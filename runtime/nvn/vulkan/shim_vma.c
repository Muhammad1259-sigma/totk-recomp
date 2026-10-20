// TotK: dedicated-allocation stand-in for the VMA calls the LASRecomp Vulkan backend makes (shim/vma/vk_mem_alloc.h).
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vma/vk_mem_alloc.h>

struct VmaAllocator_T { VkDevice dev; VkPhysicalDeviceMemoryProperties mp; uint64_t vram_cap; uint32_t vram_heap; };
struct VmaAllocation_T { VkDeviceMemory mem; void *mapped; uint32_t type; VkDeviceSize size; uint32_t tag; };
// Live bytes per caller tag (Deck d16 2026-10-16: 3 GB asked for in a 1 GB VRAM carve-out; which kind of image holds it?).
// The caller sets vma_shim_tag around a create (textures.c: 1 ASTC decoded to RGBA8, 2 other texture images, 3 block shadows).
_Thread_local int vma_shim_tag;
static _Atomic uint64_t tag_bytes[8];
static _Atomic uint32_t tag_count[8];
uint64_t vma_shim_tag_bytes(int tag, uint32_t *count) { if (count) *count = tag_count[tag & 7]; return tag_bytes[tag & 7]; }
VkDeviceSize vma_shim_size(VmaAllocation al) { return al ? al->size : 0; }
// live bytes / allocations per memory heap (device.c vk_mem_report, LARECOMP_MEMREPORT_AT)
static _Atomic uint64_t heap_bytes[VK_MAX_MEMORY_HEAPS];
static _Atomic uint32_t heap_count[VK_MAX_MEMORY_HEAPS];
uint32_t vma_shim_heaps(VmaAllocator a, uint64_t *bytes, uint32_t *count, uint64_t *size, uint32_t *flags) {
    for (uint32_t h = 0; h < a->mp.memoryHeapCount; h++) {
        bytes[h] = heap_bytes[h]; count[h] = heap_count[h];
        size[h] = a->mp.memoryHeaps[h].size; flags[h] = a->mp.memoryHeaps[h].flags;
    }
    return a->mp.memoryHeapCount;
}

VkResult vmaCreateAllocator(const VmaAllocatorCreateInfo *ci, VmaAllocator *out) {
    VmaAllocator a = calloc(1, sizeof *a);
    a->dev = ci->device;
    vkGetPhysicalDeviceMemoryProperties(ci->physicalDevice, &a->mp);
    // VRAM cap on an integrated GPU (Steam Deck: a 1 GB carve-out of the shared RAM). amdgpu does not fail a VRAM allocation
    // when the carve-out is full: it evicts other buffers (ours and the compositor's) to GTT and moves them back at later
    // submits. Deck v2 log (2026-10-16): VRAM 1016/1024 MB, then a 6 s stall inside vkQueueSubmit2 with the whole Deck lagging.
    // There VRAM is the same memory as GTT, so past the cap allocations simply take the other heap's types.
    // LARECOMP_VRAM_CAP_MB=<MB> (0 = no cap); default on an integrated GPU: the device-local heap minus 384 MB, at least half of it.
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(ci->physicalDevice, &pp);
    for (uint32_t h = 0; h < a->mp.memoryHeapCount; h++)
        if ((a->mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            (!a->vram_cap || a->mp.memoryHeaps[h].size > a->mp.memoryHeaps[a->vram_heap].size))
            a->vram_heap = h, a->vram_cap = 1;
    uint64_t hs = a->vram_cap ? a->mp.memoryHeaps[a->vram_heap].size : 0, cap = 0;
    const char *e = getenv("LARECOMP_VRAM_CAP_MB");
    if (e) cap = strtoull(e, NULL, 0) << 20;
    else if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && hs) cap = hs - (hs / 2 < (384ull << 20) ? hs / 2 : 384ull << 20);
    a->vram_cap = cap;
    if (cap) fprintf(stderr, "[vma] VRAM cap %llu MB of the %llu MB device-local heap %u (LARECOMP_VRAM_CAP_MB): allocations past it go to the other heap\n",
                     (unsigned long long)(cap >> 20), (unsigned long long)(hs >> 20), a->vram_heap);
    *out = a;
    return VK_SUCCESS;
}

// best memory type: all `need` flags, as many `want` flags as possible, fewest unwanted flags
static int pick_type(VmaAllocator a, uint32_t bits, VkMemoryPropertyFlags need, VkMemoryPropertyFlags want) {
    int best = -1, best_score = -1000;
    for (uint32_t i = 0; i < a->mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = a->mp.memoryTypes[i].propertyFlags;
        if ((f & need) != need) continue;
        int score = __builtin_popcount(f & want) * 4 - __builtin_popcount(f & ~(need | want));
        if (score > best_score) best_score = score, best = (int)i;
    }
    return best;
}

static VkResult alloc_for(VmaAllocator a, VkMemoryRequirements req, const VmaAllocationCreateInfo *ai, VmaAllocation *out,
                          VmaAllocationInfo *info) {
    bool host = ai->flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT);
    VkMemoryPropertyFlags need = ai->requiredFlags, want = ai->preferredFlags;
    if (host) need |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (ai->flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) want |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    else if (ai->usage != VMA_MEMORY_USAGE_AUTO_PREFER_HOST) want |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    // A full memory heap fails the allocation of its best type (a Steam Deck has a 1 GB VRAM carve-out next to ~8 GB of GTT): the
    // next best type of the same request is tried before giving up (`need` flags stay required, so the alias mirror's
    // DEVICE_LOCAL buffer still fails cleanly and exec.c falls back).
    uint32_t bits = req.memoryTypeBits;
    if (a->vram_cap && !(need & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && heap_bytes[a->vram_heap] + req.size > a->vram_cap) {
        uint32_t other = 0;
        for (uint32_t i = 0; i < a->mp.memoryTypeCount; i++)
            if (a->mp.memoryTypes[i].heapIndex != a->vram_heap) other |= 1u << i;
        if (bits & other) {
            static int logged;
            if (!logged++) fprintf(stderr, "[vma] VRAM cap reached (%llu MB in use): further allocations in the other heap\n",
                                   (unsigned long long)(heap_bytes[a->vram_heap] >> 20));
            bits &= other;
            want &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
    }
    VkResult r;
    VmaAllocation al = NULL;
    int t;
    for (bool first = true;; first = false) {
        t = pick_type(a, bits, need, want);
        if (first) {   // (first try only: a retry keeps every required flag)
            if (t < 0 && host) t = pick_type(a, bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, want);
            if (t < 0) t = pick_type(a, bits, 0, want);
        }
        if (t < 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        VkMemoryAllocateInfo mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = (uint32_t)t};
        al = calloc(1, sizeof *al);
        r = vkAllocateMemory(a->dev, &mi, NULL, &al->mem);
        if (r == VK_SUCCESS) break;
        free(al);
        if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY && r != VK_ERROR_OUT_OF_HOST_MEMORY) return r;
        {   // (the Deck's VRAM carve-out is 1-4 GB: the log shows when a heap filled and allocations moved to the next type)
            static int logged;
            if (logged++ < 8)
                fprintf(stderr, "[vma] memory type %d (heap %u, %llu MB in use) full for %llu KB: trying the next type\n", t,
                        a->mp.memoryTypes[t].heapIndex, (unsigned long long)(heap_bytes[a->mp.memoryTypes[t].heapIndex] >> 20),
                        (unsigned long long)(req.size >> 10));
        }
        bits &= ~(1u << t);
        if (!bits) return r;
    }
    al->type = (uint32_t)t; al->size = req.size; al->tag = (uint32_t)vma_shim_tag & 7;
    tag_bytes[al->tag] += req.size; tag_count[al->tag]++;
    heap_bytes[a->mp.memoryTypes[t].heapIndex] += req.size;
    heap_count[a->mp.memoryTypes[t].heapIndex]++;
    if (host) {
        r = vkMapMemory(a->dev, al->mem, 0, VK_WHOLE_SIZE, 0, &al->mapped);
        if (r != VK_SUCCESS) { vkFreeMemory(a->dev, al->mem, NULL); free(al); return r; }
    }
    if (info) *info = (VmaAllocationInfo){.memoryType = al->type, .deviceMemory = al->mem, .size = req.size, .pMappedData = al->mapped};
    *out = al;
    return VK_SUCCESS;
}

VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *bi, const VmaAllocationCreateInfo *ai, VkBuffer *buf,
                         VmaAllocation *alloc, VmaAllocationInfo *info) {
    VkResult r = vkCreateBuffer(a->dev, bi, NULL, buf);
    if (r != VK_SUCCESS) return r;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(a->dev, *buf, &req);
    if ((r = alloc_for(a, req, ai, alloc, info)) != VK_SUCCESS) { vkDestroyBuffer(a->dev, *buf, NULL); return r; }
    return vkBindBufferMemory(a->dev, *buf, (*alloc)->mem, 0);
}
VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *ii, const VmaAllocationCreateInfo *ai, VkImage *img,
                        VmaAllocation *alloc, VmaAllocationInfo *info) {
    VkResult r = vkCreateImage(a->dev, ii, NULL, img);
    if (r != VK_SUCCESS) return r;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(a->dev, *img, &req);
    if ((r = alloc_for(a, req, ai, alloc, info)) != VK_SUCCESS) { vkDestroyImage(a->dev, *img, NULL); return r; }
    return vkBindImageMemory(a->dev, *img, (*alloc)->mem, 0);
}
static void free_alloc(VmaAllocator a, VmaAllocation al) {
    if (!al) return;
    heap_bytes[a->mp.memoryTypes[al->type].heapIndex] -= al->size;
    heap_count[a->mp.memoryTypes[al->type].heapIndex]--;
    tag_bytes[al->tag] -= al->size; tag_count[al->tag]--;
    if (al->mapped) vkUnmapMemory(a->dev, al->mem);
    vkFreeMemory(a->dev, al->mem, NULL);
    free(al);
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buf, VmaAllocation al) { if (buf) vkDestroyBuffer(a->dev, buf, NULL); free_alloc(a, al); }
void vmaDestroyImage(VmaAllocator a, VkImage img, VmaAllocation al) { if (img) vkDestroyImage(a->dev, img, NULL); free_alloc(a, al); }

// TotK: the Vulkan Memory Allocator subset LASRecomp's backend uses, as one dedicated vkAllocateMemory per
// resource (runtime/nvn/vulkan/shim_vma.c). Fine for bring-up; switch to real VMA if allocation count/speed bites.
#pragma once
#include <vulkan/vulkan.h>

typedef struct VmaAllocator_T *VmaAllocator;
typedef struct VmaAllocation_T *VmaAllocation;

typedef enum {
    VMA_MEMORY_USAGE_UNKNOWN = 0, VMA_MEMORY_USAGE_AUTO = 7, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8,
    VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9
} VmaMemoryUsage;
enum {
    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x1, VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x4,
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x400, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x800
};
typedef VkFlags VmaAllocationCreateFlags;

typedef struct {
    VmaAllocationCreateFlags flags;
    VmaMemoryUsage usage;
    VkMemoryPropertyFlags requiredFlags, preferredFlags;
    uint32_t memoryTypeBits;
    void *pool, *pUserData;
    float priority;
} VmaAllocationCreateInfo;
typedef struct {
    uint32_t memoryType;
    VkDeviceMemory deviceMemory;
    VkDeviceSize offset, size;
    void *pMappedData, *pUserData;
    const char *pName;
} VmaAllocationInfo;
typedef struct {
    VkFlags flags;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkDeviceSize preferredLargeHeapBlockSize;
    const void *pAllocationCallbacks, *pDeviceMemoryCallbacks, *pHeapSizeLimit, *pVulkanFunctions;
    VkInstance instance;
    uint32_t vulkanApiVersion;
    const void *pTypeExternalMemoryHandleTypes;
} VmaAllocatorCreateInfo;

#ifdef __cplusplus
extern "C" {
#endif
VkResult vmaCreateAllocator(const VmaAllocatorCreateInfo *ci, VmaAllocator *out);
VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *bi, const VmaAllocationCreateInfo *ai, VkBuffer *buf,
                         VmaAllocation *alloc, VmaAllocationInfo *info);
VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *ii, const VmaAllocationCreateInfo *ai, VkImage *img,
                        VmaAllocation *alloc, VmaAllocationInfo *info);
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buf, VmaAllocation alloc);
void vmaDestroyImage(VmaAllocator a, VkImage img, VmaAllocation alloc);
#ifdef __cplusplus
}
#endif

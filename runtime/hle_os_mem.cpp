// HLE: nn::os memory (heap, memory blocks, pages, QueryMemoryInfo) and nn::mem::StandardAllocator.
#include "hle.h"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <vector>
#include <map>
#include <atomic>
#ifndef _WIN32
#include <sys/mman.h>
// runtime/linux/plat_posix.cpp: the range in a driver-owned GTT buffer (amdgpu) instead of host memory the GPU imports as userptr
extern "C" int rt_gpu_bo_map(void* addr, size_t size, int prot);
extern "C" int rt_gpu_bo_available(void);
#endif

// ---- process memory model (Switch application pool ~3.25 GB)
static const u64 kAvailable = 0xD0000000ull;
static const u64 kProgram   = 0x5000000ull;      // main image + stacks, rough
static u64 g_heap_base, g_heap_size, g_heap_used;
// for the Vulkan backend (runtime/nvn/shim/las.h): the heap is imported as one GPU buffer (VK_EXT_external_memory_host)
extern "C" u64 g_rt_heap_base;
u64 g_rt_heap_base;
extern "C" u64 kernel_heap_size(void) { return g_heap_used; }   // committed bytes from g_rt_heap_base
static std::mutex g_heap_mtx;

// nn::os::MemoryInfo: +0 totalAvailableMemorySize, +8 totalUsedMemorySize (read by nninitStartup:
// heap = ((avail - used) & ~2MB) - 0x900000), further fields zero for now.
HLE_AS(os_QueryMemoryInfo, "_ZN2nn2os15QueryMemoryInfoEPNS0_10MemoryInfoE") {
    u64* mi = (u64*)c->x[0];
    memset(mi, 0, 0x38);
    mi[0] = kAvailable; mi[1] = kProgram + g_heap_used;
    c->x[0] = 0;
}
// Write watch default. Windows: on. Linux: OFF (TOTK_WW=1 turns it on). There it is mprotect + SIGSEGV (runtime/linux/plat_posix.cpp), and
// protecting pages that the Vulkan backend has imported as GPU buffers (VK_EXT_external_memory_host: the guest heap for shader global
// memory, the alias window for vertex / index data) is hostile to amdgpu: a host-pointer import is a userptr BO that is revalidated with
// write access after every PTE change in its range, which fails (EPERM / EFAULT) on read-only pages and costs a walk of the whole range
// (2-3 GB) at the next submit when it succeeds. Off: heap_ww_changed() answers "unknown" (textures.c falls back to its sampled /
// periodic hashing) and exec.c binds the alias window's host import directly (no VRAM mirror: nothing to gain on the Deck's unified memory).
static bool ww_default() {
#ifdef _WIN32
    return true;
#else
    const char* e = getenv("TOTK_WW");
    return e && *e == '1';
#endif
}
static bool g_heap_ww;
static std::vector<u32> g_page_epoch;   // per 4 KiB heap page: epoch of the last check that found it written
static u32 g_ww_epoch;
static std::mutex g_ww_mtx;
// Were any pages of [addr, addr + size) in the heap written since *seen (the caller's epoch from its previous call)?
// 1 yes, 0 no, -1 unknown (no write watch, outside the committed heap, or the call failed). The watch bits of the range
// are fetched and reset in one GetWriteWatch, and every written page gets the new epoch, so textures sharing pages all
// see the write (a plain per-range reset would hide it from the others). A write after the fetch shows up next time.
// CPU writes only: GPU / DMA writes into the imported heap are not seen (callers keep a periodic full check).
extern "C" int heap_ww_changed(u64 addr, u64 size, u64* seen) {
    if (!g_heap_ww || !size || addr < g_heap_base || addr + size > g_heap_base + g_heap_used) return -1;
    u64 a = addr & ~0xFFFull, e = (addr + size + 0xFFF) & ~0xFFFull;
    u64 first = (a - g_heap_base) >> 12, npages = (e - a) >> 12;
    std::lock_guard<std::mutex> l(g_ww_mtx);
    if (g_page_epoch.size() < (g_heap_size >> 12)) g_page_epoch.resize(g_heap_size >> 12, 0);
    static std::vector<void*> hits;
    if (hits.size() < npages) hits.resize(npages);
    ULONG_PTR count = npages;
    DWORD gran = 0;
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void*)a, (SIZE_T)(e - a), hits.data(), &count, &gran) != 0) return -1;
    if (count) {
        ++g_ww_epoch;
        for (ULONG_PTR i = 0; i < count; i++) g_page_epoch[((u64)hits[i] - g_heap_base) >> 12] = g_ww_epoch;
    }
    u32 newest = 0;
    for (u64 i = 0; i < npages; i++) if (g_page_epoch[first + i] > newest) newest = g_page_epoch[first + i];   // (windows.h max macro)
    bool changed = newest > *seen;
    *seen = g_ww_epoch;
    return changed;
}
// exec.c's VRAM mirror of the heap (vertex / index data): the pages of [addr, addr + len) written since the mirror last
// copied them, or never copied, as (offset from the heap base, length) runs, and marks them copied. Shares the page
// epochs with heap_ww_changed, so neither caller's GetWriteWatch reset hides a write from the other. -1: no write watch
// or outside the committed heap. len 0: probe. CPU writes only (exec.c tracks GPU writes itself).
static std::vector<u32> g_mirror_epoch;   // per page: 0 never copied, else g_ww_epoch + 1 when copied
extern "C" int heap_mirror_take(u64 addr, u64 len, u64* offs, u64* lens, int max, u64* bytes) {
    *bytes = 0;
    if (!g_heap_ww) return -1;
    if (!len) return 0;
    if (addr < g_heap_base || addr >= g_heap_base + g_heap_used) return -1;
    if (addr + len > g_heap_base + g_heap_used) len = g_heap_base + g_heap_used - addr;
    u64 a = addr & ~0xFFFull, e = (addr + len + 0xFFF) & ~0xFFFull;
    u64 first = (a - g_heap_base) >> 12, npages = (e - a) >> 12;
    std::lock_guard<std::mutex> l(g_ww_mtx);
    if (g_page_epoch.size() < (g_heap_size >> 12)) g_page_epoch.resize(g_heap_size >> 12, 0);
    if (g_mirror_epoch.size() < (g_heap_size >> 12)) g_mirror_epoch.resize(g_heap_size >> 12, 0);
    static std::vector<void*> hits;
    if (hits.size() < npages) hits.resize(npages);
    ULONG_PTR count = npages;
    DWORD gran = 0;
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void*)a, (SIZE_T)(e - a), hits.data(), &count, &gran) != 0) return -1;
    if (count) {
        ++g_ww_epoch;
        for (ULONG_PTR i = 0; i < count; i++) g_page_epoch[((u64)hits[i] - g_heap_base) >> 12] = g_ww_epoch;
    }
    int n = 0;
    for (u64 i = 0; i < npages; i++) {
        u64 p = first + i;
        u32 m = g_mirror_epoch[p];
        if (m && g_page_epoch[p] < m) continue;
        g_mirror_epoch[p] = g_ww_epoch + 1;
        u64 o = p << 12;
        if (n && offs[n - 1] + lens[n - 1] == o) lens[n - 1] += 4096;
        else if (n == max) lens[n - 1] = o + 4096 - offs[n - 1];
        else { offs[n] = o; lens[n] = 4096; n++; }
        *bytes += 4096;
    }
    return n;
}
HLE_AS(os_SetMemoryHeapSize, "_ZN2nn2os17SetMemoryHeapSizeEm") {
    std::lock_guard<std::mutex> l(g_heap_mtx);
    u64 size = c->x[0];
    if (!g_heap_base && size) {
        // At a fixed address (TOTK_HEAP_ADDR, default 0x1000000000; 0 = anywhere): translated shaders bake the heap base
        // into every global-memory access, so a heap at a new address each run made those shaders miss the SPIR-V and
        // pipeline caches every run (run204: 186 of 193 SPIR-V misses contained the base) = compile hitches every session.
        const char* e = getenv("TOTK_HEAP_ADDR");
        u64 want = e ? strtoull(e, nullptr, 0) : 0x1000000000ull;
        // MEM_WRITE_WATCH (TOTK_HEAP_WW=0: off): textures.c asks heap_ww_changed() whether a texture's pages were written
        // instead of hashing it (run269: two never-changing textures, 5120x5120 + 2048x4096 linear, were hashed every
        // frame = 3.2 ms per frame of the render thread)
#ifdef _WIN32
        const DWORD ww = (getenv("TOTK_HEAP_WW") && !strcmp(getenv("TOTK_HEAP_WW"), "0")) || !ww_default() ? 0 : MEM_WRITE_WATCH;   // run273 alone: clean, Sky Island 50.8 -> 56.5 fps
#else
        // Linux: on when the heap becomes a driver-owned GPU buffer (rt_gpu_bo_map below; mprotect was only hostile to the old
        // userptr import). Deck v4 log: the same two textures hashed ~every frame, 2 s of render-thread time per 600 frames.
        // TOTK_HEAP_WW=1 forces it on, =0 off.
        const char* hw = getenv("TOTK_HEAP_WW");
        const DWORD ww = (hw ? *hw == '1' : rt_gpu_bo_available() != 0) ? MEM_WRITE_WATCH : 0;
#endif
        if (want) g_heap_base = (u64)VirtualAlloc((void*)want, size, MEM_RESERVE | ww, PAGE_READWRITE);
        if (want && !g_heap_base) printf("[os] heap: %llx is taken (err %lu), placing it anywhere\n", want, GetLastError());
        if (!g_heap_base) g_heap_base = (u64)VirtualAlloc(nullptr, size, MEM_RESERVE | ww, PAGE_READWRITE);
        g_heap_ww = g_heap_base && ww;
        if (!g_heap_base) rt_trap(c, c->x[30], "SetMemoryHeapSize: reserve failed");
        g_heap_size = size;
        g_rt_heap_base = g_heap_base;
#ifndef _WIN32
        // (before anything is committed / written: AllocateMemoryBlock mprotects)
        if (rt_gpu_bo_map((void*)g_heap_base, size, PROT_NONE) < 0 && g_heap_ww && !(hw && *hw == '1')) {
            g_heap_ww = false;   // (pages stay unarmed: heap_ww_changed answers "unknown")
            printf("[os] heap write watch off: the heap is not a driver-owned GPU buffer\n");
        }
        printf("[os] heap write watch %s\n", g_heap_ww ? "on" : "off");
#endif
    }
    printf("[os] SetMemoryHeapSize(%llx) -> heap %llx\n", size, g_heap_base);
    c->x[0] = 0;
}
HLE_AS(os_AllocateMemoryBlock, "_ZN2nn2os19AllocateMemoryBlockEPmm") {
    std::lock_guard<std::mutex> l(g_heap_mtx);
    u64 size = (c->x[1] + 0x1FFFFF) & ~0x1FFFFFull;
    if (g_heap_used + size > g_heap_size) { printf("[os] AllocateMemoryBlock(%llx): out of heap\n", size); c->x[0] = 0x1A01; return; }
    u64 a = g_heap_base + g_heap_used;
    if (!VirtualAlloc((void*)a, size, MEM_COMMIT, PAGE_READWRITE)) rt_trap(c, c->x[30], "AllocateMemoryBlock: commit failed");
    g_heap_used += size;
    *(u64*)c->x[0] = a;
    printf("[os] AllocateMemoryBlock(%llx) -> %llx\n", size, a);
    c->x[0] = 0;
}
// ---- address regions: carved from ONE reserved range (the Switch "alias" region), lowest address first, like the
// kernel does. Separate VirtualAlloc reservations scatter regions around the main heap; sead's PageHeap grows by many
// 64 KiB regions and its isInclude (f_29b7ed0) is the range [lowest, highest), so that range then covered main-heap
// blocks: sead free() handed ExpHeap blocks to the PageHeap, which FreeMemoryPages'd a main-heap page (run51/55: heap
// crash in f_b21d50; run59: 1,652 regions scattered).
static const u64 kAliasSize = 0x1000000000ull;    // 64 GB, the alias region size for 39-bit Switch processes
static const u64 kRegionAlign = 0x10000;          // PageHeap masks pointers with ~0xFFFF: keep 64 KiB alignment
static u64 g_alias_base, g_alias_size;
static std::map<u64, u64> g_alias_free, g_alias_used;   // base -> size
static std::mutex g_alias_mtx;
static bool in_alias(u64 a, u64 n) { return g_alias_base && a >= g_alias_base && a + n <= g_alias_base + g_alias_size; }
// Import window (TOTK_ALIAS_IMPORT_MB, default 2048, 0 = off): the first W bytes of the alias region stay committed so the
// Vulkan backend can import them as one GPU buffer (VK_EXT_external_memory_host, exec.c alias_import) and bind vertex /
// index data there in place. run187/188: ~100k vertex copies (~5 GB) per 2 s came from alias regions (high-water 518 MB).
// An import pins its pages, so FreeMemoryPages must not decommit inside the window; instead a per-4 KiB "mapped" bitmap
// keeps the kernel's MapPhysicalMemory semantics: pages that become mapped again read as zero, mapped pages keep data.
extern "C" u64 g_alias_import_base, g_alias_import_size;
u64 g_alias_import_base, g_alias_import_size;
static u8* g_alias_mapped;                       // 1 bit per 4 KiB page of the window
static u64 g_alias_hw;                           // high-water offset of carved regions (bounds the write-watch scan)
static bool g_alias_ww;                          // reserved with MEM_WRITE_WATCH (exec.c keeps a VRAM mirror of the window)
// Dirty ranges of the window since the previous call (GetWriteWatch with reset), as (offset, length) pairs from the
// window base, coalesced; at most `max` ranges (the last one widens to cover the rest). Returns -1 without write watch.
extern "C" int alias_dirty_take(u64* offs, u64* lens, int max, u64* bytes) {
    if (!g_alias_ww || !g_alias_import_size) return -1;
    u64 span = (g_alias_hw + 0xFFFF) & ~0xFFFFull;
    if (span > g_alias_import_size) span = g_alias_import_size;
    *bytes = 0;
    if (!span) return 0;
    if (!max) { ResetWriteWatch((void*)g_alias_import_base, (SIZE_T)span); return 0; }   // (probe: start tracking from now)
    static void** pages = (void**)malloc((g_alias_import_size >> 12) * sizeof(void*));
    ULONG_PTR count = (ULONG_PTR)(g_alias_import_size >> 12);
    ULONG gran = 0;
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void*)g_alias_import_base, (SIZE_T)span, pages, &count, &gran) != 0) {
        offs[0] = 0; lens[0] = span; *bytes = span;   // (failed: copy all)
        return 1;
    }
    int n = 0;
    for (ULONG_PTR i = 0; i < count; i++) {
        u64 o = (u64)pages[i] - g_alias_import_base;
        if (n && offs[n - 1] + lens[n - 1] == o) { lens[n - 1] += gran; continue; }
        if (n == max) { lens[n - 1] = o + gran - offs[n - 1]; continue; }
        offs[n] = o; lens[n] = gran; n++;
    }
    for (int i = 0; i < n; i++) *bytes += lens[i];
    return n;
}
// The same for [off, off + len) of the window only (exec.c lazy mirror sync: just the 1 MB chunks draws bind). A query
// costs ~0.74 ms per GB whatever the split (scratch wwbench 2026-10-15: 1 GB whole 0.74 ms, 1024 x 1 MB 0.88 ms), so
// scanning only bound chunks saves in proportion to what a list does not bind. Pages outside the range keep their bits.
extern "C" int alias_dirty_take_range(u64 off, u64 len, u64* offs, u64* lens, int max, u64* bytes) {
    *bytes = 0;
    if (!g_alias_ww || !g_alias_import_size || off >= g_alias_import_size) return -1;
    if (off + len > g_alias_import_size) len = g_alias_import_size - off;
    static void* pages[(1u << 20) >> 12];   // (callers pass at most 1 MB)
    if (len > sizeof pages / sizeof pages[0] * 4096ull) len = sizeof pages / sizeof pages[0] * 4096ull;
    ULONG_PTR count = (ULONG_PTR)(len >> 12);
    ULONG gran = 0;
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void*)(g_alias_import_base + off), (SIZE_T)len, pages, &count, &gran) != 0) {
        offs[0] = off; lens[0] = len; *bytes = len;   // (failed: copy all of it)
        return 1;
    }
    int n = 0;
    for (ULONG_PTR i = 0; i < count; i++) {
        u64 o = (u64)pages[i] - g_alias_import_base;
        if (n && offs[n - 1] + lens[n - 1] == o) { lens[n - 1] += gran; continue; }
        if (n == max) { lens[n - 1] = o + gran - offs[n - 1]; continue; }
        offs[n] = o; lens[n] = gran; n++;
    }
    for (int i = 0; i < n; i++) *bytes += lens[i];
    return n;
}
static u64 g_alias_zeroed;                       // bytes zero-filled on (re)map, for the log
static bool in_window(u64 a, u64 n) { return g_alias_import_size && a >= g_alias_import_base && a + n <= g_alias_import_base + g_alias_import_size; }
static void window_map(u64 a, u64 n, bool map) {
    for (u64 p = (a - g_alias_import_base) >> 12, e = (a + n - g_alias_import_base + 0xFFF) >> 12; p < e; p++) {
        bool was = g_alias_mapped[p >> 3] >> (p & 7) & 1;
        if (map && !was) { memset((void*)(g_alias_import_base + (p << 12)), 0, 0x1000); g_alias_zeroed += 0x1000; }
        if (map) g_alias_mapped[p >> 3] |= (u8)(1u << (p & 7)); else g_alias_mapped[p >> 3] &= (u8)~(1u << (p & 7));
    }
}
HLE_AS(os_AllocateAddressRegion, "_ZN2nn2os21AllocateAddressRegionEPmm") {
    std::lock_guard<std::mutex> l(g_alias_mtx);
    if (!g_alias_base) {
        for (u64 sz = kAliasSize; sz >= 0x100000000ull && !g_alias_base; sz >>= 1)
            if (void* p = VirtualAlloc(nullptr, sz, MEM_RESERVE | ((getenv("TOTK_ALIAS_NO_WATCH") || !ww_default()) ? 0 : MEM_WRITE_WATCH), PAGE_READWRITE)) {
                g_alias_base = (u64)p; g_alias_size = sz; g_alias_ww = !getenv("TOTK_ALIAS_NO_WATCH") && ww_default();
            }
        if (!g_alias_base) rt_trap(c, c->x[30], "AllocateAddressRegion: cannot reserve the alias region");
        g_alias_free[g_alias_base] = g_alias_size;
        printf("[os] alias region %llx + %llx\n", g_alias_base, g_alias_size);
        const char* e = getenv("TOTK_ALIAS_IMPORT_MB");
        u64 w = (e ? strtoull(e, nullptr, 0) : 2048) << 20;   // (1024 until 10-15 23:10: the user's save reached 1377 MB -> 3.7 GB of vertex copies per 2 s, run359)
        if (w > g_alias_size) w = g_alias_size;
        if (w && VirtualAlloc((void*)g_alias_base, w, MEM_COMMIT, PAGE_READWRITE)) {
#ifndef _WIN32
            if (!g_alias_ww) rt_gpu_bo_map((void*)g_alias_base, w, PROT_READ | PROT_WRITE);   // (nothing written there yet)
#endif
            g_alias_mapped = (u8*)calloc(w >> 15, 1);
            g_alias_import_base = g_alias_base; g_alias_import_size = w;   // (published last: exec.c reads it unlocked)
            printf("[os] alias import window %llx + %llu MB committed\n", g_alias_base, (unsigned long long)(w >> 20));
        } else if (w) printf("[os] alias import window: commit of %llu MB failed, vertex data there is copied\n", (unsigned long long)(w >> 20));
    }
    u64 size = (c->x[1] + kRegionAlign - 1) & ~(kRegionAlign - 1), got = 0;
    for (auto it = g_alias_free.begin(); it != g_alias_free.end(); ++it) {
        if (it->second < size) continue;
        got = it->first; u64 rest = it->second - size;
        g_alias_free.erase(it);
        if (rest) g_alias_free[got + size] = rest;
        break;
    }
    if (!got) { printf("[os] AllocateAddressRegion(%llx): alias region full\n", c->x[1]); c->x[0] = 0x1A01; return; }
    g_alias_used[got] = size;
    if (got + size - g_alias_base > g_alias_hw) g_alias_hw = got + size - g_alias_base;
    static int n; if (n++ < 16 || size >= (1ull << 20)) printf("[os] AllocateAddressRegion(%llx) -> %llx lr=%llx\n", c->x[1], got, c->x[30]);
    static u64 high;  // high-water mark of the alias region (sizing a Vulkan import of it)
    if (got + size - g_alias_base > high + (64ull << 20)) {
        high = got + size - g_alias_base;
        printf("[os] alias region high-water %llu MB (%zu regions, %llu MB zero-filled in the import window)%s\n", (unsigned long long)(high >> 20),
               g_alias_used.size(), (unsigned long long)(g_alias_zeroed >> 20),
               high > g_alias_import_size ? " PAST THE IMPORT WINDOW: vertex data there is copied (raise TOTK_ALIAS_IMPORT_MB)" : "");
    }
    *(u64*)c->x[0] = got; c->x[0] = 0;
}
HLE_AS(os_FreeAddressRegion, "_ZN2nn2os17FreeAddressRegionEm") {
    std::lock_guard<std::mutex> l(g_alias_mtx);
    auto it = g_alias_used.find(c->x[0]);
    if (it == g_alias_used.end()) { printf("[os] FreeAddressRegion(%llx): not a region\n", c->x[0]); c->x[0] = 0; return; }
    u64 a = it->first, n = it->second;
    g_alias_used.erase(it);
    if (in_window(a, n)) window_map(a, n, false);   // the import pins these pages: keep them committed
    else if (g_alias_import_size && a < g_alias_import_base + g_alias_import_size) {   // straddles the window end
        u64 we = g_alias_import_base + g_alias_import_size;
        window_map(a, we - a, false);
        VirtualFree((void*)we, a + n - we, MEM_DECOMMIT);
    } else VirtualFree((void*)a, n, MEM_DECOMMIT);
    auto nx = g_alias_free.lower_bound(a);                  // coalesce with the free neighbours
    if (nx != g_alias_free.end() && nx->first == a + n) { n += nx->second; g_alias_free.erase(nx); }
    auto pv = g_alias_free.lower_bound(a);
    if (pv != g_alias_free.begin() && (--pv, pv->first + pv->second == a)) { pv->second += n; }
    else g_alias_free[a] = n;
    c->x[0] = 0;
}
HLE_AS(os_AllocateMemoryPages, "_ZN2nn2os19AllocateMemoryPagesEmm") {
    static int nlog; if (nlog++ < 64) printf("[os] AllocateMemoryPages(%llx, %llx)\n", c->x[0], c->x[1]);
    u64 a = c->x[0], n = c->x[1];
    if (g_alias_import_size && a >= g_alias_import_base && a < g_alias_import_base + g_alias_import_size) {
        std::lock_guard<std::mutex> l(g_alias_mtx);
        u64 we = g_alias_import_base + g_alias_import_size, wn = a + n <= we ? n : we - a;
        window_map(a, wn, true);                     // (committed already: zero the pages that were unmapped)
        if (wn < n && !VirtualAlloc((void*)we, n - wn, MEM_COMMIT, PAGE_READWRITE)) { c->x[0] = 0x1A01; return; }
        c->x[0] = 0;
        return;
    }
    c->x[0] = VirtualAlloc((void*)a, n, MEM_COMMIT, PAGE_READWRITE) ? 0 : 0x1A01;
}
HLE_AS(os_FreeMemoryPages, "_ZN2nn2os15FreeMemoryPagesEmm") {
    static int nlog; if (nlog++ < 64) printf("[os] FreeMemoryPages(%llx, %llx) lr=%llx\n", c->x[0], c->x[1], c->x[30]);
    if (c->x[0] < g_heap_base + g_heap_size && c->x[0] + c->x[1] > g_heap_base) {   // pages API is for address regions only
        printf("[os] FreeMemoryPages(%llx, %llx) INSIDE the main heap, lr=%llx x19-x23 %llx %llx %llx %llx %llx bt:", c->x[0], c->x[1],
               c->x[30], c->x[19], c->x[20], c->x[21], c->x[22], c->x[23]);
        u64 fp = c->x[29];
        for (int i = 0; i < 12 && fp >= c->sp && fp < c->sp + (1ull << 24) && !(fp & 7); i++) {
            printf(" %llx", ((u64*)fp)[1]);
            if (((u64*)fp)[0] <= fp) break;
            fp = ((u64*)fp)[0];
        }
        printf("\n[os]   page heap x20 range [+e0 %llx, +e8 %llx)  main heap [%llx, %llx)\n", ((u64*)c->x[20])[0xe0 / 8],
               ((u64*)c->x[20])[0xe8 / 8], g_heap_base, g_heap_base + g_heap_size);
        fflush(stdout);
    }
    // The kernel only unmaps pages in the alias region (svcUnmapPhysicalMemory); never decommit anything else.
    if (!in_alias(c->x[0], c->x[1])) { printf("[os] FreeMemoryPages(%llx, %llx) outside the alias region: ignored\n", c->x[0], c->x[1]); c->x[0] = 0; return; }
    u64 a = c->x[0], n = c->x[1];
    if (g_alias_import_size && a >= g_alias_import_base && a < g_alias_import_base + g_alias_import_size) {
        std::lock_guard<std::mutex> l(g_alias_mtx);
        u64 we = g_alias_import_base + g_alias_import_size, wn = a + n <= we ? n : we - a;
        window_map(a, wn, false);                    // the import pins these pages: keep them committed
        if (wn < n) VirtualFree((void*)we, n - wn, MEM_DECOMMIT);
        c->x[0] = 0;
        return;
    }
    VirtualFree((void*)a, n, MEM_DECOMMIT); c->x[0] = 0;
}
static u64 g_tl_alloc, g_tl_free;
HLE_AS(os_SetMemoryAllocatorForThreadLocal, "_ZN2nn2os32SetMemoryAllocatorForThreadLocalEPFPvmmEPFvS1_mE") {
    g_tl_alloc = c->x[0]; g_tl_free = c->x[1];
}

// ---- nn::mem::StandardAllocator, backed by the host heap.
// The guest object is opaque to the game; we keep our state in a host struct whose pointer sits in the
// object's first 8 bytes (+8 holds a magic). Each block has a 16-byte header {size, offset to raw}.
struct StdAlloc { u64 magic; u64 cap; std::atomic<u64> used; };
static const u64 kMagic = 0x434F4C4C41445453ull;
static StdAlloc* sa(Ctx* c) {
    StdAlloc* s = *(StdAlloc**)c->x[0];
    if (!s || s->magic != kMagic) rt_trap(c, c->x[30], "StandardAllocator used before Initialize");
    return s;
}
static void sa_init(Ctx* c, u64 size) {
    StdAlloc* s = new StdAlloc{ kMagic, size, {0} };
    *(StdAlloc**)c->x[0] = s; ((u64*)c->x[0])[1] = kMagic;
}
static u64 blk_alloc(StdAlloc* s, u64 size, u64 align) {
    if (align < 16) align = 16;
    u8* raw = (u8*)malloc(size + align + 16);
    if (!raw) return 0;
    u64 p = ((u64)raw + 16 + align - 1) & ~(align - 1);
    ((u64*)p)[-2] = size; ((u64*)p)[-1] = p - (u64)raw;
    s->used += size;
    return p;
}
static void blk_free(StdAlloc* s, u64 p) {
    if (!p) return;
    s->used -= ((u64*)p)[-2];
    free((u8*)p - ((u64*)p)[-1]);
}
HLE_AS(sa_ctor, "_ZN2nn3mem17StandardAllocatorC1Ev") { memset((void*)c->x[0], 0, 16); }
HLE_AS(sa_ctor2, "_ZN2nn3mem17StandardAllocatorC1EPvm") { sa_init(c, c->x[2]); }
HLE_AS(sa_init3, "_ZN2nn3mem17StandardAllocator10InitializeEPvmb") { sa_init(c, c->x[2]); }
HLE_AS(sa_init2, "_ZN2nn3mem17StandardAllocator10InitializeEPvm") { sa_init(c, c->x[2]); }
HLE_AS(sa_fini, "_ZN2nn3mem17StandardAllocator8FinalizeEv") { }
HLE_AS(sa_alloc1, "_ZN2nn3mem17StandardAllocator8AllocateEm") { c->x[0] = blk_alloc(sa(c), c->x[1], 16); }
HLE_AS(sa_alloc2, "_ZN2nn3mem17StandardAllocator8AllocateEmm") { c->x[0] = blk_alloc(sa(c), c->x[1], c->x[2]); }
HLE_AS(sa_free, "_ZN2nn3mem17StandardAllocator4FreeEPv") { blk_free(sa(c), c->x[1]); }
HLE_AS(sa_realloc, "_ZN2nn3mem17StandardAllocator10ReallocateEPvm") {
    StdAlloc* s = sa(c); u64 old = c->x[1], size = c->x[2];
    if (!old) { c->x[0] = blk_alloc(s, size, 16); return; }
    if (!size) { blk_free(s, old); c->x[0] = 0; return; }
    u64 n = blk_alloc(s, size, 16);
    if (n) { u64 os = ((u64*)old)[-2]; memcpy((void*)n, (void*)old, os < size ? os : size); blk_free(s, old); }
    c->x[0] = n;
}
HLE_AS(sa_shrink, "_ZN2nn3mem17StandardAllocator6ShrinkEPvm") { c->x[0] = 0; }   // keep the block as is
HLE_AS(sa_free_size, "_ZNK2nn3mem17StandardAllocator16GetTotalFreeSizeEv") { StdAlloc* s = sa(c); c->x[0] = s->cap > s->used ? s->cap - s->used : 0; }
HLE_AS(sa_allocatable, "_ZNK2nn3mem17StandardAllocator18GetAllocatableSizeEv") { StdAlloc* s = sa(c); c->x[0] = s->cap > s->used ? s->cap - s->used : 0; }
HLE_AS(sa_dump, "_ZNK2nn3mem17StandardAllocator4DumpEv") { }
HLE_AS(sa_walk, "_ZNK2nn3mem17StandardAllocator19WalkAllocatedBlocksEPFiPvmS2_ES2_") { }

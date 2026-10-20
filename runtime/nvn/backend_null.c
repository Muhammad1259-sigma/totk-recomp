// Step A backend: no GPU. Records are built and kept by cmdbuf.c; executing/presenting them does nothing yet.
// Replaced by the Vulkan backend (LASRecomp runtime/nvn/vulkan) in step B.
#include "nvn.h"

static uint64_t g_seq, g_presents, g_execs;
void backend_init(void) { LOG("[nvn] null backend (no rendering yet)\n"); }
void backend_execute(uint64_t handle) { (void)handle; g_execs++; }
void backend_exec_lock(void) {}
void backend_exec_unlock(void) {}
void backend_present(NvnTexture *tex) {
    if (g_presents++ < 5 || g_presents % 300 == 0)
        LOG("[nvn] present #%llu tex %ux%u fmt %#x (command lists executed: %llu)\n", (unsigned long long)g_presents,
            tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->format : 0, (unsigned long long)g_execs);
}
void backend_texture_destroyed(NvnTexture *tex) { (void)tex; }
void backend_pool_written(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
uint64_t renderer_fence_seq(void) { return ++g_seq; }
void renderer_wait_seq(uint64_t seq) { (void)seq; }
void renderer_flush(void) {}
void renderer_sync(void) {}

#include <windows.h>
uint64_t plat_now_ns(void) {
    static LARGE_INTEGER f; LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (uint64_t)((__uint128_t)t.QuadPart * 1000000000u / (uint64_t)f.QuadPart);
}
// main thread parks here (rt_main.cpp); the boot watcher thread ends the process when nnMain returns
int ui_run(void) { for (;;) Sleep(1000); }

// TotK Linux port: the symbols the runtime takes from the Vulkan backend (totk_hooks.c, exec.c, textures.c, device.c), for the
// null-backend build (mkninja_linux.py --null: headless smoke tests; no window, no GPU, no shaderc). Linux build only.
#ifndef _WIN32
#include <stdbool.h>
#include <stdint.h>
#include "nvn.h"

volatile int g_debug_frame_request;              // totk_hooks.c: hle_hid.cpp F9
bool g_debug_frame;                              // exec.c: objects.c queue logging
int vk_precompile_recorded_progress(int *done, int *total) { (void)done; (void)total; return 0; }   // exec.c: hle_osd.cpp banner

// Async execute (exec.c): never on here, so objects.c takes its synchronous paths; backend_async_call runs the callback right away
// like the synchronous mode of the real one.
bool backend_async_on(void) { return false; }
void backend_async_call(void (*fn)(void *), void *arg) { fn(arg); }
void backend_async_fence(NvnSync *s) { (void)s; }
void backend_async_present(NvnTexture *tex) { (void)tex; }
void backend_async_wait(uint64_t ticket) { (void)ticket; }
// textures.c / device.c: texture ownership and the frame counter the null backend does not have
void backend_texture_cpu_written(NvnTexture *t) { (void)t; }
bool renderer_texture_gpu_owned(NvnTexture *t) { (void)t; return false; }
uint64_t renderer_present_count(void) { return 0; }
void tex_watch_add(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
bool tex_watched(uint64_t addr) { (void)addr; return false; }
#endif

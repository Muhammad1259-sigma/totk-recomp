// TotK Linux port: force-included (-include) into the Vulkan backend TUs (runtime/nvn/vulkan, maxwell) instead of
// runtime/nvn/shim/compat.h (the Windows/UCRT one). Those TUs compile against the REAL SDL3 headers here (the declare-only
// subset in runtime/nvn/shim/SDL3 is the Windows stand-in); the three SDL calls below get thin wrappers in shim_sdl3.c:
//  - SDL_WaitEventTimeout: the UI loop (device.c ui_run, main thread) also handles the window hotkeys, title / fullscreen requests
//    posted by other threads, and swallows every event but quit, like shim_win.c's message pump
//  - SDL_SetWindowTitle: called from the present thread; SDL wants window calls on the thread that pumps events
//  - SDL_SavePNG: only in SDL >= 3.4; the PNG writer (stored deflate blocks) is shim_win.c's
#pragma once
#include <windows.h>      // runtime/linux shim: GetCurrentThread / SetThreadPriority (exec.c async_main), THREAD_PRIORITY_*
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>

#define SDL_WaitEventTimeout totk_wait_event
#define SDL_SetWindowTitle totk_set_window_title
#define SDL_SavePNG totk_save_png
typedef struct SDL_Surface SDL_Surface;
bool totk_save_png(SDL_Surface *s, const char *path);

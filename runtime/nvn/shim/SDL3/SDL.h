// TotK: minimal SDL3 subset on Win32 for the LASRecomp Vulkan backend (window, Vulkan surface, timers, events).
// Only what runtime/nvn/vulkan/*.c call; implemented in runtime/nvn/vulkan/shim_win.c.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef uint32_t Uint32;
typedef uint64_t Uint64;
typedef uint64_t SDL_WindowFlags;
typedef struct SDL_Window SDL_Window;
typedef struct SDL_Surface SDL_Surface;

#define SDL_INIT_VIDEO 0x20u
#define SDL_INIT_GAMEPAD 0x2000u
#define SDL_WINDOW_RESIZABLE 0x20ull
#define SDL_WINDOW_HIDDEN 0x8ull
#define SDL_WINDOW_HIGH_PIXEL_DENSITY 0x2000ull
#define SDL_WINDOW_VULKAN 0x10000000ull
#define SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS "SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"
#define SDL_PIXELFORMAT_ABGR8888 0x16762004u

enum { SDL_EVENT_QUIT = 0x100, SDL_EVENT_KEY_DOWN = 0x300, SDL_EVENT_KEY_UP = 0x301, SDL_EVENT_WINDOW_RESIZED = 0x206 };
typedef struct { uint32_t type; uint32_t key; uint64_t timestamp; int32_t data1, data2; } SDL_Event;

bool SDL_Init(uint32_t flags);
bool SDL_SetHint(const char *name, const char *value);
const char *SDL_GetError(void);
SDL_Window *SDL_CreateWindow(const char *title, int w, int h, SDL_WindowFlags flags);
bool SDL_SetWindowFullscreen(SDL_Window *w, bool fs);
bool SDL_SetWindowTitle(SDL_Window *w, const char *title);
bool SDL_GetWindowSizeInPixels(SDL_Window *w, int *pw, int *ph);
const char *const *SDL_Vulkan_GetInstanceExtensions(Uint32 *count);
bool SDL_Vulkan_CreateSurface(SDL_Window *w, VkInstance inst, const VkAllocationCallbacks *a, VkSurfaceKHR *out);
bool SDL_WaitEventTimeout(SDL_Event *e, int32_t ms);
Uint64 SDL_GetTicksNS(void);
Uint64 SDL_GetTicks(void);
void SDL_Delay(Uint32 ms);
int SDL_GetNumLogicalCPUCores(void);
SDL_Surface *SDL_CreateSurfaceFrom(int w, int h, uint32_t fmt, void *pixels, int pitch);
bool SDL_SavePNG(SDL_Surface *s, const char *path);
void SDL_DestroySurface(SDL_Surface *s);

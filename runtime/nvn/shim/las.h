// TotK: stand-ins for the LASRecomp app layer (settings, menu/ImGui overlay, kernel heap, texture patches) that its
// Vulkan backend calls. TotK has no overlay or randomizer; the heap is runtime/hle_os_mem.cpp's SetMemoryHeapSize region.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "../glue.h"
#include "../nvn.h"
#include <SDL3/SDL.h>

// ---- settings (settings.h)
enum { ASPECT_UNLOCKED = 0, ASPECT_ORIGINAL = 1 };
enum { RES_MATCH = 0, RES_DYNAMIC = 9, RES_Q75 = 10, RES_Q25 = 11, RES_Q50 = 12 };
enum { UPSCALE_BILINEAR = 0, UPSCALE_METALFX = 1, UPSCALE_FSR1 = 2 };
enum { AA_OFF = 0, AA_SMAA = 1 };
typedef struct { int aspect; bool fullscreen; int res_mode, upscaler, antialias; bool show_fps; } Settings;
extern Settings g_settings;
void settings_load(void);

// ---- kernel / platform (kernel.h, rt.h, platform.h)
extern uint64_t g_rt_heap_base;          // runtime/hle_os_mem.cpp
uint64_t kernel_heap_size(void);
#define HEAP_BASE g_rt_heap_base
#define HEAP_MAX 0x00D0000000ull
void plat_terminate(void);
int rt_reboot_requested(void);
void rt_window_icon(SDL_Window *w);
const char *game_level_name(void);

// ---- menu / ui (ui/menu.h): no overlay in TotK
void menu_render(void *cb, uint32_t w, uint32_t h);
bool menu_handle_event(const SDL_Event *e);
void menu_apply_window_requests(SDL_Window *w);
void menu_test_open(void);
void input_handle_event(const SDL_Event *e);
void input_frame(uint64_t frame);
int ui_run(void);
void ui_window_init(void);
SDL_Window *ui_window(void);
void ui_size(int *w, int *h);

// ---- texture patches (texpatch.h): LASRecomp-specific, none in TotK
int texpatch_identify(const NvnTexture *t);
int texpatch_bottle_kind(int id);
bool texpatch_is_item_cursor(int id);
uint32_t texpatch_generation(void);
void texpatch_apply(int id, uint32_t lv, uint8_t *blocks, uint32_t w, uint32_t h, uint32_t row);
int texpatch_heart_kind(int id);
void texpatch_heart_tint(int kind, float tint[8]);

// ---- renderer.h (the non-Metal part)
bool renderer_is_gpu_owned(NvnTexture *t);
void renderer_flush(void);
bool renderer_origin_lower_left(void);
const char *renderer_stats(void);
bool renderer_format_is_depth(uint32_t f);
bool renderer_format_has_stencil(uint32_t f);
void renderer_mark_gpu_owned(NvnTexture *t);
bool input_user_blocked(const SDL_Event *e);

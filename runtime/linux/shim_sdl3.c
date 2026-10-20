// TotK Linux port: the window / event glue of the LASRecomp Vulkan backend on REAL SDL3 (counterpart of nvn/vulkan/shim_win.c, which
// implements the declare-only SDL3 subset on Win32), plus the same LASRecomp app-layer stand-ins (settings, menu, input, texpatch).
// Linux build only (mkninja_linux.py); nothing globs runtime/linux on Windows.
#ifndef _WIN32
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../nvn/shim/las.h"
// compat_posix.h (force-included) renamed these three; the wrappers below call the real ones
#undef SDL_WaitEventTimeout
#undef SDL_SetWindowTitle
#undef SDL_SavePNG

extern void totk_hid_sensor(int type, const float *data, Uint64 ts_ns);   // runtime/hle_hid.cpp

// ------------------------------------------------------------------ window requests from other threads
// device.c calls SDL_SetWindowTitle from the present thread every 0.5 s: the text goes through g_title and is applied by the UI
// thread (the one that pumps events) inside totk_wait_event.
static pthread_mutex_t g_title_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_title[256];
static atomic_int g_title_dirty;
static atomic_int g_fs_req = -1;    // -1 none, 0 windowed, 1 fullscreen (SDL_SetWindowFullscreen from another thread)
static atomic_int g_fps_osd = -2;   // -1 automatic (fullscreen only), 0 off, 1 on (TOTK_FPS_OVERLAY=1 / 0 at start); F3 toggles

bool totk_set_window_title(SDL_Window *w, const char *title) {
    if (!w) return false;
    pthread_mutex_lock(&g_title_lock);
    snprintf(g_title, sizeof g_title, "%s", title);
    pthread_mutex_unlock(&g_title_lock);
    atomic_store(&g_title_dirty, 1);
    return true;
}

// Fullscreen = SDL's borderless desktop mode (the window covers the display at its native size; the compositor / gamescope can
// scan it out directly). The cursor hides over the game while fullscreen.
bool shim_window_fullscreen(void) {
    SDL_Window *w = ui_window();
    return w && (SDL_GetWindowFlags(w) & SDL_WINDOW_FULLSCREEN) != 0;
}
// on-screen fps readout (device.c): shown while fullscreen (no title bar there), F3 toggles it in either mode
bool shim_fps_overlay(void) {
    if (atomic_load(&g_fps_osd) == -2) { const char *e = getenv("TOTK_FPS_OVERLAY"); atomic_store(&g_fps_osd, e ? (*e != '0') : -1); }
    int v = atomic_load(&g_fps_osd);
    return v < 0 ? shim_window_fullscreen() : v != 0;
}
static void apply_fullscreen(bool fs) {
    SDL_Window *w = ui_window();
    if (!w || fs == shim_window_fullscreen()) return;
    SDL_SetWindowFullscreen(w, fs);
    if (fs) SDL_HideCursor(); else SDL_ShowCursor();
    fprintf(stderr, "[win] %s\n", fs ? "fullscreen" : "windowed");
}
// the hook device.c does not have: SDL_SetWindowFullscreen from the present thread goes through here (not called today)
void shim_request_fullscreen(bool fs) { atomic_store(&g_fs_req, fs ? 1 : 0); }

static void apply_requests(void) {
    int fs = atomic_exchange(&g_fs_req, -1);
    if (fs >= 0) apply_fullscreen(fs != 0);
    if (atomic_exchange(&g_title_dirty, 0)) {
        char t[256];
        pthread_mutex_lock(&g_title_lock);
        memcpy(t, g_title, sizeof t);
        pthread_mutex_unlock(&g_title_lock);
        SDL_SetWindowTitle(ui_window(), t);
    }
}

// The UI loop's event wait (device.c ui_run): pumps SDL (windows, keyboard, gamepads - hle_hid.cpp reads their state through the SDL
// API), handles F11 / Alt+Enter (fullscreen) and F3 (fps readout), and returns only a quit event, like shim_win.c.
bool totk_wait_event(SDL_Event *out, Sint32 ms) {
    apply_requests();
    Uint64 end = ms < 0 ? 0 : SDL_GetTicks() + (Uint64)ms;
    for (;;) {
        SDL_Event e;
        Sint32 left = -1;
        if (ms >= 0) { Uint64 now = SDL_GetTicks(); left = now >= end ? 0 : (Sint32)(end - now); }
        if (!SDL_WaitEventTimeout(&e, left)) { apply_requests(); return false; }
        switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            out->type = SDL_EVENT_QUIT;
            return true;
        case SDL_EVENT_GAMEPAD_SENSOR_UPDATE:   // accelerometer / gyro of the open gamepad -> hle_hid.cpp's motion fusion
            totk_hid_sensor((int)e.gsensor.sensor, e.gsensor.data, e.gsensor.sensor_timestamp);
            break;
        case SDL_EVENT_KEY_DOWN:
            if (e.key.repeat) break;
            if (e.key.key == SDLK_F11 || (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT))) apply_fullscreen(!shim_window_fullscreen());
            else if (e.key.key == SDLK_F3) atomic_store(&g_fps_osd, shim_fps_overlay() ? 0 : 1);
            break;
        default: break;
        }
        apply_requests();
    }
}

// ------------------------------------------------------------------ PNG (only SDL >= 3.4 has SDL_SavePNG)
// Surfaces only carry RGBA8 pixels (SDL_PIXELFORMAT_ABGR8888 = R,G,B,A bytes) for the PNG writer below.
static uint32_t png_crc(uint32_t c, const uint8_t *p, size_t n) {
    static uint32_t t[256];
    if (!t[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t v = i; for (int k = 0; k < 8; k++) v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1; t[i] = v; }
    c = ~c;
    while (n--) c = t[(c ^ *p++) & 0xFF] ^ (c >> 8);
    return ~c;
}
static void png_be32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void png_chunk(FILE *f, const char *type, const uint8_t *data, size_t n) {
    uint8_t b[4]; png_be32(b, (uint32_t)n); fwrite(b, 1, 4, f);
    uint32_t c = png_crc(0, (const uint8_t *)type, 4); c = png_crc(c, data, n);
    fwrite(type, 1, 4, f); fwrite(data, 1, n, f); png_be32(b, c); fwrite(b, 1, 4, f);
}
bool totk_save_png(SDL_Surface *s, const char *path) {
    if (!s) return false;
    size_t row = (size_t)s->w * 4 + 1, raw_n = row * s->h;
    uint8_t *raw = malloc(raw_n);
    for (int y = 0; y < s->h; y++) { raw[y * row] = 0; memcpy(raw + y * row + 1, (const uint8_t *)s->pixels + (size_t)y * s->pitch, row - 1); }
    size_t blocks = (raw_n + 65534) / 65535, z_n = 2 + raw_n + blocks * 5 + 4;
    uint8_t *z = malloc(z_n), *q = z;
    *q++ = 0x78; *q++ = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t off = 0; off < raw_n; off += 65535) {
        size_t n = raw_n - off < 65535 ? raw_n - off : 65535;
        *q++ = off + n == raw_n; *q++ = n & 0xFF; *q++ = n >> 8; *q++ = ~n & 0xFF; *q++ = (~n >> 8) & 0xFF;
        memcpy(q, raw + off, n); q += n;
        for (size_t i = 0; i < n; i++) { a = (a + raw[off + i]) % 65521; b = (b + a) % 65521; }
    }
    png_be32(q, (b << 16) | a); q += 4;
    FILE *f = fopen(path, "wb");
    if (!f) { free(raw); free(z); return false; }
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    uint8_t ihdr[13]; png_be32(ihdr, s->w); png_be32(ihdr + 4, s->h); ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    fwrite(sig, 1, 8, f); png_chunk(f, "IHDR", ihdr, 13); png_chunk(f, "IDAT", z, (size_t)(q - z)); png_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(z);
    return true;
}

// ------------------------------------------------------------------ LASRecomp app layer (same stand-ins as shim_win.c)
// Fullscreen is the default on Linux (the Deck, a TV); TOTK_FULLSCREEN=0 or LARECOMP_WINDOW=WxH start windowed.
Settings g_settings = {.aspect = ASPECT_ORIGINAL, .fullscreen = true, .res_mode = RES_MATCH, .upscaler = UPSCALE_BILINEAR,
                       .antialias = AA_OFF};
void settings_load(void) {
    const char *f = getenv("TOTK_FULLSCREEN");
    g_settings.fullscreen = f ? (*f && strcmp(f, "0") != 0) : true;
    // TOTK_AA=smaa / TOTK_UPSCALE=fsr: as on Windows (shim_win.c settings_load)
    const char *aa = getenv("TOTK_AA"), *up = getenv("TOTK_UPSCALE");
    if (aa && !strcmp(aa, "smaa")) g_settings.antialias = AA_SMAA;
    if (up && !strcmp(up, "fsr")) g_settings.upscaler = UPSCALE_FSR1;
}
void plat_terminate(void) { _exit(0); }
int rt_reboot_requested(void) { return 0; }
void rt_window_icon(SDL_Window *w) { (void)w; }
const char *game_level_name(void) { return "-"; }
void menu_render(void *cb, uint32_t w, uint32_t h) { (void)cb, (void)w, (void)h; }
bool menu_handle_event(const SDL_Event *e) { (void)e; return false; }
void menu_apply_window_requests(SDL_Window *w) { (void)w; }
void menu_test_open(void) {}
void input_handle_event(const SDL_Event *e) { (void)e; }
void input_frame(uint64_t frame) { (void)frame; }
bool input_user_blocked(const SDL_Event *e) { (void)e; return false; }
void ui_size(int *w, int *h) { int a = 0, b = 0; SDL_GetWindowSizeInPixels(ui_window(), &a, &b); *w = a; *h = b; }
int texpatch_identify(const NvnTexture *t) { (void)t; return -1; }
int texpatch_bottle_kind(int id) { (void)id; return 0; }
bool texpatch_is_item_cursor(int id) { (void)id; return false; }
uint32_t texpatch_generation(void) { return 0; }
void texpatch_apply(int id, uint32_t lv, uint8_t *b, uint32_t w, uint32_t h, uint32_t row) { (void)id, (void)lv, (void)b, (void)w, (void)h, (void)row; }
int texpatch_heart_kind(int id) { (void)id; return 0; }  // (was -1: truthy, so any texture with tp > 0 tinted its draw)
void texpatch_heart_tint(int kind, float tint[8]) { (void)kind; for (int i = 0; i < 8; i++) tint[i] = 1.0f; }
#endif

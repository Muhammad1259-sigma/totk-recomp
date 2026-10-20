// TotK: Win32 implementation of the SDL3 subset in shim/SDL3/SDL.h, plus the LASRecomp app-layer stand-ins
// (settings, menu, input, texpatch, kernel heap) that runtime/nvn/vulkan/*.c call.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../shim/las.h"

struct SDL_Window { HWND hwnd; };
static SDL_Window g_win;
static volatile LONG g_quit;
static char g_err[256] = "";

// SDL_SetWindowTitle from the present thread: the text goes through g_title and a posted message, so the window
// thread sets it while pumping (SetWindowTextW from another thread would block on it)
#define WM_TOTK_TITLE (WM_APP + 1)
static SRWLOCK g_title_lock = SRWLOCK_INIT;
static wchar_t g_title[256];

// Borderless fullscreen on the window's monitor (user 2026-10-15: "add a fulscreen option"): F11 or Alt+Enter
// toggles, TOTK_FULLSCREEN=1 starts in it. A popup covering the monitor lets the driver flip the swapchain directly
// (no desktop composition step = less latency). The cursor hides over the game while fullscreen.
#define WM_TOTK_FULLSCREEN (WM_APP + 2)
static volatile LONG g_fs;
static WINDOWPLACEMENT g_wp = {sizeof g_wp};
static void apply_fullscreen(HWND h, bool fs) {
    if (fs == (g_fs != 0)) return;
    LONG style = GetWindowLongW(h, GWL_STYLE);
    if (fs) {
        MONITORINFO mi = {sizeof mi};
        GetWindowPlacement(h, &g_wp);
        if (!GetMonitorInfoW(MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST), &mi)) return;
        SetWindowLongW(h, GWL_STYLE, (style & ~WS_OVERLAPPEDWINDOW) | WS_POPUP);
        SetWindowPos(h, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
        SetWindowLongW(h, GWL_STYLE, (style & ~WS_POPUP) | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(h, &g_wp);
        SetWindowPos(h, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    }
    InterlockedExchange(&g_fs, fs);
    fprintf(stderr, "[win] %s\n", fs ? "fullscreen" : "windowed");
}
bool shim_window_fullscreen(void) { return g_fs != 0; }
// on-screen fps readout (device.c): shown while fullscreen (no title bar there), F3 toggles it in either mode
static volatile LONG g_fps_osd = -2;   // -1 = automatic (fullscreen only), 0 off, 1 on (TOTK_FPS_OVERLAY=1 / 0 at start)
bool shim_fps_overlay(void) {
    if (g_fps_osd == -2) { const char *e = getenv("TOTK_FPS_OVERLAY"); InterlockedExchange(&g_fps_osd, e ? (*e != '0') : -1); }
    return g_fps_osd < 0 ? g_fs != 0 : g_fps_osd != 0;
}

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_TOTK_FULLSCREEN: apply_fullscreen(h, w != 0); return 0;
    case WM_KEYDOWN:
        if (w == VK_F11 && !(l & (1 << 30))) { apply_fullscreen(h, !g_fs); return 0; }
        if (w == VK_F3 && !(l & (1 << 30))) { InterlockedExchange(&g_fps_osd, shim_fps_overlay() ? 0 : 1); return 0; }
        break;
    case WM_SYSKEYDOWN:   // Alt+Enter
        if (w == VK_RETURN && (l & (1 << 29)) && !(l & (1 << 30))) { apply_fullscreen(h, !g_fs); return 0; }
        break;
    case WM_SYSCHAR:
        if (w == '\r') return 0;   // (no beep on Alt+Enter)
        break;
    case WM_SETCURSOR:
        if (g_fs && LOWORD(l) == HTCLIENT) { SetCursor(NULL); return TRUE; }
        break;
    case WM_TOTK_TITLE: {
        wchar_t t[256];
        AcquireSRWLockShared(&g_title_lock);
        memcpy(t, g_title, sizeof t);
        ReleaseSRWLockShared(&g_title_lock);
        SetWindowTextW(h, t);
        return 0;
    }
    case WM_CLOSE: InterlockedExchange(&g_quit, 1); return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

bool SDL_Init(uint32_t flags) { (void)flags; SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2); return true; }
bool SDL_SetHint(const char *n, const char *v) { (void)n, (void)v; return true; }
const char *SDL_GetError(void) { return g_err; }

SDL_Window *SDL_CreateWindow(const char *title, int w, int h, SDL_WindowFlags flags) {
    HINSTANCE inst = GetModuleHandleW(NULL);
    WNDCLASSEXW wc = {sizeof wc, CS_HREDRAW | CS_VREDRAW, wndproc, 0, 0, inst, LoadIconW(NULL, (LPCWSTR)IDI_APPLICATION),
                      LoadCursorW(NULL, (LPCWSTR)IDC_ARROW), NULL, NULL, L"TotKRecomp", NULL};
    RegisterClassExW(&wc);
    RECT r = {0, 0, w, h};
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&r, style, FALSE);
    wchar_t wt[128];
    MultiByteToWideChar(CP_UTF8, 0, title, -1, wt, 128);
    g_win.hwnd = CreateWindowExW(0, L"TotKRecomp", wt, style, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                                 NULL, NULL, inst, NULL);
    if (!g_win.hwnd) { snprintf(g_err, sizeof g_err, "CreateWindowExW failed (%lu)", GetLastError()); return NULL; }
    // TOTK_NO_FOCUS=1 (tools/run_test.sh; user 2026-10-18 "make sure they open up without focus"): shown behind the other
    // windows, not activated (not minimized: a minimized window has no swapchain extent, so nothing is presented)
    const char *nf = getenv("TOTK_NO_FOCUS");
    if (flags & SDL_WINDOW_HIDDEN) {}
    else if (nf && *nf && strcmp(nf, "0")) SetWindowPos(g_win.hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    else ShowWindow(g_win.hwnd, SW_SHOW);
    return &g_win;
}
bool SDL_SetWindowFullscreen(SDL_Window *w, bool fs) {
    return w && w->hwnd && PostMessageW(w->hwnd, WM_TOTK_FULLSCREEN, fs, 0) != 0;
}
bool SDL_SetWindowTitle(SDL_Window *w, const char *title) {
    if (!w || !w->hwnd) return false;
    AcquireSRWLockExclusive(&g_title_lock);
    MultiByteToWideChar(CP_UTF8, 0, title, -1, g_title, 256);
    g_title[255] = 0;
    ReleaseSRWLockExclusive(&g_title_lock);
    return PostMessageW(w->hwnd, WM_TOTK_TITLE, 0, 0) != 0;
}
bool SDL_GetWindowSizeInPixels(SDL_Window *w, int *pw, int *ph) {
    RECT r; GetClientRect(w->hwnd, &r);
    *pw = r.right - r.left; *ph = r.bottom - r.top;
    return true;
}
const char *const *SDL_Vulkan_GetInstanceExtensions(Uint32 *count) {
    static const char *const ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    *count = 2;
    return ext;
}
bool SDL_Vulkan_CreateSurface(SDL_Window *w, VkInstance inst, const VkAllocationCallbacks *a, VkSurfaceKHR *out) {
    VkWin32SurfaceCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, .hinstance = GetModuleHandleW(NULL),
                                      .hwnd = w->hwnd};
    VkResult r = vkCreateWin32SurfaceKHR(inst, &ci, a, out);
    if (r != VK_SUCCESS) snprintf(g_err, sizeof g_err, "vkCreateWin32SurfaceKHR %d", r);
    return r == VK_SUCCESS;
}
// pumps this thread's window messages; returns one event at a time (only quit for now)
bool SDL_WaitEventTimeout(SDL_Event *e, int32_t ms) {
    if (InterlockedExchange(&g_quit, 0)) { e->type = SDL_EVENT_QUIT; return true; }
    MsgWaitForMultipleObjects(0, NULL, FALSE, ms < 0 ? INFINITE : (DWORD)ms, QS_ALLINPUT);
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    if (InterlockedExchange(&g_quit, 0)) { e->type = SDL_EVENT_QUIT; return true; }
    return false;
}
Uint64 SDL_GetTicksNS(void) { return plat_now_ns(); }
Uint64 SDL_GetTicks(void) { return plat_now_ns() / 1000000u; }
void SDL_Delay(Uint32 ms) { Sleep(ms); }
int SDL_GetNumLogicalCPUCores(void) { SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }
// Surfaces only carry RGBA8 pixels (SDL_PIXELFORMAT_ABGR8888 = R,G,B,A bytes) for the PNG writer below.
struct SDL_Surface { int w, h, pitch; const uint8_t *px; };
SDL_Surface *SDL_CreateSurfaceFrom(int w, int h, uint32_t f, void *p, int pitch) {
    (void)f;
    SDL_Surface *s = malloc(sizeof *s);
    s->w = w; s->h = h; s->pitch = pitch; s->px = p;
    return s;
}
// Minimal PNG: 8-bit RGBA, zlib with stored (uncompressed) deflate blocks. Big files, no dependencies.
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
bool SDL_SavePNG(SDL_Surface *s, const char *path) {
    if (!s) { snprintf(g_err, sizeof g_err, "no surface"); return false; }
    size_t row = (size_t)s->w * 4 + 1, raw_n = row * s->h;
    uint8_t *raw = malloc(raw_n);
    for (int y = 0; y < s->h; y++) { raw[y * row] = 0; memcpy(raw + y * row + 1, s->px + (size_t)y * s->pitch, row - 1); }
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
    if (!f) { snprintf(g_err, sizeof g_err, "cannot open %s", path); free(raw); free(z); return false; }
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    uint8_t ihdr[13]; png_be32(ihdr, s->w); png_be32(ihdr + 4, s->h); ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    fwrite(sig, 1, 8, f); png_chunk(f, "IHDR", ihdr, 13); png_chunk(f, "IDAT", z, (size_t)(q - z)); png_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(z);
    return true;
}
void SDL_DestroySurface(SDL_Surface *s) { free(s); }

// ---- LASRecomp app layer
Settings g_settings = {.aspect = ASPECT_ORIGINAL, .fullscreen = false, .res_mode = RES_MATCH, .upscaler = UPSCALE_BILINEAR,
                       .antialias = AA_OFF};
void settings_load(void) {
    const char *f = getenv("TOTK_FULLSCREEN");
    g_settings.fullscreen = f && *f && strcmp(f, "0") != 0;
    // TOTK_AA=smaa: SMAA 1x on the presented image (exec.c post_process); TOTK_UPSCALE=fsr: FSR 1 (EASU + RCAS) instead
    // of bilinear when the window is larger than the image
    const char *aa = getenv("TOTK_AA"), *up = getenv("TOTK_UPSCALE");
    if (aa && !strcmp(aa, "smaa")) g_settings.antialias = AA_SMAA;
    if (up && !strcmp(up, "fsr")) g_settings.upscaler = UPSCALE_FSR1;
}
void plat_terminate(void) { TerminateProcess(GetCurrentProcess(), 0); }
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
void ui_size(int *w, int *h) { int a, b; SDL_GetWindowSizeInPixels(&g_win, &a, &b); *w = a; *h = b; }
int texpatch_identify(const NvnTexture *t) { (void)t; return -1; }
int texpatch_bottle_kind(int id) { (void)id; return 0; }
bool texpatch_is_item_cursor(int id) { (void)id; return false; }
uint32_t texpatch_generation(void) { return 0; }
void texpatch_apply(int id, uint32_t lv, uint8_t *b, uint32_t w, uint32_t h, uint32_t row) { (void)id, (void)lv, (void)b, (void)w, (void)h, (void)row; }
int texpatch_heart_kind(int id) { (void)id; return 0; }  // (was -1: truthy, so any texture with tp > 0 tinted its draw)
void texpatch_heart_tint(int kind, float tint[8]) { (void)kind; for (int i = 0; i < 8; i++) tint[i] = 1.0f; }

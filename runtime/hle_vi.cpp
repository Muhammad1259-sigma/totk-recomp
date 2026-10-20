// HLE: nn::vi (display). One default display with one layer; the NVN window draws through our own backend, so the
// display, layer and native window are opaque tokens. The display's vsync SystemEvent is signalled by a host thread
// at 60 Hz (auto-clear): the game's main loop waits on it every frame (run41-45 watchdog: main thread parked in
// WaitSystemEvent at 0x7100a922bc on this event while GetDisplayVsyncEvent was unimplemented).
#include "hle.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#define NOMINMAX
#include <windows.h>

void hle_sysevent_init(u64 ev, bool autoclear);
void hle_sysevent_signal(u64 ev);

static u64 g_display_tok[4], g_layer_tok[4], g_window_tok[8];   // stable addresses handed to the game
static std::atomic<u64> g_vsync_ev{0};
static std::atomic<u64> g_vsync_count{0};

static std::mutex g_vsync_mtx;
static std::condition_variable g_vsync_cv;

// The tick wait: a high-resolution waitable timer (Windows 10 1803+; ~0.5 ms precision without raising the global
// timer resolution). std::this_thread::sleep_until rides the default 15.6 ms scheduler tick, so ticks could land
// up to a tick late (run194: 60.0 fps every 2-s window but present gaps up to 35 ms). TOTK_VSYNC_SLEEP=1: the old
// sleep_until (A/B). "[vsync]" logs the tick interval spread every 10 s.
static void vsync_wait_until(std::chrono::steady_clock::time_point t) {
    static thread_local HANDLE timer = getenv("TOTK_VSYNC_SLEEP") ? nullptr   // (per thread: vsync tick + TOTK_FPS > 60 pacing)
        : CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    auto now = std::chrono::steady_clock::now();
    if (t <= now) return;
    if (!timer) { std::this_thread::sleep_until(t); return; }
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)(std::chrono::duration_cast<std::chrono::nanoseconds>(t - now).count() / 100);  // relative, 100 ns
    if (!SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) { std::this_thread::sleep_until(t); return; }
    WaitForSingleObject(timer, INFINITE);
}

static void vsync_thread() {
    using clk = std::chrono::steady_clock;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    auto next = clk::now(), prev = next;
    double gmin = 1e9, gmax = 0;
    u64 ticks = 0;
    for (;;) {
        next += std::chrono::nanoseconds(16666667);
        vsync_wait_until(next);
        auto now = clk::now();
        double g = std::chrono::duration<double, std::milli>(now - prev).count();
        prev = now;
        gmin = std::min(gmin, g); gmax = std::max(gmax, g);
        if (++ticks % 600 == 0) {
            printf("[vsync] %llu ticks, interval min %.2f max %.2f ms (%s)\n", (unsigned long long)ticks, gmin, gmax,
                   getenv("TOTK_VSYNC_SLEEP") ? "sleep_until" : "high-res timer");
            gmin = 1e9; gmax = 0;
        }
        if (u64 ev = g_vsync_ev.load()) {
            hle_sysevent_signal(ev);
            { std::lock_guard<std::mutex> l(g_vsync_mtx); g_vsync_count++; }
            g_vsync_cv.notify_all();
        }
    }
}

// Present interval (nvnWindowSetPresentInterval; TotK sets 2 = 30 fps): on the Switch each presented frame stays on
// screen `interval` vblanks and the game blocks until its buffer comes back. Called by nvnQueuePresentTexture: waits
// until `interval` vsyncs have passed since the previous present (a late frame resyncs instead of catching up).
// TOTK_UNLOCK_FPS=1 turns pacing off (measures headroom).
// TOTK_SCAN_VFR=<present>: at that present, scan main's .data/.bss and the committed heap for the frame-timing
// struct (BotW's ksys::VFR keeps {u32 interval, f32 frameRate = 60 / interval, f32 frameTime = 1 / frameRate};
// TotK presents at interval 2 -> 30.0f, 1/30f adjacent) and log each hit with the words around it.
extern "C" u64 g_rt_heap_base;
extern "C" u64 kernel_heap_size(void);
extern "C" void rt_watch_write(u64 addr);
static void vfr_scan(void) {
    struct R { u64 a, b; } rs[2] = {{0x7100000000ull + 68026368, 0x7100000000ull + 73817184},
                                    {g_rt_heap_base, g_rt_heap_base + kernel_heap_size()}};
    int hits = 0;
    for (const R &r : rs) {
        if (!r.a || r.b <= r.a) continue;
        // TOTK_SCAN_VFR_ANY=1: every 1/30f (0x3D088889) on its own, not only after 30.0f
        static const bool any = getenv("TOTK_SCAN_VFR_ANY") != nullptr;
        for (u64 p = r.a & ~3ull; p + 8 <= r.b && hits < 80; p += 4) {
            const u32 *w = (const u32 *)p;
            if (any ? w[1] != 0x3D088889u : (w[0] != 0x41F00000u || w[1] != 0x3D088889u)) continue;
            hits++;
            printf("[vfr] 30.0f, 1/30f at %llx (%s):", (unsigned long long)p, r.a == rs[0].a ? "main data" : "heap");
            const u32 *b = (const u32 *)(p - 16);
            for (int i = 0; i < 24; i++) printf(" %08x", b[i]);
            printf("\n");
        }
    }
    printf("[vfr] scan done: %d hits (main data %llx-%llx, heap %llx +%llx)\n", hits, (unsigned long long)rs[0].a,
           (unsigned long long)rs[0].b, (unsigned long long)rs[1].a, (unsigned long long)(rs[1].b - rs[1].a));
    // TOTK_VFR_DUMP=<heap offset hex>: dump 0x400 bytes before/after that heap offset, and every 8-byte pointer into
    // that +-0x400 window held in main .data/.bss or the first 256 MiB of the heap (who owns the block)
    if (const char *e = getenv("TOTK_VFR_DUMP")) {
        u64 at = rs[1].a + strtoull(e, nullptr, 16), lo = at - 0x400, hi = at + 0x400;
        for (u64 p = lo; p < hi; p += 32) {
            const u32 *w = (const u32 *)p;
            printf("[vfr] %llx (+%llx):", (unsigned long long)p, (unsigned long long)(p - rs[1].a));
            for (int i = 0; i < 8; i++) printf(" %08x", w[i]);
            printf("\n");
        }
        int n = 0;
        for (int k = 0; k < 2; k++) {
            u64 a = rs[k].a, b = k ? rs[k].a + std::min<u64>(rs[k].b - rs[k].a, 256ull << 20) : rs[k].b;
            for (u64 p = a; p + 8 <= b && n < 60; p += 8) {
                u64 v = *(const u64 *)p;
                if (v < lo || v >= hi || (p >= lo && p < hi)) continue;
                n++;
                printf("[vfr] ptr at %llx (%s +%llx) -> %llx (block %+lld)\n", (unsigned long long)p, k ? "heap" : "main",
                       (unsigned long long)(p - (k ? rs[1].a : 0x7100000000ull)), (unsigned long long)v, (long long)(v - at));
            }
        }
    }
}

extern "C" volatile int g_dpad_switch_timing;   // hle_hid.cpp: a quick menu may open (d-pad held) -> Switch timing
extern "C" int vk_precompile_warmup_active(void);   // nvn/vulkan/exec.c: TOTK_WARMUP until "play now"
extern "C" void vi_present_pace(int interval) {
    static const bool off = getenv("TOTK_UNLOCK_FPS") != nullptr;
    static u64 last;
    {
        static const long scan_at = getenv("TOTK_SCAN_VFR") ? atol(getenv("TOTK_SCAN_VFR")) : 0;
        static long presents;
        if (++presents == scan_at) vfr_scan();
    }
    {   // TOTK_VFR_SET=<delta frames>: every present, write {d, d / 30} into the 3 per-core {delta frame, delta time}
        // pairs main+0x4661598 points at (run177: {1.0, 1/30} each at 30 fps); logs once per 60 presents whether the
        // game rewrote them since the last present (A/B lever test for frame-rate independent speed)
        static const long watch_at = getenv("TOTK_VFR_WATCH") ? atol(getenv("TOTK_VFR_WATCH")) : 0;  // =<present>: hardware
        static long wn;                                                                               // write watch on pair 0
        if (watch_at && ++wn == watch_at) {
            u64 p = *(const u64 *)(0x7100000000ull + 0x4661598);
            if (p) rt_watch_write(p);
        }
        // TOTK_WATCH=<hex address>: hardware write watch on any address from present 30, re-armed every 300 presents so
        // threads created later are covered too (run224: who overwrites NVN pool 1077812060's handle with -1)
        static const u64 watch_addr = getenv("TOTK_WATCH") ? strtoull(getenv("TOTK_WATCH"), nullptr, 16) : 0;
        static long wa;
        if (watch_addr && ++wa >= 30 && (wa - 30) % 300 == 0) rt_watch_write(watch_addr);
        static const char *vs = getenv("TOTK_VFR_SET");
        static long n;
        static float last_d = -1;
        if (vs) {
            float d = (float)atof(vs);
            u64 p = *(const u64 *)(0x7100000000ull + 0x4661598);
            if (p >= g_rt_heap_base && p < g_rt_heap_base + kernel_heap_size()) {
                float *f = (float *)p;
                if (++n % 60 == 0)
                    printf("[vfr] pairs before write: %g %g | %g %g | %g %g (last written %g)\n", f[0], f[1], f[2], f[3], f[4], f[5], last_d);
                for (int i = 0; i < 3; i++) { f[i * 2] = d; f[i * 2 + 1] = d / 30.0f; }
                last_d = d;
            }
        }
    }
    // TOTK_FPS=60: pace at every vsync (interval 1) instead of the game's 2 (needs the speed fix above for 100 % speed)
    static const int fps = getenv("TOTK_FPS") ? atoi(getenv("TOTK_FPS")) : 0;
    // TOTK_WARMUP=1 (shader warm-up only, tools/linux/warmup.sh, 2026-10-17): the game is paced at 10 fps so its own threads
    // (and their share of the Steam Deck's power budget) go to the shader compile workers; only the title screen runs.
    // ("play now" on the warm-up screen ends this pacing: exec.c vk_precompile_play_now)
    if (vk_precompile_warmup_active()) {
        using clk = std::chrono::steady_clock;
        static clk::time_point wnext;
        const auto period = std::chrono::milliseconds(100);
        auto now = clk::now();
        if (wnext.time_since_epoch().count() == 0 || now > wnext + period) wnext = now;
        else vsync_wait_until(wnext);
        wnext += period;
        return;
    }
    if (off) return;
    // TOTK_FPS=90/120/144... (above the 60 Hz vsync tick) or 31..59 (e.g. 40 / 45 / 50 for a Steam Deck screen set to that
    // refresh rate): pace on the clock (high-res timer); a late frame
    // (a quick menu, d-pad held: Switch timing below like at 60 fps, else its background goes black; 2026-10-17)
    if ((fps > 60 || (fps > 30 && fps < 60)) && !g_dpad_switch_timing) {
        using clk = std::chrono::steady_clock;   // resyncs instead of catching up
        static clk::time_point next;
        const auto period = std::chrono::nanoseconds(1000000000ll / fps);
        auto now = clk::now();
        if (next.time_since_epoch().count() == 0 || now > next + period) next = now;
        else vsync_wait_until(next);
        next += period;
        return;
    }
    if (fps >= 60 && interval > 1 && !g_dpad_switch_timing) interval = 1;
    // (Switch timing = 30 fps: with Dynamic resolution on, the scaled GPU clock puts TotK in its interval-3 mode for the
    // whole session (user 2026-10-19 logs: DYNRES 1 -> interval 3, DYNRES 0 -> 2), and the quick menus ran at 20 fps)
    if (g_dpad_switch_timing && interval > 2) interval = 2;
    if (interval < 1 || !g_vsync_ev.load()) return;
    std::unique_lock<std::mutex> l(g_vsync_mtx);
    u64 want = last + (u64)interval;
    if (g_vsync_count < want) g_vsync_cv.wait_for(l, std::chrono::milliseconds(100), [&] { return g_vsync_count >= want; });
    last = g_vsync_count;
}

// ---------------------------------------------------------------- game speed independent of frame rate
// TotK's frame-delta setter main+0x7a83e4 (recompiled with gen.py --hook 7a83e4; run179 hardware write watch): called
// once per game frame as f(this, w1 = present interval 2). It stores ratio = interval / 2 and frameTime = interval / 60
// at this+8, then deltaFrame = ratio * speed multiplier, deltaTime = deltaFrame / 30 at this+0 and per core at this+16,
// +56, +96. With the fixed interval every frame advanced 1/30 s of game time: 2.06x speed at 62 fps (runs 174/175).
// The hook scales those outputs by k = measured frame time / (interval / 60). Astro Bot lessons: the raw frame time is
// clamped to [1/240, 1/20] s (a load stall must not inject a huge step, physics dislikes > 1/20 s) and smoothed with an
// EMA of 0.08 (instant jumps crashed Astro). TOTK_SPEED_FIX=0: old behaviour (fixed step); [speedfix] logs k every
// 300 calls.
void f_7a83e4_orig(Ctx* __restrict c);
void hook_f_7a83e4(Ctx* __restrict c) {
    static const bool on = !getenv("TOTK_SPEED_FIX") || strcmp(getenv("TOTK_SPEED_FIX"), "0");
    u64 self = c->x[0];
    u32 interval = (u32)c->x[1];
    f_7a83e4_orig(c);
    if (interval >= 1 && interval <= 4) {   // [timemult] the game's time multiplier (deltaFrame / ratio) and call gaps, on change:
        static u64 tself; static float tm = -1; static std::chrono::steady_clock::time_point tl; static int tlogs;   // which menus
        const float *g = (const float *)self;                                       // pause or slow the world (2026-10-19)
        auto tn = std::chrono::steady_clock::now();
        if (!tself) tself = self;
        if (self == tself) {
            float m = g[2] > 0 ? g[0] / g[2] : -1;
            double gap = tm < 0 ? 0 : std::chrono::duration<double, std::milli>(tn - tl).count();
            if ((fabsf(m - tm) > 0.01f || gap > 100) && tlogs++ < 400)
                printf("[timemult] %.3f -> %.3f (gap %.0f ms, dpad timing %d)\n", tm, m, gap, (int)g_dpad_switch_timing);
            tm = m; tl = tn;
        }
    }
    if (!on || interval < 1 || interval > 4) return;
    static struct { u64 self; std::chrono::steady_clock::time_point last; double ema; u64 calls; } st[8];
    int i = 0;
    while (i < 8 && st[i].self && st[i].self != self) i++;
    if (i == 8) return;
    auto now = std::chrono::steady_clock::now();
    double nominal = interval / 60.0;
    if (!st[i].self) { st[i].self = self; st[i].last = now; st[i].ema = nominal; return; }
    double dt = std::chrono::duration<double>(now - st[i].last).count();
    st[i].last = now;
    if (g_dpad_switch_timing) {   // (game paced at its own interval: no scaling; resume at 60)
        st[i].ema = 1.0 / 60;
        if (interval > 2) {   // (paced at 2 instead, above: the outputs interval 2 gives, so ratio 1 as on the Switch)
            float *f = (float *)self;
            static const int all[] = {0, 1, 4, 5, 14, 15, 24, 25, 2, 3};
            for (int j : all) f[j] *= 2.0f / interval;
        }
        return;
    }
    dt = std::min(std::max(dt, 1.0 / 240), 1.0 / 20);
    st[i].ema += 0.08 * (dt - st[i].ema);
    float k = (float)(st[i].ema / nominal);
    if (++st[i].calls % 300 == 0)
        printf("[speedfix] timer %d (%llx): interval %u raw dt %.2f ms ema %.2f ms -> k %.3f\n", i, (unsigned long long)self,
               interval, dt * 1e3, st[i].ema * 1e3, k);
    float *f = (float *)self;
    static const int idx[] = {0, 1, 4, 5, 14, 15, 24, 25, 2, 3};  // this+0/+4, +16/+20, +56/+60, +96/+100, (+8/+12)
    // A/B modes: 2 = deltas only (keep ratio + frame time), 3 = deltas + frame time (keep ratio), 4 = deltas + ratio
    static const int mode = getenv("TOTK_SPEED_FIX") ? atoi(getenv("TOTK_SPEED_FIX")) : 1;
    // mode 5: as 1, but the ratio stays unscaled while the game's time multiplier (deltaFrame / ratio, read before
    // scaling) is below 1 = world paused or slowed (weapon quick menu: a ratio of 0.5 made TotK skip the scene in the
    // frame that captures the menu background -> black menu, w4/w8 runs 2026-10-17)
    bool keep_ratio = false;
    if (mode == 5 && f[2] > 0) {
        float mult = f[0] / f[2];
        keep_ratio = mult < 0.99f;
        static bool was; static int logs;
        if (keep_ratio != was && logs++ < 20) printf("[speedfix] time multiplier %.3f: ratio %s\n", mult, keep_ratio ? "kept" : "scaled");
        was = keep_ratio;
    }
    for (int j = 0; j < 8; j++) f[idx[j]] *= k;
    if (mode != 2 && mode != 3 && !keep_ratio) f[2] *= k;
    if (mode != 2 && mode != 4) f[3] *= k;
}

HLE_AS(vi_Initialize, "_ZN2nn2vi10InitializeEv") {}
HLE_AS(vi_OpenDefaultDisplay, "_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE") {
    *(u64*)c->x[0] = (u64)g_display_tok; c->x[0] = 0;
}
HLE_AS(vi_CreateLayer, "_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE") {
    *(u64*)c->x[0] = (u64)g_layer_tok; c->x[0] = 0;
}
HLE_AS(vi_GetNativeWindow, "_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE") {
    *(u64*)c->x[0] = (u64)g_window_tok; c->x[0] = 0;
}
HLE_AS(vi_GetDisplayVsyncEvent, "_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE") {
    u64 ev = c->x[0];
    hle_sysevent_init(ev, true);
    static std::once_flag once;
    g_vsync_ev = ev;
    std::call_once(once, [] { std::thread(vsync_thread).detach(); });
    printf("[vi] vsync event %llx (60 Hz host timer)\n", ev);
    c->x[0] = 0;
}

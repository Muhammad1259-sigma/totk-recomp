// HLE: nn::os threads, TLS slots, synchronisation primitives, ticks.
// Host state lives in maps keyed by the guest object's address (not inside the guest struct), so objects the game
// set up with static initializers (never passed to Initialize*) still work; Initialize* (re)creates the state.
#include "hle.h"
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <unordered_map>
#include <deque>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>

using Clock = std::chrono::steady_clock;
static const u64 kTickHz = 19200000;            // Switch system tick (19.2 MHz)

// Guest object address -> host object. get() first looks in a small per-thread cache (run362/363 profiles: LockMutex /
// UnlockMutex spent most of their own time in Registry<HMutex>::get, a shared_mutex + hash lookup per call); any
// reset / drop bumps gen, which empties every thread's cache before a deleted object could be returned.
// Behind the cache sits a lock-free open-addressing index (d49_perf on the Deck, 2026-10-17: 84 % of hle_os_LockMutex's
// self time, 2.8 % of all samples, was the shared_mutex + unordered_map walk after a cache miss = two atomic RMWs on a line
// every worker thread shares + two dependent cache misses). Readers probe it without a lock; inserts and removals happen
// under the unique lock. Key 0 = empty, 1 = removed. Objects that do not fit (load > 3/4, or 32 probes) live in the map only.
template <class T> struct Registry {
    std::shared_mutex m; std::unordered_map<u64, T*> map;
    std::atomic<u64> gen{1};
    struct Slot { u64 a, gen; T* p; };
    static constexpr u32 kBits = 17; static constexpr u64 kCap = 1ull << kBits;
    struct E { std::atomic<u64> a; std::atomic<T*> p; };
    E* tab = (E*)calloc(kCap, sizeof(E)); u64 used = 0;
    static u64 hidx(u64 a) { return (a * 0x9E3779B97F4A7C15ull) >> (64 - kBits); }
    T* find_fast(u64 a) {
        for (u64 i = hidx(a), n = 0; n < 32; i = (i + 1) & (kCap - 1), n++) {
            u64 k = tab[i].a.load(std::memory_order_acquire);
            if (k == a) return tab[i].p.load(std::memory_order_acquire);
            if (k == 0) return nullptr;
        }
        return nullptr;
    }
    void put_fast(u64 a, T* p) {      // under the unique lock, a not in the index yet
        if (used >= kCap / 4 * 3) return;
        for (u64 i = hidx(a), n = 0; n < 32; i = (i + 1) & (kCap - 1), n++) {
            u64 k = tab[i].a.load(std::memory_order_relaxed);
            if (k > 1) continue;
            tab[i].p.store(p, std::memory_order_relaxed); tab[i].a.store(a, std::memory_order_release);
            if (k == 0 && ++used >= 4096 && !(used & (used - 1))) fprintf(stderr, "[os] registry %zu-byte objects: %llu indexed\n", sizeof(T), (unsigned long long)used);
            return;
        }
    }
    T* get(u64 a) {
        static thread_local Slot cache[64];
        Slot& s = cache[(a >> 3 ^ a >> 9) & 63];
        u64 g = gen.load(std::memory_order_acquire);
        if (s.a == a && s.gen == g) return s.p;
        T* p = lookup(a);
        s = {a, g, p};
        return p;
    }
    T* lookup(u64 a) {
        if (T* p = find_fast(a)) return p;
        { std::shared_lock<std::shared_mutex> l(m); auto it = map.find(a); if (it != map.end()) return it->second; }
        std::unique_lock<std::shared_mutex> l(m); T*& p = map[a]; if (!p) { p = new T(); put_fast(a, p); } return p;
    }
    // (re-initialised in place: cached pointers stay valid, so no gen bump. A bump on every Initialize* flushed every
    // thread's cache and sent their locks through the shared_mutex + map: hle_os_LockMutex 1.3-2 % of all samples on the
    // Deck worker threads, d45, 2026-10-17)
    T* reset(u64 a) {
        std::unique_lock<std::shared_mutex> l(m); T*& p = map[a];
        if (p) { p->~T(); new (p) T(); } else { p = new T(); put_fast(a, p); }
        return p;
    }
    void drop(u64 a) {
        std::unique_lock<std::shared_mutex> l(m); gen++;
        for (u64 i = hidx(a), n = 0; n < 32; i = (i + 1) & (kCap - 1), n++) {
            u64 k = tab[i].a.load(std::memory_order_relaxed);
            if (k == a) { tab[i].p.store(nullptr, std::memory_order_relaxed); tab[i].a.store(1, std::memory_order_release); break; }
            if (k == 0) break;
        }
        auto it = map.find(a); if (it != map.end()) { delete it->second; map.erase(it); }
    }
};

// ---------------------------------------------------------------- threads
struct HThread {
    u64 type = 0, fn = 0, arg = 0, stack = 0, stack_size = 0;
    int prio = 16, core = -1; u64 core_mask = 7;
    HANDLE h = nullptr; DWORD tid = 0; std::string name;
};
static Registry<HThread> g_threads;
static thread_local HThread* t_self;
static std::atomic<u64> g_thread_ids{1};

static const int kTlsSlots = 64;
static std::atomic<int> g_tls_next{0};
static u64 g_tls_dtor[kTlsSlots];
static thread_local u64 t_tls[kTlsSlots];

void os_register_main_thread(Ctx* c, u64 stack_lo, u64 stack_size) {
    u64 tt = (u64)_aligned_malloc(0x200, 16); memset((void*)tt, 0, 0x200);   // guest-visible nn::os::ThreadType
    HThread* t = g_threads.get(tt);
    t->type = tt; t->stack = stack_lo; t->stack_size = stack_size; t->prio = 28; t->name = "MainThread";
    t->h = GetCurrentThread(); t->tid = GetCurrentThreadId();
    t_self = t;
    (void)c;
}

#ifndef _WIN32
extern "C" void host_affinity(int kind, unsigned long long mask);   // (linux/plat_posix.cpp, TOTK_AFFINITY; no-op when off)
#endif
static DWORD WINAPI thread_entry(LPVOID p) {
    HThread* t = (HThread*)p; t_self = t;
#ifndef _WIN32
    host_affinity(0, t->core_mask);
#endif
    Ctx* c = rt_new_ctx(t->stack + t->stack_size, nullptr, nullptr);
    rt_bind_thread(c);
    if (!t->name.empty()) {
        std::wstring w(t->name.begin(), t->name.end());
        SetThreadDescription(GetCurrentThread(), w.c_str());
    }
    c->x[0] = t->arg; c->x[30] = 0;
    call_ind(c, t->fn);
    for (int i = 0; i < kTlsSlots; i++)                     // TLS destructors
        if (g_tls_dtor[i] && t_tls[i]) { u64 v = t_tls[i]; t_tls[i] = 0; c->x[0] = v; c->x[30] = 0; call_ind(c, g_tls_dtor[i]); }
    rt_unbind_thread();
    return 0;
}

HLE_AS(os_CreateThread, "_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi") {
    HThread* t = g_threads.reset(c->x[0]);
    t->type = c->x[0]; t->fn = c->x[1]; t->arg = c->x[2]; t->stack = c->x[3]; t->stack_size = c->x[4]; t->prio = (int)c->x[5];
    printf("[os] CreateThread %llx fn=%llx stack=%llx+%llx prio=%d\n", t->type, t->fn, t->stack, t->stack_size, t->prio);
    c->x[0] = 0;
}
HLE_AS(os_StartThread, "_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE") {
    HThread* t = g_threads.get(c->x[0]);
    t->h = CreateThread(nullptr, 256ull << 20, thread_entry, t, STACK_SIZE_PARAM_IS_A_RESERVATION, &t->tid);
    if (!t->h) rt_trap(c, c->x[30], "StartThread: CreateThread failed");
}
HLE_AS(os_WaitThread, "_ZN2nn2os10WaitThreadEPNS0_10ThreadTypeE") {
    HThread* t = g_threads.get(c->x[0]);
    if (t->h) WaitForSingleObject(t->h, INFINITE);
}
HLE_AS(os_DestroyThread, "_ZN2nn2os13DestroyThreadEPNS0_10ThreadTypeE") {
    HThread* t = g_threads.get(c->x[0]);
    if (t->h) { WaitForSingleObject(t->h, INFINITE); CloseHandle(t->h); t->h = nullptr; }
}
HLE_AS(os_SetThreadName, "_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc") {
    HThread* t = g_threads.get(c->x[0]); t->name = c->x[1] ? (const char*)c->x[1] : "";
    if (t->h && c->x[1]) {   // host thread description: debuggers and rt_main's sampling profiler show the guest name
        wchar_t wn[64]; int i = 0;
        for (const char* s = (const char*)c->x[1]; *s && i < 63; s++) wn[i++] = (wchar_t)(unsigned char)*s;
        wn[i] = 0;
        SetThreadDescription(t->h, wn);
    }
}
HLE_AS(os_GetCurrentThread, "_ZN2nn2os16GetCurrentThreadEv") { c->x[0] = t_self ? t_self->type : 0; }
HLE_AS(os_GetThreadId, "_ZN2nn2os11GetThreadIdEPKNS0_10ThreadTypeE") {
    HThread* t = g_threads.get(c->x[0]); c->x[0] = t->tid;
}
HLE_AS(os_GetThreadPriority, "_ZN2nn2os17GetThreadPriorityEPKNS0_10ThreadTypeE") { c->x[0] = (u64)g_threads.get(c->x[0])->prio; }
HLE_AS(os_ChangeThreadPriority, "_ZN2nn2os20ChangeThreadPriorityEPNS0_10ThreadTypeEi") {
    HThread* t = g_threads.get(c->x[0]); int old = t->prio; t->prio = (int)c->x[1]; c->x[0] = (u64)old;
}
// idealCore: >= 0 a core; -1 IdealCoreDontCare, -2 IdealCoreUseDefault (-> lowest core of the mask), -3 IdealCoreNoUpdate.
HLE_AS(os_SetThreadCoreMask, "_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim") {
    HThread* t = g_threads.get(c->x[0]); int ideal = (int)c->x[1];
    if (ideal != -3) t->core = ideal;
    t->core_mask = c->x[2];
    static int n; if (n++ < 64) printf("[os] SetThreadCoreMask %s core %d mask %llx\n", t->name.c_str(), ideal, c->x[2]);
#ifndef _WIN32
    if (t == t_self) host_affinity(0, t->core_mask);   // (other threads: applied when they start, thread_entry)
#endif
}
HLE_AS(os_GetThreadAvailableCoreMask, "_ZN2nn2os26GetThreadAvailableCoreMaskEv") { c->x[0] = 7; }
// The guest's core, not the host CPU: the job system matches workers to cores by this number (f_7a806c: a non-worker
// joins pool W with id = its core; the thread trampoline 0x29bb758 reads it at start). The host CPU number % 3 was
// random per call (run66: main joined as "core 2"). A Switch thread with a one-core mask always reports that core;
// main (no mask set) runs on core 0.
HLE_AS(os_GetCurrentCoreNumber, "_ZN2nn2os20GetCurrentCoreNumberEv") {
    HThread* t = t_self; int core = 0;
    if (t && t->core >= 0) core = t->core;
    else if (t && (t->core_mask & 7)) { u64 m = t->core_mask & 7; while (!(m & (1ull << core))) core++; }
    c->x[0] = (u64)core;
}
HLE_AS(os_GetCurrentStackInfo, "_ZN2nn2os19GetCurrentStackInfoEPmS1_") {
    *(u64*)c->x[0] = t_self ? t_self->stack : 0; *(u64*)c->x[1] = t_self ? t_self->stack_size : 0;
}
static s64 now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
static void evlog(char op, u64 ev, u64 lr);   // (wait-chain ring, below)
// "[sleep]" every ~5 s: SleepThread calls by requested length, requested vs actual time (sleep_for rides the Windows
// timer: a short guest sleep on the game's critical path can come back late; run365: gameplay ~88 fps with no draws at
// all, game threads 13-14 % in SleepThread)
static std::atomic<u64> g_sl_n[4], g_sl_req[4], g_sl_act[4], g_sl_t0;
static std::atomic<u64> g_tw_n, g_tw_req, g_tw_act, g_tw_out;   // timed event waits (ev_wait with a timeout > 0)
// Handoff latency (signal / send -> the blocked waiter running again), "[sleep] ... | ev ... | mq ..." every ~5 s:
// kind 0 = event waits (Event / LightEvent / SystemEvent), 1 = message-queue receives. Buckets <20 us, <100 us, <1 ms, more.
static std::atomic<u64> g_hw_n[2], g_hw_blk[2], g_hw_lat[2], g_hw_b[2][4];
static void hw_note(int k, bool blocked, s64 lat) {
    g_hw_n[k]++;
    if (!blocked) return;
    g_hw_blk[k]++;
    if (lat < 0) lat = 0;
    g_hw_lat[k] += (u64)lat;
    g_hw_b[k][lat < 20000 ? 0 : lat < 100000 ? 1 : lat < 1000000 ? 2 : 3]++;
}
// Guest sleeps at the Switch's precision. sleep_for rounds up to the Windows timer: run366 gameplay, SleepThread(20 us)
// took 1.59 ms on average (~310 calls/s, the game's job waits) and SleepThread(1 ms) 1.75 ms. Below 1 ms: yield until
// the deadline; from 1 ms: a high-resolution waitable timer (~0.5 ms precision). TOTK_SLEEP_PRECISE=0: sleep_for (A/B).
static void guest_sleep_ns(s64 ns) {
    static const bool off = getenv("TOTK_SLEEP_PRECISE") && !strcmp(getenv("TOTK_SLEEP_PRECISE"), "0");
    if (off) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); return; }
    if (ns < 1000000) {
        s64 end = now_ns() + ns;
        do SwitchToThread(); while (now_ns() < end);
        return;
    }
    // (run367: the timer alone made SleepThread(1 ms) 1.44 ms: up to 4 ms, sleep to 0.5 ms before the deadline and yield
    // the rest; the ~100/s background sleeps of ~40 ms keep the plain timer, their 0.5 ms does not matter)
    s64 end = now_ns() + ns;
    s64 early = ns <= 4000000 ? 500000 : 0;
    static thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)((ns - early) / 100);   // relative, 100 ns units
    if (timer && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, INFINITE);
    else std::this_thread::sleep_for(std::chrono::nanoseconds(ns - early));
    if (early) while (now_ns() < end) SwitchToThread();
}
HLE_AS(os_SleepThread, "_ZN2nn2os11SleepThreadENS_8TimeSpanE") {
    s64 ns = timespan_ns(c->x[0]);
    s64 t0 = now_ns();
    if (ns <= 0) SwitchToThread(); else guest_sleep_ns(ns);
    s64 t1 = now_ns();
    int b = ns <= 0 ? 0 : ns < 1000000 ? 1 : ns <= 2000000 ? 2 : 3;
    g_sl_n[b]++; g_sl_req[b] += ns > 0 ? (u64)ns : 0; g_sl_act[b] += (u64)(t1 - t0);
    u64 last = g_sl_t0.load();
    if (!last) g_sl_t0.compare_exchange_strong(last, (u64)t1);
    else if ((u64)t1 - last > 5000000000ull && g_sl_t0.compare_exchange_strong(last, (u64)t1)) {
        static const char* nm[4] = {"yield", "<1ms", "1-2ms", ">2ms"};
        char buf[1024]; int o = snprintf(buf, sizeof buf, "[sleep] %.1f s:", ((u64)t1 - last) / 1e9);
        for (int i = 0; i < 4; i++) {
            u64 n = g_sl_n[i].exchange(0), rq = g_sl_req[i].exchange(0), ac = g_sl_act[i].exchange(0);
            if (n) o += snprintf(buf + o, sizeof buf - o, " %s %llu calls req %.3f act %.3f ms avg (%.0f ms total)", nm[i],
                                 (unsigned long long)n, rq / 1e6 / n, ac / 1e6 / n, ac / 1e6);
        }
        if (u64 n = g_tw_n.exchange(0)) {
            u64 rq = g_tw_req.exchange(0), ac = g_tw_act.exchange(0), to = g_tw_out.exchange(0);
            o += snprintf(buf + o, sizeof buf - o, " | timed waits %llu (%llu timed out) req %.3f act %.3f ms avg", (unsigned long long)n,
                          (unsigned long long)to, rq / 1e6 / n, ac / 1e6 / n);
        }
        static const char* hk[2] = {"ev", "mq"};
        for (int k = 0; k < 2; k++) {
            u64 n = g_hw_n[k].exchange(0), b = g_hw_blk[k].exchange(0), lat = g_hw_lat[k].exchange(0);
            u64 h0 = g_hw_b[k][0].exchange(0), h1 = g_hw_b[k][1].exchange(0), h2 = g_hw_b[k][2].exchange(0), h3 = g_hw_b[k][3].exchange(0);
            if (n) o += snprintf(buf + o, sizeof buf - o, " | %s %llu waits, %llu blocked, wake %.1f us avg (<20us %llu <100us %llu <1ms %llu more %llu)",
                                 hk[k], (unsigned long long)n, (unsigned long long)b, b ? lat / 1e3 / b : 0.0, (unsigned long long)h0,
                                 (unsigned long long)h1, (unsigned long long)h2, (unsigned long long)h3);
        }
        printf("%s\n", buf);
    }
}
HLE_AS(os_YieldThread, "_ZN2nn2os11YieldThreadEv") { SwitchToThread(); }
HLE_AS(os_GetCurrentFiber, "_ZN2nn2os15GetCurrentFiberEv") { c->x[0] = 0; }
HLE_AS(os_SwitchToFiber, "_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE") { rt_trap(c, c->x[30], "SwitchToFiber not implemented"); }

// ---------------------------------------------------------------- TLS slots
HLE_AS(os_AllocateTlsSlot, "_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE") {
    int i = g_tls_next++;
    if (i >= kTlsSlots) rt_trap(c, c->x[30], "out of TLS slots");
    g_tls_dtor[i] = c->x[1]; *(u32*)c->x[0] = (u32)i; c->x[0] = 0;
}
HLE_AS(os_FreeTlsSlot, "_ZN2nn2os11FreeTlsSlotENS0_7TlsSlotE") { g_tls_dtor[(u32)c->x[0] % kTlsSlots] = 0; }
HLE_AS(os_GetTlsValue, "_ZN2nn2os11GetTlsValueENS0_7TlsSlotE") { c->x[0] = t_tls[(u32)c->x[0] % kTlsSlots]; }
HLE_AS(os_SetTlsValue, "_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm") { t_tls[(u32)c->x[0] % kTlsSlots] = c->x[1]; }

// ---------------------------------------------------------------- ticks
static u64 now_ticks() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (u64)((u128)t.QuadPart * kTickHz / (u64)f.QuadPart);
}
u64 rt_cntpct() { return now_ticks(); }      // CNTPCT_EL0 reads in lifted sdk code
HLE_AS(os_GetSystemTick, "_ZN2nn2os13GetSystemTickEv") { c->x[0] = now_ticks(); }
HLE_AS(os_GetSystemTickFrequency, "_ZN2nn2os22GetSystemTickFrequencyEv") { c->x[0] = kTickHz; }
HLE_AS(os_ConvertToTick, "_ZN2nn2os13ConvertToTickENS_8TimeSpanE") { c->x[0] = (u64)((s128)(s64)c->x[0] * kTickHz / 1000000000); }
HLE_AS(os_ConvertToTimeSpan, "_ZN2nn2os17ConvertToTimeSpanENS0_4TickE") { c->x[0] = (u64)((s128)(s64)c->x[0] * 1000000000 / kTickHz); }

// ---------------------------------------------------------------- mutex / condition variable
// Host mutexes live in one arena, and the guest MutexType remembers its host mutex at +0x18: the SDK's internal critical
// section storage (InitializeMutex at sdk 0x127dc0 sets up +0x18 by a call and writes state @0, recursive @1, lock level
// @4, nest count @8, owner @0x10; LockMutex 0x127e20 enters +0x18). With every MutexType entry point HLE'd (main imports
// exactly these; no SDK-internal caller of the real LockMutex) nothing else reads it. A cached value is used only when it
// points into the arena at a mutex whose guest address matches, so a copied struct or stale bytes fall back to the
// Registry. (d50_perf: after the lock-free index, LockMutex was still 1.3 % of all Deck samples, 72 % of it the index
// cache miss.) LARECOMP_MUTEX_INLINE=0: Registry only.
static const size_t kMtxArena = 1u << 18;
struct HMutex;
alignas(64) static unsigned char g_mtx_arena[kMtxArena * 64];
static std::atomic<size_t> g_mtx_next{0};
struct HMutex {
    std::recursive_mutex m; std::atomic<DWORD> owner{0}; int depth = 0; u64 guest = 0;
    static void* operator new(size_t n) {
        size_t i = g_mtx_next.fetch_add(1, std::memory_order_relaxed);
        if (n <= 64 && i < kMtxArena) return g_mtx_arena + i * 64;
        return ::operator new(n);
    }
    static void* operator new(size_t, void* q) { return q; }   // (Registry::reset re-initialises in place)
    static void operator delete(void* q) {
        if ((unsigned char*)q >= g_mtx_arena && (unsigned char*)q < g_mtx_arena + sizeof g_mtx_arena) return;
        ::operator delete(q);
    }
};
static Registry<HMutex> g_mutex;
static int g_mtx_inline = -1;
static HMutex* mtx_get(u64 a) {
    if (g_mtx_inline > 0) {
        u64 q = *(volatile u64*)(a + 0x18) - (u64)(uintptr_t)g_mtx_arena;
        if (q < sizeof g_mtx_arena && !(q & 63)) {
            HMutex* m = (HMutex*)(g_mtx_arena + q);
            if (m->guest == a) return m;
        }
    } else if (g_mtx_inline < 0) {
        const char* e = getenv("LARECOMP_MUTEX_INLINE");
        g_mtx_inline = !(e && *e == '0');
    }
    HMutex* m = g_mutex.get(a);
    if (g_mtx_inline > 0 && (unsigned char*)m >= g_mtx_arena && (unsigned char*)m < g_mtx_arena + sizeof g_mtx_arena) {
        m->guest = a;
        *(volatile u64*)(a + 0x18) = (u64)(uintptr_t)m;
    }
    return m;
}
static void mtx_lock(HMutex* m) { m->m.lock(); m->owner = GetCurrentThreadId(); m->depth++; }
static void mtx_unlock(HMutex* m) { if (--m->depth == 0) m->owner = 0; m->m.unlock(); }
HLE_AS(os_InitializeMutex, "_ZN2nn2os15InitializeMutexEPNS0_9MutexTypeEbi") { g_mutex.reset(c->x[0]); mtx_get(c->x[0]); }
HLE_AS(os_FinalizeMutex, "_ZN2nn2os13FinalizeMutexEPNS0_9MutexTypeE") { }
HLE_AS(os_LockMutex, "_ZN2nn2os9LockMutexEPNS0_9MutexTypeE") { mtx_lock(mtx_get(c->x[0])); }
HLE_AS(os_TryLockMutex, "_ZN2nn2os12TryLockMutexEPNS0_9MutexTypeE") {
    HMutex* m = mtx_get(c->x[0]);
    if (m->m.try_lock()) { m->owner = GetCurrentThreadId(); m->depth++; c->x[0] = 1; } else c->x[0] = 0;
}
HLE_AS(os_UnlockMutex, "_ZN2nn2os11UnlockMutexEPNS0_9MutexTypeE") { mtx_unlock(mtx_get(c->x[0])); }
HLE_AS(os_IsMutexLockedByCurrentThread, "_ZN2nn2os28IsMutexLockedByCurrentThreadEPKNS0_9MutexTypeE") {
    c->x[0] = mtx_get(c->x[0])->owner == GetCurrentThreadId();
}
struct HCond { std::condition_variable_any cv; };
static Registry<HCond> g_cond;
HLE_AS(os_InitializeConditionVariable, "_ZN2nn2os27InitializeConditionVariableEPNS0_21ConditionVariableTypeE") { g_cond.reset(c->x[0]); }
HLE_AS(os_FinalizeConditionVariable, "_ZN2nn2os25FinalizeConditionVariableEPNS0_21ConditionVariableTypeE") { }
HLE_AS(os_WaitConditionVariable, "_ZN2nn2os21WaitConditionVariableEPNS0_21ConditionVariableTypeEPNS0_9MutexTypeE") {
    HCond* cv = g_cond.get(c->x[0]); HMutex* m = mtx_get(c->x[1]);
    int d = m->depth; m->depth = 0; m->owner = 0;            // the SDK requires depth 1 here
    std::unique_lock<std::recursive_mutex> l(m->m, std::adopt_lock);
    evlog('V', c->x[0], c->x[30]);
    cv->cv.wait(l);
    evlog('v', c->x[0], c->x[30]);
    l.release(); m->owner = GetCurrentThreadId(); m->depth = d;
}
HLE_AS(os_SignalConditionVariable, "_ZN2nn2os23SignalConditionVariableEPNS0_21ConditionVariableTypeE") { evlog('U', c->x[0], c->x[30]); g_cond.get(c->x[0])->cv.notify_one(); }

// ---------------------------------------------------------------- reader/writer lock (write owner may also read/recurse)
struct HRw { std::mutex m; std::condition_variable cv; int readers = 0; DWORD writer = 0; int wdepth = 0; };
static Registry<HRw> g_rw;
HLE_AS(os_InitializeReaderWriterLock, "_ZN2nn2os26InitializeReaderWriterLockEPNS0_20ReaderWriterLockTypeE") { g_rw.reset(c->x[0]); }
HLE_AS(os_FinalizeReaderWriterLock, "_ZN2nn2os24FinalizeReaderWriterLockEPNS0_20ReaderWriterLockTypeE") { }
static bool rw_read(HRw* r, bool block) {
    std::unique_lock<std::mutex> l(r->m); DWORD me = GetCurrentThreadId();
    if (r->writer == me) { r->readers++; return true; }
    if (!block && r->writer) return false;
    r->cv.wait(l, [&] { return r->writer == 0; }); r->readers++; return true;
}
static bool rw_write(HRw* r, bool block) {
    std::unique_lock<std::mutex> l(r->m); DWORD me = GetCurrentThreadId();
    if (r->writer == me) { r->wdepth++; return true; }
    if (!block && (r->writer || r->readers)) return false;
    r->cv.wait(l, [&] { return r->writer == 0 && r->readers == 0; }); r->writer = me; r->wdepth = 1; return true;
}
HLE_AS(os_AcquireReadLock, "_ZN2nn2os15AcquireReadLockEPNS0_20ReaderWriterLockTypeE") { rw_read(g_rw.get(c->x[0]), true); }
HLE_AS(os_TryAcquireReadLock, "_ZN2nn2os18TryAcquireReadLockEPNS0_20ReaderWriterLockTypeE") { c->x[0] = rw_read(g_rw.get(c->x[0]), false); }
HLE_AS(os_ReleaseReadLock, "_ZN2nn2os15ReleaseReadLockEPNS0_20ReaderWriterLockTypeE") {
    HRw* r = g_rw.get(c->x[0]); { std::lock_guard<std::mutex> l(r->m); r->readers--; } r->cv.notify_all();
}
HLE_AS(os_AcquireWriteLock, "_ZN2nn2os16AcquireWriteLockEPNS0_20ReaderWriterLockTypeE") { rw_write(g_rw.get(c->x[0]), true); }
HLE_AS(os_TryAcquireWriteLock, "_ZN2nn2os19TryAcquireWriteLockEPNS0_20ReaderWriterLockTypeE") { c->x[0] = rw_write(g_rw.get(c->x[0]), false); }
HLE_AS(os_ReleaseWriteLock, "_ZN2nn2os16ReleaseWriteLockEPNS0_20ReaderWriterLockTypeE") {
    HRw* r = g_rw.get(c->x[0]); { std::lock_guard<std::mutex> l(r->m); if (--r->wdepth == 0) r->writer = 0; } r->cv.notify_all();
}
HLE_AS(os_IsWriteLockHeldByCurrentThread, "_ZN2nn2os30IsWriteLockHeldByCurrentThreadEPKNS0_20ReaderWriterLockTypeE") {
    c->x[0] = g_rw.get(c->x[0])->writer == GetCurrentThreadId();
}

// ---------------------------------------------------------------- semaphores
struct HSema { std::mutex m; std::condition_variable cv; int count = 0, max = 0x7FFFFFFF; };
static Registry<HSema> g_sema;
static void sema_init(u64 a, int cnt, int max) { HSema* s = g_sema.reset(a); s->count = cnt; s->max = max; }
static void sema_acquire(HSema* s) { std::unique_lock<std::mutex> l(s->m); s->cv.wait(l, [&] { return s->count > 0; }); s->count--; }
static void sema_release(HSema* s, int n) { { std::lock_guard<std::mutex> l(s->m); s->count += n; if (s->count > s->max) s->count = s->max; } s->cv.notify_all(); }
HLE_AS(os_InitializeSemaphore, "_ZN2nn2os19InitializeSemaphoreEPNS0_13SemaphoreTypeEii") { sema_init(c->x[0], (int)c->x[1], (int)c->x[2]); }
HLE_AS(os_FinalizeSemaphore, "_ZN2nn2os17FinalizeSemaphoreEPNS0_13SemaphoreTypeE") { }
HLE_AS(os_AcquireSemaphore, "_ZN2nn2os16AcquireSemaphoreEPNS0_13SemaphoreTypeE") { sema_acquire(g_sema.get(c->x[0])); }
HLE_AS(os_ReleaseSemaphore, "_ZN2nn2os16ReleaseSemaphoreEPNS0_13SemaphoreTypeEi") { sema_release(g_sema.get(c->x[0]), (int)c->x[1]); }
HLE_AS(os_InitializeLightSemaphore, "_ZN2nn2os24InitializeLightSemaphoreEPNS0_18LightSemaphoreTypeEii") { sema_init(c->x[0], (int)c->x[1], (int)c->x[2]); }
HLE_AS(os_FinalizeLightSemaphore, "_ZN2nn2os22FinalizeLightSemaphoreEPNS0_18LightSemaphoreTypeE") { }
HLE_AS(os_AcquireLightSemaphore, "_ZN2nn2os21AcquireLightSemaphoreEPNS0_18LightSemaphoreTypeE") { sema_acquire(g_sema.get(c->x[0])); }
HLE_AS(os_ReleaseLightSemaphore, "_ZN2nn2os21ReleaseLightSemaphoreEPNS0_18LightSemaphoreTypeE") { sema_release(g_sema.get(c->x[0]), 1); }

// ---------------------------------------------------------------- events (Event, LightEvent, SystemEvent, TimerEvent)
struct HEvent {
    std::mutex m; std::condition_variable cv; bool sig = false, autoclear = false;
    s64 timer_next = 0, timer_period = 0;                     // TimerEvent: ns since steady epoch
    s64 sig_t = 0;                                            // (handoff latency stats: last signal)
};
static Registry<HEvent> g_event;
static void ev_init(u64 a, bool sig, int clear_mode) { HEvent* e = g_event.reset(a); e->sig = sig; e->autoclear = clear_mode == 1; }
static void ev_signal(HEvent* e) { { std::lock_guard<std::mutex> l(e->m); e->sig = true; e->sig_t = now_ns(); } e->cv.notify_all(); }
static void ev_clear(HEvent* e) { std::lock_guard<std::mutex> l(e->m); e->sig = false; }
static bool ev_wait(HEvent* e, s64 timeout_ns) {               // timeout < 0: infinite
    std::unique_lock<std::mutex> l(e->m);
    auto pred = [&] { return e->sig; };
    if (timeout_ns < 0) { bool blk = !e->sig; e->cv.wait(l, pred); hw_note(0, blk, blk ? now_ns() - e->sig_t : 0); }
    else if (timeout_ns > 0 && !e->sig) {   // (timed: measured in "[sleep] ... timed waits")
        s64 t0 = now_ns();
        bool ok = e->cv.wait_for(l, std::chrono::nanoseconds(timeout_ns), pred);
        g_tw_n++; g_tw_req += (u64)timeout_ns; g_tw_act += (u64)(now_ns() - t0); if (!ok) g_tw_out++;
        if (!ok) return false;
    } else if (!e->sig) return false;
    if (e->autoclear) e->sig = false;
    return true;
}
static bool ev_try(HEvent* e) { std::lock_guard<std::mutex> l(e->m); bool s = e->sig; if (s && e->autoclear) e->sig = false; return s; }
HLE_AS(os_InitializeEvent, "_ZN2nn2os15InitializeEventEPNS0_9EventTypeEbNS0_14EventClearModeE") { ev_init(c->x[0], c->x[1] & 1, (int)c->x[2]); }
HLE_AS(os_FinalizeEvent, "_ZN2nn2os13FinalizeEventEPNS0_9EventTypeE") { }
HLE_AS(os_SignalEvent, "_ZN2nn2os11SignalEventEPNS0_9EventTypeE") { ev_signal(g_event.get(c->x[0])); }
HLE_AS(os_ClearEvent, "_ZN2nn2os10ClearEventEPNS0_9EventTypeE") { ev_clear(g_event.get(c->x[0])); }
// Light-event history ring (job-system hang, then the wait-chain trace of 2026-10-16): op I(nit) S(ignal) C(lear)
// W(ait enter) w(ait done); T/t timed light-event wait; V/v condition-variable wait, U its signal; R/r light message
// queue receive, Q send / jam; Y/y nvnSyncWait (objects.c). Dumped by the watchdog; TOTK_EVRING_DUMP=<s> writes the
// whole ring (1M ops) to evring.txt at that time for scratchpad analysis (who the main thread waits on per frame).
struct EvOp { s64 t; u64 ev, lr; u32 tid; char op; };
enum : u64 { EVRING = 1u << 20 };
static EvOp g_evring[EVRING];
static std::atomic<u64> g_evpos{0};
static void evlog(char op, u64 ev, u64 lr) { u64 i = g_evpos++ & (EVRING - 1); g_evring[i] = { now_ns(), ev, lr, (u32)GetCurrentThreadId(), op }; }
extern "C" void hle_evlog(char op, u64 ev, u64 lr) { evlog(op, ev, lr); }
static void evring_save(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;
    u64 end = g_evpos.load(), beg = end > EVRING ? end - EVRING : 0;
    for (u64 i = beg; i < end; i++) {
        const EvOp& o = g_evring[i & (EVRING - 1)];
        fprintf(f, "%lld %c %llx %llx %u\n", (long long)o.t, o.op ? o.op : '?', (unsigned long long)o.ev, (unsigned long long)o.lr, o.tid);
    }
    fclose(f);
    printf("[evring] %llu ops written to %s\n", (unsigned long long)(end - beg), path);
    fflush(stdout);
}
static int s_evring_dump_init = [] {
    if (const char* e = getenv("TOTK_EVRING_DUMP")) {
        int sec = atoi(e);
        std::thread([sec] { std::this_thread::sleep_for(std::chrono::seconds(sec)); evring_save("evring.txt"); }).detach();
    }
    return 0;
}();
void hle_evring_dump(u64 focus) {
    u64 end = g_evpos.load(), beg = end > EVRING ? end - EVRING : 0; int shown = 0;
    printf("[evring] %llu ops; last 24 on %llx:\n", end, focus);
    std::vector<u64> hits;
    for (u64 i = end; i-- > beg && hits.size() < 24;) if (g_evring[i & (EVRING - 1)].ev == focus) hits.push_back(i);
    for (auto it = hits.rbegin(); it != hits.rend(); ++it) {
        const EvOp& o = g_evring[*it & (EVRING - 1)];
        printf("[evring]   #%llu %c t=%.3f s tid %u lr %llx\n", *it, o.op, o.t / 1e9, o.tid, o.lr);
    }
    printf("[evring] last 16 ops overall:\n");
    for (u64 i = end > 16 ? end - 16 : 0; i < end; i++, shown++) {
        const EvOp& o = g_evring[i & (EVRING - 1)];
        printf("[evring]   #%llu %c ev %llx t=%.3f s tid %u lr %llx\n", i, o.op, o.ev, o.t / 1e9, o.tid, o.lr);
    }
}
HLE_AS(os_InitializeLightEvent, "_ZN2nn2os20InitializeLightEventEPNS0_14LightEventTypeEbNS0_14EventClearModeE") { evlog('I', c->x[0], c->x[30]); ev_init(c->x[0], c->x[1] & 1, (int)c->x[2]); }
HLE_AS(os_FinalizeLightEvent, "_ZN2nn2os18FinalizeLightEventEPNS0_14LightEventTypeE") { }
HLE_AS(os_SignalLightEvent, "_ZN2nn2os16SignalLightEventEPNS0_14LightEventTypeE") { evlog('S', c->x[0], c->x[30]); ev_signal(g_event.get(c->x[0])); }
HLE_AS(os_ClearLightEvent, "_ZN2nn2os15ClearLightEventEPNS0_14LightEventTypeE") { evlog('C', c->x[0], c->x[30]); ev_clear(g_event.get(c->x[0])); }
HLE_AS(os_WaitLightEvent, "_ZN2nn2os14WaitLightEventEPNS0_14LightEventTypeE") {
    evlog('W', c->x[0], c->x[30]); ev_wait(g_event.get(c->x[0]), -1); evlog('w', c->x[0], c->x[30]);
}
HLE_AS(os_TimedWaitLightEvent, "_ZN2nn2os19TimedWaitLightEventEPNS0_14LightEventTypeENS_8TimeSpanE") {
    u64 ev = c->x[0], lr = c->x[30];
    evlog('T', ev, lr);
    s64 ns = timespan_ns(c->x[1]); c->x[0] = ev_wait(g_event.get(ev), ns < 0 ? 0 : ns);
    evlog('t', ev, lr);
}
// For HLE modules that own a system event (vi vsync, audio renderer): set it up and signal it from host code.
void hle_sysevent_init(u64 ev, bool autoclear) { ev_init(ev, false, autoclear ? 1 : 0); }
void hle_sysevent_signal(u64 ev) { evlog('S', ev, 0); ev_signal(g_event.get(ev)); }
// System events belong to SDK services we do not run yet: never signalled unless an HLE module signals them.
HLE_AS(os_DestroySystemEvent, "_ZN2nn2os18DestroySystemEventEPNS0_15SystemEventTypeE") { g_event.drop(c->x[0]); }
HLE_AS(os_ClearSystemEvent, "_ZN2nn2os16ClearSystemEventEPNS0_15SystemEventTypeE") { ev_clear(g_event.get(c->x[0])); }
HLE_AS(os_TryWaitSystemEvent, "_ZN2nn2os18TryWaitSystemEventEPNS0_15SystemEventTypeE") { c->x[0] = ev_try(g_event.get(c->x[0])); }
HLE_AS(os_WaitSystemEvent, "_ZN2nn2os15WaitSystemEventEPNS0_15SystemEventTypeE") {
    static std::mutex lm; static std::unordered_map<u64, int> logged;   // first few per event (vsync waits: 60/s)
    { std::lock_guard<std::mutex> l(lm); if (logged[c->x[0]]++ < 8) { printf("[os] WaitSystemEvent(%llx) lr=%llx (blocks until signalled)\n", c->x[0], c->x[30]); fflush(stdout); } }
    evlog('W', c->x[0], c->x[30]);
    ev_wait(g_event.get(c->x[0]), -1);
    evlog('w', c->x[0], c->x[30]);
}
HLE_AS(os_InitializeTimerEvent, "_ZN2nn2os20InitializeTimerEventEPNS0_14TimerEventTypeENS0_14EventClearModeE") { ev_init(c->x[0], false, (int)c->x[1]); }
HLE_AS(os_FinalizeTimerEvent, "_ZN2nn2os18FinalizeTimerEventEPNS0_14TimerEventTypeE") { }
HLE_AS(os_StartPeriodicTimerEvent, "_ZN2nn2os23StartPeriodicTimerEventEPNS0_14TimerEventTypeENS_8TimeSpanES3_") {
    HEvent* e = g_event.get(c->x[0]); std::lock_guard<std::mutex> l(e->m);
    e->timer_next = now_ns() + timespan_ns(c->x[1]); e->timer_period = timespan_ns(c->x[2]);
}
HLE_AS(os_WaitTimerEvent, "_ZN2nn2os14WaitTimerEventEPNS0_14TimerEventTypeE") {
    HEvent* e = g_event.get(c->x[0]);
    s64 due;
    { std::lock_guard<std::mutex> l(e->m); if (e->sig) { if (e->autoclear) e->sig = false; return; } due = e->timer_next; }
    if (!due) { ev_wait(e, -1); return; }
    s64 d = due - now_ns();
    if (d > 0) guest_sleep_ns(d);
    std::lock_guard<std::mutex> l(e->m);
    if (e->timer_period > 0) { s64 n = now_ns(); while (e->timer_next <= n) e->timer_next += e->timer_period; }
}

// ---------------------------------------------------------------- message queues (guest buffer unused)
struct HQueue { std::mutex m; std::condition_variable cv; std::deque<u64> q; size_t cap = 1; s64 send_t = 0; };
static Registry<HQueue> g_queue;
static void q_init(u64 a, u64 cap) { HQueue* q = g_queue.reset(a); q->cap = cap ? cap : 1; }
static bool q_send(HQueue* q, u64 v, bool block, bool front) {
    std::unique_lock<std::mutex> l(q->m);
    if (q->q.size() >= q->cap) { if (!block) return false; q->cv.wait(l, [&] { return q->q.size() < q->cap; }); }
    if (front) q->q.push_front(v); else q->q.push_back(v);
    q->send_t = now_ns();
    q->cv.notify_all(); return true;
}
static bool q_recv(HQueue* q, u64* out, bool block, bool peek, s64 timeout_ns = -1) {
    std::unique_lock<std::mutex> l(q->m);
    auto pred = [&] { return !q->q.empty(); };
    if (q->q.empty()) {
        if (!block) return false;
        if (timeout_ns < 0) { q->cv.wait(l, pred); hw_note(1, true, now_ns() - q->send_t); }
        else if (!q->cv.wait_for(l, std::chrono::nanoseconds(timeout_ns), pred)) return false;
    } else if (block) hw_note(1, false, 0);
    *out = q->q.front();
    if (!peek) { q->q.pop_front(); q->cv.notify_all(); }
    return true;
}
HLE_AS(os_InitializeMessageQueue, "_ZN2nn2os22InitializeMessageQueueEPNS0_16MessageQueueTypeEPmm") { q_init(c->x[0], c->x[2]); }
HLE_AS(os_FinalizeMessageQueue, "_ZN2nn2os20FinalizeMessageQueueEPNS0_16MessageQueueTypeE") { }
HLE_AS(os_SendMessageQueue, "_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm") { q_send(g_queue.get(c->x[0]), c->x[1], true, false); }
HLE_AS(os_ReceiveMessageQueue, "_ZN2nn2os19ReceiveMessageQueueEPmPNS0_16MessageQueueTypeE") { q_recv(g_queue.get(c->x[1]), (u64*)c->x[0], true, false); }
HLE_AS(os_InitializeLightMessageQueue, "_ZN2nn2os27InitializeLightMessageQueueEPNS0_21LightMessageQueueTypeEPmm") { q_init(c->x[0], c->x[2]); }
HLE_AS(os_FinalizeLightMessageQueue, "_ZN2nn2os25FinalizeLightMessageQueueEPNS0_21LightMessageQueueTypeE") { }
HLE_AS(os_SendLightMessageQueue, "_ZN2nn2os21SendLightMessageQueueEPNS0_21LightMessageQueueTypeEm") { evlog('Q', c->x[0], c->x[30]); q_send(g_queue.get(c->x[0]), c->x[1], true, false); }
HLE_AS(os_TrySendLightMessageQueue, "_ZN2nn2os24TrySendLightMessageQueueEPNS0_21LightMessageQueueTypeEm") { evlog('Q', c->x[0], c->x[30]); c->x[0] = q_send(g_queue.get(c->x[0]), c->x[1], false, false); }
HLE_AS(os_JamLightMessageQueue, "_ZN2nn2os20JamLightMessageQueueEPNS0_21LightMessageQueueTypeEm") { evlog('Q', c->x[0], c->x[30]); q_send(g_queue.get(c->x[0]), c->x[1], true, true); }
HLE_AS(os_TryJamLightMessageQueue, "_ZN2nn2os23TryJamLightMessageQueueEPNS0_21LightMessageQueueTypeEm") { c->x[0] = q_send(g_queue.get(c->x[0]), c->x[1], false, true); }
HLE_AS(os_ReceiveLightMessageQueue, "_ZN2nn2os24ReceiveLightMessageQueueEPmPNS0_21LightMessageQueueTypeE") {
    u64 q = c->x[1], lr = c->x[30];
    evlog('R', q, lr);
    q_recv(g_queue.get(q), (u64*)c->x[0], true, false);
    evlog('r', q, lr);
}
HLE_AS(os_TryReceiveLightMessageQueue, "_ZN2nn2os27TryReceiveLightMessageQueueEPmPNS0_21LightMessageQueueTypeE") { c->x[0] = q_recv(g_queue.get(c->x[1]), (u64*)c->x[0], false, false); }
HLE_AS(os_TryPeekLightMessageQueue, "_ZN2nn2os24TryPeekLightMessageQueueEPmPKNS0_21LightMessageQueueTypeE") { c->x[0] = q_recv(g_queue.get(c->x[1]), (u64*)c->x[0], false, true); }
HLE_AS(os_TimedPeekLightMessageQueue, "_ZN2nn2os26TimedPeekLightMessageQueueEPmPKNS0_21LightMessageQueueTypeENS_8TimeSpanE") {
    s64 ns = timespan_ns(c->x[2]); c->x[0] = q_recv(g_queue.get(c->x[1]), (u64*)c->x[0], true, true, ns < 0 ? 0 : ns);
}

// ---------------------------------------------------------------- multi-wait (polling; holders bound to Events)
struct HHolder { u64 event = 0; u64 multi = 0; };
struct HMulti { std::mutex m; std::vector<u64> holders; };
static Registry<HHolder> g_holder;
static Registry<HMulti> g_multi;
HLE_AS(os_InitializeMultiWait, "_ZN2nn2os19InitializeMultiWaitEPNS0_13MultiWaitTypeE") { g_multi.reset(c->x[0]); }
HLE_AS(os_FinalizeMultiWait, "_ZN2nn2os17FinalizeMultiWaitEPNS0_13MultiWaitTypeE") { }
HLE_AS(os_InitializeMultiWaitHolder, "_ZN2nn2os25InitializeMultiWaitHolderEPNS0_19MultiWaitHolderTypeEPNS0_9EventTypeE") { g_holder.reset(c->x[0])->event = c->x[1]; }
HLE_AS(os_FinalizeMultiWaitHolder, "_ZN2nn2os23FinalizeMultiWaitHolderEPNS0_19MultiWaitHolderTypeE") { }
HLE_AS(os_LinkMultiWaitHolder, "_ZN2nn2os19LinkMultiWaitHolderEPNS0_13MultiWaitTypeEPNS0_19MultiWaitHolderTypeE") {
    HMulti* m = g_multi.get(c->x[0]); std::lock_guard<std::mutex> l(m->m); m->holders.push_back(c->x[1]); g_holder.get(c->x[1])->multi = c->x[0];
}
HLE_AS(os_UnlinkMultiWaitHolder, "_ZN2nn2os21UnlinkMultiWaitHolderEPNS0_19MultiWaitHolderTypeE") {
    HHolder* h = g_holder.get(c->x[0]); if (!h->multi) return;
    HMulti* m = g_multi.get(h->multi); std::lock_guard<std::mutex> l(m->m);
    std::erase(m->holders, c->x[0]); h->multi = 0;
}
HLE_AS(os_WaitAny, "_ZN2nn2os7WaitAnyEPNS0_13MultiWaitTypeE") {
    HMulti* m = g_multi.get(c->x[0]);
    for (;;) {
        std::vector<u64> hs; { std::lock_guard<std::mutex> l(m->m); hs = m->holders; }
        for (u64 h : hs) {
            HEvent* e = g_event.get(g_holder.get(h)->event);
            std::lock_guard<std::mutex> l(e->m);
            if (e->sig) { c->x[0] = h; return; }               // WaitAny does not clear (SDK semantics)
        }
        Sleep(1);
    }
}

// ---------------------------------------------------------------- address space
// AllocateAddressRegion / FreeAddressRegion live in hle_os_mem.cpp (alias region allocator).
HLE_AS(os_SetDyingMessageRegion, "_ZN2nn2os21SetDyingMessageRegionEmm") { c->x[0] = 0; }

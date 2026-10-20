// TotK Linux port: POSIX implementation of the Win32 subset in runtime/linux/windows.h, the fatal-signal plumbing and the
// write-watch tracker. Linux build only (mkninja_linux.py); not part of the Windows build (nothing globs runtime/linux).
//
// WRITE WATCH (GetWriteWatch / ResetWriteWatch on a MEM_WRITE_WATCH region; used by hle_os_mem.cpp for the guest heap (textures.c asks
// whether a texture's pages were written) and the alias import window (exec.c keeps a VRAM mirror and copies the dirty ranges)).
// Windows tracks writes in the page tables for free; Linux has no equivalent that can be reset per range:
//   - /proc/self/pagemap soft-dirty can only be cleared for the whole process (clear_refs write-protects every page of 16 GB), and
//     kernels built without CONFIG_MEM_SOFT_DIRTY have nothing;
//   - userfaultfd write-protect mode fails kernel-mode writes (read(2) into a guest buffer) and needs privileges / newer kernels.
// So: mprotect(PROT_READ) + a SIGSEGV handler. Per 4 KiB page of a tracked region: st = UNARMED (never watched, writable, reported
// as written the first time it is queried - conservative) / ARMED (read-only, clean) / PENDING (fault being handled) / DIRTY
// (written since it was armed, writable again). The first write to an ARMED page faults once (~3-6 us: signal frame, mprotect,
// sigreturn); the handler marks it and unprotects it, so later writes cost nothing until the next query re-arms it.
// A query reports every page that is not ARMED and, with RESET, re-arms the reported ones (state first, then mprotect: a write that
// slips in between is visible to the caller's copy, which runs after the query returns; a write after the mprotect faults again).
// Lock-free handler, queries are serialised by one mutex. Cost model: ~5 us per first-written page per query cycle, plus one
// mprotect per run of re-armed pages; a scan is a byte loop over `st` (2 GiB window = 512 KiB of states). Pages that were never
// queried are never protected, so ordinary heap writes outside textures / alias windows do not fault.
// Kernel writes (read(2) into a guest buffer) get EFAULT on an ARMED page: hle_fs.cpp calls rt_ww_prepare_kernel_write first.
// If mprotect fails (vm.max_map_count exhausted by fragmentation) the tracker switches itself off for good: every region is made
// writable, GetWriteWatch fails (callers copy / hash everything, as with TOTK_HEAP_WW=0 / TOTK_ALIAS_NO_WATCH).
#ifndef _WIN32
#include "windows.h"
#include "plat_posix.h"
#include <dirent.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <semaphore.h>
#include <signal.h>
#include <atomic>
#include <sched.h>
#include <condition_variable>
#include <cstdio>
#include <algorithm>
#include <ctime>
#include <map>
#include <vector>
#include <mutex>
#include <string>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C" char __ehdr_start;

namespace {

constexpr size_t kPg = 4096, kGran = 65536;
size_t round_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// ------------------------------------------------------------------ virtual memory
enum : uint8_t { W_UNARMED = 0, W_ARMED = 1, W_DIRTY = 2, W_PENDING = 3 };
struct Resv {
    uintptr_t base; size_t size;
    bool ww;
    uint8_t *st, *com;      // ww only: per page state / committed
};
std::mutex g_vm_mu;
std::map<uintptr_t, Resv*> g_resv;
Resv* g_wwtab[64];
std::atomic<int> g_wwn{0};
std::mutex g_ww_query_mu;
std::atomic<bool> g_ww_off{false};
std::atomic<unsigned long long> g_ww_faults_n{0}, g_ww_arm_fail{0};

Resv* resv_of(uintptr_t a) {   // g_vm_mu held
    auto it = g_resv.upper_bound(a);
    if (it == g_resv.begin()) return nullptr;
    --it;
    return a - it->first < it->second->size ? it->second : nullptr;
}
Resv* ww_find(uintptr_t a) {   // lock-free (signal handler)
    int n = g_wwn.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) { Resv* r = g_wwtab[i]; if (r && a - r->base < r->size) return r; }
    return nullptr;
}

// Tracking failed (mprotect ENOMEM): make every tracked region writable and stop answering.
void ww_panic() {
    if (g_ww_off.exchange(true)) return;
    int n = g_wwn.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) {
        Resv* r = g_wwtab[i];
        if (!r || !r->size) continue;
        size_t np = r->size / kPg;
        for (size_t p = 0; p < np;) {
            if (!r->com[p]) { p++; continue; }
            size_t q = p; while (q < np && r->com[q]) q++;
            mprotect((void*)(r->base + p * kPg), (q - p) * kPg, PROT_READ | PROT_WRITE);
            p = q;
        }
    }
    static const char msg[] = "[os] write watch: mprotect failed (vm.max_map_count?), tracking OFF: full copies from here on\n";
    (void)!write(2, msg, sizeof msg - 1);
}

bool ww_fault(uintptr_t addr) {   // signal handler: true = handled, retry the faulting instruction
    Resv* r = ww_find(addr);
    if (!r) return false;
    size_t i = (addr - r->base) / kPg;
    if (!r->com[i]) return false;
    uint8_t exp = W_ARMED;
    if (__atomic_compare_exchange_n(&r->st[i], &exp, W_PENDING, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        if (mprotect((void*)(r->base + i * kPg), kPg, PROT_READ | PROT_WRITE) != 0) { g_ww_arm_fail++; ww_panic(); return true; }
        __atomic_store_n(&r->st[i], W_DIRTY, __ATOMIC_RELEASE);
        g_ww_faults_n++;
        return true;
    }
    return g_ww_off.load() || exp == W_PENDING || exp == W_DIRTY;   // a concurrent handler / query: retry
}

}  // namespace

extern "C" {

LPVOID VirtualAlloc(LPVOID addr, SIZE_T size, DWORD type, DWORD prot) {
    (void)prot;
    if (!size) { errno = EINVAL; return nullptr; }
    uintptr_t a = (uintptr_t)addr;
    bool fresh = (type & MEM_RESERVE) || !addr;
    if (fresh) {
        size_t sz = round_up(size, kGran);
        int fl = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
        void* p;
        if (addr) {
            p = mmap(addr, sz, PROT_NONE, fl | MAP_FIXED_NOREPLACE, -1, 0);
            if (p == MAP_FAILED) return nullptr;
            if (p != addr) { munmap(p, sz); errno = EEXIST; return nullptr; }   // (kernels without MAP_FIXED_NOREPLACE treat it as a hint)
        } else {
            p = mmap(nullptr, sz + kGran, PROT_NONE, fl, -1, 0);                 // 64 KiB aligned like a Windows reservation
            if (p == MAP_FAILED) return nullptr;
            uintptr_t al = round_up((uintptr_t)p, kGran);
            if (al > (uintptr_t)p) munmap(p, al - (uintptr_t)p);
            uintptr_t tail = al + sz, end = (uintptr_t)p + sz + kGran;
            if (end > tail) munmap((void*)tail, end - tail);
            p = (void*)al;
        }
        Resv* r = new Resv{(uintptr_t)p, sz, (type & MEM_WRITE_WATCH) != 0, nullptr, nullptr};
        if (r->ww) {
            r->st = (uint8_t*)calloc(sz / kPg, 1);
            r->com = (uint8_t*)calloc(sz / kPg, 1);
            if (!r->st || !r->com) { munmap(p, sz); delete r; errno = ENOMEM; return nullptr; }
        }
        if (type & MEM_COMMIT) {
            if (mprotect(p, sz, PROT_READ | PROT_WRITE) != 0) { munmap(p, sz); delete r; return nullptr; }
            if (r->ww) memset(r->com, 1, sz / kPg);
        }
        {
            std::lock_guard<std::mutex> l(g_vm_mu);
            g_resv[r->base] = r;
            if (r->ww) {
                int n = g_wwn.load();
                if (n < 64) { g_wwtab[n] = r; g_wwn.store(n + 1, std::memory_order_release); }
            }
        }
        return p;
    }
    // commit inside an existing reservation
    std::lock_guard<std::mutex> l(g_vm_mu);
    Resv* r = resv_of(a);
    uintptr_t lo = a & ~(kPg - 1), hi = round_up(a + size, kPg);
    if (!r || hi > r->base + r->size) { errno = EINVAL; return nullptr; }
    if (!r->ww) {
        if (mprotect((void*)lo, hi - lo, PROT_READ | PROT_WRITE) != 0) return nullptr;
        return addr;
    }
    size_t p0 = (lo - r->base) / kPg, p1 = (hi - r->base) / kPg;
    for (size_t p = p0; p < p1;) {   // only pages that are not committed yet: an ARMED (read-only) page must stay armed
        if (r->com[p]) { p++; continue; }
        size_t q = p; while (q < p1 && !r->com[q]) q++;
        if (mprotect((void*)(r->base + p * kPg), (q - p) * kPg, PROT_READ | PROT_WRITE) != 0) return nullptr;
        for (size_t k = p; k < q; k++) { r->st[k] = W_UNARMED; r->com[k] = 1; }
        p = q;
    }
    return addr;
}

BOOL VirtualFree(LPVOID addr, SIZE_T size, DWORD type) {
    std::lock_guard<std::mutex> l(g_vm_mu);
    uintptr_t a = (uintptr_t)addr;
    Resv* r = resv_of(a);
    if (!r) { errno = EINVAL; return FALSE; }
    if (type & MEM_RELEASE) {
        if (a != r->base) { errno = EINVAL; return FALSE; }
        munmap((void*)r->base, r->size);
        g_resv.erase(r->base);
        if (r->ww) r->size = 0;       // (the Resv and its state arrays leak: the signal handler may still hold the pointer)
        else delete r;
        return TRUE;
    }
    uintptr_t lo = a & ~(kPg - 1), hi = round_up(a + size, kPg);
    if (hi > r->base + r->size) { errno = EINVAL; return FALSE; }
    madvise((void*)lo, hi - lo, MADV_DONTNEED);
    mprotect((void*)lo, hi - lo, PROT_NONE);
    if (r->ww)
        for (size_t p = (lo - r->base) / kPg; p < (hi - r->base) / kPg; p++) { r->com[p] = 0; r->st[p] = W_UNARMED; }
    return TRUE;
}

UINT GetWriteWatch(DWORD flags, PVOID base, SIZE_T size, PVOID* addrs, ULONG_PTR* count, DWORD* gran) {
    if (g_ww_off.load()) return 87;
    uintptr_t b = (uintptr_t)base;
    Resv* r = ww_find(b);
    if (!r || !size || b + size > r->base + r->size) return 87;   // ERROR_INVALID_PARAMETER
    size_t p0 = (b - r->base) / kPg, p1 = (b + size - 1 - r->base) / kPg + 1, cap = (size_t)*count, n = 0;
    bool reset = (flags & WRITE_WATCH_FLAG_RESET) != 0;
    std::lock_guard<std::mutex> ql(g_ww_query_mu);
    size_t run_lo = 0, run_hi = 0;
    auto flush = [&] {
        if (run_hi > run_lo && mprotect((void*)(r->base + run_lo * kPg), (run_hi - run_lo) * kPg, PROT_READ) != 0) { g_ww_arm_fail++; ww_panic(); }
        run_lo = run_hi = 0;
    };
    for (size_t i = p0; i < p1; i++) {
        if (!r->com[i]) continue;
        uint8_t s = __atomic_load_n(&r->st[i], __ATOMIC_ACQUIRE);
        if (s == W_ARMED) continue;
        if (n == cap) break;
        addrs[n++] = (PVOID)(r->base + i * kPg);
        if (reset && s != W_PENDING) {
            __atomic_store_n(&r->st[i], W_ARMED, __ATOMIC_RELEASE);   // state first, then the page goes read-only
            if (run_hi == i && run_hi > run_lo) run_hi = i + 1; else { flush(); run_lo = i; run_hi = i + 1; }
        }
    }
    flush();
    if (g_ww_off.load()) return 87;
    *count = n; *gran = (DWORD)kPg;
    return 0;
}

UINT ResetWriteWatch(PVOID base, SIZE_T size) {
    if (g_ww_off.load()) return 87;
    uintptr_t b = (uintptr_t)base;
    Resv* r = ww_find(b);
    if (!r || !size || b + size > r->base + r->size) return 87;
    size_t p0 = (b - r->base) / kPg, p1 = (b + size - 1 - r->base) / kPg + 1;
    std::lock_guard<std::mutex> ql(g_ww_query_mu);
    for (size_t p = p0; p < p1;) {
        if (!r->com[p]) { p++; continue; }
        size_t q = p;
        while (q < p1 && r->com[q]) { __atomic_store_n(&r->st[q], W_ARMED, __ATOMIC_RELEASE); q++; }
        if (mprotect((void*)(r->base + p * kPg), (q - p) * kPg, PROT_READ) != 0) { g_ww_arm_fail++; ww_panic(); return 87; }
        p = q;
    }
    return 0;
}

void rt_ww_prepare_kernel_write(const void* ptr, size_t len) {
    uintptr_t b = (uintptr_t)ptr;
    Resv* r = len ? ww_find(b) : nullptr;
    if (!r || g_ww_off.load()) return;
    uintptr_t end = b + len;
    if (end > r->base + r->size) end = r->base + r->size;
    size_t p0 = (b - r->base) / kPg, p1 = (end - 1 - r->base) / kPg + 1;
    for (size_t i = p0; i < p1; i++) {
        if (!r->com[i]) continue;
        uint8_t exp = W_ARMED;
        if (!__atomic_compare_exchange_n(&r->st[i], &exp, W_PENDING, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        if (mprotect((void*)(r->base + i * kPg), kPg, PROT_READ | PROT_WRITE) != 0) { g_ww_arm_fail++; ww_panic(); return; }
        __atomic_store_n(&r->st[i], W_DIRTY, __ATOMIC_RELEASE);
    }
}

unsigned long long rt_ww_faults(void) { return g_ww_faults_n.load(); }

int rt_posix_readable(unsigned long long a, unsigned long long n) {
    static std::mutex mu;
    static int pfd[2] = {-1, -1};
    std::lock_guard<std::mutex> l(mu);
    if (pfd[0] < 0 && pipe2(pfd, O_NONBLOCK | O_CLOEXEC) != 0) return 0;
    char sink[1024];
    while (n) {
        size_t c = n < sizeof sink ? (size_t)n : sizeof sink;
        ssize_t w = write(pfd[1], (const void*)a, c);     // EFAULT on an unreadable page, no signal
        if (w != (ssize_t)c) { if (w > 0) (void)!read(pfd[0], sink, (size_t)w); return 0; }
        (void)!read(pfd[0], sink, c);
        a += c; n -= c;
    }
    return 1;
}

// ------------------------------------------------------------------ threads / timers
}  // extern "C"

namespace {
enum { K_THREAD = 1, K_TIMER = 2 };
struct Obj { int kind; };
struct ThreadObj : Obj {
    pthread_t th; std::atomic<pid_t> tid{0};
    LPTHREAD_START_ROUTINE fn; LPVOID arg;
    std::mutex m; std::condition_variable cv; bool done = false;
    std::atomic<int> refs{2};      // the thread + the handle
    sem_t started;
    ThreadObj() { kind = K_THREAD; }
};
struct TimerObj : Obj { int fd; TimerObj() { kind = K_TIMER; } };
void unref(ThreadObj* t) { if (t->refs.fetch_sub(1) == 1) { sem_destroy(&t->started); delete t; } }
thread_local pid_t t_tid;
pid_t my_tid() { if (!t_tid) t_tid = (pid_t)syscall(SYS_gettid); return t_tid; }
const HANDLE kSelfThread = (HANDLE)(intptr_t)-2;

void* thread_main(void* p) {
    ThreadObj* t = (ThreadObj*)p;
    rt_posix_thread_init();
    t->tid = my_tid();
    sem_post(&t->started);
    t->fn(t->arg);
    { std::lock_guard<std::mutex> l(t->m); t->done = true; }
    t->cv.notify_all();
    unref(t);
    return nullptr;
}
__attribute__((constructor)) void plat_init() { prctl(PR_SET_TIMERSLACK, 1, 0, 0, 0); }   // inherited by every thread: sub-50 us timer waits
}  // namespace

extern "C" {

// TOTK_AFFINITY (Deck, 2026-10-17; default off): pin guest threads by their guest core mask and the render-side host
// threads (render, vk-record) to their own CPUs, so the always-busy render threads stop sharing SMT cores with the guest
// critical path (d45: guest code ~1.14x slower from SMT/memory contention with rendering on). "1" = "0-1,2-3,4-5:6-7"
// (guest core k -> Deck physical core k = CPUs 2k, 2k+1; render side -> core 3); or give "<core0>,<core1>,<core2>:<render>"
// CPU lists. kind 0: calling guest thread, arg = guest core mask (bits 0..2); kind 1: calling render-side thread.
void host_affinity(int kind, unsigned long long mask) {
    static int on = -1;
    static cpu_set_t gset[3], rset;
    if (on < 0) {
        const char* e = getenv("TOTK_AFFINITY");
        on = e && *e && strcmp(e, "0") != 0;
        if (on) {
            const char* spec = strcmp(e, "1") == 0 ? "0-1,2-3,4-5:6-7" : e;
            auto parse = [](const char* s, const char* end, cpu_set_t* set) {   // "a-b" or "a" or "a-b+c"
                CPU_ZERO(set);
                while (s < end) {
                    int a = atoi(s), b = a;
                    while (s < end && *s >= '0' && *s <= '9') s++;
                    if (s < end && *s == '-') { b = atoi(++s); while (s < end && *s >= '0' && *s <= '9') s++; }
                    for (int i = a; i <= b && i < CPU_SETSIZE; i++) CPU_SET(i, set);
                    if (s < end) s++;   // '+'
                }
            };
            const char* colon = strchr(spec, ':');
            const char* gend = colon ? colon : spec + strlen(spec);
            const char* s = spec;
            for (int k = 0; k < 3; k++) {
                const char* comma = (const char*)memchr(s, ',', gend - s);
                const char* tend = comma ? comma : gend;
                parse(s, tend, &gset[k]);
                s = comma ? comma + 1 : gend;
            }
            if (colon) parse(colon + 1, colon + strlen(colon), &rset); else CPU_ZERO(&rset);
            fprintf(stderr, "[affinity] TOTK_AFFINITY=%s (guest cores 0/1/2 + render side)\n", spec);
        }
    }
    if (!on) return;
    cpu_set_t set;
    if (kind == 1) set = rset;
    else {
        CPU_ZERO(&set);
        for (int k = 0; k < 3; k++) if (mask >> k & 1) CPU_OR(&set, &set, &gset[k]);
    }
    if (CPU_COUNT(&set) == 0) return;
    int r = sched_setaffinity(0, sizeof set, &set);
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 24) fprintf(stderr, "[affinity] tid %d kind %d mask %llx -> %d CPUs%s\n", (int)my_tid(), kind, mask,
                                          CPU_COUNT(&set), r ? " (failed)" : "");
}

DWORD GetCurrentThreadId(void) { return (DWORD)my_tid(); }
HANDLE GetCurrentThread(void) { return kSelfThread; }

HANDLE CreateThread(LPSECURITY_ATTRIBUTES, SIZE_T stack, LPTHREAD_START_ROUTINE fn, LPVOID arg, DWORD, LPDWORD tid) {
    ThreadObj* t = new ThreadObj();
    t->fn = fn; t->arg = arg;
    sem_init(&t->started, 0, 0);
    pthread_attr_t at; pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&at, stack >= (256 << 10) ? stack : 8u << 20);
    int rc = pthread_create(&t->th, &at, thread_main, t);
    pthread_attr_destroy(&at);
    if (rc) { sem_destroy(&t->started); delete t; errno = rc; return nullptr; }
    while (sem_wait(&t->started) != 0 && errno == EINTR) {}
    if (tid) *tid = (DWORD)t->tid.load();
    return t;
}

BOOL CloseHandle(HANDLE h) {
    if (!h || h == INVALID_HANDLE_VALUE || h == kSelfThread) return TRUE;
    Obj* o = (Obj*)h;
    if (o->kind == K_THREAD) unref((ThreadObj*)o);
    else if (o->kind == K_TIMER) { close(((TimerObj*)o)->fd); delete (TimerObj*)o; }
    return TRUE;
}

DWORD WaitForSingleObject(HANDLE h, DWORD ms) {
    if (!h || h == INVALID_HANDLE_VALUE || h == kSelfThread) return WAIT_FAILED;
    Obj* o = (Obj*)h;
    if (o->kind == K_THREAD) {
        ThreadObj* t = (ThreadObj*)o;
        std::unique_lock<std::mutex> l(t->m);
        if (ms == INFINITE) { t->cv.wait(l, [&] { return t->done; }); return WAIT_OBJECT_0; }
        return t->cv.wait_for(l, std::chrono::milliseconds(ms), [&] { return t->done; }) ? WAIT_OBJECT_0 : WAIT_TIMEOUT;
    }
    if (o->kind == K_TIMER) {
        TimerObj* t = (TimerObj*)o;
        struct pollfd pf = {t->fd, POLLIN, 0};
        int rc;
        while ((rc = poll(&pf, 1, ms == INFINITE ? -1 : (int)ms)) < 0 && errno == EINTR) {}
        if (rc <= 0) return WAIT_TIMEOUT;
        uint64_t exp; (void)!read(t->fd, &exp, sizeof exp);
        return WAIT_OBJECT_0;
    }
    return WAIT_FAILED;
}

BOOL SetThreadPriority(HANDLE h, int prio) {
    pid_t tid = 0;
    if (h == kSelfThread) tid = my_tid();
    else if (h && h != INVALID_HANDLE_VALUE && ((Obj*)h)->kind == K_THREAD) tid = ((ThreadObj*)h)->tid.load();
    if (!tid) return FALSE;
    int nice = prio >= THREAD_PRIORITY_TIME_CRITICAL ? -10 : prio > 0 ? -5 * prio : prio < 0 ? -5 * prio : 0;
    return setpriority(PRIO_PROCESS, (id_t)tid, nice) == 0;   // (raising needs RLIMIT_NICE / CAP_SYS_NICE: normally refused, harmless)
}

HRESULT SetThreadDescription(HANDLE h, PCWSTR name) {
    char nm[16]; int i = 0;
    for (; name && name[i] && i < 15; i++) nm[i] = (char)name[i];
    nm[i] = 0;
    if (h == kSelfThread) pthread_setname_np(pthread_self(), nm);
    else if (h && h != INVALID_HANDLE_VALUE && ((Obj*)h)->kind == K_THREAD) pthread_setname_np(((ThreadObj*)h)->th, nm);
    return 0;
}

HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES, const void*, DWORD, DWORD) {
    int fd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    if (fd < 0) return nullptr;
    TimerObj* t = new TimerObj(); t->fd = fd;
    return t;
}

BOOL SetWaitableTimerEx(HANDLE h, const LARGE_INTEGER* due, LONG period, void*, LPVOID, void*, ULONG) {
    if (!h || ((Obj*)h)->kind != K_TIMER) return FALSE;
    long long ns = due->QuadPart < 0 ? -due->QuadPart * 100 : 1;     // relative (negative, 100 ns units); absolute times are not used
    if (ns < 1) ns = 1;
    struct itimerspec its = {};
    its.it_value.tv_sec = ns / 1000000000ll; its.it_value.tv_nsec = ns % 1000000000ll;
    its.it_interval.tv_sec = period / 1000; its.it_interval.tv_nsec = (period % 1000) * 1000000ll;
    return timerfd_settime(((TimerObj*)h)->fd, 0, &its, nullptr) == 0;
}

// ------------------------------------------------------------------ signals, exe location
}  // extern "C"

namespace {
RtCrashFn g_crash_cb;
void sig_handler(int sig, siginfo_t* si, void* uc) {
    if (sig == SIGSEGV && si->si_code == SEGV_ACCERR && ww_fault((uintptr_t)si->si_addr)) return;
    alarm(10);                            // the dumper uses stdio: if the fault was inside it, die anyway after 10 s
    if (g_crash_cb) g_crash_cb(sig, si, uc);
    signal(sig, SIG_DFL);                 // no callback / it returned: the default action (core dump) on return or re-raise
    if (sig == SIGABRT) raise(sig);
}
}  // namespace

extern "C" {

void rt_posix_thread_init(void) {
    static thread_local bool done;
    if (done) return;
    done = true;
    const size_t sz = 256 << 10;
    void* s = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (s == MAP_FAILED) return;
    stack_t ss = {};
    ss.ss_sp = s; ss.ss_size = sz;
    sigaltstack(&ss, nullptr);
}

void rt_posix_install_signals(RtCrashFn cb) {
    g_crash_cb = cb;
    rt_posix_thread_init();
    struct sigaction sa = {};
    sa.sa_sigaction = sig_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (int s : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(s, &sa, nullptr);
}

const char* rt_posix_exe_dir(void) {
    static std::string dir = [] {
        char b[4096]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
        if (n <= 0) return std::string(".");
        b[n] = 0;
        std::string s = b; size_t k = s.rfind('/');
        return k == std::string::npos ? std::string(".") : s.substr(0, k);
    }();
    return dir.c_str();
}
unsigned long long rt_posix_exe_base(void) { return (unsigned long long)(uintptr_t)&__ehdr_start; }

// ---- GPU-owned guest memory (amdgpu). The guest heap (2.1 GB, shader global memory) and the alias window (2 GB, vertex / index
// data) used to reach the GPU as host-pointer imports, which amdgpu turns into userptr buffer objects watched by an MMU notifier:
// whenever the kernel swaps out, migrates (compaction) or collapses (THP) a page in the range, the notifier waits without
// interruption for all GPU work using the buffer, and the next submit faults in and re-maps the whole range. Deck v2 log
// (2026-10-16): no frame for 6 s, the submit thread in D state in dma_fence_wait_any_timeout inside vkQueueSubmit2, 305 MB of swap
// in use, VRAM 1016/1024 MB; the Deck lagged as a whole and could not suspend. A GTT buffer object allocated by the driver has
// none of this: its pages are not on the LRU, so reclaim and compaction leave them alone. rt_gpu_bo_map puts one at the address
// of an existing reservation (MAP_FIXED over it, before the game writes there), and exec.c imports it into Vulkan by dma-buf.
// Only with exactly one DRM render node and it amdgpu (the Deck); TOTK_GPU_BO=0 turns it off, =1 allows several render nodes.
namespace {
struct RtDrmVersion { int major, minor, patch; size_t name_len; char* name; size_t date_len; char* date; size_t desc_len; char* desc; };
struct RtAmdGemCreate { uint64_t bo_size, alignment, domains, domain_flags; };   // union drm_amdgpu_gem_create (out: u32 handle)
struct RtAmdGemMmap { uint64_t v; };                                             // union drm_amdgpu_gem_mmap (in: u32 handle, out: u64 offset)
struct RtDrmPrimeHandle { uint32_t handle, flags; int32_t fd; };
struct RtDrmGemClose { uint32_t handle, pad; };
struct GpuBlock { uintptr_t base; size_t size; int fd; };
GpuBlock g_gpu_blocks[8];
int g_gpu_nblocks, g_gpu_fd = -2;
std::mutex g_gpu_mu;

int gpu_node(void) {   // the amdgpu render node, or -1 (under g_gpu_mu)
    if (g_gpu_fd != -2) return g_gpu_fd;
    g_gpu_fd = -1;
    const char* e = getenv("TOTK_GPU_BO");
    if (e && *e == '0') { printf("[gpu] driver-owned guest memory off (TOTK_GPU_BO=0)\n"); return -1; }
    int amd = -1, nodes = 0;
    for (int i = 128; i < 192; i++) {
        char p[32]; snprintf(p, sizeof p, "/dev/dri/renderD%d", i);
        int fd = open(p, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        nodes++;
        char name[32] = {0};
        RtDrmVersion v = {}; v.name = name; v.name_len = sizeof name - 1;
        if (ioctl(fd, _IOWR('d', 0x00, RtDrmVersion), &v) == 0 && !strcmp(name, "amdgpu") && amd < 0) amd = fd;
        else close(fd);
    }
    if (amd >= 0 && (nodes == 1 || (e && *e == '1'))) g_gpu_fd = amd;
    else if (amd >= 0) close(amd);
    printf("[gpu] %d render node(s), amdgpu %s: guest heap / alias window %s\n", nodes, amd >= 0 ? "yes" : "no",
           g_gpu_fd >= 0 ? "in driver-owned GTT buffers" : "stay host memory (host-pointer import)");
    return g_gpu_fd;
}
}  // namespace

int rt_gpu_bo_map(void* addr, size_t size, int prot) {
    std::lock_guard<std::mutex> l(g_gpu_mu);
    int fd = gpu_node();
    if (fd < 0 || g_gpu_nblocks == 8) return -1;
    size = round_up(size, kPg);
    RtAmdGemCreate cr = {size, 2u << 20, 0x2 /* AMDGPU_GEM_DOMAIN_GTT */, 1u << 7 /* AMDGPU_GEM_CREATE_EXPLICIT_SYNC */};
    if (ioctl(fd, _IOWR('d', 0x40, RtAmdGemCreate), &cr) != 0) {
        printf("[gpu] GTT buffer of %zu MB: create failed (errno %d)\n", size >> 20, errno);
        return -1;
    }
    uint32_t handle = (uint32_t)cr.bo_size;   // (out.handle overlays the first field)
    RtAmdGemMmap mm = {handle};
    RtDrmPrimeHandle ph = {handle, O_CLOEXEC | O_RDWR, -1};
    void* p = MAP_FAILED;
    if (ioctl(fd, _IOWR('d', 0x41, RtAmdGemMmap), &mm) == 0 && ioctl(fd, _IOWR('d', 0x2d, RtDrmPrimeHandle), &ph) == 0)
        p = mmap(addr, size, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)mm.v);
    RtDrmGemClose gc = {handle, 0};
    ioctl(fd, _IOW('d', 0x09, RtDrmGemClose), &gc);   // (the mapping and the dma-buf keep the buffer alive)
    if (p != addr) {
        printf("[gpu] GTT buffer of %zu MB at %p: %s failed (errno %d)\n", size >> 20, addr, ph.fd < 0 ? "export" : "map", errno);
        if (p != MAP_FAILED) munmap(p, size);
        if (ph.fd >= 0) close(ph.fd);
        // MAP_FIXED may already have replaced the reservation: put the anonymous PROT_NONE / RW mapping back
        mmap(addr, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
        return -1;
    }
    g_gpu_blocks[g_gpu_nblocks++] = {(uintptr_t)addr, size, ph.fd};
    printf("[gpu] %p + %zu MB: driver-owned GTT buffer (dma-buf %d)\n", addr, size >> 20, ph.fd);
    return ph.fd;
}

int rt_gpu_bo_available(void) {
    std::lock_guard<std::mutex> l(g_gpu_mu);
    return gpu_node() >= 0;
}

int rt_gpu_bo_find(unsigned long long addr, unsigned long long size, unsigned long long* block_size) {
    std::lock_guard<std::mutex> l(g_gpu_mu);
    for (int i = 0; i < g_gpu_nblocks; i++)
        if (addr == g_gpu_blocks[i].base && size <= g_gpu_blocks[i].size) { *block_size = g_gpu_blocks[i].size; return g_gpu_blocks[i].fd; }
    return -1;
}

// ---- system and hang diagnostics (nvn/vulkan/device.c). The first Deck run (2026-10-16) stopped rendering at 274 s with no
// GPU-hang report, and the process later could not be frozen for suspend ("Failed to freeze unit user.slice"): a thread stuck
// in the kernel. rt_sys_line: one line of memory / GPU memory use per stats window. rt_threads_dump: per thread the
// scheduler state, the kernel function it sleeps in (wchan) and its syscall; plus a user-space backtrace of one thread.
static long kb_field(const char* path, const char* key) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    size_t kl = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, kl)) { v = atol(line + kl); break; }
    fclose(f);
    return v;
}
static long long file_ll(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return -1;
    long long v = -1;
    if (fscanf(f, "%lld", &v) != 1) v = -1;
    fclose(f);
    return v;
}
static std::string drm_dir() {   // first DRM card that reports amdgpu memory use
    static std::string d = [] {
        for (int i = 0; i < 8; i++) {
            std::string p = "/sys/class/drm/card" + std::to_string(i) + "/device";
            if (file_ll(p + "/mem_info_gtt_total") > 0) return p;
        }
        return std::string();
    }();
    return d;
}
static std::string first_line(const std::string& path);
// This process's GPU memory placement and engine time from the DRM fdinfo of its render-node fds (kernel 6.x amdgpu: drm-memory-vram
// / -gtt, amd-requested-vram / -gtt, amd-evicted-vram, drm-engine-gfx ns), one client per drm-client-id. d15 2026-10-16: RADV's
// "device-local" heap is 6 GB on the Deck but VRAM is a 1 GB carve-out (full), so the kernel decides which buffers live in VRAM;
// GPU ms per frame in the same scene differed 14 vs 20 between runs.
static void drm_fdinfo(char* out, size_t n) {
    std::map<std::string, long long> sum;
    std::map<std::string, bool> seen;
    if (DIR* fd = opendir("/proc/self/fd")) {
        while (dirent* e = readdir(fd)) {
            char link[256], tgt[256];
            snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
            ssize_t k = readlink(link, tgt, sizeof tgt - 1);
            if (k <= 0) continue;
            tgt[k] = 0;
            if (strncmp(tgt, "/dev/dri/", 9)) continue;
            snprintf(link, sizeof link, "/proc/self/fdinfo/%s", e->d_name);
            FILE* f = fopen(link, "r");
            if (!f) continue;
            char line[256];
            std::map<std::string, long long> one;
            std::string id;
            while (fgets(line, sizeof line, f)) {
                char key[96];
                long long v;
                if (sscanf(line, "drm-client-id: %95s", key) == 1) { id = key; continue; }
                if (sscanf(line, "%95[^:]: %lld", key, &v) == 2) one[key] = v;
            }
            fclose(f);
            if (id.empty() || seen[id]) continue;
            seen[id] = true;
            for (auto& kv : one) sum[kv.first] += kv.second;
        }
        closedir(fd);
    }
    static long long prev_gfx = -1, prev_t = 0;
    timespec ts_;
    clock_gettime(CLOCK_MONOTONIC, &ts_);
    long long now = ts_.tv_sec * 1000000000ll + ts_.tv_nsec, gfx = sum.count("drm-engine-gfx") ? sum["drm-engine-gfx"] : -1;
    double busy = (prev_gfx >= 0 && gfx >= 0 && now > prev_t) ? 100.0 * (gfx - prev_gfx) / (now - prev_t) : -1;
    prev_gfx = gfx; prev_t = now;
    // (memory values are KiB in fdinfo)
    snprintf(out, n, "drm: vram %lld MB (req %lld, evicted %lld), gtt %lld MB (req %lld), gfx busy %.0f %%",
             sum["drm-memory-vram"] >> 10, sum["amd-requested-vram"] >> 10, sum["amd-evicted-vram"] >> 10, sum["drm-memory-gtt"] >> 10,
             sum["amd-requested-gtt"] >> 10, busy);
}
void rt_sys_line(char* out, size_t n) {
    long rss = kb_field("/proc/self/status", "VmRSS:"), avail = kb_field("/proc/meminfo", "MemAvailable:");
    long swt = kb_field("/proc/meminfo", "SwapTotal:"), swf = kb_field("/proc/meminfo", "SwapFree:");
    std::string d = drm_dir();
    long long vu = -1, vt = -1, gu = -1, gt = -1;
    if (!d.empty()) {
        vu = file_ll(d + "/mem_info_vram_used"); vt = file_ll(d + "/mem_info_vram_total");
        gu = file_ll(d + "/mem_info_gtt_used"); gt = file_ll(d + "/mem_info_gtt_total");
    }
    long long busy = d.empty() ? -1 : file_ll(d + "/gpu_busy_percent");   // (amdgpu: an instant sample, not an average)
    // GPU core clock now (the "*" row of pp_dpm_sclk) and the DPM mode: the Deck's per-game "manual GPU clock" pins it (user: 1600 MHz)
    std::string clk = "?", lvl = d.empty() ? "?" : first_line(d + "/power_dpm_force_performance_level");
    if (FILE* f = d.empty() ? nullptr : fopen((d + "/pp_dpm_sclk").c_str(), "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (strchr(line, '*')) { const char* c = strchr(line, ':'); clk = c ? c + 1 : line; clk.erase(clk.find_last_not_of(" *\n") + 1); clk.erase(0, clk.find_first_not_of(' ')); break; }
        fclose(f);
    }
    while (!lvl.empty() && (lvl.back() == '\n' || lvl.back() == ' ')) lvl.pop_back();
    // CPU core clocks now (cpufreq, kHz): the Deck shares one power budget between CPU and GPU, so a pinned GPU clock can
    // pull the CPU cores down (d8 2026-10-16: 44 fps with neither a thread nor the GPU saturated)
    long long fmin = -1, fmax = -1, fsum = 0;
    int fn = 0;
    for (int c = 0; c < 64; c++) {
        long long f = file_ll("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/scaling_cur_freq");
        if (f <= 0) { if (c >= 8) break; continue; }
        fsum += f; fn++;
        if (fmin < 0 || f < fmin) fmin = f;
        if (f > fmax) fmax = f;
    }
    // Real GPU clock and APU power (amdgpu hwmon: freq1 = sclk in Hz, power1/2 = slowPPT/fastPPT in uW, temp1 in m°C): pp_dpm_sclk's
    // "*" row is the set point; d14 2026-10-16 showed 14 vs 20 ms GPU per frame in the same scene between runs whose CPU clocks
    // differed (shared 15 W budget?)
    long long sclk = -1, p1 = -1, p2 = -1, cap = -1, temp = -1;
    if (!d.empty())
        if (DIR* hd = opendir((d + "/hwmon").c_str())) {
            while (dirent* e = readdir(hd))
                if (!strncmp(e->d_name, "hwmon", 5)) {
                    std::string h = d + "/hwmon/" + e->d_name;
                    sclk = file_ll(h + "/freq1_input"); p1 = file_ll(h + "/power1_average"); p2 = file_ll(h + "/power2_average");
                    cap = file_ll(h + "/power1_cap"); temp = file_ll(h + "/temp1_input");
                    break;
                }
            closedir(hd);
        }
    snprintf(out, n, "game RSS %ld MB | RAM available %ld MB | swap used %ld MB | VRAM %lld/%lld MB | GTT %lld/%lld MB | GPU busy now %lld %%, clock %s (%s) | CPU MHz %lld..%lld avg %lld"
             " | sclk now %lld MHz, APU %.1f W (fast %.1f, cap %.0f), %lld C",
             rss >> 10, avail >> 10, (swt - swf) >> 10, vu >> 20, vt >> 20, gu >> 20, gt >> 20, busy, clk.c_str(), lvl.c_str(),
             fmin / 1000, fmax / 1000, fn ? fsum / fn / 1000 : -1, sclk > 0 ? sclk / 1000000 : -1, p1 / 1e6, p2 / 1e6, cap / 1e6,
             temp > 0 ? temp / 1000 : -1);
    size_t l = strlen(out);
    if (l + 8 < n) { snprintf(out + l, n - l, " | "); drm_fdinfo(out + l + 3, n - l - 3); }
}
// Per-thread CPU use since the previous call (Deck v3, 2026-10-16: 25-30 fps in gameplay, CPU- or GPU-bound unknown; TOTK_PROF
// sampling is Windows-only): the process total in % of one core, then the busiest threads (name from SetThreadDescription /
// pthread_setname_np: guest threads, "render") with their % of one core.
void rt_cpu_line(char* out, size_t n) {
    static std::map<int, long long> prev;
    static long long prev_ns;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    long long now = ts.tv_sec * 1000000000ll + ts.tv_nsec, dt = now - prev_ns;
    double hz = (double)sysconf(_SC_CLK_TCK);
    std::map<int, long long> cur;
    std::vector<std::pair<long long, std::string>> rows;
    long long total = 0;
    if (DIR* dir = opendir("/proc/self/task")) {
        while (dirent* e = readdir(dir)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            int tid = atoi(e->d_name);
            std::string st = first_line("/proc/self/task/" + std::string(e->d_name) + "/stat");
            size_t lp = st.find('('), rp = st.rfind(')');
            if (lp == std::string::npos || rp == std::string::npos) continue;
            std::string comm = st.substr(lp + 1, rp - lp - 1);
            long long ut = 0, stt = 0; int f = 0;   // fields after ")": state(3) ... utime(14) stime(15)
            for (const char* p = st.c_str() + rp + 2; *p && f < 13; p++) if (*p == ' ') { f++; if (f == 11) ut = atoll(p + 1); if (f == 12) stt = atoll(p + 1); }
            long long t = ut + stt;
            cur[tid] = t;
            auto it = prev.find(tid);
            long long d = it == prev.end() ? 0 : t - it->second;
            if (prev_ns && d > 0) { total += d; rows.push_back({d, comm + "/" + std::to_string(tid)}); }
        }
        closedir(dir);
    }
    prev.swap(cur);
    long long pns = prev_ns; prev_ns = now;
    if (!pns || dt <= 0) { snprintf(out, n, "(first sample)"); return; }
    auto pct = [&](long long ticks) { return ticks / hz * 1e9 / (double)dt * 100.0; };
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    int k = snprintf(out, n, "process %.0f %% of a core (%ld cores) |", pct(total), sysconf(_SC_NPROCESSORS_ONLN));
    for (size_t i = 0; i < rows.size() && i < 8 && k > 0 && (size_t)k < n; i++)
        k += snprintf(out + k, n - k, " %s %.0f", rows[i].second.c_str(), pct(rows[i].first));
}
static void* g_bt[48];
static std::atomic<int> g_bt_n{-1};
static void bt_handler(int) { g_bt_n.store(backtrace(g_bt, 48)); }
static std::string first_line(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "?";
    char b[512] = "";
    if (!fgets(b, sizeof b, f)) b[0] = 0;
    fclose(f);
    std::string s = b;
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.empty() ? "-" : s;
}
void rt_threads_dump(int bt_tid) {
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return;
    int n = 0;
    while (dirent* e = readdir(dir)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        std::string t = std::string("/proc/self/task/") + e->d_name;
        std::string st = first_line(t + "/stat");
        size_t k = st.rfind(')');
        char state = k != std::string::npos && k + 2 < st.size() ? st[k + 2] : '?';
        std::string sc = first_line(t + "/syscall");
        if (sc.size() > 40) sc = sc.substr(0, 40);
        // (sleeping threads that wait in futexes / poll are the normal idle state: list the rest in full, those briefly)
        std::string wc = first_line(t + "/wchan");
        bool idle = state == 'S' && (wc.find("futex") != std::string::npos || wc.find("poll") != std::string::npos ||
                                     wc.find("nanosleep") != std::string::npos || wc.find("hrtimer") != std::string::npos);
        if (idle && atoi(e->d_name) != bt_tid) { n++; continue; }
        printf("[hang] thread %s %s: state %c, kernel wait %s, syscall %s\n", e->d_name, first_line(t + "/comm").c_str(), state,
               wc.c_str(), sc.c_str());
    }
    closedir(dir);
    printf("[hang] (%d more threads idle in futex / poll / sleep waits)\n", n);
    if (bt_tid > 0) {
        backtrace(g_bt, 1);   // (loads libgcc's unwinder outside the signal handler)
        struct sigaction sa = {}, old = {};
        sa.sa_handler = bt_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGUSR2, &sa, &old);
        g_bt_n.store(-1);
        syscall(SYS_tgkill, getpid(), bt_tid, SIGUSR2);
        for (int i = 0; i < 100 && g_bt_n.load() < 0; i++) usleep(10000);
        int m = g_bt_n.load();
        if (m < 0) printf("[hang] thread %d did not take a signal within 1 s: blocked inside the kernel\n", bt_tid);
        for (int i = 2; i < m; i++) {   // (0-1: the handler and the signal frame)
            Dl_info di = {};
            if (dladdr(g_bt[i], &di) && di.dli_fname)
                printf("[hang]   #%d %s+%#lx %s\n", i - 2, strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname,
                       (unsigned long)((uintptr_t)g_bt[i] - (uintptr_t)di.dli_fbase), di.dli_sname ? di.dli_sname : "");
            else printf("[hang]   #%d %p\n", i - 2, g_bt[i]);
        }
        sigaction(SIGUSR2, &old, nullptr);
    }
    fflush(stdout);
}

}  // extern "C"
#endif

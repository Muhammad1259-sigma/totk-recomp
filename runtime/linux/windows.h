// TotK Linux port: the subset of <windows.h> the runtime and the NVN layer use, implemented on POSIX (runtime/linux/plat_posix.cpp).
// Only the Linux build puts runtime/linux on the include path (tools/recomp/mkninja_linux.py), so `#include <windows.h>` in
// hle_os_mem.cpp, hle_os_thread.cpp, hle_vi.cpp, hle_time.cpp, nvn/cmdbuf.c, ... resolves to this file there and to the real header on
// Windows. LP64: DWORD/LONG are 64-bit here (printf("%lu") of a DWORD, `long` variables passed to Interlocked*).
// Not emulated (the files that use them have #ifdef _WIN32 branches): XInput / raw HID, WASAPI, GDI, Media Foundation, SEH,
// thread suspend / context, VirtualQuery.
#pragma once
#ifndef _WIN32
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WINAPI
#define CALLBACK
#define WIN32_LEAN_AND_MEAN
#define INFINITE 0xFFFFFFFFul
#define TRUE 1
#define FALSE 0
typedef unsigned long DWORD, ULONG;
typedef long LONG;
typedef long long LONGLONG;
typedef unsigned long long ULONGLONG, DWORD64;
typedef unsigned short WORD, USHORT;
typedef short SHORT;
typedef unsigned char BYTE;
typedef unsigned char BOOLEAN;
typedef int BOOL;
typedef unsigned int UINT;
typedef void *HANDLE, *LPVOID, *PVOID;
typedef const void *LPCVOID;
typedef size_t SIZE_T;
typedef uintptr_t ULONG_PTR;
typedef DWORD *LPDWORD;
typedef const wchar_t *PCWSTR;            // (wchar_t is 32-bit here; SetThreadDescription callers pass ASCII thread names)
typedef long HRESULT;
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define SUCCEEDED(hr) ((hr) >= 0)
#define FAILED(hr) ((hr) < 0)

typedef union _LARGE_INTEGER {
    struct { DWORD LowPart; LONG HighPart; };
    LONGLONG QuadPart;
} LARGE_INTEGER;

// ---- time
static inline BOOL QueryPerformanceFrequency(LARGE_INTEGER *f) { f->QuadPart = 1000000000ll; return TRUE; }   // ns ticks
static inline BOOL QueryPerformanceCounter(LARGE_INTEGER *c) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    c->QuadPart = (LONGLONG)ts.tv_sec * 1000000000ll + ts.tv_nsec;
    return TRUE;
}
static inline ULONGLONG GetTickCount64(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ULONGLONG)ts.tv_sec * 1000ull + (ULONGLONG)ts.tv_nsec / 1000000ull;
}
static inline void Sleep(DWORD ms) {
    struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000l};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}
static inline BOOL SwitchToThread(void) { sched_yield(); return TRUE; }

typedef int64_t __time64_t;
static inline __time64_t _time64(__time64_t *p) { time_t t = time(NULL); if (p) *p = t; return t; }
static inline int _gmtime64_s(struct tm *out, const __time64_t *t) { time_t tt = (time_t)*t; return gmtime_r(&tt, out) ? 0 : EINVAL; }
static inline int _localtime64_s(struct tm *out, const __time64_t *t) { time_t tt = (time_t)*t; return localtime_r(&tt, out) ? 0 : EINVAL; }
static inline __time64_t _mkgmtime64(struct tm *tm) { return (__time64_t)timegm(tm); }

// ---- misc CRT
#define _strnicmp strncasecmp
static inline void *_aligned_malloc(size_t size, size_t align) { void *p = NULL; if (align < sizeof(void *)) align = sizeof(void *); return posix_memalign(&p, align, size) ? NULL : p; }
static inline void _aligned_free(void *p) { free(p); }

// ---- interlocked / SRW locks (cmdbuf.c)
static inline LONG InterlockedExchange(volatile LONG *p, LONG v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
static inline LONG InterlockedCompareExchange(volatile LONG *p, LONG exch, LONG cmp) { __atomic_compare_exchange_n(p, &cmp, exch, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); return cmp; }
static inline LONG InterlockedIncrement(volatile LONG *p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
static inline LONG InterlockedDecrement(volatile LONG *p) { return __atomic_sub_fetch(p, 1, __ATOMIC_SEQ_CST); }
typedef pthread_mutex_t SRWLOCK;
#define SRWLOCK_INIT PTHREAD_MUTEX_INITIALIZER
static inline void InitializeSRWLock(SRWLOCK *l) { pthread_mutex_init(l, NULL); }
static inline void AcquireSRWLockExclusive(SRWLOCK *l) { pthread_mutex_lock(l); }
static inline void ReleaseSRWLockExclusive(SRWLOCK *l) { pthread_mutex_unlock(l); }
static inline void AcquireSRWLockShared(SRWLOCK *l) { pthread_mutex_lock(l); }
static inline void ReleaseSRWLockShared(SRWLOCK *l) { pthread_mutex_unlock(l); }

// ---- virtual memory (plat_posix.cpp): reserve = PROT_NONE mapping, commit = mprotect(RW), decommit = MADV_DONTNEED + PROT_NONE.
// MEM_WRITE_WATCH regions track writes with mprotect + a SIGSEGV handler (see plat_posix.cpp for the cost and the protocol).
#define MEM_COMMIT 0x00001000u
#define MEM_RESERVE 0x00002000u
#define MEM_DECOMMIT 0x00004000u
#define MEM_RELEASE 0x00008000u
#define MEM_WRITE_WATCH 0x00200000u
#define PAGE_NOACCESS 0x01u
#define PAGE_READONLY 0x02u
#define PAGE_READWRITE 0x04u
#define WRITE_WATCH_FLAG_RESET 1u
LPVOID VirtualAlloc(LPVOID addr, SIZE_T size, DWORD type, DWORD prot);
BOOL VirtualFree(LPVOID addr, SIZE_T size, DWORD type);
UINT GetWriteWatch(DWORD flags, PVOID base, SIZE_T size, PVOID *addrs, ULONG_PTR *count, DWORD *gran);
UINT ResetWriteWatch(PVOID base, SIZE_T size);
// Kernel writes into a tracked region (read(2) into a guest buffer) fail with EFAULT on a write-protected page instead of raising
// SIGSEGV: call this first on the destination (marks the pages written and unprotects them). No-op outside tracked regions.
void rt_ww_prepare_kernel_write(const void *p, size_t n);
// Is [p, p + n) readable? (process_vm_readv probe: EFAULT instead of a signal.) Used by the crash / watchdog dumpers.
int rt_posix_readable(unsigned long long p, unsigned long long n);

// ---- threads / waitable timers (plat_posix.cpp)
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x00010000u
#define THREAD_PRIORITY_LOWEST (-2)
#define THREAD_PRIORITY_BELOW_NORMAL (-1)
#define THREAD_PRIORITY_NORMAL 0
#define THREAD_PRIORITY_ABOVE_NORMAL 1
#define THREAD_PRIORITY_TIME_CRITICAL 15
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2u
#define TIMER_ALL_ACCESS 0x1F0003u
#define WAIT_OBJECT_0 0
#define WAIT_TIMEOUT 0x102ul
#define WAIT_FAILED 0xFFFFFFFFul
typedef DWORD (*LPTHREAD_START_ROUTINE)(LPVOID);
typedef void *LPSECURITY_ATTRIBUTES;
HANDLE CreateThread(LPSECURITY_ATTRIBUTES sa, SIZE_T stack, LPTHREAD_START_ROUTINE fn, LPVOID arg, DWORD flags, LPDWORD tid);
DWORD WaitForSingleObject(HANDLE h, DWORD ms);   // thread handles (done) and waitable timers (fired)
BOOL CloseHandle(HANDLE h);
HANDLE GetCurrentThread(void);                   // pseudo handle
DWORD GetCurrentThreadId(void);
BOOL SetThreadPriority(HANDLE h, int prio);      // lowering works unprivileged; raising is tried and ignored on EACCES
HRESULT SetThreadDescription(HANDLE h, PCWSTR name);
HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES sa, const void *name, DWORD flags, DWORD access);
BOOL SetWaitableTimerEx(HANDLE h, const LARGE_INTEGER *due, LONG period, void *cb, LPVOID arg, void *ctx, ULONG delay);
static inline HANDLE GetCurrentProcess(void) { return (HANDLE)(intptr_t)-1; }
static inline BOOL TerminateProcess(HANDLE h, unsigned code) { (void)h; _exit((int)code); return TRUE; }
static inline void ExitProcess(unsigned code) { _exit((int)code); }
static inline DWORD GetLastError(void) { return (DWORD)errno; }

#ifdef __cplusplus
}
#endif
#endif   // !_WIN32

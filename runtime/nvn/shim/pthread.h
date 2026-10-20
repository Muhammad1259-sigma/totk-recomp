// TotK: the pthread subset LASRecomp's Vulkan backend uses, on Win32 SRW locks / condition variables / threads.
#pragma once
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
typedef void pthread_attr_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

static inline int pthread_mutex_init(pthread_mutex_t *m, const void *a) { (void)a; InitializeSRWLock(m); return 0; }
static inline int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m) { AcquireSRWLockExclusive(m); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(m); return 0; }
static inline int pthread_cond_init(pthread_cond_t *c, const void *a) { (void)a; InitializeConditionVariable(c); return 0; }
static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) { SleepConditionVariableSRW(c, m, INFINITE, 0); return 0; }
static inline int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }

typedef struct { void *(*fn)(void *); void *arg; } pthread_start_;
static DWORD WINAPI pthread_tramp_(LPVOID p) {
    pthread_start_ s = *(pthread_start_ *)p; HeapFree(GetProcessHeap(), 0, p); s.fn(s.arg); return 0;
}
static inline int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    (void)a;
    pthread_start_ *s = (pthread_start_ *)HeapAlloc(GetProcessHeap(), 0, sizeof *s);
    s->fn = fn; s->arg = arg;
    *t = CreateThread(NULL, 8u << 20, pthread_tramp_, s, 0, NULL);
    return *t ? 0 : -1;
}
static inline int pthread_detach(pthread_t t) { CloseHandle(t); return 0; }
static inline int pthread_join(pthread_t t, void **r) { (void)r; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0; }

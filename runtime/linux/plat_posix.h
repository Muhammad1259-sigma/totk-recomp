// TotK Linux port: platform services that have no Win32 spelling in the runtime (signals, exe location). Linux build only.
#pragma once
#ifndef _WIN32
#include <signal.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Fatal-signal callback: called on the faulting thread (alternate signal stack) for SIGSEGV that is not a write-watch fault, SIGBUS,
// SIGILL, SIGFPE, SIGABRT. Must not return (print, then _exit); if it does the default action runs.
typedef void (*RtCrashFn)(int sig, siginfo_t *si, void *ucontext);
void rt_posix_install_signals(RtCrashFn cb);
// sigaltstack for the calling thread (the host stacks are huge but a guest stack overflow must still reach the crash dumper)
void rt_posix_thread_init(void);
// directory of the running executable (no trailing slash) and its load address (non-PIE: the ELF header, for "exe+offset" in dumps)
const char *rt_posix_exe_dir(void);
unsigned long long rt_posix_exe_base(void);
void rt_sys_line(char* out, size_t n);   // memory / GPU memory use, one line (device.c stats)
void rt_cpu_line(char* out, size_t n);   // per-thread CPU % since the previous call (device.c stats)
void rt_threads_dump(int bt_tid);       // per-thread kernel waits + a backtrace of bt_tid (device.c hang watch)
// GPU-owned guest memory (amdgpu only): replaces [addr, addr + size) of an existing reservation with a driver-owned GTT buffer
// mapped at the same address (prot PROT_NONE or PROT_READ | PROT_WRITE). Returns its dma-buf fd, or -1 (keep host memory).
int rt_gpu_bo_map(void* addr, size_t size, int prot);
int rt_gpu_bo_available(void);   // 1 = rt_gpu_bo_map can work here (one render node, amdgpu, TOTK_GPU_BO not 0)
// dma-buf fd of the block that starts at addr and covers size bytes (its size in *block_size), or -1
int rt_gpu_bo_find(unsigned long long addr, unsigned long long size, unsigned long long* block_size);
// number of write-watch faults handled / mprotect failures (stats line)
unsigned long long rt_ww_faults(void);
#ifdef __cplusplus
}
#endif
#endif

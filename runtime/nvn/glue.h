// Glue between the NVN layer ported from LASRecomp (C, MIT, see LICENSE.LASRecomp) and the TotK runtime (C++).
// Replaces LASRecomp's ../hle.h, ../kernel.h, ../rt.h for these files.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Guest CPU context: the C view of runtime/rt.h `struct Ctx` (static_assert'd on the C++ side as well).
typedef __uint128_t u128a __attribute__((aligned(8)));
typedef struct Cpu {
    uint64_t x[32];
    uint64_t sp;
    u128a v[32];
    uint8_t n, z, c_, vf;
    uint32_t fpcr, fpsr;
    uint64_t tpidr;
} Cpu;
_Static_assert(offsetof(Cpu, sp) == 256 && offsetof(Cpu, v) == 264, "Cpu must match runtime/rt.h Ctx");
typedef void (*HostFn)(Cpu *c);

// NVN proc registration: hle_nvn.cpp hands these out from nvnBootstrapLoader / nvnDeviceGetProcAddress.
void nvn_register_impl(const char *name, HostFn fn);
#define HLE_CAT2(a, b) a##b
#define HLE_CAT(a, b) HLE_CAT2(a, b)
#define HLE_FUNC(sym, fn)                                                                   \
    static void fn(Cpu *c);                                                                 \
    __attribute__((constructor)) static void HLE_CAT(hle_reg_, __LINE__)(void) { nvn_register_impl(sym, fn); } \
    static void fn(Cpu *c)

#define LOG(...) printf(__VA_ARGS__)
_Noreturn void rt_fatal_guest(const char *msg);    // runtime/rt_main.cpp: message + guest backtrace, exit 4
#define FATAL(...) do { char fatal_[768]; snprintf(fatal_, sizeof fatal_, __VA_ARGS__); rt_fatal_guest(fatal_); } while (0)
#define ENV(name) getenv(name)

// Call a guest function at `addr` with the current context (runtime/rt_main.cpp).
void guest_call(Cpu *c, uint64_t addr);

// LASRecomp tears the game down in place for reboots; TotK doesn't, so reset hooks are never called.
#define RT_ON_RESET(fn) static void fn(void) __attribute__((unused)); static void fn(void)
uint64_t plat_now_ns(void);   // monotonic ns (backend_null.c / platform)

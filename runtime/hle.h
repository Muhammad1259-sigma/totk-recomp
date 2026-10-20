// HLE registration: native implementations of SDK imports, looked up by symbol name at load time.
#pragma once
#include "rt.h"

constexpr int kMaxImports = 1024;
constexpr int kMaxHostFns = 4096;
u64 rt_host_fn_addr(GuestFn fn);   // host function -> callable guest address

struct HleReg { HleReg(const char* name, GuestFn fn); };

// HLE(memcpy) { ... }            for C names
// HLE_AS(nn_os_Foo, "_ZN2nn2os3FooEv") { ... }   for mangled names
#define HLE_AS(id, sym) static void hle_##id(Ctx* c); static HleReg reg_##id(sym, hle_##id); static void hle_##id(Ctx* c)
#define HLE(id) HLE_AS(id, #id)

// argument / return helpers (AAPCS64)
RT_INLINE void* P(u64 a) { return (void*)a; }
RT_INLINE float  F(Ctx* c, int i) { return c->v[i].f[0]; }
RT_INLINE double D(Ctx* c, int i) { return c->v[i].fd[0]; }
RT_INLINE void retf(Ctx* c, float v)  { c->v[0] = V128{}; c->v[0].f[0] = v; }
RT_INLINE void retd(Ctx* c, double v) { c->v[0] = V128{}; c->v[0].fd[0] = v; }

// loader services for HLE modules (rt_main.cpp)
Ctx* rt_new_ctx(u64 stack_top, u64* stack_lo, u64* stack_size);
void rt_bind_thread(Ctx* c);
void rt_unbind_thread();
RT_INLINE s64 timespan_ns(u64 v) { return (s64)v; }        // nn::TimeSpan is passed as one s64 of nanoseconds

extern thread_local const char* t_hle_cur;   // set by the import / NVN dispatchers, printed by the crash handler

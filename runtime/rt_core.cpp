// Runtime pieces that generated code needs regardless of the OS layer: ARM FP estimates, FPCR, traps.
#include "rt.h"
#include <cstdio>
#include <cstdlib>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// ---- ARMv8 FPRecipEstimate / FPRSqrtEstimate for single precision (FPCR.FZ=0, DN=0, RN), per the ARM ARM pseudocode.
static u32 recip_estimate(u32 a) {            // 256..511 -> 256..511
    a = a * 2 + 1;
    u32 b = (1u << 19) / a;
    return (b + 1) / 2;
}
static u32 recip_sqrt_estimate(u32 a) {       // 128..511 -> 256..511
    if (a < 256) a = a * 2 + 1;
    else { a = (a >> 1) << 1; a = (a + 1) * 2; }
    u32 b = 512;
    while ((u64)a * (b + 1) * (b + 1) < (1ull << 28)) b++;
    return (b + 1) / 2;
}
// The loop above runs up to ~500 times a call: run386 had frsqrte_f32 as the top host function in guest samples
// (3.5 %). Same function, evaluated once per input (128..511) at startup.
static const struct RsqrtTab {
    u16 v[512];
    RsqrtTab() { for (u32 a = 128; a < 512; a++) v[a] = (u16)recip_sqrt_estimate(a); }
} k_rsqrt;

float frecpe_f32(float x) {
    u32 op = fbits(x);
    u32 sign = op & 0x80000000u;
    s32 exp = (op >> 23) & 0xFF;
    u64 frac = (u64)(op & 0x7FFFFF) << 29;      // 52-bit fraction
    if (exp == 0xFF) {
        if (frac) return bitsf(op | 0x00400000u); // NaN -> quiet
        return bitsf(sign);                        // inf -> 0
    }
    if (exp == 0 && frac == 0) return bitsf(sign | 0x7F800000u);   // 0 -> inf
    if (std::fabs(x) < 0x1p-128f) return bitsf(sign | 0x7F800000u); // overflow -> inf (RN)
    if (exp == 0) {
        if (((frac >> 51) & 1) == 0) { exp = -1; frac = (frac & ((1ull << 50) - 1)) << 2; }
        else frac = (frac & ((1ull << 51) - 1)) << 1;
    }
    u32 scaled = 0x100 | (u32)((frac >> 44) & 0xFF);
    s32 rexp = 253 - exp;
    u32 est = recip_estimate(scaled);
    u64 f = (u64)(est & 0xFF) << 44;
    if (rexp == 0) f = (1ull << 51) | (f >> 1);
    else if (rexp == -1) { f = (1ull << 50) | (f >> 2); rexp = 0; }
    return bitsf(sign | ((u32)(rexp & 0xFF) << 23) | (u32)((f >> 29) & 0x7FFFFF));
}

float frsqrte_f32(float x) {
    u32 op = fbits(x);
    u32 sign = op & 0x80000000u;
    s32 exp = (op >> 23) & 0xFF;
    u64 frac = (u64)(op & 0x7FFFFF) << 29;
    if (exp == 0xFF && frac) return bitsf(op | 0x00400000u);        // NaN
    if (exp == 0 && frac == 0) return bitsf(sign | 0x7F800000u);     // +-0 -> +-inf
    if (sign) return bitsf(0x7FC00000u);                              // negative -> default NaN
    if (exp == 0xFF) return 0.0f;                                     // +inf -> +0
    if (exp == 0) {
        while (((frac >> 51) & 1) == 0) { frac = (frac << 1) & ((1ull << 52) - 1); exp--; }
        frac = (frac << 1) & ((1ull << 52) - 1);
    }
    u32 scaled = (exp & 1) == 0 ? (0x100 | (u32)((frac >> 44) & 0xFF)) : (0x80 | (u32)((frac >> 45) & 0x7F));
    s32 num = 380 - exp;
    s32 rexp = num >= 0 ? num / 2 : -((-num + 1) / 2);              // floor division
    u32 est = k_rsqrt.v[scaled];
    return bitsf(((u32)(rexp & 0xFF) << 23) | ((est & 0xFF) << 15));
}

u64 rt_read_fpcr(Ctx* c) { return c->fpcr; }
void rt_write_fpcr(Ctx* c, u64 v) { c->fpcr = (u32)v; }   // TODO(R1): map RMode/FZ to MXCSR if the game relies on them

void rt_mrs_unknown(Ctx* c, u64 pc) { rt_trap(c, pc, "mrs"); }

[[noreturn]] void rt_trap(Ctx* c, u64 pc, const char* why) {
    fprintf(stderr, "[rt] TRAP at %llx: %s (lr=%llx sp=%llx x0=%llx)\n", (unsigned long long)pc, why,
            (unsigned long long)c->x[30], (unsigned long long)c->sp, (unsigned long long)c->x[0]);
    // guest backtrace through the x29 frame chain (frames live on the guest stack, above sp)
    u64 fp = c->x[29];
    for (int i = 0; i < 24 && fp >= c->sp && fp < c->sp + (256ull << 20) && !(fp & 7); i++) {
        u64 nfp = *(u64*)fp, lr = *(u64*)(fp + 8);
        fprintf(stderr, "[rt]   #%-2d lr=%llx\n", i, (unsigned long long)lr);
        if (nfp <= fp) break;
        fp = nfp;
    }
    // x0 is usually `this` of a bad virtual call: dump the object and what it holds as its vtable
#ifdef _WIN32
    auto readable = [](u64 a, u64 n) {
        MEMORY_BASIC_INFORMATION mi;
        return a && VirtualQuery((void*)a, &mi, sizeof mi) && mi.State == MEM_COMMIT &&
               !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) && a + n <= (u64)mi.BaseAddress + mi.RegionSize;
    };
#else
    auto readable = [](u64 a, u64 n) { return a && rt_posix_readable(a, n) != 0; };   // (runtime/linux/plat_posix.cpp)
#endif
    for (u64 a = c->x[0], k = 0; k < 2 && readable(a & ~7ull, 0x80); k++) {
        const u64* w = (const u64*)(a & ~7ull);
        fprintf(stderr, "[rt]   %s %llx:", k ? "vtable" : "x0 obj", (unsigned long long)a);
        for (int i = 0; i < 16; i++) fprintf(stderr, "%s%llx", i % 4 ? " " : "\n[rt]     ", (unsigned long long)w[i]);
        fprintf(stderr, "\n");
        a = w[0];
    }
    fflush(stderr);
    abort();
}

// TotK recomp runtime: guest CPU state and AArch64 instruction semantics used by generated code.
// Guest memory is identity-mapped: a guest address is a host address.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <climits>
#include <limits>
#include <utility>
#include <type_traits>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef __int128 s128; typedef unsigned __int128 u128;

#define RT_INLINE static inline __attribute__((always_inline))

// Compile-time unrolled lane loop: LANES(B, E, stmt using constexpr int i). Generated code never contains real
// loops for lanes: thousands of tiny loops in one huge function make LLVM's per-loop passes (LICM, LoopRotate)
// quadratic (measured 2026-10-14: an 18k-insn function took > 300 s at -O1 / -Og, 6 s at -O0).
template <int B, class F, int... I> RT_INLINE void rt_unroll_(F& f, std::integer_sequence<int, I...>) { (f(std::integral_constant<int, B + I>{}), ...); }
template <int B, int E, class F> RT_INLINE void rt_unroll(F&& f) { if constexpr (E > B) rt_unroll_<B>(f, std::make_integer_sequence<int, E - B>{}); }
#define LANES(B, E, ...) rt_unroll<(B), (E)>([&](auto i_) __attribute__((always_inline)) { constexpr int i = decltype(i_)::value; __VA_ARGS__; })

union V128 {
    u8 b[16]; u16 h[8]; u32 s[4]; u64 d[2];
    s8 sb[16]; s16 sh[8]; s32 ss[4]; s64 sd[2];
    float f[4]; double fd[2];
};

struct Ctx {
    u64 x[32];          // x[31] unused: xzr reads are emitted as 0, sp lives in .sp
    u64 sp;
    V128 v[32];
    u8 n, z, c, vf;     // NZCV
    u32 fpcr, fpsr;
    u64 tpidr;          // TPIDR_EL0
    u64 excl_addr;      // exclusive monitor (~0 = open)
    u64 excl_val;
    u64 excl_val2;
    void* host;         // runtime per-thread data
};

// ---------------------------------------------------------------- memory (volatile: keep guest access order/count)
typedef u16 u16a __attribute__((aligned(1))); typedef u32 u32a __attribute__((aligned(1)));
typedef u64 u64a __attribute__((aligned(1)));
RT_INLINE u8  ld8 (u64 a) { return *(volatile u8*)a; }
RT_INLINE u16 ld16(u64 a) { return *(volatile u16a*)a; }
RT_INLINE u32 ld32(u64 a) { return *(volatile u32a*)a; }
RT_INLINE u64 ld64(u64 a) { return *(volatile u64a*)a; }
RT_INLINE void st8 (u64 a, u8 v)  { *(volatile u8*)a = v; }
RT_INLINE void st16(u64 a, u16 v) { *(volatile u16a*)a = v; }
RT_INLINE void st32(u64 a, u32 v) { *(volatile u32a*)a = v; }
RT_INLINE void st64(u64 a, u64 v) { *(volatile u64a*)a = v; }
RT_INLINE V128 ld128(u64 a) { V128 r; r.d[0] = ld64(a); r.d[1] = ld64(a + 8); return r; }
RT_INLINE void st128(u64 a, const V128& v) { st64(a, v.d[0]); st64(a + 8, v.d[1]); }
RT_INLINE float  ldf32(u64 a) { u32 b = ld32(a); float f; memcpy(&f, &b, 4); return f; }

// ---------------------------------------------------------------- vector register helpers
RT_INLINE V128 vzero() { V128 r; r.d[0] = 0; r.d[1] = 0; return r; }
RT_INLINE V128 vS(float f) { V128 r = vzero(); r.f[0] = f; return r; }
RT_INLINE V128 vD(double d) { V128 r = vzero(); r.fd[0] = d; return r; }
RT_INLINE V128 vU32(u32 x) { V128 r = vzero(); r.s[0] = x; return r; }
RT_INLINE V128 vU64(u64 x) { V128 r = vzero(); r.d[0] = x; return r; }
RT_INLINE V128 vU16(u16 x) { V128 r = vzero(); r.h[0] = x; return r; }
RT_INLINE V128 vU8(u8 x) { V128 r = vzero(); r.b[0] = x; return r; }
RT_INLINE V128 vlo(V128 a) { a.d[1] = 0; return a; }          // 64-bit arrangement result
RT_INLINE V128 v2(u64 lo, u64 hi) { V128 r; r.d[0] = lo; r.d[1] = hi; return r; }
RT_INLINE float  bitsf(u32 x) { float f; memcpy(&f, &x, 4); return f; }
RT_INLINE double bitsd(u64 x) { double f; memcpy(&f, &x, 8); return f; }
RT_INLINE u32 fbits(float f) { u32 x; memcpy(&x, &f, 4); return x; }
RT_INLINE u64 dbits(double f) { u64 x; memcpy(&x, &f, 8); return x; }

// ---------------------------------------------------------------- flags
// (templates: generated code passes its LCtx, see the end of this file; runtime code a Ctx)
template <class C> RT_INLINE void nz32(C* c, u32 r) { c->n = r >> 31; c->z = r == 0; }
template <class C> RT_INLINE void nz64(C* c, u64 r) { c->n = r >> 63; c->z = r == 0; }
template <class C> RT_INLINE u64 adds64(C* c, u64 a, u64 b, u64 cin = 0) {
    u128 w = (u128)a + b + cin; u64 r = (u64)w;
    c->n = r >> 63; c->z = r == 0; c->c = (u8)(w >> 64);
    c->vf = (u8)((~(a ^ b) & (a ^ r)) >> 63); return r;
}
template <class C> RT_INLINE u32 adds32(C* c, u32 a, u32 b, u32 cin = 0) {
    u64 w = (u64)a + b + cin; u32 r = (u32)w;
    c->n = r >> 31; c->z = r == 0; c->c = (u8)(w >> 32);
    c->vf = (u8)((~(a ^ b) & (a ^ r)) >> 31); return r;
}
template <class C> RT_INLINE u64 subs64(C* c, u64 a, u64 b) { return adds64(c, a, ~b, (u64)1); }
template <class C> RT_INLINE u32 subs32(C* c, u32 a, u32 b) { return adds32(c, a, ~b, (u32)1); }
template <class C> RT_INLINE void setnzcv(C* c, u32 f) { c->n = (f >> 3) & 1; c->z = (f >> 2) & 1; c->c = (f >> 1) & 1; c->vf = f & 1; }

#define C_EQ (c->z)
#define C_NE (!c->z)
#define C_HS (c->c)
#define C_LO (!c->c)
#define C_MI (c->n)
#define C_PL (!c->n)
#define C_VS (c->vf)
#define C_VC (!c->vf)
#define C_HI (c->c && !c->z)
#define C_LS (!(c->c && !c->z))
#define C_GE (c->n == c->vf)
#define C_LT (c->n != c->vf)
#define C_GT (!c->z && c->n == c->vf)
#define C_LE (!(!c->z && c->n == c->vf))
#define C_AL (1)

// ---------------------------------------------------------------- integer helpers
RT_INLINE u32 ror32(u32 x, unsigned s) { s &= 31; return s ? (x >> s) | (x << (32 - s)) : x; }
RT_INLINE u64 ror64(u64 x, unsigned s) { s &= 63; return s ? (x >> s) | (x << (64 - s)) : x; }
RT_INLINE u32 asr32(u32 x, unsigned s) { return (u32)((s32)x >> (s & 31)); }
RT_INLINE u64 asr64(u64 x, unsigned s) { return (u64)((s64)x >> (s & 63)); }
RT_INLINE u32 lsl32(u32 x, unsigned s) { return x << (s & 31); }
RT_INLINE u64 lsl64(u64 x, unsigned s) { return x << (s & 63); }
RT_INLINE u32 lsr32(u32 x, unsigned s) { return x >> (s & 31); }
RT_INLINE u64 lsr64(u64 x, unsigned s) { return x >> (s & 63); }
RT_INLINE u32 clz32(u32 x) { return x ? __builtin_clz(x) : 32; }
RT_INLINE u64 clz64(u64 x) { return x ? __builtin_clzll(x) : 64; }
RT_INLINE u64 rbit64(u64 x) {
    x = ((x >> 1) & 0x5555555555555555ull) | ((x & 0x5555555555555555ull) << 1);
    x = ((x >> 2) & 0x3333333333333333ull) | ((x & 0x3333333333333333ull) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((x & 0x0F0F0F0F0F0F0F0Full) << 4);
    return __builtin_bswap64(x);
}
RT_INLINE u32 rbit32(u32 x) { return (u32)(rbit64(x) >> 32); }
RT_INLINE u32 rev16_32(u32 x) { return ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu); }
RT_INLINE s32 sdiv32(s32 a, s32 b) { return b == 0 ? 0 : (a == INT32_MIN && b == -1) ? INT32_MIN : a / b; }
RT_INLINE s64 sdiv64(s64 a, s64 b) { return b == 0 ? 0 : (a == INT64_MIN && b == -1) ? INT64_MIN : a / b; }
RT_INLINE u32 udiv32(u32 a, u32 b) { return b ? a / b : 0; }
RT_INLINE u64 udiv64(u64 a, u64 b) { return b ? a / b : 0; }
RT_INLINE u64 smulh(s64 a, s64 b) { return (u64)(((s128)a * b) >> 64); }
RT_INLINE u64 umulh(u64 a, u64 b) { return (u64)(((u128)a * b) >> 64); }
// bitfield move (UBFM/SBFM/BFM semantics expressed via aliases in the emitter)
RT_INLINE u64 ubfx64(u64 x, unsigned lsb, unsigned w) { return w >= 64 ? x >> lsb : (x >> lsb) & ((1ull << w) - 1); }
RT_INLINE u64 sbfx64(u64 x, unsigned lsb, unsigned w) { return (u64)(((s64)(x << (64 - lsb - w))) >> (64 - w)); }
RT_INLINE u32 ubfx32(u32 x, unsigned lsb, unsigned w) { return w >= 32 ? x >> lsb : (x >> lsb) & ((1u << w) - 1); }
RT_INLINE u32 sbfx32(u32 x, unsigned lsb, unsigned w) { return (u32)(((s32)(x << (32 - lsb - w))) >> (32 - w)); }
RT_INLINE u64 bfi64(u64 d, u64 s, unsigned lsb, unsigned w) { u64 m = (w >= 64 ? ~0ull : ((1ull << w) - 1)) << lsb; return (d & ~m) | ((s << lsb) & m); }
RT_INLINE u32 bfi32(u32 d, u32 s, unsigned lsb, unsigned w) { u32 m = (w >= 32 ? ~0u : ((1u << w) - 1)) << lsb; return (d & ~m) | ((s << lsb) & m); }
RT_INLINE u64 extr64(u64 hi, u64 lo, unsigned s) { return s ? (lo >> s) | (hi << (64 - s)) : lo; }
RT_INLINE u32 extr32(u32 hi, u32 lo, unsigned s) { return s ? (lo >> s) | (hi << (32 - s)) : lo; }

// CRC32 (ISO-HDLC poly 0x04C11DB7 reflected 0xEDB88320) and CRC32C (0x82F63B78)
template <int BITS> RT_INLINE u32 crc_bits_(u32 crc, u64 val, u32 poly) {
    crc ^= (u32)val; u64 v = val >> 32;
    LANES(0, BITS, crc = (crc >> 1) ^ (poly & (0u - (crc & 1))); if (i == 31) crc ^= (u32)v;);
    return crc;
}
RT_INLINE u32 crc_bits(u32 crc, u64 val, int bits, u32 poly) {
    switch (bits) { case 8: return crc_bits_<8>(crc, val, poly); case 16: return crc_bits_<16>(crc, val, poly);
                    case 32: return crc_bits_<32>(crc, val, poly); default: return crc_bits_<64>(crc, val, poly); }
}
RT_INLINE u32 crc32b(u32 a, u32 v) { return crc_bits(a, v & 0xFF, 8, 0xEDB88320u); }
RT_INLINE u32 crc32h(u32 a, u32 v) { return crc_bits(a, v & 0xFFFF, 16, 0xEDB88320u); }
RT_INLINE u32 crc32w(u32 a, u32 v) { return crc_bits(a, v, 32, 0xEDB88320u); }
RT_INLINE u32 crc32x(u32 a, u64 v) { return crc_bits(a, v, 64, 0xEDB88320u); }
RT_INLINE u32 crc32cb(u32 a, u32 v) { return crc_bits(a, v & 0xFF, 8, 0x82F63B78u); }
RT_INLINE u32 crc32ch(u32 a, u32 v) { return crc_bits(a, v & 0xFFFF, 16, 0x82F63B78u); }
RT_INLINE u32 crc32cw(u32 a, u32 v) { return crc_bits(a, v, 32, 0x82F63B78u); }
RT_INLINE u32 crc32cx(u32 a, u64 v) { return crc_bits(a, v, 64, 0x82F63B78u); }

// ---------------------------------------------------------------- exclusives / barriers
template <class C> RT_INLINE u64 ldx64(C* c, u64 a) { u64 v = __atomic_load_n((u64*)a, __ATOMIC_SEQ_CST); c->excl_addr = a; c->excl_val = v; return v; }
template <class C> RT_INLINE u32 ldx32(C* c, u64 a) { u32 v = __atomic_load_n((u32*)a, __ATOMIC_SEQ_CST); c->excl_addr = a; c->excl_val = v; return v; }
template <class C> RT_INLINE u8  ldx8 (C* c, u64 a) { u8 v = __atomic_load_n((u8*)a, __ATOMIC_SEQ_CST); c->excl_addr = a; c->excl_val = v; return v; }
template <class C> RT_INLINE u32 stx64(C* c, u64 a, u64 v) {
    if (c->excl_addr != a) return 1;
    c->excl_addr = ~0ull; u64 e = c->excl_val;
    return __atomic_compare_exchange_n((u64*)a, &e, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
}
template <class C> RT_INLINE u32 stx32(C* c, u64 a, u32 v) {
    if (c->excl_addr != a) return 1;
    c->excl_addr = ~0ull; u32 e = (u32)c->excl_val;
    return __atomic_compare_exchange_n((u32*)a, &e, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
}
template <class C> RT_INLINE u32 stx8(C* c, u64 a, u8 v) {
    if (c->excl_addr != a) return 1;
    c->excl_addr = ~0ull; u8 e = (u8)c->excl_val;
    return __atomic_compare_exchange_n((u8*)a, &e, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
}
template <class C> RT_INLINE void clrex(C* c) { c->excl_addr = ~0ull; }
RT_INLINE u64 lda64(u64 a) { return __atomic_load_n((u64*)a, __ATOMIC_SEQ_CST); }
RT_INLINE u32 lda32(u64 a) { return __atomic_load_n((u32*)a, __ATOMIC_SEQ_CST); }
RT_INLINE u8  lda8 (u64 a) { return __atomic_load_n((u8*)a, __ATOMIC_SEQ_CST); }
RT_INLINE void stl64(u64 a, u64 v) { __atomic_store_n((u64*)a, v, __ATOMIC_SEQ_CST); }
RT_INLINE void stl32(u64 a, u32 v) { __atomic_store_n((u32*)a, v, __ATOMIC_SEQ_CST); }
RT_INLINE void stl8 (u64 a, u8 v)  { __atomic_store_n((u8*)a, v, __ATOMIC_SEQ_CST); }
RT_INLINE void dmb_full() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
RT_INLINE void dmb_light() { __atomic_signal_fence(__ATOMIC_SEQ_CST); }

// ---------------------------------------------------------------- scalar floating point
// All helpers below are BRANCH-FREE (integer masks + selects): unrolled NEON code calls them per lane, and
// branchy NaN checks turned one 18k-insn function into ~100k basic blocks (clang > 300 s). Same ARM semantics,
// checked by tools/recomp/difftest.py against Unicorn.
template <class F> RT_INLINE bool isnanT(F a) { return a != a; }
template <class F> using UB = std::conditional_t<sizeof(F) == 4, u32, u64>;
template <class F> RT_INLINE UB<F> fb(F a) { UB<F> u; memcpy(&u, &a, sizeof u); return u; }
template <class F> RT_INLINE F bf(UB<F> u) { F a; memcpy(&a, &u, sizeof a); return a; }
template <class F> constexpr UB<F> kQB = sizeof(F) == 4 ? (UB<F>)0x00400000u : (UB<F>)0x0008000000000000ull;   // quiet bit
template <class F> constexpr UB<F> kDN = sizeof(F) == 4 ? (UB<F>)0x7FC00000u : (UB<F>)0x7FF8000000000000ull;   // ARM default NaN
template <class F> RT_INLINE UB<F> mk(bool b) { return (UB<F>)0 - (UB<F>)b; }
template <class U> RT_INLINE U sel(U m, U x, U y) { return (x & m) | (y & ~m); }
RT_INLINE float  qnan_f(float a) { return bitsf(fbits(a) | 0x00400000u); }
RT_INLINE double qnan_d(double a) { return bitsd(dbits(a) | 0x0008000000000000ull); }
RT_INLINE float  qn(float a) { return qnan_f(a); }
RT_INLINE double qn(double a) { return qnan_d(a); }
template <class F> RT_INLINE bool issnan(F a) { return isnanT(a) & !(fb(a) & kQB<F>); }
template <class F> RT_INLINE UB<F> mnan(F a) { return mk<F>(isnanT(a)); }
template <class F> RT_INLINE UB<F> msnan(F a) { return mk<F>(isnanT(a) & !(fb(a) & kQB<F>)); }
// ARM FPProcessNaNs (2 operands), bits of the quietened result; only meaningful when a or b is NaN
template <class F> RT_INLINE UB<F> nanprop2(F a, F b) {
    UB<F> t = fb(b);
    t = sel(mnan(a), fb(a), t);       // first QNaN ...
    t = sel(msnan(b), fb(b), t);      // ... but SNaNs win, a before b
    t = sel(msnan(a), fb(a), t);
    return t | kQB<F>;
}
template <class F> RT_INLINE F nanprop(F a, F b) { return bf<F>(nanprop2(a, b)); }
template <class F> RT_INLINE F fmaxA(F a, F b) {
    UB<F> r = sel(mk<F>(a > b), fb(a), fb(b));
    r = sel(mk<F>((a == 0) & (b == 0)), std::signbit(a) ? fb(b) : fb(a), r);
    return bf<F>(sel(mnan(a) | mnan(b), nanprop2(a, b), r));
}
template <class F> RT_INLINE F fminA(F a, F b) {
    UB<F> r = sel(mk<F>(a < b), fb(a), fb(b));
    r = sel(mk<F>((a == 0) & (b == 0)), std::signbit(a) ? fb(a) : fb(b), r);
    return bf<F>(sel(mnan(a) | mnan(b), nanprop2(a, b), r));
}
template <class F> RT_INLINE F fmaxnmA(F a, F b) {   // only quiet NaNs give way to a number
    UB<F> r = fb(fmaxA(a, b));
    r = sel((mnan(b) & ~msnan(b)) & ~mnan(a), fb(a), r);
    r = sel((mnan(a) & ~msnan(a)) & ~mnan(b), fb(b), r);
    return bf<F>(r);
}
template <class F> RT_INLINE F fminnmA(F a, F b) {
    UB<F> r = fb(fminA(a, b));
    r = sel((mnan(b) & ~msnan(b)) & ~mnan(a), fb(a), r);
    r = sel((mnan(a) & ~msnan(a)) & ~mnan(b), fb(b), r);
    return bf<F>(r);
}
template <class F> RT_INLINE F fabdA(F a, F b) { return std::fabs(a - b); }
template <class C, class F> RT_INLINE void fcmpA(C* c, F a, F b) {
    // gt 0010, lt 1000, eq 0110, unordered 0011 = n: lt, z: eq, c: !lt, v: unordered. Plain compares per flag so clang can
    // fold fcmp + b.cond into one ucomis + jcc (the old packed 2 + 6*lt + 4*eq form with masks was not folded; 757 fcmpA in
    // the 8 hottest FP functions, Deck d48, 2026-10-17).
    bool lt = a < b;
    c->n = lt; c->z = a == b; c->c = !lt; c->vf = isnanT(a) | isnanT(b);
}
// float -> int, round toward zero, saturating, NaN -> 0
template <class I, class F> RT_INLINE I fcvtzT(F a) {
    using U = std::make_unsigned_t<I>;
    bool hi = a >= (F)std::numeric_limits<I>::max(), lo = a <= (F)std::numeric_limits<I>::min(), nn = isnanT(a);
    F safe = bf<F>(sel(mk<F>(hi | lo | nn), (UB<F>)0, fb(a)));
    U r = (U)(I)safe;
    r = sel((U)(0 - (U)hi), (U)std::numeric_limits<I>::max(), r);
    r = sel((U)(0 - (U)lo), (U)std::numeric_limits<I>::min(), r);
    return (I)r;
}
template <class F> RT_INLINE F rnd(F a, int mode) {   // 0 nearest-even, 1 +inf, 2 -inf, 3 zero, 4 nearest-away
    return mode == 0 ? std::nearbyint(a) : mode == 1 ? std::ceil(a) : mode == 2 ? std::floor(a) : mode == 3 ? std::trunc(a) : std::round(a);
}
template <class I, class F> RT_INLINE I fcvt_rm(F a, int mode) { return fcvtzT<I, F>(rnd(a, mode)); }   // NaN -> 0 via fcvtzT
template <class F> RT_INLINE F frintA(F a, int mode) { return bf<F>(sel(mnan(a), fb(a) | kQB<F>, fb(rnd(a, mode)))); }
// arithmetic result fix-up: x86 makes a negative default NaN and does not prioritise SNaNs; ARM does.
template <class F> RT_INLINE F fixnan(F r, F a, F b) {
    UB<F> n = sel(mnan(a) | mnan(b), nanprop2(a, b), kDN<F>);
    return bf<F>(sel(mnan(r), n, fb(r)));
}
template <class F> RT_INLINE F fixnan1(F r, F a) {
    UB<F> n = sel(mnan(a), fb(a) | kQB<F>, kDN<F>);
    return bf<F>(sel(mnan(r), n, fb(r)));
}
// (the "A" suffix: glibc's <math.h> declares the C23 narrowing float fadd/fsub/fmul/fdiv(double, double); a non-template
// wins the overload tie, so on Linux every DOUBLE add/sub/mul/div called libm and came back rounded to float, without the
// NaN fix-up (Deck perf d45: f32mulf32x / f32addf32x in the worker threads, 2026-10-17). Windows' UCRT has no such names.
// The macros keep code generated before the rename compiling; gen.py emits the A names.)
template <class F> RT_INLINE F faddA(F a, F b) { return fixnan(a + b, a, b); }
template <class F> RT_INLINE F fsubA(F a, F b) { return fixnan(a - b, a, b); }
template <class F> RT_INLINE F fmulA(F a, F b) { return fixnan(a * b, a, b); }
template <class F> RT_INLINE F fdivA(F a, F b) { return fixnan(a / b, a, b); }
#define fadd faddA
#define fsub fsubA
#define fmul fmulA
#define fdiv fdivA
template <class F> RT_INLINE F fnmul(F a, F b) { return -fmulA(a, b); }
template <class F> RT_INLINE F fsqrtA(F a) { return fixnan1(std::sqrt(a), a); }
template <class F> RT_INLINE F fmlaA(F acc, F a, F b) {   // FPProcessNaNs3: SNaN acc,a,b then QNaN acc,a,b
    F r = std::fma(a, b, acc);
    UB<F> t = fb(b);
    t = sel(mnan(a), fb(a), t);
    t = sel(mnan(acc), fb(acc), t);
    t = sel(msnan(b), fb(b), t);
    t = sel(msnan(a), fb(a), t);
    t = sel(msnan(acc), fb(acc), t);
    UB<F> n = sel(mnan(acc) | mnan(a) | mnan(b), t | kQB<F>, kDN<F>);
    // FPMulAdd: a QUIET NaN addend with an inf * 0 product gives the default NaN (invalid op), not the addend (difftest
    // seed 11, fmls v.4s by element, 2026-10-17)
    bool i0 = (std::isinf(a) & (b == 0)) | (std::isinf(b) & (a == 0));
    n = sel(mk<F>(isnanT(acc) & !issnan(acc) & i0), kDN<F>, n);
    return bf<F>(sel(mnan(r), n, fb(r)));
}
// Fast forms (gen.py: scalar ops here; vector ops test all lanes at once, gen.py lanes_fp): the native result, and the
// exact ARM NaN handling (the A forms) out of line only when the result is NaN. A NaN operand always gives a NaN result
// for add/sub/mul/div/fma, so testing the result covers the operands too: same values as the A forms. The A forms are ~19
// x86 insns per scalar op (branch-free masks; x4 per vector op) = over half of the host code of FP-heavy hot functions
// (Deck d47, 2026-10-17). One test per op (not per lane) keeps the basic-block count sane (see the comment above).
template <class F> __attribute__((noinline, cold)) F fixnan_cold(F r, F a, F b) { return fixnan(r, a, b); }
template <class F> __attribute__((noinline, cold)) F fmla_cold(F acc, F a, F b) { return fmlaA(acc, a, b); }
template <class F> RT_INLINE F faddS(F a, F b) { F r = a + b; if (__builtin_expect(r != r, 0)) r = fixnan_cold(r, a, b); return r; }
template <class F> RT_INLINE F fsubS(F a, F b) { F r = a - b; if (__builtin_expect(r != r, 0)) r = fixnan_cold(r, a, b); return r; }
template <class F> RT_INLINE F fmulS(F a, F b) { F r = a * b; if (__builtin_expect(r != r, 0)) r = fixnan_cold(r, a, b); return r; }
template <class F> RT_INLINE F fdivS(F a, F b) { F r = a / b; if (__builtin_expect(r != r, 0)) r = fixnan_cold(r, a, b); return r; }
template <class F> RT_INLINE F fnmulS(F a, F b) { return -fmulS(a, b); }
template <class F> RT_INLINE F fmlaS(F acc, F a, F b) {
    F r = std::fma(a, b, acc);
    if (__builtin_expect(r != r, 0)) r = fmla_cold(acc, a, b);
    return r;
}
// Vector add/sub/mul/div/fmla/fmls (gen.py _fbin vop, 2026-10-17): ONE SIMD op + ONE NaN test over the live lanes; the exact
// per-lane A forms only out of line (vfop_cold). The earlier lanes_fp form (per-lane scalar ops and tests, exact path inlined
// at every site) made f_170744 21 host insns per guest insn. Same values: a lane is NaN in the native result iff the A form
// would do NaN handling there. OP: 0 add, 1 sub, 2 mul, 3 div, 4 fmla (acc + a*b, fused), 5 fmls (acc - a*b, fused).
// Lanes >= CNT come out zero (64-bit arrangements), as before.
typedef float vf32x4 __attribute__((vector_size(16)));
typedef double vf64x2 __attribute__((vector_size(16)));
typedef s32 vm32x4 __attribute__((vector_size(16)));
typedef s64 vm64x2 __attribute__((vector_size(16)));
template <class F> RT_INLINE F* vlanes(V128& v) { if constexpr (sizeof(F) == 4) return v.f; else return v.fd; }
template <class F> RT_INLINE V128 vbcast(F s) {   // (vector literal -> vbroadcastss / vmovddup)
    V128 r;
    if constexpr (sizeof(F) == 4) { vf32x4 v = {s, s, s, s}; memcpy(&r, &v, 16); }
    else { vf64x2 v = {s, s}; memcpy(&r, &v, 16); }
    return r;
}
template <int OP, int CNT, class F> __attribute__((noinline, cold)) V128 vfop_cold(V128 a, V128 b, V128 acc) {
    V128 r = vzero();
    F *x = vlanes<F>(a), *y = vlanes<F>(b), *z = vlanes<F>(acc), *o = vlanes<F>(r);
    for (int i = 0; i < CNT; i++)
        o[i] = OP == 0 ? faddA(x[i], y[i]) : OP == 1 ? fsubA(x[i], y[i]) : OP == 2 ? fmulA(x[i], y[i]) : OP == 3 ? fdivA(x[i], y[i])
             : OP == 4 ? fmlaA(z[i], x[i], y[i]) : fmlaA(z[i], -x[i], y[i]);
    return r;
}
template <int OP, int CNT, class F> RT_INLINE V128 vfop(const V128& a, const V128& b, const V128& acc) {
    static_assert(CNT * sizeof(F) == 8 || CNT * sizeof(F) == 16, "vfop arrangement");
    using V = std::conditional_t<sizeof(F) == 4, vf32x4, vf64x2>;
    using M = std::conditional_t<sizeof(F) == 4, vm32x4, vm64x2>;
    V x, y, z, r;
    memcpy(&x, &a, 16); memcpy(&y, &b, 16);
    if constexpr (OP == 0) r = x + y;
    else if constexpr (OP == 1) r = x - y;
    else if constexpr (OP == 2) r = x * y;
    else if constexpr (OP == 3) r = x / y;
    else { memcpy(&z, &acc, 16); r = __builtin_elementwise_fma(OP == 4 ? x : -x, y, z); }
    M m = r != r;
    bool nan;
    if constexpr (CNT * sizeof(F) == 16) nan = __builtin_reduce_or(m) != 0;
    else nan = (m[0] | m[1]) != 0;                                  // 2S: lanes 0..1 only
    V128 o; memcpy(&o, &r, 16);
    if constexpr (CNT * sizeof(F) == 8) o.d[1] = 0;
    if (__builtin_expect(nan, 0)) o = vfop_cold<OP, CNT, F>(a, b, acc);
    return o;
}
float frecpe_f32(float a);
float frsqrte_f32(float a);
RT_INLINE bool inf_times_zero(float a, float b) { return (std::isinf(a) & (b == 0)) | (std::isinf(b) & (a == 0)); }
RT_INLINE float frecps_f32(float a, float b) {   // 2 - a*b, with inf*0 = 2; ARM negates op1 before NaN processing
    float r = fixnan(std::fma(-a, b, 2.0f), -a, b);
    return bf<float>(sel(mk<float>(inf_times_zero(a, b)), fbits(2.0f), fbits(r)));
}
RT_INLINE float frsqrts_f32(float a, float b) {  // (3 - a*b)/2, with inf*0 = 1.5; ARM halves before rounding
    float r = fixnan((float)(std::fma(-(double)a, (double)b, 3.0) * 0.5), -a, b);
    return bf<float>(sel(mk<float>(inf_times_zero(a, b)), fbits(1.5f), fbits(r)));
}
RT_INLINE float  f16tof32(u16 h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
RT_INLINE u16    f32tof16(float f) { _Float16 x = (_Float16)f; u16 h; memcpy(&h, &x, 2); return h; }

// ---------------------------------------------------------------- vector lane macros
// VMAP(T, N, expr using a[i], b[i]) -> V128 with N lanes of T computed; upper lanes zero when N*sizeof(T)==8
#define VLANES(T) (16 / sizeof(T))
template <class T> RT_INLINE T* lanes(V128& v) { return (T*)&v; }
template <class T> RT_INLINE const T* lanes(const V128& v) { return (const T*)&v; }
#define VBIN(T, N, A, B, EXPR) ([&]() { V128 r_ = vzero(); const T* a = lanes<T>(A); const T* b = lanes<T>(B); T* r = lanes<T>(r_); \
    LANES(0, (N), r[i] = (T)(EXPR)); return r_; }())
#define VUN(T, N, A, EXPR) ([&]() { V128 r_ = vzero(); const T* a = lanes<T>(A); T* r = lanes<T>(r_); \
    LANES(0, (N), r[i] = (T)(EXPR)); return r_; }())
#define VTER(T, N, D, A, B, EXPR) ([&]() { V128 r_ = vzero(); const T* d = lanes<T>(D); const T* a = lanes<T>(A); const T* b = lanes<T>(B); T* r = lanes<T>(r_); \
    LANES(0, (N), r[i] = (T)(EXPR)); return r_; }())

RT_INLINE V128 vins(V128 d, int bytes, int idx, u64 val) { memcpy(d.b + idx * bytes, &val, bytes); return d; }
RT_INLINE u64 vext_lane(const V128& s, int bytes, int idx) { u64 v = 0; memcpy(&v, s.b + idx * bytes, bytes); return v; }
RT_INLINE V128 vdup(int bytes, int nbytes_total, u64 val) {
    u64 m = bytes >= 8 ? ~0ull : (1ull << (bytes * 8)) - 1, rep = bytes == 1 ? 0x0101010101010101ull : bytes == 2 ? 0x0001000100010001ull : bytes == 4 ? 0x0000000100000001ull : 1;
    V128 r = vzero(); r.d[0] = (val & m) * rep; if (nbytes_total == 16) r.d[1] = r.d[0]; return r;
}
RT_INLINE V128 vext(const V128& a, const V128& b, int idx, int total) {   // EXT
    u8 cat[32]; memcpy(cat, a.b, total); memcpy(cat + total, b.b, total);
    V128 r = vzero(); memcpy(r.b, cat + idx, total); return r;
}
RT_INLINE V128 vtbl(const V128* tab, int ntab, const V128& idx, int total) {
    V128 r = vzero();
    auto one = [&](int i) __attribute__((always_inline)) { unsigned j = idx.b[i]; r.b[i] = j < 16u * ntab ? tab[j / 16].b[j % 16] : 0; };
    if (total == 16) LANES(0, 16, one(i)); else LANES(0, 8, one(i));
    return r;
}
template <class T> RT_INLINE V128 vzip(const V128& a, const V128& b, int part, int total) {
    V128 r = vzero(); int n = total / sizeof(T); const T* x = lanes<T>(a); const T* y = lanes<T>(b); T* o = lanes<T>(r);
    int base = part * n / 2;
    if (total == 16) LANES(0, 8 / (int)sizeof(T), { o[2 * i] = x[base + i]; o[2 * i + 1] = y[base + i]; }); else LANES(0, 4 / (int)sizeof(T), { o[2 * i] = x[base + i]; o[2 * i + 1] = y[base + i]; });
    return r;
}
template <class T> RT_INLINE V128 vuzp(const V128& a, const V128& b, int part, int total) {
    V128 r = vzero(); int n = total / sizeof(T); const T* x = lanes<T>(a); const T* y = lanes<T>(b); T* o = lanes<T>(r);
    if (total == 16) LANES(0, 8 / (int)sizeof(T), { o[i] = x[2 * i + part]; o[n / 2 + i] = y[2 * i + part]; }); else LANES(0, 4 / (int)sizeof(T), { o[i] = x[2 * i + part]; o[n / 2 + i] = y[2 * i + part]; });
    return r;
}
template <class T> RT_INLINE V128 vtrn(const V128& a, const V128& b, int part, int total) {
    V128 r = vzero(); int n = total / sizeof(T); const T* x = lanes<T>(a); const T* y = lanes<T>(b); T* o = lanes<T>(r);
    if (total == 16) LANES(0, 8 / (int)sizeof(T), { o[2 * i] = x[2 * i + part]; o[2 * i + 1] = y[2 * i + part]; }); else LANES(0, 4 / (int)sizeof(T), { o[2 * i] = x[2 * i + part]; o[2 * i + 1] = y[2 * i + part]; });
    return r;
}
template <class T> RT_INLINE V128 vrev64(const V128& a, int total) {
    V128 r = vzero(); int per = 8 / sizeof(T); const T* x = lanes<T>(a); T* o = lanes<T>(r);
    if (total == 16) LANES(0, 16 / (int)sizeof(T), o[i] = x[(i / per) * per + (per - 1 - i % per)]); else LANES(0, 8 / (int)sizeof(T), o[i] = x[(i / per) * per + (per - 1 - i % per)]);
    return r;
}
template <class T> RT_INLINE T satT(s64 v) {
    if (v > (s64)std::numeric_limits<T>::max()) return std::numeric_limits<T>::max();
    if (v < (s64)std::numeric_limits<T>::min()) return std::numeric_limits<T>::min();
    return (T)v;
}

// ---------------------------------------------------------------- control flow (implemented by the runtime)
typedef void (*GuestFn)(Ctx*);
void rt_call_indirect(Ctx* c, u64 target);           // BLR / tail BR to unknown target
extern GuestFn* g_ftab;                              // indexed by (addr - FTAB_BASE) >> 2
extern u64 g_ftab_n;
#define FTAB_BASE 0x7100000000ull
RT_INLINE void call_ind(Ctx* c, u64 t) {
    u64 i = (t - FTAB_BASE) >> 2; GuestFn f;
    if (i < g_ftab_n && (f = g_ftab[i]) != nullptr) f(c); else rt_call_indirect(c, t);
}
[[noreturn]] void rt_trap(Ctx* c, u64 pc, const char* why);
void rt_mrs_unknown(Ctx* c, u64 pc);
u64  rt_read_fpcr(Ctx* c);
void rt_write_fpcr(Ctx* c, u64 v);

// ---------------------------------------------------------------- generated-function register file
// gen.py functions run on an LCtx: x0-x30, sp and NZCV are members of a local whose address never leaves the function
// (after inlining, SROA keeps them in host registers / spill slots). Measured on run386's build: with them in *Ctx, clang
// reloaded c->sp after every guest store and stored NZCV on every flag-setting instruction, because guest pointers are
// made from integers and may alias the escaped Ctx. Everything else refers to the real Ctx. gen.py writes the registers
// the function changes back before every call / return / trap (lso, mask = bits 0-30 x0-x30, 31 sp, 32 NZCV) and reloads
// the caller-saved ones after each call returns (lsi: x0-x18, x30, NZCV; x19-x29 / sp are preserved by the callee per
// AAPCS64, see lsi): the real Ctx stays exact wherever other code can look at it.
struct LCtx {
    u64 x[32];
    u64 sp;
    u8 n, z, c, vf;
    V128 (&v)[32];
    u32 &fpcr, &fpsr;
    u64 &tpidr, &excl_addr, &excl_val, &excl_val2;
    __attribute__((always_inline)) explicit LCtx(Ctx* r)
        : v(r->v), fpcr(r->fpcr), fpsr(r->fpsr), tpidr(r->tpidr), excl_addr(r->excl_addr), excl_val(r->excl_val),
          excl_val2(r->excl_val2) {
        LANES(0, 31, x[i] = r->x[i]);
        sp = r->sp; n = r->n; z = r->z; c = r->c; vf = r->vf;
    }
};
template <u64 M> RT_INLINE void lso(Ctx* __restrict r, const LCtx& l) {
    LANES(0, 31, if constexpr ((M >> i) & 1) r->x[i] = l.x[i]);
    if constexpr ((M >> 31) & 1) r->sp = l.sp;
    if constexpr ((M >> 32) & 1) { r->n = l.n; r->z = l.z; r->c = l.c; r->vf = l.vf; }
}
// After a guest call (gen.py emits lsi only after BL / BL to an import / BLR): the callee preserved x19..x29 and sp
// (AAPCS64), so only x0..x18, x30 and the flags come back from the Ctx; the locals keep x19..x29 / sp. tools/abi_scan.py
// (2026-10-17): no function in main (242,860 FDE ranges) or sdk writes x19..x29 without saving them (no outlined code);
// fibers trap; no HLE writes them. Up to ~12 fewer loads per call site and the values stay in host registers.
// LARECOMP_FULL_LSI-style A/B needs a rebuild: define RT_FULL_LSI for the old full reload.
RT_INLINE void lsi(Ctx* __restrict r, LCtx& l) {
#ifdef RT_FULL_LSI
    LANES(0, 31, l.x[i] = r->x[i]);
    l.sp = r->sp;
#else
    LANES(0, 19, l.x[i] = r->x[i]);
    l.x[30] = r->x[30];
#endif
    l.n = r->n; l.z = r->z; l.c = r->c; l.vf = r->vf;
}
// (padding to the previous length: in-place restore while a stale compile kept this header mapped, 2026-10-16)
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

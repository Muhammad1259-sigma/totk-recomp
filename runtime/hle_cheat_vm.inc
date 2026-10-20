// ---- Cheats (included at the end of hle_hid.cpp; launcher 2026-10-20). TOTK_CHEATS = an Atmosphere-format cheat file holding
// only the entries the player enabled (the launcher writes it). This is the dmnt cheat VM (research/sweep/w31_atmosphere_cheat_vm.md,
// read from Atmosphere master): a thread runs the program 12 times a second, registers reset each run, static registers persist,
// 0x8 tests HELD keys, 0xC4 has its own press edge per run. Guest memory is identity-mapped: MainNso = 0x7100000000 (FTAB_BASE),
// heap = g_rt_heap_base. Deviations: alias / aslr region accesses are skipped, pause / resume / debug-log are no-ops, and a run
// stops after 1M instructions (Atmosphere hangs on a loop count of 0). Writes into the main module's CODE change nothing here
// (the game is recompiled); data writes work.
#ifndef _WIN32
#include <sys/uio.h>
#endif
#include <cctype>
extern "C" u64 g_rt_heap_base;
namespace cheatvm {
static bool mem(u64 a, void* b, size_t n, bool wr) {   // a failed access is silent, as in dmnt
    if (!a || !n) return false;
#ifdef _WIN32
    SIZE_T got = 0;
    return wr ? WriteProcessMemory(GetCurrentProcess(), (void*)a, b, n, &got) && got == n
              : ReadProcessMemory(GetCurrentProcess(), (const void*)a, b, n, &got) && got == n;
#else
    struct iovec l = {b, n}, r = {(void*)a, n};
    return (wr ? process_vm_writev(getpid(), &l, 1, &r, 1, 0) : process_vm_readv(getpid(), &l, 1, &r, 1, 0)) == (ssize_t)n;
#endif
}
static std::vector<u32> g_prog;
static u64 g_static[0x100];
static int g_warned;
static void warn(const char* m) { if (g_warned++ < 8) printf("[cheat] %s\n", m); }
static bool region(u32 m, u64 rel, u64* out) {
    switch (m) {
        case 1: if (!g_rt_heap_base) return false; *out = g_rt_heap_base + rel; return true;
        case 2: case 3: warn("alias / aslr region access skipped (not supported)"); return false;
        case 4: *out = rel; return true;
        default: *out = 0x7100000000ull + rel; return true;   // 0 MainNso (and 5-15, like dmnt)
    }
}
static u64 trunc(u64 v, u32 t) { return t == 1 ? (u8)v : t == 2 ? (u16)v : t == 4 ? (u32)v : v; }
static bool width_ok(u32 t) { return t == 1 || t == 2 || t == 4 || t == 8; }
static bool cmp(u32 c, u64 a, u64 b) {
    switch (c) { case 1: return a > b; case 2: return a >= b; case 3: return a < b; case 4: return a <= b; case 5: return a == b; case 6: return a != b; }
    return false;
}
struct Run {
    const std::vector<u32>& p; size_t ip = 0; bool ok = true;
    u64 reg[16] = {}, saved[16] = {}; size_t loop_top[16] = {}; int depth = 0; u64 held, keyold;
    Run(const std::vector<u32>& p_, u64 h, u64 o) : p(p_), held(h), keyold(o) {}
    u32 next() { if (ip >= p.size()) { ok = false; return 0; } return p[ip++]; }
    u64 vmint(u32 t) {   // GetNextVmInt: 8 = two words, high first
        if (t == 8) { u64 hi = next(); return hi << 32 | next(); }
        if (t == 1 || t == 2 || t == 4) return trunc(next(), t);
        return 0;
    }
    u64 rd(u64 a, u32 t) { u64 v = 0; if (width_ok(t)) mem(a, &v, t, false); return v; }
    void wr(u64 a, u64 v, u32 t) { if (width_ok(t)) mem(a, &v, t, true); }
    // words of the instruction at ip, and whether it opens a conditional block; ip is left after it
    struct Skim { bool begin, end, els; };
    Skim skim() {
        u32 w = next(); if (!ok) return {};
        u32 t = w >> 28;
        if (t >= 0xC) t = t << 4 | (w >> 24 & 0xF);
        if (t >= 0xF0) t = t << 4 | (w >> 20 & 0xF);
        u32 T = w >> 24 & 0xF;
        switch (t) {
            case 0x0: next(); vmint(T); return {};
            case 0x1: next(); vmint(T); return {true, false, false};
            case 0x2: return {false, true, (w >> 24 & 0xF) == 1};
            case 0x3: if (!(w >> 24 & 0xF)) next(); return {};
            case 0x4: case 0x6: next(); next(); return {};
            case 0x5: case 0x7: next(); return {};
            case 0x8: return {true, false, false};
            case 0x9: if (w >> 8 & 0xF) vmint(T); return {};
            case 0xA: { u32 O = w >> 8 & 0xF; if (O == 2 || O == 4 || O == 5) next(); return {}; }
            case 0xC0: { u32 X = w >> 8 & 0xF, Tc = w >> 20 & 0xF; if (X == 0 || X == 2) next(); else if (X == 4) vmint(Tc); return {true, false, false}; }
            case 0xC1: case 0xC2: case 0xC3: case 0xFF0: case 0xFF1: return {};
            case 0xC4: next(); next(); return {true, false, false};
            case 0xFFF: { u32 X = w >> 8 & 0xF; if (X == 0 || X == 2) next(); return {}; }
        }
        ok = false; return {};
    }
    void skip_false() {   // condition not met: to the matching else (enter it) or end
        for (int nest = 0; ok && ip < p.size();) {
            Skim s = skim();
            if (s.begin) nest++;
            else if (s.end) {
                if (nest == 0) { if (s.els) depth++; return; }
                if (!s.els) nest--;
            }
        }
    }
    void skip_else() {    // end of a taken branch: to the matching end
        for (int nest = 0; ok && ip < p.size();) {
            Skim s = skim();
            if (s.begin) nest++;
            else if (s.end && !s.els) { if (nest == 0) { depth--; return; } nest--; }
        }
    }
    void cond(bool met) { if (met) depth++; else skip_false(); }
    void exec() {
        for (long budget = 1000000; ok && ip < p.size(); budget--) {
            if (budget <= 0) { warn("run stopped after 1M instructions (loop count 0?)"); return; }
            u32 w = next();
            u32 t = w >> 28;
            if (t >= 0xC) t = t << 4 | (w >> 24 & 0xF);
            if (t >= 0xF0) t = t << 4 | (w >> 20 & 0xF);
            u32 T = w >> 24 & 0xF, M = w >> 20 & 0xF, R = w >> 16 & 0xF;
            switch (t) {
                case 0x0: { u64 rel = (u64)(w & 0xFF) << 32 | next(); u64 v = vmint(T), a;
                            if (region(M, rel + reg[R], &a)) wr(a, v, T); break; }
                case 0x1: { u32 C = w >> 16 & 0xF, X = w >> 12 & 0xF, r = w >> 8 & 0xF;
                            u64 rel = (u64)(w & 0xFF) << 32 | next(); u64 v = vmint(T), a, m = 0;
                            if (region(M, X ? reg[r] + rel : rel, &a)) m = rd(a, T);
                            cond(cmp(C, m, v)); break; }
                case 0x2: if ((w >> 24 & 0xF) == 1) {
                              if (depth == 0) { warn("else outside a conditional (dmnt aborts): run stopped"); return; }
                              skip_else();
                          } else if (depth > 0) depth--;
                          break;
                case 0x3: if (!(w >> 24 & 0xF)) { reg[R] = next(); loop_top[R] = ip; }
                          else if (--reg[R] != 0) ip = loop_top[R];
                          break;
                case 0x4: { u64 hi = next(); reg[R] = hi << 32 | next(); break; }
                case 0x5: { u32 I = w >> 12 & 0xF, s = w >> 8 & 0xF; u64 rel = (u64)(w & 0xFF) << 32 | next(), a; bool go = true;
                            if (I == 1) a = reg[R] + rel; else if (I == 2) a = reg[s] + rel;
                            else go = region(M, I == 3 ? reg[s] + rel : rel, &a);
                            if (go && width_ok(T)) mem(a, &reg[R], T, false);   // (low T bytes only; little-endian like the guest)
                            break; }
                case 0x6: { u32 I = w >> 12 & 0xF, o = w >> 8 & 0xF, r = w >> 4 & 0xF; u64 hi = next(), v = hi << 32 | next();
                            wr(reg[R] + (o ? reg[r] : 0), v, T); if (I) reg[R] += T; break; }
                case 0x7: { u32 C = w >> 12 & 0xF; u64 v = next(), x = reg[R];
                            if (C == 0) x += v; else if (C == 1) x -= v; else if (C == 2) x *= v; else if (C == 3) x <<= (v & 63); else if (C == 4) x >>= (v & 63);
                            reg[R] = width_ok(T) ? trunc(x, T) : x; break; }
                case 0x8: { u64 k = w & 0x0FFFFFFF; cond((held & k) == k); break; }
                case 0x9: { u32 C = w >> 20 & 0xF, S = w >> 12 & 0xF, I = w >> 8 & 0xF, s = w >> 4 & 0xF;
                            u64 a = reg[S], b = I ? vmint(T) : reg[s], x = 0;
                            auto fop = [&](auto fa, auto fb) -> u64 {
                                using F = decltype(fa);
                                F r = C == 10 ? fa + fb : C == 11 ? fa - fb : C == 12 ? fa * fb : fa / fb;
                                if constexpr (sizeof(F) == 4) { u32 u; memcpy(&u, &r, 4); return u; } else { u64 u; memcpy(&u, &r, 8); return u; }
                            };
                            switch (C) {
                                case 0: x = a + b; break; case 1: x = a - b; break; case 2: x = a * b; break;
                                case 3: x = a << (b & 63); break; case 4: x = a >> (b & 63); break; case 5: x = a & b; break;
                                case 6: x = a | b; break; case 7: x = ~a; break; case 8: x = a ^ b; break; case 9: x = a; break;
                                case 10: case 11: case 12: case 13:
                                    if (T == 4) { float fa, fb; u32 ua = (u32)a, ub = (u32)b; memcpy(&fa, &ua, 4); memcpy(&fb, &ub, 4); x = fop(fa, fb); }
                                    else if (T == 8) { double fa, fb; memcpy(&fa, &a, 8); memcpy(&fb, &b, 8); x = fop(fa, fb); }
                                    break;
                            }
                            reg[R] = width_ok(T) ? trunc(x, T) : x; break; }
                case 0xA: { u32 S = w >> 20 & 0xF, I = w >> 12 & 0xF, O = w >> 8 & 0xF, x = w >> 4 & 0xF; u64 rel = 0, a = 0; bool go = true;
                            if (O == 2 || O == 4 || O == 5) rel = (u64)(w & 0xF) << 32 | next();
                            if (O == 1) a = reg[R] + reg[x]; else if (O == 2) a = reg[R] + rel;
                            else if (O == 3) go = region(x, reg[R], &a); else if (O == 4) go = region(x, rel, &a);
                            else if (O == 5) go = region(x, reg[R] + rel, &a); else a = reg[R];
                            if (go) wr(a, reg[S], T);
                            if (I) reg[R] += T;
                            break; }
                case 0xC0: { u32 Tc = w >> 20 & 0xF, c = w >> 16 & 0xF, S = w >> 12 & 0xF, X = w >> 8 & 0xF, hi = w >> 4 & 0xF, lo = w & 0xF;
                             u64 src = trunc(reg[S], Tc), v = 0, a;
                             if (X == 0) { u64 rel = (u64)lo << 32 | next(); if (region(hi, rel, &a)) v = rd(a, Tc); }
                             else if (X == 1) { if (region(hi, reg[lo], &a)) v = rd(a, Tc); }
                             else if (X == 2) { u64 rel = (u64)lo << 32 | next(); v = rd(reg[hi] + rel, Tc); }
                             else if (X == 3) v = rd(reg[hi] + reg[lo], Tc);
                             else if (X == 4) v = vmint(Tc);
                             else if (X == 5) v = trunc(reg[hi], Tc);
                             cond(cmp(c, src, v)); break; }
                case 0xC1: { u32 D = w >> 16 & 0xF, S = w >> 8 & 0xF, x = w >> 4 & 0xF;
                             if (x == 1) saved[D] = reg[S]; else if (x == 2) saved[D] = 0; else if (x == 3) reg[D] = 0; else reg[D] = saved[S];
                             break; }
                case 0xC2: { u32 x = w >> 20 & 0xF;
                             for (int i = 0; i < 16; i++) if (w >> i & 1) {
                                 if (x == 1) saved[i] = reg[i]; else if (x == 2) saved[i] = 0; else if (x == 3) reg[i] = 0; else reg[i] = saved[i];
                             }
                             break; }
                case 0xC3: { u32 XX = w >> 4 & 0xFF, x = w & 0xF; if (XX < 0x80) reg[x] = g_static[XX]; else g_static[XX] = reg[x]; break; }
                case 0xC4: { u32 r = w >> 20 & 0xF; u64 hi = next(), k = hi << 32 | next();
                             bool now = (held & k) == k; cond(r ? now : now && (keyold & k) != k); break; }
                case 0xFF0: case 0xFF1: warn("pause / resume opcode ignored"); break;
                case 0xFFF: { u32 X = w >> 8 & 0xF; if (X == 0 || X == 2) next(); break; }
                default: ok = false; warn("invalid opcode: rest of the program skipped (as in dmnt)"); break;
            }
        }
    }
};
// Atmosphere grammar: [name] / {master} headers, 8-hex-digit words, whitespace; anything else ends the parse (entries so far stay)
static bool load(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { printf("[cheat] can't open %s\n", path); return false; }
    std::string s; char b[4096]; size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0 && s.size() < (64 << 10)) s.append(b, n);
    fclose(f);
    std::vector<u32> master, rest; std::vector<u32>* cur = nullptr; int entries = 0;
    for (size_t i = 0; i < s.size();) {
        char ch = s[i];
        if (isspace((unsigned char)ch)) { i++; continue; }
        if (ch == '[' || ch == '{') {
            size_t e = s.find(ch == '[' ? ']' : '}', i); if (e == std::string::npos) break;
            cur = ch == '{' ? &master : &rest; entries++;
            printf("[cheat] %s %s\n", ch == '{' ? "master" : "on:", s.substr(i + 1, e - i - 1).c_str());
            i = e + 1; continue;
        }
        size_t j = i; while (j < s.size() && isxdigit((unsigned char)s[j])) j++;
        if (j - i != 8 || !cur) { printf("[cheat] parse stopped at offset %zu\n", i); break; }
        cur->push_back((u32)strtoul(s.substr(i, 8).c_str(), nullptr, 16));
        i = j;
    }
    g_prog = master; g_prog.insert(g_prog.end(), rest.begin(), rest.end());
    if (g_prog.size() > 0x400) { printf("[cheat] %zu words > 1024 (dmnt runs nothing)\n", g_prog.size()); g_prog.clear(); }
    printf("[cheat] %d entries, %zu words from %s\n", entries, g_prog.size(), path);
    return !g_prog.empty();
}
static struct Start {
    Start() {
        const char* p = getenv("TOTK_CHEATS");
        if (!p || !*p || !load(p)) return;
        std::thread([] {
            u64 keyold = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::microseconds(1000000 / 12));   // dmnt TimesPerSecond = 12
                u64 held = g_cheat_keys.load();
                Run r(g_prog, held, keyold);
                r.exec();
                keyold = held;
            }
        }).detach();
    }
} g_start;
}   // namespace cheatvm

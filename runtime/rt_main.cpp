// Loader + boot: maps main (and sdk for its data exports) identity-mapped, relocates, builds the function
// table, routes every SDK import through a tracing thunk (HLE if registered, else logged stub returning 0),
// then runs DT_INIT -> nninitStartup -> init_array -> nnMain on a guest thread.
#include "rt.h"
#include "hle.h"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <map>
#include <atomic>
#include <mutex>
#include <utility>
#include <array>
#include <algorithm>
#include <cstdarg>
#include <thread>
#ifdef _WIN32
#include <tlhelp32.h>
#else
#include <sys/resource.h>
#include <ucontext.h>
#include <unistd.h>
#include "linux/plat_posix.h"   // signals, exe location (runtime/linux/plat_posix.cpp)
#endif

struct FtabEnt { u64 off; GuestFn fn; };
extern const FtabEnt g_ftab_init[];
extern const u64 g_ftab_count;
extern const char* const g_imp_names[];
extern const u64 g_imp_count;

// lifted sdk (gen/sdk, gen.py --prefix s --plt-all): runs only where bound (lift whitelist / sdk-internal calls)
extern const FtabEnt g_s_ftab_init[];
extern const u64 g_s_ftab_count;
extern const char* const g_s_imp_names[];
extern const u64 g_s_imp_count;
constexpr u64 kMaxSdkImports = 16384;
GuestFn g_s_imp[kMaxSdkImports];
static GuestFn* g_sftab; static u64 g_sftab_n;         // indexed by (addr - SDK_BASE) >> 2

GuestFn* g_ftab; u64 g_ftab_n;
GuestFn g_imp[kMaxImports];

static const u64 MAIN_BASE = 0x7100000000ull;
static const u64 SDK_BASE  = 0x7180000000ull;          // sdk image: data exports + lifted code (gen/sdk)
static const u64 IMP_FAKE  = MAIN_BASE + 0x5000000ull; // fake code address of import i = IMP_FAKE + 4*i

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr); fflush(stderr);
    ExitProcess(1);
}

// ---------------------------------------------------------------- module image
struct Sym { std::string name; u8 type; u16 shndx; u64 value; };
struct Module {
    const char* name; u64 base, size;
    u64 dyn[64] = {};           // DT_* < 64
    std::vector<Sym> syms;
    std::unordered_map<std::string, u64> exports;   // name -> absolute address
};

static void load_module(Module& m, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open %s", path.c_str());
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<u8> tmp(fsz); fread(tmp.data(), 1, fsz, f); fclose(f);
    u32 mod0 = *(u32*)&tmp[4];
    if (memcmp(&tmp[mod0], "MOD0", 4)) die("%s: no MOD0", m.name);
    s32 bss_end = *(s32*)&tmp[mod0 + 12];
    m.size = ((u64)(mod0 + bss_end) + 0xFFFF) & ~0xFFFFull;
    if (m.size < (u64)fsz) m.size = ((u64)fsz + 0xFFFF) & ~0xFFFFull;
    void* p = VirtualAlloc((void*)m.base, m.size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p != (void*)m.base) die("%s: cannot map at %llx (err %lu)", m.name, m.base, GetLastError());
    memcpy(p, tmp.data(), fsz);
    u64 dyn = m.base + mod0 + *(s32*)&tmp[mod0 + 4];
    for (u64 d = dyn; ; d += 16) {
        u64 tag = *(u64*)d, val = *(u64*)(d + 8);
        if (tag == 0) break;
        if (tag < 64) m.dyn[tag] = val;
    }
    u64 symtab = m.base + m.dyn[6], strtab = m.base + m.dyn[5];
    u32 nsyms = *(u32*)(m.base + m.dyn[4] + 4);          // DT_HASH nchain
    for (u32 i = 0; i < nsyms; i++) {
        u64 s = symtab + i * 24ull;
        Sym y{ std::string((const char*)(strtab + *(u32*)s)), (u8)(*(u8*)(s + 4) & 0xF), *(u16*)(s + 6), *(u64*)(s + 8) };
        if (y.shndx != 0 && !y.name.empty()) m.exports[y.name] = m.base + y.value;
        m.syms.push_back(std::move(y));
    }
    printf("[ld] %s at %llx size %llx syms %u exports %zu\n", m.name, m.base, m.size, nsyms, m.exports.size());
}

static Module g_main{"main", MAIN_BASE}, g_sdk{"sdk", SDK_BASE};
static std::unordered_map<std::string, u32> g_imp_index;

// resolve(sym) for a module: own definition, else import (fake code addr / sdk data), else 0
static u64 resolve(Module& m, u32 si, int* unresolved) {
    const Sym& y = m.syms[si];
    if (y.shndx != 0) return m.base + y.value;
    if (&m == &g_main) {
        if (y.type == 1) {                                   // STT_OBJECT: real data from the sdk image
            auto it = g_sdk.exports.find(y.name);
            if (it != g_sdk.exports.end()) return it->second;
        } else {
            auto it = g_imp_index.find(y.name);
            if (it != g_imp_index.end()) return IMP_FAKE + 4ull * it->second;
        }
    } else {
        auto it = g_main.exports.find(y.name);
        if (it != g_main.exports.end()) return it->second;
    }
    (*unresolved)++;
    return 0;
}

static void relocate(Module& m) {
    int unresolved = 0; u64 n = 0;
    const u64 tabs[2][2] = { { m.dyn[7], m.dyn[8] }, { m.dyn[23], m.dyn[2] } };   // RELA/RELASZ, JMPREL/PLTRELSZ
    for (auto& t : tabs) {
        for (u64 r = m.base + t[0]; r < m.base + t[0] + t[1]; r += 24, n++) {
            u64 off = *(u64*)r, info = *(u64*)(r + 8); s64 add = *(s64*)(r + 16);
            u32 type = (u32)info, sym = (u32)(info >> 32);
            u64* where = (u64*)(m.base + off);
            switch (type) {
            case 1027: *where = m.base + add; break;                                       // RELATIVE
            case 257: case 1025: case 1026: *where = resolve(m, sym, &unresolved) + add; break; // ABS64 / GLOB_DAT / JUMP_SLOT
            case 0: break;
            default: die("%s: reloc type %u at %llx", m.name, type, off);
            }
        }
    }
    printf("[ld] %s: %llu relocs, %d unresolved symbol refs\n", m.name, n, unresolved);
}

// ---------------------------------------------------------------- import tracing
static GuestFn g_hle[kMaxImports];
static std::atomic<u64> g_imp_calls[kMaxImports];
static std::atomic<u64> g_imp_seq{0};
thread_local const char* t_hle_cur;      // import / NVN proc currently executing on this thread (crash reports)
static bool g_trace_all = getenv("TOTK_TRACE_ALL") != nullptr;

static std::unordered_map<std::string, GuestFn>& hle_table() { static std::unordered_map<std::string, GuestFn> t; return t; }
HleReg::HleReg(const char* name, GuestFn fn) { hle_table()[name] = fn; }

// Full call counts only with TOTK_IMP_STATS=1 (or the watchdog): run386's guest profile had 5 % of all guest samples on
// this lock xadd and nvn_dispatch's (the counter lines bounce between the cores of every calling thread). Otherwise a
// plain read once the first 3 calls are logged, and print_stats shows counts capped at 3.
#ifdef TOTK_PGO_GEN
extern "C" int __llvm_profile_write_file(void);   // (compiler-rt profile runtime)
#endif
bool g_imp_count_all = getenv("TOTK_IMP_STATS") != nullptr || getenv("TOTK_WATCHDOG") != nullptr;
static void imp_dispatch(int i, Ctx* c) {
    u64 n = g_imp_calls[i].load(std::memory_order_relaxed);
    if (n < 3 || g_imp_count_all) n = g_imp_calls[i].fetch_add(1, std::memory_order_relaxed) + 1;
    else n = 4;
    if (n <= 3 || g_trace_all) {
        printf("[imp %6llu] %s%s(%llx, %llx, %llx, %llx) lr=%llx\n", (unsigned long long)++g_imp_seq,
               g_hle[i] ? "" : "UNIMPL ", g_imp_names[i], c->x[0], c->x[1], c->x[2], c->x[3], c->x[30]);
        if (!g_hle[i]) fflush(stdout);
    }
    const char* prev = t_hle_cur; t_hle_cur = g_imp_names[i];
    if (g_hle[i]) g_hle[i](c);
    else c->x[0] = 0;
    t_hle_cur = prev;
}
template <int N> static void imp_thunk(Ctx* c) { imp_dispatch(N, c); }
template <int... I> static constexpr std::array<GuestFn, sizeof...(I)> mk_thunks(std::integer_sequence<int, I...>) { return { imp_thunk<I>... }; }
static constexpr auto k_thunks = mk_thunks(std::make_integer_sequence<int, kMaxImports>{});

void rt_call_indirect(Ctx* c, u64 t) {
    if (t >= SDK_BASE && t < SDK_BASE + g_sdk.size) {
        u64 i = (t - SDK_BASE) >> 2;
        if (i < g_sftab_n && g_sftab[i]) return g_sftab[i](c);
        rt_trap(c, t, "indirect call into sdk image (no lifted function at this address)");
    }
    rt_trap(c, t, "indirect call to unknown target");
}
static thread_local Ctx* t_ctx;     // guest context of this host thread (crash reports, rt_fatal_guest)
// Fatal error from host code (NVN layer etc.): message + the current guest thread's x29 backtrace.
extern "C" [[noreturn]] void rt_fatal_guest(const char* msg) {
    fflush(stdout);
    fprintf(stderr, "[fatal] %s", msg);
    if (Ctx* c = t_ctx) {
        fprintf(stderr, "[fatal] lr=%llx sp=%llx\n", c->x[30], c->sp);
        u64 fp = c->x[29];
        for (int i = 0; i < 24 && fp >= c->sp && fp < c->sp + (256ull << 20) && !(fp & 7); i++) {
            u64 nfp = *(u64*)fp;
            fprintf(stderr, "[fatal]   #%-2d lr=%llx\n", i, *(u64*)(fp + 8));
            if (nfp <= fp) break;
            fp = nfp;
        }
    }
    fflush(stderr);
    ExitProcess(4);
}
// gen.py --trace hook: entry (x0-x7, lr) and ret (x0) of selected functions; TOTK_TRACE_MAX lines per function
void rt_trace(Ctx* c, u64 fn, int ret) {
    static std::unordered_map<u64, int> n; static std::mutex m;
    static int maxn = getenv("TOTK_TRACE_MAX") ? atoi(getenv("TOTK_TRACE_MAX")) : 200;
    std::lock_guard<std::mutex> l(m);
    if (n[fn * 2 + ret]++ >= maxn) return;
    if (!ret) printf("[trace %5lu] > %llx(%llx, %llx, %llx, %llx, %llx, %llx, %llx, %llx) lr=%llx\n", GetCurrentThreadId(), fn,
                     c->x[0], c->x[1], c->x[2], c->x[3], c->x[4], c->x[5], c->x[6], c->x[7], c->x[30]);
    else printf("[trace %5lu] < %llx = %llx\n", GetCurrentThreadId(), fn, c->x[0]);
}
// TPIDRRO_EL0 for lifted sdk code: Horizon's per-thread 0x200-byte ThreadLocalRegion (sdk keeps its thread pointer
// at +0x1F0). One per host thread (guest threads are host threads).
u64 rt_tpidrro(Ctx*) { alignas(16) static thread_local u8 tlr[0x200]; return (u64)tlr; }
// sdk PLT slot with nothing to bind (no HLE, no main export, no lifted definition)
// The name comes from the caller's `bl <plt stub>` (lr - 4) -> stub's adrp/ldr GOT slot -> JMPREL symbol.
static void sdk_imp_missing(Ctx* c) {
    static char msg[512]; const char* nm = "?";
    u64 lr = c->x[30];
    if (lr >= SDK_BASE + 4 && lr < SDK_BASE + g_sdk.size) {
        u32 bl = *(u32*)(lr - 4);
        if ((bl >> 26) == 0x25) {
            u64 stub = lr - 4 + ((s64)((s32)(bl << 6) >> 6) << 2);
            u32 adrp = *(u32*)stub, ldr = *(u32*)(stub + 4);
            u64 imm = (((adrp >> 29) & 3) | (((adrp >> 5) & 0x7FFFF) << 2)) << 12;
            u64 got = (stub & ~0xFFFull) + imm + ((ldr >> 10) & 0xFFF) * 8 - SDK_BASE;
            for (u64 r = g_sdk.base + g_sdk.dyn[23]; r < g_sdk.base + g_sdk.dyn[23] + g_sdk.dyn[2]; r += 24)
                if (*(u64*)r == got) { nm = g_sdk.syms[*(u64*)(r + 8) >> 32].name.c_str(); break; }
        }
    }
    snprintf(msg, sizeof msg, "sdk import with no binding: %s (caller = lr)", nm);
    rt_trap(c, lr, msg);
}

// ---------------------------------------------------------------- crash reporting
static const char* volatile g_stage = "load";
static volatile u64 g_stage_idx;
static std::vector<std::pair<u64, u64>> g_fnaddr;     // host fn address -> guest offset, sorted

static u64 guest_fn_of(u64 rip) {      // guest offset; sdk functions are tagged with SDK_TAG
    if (g_fnaddr.empty()) {
        for (u64 i = 0; i < g_ftab_count; i++) g_fnaddr.push_back({ (u64)g_ftab_init[i].fn, g_ftab_init[i].off });
        for (u64 i = 0; i < g_s_ftab_count; i++) g_fnaddr.push_back({ (u64)g_s_ftab_init[i].fn, g_s_ftab_init[i].off | (1ull << 40) });
        std::sort(g_fnaddr.begin(), g_fnaddr.end());
    }
    auto it = std::upper_bound(g_fnaddr.begin(), g_fnaddr.end(), std::make_pair(rip, ~0ull));
    return it == g_fnaddr.begin() ? ~0ull : std::prev(it)->second;
}

extern "C" u64 g_rt_heap_base;
extern "C" u64 kernel_heap_size(void);
extern "C" unsigned long long renderer_present_count(void);   // (nvn backend: presents so far)
extern "C" uint64_t SDL_GetTicksNS(void);

// ---------------------------------------------------------------- hardware write watchpoint (diagnostics)
// rt_watch_write(addr): DR0 = 4-byte write watch on every thread that exists now; each hit logs the guest function
// that wrote, the written value and the guest lr (first 40 hits). Used to find which game code rewrites a variable.
#ifndef _WIN32
// Linux: no hardware watchpoints (would need perf_event_open / ptrace on every thread); the TOTK_WATCH / TOTK_VFR_WATCH diagnostics
// are Windows-only.
extern "C" void rt_watch_write(u64 addr) { printf("[watch] write watch on %llx: not available on Linux\n", (unsigned long long)addr); }
#else
static volatile u64 g_watch_addr;
static volatile long g_watch_hits;
static LONG WINAPI on_watch(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !(ep->ContextRecord->Dr6 & 1)) return EXCEPTION_CONTINUE_SEARCH;
    ep->ContextRecord->Dr6 = 0;
    long n = InterlockedIncrement(&g_watch_hits);
    static const long wmax = getenv("TOTK_WATCH_MAX") ? atol(getenv("TOTK_WATCH_MAX")) : 40;   // (TOTK_WATCH_MAX: more hits)
    if (n <= wmax) {
        u64 rip = ep->ContextRecord->Rip, gf = guest_fn_of(rip);
        Ctx* c = t_ctx;
        printf("[watch] #%ld write %08x to %llx by guest fn %s_%llx (host %llx) lr %llx frame %llu tid %lu t %llu\n", n, *(volatile u32*)g_watch_addr,
               (unsigned long long)g_watch_addr, gf >> 40 == 1 ? "s" : "f", gf & 0xFFFFFFFFFFull, rip,
               c ? (unsigned long long)c->x[30] : 0ull, (unsigned long long)renderer_present_count(), GetCurrentThreadId(),
               (unsigned long long)(SDL_GetTicksNS() / 1000));
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}
extern "C" void rt_watch_write(u64 addr) {
    g_watch_addr = addr;
    static bool handler;   // re-arming (TOTK_WATCH) must not stack handlers
    if (!handler) { handler = true; AddVectoredExceptionHandler(1, on_watch); }
    std::thread([addr] {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        THREADENTRY32 te{sizeof te};
        DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
        int n = 0;
        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
            if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
            HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, te.th32ThreadID);
            if (!h) continue;
            if (SuspendThread(h) != (DWORD)-1) {
                CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
                if (GetThreadContext(h, &ctx)) {
                    ctx.Dr0 = addr;
                    ctx.Dr7 = (ctx.Dr7 & ~0xF0003ull) | 1ull | (1ull << 16) | (3ull << 18);  // L0, RW0 = write, LEN0 = 4 bytes
                    if (SetThreadContext(h, &ctx)) n++;
                }
                ResumeThread(h);
            }
            CloseHandle(h);
        }
        CloseHandle(snap);
        printf("[watch] write watch on %llx set in %d threads\n", (unsigned long long)addr, n);
    }).join();
}
#endif

#ifndef _WIN32
// Linux fatal-signal handler (installed by main(); plat_posix.cpp runs it on the faulting thread's alternate stack after the
// write-watch tracker declined the fault): the same report as on_crash below, host registers from the ucontext.
static void on_crash_signal(int sig, siginfo_t* si, void* uc) {
    fflush(stdout);
    ucontext_t* ctx = (ucontext_t*)uc;
    u64 rip = (u64)ctx->uc_mcontext.gregs[REG_RIP];
    u64 gf = guest_fn_of(rip);
    fprintf(stderr, "[crash] signal %d (%s) code %d at host %llx (guest fn %s_%llx) addr %llx  stage %s #%llu\n", sig, strsignal(sig), si->si_code,
            (unsigned long long)rip, gf >> 40 == 1 ? "s" : "f", (unsigned long long)(gf & 0xFFFFFFFFFFull), (unsigned long long)(u64)si->si_addr,
            g_stage, (unsigned long long)g_stage_idx);
    fprintf(stderr, "[crash] exe+%llx (load %llx)  in HLE: %s\n", (unsigned long long)(rip - rt_posix_exe_base()),
            (unsigned long long)rt_posix_exe_base(), t_hle_cur ? t_hle_cur : "(none: generated code)");
    if (sig == SIGSEGV || sig == SIGBUS)   // what the faulting page is
        fprintf(stderr, "[crash] fault address %llx is %sreadable; main heap %llx+%llx\n", (unsigned long long)(u64)si->si_addr,
                rt_posix_readable((u64)si->si_addr, 1) ? "" : "NOT ", (unsigned long long)g_rt_heap_base, (unsigned long long)kernel_heap_size());
    static const char* const rn[] = {"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rdi", "rsi", "rbp", "rbx", "rdx", "rax", "rcx", "rsp", "rip"};
    for (int i = 0; i < 17; i++) fprintf(stderr, "%-3s=%016llx%s", rn[i], (unsigned long long)ctx->uc_mcontext.gregs[i], i % 4 == 3 ? "\n" : "  ");
    fprintf(stderr, "\n");
    if (Ctx* c = t_ctx) {
        for (int i = 0; i < 31; i++) fprintf(stderr, "x%-2d=%016llx%s", i, (unsigned long long)c->x[i], i % 4 == 3 ? "\n" : "  ");
        fprintf(stderr, "  sp=%016llx tpidr=%016llx\n", (unsigned long long)c->sp, (unsigned long long)c->tpidr);
        u64 fp = c->x[29];      // guest backtrace (x29 frame chain on the guest stack)
        for (int i = 0; i < 24 && fp >= c->sp && fp < c->sp + (256ull << 20) && !(fp & 7) && rt_posix_readable(fp, 16); i++) {
            u64 nfp = *(u64*)fp;
            fprintf(stderr, "[crash]   #%-2d lr=%llx\n", i, (unsigned long long)*(u64*)(fp + 8));
            if (nfp <= fp) break;
            fp = nfp;
        }
    }
    fflush(stderr);
    _exit(3);
}
#else
static LONG WINAPI on_crash(EXCEPTION_POINTERS* ep) {
    auto* er = ep->ExceptionRecord;
    if (er->ExceptionCode < 0xC0000000u) return EXCEPTION_CONTINUE_SEARCH;
    fflush(stdout);
    u64 rip = ep->ContextRecord->Rip;
    u64 gf = guest_fn_of(rip);
    fprintf(stderr, "[crash] code %08lx at host %llx (guest fn %s_%llx) addr %llx  stage %s #%llu\n", er->ExceptionCode, rip,
            gf >> 40 == 1 ? "s" : "f", gf & 0xFFFFFFFFFFull, er->NumberParameters > 1 ? (u64)er->ExceptionInformation[1] : 0ull,
            g_stage, g_stage_idx);
    fprintf(stderr, "[crash] exe+%llx  in HLE: %s\n", rip - (u64)GetModuleHandleW(nullptr), t_hle_cur ? t_hle_cur : "(none: generated code)");
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters > 1) {   // what the faulting page is
        MEMORY_BASIC_INFORMATION mi{};
        u64 fa = (u64)er->ExceptionInformation[1];
        if (VirtualQuery((void*)fa, &mi, sizeof mi))
            fprintf(stderr, "[crash] fault page: state %lx protect %lx region %llx+%llx alloc base %llx (%s)  main heap %llx+%llx\n",
                    mi.State, mi.Protect, (u64)mi.BaseAddress, (u64)mi.RegionSize, (u64)mi.AllocationBase,
                    er->ExceptionInformation[0] == 1 ? "write" : "read", g_rt_heap_base, kernel_heap_size());
    }
    if (Ctx* c = t_ctx) {
        for (int i = 0; i < 31; i++) fprintf(stderr, "x%-2d=%016llx%s", i, c->x[i], i % 4 == 3 ? "\n" : "  ");
        fprintf(stderr, "  sp=%016llx tpidr=%016llx\n", c->sp, c->tpidr);
        u64 fp = c->x[29];      // guest backtrace (x29 frame chain on the guest stack)
        for (int i = 0; i < 24 && fp >= c->sp && fp < c->sp + (256ull << 20) && !(fp & 7); i++) {
            u64 nfp = *(u64*)fp;
            fprintf(stderr, "[crash]   #%-2d lr=%llx\n", i, *(u64*)(fp + 8));
            if (nfp <= fp) break;
            fp = nfp;
        }
    }
    fflush(stderr);
    TerminateProcess(GetCurrentProcess(), 3);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

// ---------------------------------------------------------------- guest thread
// New guest CPU context. stack_top == 0: allocate a 32 MB guest stack (main thread); else the guest gave one.
Ctx* rt_new_ctx(u64 stack_top, u64* stack_lo, u64* stack_size) {
    Ctx* c = (Ctx*)_aligned_malloc(sizeof(Ctx), 64); memset(c, 0, sizeof(Ctx));
    const u64 tls_sz = 0x10000;
    if (!stack_top) {
        const u64 stack_sz = 32ull << 20;
        u8* stk = (u8*)VirtualAlloc(nullptr, stack_sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        stack_top = (u64)stk + stack_sz;
        if (stack_lo) *stack_lo = (u64)stk;
        if (stack_size) *stack_size = stack_sz;
    }
    u8* tls = (u8*)VirtualAlloc(nullptr, tls_sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    c->sp = (stack_top - 0x100) & ~15ull; c->tpidr = (u64)tls; c->excl_addr = ~0ull;
    return c;
}
void hle_evring_dump(u64 focus);
void hle_nvn_dump_top(int top);
// Guest thread registry for the watchdog (TOTK_WATCHDOG=secs): where each thread is when the game stops progressing.
struct GuestThreadRec { Ctx* c; const char* const* hle_cur; DWORD tid; };
static std::mutex g_threads_mtx;
static std::vector<GuestThreadRec> g_threads;
void rt_bind_thread(Ctx* c) {
    t_ctx = c;
    std::lock_guard<std::mutex> l(g_threads_mtx);
    g_threads.push_back({c, &t_hle_cur, GetCurrentThreadId()});
}
// A host-only thread (the NVN render thread) sampled by TOTK_PROF like a guest thread: its samples are charged to
// "hle <name>" and its host RVAs go to hostprof.txt under <name> (tools/hostprof.py --hle <name>). No Ctx.
extern "C" void rt_prof_register_host(const char* name) {
    t_hle_cur = name;
    std::lock_guard<std::mutex> l(g_threads_mtx);
    g_threads.push_back({nullptr, &t_hle_cur, GetCurrentThreadId()});
}
// Called by a guest thread before it exits: its Ctx and thread_local t_hle_cur die with it (run72: the watchdog read a
// dead thread's t_hle_cur and crashed once the game started destroying threads).
void rt_unbind_thread() {
    std::lock_guard<std::mutex> l(g_threads_mtx);
    DWORD me = GetCurrentThreadId();
    for (size_t i = 0; i < g_threads.size(); i++)
        if (g_threads[i].tid == me) { g_threads.erase(g_threads.begin() + i); break; }
}
// Sampling profiler: suspend each guest thread `samples` times (every `ms` ms), map its host RIP to the generated guest
// function (or the HLE import it is in, or "host" for code outside the exe) and print the top entries per thread and
// overall. Answers "busy or parked, and where" for guest-code threads that have no import to show.
#ifdef _WIN32
static std::string module_of(u64 rip) {   // (profile_threads) base name of the module holding rip, "?" if none
    static std::map<u64, std::string> cache;
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)rip, &m) || !m)
        return "?";
    auto it = cache.find((u64)m);
    if (it != cache.end()) return it->second;
    char path[MAX_PATH] = "";
    GetModuleFileNameA(m, path, MAX_PATH);
    const char* b = strrchr(path, '\\');
    std::string n = b ? b + 1 : path;
    cache[(u64)m] = n;
    return n;
}
static void profile_threads(int samples, int ms) {
    static u64 exe_lo, exe_hi;
    if (!exe_lo) {
        exe_lo = (u64)GetModuleHandleW(nullptr);
        auto* nt = (IMAGE_NT_HEADERS*)(exe_lo + ((IMAGE_DOS_HEADER*)exe_lo)->e_lfanew);
        exe_hi = exe_lo + nt->OptionalHeader.SizeOfImage;
    }
    struct T { DWORD tid; HANDLE h; const char* const* hle; std::unordered_map<std::string, int> hits; u64 cpu0; };
    auto cpu_of = [](HANDLE h) -> u64 { FILETIME a, b, k, u; if (!GetThreadTimes(h, &a, &b, &k, &u)) return 0;
        return ((u64)k.dwHighDateTime << 32 | k.dwLowDateTime) + ((u64)u.dwHighDateTime << 32 | u.dwLowDateTime); };  // 100 ns units
    std::vector<T> ts;
    {
        std::lock_guard<std::mutex> l(g_threads_mtx);
        for (auto& g : g_threads)
            if (HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, g.tid))
                ts.push_back({ g.tid, h, g.hle_cur, {}, cpu_of(h) });
    }
    char buf[64];
    // host code samples inside HLE calls (our own runtime: the NVN backend in nvnQueueSubmitCommands, ...): RVA -> count,
    // appended to hostprof.txt as "<hle> <rva hex> <count>"; tools/hostprof.py resolves the RVAs with totk.exe.map
    std::map<std::pair<std::string, u64>, int> host_hits;
    u64 wall0 = GetTickCount64();
    for (int s = 0; s < samples; s++) {
        for (auto& t : ts) {
            if (SuspendThread(t.h) == (DWORD)-1) continue;
            CONTEXT cx{}; cx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            BOOL ok = GetThreadContext(t.h, &cx);
            const char* hle = *t.hle;
            // host code outside the exe inside an HLE call (GPU driver, CRT, kernel waits): unwind to the first exe frame
            // while the thread is still suspended, so the sample is charged to "<hle>@<module>" at that caller
            u64 via = 0;
            std::string mod;
            if (ok && hle && (cx.Rip < exe_lo || cx.Rip >= exe_hi)) {
                mod = module_of(cx.Rip);
                CONTEXT u = cx;
                for (int d = 0; d < 48; d++) {
                    DWORD64 ib = 0;
                    PRUNTIME_FUNCTION f = RtlLookupFunctionEntry(u.Rip, &ib, nullptr);
                    if (!f) {   // leaf: the return address is at [rsp]
                        MEMORY_BASIC_INFORMATION mi;
                        if (!VirtualQuery((void*)u.Rsp, &mi, sizeof mi) || mi.State != MEM_COMMIT) break;
                        u.Rip = *(u64*)u.Rsp; u.Rsp += 8;
                    } else {
                        void* hd = nullptr; DWORD64 ef = 0;
                        RtlVirtualUnwind(UNW_FLAG_NHANDLER, ib, u.Rip, f, &u, &hd, &ef, nullptr);
                    }
                    if (!u.Rip) break;
                    if (u.Rip >= exe_lo && u.Rip < exe_hi) { via = u.Rip - exe_lo; break; }
                }
            }
            ResumeThread(t.h);
            if (!ok) continue;
            if (hle) {
                t.hits[std::string("hle ") + hle]++;
                if (cx.Rip >= exe_lo && cx.Rip < exe_hi) host_hits[{ hle, cx.Rip - exe_lo }]++;   // (host code inside an HLE call)
                else host_hits[{ std::string(hle) + "@" + mod, via }]++;                          // (outside: charged to its exe caller)
            } else if (cx.Rip < exe_lo || cx.Rip >= exe_hi) t.hits["host (outside exe)"]++;
            else {
                u64 gf = guest_fn_of(cx.Rip);
                snprintf(buf, sizeof buf, "%s_%llx", gf >> 40 == 1 ? "s" : "f", gf & 0xFFFFFFFFFFull);
                t.hits[buf]++;
                // also by exact RVA (hostprof.py + the map): guest_fn_of charges runtime code linked after the last guest
                // function to that function (run367: "f_2aa41c0", a one-shot initializer, was the top "guest" entry)
                host_hits[{ "guest", cx.Rip - exe_lo }]++;
            }
        }
        Sleep(ms);
    }
    u64 wall = GetTickCount64() - wall0; if (!wall) wall = 1;
    std::unordered_map<std::string, int> all;
    std::sort(ts.begin(), ts.end(), [&](const T& a, const T& b) { return cpu_of(a.h) - a.cpu0 > cpu_of(b.h) - b.cpu0; });
    u64 cpu_all = 0;
    for (auto& t : ts) {
        u64 cpu_ms = (cpu_of(t.h) - t.cpu0) / 10000; cpu_all += cpu_ms;
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k, n] : t.hits) { v.push_back({ n, k }); if (k.rfind("hle ", 0) && k.rfind("host", 0)) all[k] += n; }
        std::sort(v.rbegin(), v.rend());
        bool busy = (!v.empty() && v[0].second.rfind("hle ", 0) != 0) || cpu_ms * 100 >= wall * 3;  // parked threads sit in one HLE wait
        if (busy) {
            char nm[64] = "";
            PWSTR wd = nullptr;
            if (SUCCEEDED(GetThreadDescription(t.h, &wd)) && wd) {
                int i = 0; for (; wd[i] && i < 63; i++) nm[i] = (char)wd[i]; nm[i] = 0;
                LocalFree(wd);
            }
            printf("[prof] tid %5lu %-24s cpu %3llu%%:", t.tid, nm, cpu_ms * 100 / wall);
            for (size_t i = 0; i < v.size() && i < 4; i++) printf(" %s %d%%", v[i].second.c_str(), v[i].first * 100 / samples);
            printf("\n");
        }
        CloseHandle(t.h);
    }
    std::vector<std::pair<int, std::string>> v;
    for (auto& [k, n] : all) v.push_back({ n, k });
    std::sort(v.rbegin(), v.rend());
    if (FILE* hf = fopen("hostprof.txt", "a")) {
        fprintf(hf, "# sampling %d x %d ms\n", samples, ms);
        for (auto& [k, n] : host_hits) fprintf(hf, "%s %llx %d\n", k.first.c_str(), (unsigned long long)k.second, n);
        fclose(hf);
    }
    printf("[prof] guest threads cpu total %llu%% of one core over %llu ms\n", cpu_all * 100 / wall, wall);
    printf("[prof] guest code overall (thread-samples of %d):", samples);
    for (size_t i = 0; i < v.size() && i < 15; i++) printf(" %s %d", v[i].second.c_str(), v[i].first);
    printf("\n");
}
#else
// Linux: TOTK_PROF / TOTK_WATCHDOG sampling needs SuspendThread + GetThreadContext (and a PE module list): not available. The
// watchdog still lists every guest thread's HLE position and guest backtrace below; for host profiles use `perf record -g`.
static void profile_threads(int samples, int ms) {
    static bool told;
    if (!told) { told = true; printf("[prof] host-code sampling (TOTK_PROF) is Windows-only; use perf on Linux\n"); }
    (void)samples; (void)ms;
}
#endif
static void watchdog_dump() {
    profile_threads(100, 20);
#ifdef _WIN32
    auto readable = [](u64 a) {
        MEMORY_BASIC_INFORMATION mi;
        return a && !(a & 7) && VirtualQuery((void*)a, &mi, sizeof mi) && mi.State == MEM_COMMIT &&
               !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) && a + 16 <= (u64)mi.BaseAddress + mi.RegionSize;
    };
#else
    auto readable = [](u64 a) { return a && !(a & 7) && rt_posix_readable(a, 16) != 0; };
#endif
    std::lock_guard<std::mutex> l(g_threads_mtx);
    printf("[watchdog] %zu guest threads\n", g_threads.size());
    for (auto& t : g_threads) {
        if (!t.c) continue;   // (host-only thread: rt_prof_register_host)
        const char* cur = *t.hle_cur;
        printf("[watchdog] tid %5lu in %s lr=%llx sp=%llx bt:", t.tid, cur ? cur : "(guest code)", t.c->x[30], t.c->sp);
        u64 fp = t.c->x[29];
        for (int i = 0; i < 10 && readable(fp) && fp >= t.c->sp; i++) {
            u64 nfp = *(u64*)fp, lr = *(u64*)(fp + 8);
            printf(" %llx", lr);
            if (nfp <= fp) break;
            fp = nfp;
        }
        printf("\n[watchdog]   x19-x28:");
        for (int r = 19; r <= 28; r++) printf(" %llx", t.c->x[r]);
        printf("\n");
        // TEMP (job-system hang): in the waiter f_813358, x21 = pool W: pending [W+0xc0], flags [W+0xbc], list W+0x40
        if (cur && strstr(cur, "WaitLightEvent") && readable(t.c->x[21]) && readable(t.c->x[21] + 0xc8)) {
            const u64* w = (const u64*)t.c->x[21];
            for (int i = 0; i < 26; i += 4)
                printf("[watchdog]   W+%02x: %llx %llx %llx %llx\n", i * 8, w[i], w[i + 1], w[i + 2], w[i + 3]);
            u64 ev = t.c->x[0], s = ev - 0xb0;     // waiter record S: claim [S+8], flags [S+0x10], event S+0xb0, id [S+0xd8]
            if (readable(s) && readable(s + 0xd8)) {
                const u64* p = (const u64*)s;
                printf("[watchdog]   S=%llx: +08 %llx +10 %llx +d8 %llx\n", s, p[1], p[2], p[0xd8 / 8]);
            }
            hle_evring_dump(ev);
        }
    }
    hle_nvn_dump_top(25);
    fflush(stdout);
}
static_assert(offsetof(Ctx, sp) == 256 && offsetof(Ctx, v) == 264, "runtime/nvn/glue.h Cpu mirrors this layout");
// Host -> guest call (NVN memory callbacks etc.): run the guest function at addr on the current context.
extern "C" void guest_call(Ctx* c, u64 addr) { u64 lr = c->x[30]; call_ind(c, addr); c->x[30] = lr; }
void os_register_main_thread(Ctx* c, u64 stack_lo, u64 stack_size);


static DWORD WINAPI boot_thread(LPVOID) {
    u64 slo = 0, ssz = 0;
    Ctx* c = rt_new_ctx(0, &slo, &ssz); rt_bind_thread(c);
    os_register_main_thread(c, slo, ssz);
    g_stage = "DT_INIT";
    u64 init = MAIN_BASE + g_main.dyn[12];
    printf("[boot] DT_INIT %llx\n", init); fflush(stdout);
    guest_call(c, init);
    auto e = g_main.exports.find("nninitStartup");
    if (e == g_main.exports.end()) die("no nninitStartup export");
    printf("[boot] nninitStartup %llx\n", e->second); fflush(stdout);
    g_stage = "nninitStartup"; guest_call(c, e->second);
    g_stage = "init_array";
    u64 ia = MAIN_BASE + g_main.dyn[25], ian = g_main.dyn[27] / 8;
    printf("[boot] init_array %llu entries\n", ian); fflush(stdout);
    for (u64 i = 0; i < ian; i++) {
        u64 fn = ((u64*)ia)[i];
        if (fn == 0 || fn == ~0ull) continue;
        c->x[0] = 0; c->x[1] = 0; c->x[2] = 0; g_stage_idx = i;
        guest_call(c, fn);
        if ((i + 1) % 1000 == 0) { printf("[boot] init_array %llu/%llu\n", i + 1, ian); fflush(stdout); }
    }
    e = g_main.exports.find("nnMain");
    printf("[boot] nnMain entered (%llx)\n", e->second); fflush(stdout); g_stage = "nnMain";
    guest_call(c, e->second);
    printf("[boot] nnMain returned x0=%llx\n", c->x[0]);
    return 0;
}

extern "C" int ui_run(void);
static void print_stats() {
    fflush(stdout);
    std::vector<std::pair<u64, int>> top;
    for (u32 i = 0; i < g_imp_count; i++) if (g_imp_calls[i]) top.push_back({ g_imp_calls[i].load(), (int)i });
    std::sort(top.rbegin(), top.rend());
    printf("[stats] %zu distinct imports called%s\n", top.size(), g_imp_count_all ? "" : " (counts capped at 3; TOTK_IMP_STATS=1 counts all)");
    for (auto& [n, i] : top) printf("  %10llu %s%s\n", n, g_hle[i] ? "" : "UNIMPL ", g_imp_names[i]);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOFBF, 1 << 16);
    // Idle priority class by default: test runs shared the PC with the user's video playback and made its audio crackle
    // (user 2026-10-14). TOTK_PRIORITY=normal opts out (e.g. for clean perf measurements).
#ifdef _WIN32
    { const char* pr = getenv("TOTK_PRIORITY"); if (!pr || strcmp(pr, "normal") != 0) SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS); }
#else
    // Linux: normal priority by default (this is the game on the Deck); TOTK_PRIORITY=idle = nice 19 for background test runs
    { const char* pr = getenv("TOTK_PRIORITY"); if (pr && strcmp(pr, "idle") == 0) (void)!setpriority(PRIO_PROCESS, 0, 19); }
    rt_posix_install_signals(on_crash_signal);
    signal(SIGPIPE, SIG_IGN);   // a log piped into a closed reader must not end the game
#endif
    // flush once a second: a run killed by `timeout` otherwise loses its last 64 KiB of log (run83 lost its ending)
    CloseHandle(CreateThread(nullptr, 0, [](LPVOID) -> DWORD { for (;;) { Sleep(1000); fflush(stdout); fflush(stderr); } }, nullptr, 0, nullptr));
#ifdef _WIN32
    std::string dir = argc > 1 ? argv[1] : "C:/totk/extract/img";
#else
    // the decompressed NSO images (tools/nso.py): argv[1], else $TOTK_IMG, else <exe dir>/data/img
    std::string dir = argc > 1 ? argv[1] : getenv("TOTK_IMG") && *getenv("TOTK_IMG") ? getenv("TOTK_IMG") : std::string(rt_posix_exe_dir()) + "/data/img";
#endif
    if (g_imp_count > (u64)kMaxImports) die("too many imports %llu", g_imp_count);
    for (u32 i = 0; i < g_imp_count; i++) g_imp_index[g_imp_names[i]] = i;

    load_module(g_main, dir + "/main.bin");
    load_module(g_sdk, dir + "/sdk.bin");
    relocate(g_sdk);
    relocate(g_main);

    // function table: every FDE start, plus the import fake addresses
    g_ftab_n = (IMP_FAKE - MAIN_BASE) / 4 + kMaxImports + kMaxHostFns;
    g_ftab = (GuestFn*)VirtualAlloc(nullptr, g_ftab_n * sizeof(GuestFn), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    for (u64 i = 0; i < g_ftab_count; i++) g_ftab[g_ftab_init[i].off >> 2] = g_ftab_init[i].fn;
    // sdk function table (sdk .text only)
    g_sftab_n = 0;
    for (u64 i = 0; i < g_s_ftab_count; i++) if ((g_s_ftab_init[i].off >> 2) + 1 > g_sftab_n) g_sftab_n = (g_s_ftab_init[i].off >> 2) + 1;
    g_sftab = (GuestFn*)VirtualAlloc(nullptr, g_sftab_n * sizeof(GuestFn), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    for (u64 i = 0; i < g_s_ftab_count; i++) g_sftab[g_s_ftab_init[i].off >> 2] = g_s_ftab_init[i].fn;
    auto lifted = [](const std::string& name) -> GuestFn {     // sdk export -> its lifted function
        auto e = g_sdk.exports.find(name);
        if (e == g_sdk.exports.end() || e->second < SDK_BASE) return nullptr;
        u64 i = (e->second - SDK_BASE) >> 2;
        return i < g_sftab_n ? g_sftab[i] : nullptr;
    };
    // main imports in the lift whitelist (namespace prefixes; TOTK_LIFT overrides, comma-separated) run lifted sdk code
    // instead of HLE: big pure libraries the game uses through many calls (font engine, codecs). "!prefix" excludes
    // (those names keep their HLE): prepo's msgpack builder is lifted, its Save* (a service call) stays HLE.
    std::vector<std::string> lift_prefix, lift_excl;
    {
        std::string s = getenv("TOTK_LIFT") ? getenv("TOTK_LIFT")
                      : "_ZN2nn6fontll,_ZNK2nn6fontll,_ZN2nn4util,_ZNK2nn4util,_ZN2nn5image,_ZNK2nn5image,"
                        "_ZN2nn5prepo,_ZNK2nn5prepo,!_ZN2nn5prepo10PlayReport4Save,!_ZN2nn5prepo28RequestImmediateTransmission";
        for (size_t p = 0, q; p < s.size(); p = q + 1) {
            q = s.find(',', p); if (q == std::string::npos) q = s.size();
            if (q > p) { std::string it = s.substr(p, q - p); if (it[0] == '!') lift_excl.push_back(it.substr(1)); else lift_prefix.push_back(it); }
        }
    }
    auto in_lift = [&](const std::string& nm) {
        for (auto& p : lift_excl) if (nm.compare(0, p.size(), p) == 0) return false;
        for (auto& p : lift_prefix) if (nm.compare(0, p.size(), p) == 0) return true;
        return false;
    };
    int nhle = 0, nlift = 0;
    for (u32 i = 0; i < g_imp_count; i++) {
        std::string nm = g_imp_names[i];
        GuestFn lf = nullptr;
        if (in_lift(nm)) lf = lifted(nm);
        auto it = hle_table().find(nm);
        if (lf) { g_hle[i] = lf; nlift++; }
        else if (it != hle_table().end()) { g_hle[i] = it->second; nhle++; }
        g_imp[i] = k_thunks[i];
        g_ftab[(IMP_FAKE - MAIN_BASE) / 4 + i] = k_thunks[i];
    }
    printf("[ld] ftab %llu functions, %llu imports (%d HLE, %d lifted sdk, %llu stubbed)\n", g_ftab_count, g_imp_count, nhle,
           nlift, g_imp_count - nhle - nlift);
    // sdk PLT slots: our HLE (same names main uses), else main's own export (rtld resolves main first), else the sdk's
    // lifted definition
    if (g_s_imp_count > kMaxSdkImports) die("too many sdk imports %llu", g_s_imp_count);
    int sh = 0, sm = 0, ss = 0, sx = 0;
    for (u32 i = 0; i < g_s_imp_count; i++) {
        std::string nm = g_s_imp_names[i];
        GuestFn f = nullptr;
        bool white = false;
        white = in_lift(nm);
        auto it = hle_table().find(nm);
        auto me = g_main.exports.find(nm);
        if (!white && it != hle_table().end()) { f = it->second; sh++; }
        else if (me != g_main.exports.end() && ((me->second - MAIN_BASE) >> 2) < g_ftab_n && g_ftab[(me->second - MAIN_BASE) >> 2]) {
            f = g_ftab[(me->second - MAIN_BASE) >> 2]; sm++;
        } else if ((f = lifted(nm))) ss++;
        else { f = sdk_imp_missing; sx++; printf("[ld] sdk unbound: %s\n", nm.c_str()); }
        g_s_imp[i] = f;
    }
    printf("[ld] sdk: %llu lifted functions; %llu PLT slots -> %d HLE, %d main, %d lifted, %d unbound\n", g_s_ftab_count,
           g_s_imp_count, sh, sm, ss, sx);
    fflush(stdout);

#ifdef _WIN32
    AddVectoredExceptionHandler(0, on_crash);
#endif
    // TOTK_NVN_PROBE=name1,name2,...: resolve NVN procs through the game's own (lifted) driver loader and print their
    // sdk offsets, to read real driver behaviour (e.g. nvnDeviceGetInteger constants) out of the sdk code.
    if (const char* probe = getenv("TOTK_NVN_PROBE")) {
        Ctx* c = rt_new_ctx(0, nullptr, nullptr); rt_bind_thread(c);
        GuestFn boot = lifted("nvnBootstrapLoader");
        if (!boot) die("probe: no lifted nvnBootstrapLoader");
        static char nm[256];
        strcpy(nm, "nvnDeviceGetProcAddress"); c->x[0] = (u64)nm; c->x[30] = 0; boot(c);
        u64 gpa = c->x[0];
        printf("[probe] nvnBootstrapLoader(nvnDeviceGetProcAddress) = %llx (sdk+%llx)\n", gpa, gpa - SDK_BASE);
        for (std::string s = probe; !s.empty();) {
            size_t q = s.find(','); std::string one = s.substr(0, q);
            strcpy(nm, one.c_str());
            c->x[0] = (u64)nm; boot(c); u64 a1 = c->x[0];
            u64 a2 = 0;
            if (gpa) { c->x[0] = 0; c->x[1] = (u64)nm; c->x[30] = 0; call_ind(c, gpa); a2 = c->x[0]; }
            printf("[probe] %s: bootstrap %llx (sdk+%llx)  getprocaddress %llx (sdk+%llx)\n", nm, a1, a1 - SDK_BASE, a2, a2 - SDK_BASE);
            s = q == std::string::npos ? "" : s.substr(q + 1);
        }
        fflush(stdout);
        return 0;
    }
    HANDLE h = CreateThread(nullptr, 1ull << 30, boot_thread, nullptr, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (const char* w = getenv("TOTK_WATCHDOG"))      // dump every guest thread's position every N seconds (3 times)
        CreateThread(nullptr, 0, [](LPVOID p) -> DWORD {
            for (int i = 0; i < 3; i++) { Sleep((DWORD)(uintptr_t)p * 1000); watchdog_dump(); }
            return 0;
        }, (LPVOID)(uintptr_t)atoi(w), 0, nullptr);
#ifdef TOTK_PGO_GEN
    // PGO instrumented build (mkninja TOTK_PGO=gen): write the profile at TOTK_PGO_DUMP seconds (default 150), since test
    // runs end by timeout and the at-exit write never runs then
    CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
        const char* e = getenv("TOTK_PGO_DUMP");
        Sleep((DWORD)(e ? atoi(e) : 150) * 1000);
        int r = __llvm_profile_write_file();
        printf("[pgo] profile written (%d)\n", r);
        fflush(stdout);
        return 0;
    }, nullptr, 0, nullptr);
#endif
    if (const char* w = getenv("TOTK_PROF"))          // TOTK_PROF=<s>[,<s>...]: sample every guest thread for ~3 s at those times
        CreateThread(nullptr, 0, [](LPVOID p) -> DWORD {
            const char* s = (const char*)p;
            DWORD at = 0;
            while (*s) {
                DWORD t = (DWORD)atoi(s);
                if (t > at) Sleep((t - at) * 1000);
                at = t;
                printf("[prof] --- sampling at %lus ---\n", (unsigned long)t);
                profile_threads(1000, 3);
                fflush(stdout);
                while (*s && *s != ',') s++;
                if (*s) s++;
            }
            return 0;
        }, (LPVOID)w, 0, nullptr);
    // The window + Vulkan device live on the main thread (runtime/nvn/vulkan/device.c ui_run: creates them, then pumps
    // window events forever); the game boots on its own thread and its first nvnDeviceInitialize waits for the UI.
    // A watcher ends the process when nnMain returns. TOTK_HEADLESS=1 keeps the old console-only run.
    if (!getenv("TOTK_HEADLESS")) {
        static HANDLE boot; boot = h;
        CreateThread(nullptr, 0, [](LPVOID) -> DWORD { WaitForSingleObject(boot, INFINITE); print_stats(); fflush(stdout); ExitProcess(0); }, nullptr, 0, nullptr);
        return ui_run();
    }
    WaitForSingleObject(h, INFINITE);
    print_stats();
    return 0;
}

// Give a host function a guest code address (after the import thunks) so guest code can call it through a pointer
// (e.g. NVN procs returned by nvnDeviceGetProcAddress).
u64 rt_host_fn_addr(GuestFn fn) {
    static std::atomic<u32> next{0};
    u32 i = next++;
    if (i >= (u32)kMaxHostFns) { fprintf(stderr, "[rt] out of host fn slots\n"); abort(); }
    g_ftab[(IMP_FAKE - MAIN_BASE) / 4 + kMaxImports + i] = fn;
    return IMP_FAKE + 4ull * (kMaxImports + i);
}

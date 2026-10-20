// NVN entry (R3 step 1: measure). nvnBootstrapLoader / nvnDeviceGetProcAddress hand the game a tracing thunk per
// proc name, so a run lists which NVN procs the game uses and in what order. "...Initialize" procs return
// NVN_TRUE so init proceeds; everything else returns 0 until the Vulkan backend implements it.
#include "hle.h"
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <atomic>
#include <array>
#include <utility>
#include <vector>
#include <algorithm>

constexpr int kMaxNvn = 1024;
static std::mutex g_nvn_mtx;
static std::unordered_map<std::string, u64> g_nvn_addr;
static std::string g_nvn_name[kMaxNvn];
static std::atomic<u64> g_nvn_calls[kMaxNvn];
static std::atomic<int> g_nvn_n{0};
static std::atomic<u64> g_nvn_seq{0};
static GuestFn g_nvn_impl[kMaxNvn];

// ---- implemented procs: the NVN layer ported from LASRecomp (runtime/nvn/*.c) registers itself here from static
// constructors; procs it doesn't have stay tracing stubs.
static std::unordered_map<std::string, GuestFn>& impl_table() { static std::unordered_map<std::string, GuestFn> t; return t; }
extern "C" void nvn_register_impl(const char* name, GuestFn fn) { impl_table()[name] = fn; }
static std::atomic<u64> g_nvn_unimpl{0};

static void nvn_dispatch(int i, Ctx* c) {
    const std::string& n = g_nvn_name[i];
    extern bool g_imp_count_all;   // (rt_main.cpp imp_dispatch: full counts only with TOTK_IMP_STATS / the watchdog)
    u64 k = g_nvn_calls[i].load(std::memory_order_relaxed);
    if (k < 2 || g_imp_count_all) k = g_nvn_calls[i].fetch_add(1, std::memory_order_relaxed) + 1;
    else k = 3;
    if (k <= 2) printf("[nvn %5llu] %s(%llx, %llx, %llx, %llx) lr=%llx\n", (unsigned long long)++g_nvn_seq, n.c_str(),
                       c->x[0], c->x[1], c->x[2], c->x[3], c->x[30]);
    const char* prev = t_hle_cur; t_hle_cur = n.c_str();
    if (g_nvn_impl[i]) g_nvn_impl[i](c);
    else {
        c->x[0] = n.find("Initialize") != std::string::npos ? 1 : 0;
        if (k == 1) { ++g_nvn_unimpl; printf("[nvn] UNIMPL %s\n", n.c_str()); }
    }
    t_hle_cur = prev;
}
// Watchdog: the most-called NVN procs so far (per-frame work shows up as counts growing between dumps).
void hle_nvn_dump_top(int top) {
    std::vector<std::pair<u64, int>> v;
    for (int i = 0; i < g_nvn_n.load() && i < kMaxNvn; i++) v.push_back({ g_nvn_calls[i].load(), i });
    std::sort(v.rbegin(), v.rend());
    printf("[nvn] %d procs used, %llu unimplemented; top %d by calls:\n", g_nvn_n.load(), (unsigned long long)g_nvn_unimpl.load(), top);
    for (int k = 0; k < top && k < (int)v.size(); k++)
        printf("[nvn]   %10llu %s%s\n", (unsigned long long)v[k].first, g_nvn_name[v[k].second].c_str(), g_nvn_impl[v[k].second] ? "" : " (UNIMPL)");
}
template <int N> static void nvn_thunk(Ctx* c) { nvn_dispatch(N, c); }
template <int... I> static constexpr std::array<GuestFn, sizeof...(I)> mk(std::integer_sequence<int, I...>) { return { nvn_thunk<I>... }; }
static constexpr auto k_nvn = mk(std::make_integer_sequence<int, kMaxNvn>{});

static void nvn_getproc(Ctx* c);
static u64 proc_addr(const std::string& name) {
    std::lock_guard<std::mutex> l(g_nvn_mtx);
    auto it = g_nvn_addr.find(name);
    if (it != g_nvn_addr.end()) return it->second;
    u64 a;
    if (name == "nvnDeviceGetProcAddress") a = rt_host_fn_addr(nvn_getproc);
    else {
        int i = g_nvn_n++;
        if (i >= kMaxNvn) { fprintf(stderr, "[nvn] too many procs\n"); abort(); }
        g_nvn_name[i] = name; a = rt_host_fn_addr(k_nvn[i]);
        auto im = impl_table().find(name); if (im != impl_table().end()) g_nvn_impl[i] = im->second;
    }
    g_nvn_addr[name] = a;
    return a;
}
static void nvn_getproc(Ctx* c) { c->x[0] = c->x[1] ? proc_addr((const char*)c->x[1]) : 0; }   // (device, name)

HLE(nvnBootstrapLoader) {
    const char* name = (const char*)c->x[0];
    printf("[nvn] bootstrap %s\n", name ? name : "(null)");
    c->x[0] = name ? proc_addr(name) : 0;
}

#include <atomic>
// HLE: nn::fs on host folders. Mount name -> host dir: the rom mount -> extract/romfs (read-only),
// save/cache/temporary storage -> userdata/<kind>. Paths are "<mount>:/a/b".
#include "hle.h"
#include <windows.h>
#include <cstdio>
#include <string>
#include <mutex>
#include <unordered_map>
#include <set>
#include <vector>
#include <filesystem>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>
#include "linux/plat_posix.h"
#endif

namespace fsys = std::filesystem;
#ifdef _WIN32
static const char* kRomDir  = "C:/totk/extract/romfs";
// TOTK_USERDATA=<dir>: saves/cache elsewhere (test runs that start a New Game must not touch the player's saves)
static const char* kUserDir = getenv("TOTK_USERDATA") ? getenv("TOTK_USERDATA") : "C:/totk/userdata";
static const std::string kOverlayDir = "C:/totk/overlay/romfs";   // tools/rstb_slack.py writes here (U9)
// Skip the opening (user 2026-10-15): while on, overlay/skip_opening/romfs (tools/make_skip_opening.py: the opening's
// event flow jumps straight to its own end-of-prologue chain -> Great Sky Island wake-up) is served before the romfs.
// TOTK_SKIP_OPENING=1 starts with it on; F2 / the pad's Minus toggles it on the title screen (hle_hid.cpp).
static const std::string kSkipDir = getenv("TOTK_SKIP_DIR") ? getenv("TOTK_SKIP_DIR") : "C:/totk/overlay/skip_opening/romfs";
#else
// Linux: no drive letters. TOTK_ROMFS = the extracted romfs (default <exe dir>/data/romfs), TOTK_USERDATA = saves / cache
// (default $XDG_DATA_HOME/totk-recomp/userdata), TOTK_OVERLAY = LayeredFS-style replacement files (default <exe dir>/data/overlay/romfs,
// normally absent), TOTK_SKIP_DIR = the skip-opening overlay (default <exe dir>/data/overlay/skip_opening/romfs). See README in the package.
static std::string env_or(const char* name, const std::string& def) { const char* e = getenv(name); return e && *e ? e : def; }
static std::string data_home() {
    const char* x = getenv("XDG_DATA_HOME"); const char* h = getenv("HOME");
    return (x && *x ? std::string(x) : std::string(h && *h ? h : ".") + "/.local/share") + "/totk-recomp";
}
static const std::string kRomStr  = env_or("TOTK_ROMFS", std::string(rt_posix_exe_dir()) + "/data/romfs");
static const std::string kUserStr = env_or("TOTK_USERDATA", data_home() + "/userdata");
static const char* kRomDir  = kRomStr.c_str();
static const char* kUserDir = kUserStr.c_str();
static const std::string kOverlayDir = env_or("TOTK_OVERLAY", std::string(rt_posix_exe_dir()) + "/data/overlay/romfs");
static const std::string kSkipDir = env_or("TOTK_SKIP_DIR", std::string(rt_posix_exe_dir()) + "/data/overlay/skip_opening/romfs");
#endif
extern "C" { volatile int g_skip_opening = 0, g_title_phase = 1, g_in_opening = 0; volatile uint64_t g_in_opening_since = 0; }
static const int skip_init = (g_skip_opening = getenv("TOTK_SKIP_OPENING") && atoi(getenv("TOTK_SKIP_OPENING")) ? 1 : 0);
extern "C" uint64_t nvn_present_count(void);

static const u32 kOk = 0, kPathNotFound = 0x202, kPathExists = 0x402, kOutOfRange = 0x3E02 /* not exact */, kUnsupported = 0x6402;

static std::mutex g_fs_mtx;
static std::unordered_map<std::string, std::string> g_mounts;       // name -> host dir
static bool g_trace_fs = getenv("TOTK_TRACE_FS") != nullptr;
static int g_open_logged;
static std::atomic<u64> g_rd_bytes{0}, g_rd_calls{0}, g_opens_ok{0};

#ifdef _WIN32
static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n ? n - 1 : 0, 0); MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n); return w;
}
// is `p` an existing file (not a directory)? overlay lookups
static bool host_file_exists(const std::string& p) {
    DWORD at = GetFileAttributesW(widen(p).c_str());
    return at != INVALID_FILE_ATTRIBUTES && !(at & FILE_ATTRIBUTE_DIRECTORY);
}
#else
static const std::string& widen(const std::string& s) { return s; }   // (host paths are UTF-8 already)
// Windows file systems ignore case, Linux ones do not: the guest asks for the names the romfs lists, but when a lookup fails
// each path component is matched case-insensitively against its directory (the first hit wins; result cached per path).
static std::mutex g_ci_mtx;
static std::unordered_map<std::string, std::string> g_ci_cache;
static std::string resolve_ci(const std::string& path) {
    {
        std::lock_guard<std::mutex> l(g_ci_mtx);
        auto it = g_ci_cache.find(path);
        if (it != g_ci_cache.end()) return it->second;
    }
    std::string cur = path.size() && path[0] == '/' ? "/" : "";
    size_t i = path.size() && path[0] == '/' ? 1 : 0;
    bool ok = true;
    while (ok && i <= path.size()) {
        size_t j = path.find('/', i); if (j == std::string::npos) j = path.size();
        std::string part = path.substr(i, j - i);
        i = j + 1;
        if (part.empty() || part == ".") continue;
        std::string cand = cur + (cur.empty() || cur.back() == '/' ? "" : "/") + part;
        struct stat st;
        if (stat(cand.c_str(), &st) == 0) { cur = cand; continue; }
        ok = false;
        if (DIR* d = opendir(cur.empty() ? "." : cur.c_str())) {
            while (dirent* e = readdir(d))
                if (!strcasecmp(e->d_name, part.c_str())) { cur = cur + (cur.empty() || cur.back() == '/' ? "" : "/") + e->d_name; ok = true; break; }
            closedir(d);
        }
    }
    std::string r = ok ? cur : path;
    std::lock_guard<std::mutex> l(g_ci_mtx);
    g_ci_cache[path] = r;
    return r;
}
static bool host_file_exists(const std::string& p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0 && stat(resolve_ci(p).c_str(), &st) != 0) return false;
    return S_ISREG(st.st_mode);
}
static void ci_fix(std::string& p) { struct stat st; if (stat(p.c_str(), &st) != 0) p = resolve_ci(p); }
#endif
#ifdef _WIN32
static void ci_fix(std::string&) {}   // (NTFS is case-insensitive)
#endif
static bool host_path(const char* p, std::string* out) {
    std::string s = p; size_t k = s.find(':');
    if (k == std::string::npos) return false;
    std::lock_guard<std::mutex> l(g_fs_mtx);
    auto it = g_mounts.find(s.substr(0, k));
    if (it == g_mounts.end()) return false;
    *out = it->second + s.substr(k + 1);
    static const bool no_overlay = getenv("TOTK_NO_OVERLAY") != nullptr;   // A/B: run on the pristine romfs
    std::string rel = s.substr(k + 1);
    if (it->second == kRomDir && rel.find("Event/EventFlow/OpeningEvent.bfevfl") != std::string::npos) {
        printf("[fs] OpeningEvent opened at present %llu: skip opening %s\n", (unsigned long long)nvn_present_count(), g_skip_opening ? "ON" : "OFF");
        g_title_phase = 0;
    }
    if (it->second == kRomDir && (rel.rfind("/Banc/NormalStage/", 0) == 0 || rel.rfind("/Bake/Scene/", 0) == 0)) {
        // stage files: the under-castle opening is the OpeningField stage (in-game skip button, hle_hid.cpp)
        static std::set<std::string> seen;
        if (seen.insert(rel).second) printf("[fs] stage file %s at present %llu" "\n", rel.c_str(), (unsigned long long)nvn_present_count());
        bool op = rel.find("OpeningField") != std::string::npos;
        // any stage but the title's own (TitleScene_*) ends the title: Continue / Load go straight to MainField without
        // opening OpeningEvent, and the skip-opening banner stayed up in gameplay (user screenshot 2026-10-15 15:50)
        if (g_title_phase && rel.find("TitleScene") == std::string::npos) {
            g_title_phase = 0;
            printf("[fs] title over: stage %s at present %llu\n", rel.c_str(), (unsigned long long)nvn_present_count());
        }
        if (op && !g_in_opening) { g_in_opening = 1; g_in_opening_since = nvn_present_count(); g_title_phase = 0; printf("[fs] entered the opening stage" "\n"); }
        else if (!op && g_in_opening) { g_in_opening = 0; printf("[fs] left the opening stage (%s)" "\n", rel.c_str()); }
    }
    {   // measure what marks the end of the title screen (Continue / Load / New Game)
        static int n = 0;
        if (g_title_phase && n < 40 && (rel.find("Banc/") != std::string::npos || it->second != kRomDir)) {
            n++; printf("[fs] title-phase open %s at present %llu\n", p, (unsigned long long)nvn_present_count());
        }
    }
    if (it->second == kRomDir && g_skip_opening && !no_overlay) {
        std::string sk = kSkipDir + rel;
        if (host_file_exists(sk)) {
            printf("[fs] skip-opening overlay %s\n", sk.c_str());
            *out = sk;
            return true;
        }
    }
    if (it->second == kRomDir && !no_overlay) {    // TOTK_MODS (launcher, 2026-10-20): romfs folders of enabled mods, first listed wins
        // shortcut: replacement files only (a mod's NEW files don't show in directory listings, no RSTB merge); fine for
        // the usual texture / model / param replacements, upgrade when a mod needs added files.
        static const std::vector<std::string> mods = [] {
            std::vector<std::string> v; const char* e = getenv("TOTK_MODS");
#ifdef _WIN32
            const char sep = ';';
#else
            const char sep = ':';
#endif
            for (std::string m = e ? e : ""; !m.empty();) {
                size_t c = m.find(sep); std::string d = m.substr(0, c);
                while (d.size() > 1 && (d.back() == '/' || d.back() == '\\')) d.pop_back();
                if (!d.empty()) { v.push_back(d); printf("[fs] mod %s\n", d.c_str()); }
                if (c == std::string::npos) break;
                m = m.substr(c + 1);
            }
            return v;
        }();
        for (const auto& d : mods) {
            std::string mp = d + rel;
            if (host_file_exists(mp)) {
                static int logged; if (logged++ < 64) printf("[fs] mod file %s\n", mp.c_str());
                *out = mp;
                return true;
            }
        }
    }
    if (it->second == kRomDir && !no_overlay) {    // LayeredFS-style overlay: an existing file overlay/romfs/<path> replaces the romfs one
        std::string ov = kOverlayDir + s.substr(k + 1);
        if (host_file_exists(ov)) {
            static int logged; if (logged++ < 32) printf("[fs] overlay %s\n", ov.c_str());
            *out = ov;
        }
    }
    return true;
}
bool hle_fs_host_path(const char* p, std::string* out) { bool r = host_path(p, out); if (r) ci_fix(*out); return r; }   // hle_movie.cpp (movie files)
static void mount(const char* name, const std::string& dir) {
    fsys::create_directories(widen(dir));
    std::lock_guard<std::mutex> l(g_fs_mtx); g_mounts[name] = dir;
    printf("[fs] mount %s: -> %s\n", name, dir.c_str());
}

// ---- handles
#ifdef _WIN32
typedef HANDLE FH;
static const FH kNoFile = INVALID_HANDLE_VALUE, kClosed = nullptr;
#else
typedef int FH;                                   // file descriptor
static const FH kNoFile = -1, kClosed = -1;
// A file the guest opened for reading. Linux opens a directory with O_RDONLY, Windows CreateFileW refuses: refuse it here too.
static FH fh_open(const std::string& path, int mode) {
    int fl = ((mode & 2) ? O_RDWR : O_RDONLY) | O_CLOEXEC;
    int fd = open(path.c_str(), fl);
    if (fd < 0 && errno == ENOENT) fd = open(resolve_ci(path).c_str(), fl);
    if (fd < 0) return kNoFile;
    struct stat st;
    if (fstat(fd, &st) != 0 || S_ISDIR(st.st_mode)) { close(fd); return kNoFile; }
    return fd;
}
#endif
struct HFile { FH h; int mode; std::string path; };
static std::mutex g_h_mtx;
static std::vector<HFile*> g_files;                         // handle value = index + 1
static HFile* file(u64 v) { std::lock_guard<std::mutex> l(g_h_mtx); return v && v <= g_files.size() ? g_files[v - 1] : nullptr; }

HLE_AS(fs_QueryMountRomCacheSize, "_ZN2nn2fs22QueryMountRomCacheSizeEPm") { *(u64*)c->x[0] = 0x1000; c->x[0] = 0; }
HLE_AS(fs_MountRom, "_ZN2nn2fs8MountRomEPKcPvm") { mount((const char*)c->x[0], kRomDir); c->x[0] = 0; }
HLE_AS(fs_MountSaveData, "_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE") { mount((const char*)c->x[0], std::string(kUserDir) + "/save"); c->x[0] = 0; }
HLE_AS(fs_MountSaveDataReadOnly, "_ZN2nn2fs21MountSaveDataReadOnlyEPKcNS_13ApplicationIdERKNS_7account3UidE") { mount((const char*)c->x[0], std::string(kUserDir) + "/save"); c->x[0] = 0; }
HLE_AS(fs_IsSaveDataExisting, "_ZN2nn2fs18IsSaveDataExistingENS_13ApplicationIdERKNS_7account3UidE") { c->x[0] = 1; }
HLE_AS(fs_MountTemporaryStorage, "_ZN2nn2fs21MountTemporaryStorageEPKc") { mount((const char*)c->x[0], std::string(kUserDir) + "/temp"); c->x[0] = 0; }
HLE_AS(fs_MountCacheStorage, "_ZN2nn2fs17MountCacheStorageEPKci") { mount((const char*)c->x[0], std::string(kUserDir) + "/cache" + std::to_string((int)c->x[1])); c->x[0] = 0; }
HLE_AS(fs_CreateCacheStorage, "_ZN2nn2fs18CreateCacheStorageEill") { c->x[0] = 0; }
HLE_AS(fs_DeleteCacheStorage, "_ZN2nn2fs18DeleteCacheStorageEi") { c->x[0] = 0; }
HLE_AS(fs_OpenCacheStorageList, "_ZN2nn2fs20OpenCacheStorageListEPNS0_22CacheStorageListHandleE") { *(u64*)c->x[0] = 1; c->x[0] = 0; }
HLE_AS(fs_ReadCacheStorageList, "_ZN2nn2fs20ReadCacheStorageListEPiPNS0_16CacheStorageInfoENS0_22CacheStorageListHandleEi") { *(s32*)c->x[0] = 0; c->x[0] = 0; }
HLE_AS(fs_CloseCacheStorageList, "_ZN2nn2fs21CloseCacheStorageListENS0_22CacheStorageListHandleE") { }
HLE_AS(fs_Unmount, "_ZN2nn2fs7UnmountEPKc") { std::lock_guard<std::mutex> l(g_fs_mtx); g_mounts.erase((const char*)c->x[0]); }
HLE_AS(fs_Commit, "_ZN2nn2fs6CommitEPKc") { c->x[0] = 0; }
HLE_AS(fs_SetPriorityOnCurrentThread, "_ZN2nn2fs26SetPriorityOnCurrentThreadENS0_8PriorityE") { }

HLE_AS(fs_OpenFile, "_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci") {
    const char* p = (const char*)c->x[1]; int mode = (int)c->x[2]; std::string hp;
    if (!host_path(p, &hp)) { printf("[fs] OpenFile %s: unknown mount\n", p); c->x[0] = kPathNotFound; return; }
#ifdef _WIN32
    DWORD acc = GENERIC_READ | ((mode & 2) ? GENERIC_WRITE : 0);
    HANDLE h = CreateFileW(widen(hp).c_str(), acc, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
    FH h = fh_open(hp, mode);
#endif
    if (g_trace_fs || g_open_logged < 40 || h == kNoFile) {
        g_open_logged++; printf("[fs] OpenFile %s mode %d%s\n", p, mode, h == kNoFile ? " -> NOT FOUND" : "");
    }
    if (h == kNoFile) { c->x[0] = kPathNotFound; return; }
    g_opens_ok++;
    std::lock_guard<std::mutex> l(g_h_mtx);
    g_files.push_back(new HFile{ h, mode, hp });
    *(u64*)c->x[0] = g_files.size(); c->x[0] = 0;
}
HLE_AS(fs_CloseFile, "_ZN2nn2fs9CloseFileENS0_10FileHandleE") {
#ifdef _WIN32
    HFile* f = file(c->x[0]); if (f && f->h) { CloseHandle(f->h); f->h = nullptr; }
#else
    HFile* f = file(c->x[0]); if (f && f->h != kClosed) { close(f->h); f->h = kClosed; }
#endif
}
// load-progress counters: a line every 64 MiB read (is the loading screen streaming or stalled?)
static void count_read(u64 n) {
    static const u64 t0 = GetTickCount64();
    u64 before = g_rd_bytes.fetch_add(n); g_rd_calls++;
    if ((before >> 26) != ((before + n) >> 26))
        printf("[fs] progress: %llu MiB in %llu reads, %llu opens, t=%.1f s\n", (unsigned long long)((before + n) >> 20),
               (unsigned long long)g_rd_calls.load(), (unsigned long long)g_opens_ok.load(), (GetTickCount64() - t0) / 1000.0);
}
extern "C" unsigned long long hle_fs_read_bytes(void) { return g_rd_bytes.load(std::memory_order_relaxed); }   // ([hitch] log)
static u64 host_pread(HFile* f, s64 off, void* buf, u64 size) {
    u64 done = 0;
    while (done < size) {
#ifdef _WIN32
        OVERLAPPED ov{}; u64 o = (u64)off + done; ov.Offset = (DWORD)o; ov.OffsetHigh = (DWORD)(o >> 32);
        DWORD n = 0, want = (DWORD)std::min<u64>(size - done, 1u << 30);
        if (!ReadFile(f->h, (u8*)buf + done, want, &n, &ov) || n == 0) break;
        done += n;
#else
        size_t want = (size_t)std::min<u64>(size - done, 1u << 30);
        rt_ww_prepare_kernel_write((u8*)buf + done, want);   // a write-watched (read-only) page would give EFAULT, not a fault we handle
        ssize_t n = ::pread(f->h, (u8*)buf + done, want, (off_t)((u64)off + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        done += (u64)n;
#endif
    }
    count_read(done);
    return done;
}
// ReadFile(size_t* out, FileHandle, s64 off, void* buf, size_t size[, ReadOption const&])
static void read_out(Ctx* c) {
    HFile* f = file(c->x[1]); if (!f) { c->x[0] = kUnsupported; return; }
    *(u64*)c->x[0] = host_pread(f, (s64)c->x[2], (void*)c->x[3], c->x[4]); c->x[0] = 0;
}
HLE_AS(fs_ReadFile6, "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvmRKNS0_10ReadOptionE") { read_out(c); }
HLE_AS(fs_ReadFile5, "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm") { read_out(c); }
// ReadFile(FileHandle, s64 off, void* buf, size_t size): must read everything
HLE_AS(fs_ReadFile4, "_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm") {
    HFile* f = file(c->x[0]); if (!f) { c->x[0] = kUnsupported; return; }
    u64 n = host_pread(f, (s64)c->x[1], (void*)c->x[2], c->x[3]);
    c->x[0] = n == c->x[3] ? kOk : kOutOfRange;
}
HLE_AS(fs_WriteFile, "_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE") {
    HFile* f = file(c->x[0]); if (!f) { c->x[0] = kUnsupported; return; }
#ifdef _WIN32
    OVERLAPPED ov{}; u64 o = c->x[1]; ov.Offset = (DWORD)o; ov.OffsetHigh = (DWORD)(o >> 32);
    DWORD n = 0; BOOL ok = WriteFile(f->h, (void*)c->x[2], (DWORD)c->x[3], &n, &ov);
    c->x[0] = ok && n == c->x[3] ? kOk : kUnsupported;
#else
    u64 done = 0, want = c->x[3];
    while (done < want) {
        ssize_t n = ::pwrite(f->h, (const u8*)c->x[2] + done, (size_t)(want - done), (off_t)(c->x[1] + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        done += (u64)n;
    }
    c->x[0] = done == want ? kOk : kUnsupported;
#endif
}
#ifdef _WIN32
HLE_AS(fs_FlushFile, "_ZN2nn2fs9FlushFileENS0_10FileHandleE") { if (HFile* f = file(c->x[0])) FlushFileBuffers(f->h); c->x[0] = 0; }
#else
HLE_AS(fs_FlushFile, "_ZN2nn2fs9FlushFileENS0_10FileHandleE") { if (HFile* f = file(c->x[0])) fsync(f->h); c->x[0] = 0; }
#endif
HLE_AS(fs_GetFileSize, "_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE") {
#ifdef _WIN32
    HFile* f = file(c->x[1]); LARGE_INTEGER s{};
    if (!f || !GetFileSizeEx(f->h, &s)) { c->x[0] = kUnsupported; return; }
    *(s64*)c->x[0] = s.QuadPart; c->x[0] = 0;
#else
    HFile* f = file(c->x[1]); struct stat st;
    if (!f || fstat(f->h, &st) != 0) { c->x[0] = kUnsupported; return; }
    *(s64*)c->x[0] = (s64)st.st_size; c->x[0] = 0;
#endif
}
HLE_AS(fs_SetFileSize, "_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl") {
    HFile* f = file(c->x[0]); if (!f) { c->x[0] = kUnsupported; return; }
#ifdef _WIN32
    FILE_END_OF_FILE_INFO e; e.EndOfFile.QuadPart = (s64)c->x[1];
    c->x[0] = SetFileInformationByHandle(f->h, FileEndOfFileInfo, &e, sizeof e) ? 0 : kUnsupported;
#else
    c->x[0] = ftruncate(f->h, (off_t)c->x[1]) == 0 ? 0 : kUnsupported;
#endif
}

// ---- paths
static u32 fs_err(const std::error_code& ec) { return ec ? (ec == std::errc::file_exists ? kPathExists : kPathNotFound) : kOk; }
HLE_AS(fs_GetEntryType, "_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc") {
    std::string hp; std::error_code ec;
    if (!host_path((const char*)c->x[1], &hp)) { c->x[0] = kPathNotFound; return; }
    ci_fix(hp);
    auto st = fsys::status(widen(hp), ec);
    if (ec || !fsys::exists(st)) { c->x[0] = kPathNotFound; return; }
    *(u32*)c->x[0] = fsys::is_directory(st) ? 0 : 1; c->x[0] = 0;
}
HLE_AS(fs_CreateFile, "_ZN2nn2fs10CreateFileEPKcl") {
    std::string hp; if (!host_path((const char*)c->x[0], &hp)) { c->x[0] = kPathNotFound; return; }
#ifdef _WIN32
    HANDLE h = CreateFileW(widen(hp).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { c->x[0] = GetLastError() == ERROR_FILE_EXISTS ? kPathExists : kPathNotFound; return; }
    FILE_END_OF_FILE_INFO e; e.EndOfFile.QuadPart = (s64)c->x[1]; SetFileInformationByHandle(h, FileEndOfFileInfo, &e, sizeof e);
    CloseHandle(h); c->x[0] = 0;
#else
    int h = open(hp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (h < 0) { c->x[0] = errno == EEXIST ? kPathExists : kPathNotFound; return; }
    if (ftruncate(h, (off_t)c->x[1]) != 0) { /* (size is best effort, like the Windows path) */ }
    close(h); c->x[0] = 0;
#endif
}
HLE_AS(fs_DeleteFile, "_ZN2nn2fs10DeleteFileEPKc") {
    std::string hp; std::error_code ec; if (!host_path((const char*)c->x[0], &hp)) { c->x[0] = kPathNotFound; return; }
    c->x[0] = fsys::remove(widen(hp), ec) ? 0 : kPathNotFound;
}
HLE_AS(fs_RenameFile, "_ZN2nn2fs10RenameFileEPKcS2_") {
    std::string a, b; std::error_code ec;
    if (!host_path((const char*)c->x[0], &a) || !host_path((const char*)c->x[1], &b)) { c->x[0] = kPathNotFound; return; }
    fsys::rename(widen(a), widen(b), ec); c->x[0] = fs_err(ec);
}
HLE_AS(fs_CreateDirectory, "_ZN2nn2fs15CreateDirectoryEPKc") {
    std::string hp; std::error_code ec; if (!host_path((const char*)c->x[0], &hp)) { c->x[0] = kPathNotFound; return; }
    c->x[0] = fsys::create_directory(widen(hp), ec) ? 0 : (ec ? kPathNotFound : kPathExists);
}
HLE_AS(fs_DeleteDirectoryRecursively, "_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc") {
    std::string hp; std::error_code ec; if (!host_path((const char*)c->x[0], &hp)) { c->x[0] = kPathNotFound; return; }
    if (hp.rfind(kRomDir, 0) == 0) { c->x[0] = kUnsupported; return; }      // never touch the extracted romfs
    fsys::remove_all(widen(hp), ec); c->x[0] = fs_err(ec);
}

// ---- directories. nn::fs::DirectoryEntry: name[0x301], pad[3], s8 type @0x304, pad[3], s64 size @0x308 (0x310)
struct HDir { std::vector<std::pair<std::string, std::pair<int, s64>>> ents; size_t pos = 0; };
static std::vector<HDir*> g_dirs;
HLE_AS(fs_OpenDirectory, "_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci") {
    std::string hp; std::error_code ec; int mode = (int)c->x[2];
    if (!host_path((const char*)c->x[1], &hp) || (ci_fix(hp), !fsys::is_directory(widen(hp), ec))) { c->x[0] = kPathNotFound; return; }
    HDir* d = new HDir();
    for (auto& e : fsys::directory_iterator(widen(hp), ec)) {
        bool dir = e.is_directory(ec);
        if ((dir && !(mode & 1)) || (!dir && !(mode & 2))) continue;
        std::u8string nm = e.path().filename().u8string();
        d->ents.push_back({ std::string(nm.begin(), nm.end()), { dir ? 0 : 1, dir ? 0 : (s64)e.file_size(ec) } });
    }
    std::lock_guard<std::mutex> l(g_h_mtx); g_dirs.push_back(d);
    *(u64*)c->x[0] = g_dirs.size(); c->x[0] = 0;
}
HLE_AS(fs_ReadDirectory, "_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl") {
    HDir* d; { std::lock_guard<std::mutex> l(g_h_mtx); u64 v = c->x[2]; d = v && v <= g_dirs.size() ? g_dirs[v - 1] : nullptr; }
    if (!d) { c->x[0] = kUnsupported; return; }
    u8* out = (u8*)c->x[1]; s64 max = (s64)c->x[3], n = 0;
    for (; n < max && d->pos < d->ents.size(); n++, d->pos++) {
        u8* e = out + n * 0x310; memset(e, 0, 0x310);
        auto& [name, ts] = d->ents[d->pos];
        memcpy(e, name.c_str(), std::min<size_t>(name.size(), 0x300));
        e[0x304] = (u8)ts.first; *(s64*)(e + 0x308) = ts.second;
    }
    *(s64*)c->x[0] = n; c->x[0] = 0;
}
HLE_AS(fs_CloseDirectory, "_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE") { }

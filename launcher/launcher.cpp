// TotK Recomp launcher (Steam Deck / Linux): everything that is configuration or preparation lives here, the game itself
// only plays (user 2026-10-19: "keep all the configuration settings and warming inside the launcher and leave in game to
// actually playing it"). SDL3 + Dear ImGui; works with the Deck's controls (Steam Input gamepad), touch, mouse, keyboard.
//
// Files: <state> = ${TOTK_STATE:-${XDG_DATA_HOME:-~/.local/share}/totk-recomp}
//   <state>/launcher.ini          launcher settings (key=value)
//   <state>/recomp_settings.ini   the game's graphics variables (same file the old in-game menu wrote)
//   <state>/backups/              save backups (before each launch + by hand)
//   <state>/cheats/<buildid>.txt  Atmosphere-format cheat files      <state>/mods/<name>/romfs   romfs mods
//   <install>/run.sh, totk, data/  the game (built from the player's own dump; nothing of the game ships with the launcher)
// Test mode (no display needed): launcher --shot out.png [--page N] [--frames N] [--keys "down,down,enter"]
//   with SDL_VIDEODRIVER=offscreen it renders with the software renderer and writes one PNG.

#include <SDL3/SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "imstb_truetype.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using std::string;
using std::vector;

static const char* kVersion = "0.1.0";
static const char* kRepo = "";   // "owner/name" of the GitHub repository once it is published (update check)
static const uint64_t kTitleId = 0x0100F2C0115B6000ull;

// ------------------------------------------------------------------------------------------------ small helpers
static string sq(const string& s) {   // single-quote for sh
    string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}
static string env_or(const char* k, const string& d) { const char* v = getenv(k); return v && *v ? v : d; }
static string now_stamp(const char* fmt = "%Y%m%d-%H%M%S") {
    char b[64]; time_t t = time(nullptr); strftime(b, sizeof b, fmt, localtime(&t)); return b;
}
static string read_file(const string& p, size_t max = 1 << 20) {
    std::ifstream f(p, std::ios::binary); if (!f) return "";
    string s; s.resize(max); f.read(&s[0], max); s.resize(f.gcount()); return s;
}
static string trim(string s) {
    while (!s.empty() && strchr(" \t\r\n", s.back())) s.pop_back();
    size_t i = 0; while (i < s.size() && strchr(" \t\r\n", s[i])) i++;
    return s.substr(i);
}
static string human(double b) {
    const char* u[] = {"B", "KB", "MB", "GB", "TB"}; int i = 0;
    while (b >= 1024 && i < 4) { b /= 1024; i++; }
    char o[32]; snprintf(o, sizeof o, i ? "%.1f %s" : "%.0f %s", b, u[i]); return o;
}
static uint64_t dir_size(const string& p) {
    uint64_t n = 0; std::error_code ec;
    if (!fs::exists(p, ec)) return 0;
    if (fs::is_regular_file(p, ec)) return fs::file_size(p, ec);
    for (auto it = fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec))
        if (!ec && it->is_regular_file(ec)) n += it->file_size(ec);
    return n;
}
static uint64_t free_bytes(const string& p) {
    struct statvfs s; string q = p;
    while (!q.empty() && statvfs(q.c_str(), &s) != 0) q = fs::path(q).parent_path().string();
    if (q.empty()) return 0;
    return (uint64_t)s.f_bavail * s.f_frsize;
}
static string run_capture(const string& cmd) {   // short commands only (blocks)
    string out; FILE* p = popen((cmd + " 2>&1").c_str(), "r"); if (!p) return out;
    char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, p)) > 0) out.append(b, n);
    pclose(p); return out;
}

// ------------------------------------------------------------------------------------------------ paths + config
struct Paths { string exe_dir, state, install, cfg, game_cfg, backups, cheats, mods, tmp, log; } P;

static void init_paths() {
    char b[4096]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    P.exe_dir = n > 0 ? fs::path(string(b, n)).parent_path().string() : ".";
    string home = env_or("HOME", ".");
    P.state = env_or("TOTK_STATE", env_or("XDG_DATA_HOME", home + "/.local/share") + "/totk-recomp");
    P.install = env_or("TOTK_INSTALL", P.exe_dir);
    P.cfg = P.state + "/launcher.ini";
    P.game_cfg = P.state + "/recomp_settings.ini";
    P.backups = P.state + "/backups";
    P.cheats = P.state + "/cheats";
    P.mods = P.state + "/mods";
    P.tmp = P.state + "/tmp";
    P.log = P.state + "/launcher.log";
    for (auto& d : {P.state, P.backups, P.cheats, P.mods, P.tmp, P.state + "/userdata"}) fs::create_directories(d);
}
static FILE* g_log;
static void lg(const char* fmt, ...) {
    if (!g_log) return;
    va_list a; va_start(a, fmt);
    fprintf(g_log, "[%s] ", now_stamp("%H:%M:%S").c_str()); vfprintf(g_log, fmt, a); fputc('\n', g_log); fflush(g_log);
    va_end(a);
}

struct Config {
    std::map<string, string> kv;
    string get(const string& k, const string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    bool on(const string& k, bool d = false) const { auto v = get(k, d ? "1" : "0"); return v == "1" || v == "true"; }
    void set(const string& k, const string& v) { kv[k] = v; dirty = true; }
    bool dirty = false;
    void load() {
        std::ifstream f(P.cfg); string l;
        while (std::getline(f, l)) { l = trim(l); if (l.empty() || l[0] == '#') continue;
            auto e = l.find('='); if (e != string::npos) kv[trim(l.substr(0, e))] = trim(l.substr(e + 1)); }
    }
    void save() {
        if (!dirty) return;
        std::ofstream f(P.cfg + ".tmp");
        f << "# TotK Recomp launcher settings\n";
        for (auto& [k, v] : kv) f << k << "=" << v << "\n";
        f.close(); fs::rename(P.cfg + ".tmp", P.cfg); dirty = false;
    }
} C;

// Deck model + display (auto defaults: user 2026-10-19 "for lcd decks js change target fps to 60")
struct Sys { string product, model; bool deck = false, oled = false, docked = false; int cores = 1; uint64_t ram = 0; } S;
static void detect_system() {
    S.product = trim(read_file("/sys/devices/virtual/dmi/id/product_name", 256));
    S.deck = S.product == "Jupiter" || S.product == "Galileo";
    S.oled = S.product == "Galileo";
    S.model = S.product == "Galileo" ? "Steam Deck OLED" : S.product == "Jupiter" ? "Steam Deck LCD" : (S.product.empty() ? "PC" : S.product);
    std::error_code ec;
    for (auto& e : fs::directory_iterator("/sys/class/drm", ec)) {
        string n = e.path().filename().string();
        if (n.find("-DP-") != string::npos || n.find("-HDMI-") != string::npos)
            if (trim(read_file(e.path().string() + "/status", 64)) == "connected") S.docked = true;
    }
    S.cores = (int)std::thread::hardware_concurrency();
    string mi = read_file("/proc/meminfo", 4096); unsigned long long kb = 0;
    if (sscanf(mi.c_str(), "MemTotal: %llu kB", &kb) == 1) S.ram = kb * 1024;
}

// ------------------------------------------------------------------------------------------------ settings model
struct Opt { const char* val; const char* name; };
struct Setting { const char* key; const char* env; const char* label; const char* help; vector<Opt> opts; bool game_menu; };
// game_menu: one of the variables the game's own settings file carries (recomp_settings.ini, hle_osd.cpp menu_items)
static vector<Setting>& graphics_settings() {
    static vector<Setting> v = {
        {"fps", "TOTK_FPS", "Frame rate", "Frame-rate target. The game keeps normal speed at any rate.",
         {{"60", "60 fps"}, {"90", "90 fps"}, {"unlimited", "Unlimited"}}, true},
        {"dynres", "TOTK_DYNRES", "Dynamic resolution", "Lowers the 3D resolution in heavy scenes to hold the frame rate (the HUD stays sharp).",
         {{"1", "On"}, {"0", "Off"}}, true},
        {"res_scale", "TOTK_RES_SCALE", "Render resolution", "3D resolution relative to the game's own (720p handheld).",
         {{"0.67", "67 %"}, {"0.8", "80 %"}, {"0.9", "90 %"}, {"1", "100 % (native)"}, {"1.5", "150 %"}, {"2", "200 %"}}, true},
        {"handheld", "TOTK_HANDHELD", "Game mode", "Handheld = 720p rendering and handheld detail (right for the Deck). Docked = 900p, more detail, much heavier.",
         {{"1", "Handheld"}, {"0", "Docked"}}, false},
        {"ssaa", "TOTK_SSAA", "3D upscaler", "How the 3D image is scaled to the output.", {{"", "Bicubic"}, {"fetch", "FSR 1 (the game's)"}}, true},
        {"upscale", "TOTK_UPSCALE", "Window upscaler", "Scaling of the final picture to the screen.", {{"", "Bilinear"}, {"fsr", "FSR 1"}}, true},
        {"sharpen", "TOTK_SHARPEN", "Sharpening", "FSR 1 RCAS sharpening.", {{"", "Off"}, {"1", "Low"}, {"0.5", "Medium"}, {"0.2", "High"}}, true},
        {"fullscreen", "TOTK_FULLSCREEN", "Display", "Fullscreen or a window (desktop mode).", {{"1", "Fullscreen"}, {"0", "Window"}}, false},
        {"fps_overlay", "TOTK_FPS_OVERLAY", "FPS counter", "Frame-rate readout in the top-left corner.", {{"0", "Off"}, {"1", "On"}}, false},
        {"ingame_menu", "TOTK_GFX_MENU", "In-game graphics menu", "The old L3 + R3 overlay menu. Off: all settings are here in the launcher.",
         {{"0", "Off"}, {"1", "On"}}, false},
    };
    return v;
}
static vector<Setting>& control_settings() {
    static vector<Setting> v = {
        {"pad_labels", "TOTK_PAD_LABELS", "Button layout", "Nintendo = buttons by name (A right, B bottom), Xbox = by position.",
         {{"", "Nintendo (by name)"}, {"1", "Xbox (by position)"}}, false},
        {"gyro", "TOTK_GYRO", "Motion controls", "Gyro aiming from the Deck / controller.", {{"", "On"}, {"0", "Off"}}, false},
        {"rumble", "TOTK_RUMBLE", "Rumble strength", "Controller vibration gain.", {{"0", "Off"}, {"0.5", "Low"}, {"", "Normal"}, {"2", "Strong"}, {"4", "Max"}}, false},
        {"volume", "TOTK_VOLUME", "Volume", "Game audio volume.", {{"0.5", "50 %"}, {"0.75", "75 %"}, {"", "100 %"}, {"1.5", "150 %"}, {"2", "200 %"}}, false},
        {"skip_opening", "TOTK_SKIP_OPENING", "Skip opening", "New games start on Great Sky Island instead of the opening under the castle.",
         {{"", "Off"}, {"1", "On"}}, false},
    };
    return v;
}
static string recommended(const string& key) {
    if (key == "fps") return S.deck && !S.oled ? "60" : "90";
    if (key == "dynres") return "1";
    if (key == "res_scale") return "1";
    if (key == "handheld") return "1";
    if (key == "fullscreen") return "1";
    if (key == "fps_overlay") return "0";
    if (key == "ingame_menu") return "0";
    return "";
}
static string setting_val(const Setting& s) { return C.get(s.key, recommended(s.key)); }
static string setting_env_val(const Setting& s) {
    string v = setting_val(s);
    if (string(s.key) == "fps" && v == "unlimited") return "120";   // + TOTK_UNLOCK_FPS=1 (pacing off; 120 = the in-game menu's top value, dynres target)
    return v;
}

// ------------------------------------------------------------------------------------------------ background jobs
struct Job {
    string title, cmd;
    std::atomic<bool> running{false}, done{false};
    std::atomic<int> status{0};
    std::mutex mu;
    std::deque<string> lines;
    pid_t pid = 0;
    std::function<void(Job&)> on_done;
    void push(const string& l) { std::lock_guard<std::mutex> g(mu); lines.push_back(l); if (lines.size() > 400) lines.pop_front(); }
    string last() { std::lock_guard<std::mutex> g(mu); return lines.empty() ? "" : lines.back(); }
};
static Job g_job;   // one long task at a time (extraction, warm-up, build, report)

static void job_start(const string& title, const string& cmd, std::function<void(Job&)> on_done = nullptr) {
    if (g_job.running) return;
    g_job.title = title; g_job.cmd = cmd; g_job.on_done = on_done;
    { std::lock_guard<std::mutex> g(g_job.mu); g_job.lines.clear(); }
    g_job.running = true; g_job.done = false; g_job.status = 0;
    lg("job start: %s: %s", title.c_str(), cmd.c_str());
    std::thread([] {
        int fds[2]; if (pipe(fds) != 0) { g_job.running = false; return; }
        pid_t pid = fork();
        if (pid == 0) {
            setpgid(0, 0);
            dup2(fds[1], 1); dup2(fds[1], 2); close(fds[0]); close(fds[1]);
            execl("/bin/bash", "bash", "-c", g_job.cmd.c_str(), (char*)nullptr);
            _exit(127);
        }
        g_job.pid = pid;
        close(fds[1]);
        FILE* f = fdopen(fds[0], "r"); char b[4096]; string part;
        while (fgets(b, sizeof b, f)) {
            string l(b);
            // progress lines that end in \r: keep the latest
            size_t cr; while ((cr = l.find('\r')) != string::npos && cr + 1 < l.size()) l = l.substr(cr + 1);
            l = trim(l); if (!l.empty()) g_job.push(l);
        }
        fclose(f);
        int st = 0; waitpid(pid, &st, 0);
        g_job.status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
        lg("job end: %s: status %d", g_job.title.c_str(), (int)g_job.status);
        g_job.running = false; g_job.done = true;
        if (g_job.on_done) g_job.on_done(g_job);
    }).detach();
}
static void job_cancel() { if (g_job.running && g_job.pid > 0) kill(-g_job.pid, SIGTERM); }

// ------------------------------------------------------------------------------------------------ the game dump
// XCI (HFS0 root -> secure partition) and NSP (PFS0) parsing: enough to find the meta (.cnmt) and control NCAs.
struct PFile { string name; uint64_t off, size; };
static bool read_at(FILE* f, uint64_t off, void* b, size_t n) { return fseeko(f, (off_t)off, SEEK_SET) == 0 && fread(b, 1, n, f) == n; }
static vector<PFile> parse_hfs0(FILE* f, uint64_t base) {   // HFS0 or PFS0 at base
    vector<PFile> v; char hdr[16];
    if (!read_at(f, base, hdr, 16)) return v;
    bool hfs = !memcmp(hdr, "HFS0", 4), pfs = !memcmp(hdr, "PFS0", 4);
    if (!hfs && !pfs) return v;
    uint32_t n, st; memcpy(&n, hdr + 4, 4); memcpy(&st, hdr + 8, 4);
    if (n > 4096 || st > (1 << 20)) return v;
    size_t es = hfs ? 0x40 : 0x18;
    vector<char> e(n * es), s(st);
    if (!read_at(f, base + 16, e.data(), e.size()) || !read_at(f, base + 16 + e.size(), s.data(), st)) return v;
    uint64_t data = base + 16 + e.size() + st;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t off, size; uint32_t no;
        memcpy(&off, &e[i * es], 8); memcpy(&size, &e[i * es + 8], 8); memcpy(&no, &e[i * es + 16], 4);
        if (no >= st) continue;
        v.push_back({string(&s[no], strnlen(&s[no], st - no)), data + off, size});
    }
    return v;
}
static vector<PFile> dump_files(const string& path, string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "can't open the file"; return {}; }
    vector<PFile> out; char magic[4] = {};
    if (read_at(f, 0x100, magic, 4) && !memcmp(magic, "HEAD", 4)) {
        uint64_t root; read_at(f, 0x130, &root, 8);
        for (auto& p : parse_hfs0(f, root)) if (p.name == "secure") out = parse_hfs0(f, p.off);
        if (out.empty()) err = "XCI without a readable secure partition";
    } else if (read_at(f, 0, magic, 4) && !memcmp(magic, "PFS0", 4)) {
        out = parse_hfs0(f, 0);
    } else err = "not an XCI or NSP file";
    fclose(f);
    return out;
}
static bool copy_range(const string& src, uint64_t off, uint64_t size, const string& dst) {
    FILE* f = fopen(src.c_str(), "rb"); if (!f) return false;
    FILE* o = fopen(dst.c_str(), "wb"); if (!o) { fclose(f); return false; }
    vector<char> b(1 << 20); bool ok = fseeko(f, (off_t)off, SEEK_SET) == 0;
    while (ok && size) { size_t n = (size_t)std::min<uint64_t>(size, b.size()); ok = fread(b.data(), 1, n, f) == n && fwrite(b.data(), 1, n, o) == n; size -= n; }
    fclose(f); fclose(o); return ok;
}
static string hactool_path() {
    for (auto& p : {P.install + "/tools/hactool", P.exe_dir + "/tools/hactool", P.exe_dir + "/hactool"})
        if (access(p.c_str(), X_OK) == 0) return p;
    string w = trim(run_capture("command -v hactool"));
    return w;
}
// hactool aborts on lines that are not "name = hex" keys; prod.keys from other tools has extras (same filter as build_from_xci.sh)
static string filtered_keys(string keys) {
    std::error_code ec;
    if (fs::is_directory(keys, ec))   // a keys folder (emulators keep prod.keys in one): its prod.keys, else the first *.keys
        for (auto& e : fs::directory_iterator(keys, ec))
            if (e.path().filename() == "prod.keys" || (e.path().extension() == ".keys" && !fs::exists(keys + "/prod.keys"))) { keys = e.path().string(); break; }
    if (!fs::is_regular_file(keys, ec)) return "";
    std::ifstream f(keys); string l, out; std::regex re("^\\s*[A-Za-z0-9_]+\\s*=\\s*([0-9A-Fa-f]{32}|[0-9A-Fa-f]{64})\\s*$");
    while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (std::regex_match(l, re)) out += l + "\n"; }
    string p = P.tmp + "/keys.filtered";
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600); if (fd < 0) return "";
    if (write(fd, out.data(), out.size()) < 0) {} close(fd);
    if (out.empty()) { fs::remove(p, ec); return ""; }
    return p;
}

struct DumpInfo {
    bool ok = false; string err;
    uint64_t title = 0; uint32_t version = 0; string display_version, name, icon_path, kind;
    uint64_t size = 0;
} g_dump;
static std::atomic<bool> g_dump_busy{false};

// known releases: cnmt version -> display version; only 1.0.0 is the build the recompiled code was made from
struct Known { uint32_t ver; const char* disp; bool supported; };
static const Known kKnown[] = {{0, "1.0.0", true}, {65536, "1.1.0", false}, {131072, "1.1.1", false}, {196608, "1.1.2", false},
                               {262144, "1.2.0", false}, {327680, "1.2.1", false}, {393216, "1.4.0", false}, {458752, "1.4.1", false},
                               {524288, "1.4.2", false}};
static const Known* known_ver(uint32_t v) { for (auto& k : kKnown) if (k.ver == v) return &k; return nullptr; }

static void inspect_dump(const string& path, const string& keys) {
    if (g_dump_busy.exchange(true)) return;
    std::thread([path, keys] {
        DumpInfo d; std::error_code ec;
        d.size = fs::file_size(path, ec);
        d.kind = fs::path(path).extension().string();
        string err; auto files = dump_files(path, err);
        if (files.empty()) { d.err = err.empty() ? "no files in the dump" : err; g_dump = d; g_dump_busy = false; return; }
        string hac = hactool_path();
        if (hac.empty()) { d.err = "hactool is missing from the launcher's tools folder"; g_dump = d; g_dump_busy = false; return; }
        string kf = filtered_keys(keys);
        if (kf.empty()) { d.err = "no usable keys in " + keys + " (pick your prod.keys file)"; lg("inspect: %s", d.err.c_str()); g_dump = d; g_dump_busy = false; return; }
        string work = P.tmp + "/inspect"; fs::remove_all(work, ec); fs::create_directories(work);
        bool meta_found = false;
        for (auto& f : files) {
            if (f.name.size() < 9 || f.name.substr(f.name.size() - 9) != ".cnmt.nca" || f.size > (64 << 20)) continue;
            meta_found = true;
            copy_range(path, f.off, f.size, work + "/meta.nca");
            string o = run_capture(sq(hac) + " -k " + sq(kf) + " -t nca --section0dir=" + sq(work + "/cnmt") + " " + sq(work + "/meta.nca"));
            for (auto& e : fs::directory_iterator(work + "/cnmt", ec)) {
                string c = read_file(e.path().string(), 1 << 16);
                if (c.size() < 0x20) continue;
                uint64_t tid; uint32_t ver; memcpy(&tid, &c[0], 8); memcpy(&ver, &c[8], 4);
                uint8_t type = (uint8_t)c[0xC];
                if (type != 0x80) continue;   // application
                d.title = tid; d.version = ver;
                uint16_t ext, cnt; memcpy(&ext, &c[0xE], 2); memcpy(&cnt, &c[0x10], 2);
                for (int i = 0; i < cnt; i++) {   // content records: hash[32] id[16] size[6] type[1] idoff[1]
                    size_t r = 0x20 + ext + (size_t)i * 0x38;
                    if (r + 0x38 > c.size()) break;
                    if ((uint8_t)c[r + 0x36] != 3) continue;   // control
                    char id[33]; for (int k = 0; k < 16; k++) snprintf(id + 2 * k, 3, "%02x", (uint8_t)c[r + 32 + k]);
                    for (auto& cf : files) if (cf.name == string(id) + ".nca") {
                        copy_range(path, cf.off, cf.size, work + "/control.nca");
                        run_capture(sq(hac) + " -k " + sq(kf) + " -t nca --romfsdir=" + sq(work + "/control") + " " + sq(work + "/control.nca"));
                    }
                }
            }
            if (d.title) break;
        }
        // NACP: name (AmericanEnglish = entry 0, 0x200 + 0x100 bytes each), display version at 0x3060
        string nacp = read_file(work + "/control/control.nacp", 0x4000);
        if (nacp.size() >= 0x3070) {
            d.name = string(nacp.data(), strnlen(nacp.data(), 0x200));
            d.display_version = string(nacp.data() + 0x3060, strnlen(nacp.data() + 0x3060, 16));
        }
        string icon = work + "/control/icon_AmericanEnglish.dat";
        if (d.title == kTitleId && fs::exists(icon, ec)) {   // (another game's icon is not ours to show)
            fs::copy_file(icon, P.state + "/game_icon.jpg", fs::copy_options::overwrite_existing, ec); d.icon_path = P.state + "/game_icon.jpg"; }
        fs::remove(kf, ec);
        fs::remove_all(work, ec);
        if (!meta_found) d.err = "no metadata (.cnmt.nca) in the dump";
        else if (!d.title) d.err = "couldn't decrypt the metadata: are the keys right for this dump?";
        else d.ok = true;
        lg("inspect %s: ok %d title %016llx ver %u disp '%s' err '%s'", path.c_str(), d.ok, (unsigned long long)d.title, d.version,
             d.display_version.c_str(), d.err.c_str());
        g_dump = d; g_dump_busy = false;
    }).detach();
}

// ------------------------------------------------------------------------------------------------ install state
struct Install { bool game = false, romfs = false, img = false; string build_level, romfs_dir; uint64_t cache = 0; } I;
static void scan_install() {
    I.game = access((P.install + "/totk").c_str(), X_OK) == 0 && access((P.install + "/run.sh").c_str(), R_OK) == 0;
    I.romfs_dir = env_or("TOTK_ROMFS", P.install + "/data/romfs");
    I.romfs = fs::exists(I.romfs_dir + "/System/Resource/ResourceSizeTable.Product.100.rsizetable.zs");
    I.img = fs::exists(P.install + "/data/img/main.bin");
    I.build_level = trim(read_file(P.install + "/BUILD_LEVEL", 400));
    if (I.build_level.size() > 160) I.build_level = I.build_level.substr(0, 157) + "...";
}

// ------------------------------------------------------------------------------------------------ saves
static string save_dir() { return P.state + "/userdata/save"; }
static void prune_backups(int keep) {
    vector<fs::path> v; std::error_code ec;
    for (auto& e : fs::directory_iterator(P.backups, ec)) if (e.path().extension() == ".gz") v.push_back(e.path());
    std::sort(v.begin(), v.end());
    while ((int)v.size() > keep) { fs::remove(v.front(), ec); v.erase(v.begin()); }
}
static string backup_saves(const string& why) {
    std::error_code ec;
    if (!fs::exists(save_dir(), ec) || fs::is_empty(save_dir(), ec)) return "";
    string out = P.backups + "/save-" + now_stamp() + "-" + why + ".tar.gz";
    string r = run_capture("tar -C " + sq(P.state + "/userdata") + " -czf " + sq(out) + " save && echo OK");
    lg("backup %s: %s", out.c_str(), trim(r).c_str());
    prune_backups(std::max(3, atoi(C.get("backups_keep", "10").c_str())));
    return r.find("OK") != string::npos ? out : "";
}
// a save folder from the Switch (JKSV), Eden / yuzu / Ryujinx: the folder that holds slot_00 / progress files
static string find_save_root(const string& pick) {
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(pick, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it.depth() > 6) { it.disable_recursion_pending(); continue; }
        string n = it->path().filename().string();
        if (n.rfind("slot_0", 0) == 0 || n == "option.sav" || n == "progress.sav") return it->path().parent_path().string();
    }
    return "";
}

// ------------------------------------------------------------------------------------------------ cheats + mods
struct Cheat { string name; bool master; };
struct CheatFile { string path, build; vector<Cheat> cheats; };
static vector<CheatFile> g_cheats;
static void scan_cheats() {
    g_cheats.clear(); std::error_code ec;
    for (auto& e : fs::directory_iterator(P.cheats, ec)) {
        if (e.path().extension() != ".txt") continue;
        CheatFile cf; cf.path = e.path().string(); cf.build = e.path().stem().string();
        std::ifstream f(cf.path); string l;
        while (std::getline(f, l)) {
            l = trim(l);
            if (l.size() > 2 && (l[0] == '[' || l[0] == '{')) cf.cheats.push_back({l.substr(1, l.size() - 2), l[0] == '{'});
        }
        g_cheats.push_back(cf);
    }
}
static vector<string> g_mods;
static void scan_mods() {
    g_mods.clear(); std::error_code ec;
    for (auto& e : fs::directory_iterator(P.mods, ec)) if (e.is_directory()) g_mods.push_back(e.path().filename().string());
    std::sort(g_mods.begin(), g_mods.end());
}
static string mod_romfs(const string& m) {   // <mod>/romfs, or <mod> itself when it holds the romfs folders directly
    string r = P.mods + "/" + m + "/romfs";
    return fs::exists(r) ? r : P.mods + "/" + m;
}

// ------------------------------------------------------------------------------------------------ launching the game
static std::atomic<bool> g_game_running{false};
static std::atomic<int> g_game_status{-1};
static string g_last_launch_msg;

static vector<std::pair<string, string>> game_env(bool warmup) {
    vector<std::pair<string, string>> e;
    for (auto* list : {&graphics_settings(), &control_settings()})
        for (auto& s : *list) e.push_back({s.env, setting_env_val(s)});
    if (setting_val(graphics_settings()[0]) == "unlimited") e.push_back({"TOTK_UNLOCK_FPS", "1"});
    e.push_back({"TOTK_SETTINGS_FILE", P.game_cfg});
    // cheats: the enabled entries of each file -> one active file the runtime reads
    string active = P.state + "/run/cheats_active.txt"; std::ofstream cf(active); int n = 0;
    for (auto& f : g_cheats) {
        std::ifstream in(f.path); string l; bool keep = false;
        while (std::getline(in, l)) {
            string t = trim(l);
            if (t.size() > 2 && (t[0] == '[' || t[0] == '{')) {
                string nm = t.substr(1, t.size() - 2);
                keep = t[0] == '{' || C.on("cheat." + f.build + "." + nm);
                if (keep) { cf << t << "\n"; if (t[0] == '[') n++; }
            } else if (keep && !t.empty()) cf << t << "\n";
        }
    }
    cf.close();
    if (n) e.push_back({"TOTK_CHEATS", active});
    string mods;
    for (auto& m : g_mods) if (C.on("mod." + m)) mods += (mods.empty() ? "" : ":") + mod_romfs(m);
    if (!mods.empty()) e.push_back({"TOTK_MODS", mods});
    if (warmup) e.push_back({"TOTK_WARMUP", "1"});
    return e;
}
static void write_game_settings() {   // same file + keys the game reads at start (hle_osd.cpp SettingsLoad)
    std::ofstream f(P.game_cfg);
    f << "# TotK Recomp graphics settings, written by the launcher. Read when the game starts.\n";
    for (auto& s : graphics_settings()) if (s.game_menu) f << s.env << "=" << setting_env_val(s) << "\n";
}
static void launch_game(bool warmup, std::function<void()> on_exit) {
    if (g_game_running) return;
    C.save(); write_game_settings();
    if (!warmup && C.on("backup_before_launch", true)) backup_saves("auto");
    auto env = game_env(warmup);
    string runsh = P.install + "/run.sh";
    lg("launch %s (warmup %d)", runsh.c_str(), warmup);
    g_game_running = true;
    std::thread([env, runsh, on_exit] {
        pid_t pid = fork();
        if (pid == 0) {
            for (auto& [k, v] : env) { if (v.empty()) unsetenv(k.c_str()); else setenv(k.c_str(), v.c_str(), 1); }
            execl("/bin/bash", "bash", runsh.c_str(), (char*)nullptr);
            _exit(127);
        }
        int st = 0; waitpid(pid, &st, 0);
        g_game_status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
        lg("game exited: %d", (int)g_game_status);
        g_game_running = false;
        if (on_exit) on_exit();
    }).detach();
}

// ------------------------------------------------------------------------------------------------ Steam shortcut + art
// Artwork is generated here from the icon in the player's own dump (nothing of the game ships with the launcher).
struct Img {
    int w = 0, h = 0; vector<uint8_t> px;   // RGBA
    Img() {}
    Img(int w_, int h_) : w(w_), h(h_), px((size_t)w_ * h_ * 4, 0) {}
    uint8_t* at(int x, int y) { return &px[((size_t)y * w + x) * 4]; }
};
static Img img_load(const string& p) {
    Img o; int n; unsigned char* d = stbi_load(p.c_str(), &o.w, &o.h, &n, 4);
    if (d) { o.px.assign(d, d + (size_t)o.w * o.h * 4); stbi_image_free(d); }
    return o;
}
static Img img_scale(const Img& s, int w, int h) {   // bilinear
    Img o(w, h);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float fx = (x + 0.5f) * s.w / w - 0.5f, fy = (y + 0.5f) * s.h / h - 0.5f;
        int x0 = std::clamp((int)floorf(fx), 0, s.w - 1), y0 = std::clamp((int)floorf(fy), 0, s.h - 1);
        int x1 = std::min(x0 + 1, s.w - 1), y1 = std::min(y0 + 1, s.h - 1);
        float ax = std::clamp(fx - x0, 0.f, 1.f), ay = std::clamp(fy - y0, 0.f, 1.f);
        for (int c = 0; c < 4; c++) {
            auto g = [&](int xx, int yy) { return (float)s.px[((size_t)yy * s.w + xx) * 4 + c]; };
            float v = (g(x0, y0) * (1 - ax) + g(x1, y0) * ax) * (1 - ay) + (g(x0, y1) * (1 - ax) + g(x1, y1) * ax) * ay;
            o.px[((size_t)y * w + x) * 4 + c] = (uint8_t)std::clamp(v + 0.5f, 0.f, 255.f);
        }
    }
    return o;
}
static void img_blur(Img& m, int r, int passes) {   // box blur
    vector<uint8_t> t(m.px.size());
    for (int p = 0; p < passes; p++) for (int dir = 0; dir < 2; dir++) {
        int W = m.w, H = m.h;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int s[4] = {}, n = 0;
            for (int k = -r; k <= r; k++) {
                int xx = dir ? x : std::clamp(x + k, 0, W - 1), yy = dir ? std::clamp(y + k, 0, H - 1) : y;
                const uint8_t* q = &m.px[((size_t)yy * W + xx) * 4];
                for (int c = 0; c < 4; c++) s[c] += q[c];
                n++;
            }
            for (int c = 0; c < 4; c++) t[((size_t)y * W + x) * 4 + c] = (uint8_t)(s[c] / n);
        }
        m.px.swap(t);
    }
}
static void img_blit(Img& d, const Img& s, int ox, int oy, float alpha = 1.f) {
    for (int y = 0; y < s.h; y++) for (int x = 0; x < s.w; x++) {
        int X = ox + x, Y = oy + y; if (X < 0 || Y < 0 || X >= d.w || Y >= d.h) continue;
        const uint8_t* q = &s.px[((size_t)y * s.w + x) * 4]; uint8_t* p = d.at(X, Y);
        float a = q[3] / 255.f * alpha;
        for (int c = 0; c < 3; c++) p[c] = (uint8_t)(p[c] * (1 - a) + q[c] * a);
        p[3] = (uint8_t)std::max<float>(p[3], a * 255);
    }
}
static void img_shade(Img& d, float top, float bottom) {   // vertical darkening gradient
    for (int y = 0; y < d.h; y++) { float k = top + (bottom - top) * y / std::max(1, d.h - 1);
        for (int x = 0; x < d.w; x++) { uint8_t* p = d.at(x, y); for (int c = 0; c < 3; c++) p[c] = (uint8_t)(p[c] * k); } }
}
static vector<uint8_t> g_font_bold;
static void img_text(Img& d, const string& t, float px, int cx, int cy, uint32_t rgb, bool center = true) {
    if (g_font_bold.empty()) return;
    stbtt_fontinfo fi; if (!stbtt_InitFont(&fi, g_font_bold.data(), 0)) return;
    float sc = stbtt_ScaleForPixelHeight(&fi, px);
    int asc, desc, gap; stbtt_GetFontVMetrics(&fi, &asc, &desc, &gap);
    float w = 0; for (char ch : t) { int adv, lsb; stbtt_GetCodepointHMetrics(&fi, (unsigned char)ch, &adv, &lsb); w += adv * sc; }
    float x = center ? cx - w / 2 : cx; int base = cy + (int)(asc * sc / 2);
    for (size_t i = 0; i < t.size(); i++) {
        int cp = (unsigned char)t[i], adv, lsb, x0, y0, x1, y1;
        stbtt_GetCodepointHMetrics(&fi, cp, &adv, &lsb);
        stbtt_GetCodepointBitmapBox(&fi, cp, sc, sc, &x0, &y0, &x1, &y1);
        int bw = x1 - x0, bh = y1 - y0;
        if (bw > 0 && bh > 0) {
            vector<uint8_t> g((size_t)bw * bh);
            stbtt_MakeCodepointBitmap(&fi, g.data(), bw, bh, bw, sc, sc, cp);
            for (int yy = 0; yy < bh; yy++) for (int xx = 0; xx < bw; xx++) {
                int X = (int)x + x0 + xx, Y = base + y0 + yy; if (X < 0 || Y < 0 || X >= d.w || Y >= d.h) continue;
                float a = g[(size_t)yy * bw + xx] / 255.f; uint8_t* p = d.at(X, Y);
                uint8_t col[3] = {(uint8_t)(rgb >> 16), (uint8_t)(rgb >> 8), (uint8_t)rgb};
                for (int c = 0; c < 3; c++) p[c] = (uint8_t)(p[c] * (1 - a) + col[c] * a);
                p[3] = (uint8_t)std::max<float>(p[3], a * 255);
            }
        }
        x += adv * sc;
        if (i + 1 < t.size()) x += stbtt_GetCodepointKernAdvance(&fi, cp, (unsigned char)t[i + 1]) * sc;
    }
}
static Img art_background(const Img& icon, int w, int h) {
    Img bg = img_scale(icon, 64, 64); img_blur(bg, 6, 3);
    Img o = img_scale(bg, w, h);
    img_shade(o, 0.55f, 0.25f);
    return o;
}
// writes <dir>/{p,header,hero,logo,icon}.png; returns "" or an error
static string make_artwork(const string& dir) {
    Img icon = img_load(P.state + "/game_icon.jpg");
    if (!icon.w) return "no game icon yet: check the game dump in Setup first";
    fs::create_directories(dir);
    const char* title = "TEARS OF THE KINGDOM";
    {   Img o = art_background(icon, 600, 900);
        img_blit(o, img_scale(icon, 420, 420), 90, 150);
        img_text(o, title, 40, 300, 660, 0xF2E6C8); img_text(o, "PC RECOMPILATION", 26, 300, 715, 0x9FD8C8);
        stbi_write_png((dir + "/p.png").c_str(), o.w, o.h, 4, o.px.data(), o.w * 4); }
    {   Img o = art_background(icon, 920, 430);
        img_blit(o, img_scale(icon, 330, 330), 50, 50);
        img_text(o, title, 46, 410, 180, 0xF2E6C8, false); img_text(o, "PC RECOMPILATION", 28, 412, 240, 0x9FD8C8, false);
        stbi_write_png((dir + "/header.png").c_str(), o.w, o.h, 4, o.px.data(), o.w * 4); }
    {   Img o = art_background(icon, 1920, 620);
        Img big = img_scale(icon, 900, 900); img_blur(big, 3, 1);
        for (int y = 0; y < big.h; y++) for (int x = 0; x < 360; x++) big.at(x, y)[3] = (uint8_t)(big.at(x, y)[3] * x / 360);   // fade in from the left
        img_blit(o, big, 1020, -140, 0.35f);
        stbi_write_png((dir + "/hero.png").c_str(), o.w, o.h, 4, o.px.data(), o.w * 4); }
    {   Img o(1280, 360);   // transparent logo
        img_text(o, title, 96, 640, 140, 0xF2E6C8); img_text(o, "PC RECOMPILATION", 52, 640, 260, 0x9FD8C8);
        stbi_write_png((dir + "/logo.png").c_str(), o.w, o.h, 4, o.px.data(), o.w * 4); }
    {   Img o = img_scale(icon, 256, 256); stbi_write_png((dir + "/icon.png").c_str(), o.w, o.h, 4, o.px.data(), o.w * 4); }
    return "";
}

// Steam: the non-Steam shortcut is added while Steam runs (steamos-add-to-steam / steam://addnonsteamgame), then the
// artwork goes to every Steam user's grid folder under the id Steam gave the shortcut, then the player restarts Steam.
static string steam_root() {
    string h = env_or("HOME", ".");
    for (auto& p : {h + "/.local/share/Steam", h + "/.steam/steam", h + "/.steam/root"}) if (fs::exists(p + "/userdata")) return p;
    return "";
}
// binary VDF shortcuts.vdf: find each shortcut's appid + exe + name (enough to find ours)
struct Shortcut { uint32_t appid = 0; string name, exe, user; };
static vector<Shortcut> read_shortcuts() {
    vector<Shortcut> out; string root = steam_root(); if (root.empty()) return out; std::error_code ec;
    for (auto& u : fs::directory_iterator(root + "/userdata", ec)) {
        string d = read_file(u.path().string() + "/config/shortcuts.vdf", 8 << 20);
        size_t i = 0; Shortcut cur; bool in = false;
        auto rd_str = [&](size_t& k) { size_t e = d.find('\0', k); string s = d.substr(k, e - k); k = e + 1; return s; };
        while (i < d.size()) {
            uint8_t t = (uint8_t)d[i++];
            if (t == 8) { if (in && (cur.appid || !cur.exe.empty())) { cur.user = u.path().filename().string(); out.push_back(cur); } cur = {}; in = false; continue; }
            if (i >= d.size()) break;
            string key = rd_str(i);
            if (t == 0) { if (key.size() && isdigit((unsigned char)key[0])) in = true; continue; }
            if (t == 1) { string v = rd_str(i); std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                if (key == "appname") cur.name = v; else if (key == "exe") cur.exe = v; }
            else if (t == 2) { uint32_t v = 0; if (i + 4 <= d.size()) memcpy(&v, &d[i], 4); i += 4;
                string k = key; std::transform(k.begin(), k.end(), k.begin(), ::tolower); if (k == "appid") cur.appid = v; }
            else if (t == 7) i += 8;
        }
    }
    return out;
}
static vector<Shortcut> our_shortcuts() {
    vector<Shortcut> v; string me = P.exe_dir + "/launcher";
    for (auto& s : read_shortcuts()) if (s.exe.find(me) != string::npos) v.push_back(s);
    return v;
}
static string install_artwork(const vector<Shortcut>& sc) {
    string art = P.state + "/artwork"; string e = make_artwork(art); if (!e.empty()) return e;
    string root = steam_root(); int n = 0; std::error_code ec;
    for (auto& s : sc) {
        if (!s.appid) continue;
        string g = root + "/userdata/" + s.user + "/config/grid"; fs::create_directories(g, ec);
        string id = std::to_string(s.appid);
        string mark = P.state + "/steam_art_" + id;   // this launcher wrote this shortcut's art before (a file: C is not thread-safe)
        bool ours = fs::exists(mark, ec);
        auto cp = [&](const char* f, const string& to) {
            if (fs::exists(g + "/" + to, ec) && !ours) { lg("artwork %s exists, kept", to.c_str()); return; }
            fs::copy_file(art + "/" + f, g + "/" + to, fs::copy_options::overwrite_existing, ec);
        };
        cp("p.png", id + "p.png"); cp("header.png", id + ".png"); cp("hero.png", id + "_hero.png"); cp("logo.png", id + "_logo.png");
        cp("icon.png", id + "_icon.png");
        FILE* mf = fopen(mark.c_str(), "w"); if (mf) fclose(mf);
        n++;
    }
    lg("artwork for %d shortcut(s)", n);
    return n ? "" : "Steam has not saved the new shortcut yet";
}
static string g_steam_msg;
static std::atomic<int> g_steam_state{0};   // 0 idle, 1 adding, 2 added (restart needed), 3 error
static void add_to_steam() {
    if (g_steam_state == 1) return;
    g_steam_state = 1; g_steam_msg = "Adding to Steam...";
    std::thread([] {
        string me = P.exe_dir + "/launcher";
        // a .desktop entry gives the shortcut its name and icon
        string art = P.state + "/artwork"; make_artwork(art);
        string desk = P.state + "/TotK-Recomp.desktop";
        { std::ofstream f(desk);
          f << "[Desktop Entry]\nType=Application\nName=Tears of the Kingdom Recomp\nExec=\"" << me << "\"\nPath=" << P.exe_dir
            << "\nIcon=" << art << "/icon.png\nTerminal=false\nCategories=Game;\n"; }
        chmod(desk.c_str(), 0755);
        auto before = our_shortcuts();
        string r;
        if (access("/usr/bin/steamos-add-to-steam", X_OK) == 0) r = run_capture("/usr/bin/steamos-add-to-steam " + sq(desk));
        else r = run_capture("steam steam://addnonsteamgame/" + sq(desk));
        lg("add to steam: %s", trim(r).c_str());
        // Steam writes shortcuts.vdf a moment later
        vector<Shortcut> now;
        for (int i = 0; i < 40; i++) { now = our_shortcuts(); if (now.size() > before.size() || (!now.empty() && i > 6)) break; usleep(500000); }
        string e = now.empty() ? "Steam did not save the shortcut (is Steam running?)" : install_artwork(now);
        if (e.empty()) { g_steam_msg = "Added. Restart Steam to see the artwork."; g_steam_state = 2; }
        else { g_steam_msg = e; g_steam_state = 3; }
    }).detach();
}
static void restart_steam() {
    lg("restart steam");
    // Game Mode restarts the client by itself after a shutdown; on the desktop start it again
    string mode = env_or("XDG_CURRENT_DESKTOP", "");
    if (getenv("SteamDeck") || getenv("GAMESCOPE_WAYLAND_DISPLAY") || mode == "gamescope") { int r = system("steam -shutdown >/dev/null 2>&1 &"); (void)r; }
    else { int r = system("(steam -shutdown; sleep 8; setsid steam >/dev/null 2>&1 &) >/dev/null 2>&1 &"); (void)r; }
}

// ------------------------------------------------------------------------------------------------ report + cleanup + update
static string make_report() {
    string dir = P.tmp + "/report-" + now_stamp(); fs::create_directories(dir); std::error_code ec;
    for (auto& f : {P.state + "/run/totk.log", P.state + "/run/totk.prev.log", P.log, P.cfg, P.game_cfg})
        if (fs::exists(f, ec)) fs::copy_file(f, dir + "/" + fs::path(f).filename().string(), fs::copy_options::overwrite_existing, ec);
    {   std::ofstream s(dir + "/system.txt");
        s << "launcher " << kVersion << "\nmodel " << S.model << " (" << S.product << ") docked " << S.docked << "\n";
        s << "build " << I.build_level << "\ndump " << g_dump.display_version << " v" << g_dump.version << "\n\n";
        s << run_capture("uname -a; cat /etc/os-release; free -m; df -h " + sq(P.state) + "; lscpu | head -20");
        s << run_capture("vulkaninfo --summary 2>/dev/null | head -40"); }
    {   std::ofstream t(dir + "/ISSUE_TEMPLATE.md");
        t << "## What happened\n\n## What you expected\n\n## Steps to reproduce\n1.\n\n## Where in the game\n(area, what Link was doing)\n\n"
             "## Attached\nThis archive: logs, settings, system info. It contains no game files and no keys.\n"; }
    string home = env_or("HOME", ".");
    string outdir = fs::exists(home + "/Desktop") ? home + "/Desktop" : home;
    string out = outdir + "/totk-recomp-report-" + now_stamp() + ".tar.gz";
    run_capture("tar -C " + sq(P.tmp) + " -czf " + sq(out) + " " + sq(fs::path(dir).filename().string()));
    fs::remove_all(dir, ec);
    lg("report %s", out.c_str());
    return out;
}
struct CleanItem { string label, path; uint64_t size = 0; bool danger = false; };
static vector<CleanItem> g_clean; static std::atomic<bool> g_clean_busy{false};
static void scan_cleanup() {
    if (g_clean_busy.exchange(true)) return;
    std::thread([] {
        vector<CleanItem> v = {
            {"Shader cache (rebuilt by the warm-up or while playing)", P.state + "/run/cache"},
            {"Old logs", P.state + "/run/totk.prev.log"},
            {"Launcher temporary files", P.tmp},
            {"Save backups (keeps your current save)", P.backups, 0, true},
        };
        for (auto& c : v) c.size = dir_size(c.path);
        g_clean = v; g_clean_busy = false;
    }).detach();
}
static string g_update_msg;
static void check_update() {
    if (!*kRepo) { g_update_msg = "No release repository configured in this build."; return; }
    g_update_msg = "Checking...";
    std::thread([] {
        string r = run_capture(string("curl -fsSL --max-time 8 https://api.github.com/repos/") + kRepo + "/releases/latest");
        std::smatch m; std::regex re("\"tag_name\"\\s*:\\s*\"([^\"]+)\"");
        if (std::regex_search(r, m, re)) { string t = m[1]; g_update_msg = t == string("v") + kVersion || t == kVersion ? "Up to date (" + t + ")." : "New version available: " + t; }
        else g_update_msg = "Couldn't reach GitHub.";
    }).detach();
}

// ------------------------------------------------------------------------------------------------ UI
static SDL_Window* g_win; static SDL_Renderer* g_ren;
static ImFont *F_body, *F_small, *F_h1, *F_h2, *F_title;
static SDL_Texture* g_icon_tex; static int g_icon_w, g_icon_h; static string g_icon_loaded;
static int g_page = 0;
enum { PG_HOME, PG_SETUP, PG_GRAPHICS, PG_CONTROLS, PG_CHEATS, PG_MODS, PG_SAVES, PG_STEAM, PG_TOOLS, PG_ABOUT, PG_N };
static const char* kPages[PG_N] = {"Home", "Setup", "Graphics", "Controls", "Cheats", "Mods", "Saves", "Steam", "Tools", "About"};
static float g_scale = 1.f;
static bool g_gamescope;   // Game Mode: one window at a time, so the launcher hides while the game runs (default)

static const ImVec4 COL_ACCENT(0.36f, 0.78f, 0.70f, 1), COL_GOLD(0.95f, 0.85f, 0.55f, 1), COL_WARN(0.98f, 0.70f, 0.30f, 1),
    COL_BAD(0.95f, 0.40f, 0.38f, 1), COL_OK(0.45f, 0.85f, 0.50f, 1), COL_DIM(0.62f, 0.66f, 0.70f, 1);

static void style() {
    ImGuiStyle& s = ImGui::GetStyle();
    ImGui::StyleColorsDark(&s);
    s.WindowRounding = 0; s.FrameRounding = 10; s.GrabRounding = 10; s.ChildRounding = 14; s.PopupRounding = 12;
    s.FramePadding = ImVec2(16, 10); s.ItemSpacing = ImVec2(12, 12); s.WindowPadding = ImVec2(24, 20); s.ScrollbarSize = 18;
    s.ScrollbarRounding = 9; s.FrameBorderSize = 0; s.WindowBorderSize = 0;
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.082f, 0.098f, 1); c[ImGuiCol_ChildBg] = ImVec4(0.10f, 0.12f, 0.14f, 1);
    c[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.13f, 0.15f, 0.98f);
    c[ImGuiCol_FrameBg] = ImVec4(0.15f, 0.18f, 0.21f, 1); c[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.25f, 0.28f, 1);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.30f, 0.32f, 1);
    c[ImGuiCol_Button] = ImVec4(0.16f, 0.20f, 0.23f, 1); c[ImGuiCol_ButtonHovered] = ImVec4(0.22f, 0.34f, 0.35f, 1);
    c[ImGuiCol_ButtonActive] = ImVec4(0.26f, 0.48f, 0.46f, 1);
    c[ImGuiCol_Header] = ImVec4(0.18f, 0.30f, 0.31f, 1); c[ImGuiCol_HeaderHovered] = ImVec4(0.22f, 0.38f, 0.38f, 1);
    c[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.48f, 0.46f, 1);
    c[ImGuiCol_CheckMark] = COL_ACCENT; c[ImGuiCol_SliderGrab] = COL_ACCENT; c[ImGuiCol_SliderGrabActive] = COL_GOLD;
    c[ImGuiCol_NavCursor] = COL_GOLD; c[ImGuiCol_Separator] = ImVec4(0.2f, 0.24f, 0.27f, 1);
    c[ImGuiCol_Text] = ImVec4(0.92f, 0.94f, 0.95f, 1); c[ImGuiCol_TextDisabled] = COL_DIM;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.6f);
}
static void load_fonts_file() {   // the artwork's text font (stb_truetype)
    string b = read_file(P.exe_dir + "/launcher-data/Inter-Bold.ttf", 8 << 20); g_font_bold.assign(b.begin(), b.end());
}
static void load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    string d = P.exe_dir + "/launcher-data/";
    auto ld = [&](const char* f, float sz) -> ImFont* {
        string p = d + f; if (!fs::exists(p)) return nullptr;
        return io.Fonts->AddFontFromFileTTF(p.c_str(), sz * g_scale);
    };
    F_body = ld("Inter-Medium.ttf", 22); if (!F_body) F_body = io.Fonts->AddFontDefault();
    F_small = ld("Inter-Regular.ttf", 18); if (!F_small) F_small = F_body;
    F_h2 = ld("Inter-SemiBold.ttf", 26); if (!F_h2) F_h2 = F_body;
    F_h1 = ld("Inter-Bold.ttf", 34); if (!F_h1) F_h1 = F_body;
    F_title = ld("Inter-ExtraBold.ttf", 44); if (!F_title) F_title = F_h1;
    load_fonts_file();
}
static void load_icon_tex() {
    string p = P.state + "/game_icon.jpg";
    if (g_icon_loaded == p && g_icon_tex) return;
    if (!fs::exists(p)) return;
    Img m = img_load(p); if (!m.w) return;
    if (g_icon_tex) SDL_DestroyTexture(g_icon_tex);
    g_icon_tex = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, m.w, m.h);
    SDL_UpdateTexture(g_icon_tex, nullptr, m.px.data(), m.w * 4);
    g_icon_w = m.w; g_icon_h = m.h; g_icon_loaded = p;
}

static void text_wrapped_col(const ImVec4& c, const char* fmt, ...) {
    va_list a; va_start(a, fmt); char b[2048]; vsnprintf(b, sizeof b, fmt, a); va_end(a);
    ImGui::PushStyleColor(ImGuiCol_Text, c); ImGui::TextWrapped("%s", b); ImGui::PopStyleColor();
}
static void heading(const char* t, const char* sub = nullptr) {
    ImGui::PushFont(F_h1); ImGui::TextUnformatted(t); ImGui::PopFont();
    if (sub) { ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "%s", sub); ImGui::PopFont(); }
    ImGui::Dummy(ImVec2(0, 6));
}
static bool big_button(const char* label, const ImVec4& col, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_Button, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(col.x * 1.15f, col.y * 1.15f, col.z * 1.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(col.x * 0.85f, col.y * 0.85f, col.z * 0.85f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.05f, 0.07f, 0.08f, 1));
    ImGui::PushFont(F_h2);
    bool r = ImGui::Button(label, size);
    ImGui::PopFont(); ImGui::PopStyleColor(4);
    return r;
}
static void chip(const char* label, const ImVec4& c) {
    ImDrawList* dl = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::PushFont(F_small); ImVec2 ts = ImGui::CalcTextSize(label);
    float room = ImGui::GetContentRegionAvail().x - 8;
    if (ts.x + 28 > room && ImGui::GetCursorPosX() > ImGui::GetStyle().WindowPadding.x + 1) { ImGui::NewLine(); p = ImGui::GetCursorScreenPos(); room = ImGui::GetContentRegionAvail().x - 8; }
    ImVec2 sz(std::min(ts.x + 28, room), ts.y + 10);
    ImGui::PushClipRect(p, ImVec2(p.x + sz.x, p.y + sz.y), true);
    dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), ImGui::GetColorU32(ImVec4(c.x, c.y, c.z, 0.16f)), sz.y / 2);
    dl->AddCircleFilled(ImVec2(p.x + 12, p.y + sz.y / 2), 4, ImGui::GetColorU32(c));
    dl->AddText(ImVec2(p.x + 22, p.y + 5), ImGui::GetColorU32(c), label);
    ImGui::PopClipRect();
    ImGui::Dummy(sz); ImGui::PopFont();
}
// one option row: label + help on the left, choice buttons on the right (works with the d-pad: left/right moves between choices)
static bool option_row(const Setting& s) {
    bool changed = false; string cur = setting_val(s);
    ImGui::PushID(s.key);
    ImGui::BeginGroup();
    ImGui::PushFont(F_h2); ImGui::TextUnformatted(s.label); ImGui::PopFont();
    ImGui::PushFont(F_small); ImGui::PushStyleColor(ImGuiCol_Text, COL_DIM);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x); ImGui::TextUnformatted(s.help); ImGui::PopTextWrapPos();
    ImGui::PopStyleColor(); ImGui::PopFont();
    for (size_t i = 0; i < s.opts.size(); i++) {
        bool sel = cur == s.opts[i].val;
        if (i) ImGui::SameLine();
        if (sel) { ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.52f, 0.49f, 1)); ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1)); }
        if (ImGui::Button(s.opts[i].name)) { C.set(s.key, s.opts[i].val); changed = true; }
        if (sel) ImGui::PopStyleColor(2);
    }
    bool rec = cur == recommended(s.key);
    if (!recommended(s.key).empty() && !rec) { ImGui::SameLine(); ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "recommended: %s",
        [&] { for (auto& o : s.opts) if (recommended(s.key) == o.val) return o.name; return ""; }()); ImGui::PopFont(); }
    ImGui::EndGroup();
    ImGui::PopID();
    ImGui::Dummy(ImVec2(0, 6));
    return changed;
}

// ---- file browser (Game Mode has no file dialogs): big rows, d-pad friendly
struct Browser {
    bool open = false; string title, dir, target; vector<string> exts; vector<fs::directory_entry> items; bool want_dir = false;
    std::function<void(const string&)> pick;
    void refresh() {
        items.clear(); std::error_code ec;
        for (auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
            string n = e.path().filename().string(); if (n.empty() || n[0] == '.') continue;
            bool d = e.is_directory(ec);
            if (!d && !want_dir) {
                string x = e.path().extension().string(); std::transform(x.begin(), x.end(), x.begin(), ::tolower);
                bool ok = exts.empty(); for (auto& w : exts) if (x == w || n == w) ok = true;
                if (!ok) continue;
            } else if (!d) continue;
            items.push_back(e);
        }
        std::sort(items.begin(), items.end(), [](auto& a, auto& b) {
            bool da = a.is_directory(), db = b.is_directory(); if (da != db) return da;
            return a.path().filename().string() < b.path().filename().string(); });
    }
    void show(const string& t, const string& start, vector<string> e, bool dirs, std::function<void(const string&)> cb) {
        title = t; exts = e; want_dir = dirs; pick = cb; std::error_code ec;
        dir = fs::is_directory(start, ec) ? start : (fs::is_directory(fs::path(start).parent_path(), ec) ? fs::path(start).parent_path().string() : env_or("HOME", "/"));
        refresh(); open = true; ImGui::OpenPopup("##browser");
    }
    void draw() {
        if (!open) return;
        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x * 0.86f, io.DisplaySize.y * 0.86f));
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), 0, ImVec2(0.5f, 0.5f));
        if (!ImGui::IsPopupOpen("##browser")) ImGui::OpenPopup("##browser");
        if (ImGui::BeginPopupModal("##browser", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::PushFont(F_h2); ImGui::TextUnformatted(title.c_str()); ImGui::PopFont();
            ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "%s", dir.c_str()); ImGui::PopFont();
            string home = env_or("HOME", "/");
            auto place = [&](const char* n, const string& p) { std::error_code ec; if (!fs::is_directory(p, ec)) return;
                if (ImGui::Button(n)) { dir = p; refresh(); } ImGui::SameLine(); };
            if (ImGui::Button("Up")) { dir = fs::path(dir).has_parent_path() && dir != "/" ? fs::path(dir).parent_path().string() : "/"; refresh(); }
            ImGui::SameLine(); place("Home", home); place("Downloads", home + "/Downloads"); place("Desktop", home + "/Desktop");
            std::error_code ec;
            for (auto& m : {string("/run/media"), string("/run/media/") + env_or("USER", "deck")})
                if (fs::is_directory(m, ec)) for (auto& e : fs::directory_iterator(m, ec))
                    if (e.is_directory(ec) && e.path().filename() != env_or("USER", "deck")) place(("SD/USB: " + e.path().filename().string()).c_str(), e.path().string());
            ImGui::NewLine();
            float footer = ImGui::GetFrameHeightWithSpacing() * 1.3f;
            ImGui::BeginChild("list", ImVec2(0, -footer), true);
            if (items.empty()) text_wrapped_col(COL_DIM, want_dir ? "No folders here." : "No matching files here.");
            for (auto& e : items) {
                bool d = e.is_directory(ec); string n = e.path().filename().string();
                string lab = (d ? "[folder]  " : "") + n;
                if (!d) { lab += "   (" + human((double)e.file_size(ec)) + ")"; }
                if (ImGui::Selectable(lab.c_str(), false, 0, ImVec2(0, 44 * g_scale))) {
                    if (d) { dir = e.path().string(); refresh(); break; }
                    else { pick(e.path().string()); open = false; ImGui::CloseCurrentPopup(); break; }
                }
            }
            ImGui::EndChild();
            if (want_dir) { if (ImGui::Button("Use this folder")) { pick(dir); open = false; ImGui::CloseCurrentPopup(); } ImGui::SameLine(); }
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight) || ImGui::IsKeyPressed(ImGuiKey_Escape)) { open = false; ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }
    }
} B;

// ---- confirm dialog
struct Confirm { bool open = false; string text; std::function<void()> yes; } CF;
static void confirm(const string& t, std::function<void()> yes) { CF.open = true; CF.text = t; CF.yes = yes; }
static void draw_confirm() {
    if (!CF.open) return;
    if (!ImGui::IsPopupOpen("##confirm")) ImGui::OpenPopup("##confirm");
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), 0, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x * 0.55f, 0));
    if (ImGui::BeginPopupModal("##confirm", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
        ImGui::TextWrapped("%s", CF.text.c_str()); ImGui::Dummy(ImVec2(0, 8));
        if (big_button("Yes", COL_ACCENT, ImVec2(180 * g_scale, 0))) { CF.open = false; ImGui::CloseCurrentPopup(); if (CF.yes) CF.yes(); }
        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(180 * g_scale, 0)) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight) || ImGui::IsKeyPressed(ImGuiKey_Escape)) { CF.open = false; ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

// ---- pages
static string dump_status_line(ImVec4& col) {
    if (C.get("dump").empty()) { col = COL_DIM; return "No game dump selected"; }
    if (g_dump_busy) { col = COL_DIM; return "Reading the game dump..."; }
    if (!g_dump.ok) { col = g_dump.err.empty() ? COL_DIM : COL_BAD; return g_dump.err.empty() ? "Not checked yet" : g_dump.err; }
    auto k = known_ver(g_dump.version);
    string v = g_dump.display_version.empty() ? (k ? k->disp : "unknown version") : g_dump.display_version;
    if (g_dump.title != kTitleId) {
        char t[96]; snprintf(t, sizeof t, "Not Tears of the Kingdom: %s%s(%016llX)", g_dump.name.c_str(), g_dump.name.empty() ? "" : " ",
                             (unsigned long long)g_dump.title);
        col = COL_BAD; return t;
    }
    if (k && k->supported) { col = COL_OK; return "Version " + v + " - supported"; }
    col = COL_WARN; return "Version " + v + " - not supported: the recompiled code is made from 1.0.0, this may not work";
}
static void page_home() {
    ImGuiIO& io = ImGui::GetIO();
    float W = ImGui::GetContentRegionAvail().x;
    ImGui::BeginChild("hero", ImVec2(W, 300 * g_scale), true, ImGuiWindowFlags_NoScrollbar);
    {
        ImDrawList* dl = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetWindowPos(), s = ImGui::GetWindowSize();
        dl->AddRectFilledMultiColor(p, ImVec2(p.x + s.x, p.y + s.y), IM_COL32(18, 40, 44, 255), IM_COL32(30, 32, 52, 255),
                                    IM_COL32(16, 18, 26, 255), IM_COL32(14, 30, 30, 255));
        float isz = s.y - 48;
        load_icon_tex();
        if (g_icon_tex) { ImGui::SetCursorPos(ImVec2(24, 24)); ImGui::Image((ImTextureID)(intptr_t)g_icon_tex, ImVec2(isz, isz)); }
        else {
            ImVec2 a(p.x + 24, p.y + 24); dl->AddRectFilled(a, ImVec2(a.x + isz, a.y + isz), IM_COL32(40, 60, 64, 255), 16);
            dl->AddText(F_h1, F_h1->FontSize, ImVec2(a.x + isz / 2 - 30, a.y + isz / 2 - 20), IM_COL32(120, 200, 190, 255), "TotK");
        }
        ImGui::SetCursorPos(ImVec2(isz + 56, 34));
        ImGui::BeginGroup();
        ImGui::PushFont(F_title); ImGui::TextColored(COL_GOLD, "Tears of the Kingdom"); ImGui::PopFont();
        ImGui::PushFont(F_h2); ImGui::TextColored(COL_ACCENT, "PC Recompilation  -  %s", S.model.c_str()); ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 10));
        ImVec4 dc; string ds = dump_status_line(dc);
        chip(I.game ? "Game build ready" : "Game not built", I.game ? COL_OK : COL_WARN); ImGui::SameLine();
        chip(I.romfs ? "Game files found" : "Game files missing", I.romfs ? COL_OK : COL_BAD); ImGui::SameLine();
        chip(ds.c_str(), dc);
        ImGui::Dummy(ImVec2(0, 14));
        bool can = I.game && I.romfs && !g_game_running && !g_job.running;
        ImGui::BeginDisabled(!can);
        if (big_button(g_game_running ? "Running..." : "PLAY", COL_GOLD, ImVec2(260 * g_scale, 64 * g_scale)) ||
            (can && ImGui::IsKeyPressed(ImGuiKey_GamepadStart))) {
            g_last_launch_msg = "";
            launch_game(false, [] { g_last_launch_msg = g_game_status == 0 ? "Game closed." : "The game ended with status " + std::to_string(g_game_status) + ". Tools > Report a problem collects the log."; });
            if (C.on("close_on_play", g_gamescope)) SDL_HideWindow(g_win);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Settings", ImVec2(180 * g_scale, 64 * g_scale))) g_page = PG_GRAPHICS;
        ImGui::EndGroup();
    }
    ImGui::EndChild();
    if (!I.game || !I.romfs) {
        text_wrapped_col(COL_WARN, "%s", !I.game ? "The game has not been built on this device yet. Open Setup to pick your game dump and keys."
                                                 : "The game files (romfs) were not found. Open Setup to extract them from your dump.");
        if (ImGui::Button("Open Setup")) g_page = PG_SETUP;
    }
    if (!g_last_launch_msg.empty()) text_wrapped_col(COL_DIM, "%s", g_last_launch_msg.c_str());
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::Columns(2, nullptr, false);
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("This session"); ImGui::PopFont();
    auto& g = graphics_settings();
    auto name_of = [&](const Setting& s) { string v = setting_val(s); for (auto& o : s.opts) if (v == o.val) return string(o.name); return v; };
    ImGui::Text("Frame rate: %s", name_of(g[0]).c_str());
    ImGui::Text("Dynamic resolution: %s", name_of(g[1]).c_str());
    ImGui::Text("Resolution: %s, %s", name_of(g[2]).c_str(), name_of(g[3]).c_str());
    int nc = 0; for (auto& f : g_cheats) for (auto& c : f.cheats) if (!c.master && C.on("cheat." + f.build + "." + c.name)) nc++;
    int nm = 0; for (auto& m : g_mods) if (C.on("mod." + m)) nm++;
    ImGui::Text("Cheats on: %d    Mods on: %d", nc, nm);
    ImGui::NextColumn();
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Shaders"); ImGui::PopFont();
    ImGui::PushFont(F_small);
    text_wrapped_col(COL_DIM, "Run the shader warm-up once after installing or updating: it compiles every shader so the game doesn't stutter while you play. Press A on its screen to start playing at once; it keeps compiling in the background.");
    ImGui::PopFont();
    ImGui::BeginDisabled(!I.game || !I.romfs || g_game_running || g_job.running);
    if (ImGui::Button("Shader warm-up")) launch_game(true, [] { g_last_launch_msg = "Shader warm-up closed (finished shaders are kept)."; });
    ImGui::EndDisabled();
    ImGui::Columns(1);
    (void)io;
}
static void step_header(int n, const char* title, bool done) {   // numbered disc, filled green with a tick when done
    ImGui::PushFont(F_h2);
    ImDrawList* dl = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos();
    float r = ImGui::GetTextLineHeight() * 0.5f; ImVec2 c(p.x + r, p.y + r);
    if (done) {
        dl->AddCircleFilled(c, r, ImGui::GetColorU32(COL_OK));
        ImU32 k = IM_COL32(16, 22, 24, 255);
        dl->AddLine(ImVec2(c.x - r * 0.45f, c.y), ImVec2(c.x - r * 0.1f, c.y + r * 0.4f), k, r * 0.22f);
        dl->AddLine(ImVec2(c.x - r * 0.1f, c.y + r * 0.4f), ImVec2(c.x + r * 0.5f, c.y - r * 0.35f), k, r * 0.22f);
    } else {
        dl->AddCircle(c, r - 1, ImGui::GetColorU32(COL_GOLD), 0, 2.f);
        char d[8]; snprintf(d, sizeof d, "%d", n); ImVec2 ts = ImGui::CalcTextSize(d);
        dl->AddText(ImVec2(c.x - ts.x / 2, c.y - ts.y / 2), ImGui::GetColorU32(COL_GOLD), d);
    }
    ImGui::Dummy(ImVec2(r * 2, r * 2)); ImGui::SameLine(0, r);
    ImGui::TextColored(done ? COL_OK : ImVec4(0.92f, 0.94f, 0.95f, 1), "%s", title);
    ImGui::PopFont();
}
static void page_setup() {
    heading("Setup", "Everything is made on this device from your own cartridge dump and keys. Nothing from the game is downloaded or shipped. "
                     "You can leave and come back: each finished step is remembered.");
    string dump = C.get("dump"), keys = C.get("keys");
    // 1. dump
    step_header(1, "Your game dump (.xci or .nsp)", !dump.empty() && fs::exists(dump));
    ImGui::Indent();
    ImGui::TextWrapped("%s", dump.empty() ? "Not selected. Pick the file, or drag it onto this window (desktop mode)." : dump.c_str());
    if (ImGui::Button("Choose dump...")) B.show("Choose your Tears of the Kingdom dump", dump.empty() ? env_or("HOME", "/") : dump, {".xci", ".nsp"}, false,
        [](const string& p) { C.set("dump", p); C.save(); if (!C.get("keys").empty()) inspect_dump(p, C.get("keys")); });
    ImGui::Unindent();
    // 2. keys
    step_header(2, "Your prod.keys", !keys.empty() && fs::exists(keys));
    ImGui::Indent();
    ImGui::TextWrapped("%s", keys.empty() ? "Not selected. Dumped from your own console (Lockpick)." : keys.c_str());
    if (ImGui::Button("Choose prod.keys...")) B.show("Choose your prod.keys", keys.empty() ? env_or("HOME", "/") + "/.switch" : keys, {".keys", "prod.keys"}, false,
        [](const string& p) { C.set("keys", p); C.save(); if (!C.get("dump").empty()) inspect_dump(C.get("dump"), p); });
    ImGui::Unindent();
    // 3. check
    ImVec4 dc; string ds = dump_status_line(dc);
    step_header(3, "Check the game version", g_dump.ok && g_dump.title == kTitleId);
    ImGui::Indent();
    text_wrapped_col(dc, "%s", ds.c_str());
    ImGui::BeginDisabled(dump.empty() || keys.empty() || g_dump_busy);
    if (ImGui::Button("Check now")) inspect_dump(dump, keys);
    ImGui::EndDisabled();
    ImGui::PushFont(F_small);
    text_wrapped_col(COL_DIM, "Compatible: 1.0.0 (the cartridge version). Other versions can be used but may not work.");
    ImGui::PopFont();
    ImGui::Unindent();
    // 4. space
    uint64_t fr = free_bytes(P.install);
    bool space_ok = fr > 40ull << 30 || I.romfs;
    step_header(4, "Disk space", space_ok);
    ImGui::Indent();
    text_wrapped_col(space_ok ? COL_DIM : COL_WARN, "Free on this drive: %s. Extracting the game files needs about 40 GB at the peak (17 GB afterwards).",
                     human((double)fr).c_str());
    ImGui::Unindent();
    // 5. game files
    step_header(5, "Game files (romfs)", I.romfs);
    ImGui::Indent();
    ImGui::TextWrapped("%s", I.romfs ? ("Found at " + I.romfs_dir).c_str() : "Not extracted yet.");
    ImGui::BeginDisabled(I.romfs || !g_dump.ok || g_job.running || !fs::exists(P.install + "/tools/extract_romfs.sh"));
    if (ImGui::Button("Extract from the dump")) {
        string kf = filtered_keys(keys);
        job_start("Extracting game files", "bash " + sq(P.install + "/tools/extract_romfs.sh") + " " + sq(dump) + " " + sq(kf) + " " + sq(P.install + "/data") +
                  " " + sq(hactool_path()) + "; s=$?; rm -f " + sq(kf) + "; exit $s", [](Job&) { scan_install(); });
    }
    ImGui::EndDisabled();
    ImGui::Unindent();
    // 6. build
    step_header(6, "Game build", I.game);
    ImGui::Indent();
    if (I.game) text_wrapped_col(COL_DIM, "Installed: %s", I.build_level.empty() ? "yes" : I.build_level.c_str());
    else text_wrapped_col(COL_WARN, "This package does not contain a game build yet. The on-device builder (fast code generator) is still in development; "
                                    "until it ships, the build runs from build_from_xci.sh on a PC.");
    ImGui::Unindent();
    if (g_job.running || g_job.done) {
        ImGui::Separator();
        ImGui::PushFont(F_h2); ImGui::Text("%s%s", g_job.title.c_str(), g_job.running ? "..." : g_job.status == 0 ? ": done" : ": failed"); ImGui::PopFont();
        ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "%s", g_job.last().c_str()); ImGui::PopFont();
        if (g_job.running && ImGui::Button("Stop")) job_cancel();
    }
}
static void page_graphics() {
    heading("Graphics", ("Defaults are set for your " + S.model + (S.docked ? " (external display connected)" : "") + ". Changes apply the next time you press Play.").c_str());
    if (ImGui::Button("Use recommended settings")) for (auto& s : graphics_settings()) C.set(s.key, recommended(s.key));
    ImGui::Dummy(ImVec2(0, 6));
    for (auto& s : graphics_settings()) option_row(s);
}
static void page_controls() {
    heading("Controls", "Controller and audio options. Button remapping for the Deck's controls is done in Steam (controller settings for this game).");
    for (auto& s : control_settings()) option_row(s);
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Keyboard (desktop)"); ImGui::PopFont();
    static const char* keys[][2] = {{"Move", "W A S D"}, {"Camera", "Arrow keys"}, {"Jump (X)", "Space"}, {"Attack (Y)", "F / mouse"},
                                    {"A (interact)", "E"}, {"B (run / cancel)", "Shift"}, {"Menu (+)", "Enter"}, {"Map (-)", "M"},
                                    {"Fullscreen", "F11 / Alt+Enter"}, {"FPS counter", "F3"}};
    if (ImGui::BeginTable("keys", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        for (auto& k : keys) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(k[0]); ImGui::TableNextColumn(); ImGui::TextColored(COL_ACCENT, "%s", k[1]); }
        ImGui::EndTable();
    }
}
static void page_cheats() {
    heading("Cheats", "Atmosphere-format cheat files (<build id>.txt). The game's 1.0.0 build id is 082CE09B06E33A12. Enabled cheats apply from the next start.");
    if (ImGui::Button("Import a cheat file...")) B.show("Choose a cheat file (.txt)", env_or("HOME", "/") + "/Downloads", {".txt"}, false, [](const string& p) {
        std::error_code ec; string name = fs::path(p).filename().string();
        fs::copy_file(p, P.cheats + "/" + name, fs::copy_options::overwrite_existing, ec); scan_cheats(); });
    ImGui::SameLine(); if (ImGui::Button("Rescan")) scan_cheats();
    ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "Folder: %s", P.cheats.c_str()); ImGui::PopFont();
    if (g_cheats.empty()) text_wrapped_col(COL_DIM, "No cheat files yet.");
    for (auto& f : g_cheats) {
        ImGui::PushFont(F_h2); ImGui::Text("%s", f.build.c_str()); ImGui::PopFont();
        if (f.build != "082CE09B06E33A12" && f.build != "082ce09b06e33a12") text_wrapped_col(COL_WARN, "This file is for another build of the game: its cheats will not match.");
        for (auto& c : f.cheats) {
            if (c.master) { ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "{%s} (always on)", c.name.c_str()); ImGui::PopFont(); continue; }
            string k = "cheat." + f.build + "." + c.name; bool on = C.on(k);
            if (ImGui::Checkbox(c.name.c_str(), &on)) C.set(k, on ? "1" : "0");
        }
    }
}
static void page_mods() {
    heading("Mods", "romfs mods: each folder in the mods folder holds a romfs/ folder (or the romfs contents directly). Enabled mods replace game files; earlier names win.");
    if (ImGui::Button("Add a mod folder...")) B.show("Choose a mod folder (the one that contains romfs)", env_or("HOME", "/") + "/Downloads", {}, true, [](const string& p) {
        std::error_code ec; fs::copy(p, P.mods + "/" + fs::path(p).filename().string(), fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec); scan_mods(); });
    ImGui::SameLine(); if (ImGui::Button("Rescan")) scan_mods();
    ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "Folder: %s", P.mods.c_str()); ImGui::PopFont();
    if (g_mods.empty()) text_wrapped_col(COL_DIM, "No mods yet.");
    for (auto& m : g_mods) { bool on = C.on("mod." + m); if (ImGui::Checkbox(m.c_str(), &on)) C.set("mod." + m, on ? "1" : "0"); }
}
static vector<fs::path> list_backups() {
    vector<fs::path> v; std::error_code ec;
    for (auto& e : fs::directory_iterator(P.backups, ec)) if (e.path().extension() == ".gz") v.push_back(e.path());
    std::sort(v.rbegin(), v.rend()); return v;
}
static string g_save_msg;
static void page_saves() {
    heading("Saves", ("Your save: " + save_dir()).c_str());
    bool b = C.on("backup_before_launch", true);
    if (ImGui::Checkbox("Back up the save before each launch (keeps the newest 10)", &b)) C.set("backup_before_launch", b ? "1" : "0");
    if (ImGui::Button("Back up now")) { string r = backup_saves("manual"); g_save_msg = r.empty() ? "Nothing to back up yet." : "Saved " + r; }
    ImGui::SameLine();
    if (ImGui::Button("Import a save...")) B.show("Choose the folder with the save (from Eden / yuzu / Ryujinx / JKSV)", env_or("HOME", "/"), {}, true, [](const string& p) {
        string root = find_save_root(p);
        if (root.empty()) { g_save_msg = "No Tears of the Kingdom save (slot_00 / progress.sav) found in that folder."; return; }
        confirm("Replace the current save with the one in\n" + root + "?\nThe current save is backed up first.", [root] {
            backup_saves("before-import"); std::error_code ec;
            fs::remove_all(save_dir(), ec); fs::create_directories(save_dir());
            fs::copy(root, save_dir(), fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
            g_save_msg = ec ? "Import failed: " + ec.message() : "Imported from " + root;
            lg("save import %s: %s", root.c_str(), g_save_msg.c_str());
        });
    });
    if (!g_save_msg.empty()) text_wrapped_col(COL_DIM, "%s", g_save_msg.c_str());
    ImGui::Dummy(ImVec2(0, 6));
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Backups"); ImGui::PopFont();
    auto v = list_backups();
    if (v.empty()) text_wrapped_col(COL_DIM, "None yet.");
    for (auto& p : v) {
        ImGui::PushID(p.string().c_str());
        std::error_code ec;
        ImGui::Text("%s  (%s)", p.filename().string().c_str(), human((double)fs::file_size(p, ec)).c_str()); ImGui::SameLine();
        if (ImGui::SmallButton("Restore")) confirm("Restore " + p.filename().string() + "?\nThe current save is backed up first.", [p] {
            backup_saves("before-restore"); std::error_code ec; fs::remove_all(save_dir(), ec);
            string r = run_capture("tar -C " + sq(P.state + "/userdata") + " -xzf " + sq(p.string()) + " && echo OK");
            g_save_msg = r.find("OK") != string::npos ? "Restored " + p.filename().string() : "Restore failed: " + trim(r); });
        ImGui::PopID();
    }
}
static void page_steam() {
    heading("Steam", "Adds this launcher to your Steam library with artwork made from the icon in your dump. Steam can stay open; restart it afterwards to see the artwork.");
    auto sc = our_shortcuts();
    if (!sc.empty()) text_wrapped_col(COL_OK, "In your Steam library (%d shortcut%s).", (int)sc.size(), sc.size() == 1 ? "" : "s");
    else text_wrapped_col(COL_DIM, "Not in your Steam library yet.");
    ImGui::BeginDisabled(g_steam_state == 1);
    if (big_button(sc.empty() ? "Add to Steam" : "Add again / refresh artwork", COL_ACCENT, ImVec2(380 * g_scale, 56 * g_scale))) {
        if (sc.empty()) add_to_steam();
        else { string e = install_artwork(sc); g_steam_msg = e.empty() ? "Artwork updated. Restart Steam to see it." : e; g_steam_state = e.empty() ? 2 : 3; }
    }
    ImGui::EndDisabled();
    if (!g_steam_msg.empty()) text_wrapped_col(g_steam_state == 3 ? COL_BAD : COL_DIM, "%s", g_steam_msg.c_str());
    if (g_steam_state == 2) {
        ImGui::Dummy(ImVec2(0, 6));
        if (big_button("Restart Steam now", COL_GOLD, ImVec2(320 * g_scale, 56 * g_scale)))
            confirm("Restart Steam now? (Anything running from Steam closes, including this launcher.)", [] { restart_steam(); });
    }
}
static void page_tools() {
    heading("Tools");
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Report a problem"); ImGui::PopFont();
    ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "Packs the last game log, launcher log, settings and system info (no game files, no keys) into one archive "
        "on your Desktop, with an issue template. Attach it to a GitHub issue."); ImGui::PopFont();
    static string rep;
    if (ImGui::Button("Create report")) rep = "Saved: " + make_report();
    if (!rep.empty()) text_wrapped_col(COL_OK, "%s", rep.c_str());
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Clean up"); ImGui::PopFont();
    if (g_clean.empty() && !g_clean_busy) scan_cleanup();
    if (g_clean_busy) text_wrapped_col(COL_DIM, "Measuring...");
    for (auto& c : g_clean) {
        ImGui::PushID(c.path.c_str());
        ImGui::Text("%s: %s", c.label.c_str(), human((double)c.size).c_str()); ImGui::SameLine();
        ImGui::BeginDisabled(!c.size || g_game_running);
        if (ImGui::SmallButton("Delete")) { string p = c.path; confirm("Delete " + c.label + " (" + human((double)c.size) + ")?", [p] {
            std::error_code ec; if (fs::is_directory(p, ec)) { for (auto& e : fs::directory_iterator(p, ec)) fs::remove_all(e.path(), ec); } else fs::remove(p, ec);
            lg("cleanup %s", p.c_str()); g_clean.clear(); }); }
        ImGui::EndDisabled();
        ImGui::PopID();
    }
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Updates"); ImGui::PopFont();
    if (ImGui::Button("Check for updates")) check_update();
    if (!g_update_msg.empty()) { ImGui::SameLine(); text_wrapped_col(COL_DIM, "%s", g_update_msg.c_str()); }
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("After Play"); ImGui::PopFont();
    bool cl = C.on("close_on_play", g_gamescope); if (ImGui::Checkbox("Hide the launcher while the game runs", &cl)) C.set("close_on_play", cl ? "1" : "0");
}
static void page_about() {
    heading("About");
    ImGui::Text("TotK Recomp launcher %s", kVersion);
    ImGui::TextWrapped("Install: %s", P.install.c_str());
    ImGui::TextWrapped("Data: %s", P.state.c_str());
    ImGui::Text("Device: %s, %d threads, %s RAM", S.model.c_str(), S.cores, human((double)S.ram).c_str());
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushFont(F_h2); ImGui::TextUnformatted("Game versions"); ImGui::PopFont();
    if (ImGui::BeginTable("vers", 2, ImGuiTableFlags_RowBg)) {
        for (auto& k : kKnown) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(k.disp);
            ImGui::TableNextColumn(); ImGui::TextColored(k.supported ? COL_OK : COL_WARN, "%s", k.supported ? "Supported" : "Not supported (may not work)"); }
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushFont(F_small);
    text_wrapped_col(COL_DIM, "This is an unofficial fan project, not affiliated with or endorsed by Nintendo. It contains no game code or data: everything is built "
                              "from your own legally dumped cartridge and keys on your own device.\n\nThird-party: SDL3 (zlib), Dear ImGui (MIT), stb (public domain), "
                              "Inter font (SIL OFL 1.1), hactool (ISC).");
    ImGui::PopFont();
}

// ------------------------------------------------------------------------------------------------ main
static void handle_drop(const string& p) {
    string x = fs::path(p).extension().string(); std::transform(x.begin(), x.end(), x.begin(), ::tolower);
    string n = fs::path(p).filename().string();
    if (x == ".xci" || x == ".nsp") { C.set("dump", p); C.save(); g_page = PG_SETUP; if (!C.get("keys").empty()) inspect_dump(p, C.get("keys")); }
    else if (x == ".keys" || n == "prod.keys") { C.set("keys", p); C.save(); g_page = PG_SETUP; if (!C.get("dump").empty()) inspect_dump(C.get("dump"), p); }
    else if (x == ".txt") { std::error_code ec; fs::copy_file(p, P.cheats + "/" + n, fs::copy_options::overwrite_existing, ec); scan_cheats(); g_page = PG_CHEATS; }
}

int main(int argc, char** argv) {
    string shot, keys_script; int shot_page = -1, shot_frames = 8;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shot = argv[++i];
        else if (a == "--page" && i + 1 < argc) shot_page = atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) shot_frames = atoi(argv[++i]);
        else if (a == "--keys" && i + 1 < argc) keys_script = argv[++i];
        else if (a == "--version") { printf("%s\n", kVersion); return 0; }
        else if (a == "--steam-add") {   // the Steam page's "Add to Steam", from a terminal (also the remote test)
            init_paths(); g_log = fopen(P.log.c_str(), "a"); C.load(); load_fonts_file();
            add_to_steam();
            while (g_steam_state == 1) usleep(200000);
            printf("%s\n", g_steam_msg.c_str());
            for (auto& sc : our_shortcuts()) printf("shortcut appid %u user %s name '%s' exe '%s'\n", sc.appid, sc.user.c_str(), sc.name.c_str(), sc.exe.c_str());
            return g_steam_state == 2 ? 0 : 1;
        }
        else if (a == "--art" && i + 1 < argc) {   // test: write the Steam artwork made from the dump's icon into a folder
            init_paths(); load_fonts_file(); string e = make_artwork(argv[++i]);
            printf("%s\n", e.empty() ? "ok" : e.c_str()); return e.empty() ? 0 : 1;
        }
    }
    init_paths();
    g_log = fopen(P.log.c_str(), "a");
    lg("start %s (exe %s, state %s)", kVersion, P.exe_dir.c_str(), P.state.c_str());
    {   // remote tests (tools/linux/deck_launcher_test.sh): options for this one start, then the file is renamed
        string tf = P.state + "/launcher-test.env";
        std::ifstream f(tf); string l; bool any = false;
        while (std::getline(f, l)) { l = trim(l); auto e = l.find('='); if (l.empty() || l[0] == '#' || e == string::npos) continue;
            string v = trim(l.substr(e + 1)); if (v.size() >= 2 && (v[0] == '\'' || v[0] == '"')) v = v.substr(1, v.size() - 2);
            setenv(trim(l.substr(0, e)).c_str(), v.c_str(), 1); any = true; }
        if (any) { std::error_code ec; fs::rename(tf, tf + ".used", ec); lg("test options from %s", tf.c_str()); }
    }
    C.load(); detect_system(); scan_install(); scan_cheats(); scan_mods();
    if (!shot.empty()) setenv("SDL_RENDER_DRIVER", "software", 0);
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
    SDL_WindowFlags wf = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    g_win = SDL_CreateWindow("TotK Recomp", 1280, 800, wf);
    if (!g_win) { fprintf(stderr, "window: %s\n", SDL_GetError()); return 1; }
    g_ren = SDL_CreateRenderer(g_win, nullptr);
    if (!g_ren) { fprintf(stderr, "renderer: %s\n", SDL_GetError()); return 1; }
    SDL_SetRenderVSync(g_ren, 1);
    bool gamescope = g_gamescope = getenv("GAMESCOPE_WAYLAND_DISPLAY") || getenv("SteamDeck");
    if (gamescope && shot.empty()) SDL_SetWindowFullscreen(g_win, true);
    int ww, wh; SDL_GetWindowSizeInPixels(g_win, &ww, &wh);
    g_scale = std::clamp(wh / 800.f, 0.8f, 2.5f);

    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.IniFilename = nullptr;
    style(); ImGui::GetStyle().ScaleAllSizes(g_scale); load_fonts();
    ImGui_ImplSDL3_InitForSDLRenderer(g_win, g_ren);
    ImGui_ImplSDLRenderer3_Init(g_ren);
    if (!C.get("dump").empty() && !C.get("keys").empty()) inspect_dump(C.get("dump"), C.get("keys"));
    if (shot_page >= 0) g_page = shot_page;
    if (!I.game || !I.romfs) if (shot_page < 0 && C.get("dump").empty()) g_page = PG_SETUP;

    // scripted keys for test shots: comma list of down/up/left/right/enter/esc/tab/l1/r1, one per frame
    vector<string> script; { std::stringstream ss(keys_script); string t; while (std::getline(ss, t, ',')) if (!t.empty()) script.push_back(t); }
    int frame = 0; bool quit = false; bool was_running = false;
    vector<std::pair<int, string>> test_shots;   // TOTK_LAUNCHER_SHOTS="60:/tmp/a.png,200:/tmp/b.png" (real display)
    { std::stringstream ss(env_or("TOTK_LAUNCHER_SHOTS", "")); string t;
      while (std::getline(ss, t, ',')) { auto c = t.find(':'); if (c != string::npos) test_shots.push_back({atoi(t.substr(0, c).c_str()), t.substr(c + 1)}); } }
    const int test_play = atoi(env_or("TOTK_LAUNCHER_PLAY", "0").c_str());   // press Play at this frame
    const bool test_quit = env_or("TOTK_LAUNCHER_QUIT", "0") == "1";          // quit when the game has ended
    const bool test_warmup = env_or("TOTK_LAUNCHER_WARMUP", "0") == "1";      // the test press is the warm-up button
    bool test_played = false;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT) quit = true;
            if (e.type == SDL_EVENT_DROP_FILE && e.drop.data) handle_drop(e.drop.data);
        }
        if (!script.empty() && frame >= 2 && (size_t)(frame - 2) < script.size()) {
            string k = script[frame - 2]; ImGuiKey key = ImGuiKey_None;
            if (k == "down") key = ImGuiKey_DownArrow; else if (k == "up") key = ImGuiKey_UpArrow; else if (k == "left") key = ImGuiKey_LeftArrow;
            else if (k == "right") key = ImGuiKey_RightArrow; else if (k == "enter") key = ImGuiKey_Enter; else if (k == "esc") key = ImGuiKey_Escape;
            else if (k == "l1") key = ImGuiKey_GamepadL1; else if (k == "r1") key = ImGuiKey_GamepadR1;
            if (key != ImGuiKey_None) { io.AddKeyEvent(key, true); io.AddKeyEvent(key, false); }
        }
        // once only: the frame counter stands still while the game runs hidden, so it is still == test_play when the game ends
        if (test_play && !test_played && frame == test_play && I.game && I.romfs && !g_game_running) {
            lg(test_warmup ? "test: Warm up" : "test: Play"); test_played = true;
            launch_game(test_warmup, [] { g_last_launch_msg = "Game closed (status " + std::to_string(g_game_status) + ")."; });
            if (C.on("close_on_play", g_gamescope)) SDL_HideWindow(g_win);
        }
        bool running = g_game_running;
        if (was_running && !running) { SDL_ShowWindow(g_win); SDL_RaiseWindow(g_win); scan_install(); if (test_quit && test_played) quit = true; }
        was_running = running;
        if (running && C.on("close_on_play", gamescope)) { SDL_Delay(100); continue; }

        ImGui_ImplSDLRenderer3_NewFrame(); ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame();
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false)) g_page = (g_page + 1) % PG_N;
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false)) g_page = (g_page + PG_N - 1) % PG_N;

        ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("##root", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoSavedSettings);
        // sidebar
        float side = 230 * g_scale;
        ImGui::BeginChild("side", ImVec2(side, 0), false);
        ImGui::PushFont(F_h1); ImGui::TextColored(COL_GOLD, "TotK"); ImGui::SameLine(); ImGui::TextColored(COL_ACCENT, "Recomp"); ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 10));
        ImGui::PushFont(F_h2);
        for (int i = 0; i < PG_N; i++) {
            bool sel = g_page == i;
            if (sel) ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.36f, 0.36f, 1));
            if (ImGui::Selectable(kPages[i], sel, 0, ImVec2(0, 46 * g_scale))) g_page = i;
            if (sel) ImGui::PopStyleColor();
        }
        ImGui::PopFont();
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 60 * g_scale);
        ImGui::PushFont(F_small); text_wrapped_col(COL_DIM, "L1 / R1: pages   A: select   B: back"); ImGui::PopFont();
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("page", ImVec2(0, 0), false);
        switch (g_page) {
            case PG_HOME: page_home(); break;      case PG_SETUP: page_setup(); break;   case PG_GRAPHICS: page_graphics(); break;
            case PG_CONTROLS: page_controls(); break; case PG_CHEATS: page_cheats(); break; case PG_MODS: page_mods(); break;
            case PG_SAVES: page_saves(); break;    case PG_STEAM: page_steam(); break;   case PG_TOOLS: page_tools(); break;
            default: page_about(); break;
        }
        B.draw(); draw_confirm();
        ImGui::EndChild();
        ImGui::End();
        ImGui::Render();
        SDL_SetRenderDrawColor(g_ren, 17, 21, 25, 255); SDL_RenderClear(g_ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), g_ren);
        for (auto& ts : test_shots) if (ts.first == frame) {   // real-display test shot (before present)
            SDL_Surface* sf = SDL_RenderReadPixels(g_ren, nullptr);
            if (sf) { SDL_Surface* c = SDL_ConvertSurface(sf, SDL_PIXELFORMAT_ABGR8888);
                      if (c) { stbi_write_png(ts.second.c_str(), c->w, c->h, 4, c->pixels, c->pitch); SDL_DestroySurface(c); lg("test shot %s", ts.second.c_str()); }
                      SDL_DestroySurface(sf); }
        }
        static int settle = 0;   // test shots: wait for the dump check / disk scans to finish (at most ~60 s)
        if (!shot.empty() && (g_dump_busy || g_clean_busy || g_job.running) && settle < 1200) { settle++; frame--; SDL_Delay(50); }
        if (!shot.empty() && frame >= shot_frames + (int)script.size()) {
            SDL_Surface* s = SDL_RenderReadPixels(g_ren, nullptr);
            if (s) {
                SDL_Surface* c = SDL_ConvertSurface(s, SDL_PIXELFORMAT_ABGR8888);
                if (c) { stbi_write_png(shot.c_str(), c->w, c->h, 4, c->pixels, c->pitch); SDL_DestroySurface(c); }
                SDL_DestroySurface(s);
            }
            quit = true;
        }
        SDL_RenderPresent(g_ren);
        C.save();
        frame++;
    }
    C.save();
    ImGui_ImplSDLRenderer3_Shutdown(); ImGui_ImplSDL3_Shutdown(); ImGui::DestroyContext();
    SDL_DestroyRenderer(g_ren); SDL_DestroyWindow(g_win); SDL_Quit();
    lg("exit");
    return 0;
}

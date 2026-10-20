// HLE: nn::hid. One Pro Controller (FullKey) on Npad id 0 (No1). Layouts follow Eden's hid_core/hid_types.h
// (NpadFullKeyState etc. = 0x28 bytes; SixAxisSensorState 0x60; MouseState 0x28; KeyboardState 0x30).
// Input = host pads (XInput: Xbox; raw HID: DualSense, DualShock 4, Pro Controller, Joy-Cons) | keyboard (while the game window has focus) |
// TOTK_PRESS script; see host_input().
#include "hle.h"
#include <chrono>
#include <windows.h>
#ifndef _WIN32
#include <SDL3/SDL.h>      // Linux: pads, keyboard and motion come from SDL3 (Steam Input shows up as an SDL gamepad)
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <dirent.h>
#include <time.h>
#include <cerrno>
#include <sys/ioctl.h>
#include <linux/hidraw.h>
#include "linux/plat_posix.h"
#endif
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <cwchar>
#include <cwctype>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

enum : u32 { StyleFullKey = 1u << 0, StyleHandheld = 1u << 1, StyleJoyDual = 1u << 2 };
static std::atomic<s64> g_sample{1};

struct NpadState {          // NpadFullKeyState / JoyDual / Handheld / JoyLeft / JoyRight share this layout
    s64 sampling_number; u64 buttons; s32 lx, ly, rx, ry; u32 attributes; u32 reserved;
};
static_assert(sizeof(NpadState) == 0x28);
static bool npad_connected(u32 id) { return id == 0; }

// Scripted input for unattended test runs (user 2026-10-14: after the autosave, boots stop at the title screen):
// TOTK_PRESS="A@600,A@700,PLUS@900" holds each button for 6 presented frames starting at that frame (present count);
// "A@300+30x40" repeats: 40 presses, 30 frames apart, from frame 300.
// Npad button bits (Eden hid_types.h NpadButton): A 0, B 1, X 2, Y 3, LS 4, RS 5, L 6, R 7, ZL 8, ZR 9, PLUS 10,
// MINUS 11, LEFT 12, UP 13, RIGHT 14, DOWN 15.
extern "C" uint64_t nvn_present_count(void);
// Stick pseudo-buttons LUP/LDOWN/LLEFT/LRIGHT/RUP/RDOWN/RLEFT/RRIGHT = full deflection (bits 32-39, never sent as
// buttons) and an optional hold length: "LUP@2100:150" pushes the left stick up for 150 frames (default hold 6).
enum { kScriptStick = 32 };
struct Press { u64 frame, bits, hold; bool logged; };
static std::vector<Press> parse_presses() {
    static const struct { const char* n; int bit; } names[] = {{"A", 0}, {"B", 1}, {"X", 2}, {"Y", 3}, {"LS", 4}, {"RS", 5},
        {"L", 6}, {"R", 7}, {"ZL", 8}, {"ZR", 9}, {"PLUS", 10}, {"MINUS", 11}, {"LEFT", 12}, {"UP", 13}, {"RIGHT", 14}, {"DOWN", 15},
        {"LUP", 32}, {"LDOWN", 33}, {"LLEFT", 34}, {"LRIGHT", 35}, {"RUP", 36}, {"RDOWN", 37}, {"RLEFT", 38}, {"RRIGHT", 39}, {"DUMP", 40}};
    std::vector<Press> v;
    const char* e = getenv("TOTK_PRESS");
    // TOTK_AUTO_NEWGAME=1 (set by the in-game skip restart): title menu Continue -> New Game, A, "Overwrite previous
    // save data?" Cancel -> Start, A (layout and timing observed in runs 223/224/228: menu up by present 300)
    if (!e && getenv("TOTK_AUTO_NEWGAME")) e = "DOWN@350,A@420,UP@480,A@540";
    for (const char* p = e; p && *p;) {
        char name[16] = {}; unsigned long long f = 0, hold = 6; int n = 0;
        if (sscanf(p, "%15[A-Z]@%llu%n", name, &f, &n) != 2) { printf("[hid] TOTK_PRESS: cannot parse at \"%s\"\n", p); break; }
        int bit = -1;
        for (auto& nm : names) if (!strcmp(nm.n, name)) bit = nm.bit;
        p += n;
        int k = 0;
        if (*p == ':' && sscanf(p, ":%llu%n", &hold, &k) == 1) p += k;
        // repeat form "A@300+30x40": from frame 300, every 30 frames, 40 presses (user: tap through the title as soon
        // as it shows instead of guessing its frame)
        unsigned long long every = 0, count = 1; int m = 0;
        if (*p == '+' && sscanf(p, "+%llux%llu%n", &every, &count, &m) == 2) p += m; else every = 0, count = 1;
        if (bit < 0) printf("[hid] TOTK_PRESS: unknown button %s\n", name);
        else for (unsigned long long i = 0; i < count; i++) v.push_back({f + i * every, 1ull << bit, hold, false});
        if (*p == ',') p++;
    }
    return v;
}
extern "C" volatile int g_debug_frame_request;   // nvn/vulkan/totk_hooks.c (device.c dumps the next frame)
// "DUMP@N" (bit 40, never sent): request a frame capture at tick N, as F9 does (with LARECOMP_DEBUG_FRAMES=n: n frames)
static u64 scripted_buttons() {
    static std::vector<Press> presses = parse_presses();
    if (presses.empty()) return 0;
    u64 f = nvn_present_count(), b = 0;
    // TOTK_PRESS_CLOCK=wall: frame numbers are 1/30 s ticks of wall time since the first presented frame, so a script
    // covers the same real time at any frame rate (speed A/B runs at 30 vs 60 fps)
    static const bool wall = getenv("TOTK_PRESS_CLOCK") && !strcmp(getenv("TOTK_PRESS_CLOCK"), "wall");
    if (wall) {
        static auto t0 = std::chrono::steady_clock::now();
        f = (u64)(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 30.0);
    }
    for (auto& p : presses)
        if (f >= p.frame && f < p.frame + p.hold) {
            b |= p.bits & ~(1ull << 40);
            if (!p.logged && (p.bits >> 40 & 1)) g_debug_frame_request = 1;
            if (!p.logged) { p.logged = true; printf("[hid] scripted press %llx at frame %llu\n", (unsigned long long)p.bits, (unsigned long long)f); }
        }
    return b;
}
// ------------------------------------------------------------------ host input
// Pad: XInput (Xbox controllers; a DualSense / Pro Controller shows up through Steam Input or DS4Windows). Face buttons
// map by POSITION like Eden's SDL default (pad B = right = Switch A, pad A = bottom = Switch B, Y = top = X, X = left
// = Y), so on-screen prompts match where the button sits; TOTK_PAD_LABELS=1 maps by label instead (pad A -> Switch A).
// Keyboard (only while the game window is in front): WASD move (Ctrl = walk), arrows / IJKL camera, Space X (jump),
// Shift B (sprint / back), E or Enter A (action / confirm), F Y (attack), Q L (abilities), R R (throw / use),
// Z ZL (target / shield), C ZR (bow), V LS (crouch), G RS (look), Esc PLUS (menu), M MINUS (map), 1-4 D-pad L U R D,
// Backspace B.
enum : u64 { BtnA = 1u << 0, BtnB = 1u << 1, BtnX = 1u << 2, BtnY = 1u << 3, BtnLS = 1u << 4, BtnRS = 1u << 5,
             BtnL = 1u << 6, BtnR = 1u << 7, BtnZL = 1u << 8, BtnZR = 1u << 9, BtnPlus = 1u << 10, BtnMinus = 1u << 11,
             BtnLeft = 1u << 12, BtnUp = 1u << 13, BtnRight = 1u << 14, BtnDown = 1u << 15 };
struct HostInput { u64 buttons; s32 lx, ly, rx, ry; };

#ifdef _WIN32
struct XPad { WORD buttons; BYTE lt, rt; SHORT lx, ly, rx, ry; };     // XINPUT_GAMEPAD
struct XState { DWORD packet; XPad pad; };                          // XINPUT_STATE
typedef DWORD(WINAPI* XInputGetState_t)(DWORD, XState*);
static XInputGetState_t xinput_load() {
    for (const char* n : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"})
        if (HMODULE m = LoadLibraryA(n))
            if (auto f = (XInputGetState_t)GetProcAddress(m, "XInputGetState")) { printf("[hid] host pads via %s\n", n); return f; }
    printf("[hid] XInput not found: keyboard only\n");
    return nullptr;
}
#endif

// radial dead zone, rescaled so the edge of the dead zone is 0 and full deflection stays 32767 (Npad stick range)
static void stick(SHORT x, SHORT y, int dz, s32* ox, s32* oy) {
    float fx = x, fy = y, m = sqrtf(fx * fx + fy * fy);
    if (m <= dz) { *ox = *oy = 0; return; }
    float k = fminf((m - dz) / (32767.0f - dz), 1.0f) * 32767.0f / m;
    *ox = (s32)lrintf(fx * k); *oy = (s32)lrintf(fy * k);
}

#ifdef _WIN32
static std::atomic<int> g_xslot{-1};   // XInput slot in use (rumble_thread: XInputSetState)
static void pad_read(HostInput* in) {
    static XInputGetState_t get = xinput_load();
    static const bool labels = getenv("TOTK_PAD_LABELS") != nullptr;
    static int slot = -1;
    static u64 next_scan;
    if (!get) return;
    XState st;
    if (slot < 0) {   // XInputGetState on an empty slot is slow (~ms): look for a new pad once a second only
        u64 now = GetTickCount64();
        if (now < next_scan) return;
        next_scan = now + 1000;
        for (int i = 0; i < 4 && slot < 0; i++) if (get(i, &st) == ERROR_SUCCESS) { slot = i; printf("[hid] pad connected (XInput slot %d)\n", i); }
        if (slot < 0) return;
        g_xslot = slot;
    }
    if (get(slot, &st) != ERROR_SUCCESS) { printf("[hid] pad on XInput slot %d disconnected\n", slot); slot = -1; g_xslot = -1; return; }
    const XPad& p = st.pad;
    static const struct { WORD x; u64 pos, lab; } map[] = {
        {0x1000 /*A*/, BtnB, BtnA}, {0x2000 /*B*/, BtnA, BtnB}, {0x4000 /*X*/, BtnY, BtnX}, {0x8000 /*Y*/, BtnX, BtnY},
        {0x0100 /*LB*/, BtnL, BtnL}, {0x0200 /*RB*/, BtnR, BtnR}, {0x0040 /*LTHUMB*/, BtnLS, BtnLS},
        {0x0080 /*RTHUMB*/, BtnRS, BtnRS}, {0x0010 /*START*/, BtnPlus, BtnPlus}, {0x0020 /*BACK*/, BtnMinus, BtnMinus},
        {0x0001, BtnUp, BtnUp}, {0x0002, BtnDown, BtnDown}, {0x0004, BtnLeft, BtnLeft}, {0x0008, BtnRight, BtnRight}};
    for (auto& m : map) if (p.buttons & m.x) in->buttons |= labels ? m.lab : m.pos;
    if (p.lt > 30) in->buttons |= BtnZL;      // XINPUT_GAMEPAD_TRIGGER_THRESHOLD
    if (p.rt > 30) in->buttons |= BtnZR;
    stick(p.lx, p.ly, 7849, &in->lx, &in->ly);   // XINPUT_GAMEPAD_LEFT / RIGHT_THUMB_DEADZONE
    stick(p.rx, p.ry, 8689, &in->rx, &in->ry);
}
#else
static void pad_read(HostInput* in);   // SDL gamepad, below the pad state
#endif

// ---- Pads over raw HID (no XInput, no DS4Windows / BetterJoy needed), USB or Bluetooth, up to 4 at once (two Joy-Cons
// make one pad: their buttons and sticks are merged):
//  Sony DualSense / DualShock 4 (VID 054C). Report layouts as in SDL's hidapi PS4 / PS5 drivers: DS4 order LX LY RX RY
//   b0 b1 b2 L2 R2 (USB 0x01 at +1, BT 0x11 at +3, BT "simple" 0x01 at +1, also the DualSense BT simple report);
//   DualSense full order LX LY RX RY L2 R2 counter b0 b1 b2 (USB 0x01 at +1, BT 0x31 at +2). b0 = hat (low nibble,
//   8 = centred) | Square 0x10 Cross 0x20 Circle 0x40 Triangle 0x80; b1 = L1 R1 L2 R2 Create Options L3 R3; b2 = PS 1,
//   touchpad 2. Sticks 0..255, 128 centre, y down. Face buttons by position like the Xbox mapping.
//  Nintendo Pro Controller / Joy-Con L / Joy-Con R (VID 057E, PID 2009 / 2006 / 2007). Switched to the standard full
//   report 0x30 with subcommand 0x03 (USB first gets the 0x80 02/03/02/04 handshake), player light 1 via subcommand
//   0x30 (dekuNukem Nintendo_Switch_Reverse_Engineering, SDL hidapi_switch). Report: [3] Y X B A SR SL R ZR,
//   [4] - + RS LS Home Capture, [5] Down Up Right Left SR SL L ZL, [6..8] left stick, [9..11] right stick (12-bit x, y).
//   Buttons by label (Switch layout). Stick centre = first full report (sticks at rest), range grows from 1300 to
//   what the stick reaches (no SPI factory calibration read).
enum PadKind { PadNone, PadDualSense, PadDS4, PadPro, PadJoyL, PadJoyR };
// Motion in the Switch SixAxisSensor frame as Eden/yuzu feed it (x right, y forward, z up out of the face; accel in g,
// -1 on z lying flat face up; angular velocity in revolutions/s), integrated by the pad thread (motion_fuse).
struct Motion { float acc[3] = {0, 0, -1}, gyro[3] = {}, angle[3] = {}, dir[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; s64 dt_ns = 5000000; u64 n = 0; };
struct HidPad {
    std::atomic<u64> buttons{0}; std::atomic<s32> lx{0}, ly{0}, rx{0}, ry{0}; std::atomic<bool> live{false};
    std::mutex mmu; Motion m; std::atomic<bool> has_imu{false};
    // rumble output: a second handle (synchronous ReadFile on the read handle would hold WriteFile until a report lands)
    std::atomic<HANDLE> wr{INVALID_HANDLE_VALUE}; std::atomic<int> kind{PadNone}; bool bt = false; int outlen = 64; u8 seq = 0;
};
static HidPad g_hidpad[4];
static std::mutex g_hidpad_mtx;
static std::wstring g_hidpad_path[4];   // guarded by g_hidpad_mtx; empty = free slot

#ifdef _WIN32
typedef BOOLEAN(WINAPI* HidD_GetAttributes_t)(HANDLE, void*);
typedef BOOLEAN(WINAPI* HidD_GetPreparsedData_t)(HANDLE, void**);
typedef BOOLEAN(WINAPI* HidD_FreePreparsedData_t)(void*);
typedef BOOLEAN(WINAPI* HidD_GetFeature_t)(HANDLE, void*, ULONG);
typedef LONG(WINAPI* HidP_GetCaps_t)(void*, void*);
typedef DWORD(WINAPI* CM_List_Size_t)(PULONG, LPGUID, void*, ULONG);
typedef DWORD(WINAPI* CM_List_t)(LPGUID, void*, wchar_t*, ULONG, ULONG);
static struct {
    HidD_GetAttributes_t attr; HidD_GetPreparsedData_t prep; HidD_FreePreparsedData_t freep; HidP_GetCaps_t caps; HidD_GetFeature_t getfeat;
    CM_List_Size_t list_size; CM_List_t list;
} hidapi;
#endif

static void sony_decode(const u8* r, PadKind kind, bool bt, HidPad* pad) {
    const u8* p; bool ds4_order;
    if (bt && r[0] == 0x01) { p = r + 1; ds4_order = true; }                 // BT simple report (both pads)
    else if (!bt && r[0] == 0x01) { p = r + 1; ds4_order = kind == PadDS4; }
    else if (bt && kind == PadDualSense && r[0] == 0x31) { p = r + 2; ds4_order = false; }
    else if (bt && kind == PadDS4 && r[0] >= 0x11 && r[0] <= 0x19) { p = r + 3; ds4_order = true; }
    else return;
    u8 b0 = ds4_order ? p[4] : p[7], b1 = ds4_order ? p[5] : p[8], b2 = ds4_order ? p[6] : p[9];
    u8 l2 = ds4_order ? p[7] : p[4], r2 = ds4_order ? p[8] : p[5];
    u64 b = 0;
    if (b0 & 0x40) b |= BtnA;   // Circle (right)
    if (b0 & 0x20) b |= BtnB;   // Cross (bottom)
    if (b0 & 0x80) b |= BtnX;   // Triangle (top)
    if (b0 & 0x10) b |= BtnY;   // Square (left)
    static const u64 hat[8] = {BtnUp, BtnUp | BtnRight, BtnRight, BtnRight | BtnDown, BtnDown, BtnDown | BtnLeft, BtnLeft, BtnLeft | BtnUp};
    if ((b0 & 15) < 8) b |= hat[b0 & 15];
    if (b1 & 0x01) b |= BtnL;
    if (b1 & 0x02) b |= BtnR;
    if ((b1 & 0x04) || l2 > 30) b |= BtnZL;
    if ((b1 & 0x08) || r2 > 30) b |= BtnZR;
    if (b1 & 0x10) b |= BtnMinus;
    if (b1 & 0x20) b |= BtnPlus;
    if (b1 & 0x40) b |= BtnLS;
    if (b1 & 0x80) b |= BtnRS;
    if (b2 & 0x02) b |= BtnMinus;   // touchpad click = map too
    auto ax = [](u8 v) { int d = (int)v - 128; return d < 0 ? d * 32767 / 128 : d * 32767 / 127; };   // +-32767
    s32 x, y;
    stick((SHORT)ax(p[0]), (SHORT)-ax(p[1]), 5000, &x, &y); pad->lx = x; pad->ly = y;   // y down on the pad
    stick((SHORT)ax(p[2]), (SHORT)-ax(p[3]), 5000, &x, &y); pad->rx = x; pad->ry = y;
    pad->buttons = b;
}

struct NinCal { int c[4], r[4]; bool init; };
static void nin_decode(const u8* r, PadKind kind, NinCal* k, HidPad* pad) {
    if (r[0] != 0x30 && r[0] != 0x21 && r[0] != 0x31) return;   // full report / subcommand reply / NFC-IR: same header
    int raw[4] = {r[6] | (r[7] & 0xF) << 8, r[7] >> 4 | r[8] << 4, r[9] | (r[10] & 0xF) << 8, r[10] >> 4 | r[11] << 4};
    if (!k->init) {
        k->init = true;
        for (int i = 0; i < 4; i++) k->c[i] = raw[i], k->r[i] = 1300;
        printf("[hid] Nintendo pad stick centres %d,%d %d,%d (taken at rest)\n", raw[0], raw[1], raw[2], raw[3]);
    }
    s32 v[4];
    for (int i = 0; i < 4; i++) {
        int d = raw[i] - k->c[i], a = d < 0 ? -d : d;
        if (a > k->r[i]) k->r[i] = a;
        v[i] = (s32)((long long)d * 32767 / k->r[i]);
    }
    u8 b3 = r[3], b4 = r[4], b5 = r[5];
    u64 b = 0;
    if (b3 & 0x01) b |= BtnY;
    if (b3 & 0x02) b |= BtnX;
    if (b3 & 0x04) b |= BtnB;
    if (b3 & 0x08) b |= BtnA;
    if (b3 & 0x40) b |= BtnR;
    if (b3 & 0x80) b |= BtnZR;
    if (b4 & 0x01) b |= BtnMinus;
    if (b4 & 0x02) b |= BtnPlus;
    if (b4 & 0x04) b |= BtnRS;
    if (b4 & 0x08) b |= BtnLS;
    if (b5 & 0x01) b |= BtnDown;
    if (b5 & 0x02) b |= BtnUp;
    if (b5 & 0x04) b |= BtnRight;
    if (b5 & 0x08) b |= BtnLeft;
    if (b5 & 0x40) b |= BtnL;
    if (b5 & 0x80) b |= BtnZL;
    s32 x, y;
    if (kind != PadJoyR) { stick((SHORT)v[0], (SHORT)v[1], 4000, &x, &y); pad->lx = x; pad->ly = y; }   // y up already
    if (kind != PadJoyL) { stick((SHORT)v[2], (SHORT)v[3], 4000, &x, &y); pad->rx = x; pad->ry = y; }
    pad->buttons = b;
}

// ---- motion (user 2026-10-15: "add motion control support for dualsense and switch pro controllers")
// Pad thread side: raw IMU -> Switch frame (research/sweep/w30, w31: Eden/yuzu SDL mapping accel (-x, z, -y) / g,
// gyro (x, -z, y) / 2pi from SDL's frame x right, y up, z toward the player) -> fusion like Eden's motion_input.cpp
// (gyro integration + accelerometer tilt correction, Mahony gain 0.5; gyro bias averaged while the pad lies still;
// 1.3 dps dead zone). TOTK_GYRO=0: no motion (the game sees a pad lying still).
struct Fusion {
    float q[4] = {1, 0, 0, 0}, bias[3] = {}; double still_s = 0, steady_s = 0; u64 logged = 0;
    double win_s = 0, win_still = 0, win_steady = 0, win_out[3] = {}; int win_logs = 0;   // [hid] gyro log window
};
static void motion_fuse(HidPad* pad, Fusion* f, const float acc[3], const float gyro_in[3], float dt) {
    if (!(dt > 0 && dt < 0.1f)) dt = 0.005f;
    float an = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    float g[3] = {gyro_in[0] - f->bias[0], gyro_in[1] - f->bias[1], gyro_in[2] - f->bias[2]};
    float gn = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    // Gyro bias (user 2026-10-15: "theres a rightward drift on the gyro ... its contant"): the old rule learned it only
    // while the pad LAY still (< 4 dps, 1 g +- 0.05 for 0.5 s); held in the hands it never did, tremor kept |gyro| over
    // the dead zone and the residual yaw bias went to the game. Now three rates:
    //  lying still 0.5 s: 0.02 per sample (as before);
    //  held steady (< 2 dps for 1 s, 0.9..1.1 g): time constant 5 s (ours);
    //  not moving fast (< 0.05 rev/s = 18 dps, 0.9..1.1 g): time constant 40 s = Eden motion_input.cpp SetGyroscope
    //  (gyro_bias * 0.9999 + gyroscope * 0.0001 per sample at ~250 Hz while !IsMoving(IsAtRestRelaxed)).
    bool g1 = an > 0.9f && an < 1.1f;
    bool still = gn < 0.011f && fabsf(an - 1.0f) < 0.05f;           // < ~4 dps, ~1 g
    f->still_s = still ? f->still_s + dt : 0;
    f->steady_s = gn < 0.0055f && g1 ? f->steady_s + dt : 0;
    float rate = f->still_s > 0.5 ? 0.02f : f->steady_s > 1.0 ? dt / 5.0f : gn < 0.05f && g1 ? dt / 40.0f : 0.0f;
    if (rate > 0) for (int i = 0; i < 3; i++) f->bias[i] += (gyro_in[i] - f->bias[i]) * rate;
    if (gn < 0.0035f) g[0] = g[1] = g[2] = 0;
    f->win_s += dt; f->win_still += f->still_s > 0.5 ? dt : 0; f->win_steady += f->steady_s > 1.0 ? dt : 0;
    for (int i = 0; i < 3; i++) f->win_out[i] += g[i] * dt;
    if (f->win_s >= 30.0) {   // measurement for the next report: bias + what reached the game (mean dps) per 30 s
        if (f->win_logs++ < 60)
            printf("[hid] gyro 30 s: bias %.2f %.2f %.2f dps, game mean %.2f %.2f %.2f dps, lying still %.0f %%, held steady %.0f %%\n",
                   f->bias[0] * 360, f->bias[1] * 360, f->bias[2] * 360, f->win_out[0] / f->win_s * 360, f->win_out[1] / f->win_s * 360,
                   f->win_out[2] / f->win_s * 360, f->win_still / f->win_s * 100, f->win_steady / f->win_s * 100);
        f->win_s = f->win_still = f->win_steady = 0; f->win_out[0] = f->win_out[1] = f->win_out[2] = 0;
    }
    float w[3] = {g[0] * 6.2831853f, g[1] * 6.2831853f, g[2] * 6.2831853f};   // rad/s
    float* q = f->q;
    if (an > 0.75f && an < 1.25f) {   // expected "down" in the pad frame = R^T (0, 0, -1); error = measured x expected
        float vx = -2 * (q[1] * q[3] - q[0] * q[2]), vy = -2 * (q[0] * q[1] + q[2] * q[3]);
        float vz = -(q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
        float ax = acc[0] / an, ay = acc[1] / an, az = acc[2] / an;
        const float kp = 0.5f;
        w[0] += kp * (ay * vz - az * vy); w[1] += kp * (az * vx - ax * vz); w[2] += kp * (ax * vy - ay * vx);
    }
    float h = 0.5f * dt;
    float q0 = q[0] + (-q[1] * w[0] - q[2] * w[1] - q[3] * w[2]) * h, q1 = q[1] + (q[0] * w[0] + q[2] * w[2] - q[3] * w[1]) * h;
    float q2 = q[2] + (q[0] * w[1] - q[1] * w[2] + q[3] * w[0]) * h, q3 = q[3] + (q[0] * w[2] + q[1] * w[1] - q[2] * w[0]) * h;
    float qn = 1.0f / sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q[0] = q0 * qn; q[1] = q1 * qn; q[2] = q2 * qn; q[3] = q3 * qn;
    std::lock_guard<std::mutex> l(pad->mmu);
    Motion& m = pad->m;
    for (int i = 0; i < 3; i++) { m.acc[i] = acc[i]; m.gyro[i] = g[i]; m.angle[i] += g[i] * dt; }
    // direction: the pad's x / y / z axes in the world frame (columns of R; identity lying flat, pointing forward)
    float xx = q[1] * q[1], yy = q[2] * q[2], zz = q[3] * q[3], xy = q[1] * q[2], xz = q[1] * q[3], yz = q[2] * q[3];
    float wx = q[0] * q[1], wy = q[0] * q[2], wz = q[0] * q[3];
    float r[9] = {1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                  2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)};
    for (int c = 0; c < 3; c++) for (int k = 0; k < 3; k++) m.dir[c * 3 + k] = r[k * 3 + c];
    m.dt_ns = (s64)(dt * 1e9f);
    m.n++;
    pad->has_imu = true;
    if (f->logged < 3 && m.n % 2000 == 1) {
        f->logged++;
        printf("[hid] motion: accel %.2f %.2f %.2f g, gyro %.3f %.3f %.3f rev/s, bias %.4f %.4f %.4f\n", acc[0], acc[1], acc[2], g[0], g[1],
               g[2], f->bias[0], f->bias[1], f->bias[2]);
    }
}
static bool ds_haptics_on() { static const bool on = getenv("TOTK_DS_HAPTICS") && atoi(getenv("TOTK_DS_HAPTICS")) != 0; return on; }
static bool gyro_on() { static const bool on = !getenv("TOTK_GYRO") || strcmp(getenv("TOTK_GYRO"), "0") != 0; return on; }
static s16 le16(const u8* p) { return (s16)(p[0] | p[1] << 8); }

// DualSense calibration (feature report 0x05; w30: SDL formulas): gyro dps = (raw - gb) * gs, accel g = (raw - ab) * as.
struct DsCal { float gs[3] = {1 / 16.0f, 1 / 16.0f, 1 / 16.0f}, gb[3] = {}, as[3] = {1 / 8192.0f, 1 / 8192.0f, 1 / 8192.0f}, ab[3] = {}; };
static void ds_imu(const u8* r, bool bt, const DsCal& k, HidPad* pad, Fusion* f, double dt) {
    const u8* p = bt ? r + 2 : r + 1;   // state struct: gyro x y z at +15, accel x y z at +21 (int16)
    float gd[3], ag[3];
    for (int i = 0; i < 3; i++) { gd[i] = (le16(p + 15 + 2 * i) - k.gb[i]) * k.gs[i]; ag[i] = (le16(p + 21 + 2 * i) - k.ab[i]) * k.as[i]; }
    const float acc[3] = {-ag[0], ag[2], -ag[1]};
    const float gyro[3] = {gd[0] / 360.0f, -gd[2] / 360.0f, gd[1] / 360.0f};
    motion_fuse(pad, f, acc, gyro, (float)dt);
}
// Pro Controller / Joy-Con report 0x30: 3 IMU frames (accel x y z, gyro x y z) at 13 + 12 i, 5 ms apart (w31:
// 4096 LSB per g, 0.070 dps per LSB; SDL frame (-y, z, -x) for both, then the Eden mapping above).
static void nin_imu(const u8* r, HidPad* pad, Fusion* f) {
    if (r[0] != 0x30) return;
    for (int s = 0; s < 3; s++) {
        const u8* p = r + 13 + 12 * s;
        float a[3] = {le16(p) / 4096.0f, le16(p + 2) / 4096.0f, le16(p + 4) / 4096.0f};
        float gd[3] = {le16(p + 6) * 0.0700f, le16(p + 8) * 0.0700f, le16(p + 10) * 0.0700f};
        const float acc[3] = {a[1], -a[0], -a[2]};
        const float gyro[3] = {-gd[1] / 360.0f, gd[0] / 360.0f, gd[2] / 360.0f};
        motion_fuse(pad, f, acc, gyro, 0.005f);
    }
}

#ifdef _WIN32
static bool hid_write(HANDLE h, const u8* d, int n, int outlen) {
    u8 buf[1024] = {};
    if (n > (int)sizeof buf) n = sizeof buf;
    if (outlen > (int)sizeof buf) outlen = sizeof buf;
    memcpy(buf, d, n);
    DWORD w = 0;
    return WriteFile(h, buf, (DWORD)(outlen > n ? outlen : n), &w, nullptr) != 0;
}
static void nin_subcmd(HANDLE h, int outlen, u8* pkt, u8 cmd, u8 arg) {
    u8 b[12] = {0x01, (u8)((*pkt)++ & 15), 0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40, cmd, arg};   // neutral rumble
    if (!hid_write(h, b, 12, outlen)) printf("[hid] Nintendo subcommand %02x failed (%lu)\n", cmd, GetLastError());
    Sleep(60);
}

static u32 crc32_buf(const u8* d, size_t n) {   // CRC-32 (reflected 0xEDB88320), as SDL_crc32 / crc32_le
    u32 c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) { c ^= d[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1))); }
    return ~c;
}
static void ds_cal_read(HANDLE h, int featlen, DsCal* k) {
    u8 b[256] = {0x05};
    if (!hidapi.getfeat || featlen < 41 || featlen > (int)sizeof b || !hidapi.getfeat(h, b, (ULONG)featlen)) {
        printf("[hid] DualSense calibration (feature 0x05) not read (%lu): default scales\n", GetLastError());
        return;
    }
    auto v = [&](int o) { return (float)le16(b + o); };
    float speed = v(19) + v(21);
    for (int i = 0; i < 3; i++) {
        float plus = v(7 + 4 * i), minus = v(9 + 4 * i), den = plus - minus;
        float gs = den != 0 ? speed / den : 0;
        if (gs > 1 / 40.0f && gs < 1 / 8.0f) { k->gs[i] = gs; k->gb[i] = v(1 + 2 * i); }
        float ap = v(23 + 4 * i), am = v(25 + 4 * i), range = ap - am;
        if (range > 8192 && range < 32768) { k->as[i] = 2.0f / range; k->ab[i] = ap - range / 2; }
    }
    printf("[hid] DualSense calibration: gyro %.4f %.4f %.4f dps/LSB bias %.0f %.0f %.0f, accel %.6f %.6f %.6f g/LSB\n", k->gs[0], k->gs[1],
           k->gs[2], k->gb[0], k->gb[1], k->gb[2], k->as[0], k->as[1], k->as[2]);
}
// DualSense output report (w30, SDL DS5EffectsState_t): USB 0x02 (effects at [1], 48 bytes); Bluetooth 0x31 (78 bytes:
// [1] sequence << 4, [2] 0x10, effects at [3], CRC-32 of 0xA2 + bytes 0..73 at 74..77). Sending 0x31 is also what
// switches a Bluetooth DualSense from the simple 0x01 report (no motion data) to the full 0x31 report.
// effects [0] 0x01 rumble emulation | 0x02 classic rumble (audio haptics off), [2] right motor, [3] left motor,
// [38] 0x04 improved rumble emulation (firmware >= 2.24: full-range motor values).
static void ds_effects(HidPad* p, HANDLE w, u8 left, u8 right, bool haptics) {
    u8 b[80] = {};
    int off = p->bt ? 3 : 1, size = p->bt ? 78 : 48;
    if (p->bt) { b[0] = 0x31; b[1] = (u8)(p->seq++ << 4); b[2] = 0x10; } else b[0] = 0x02;
    u8* e = b + off;
    e[0] = haptics ? 0x00 : 0x03;
    e[2] = right; e[3] = left;
    e[38] = 0x04;
    if (p->bt) {
        u8 t[80]; t[0] = 0xA2; memcpy(t + 1, b, 74);
        u32 c = crc32_buf(t, 75);
        b[74] = (u8)c; b[75] = (u8)(c >> 8); b[76] = (u8)(c >> 16); b[77] = (u8)(c >> 24);
    }
    if (!hid_write(w, b, size, p->outlen)) {
        static std::atomic<int> told{0};
        if (told++ < 3) printf("[hid] DualSense output report failed (%lu)\n", GetLastError());
    }
}

static void hidpad_thread(int slot, std::wstring path, PadKind kind, bool bt, HANDLE h) {
    HidPad* pad = &g_hidpad[slot];
    int outlen = 64, featlen = 0;
    void* pd = nullptr;
    if (hidapi.prep(h, &pd)) {
        USHORT caps[32] = {};   // HIDP_CAPS: Usage, UsagePage, InputReportByteLength, OutputReportByteLength, FeatureReportByteLength
        if (hidapi.caps(pd, caps) >= 0 && caps[3]) { outlen = caps[3]; featlen = caps[4]; }
        hidapi.freep(pd);
    }
    // rumble writes go through their own handle (rumble_thread); motion/rumble state on the pad
    HANDLE w = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    pad->bt = bt; pad->outlen = outlen; pad->seq = 0; pad->has_imu = false;
    { std::lock_guard<std::mutex> l(pad->mmu); pad->m = Motion{}; }
    if (kind == PadPro || kind == PadJoyL || kind == PadJoyR) {
        if (!bt) {   // USB: handshake, 3 Mbit, handshake, HID only (no timeout) before Bluetooth-style subcommands work
            for (u8 c : {0x02, 0x03, 0x02, 0x04}) { u8 m[2] = {0x80, c}; hid_write(h, m, 2, outlen); Sleep(60); }
        }
        u8 pkt = 0;
        nin_subcmd(h, outlen, &pkt, 0x03, 0x30);   // input report mode: standard full (0x30)
        nin_subcmd(h, outlen, &pkt, 0x30, 0x01);   // player light 1
        nin_subcmd(h, outlen, &pkt, 0x40, 0x01);   // IMU on (w31)
        nin_subcmd(h, outlen, &pkt, 0x48, 0x01);   // vibration on (HD rumble)
        pad->seq = pkt;
    }
    DsCal dscal;
    if (kind == PadDualSense) {
        ds_cal_read(h, featlen, &dscal);
        if (w != INVALID_HANDLE_VALUE) ds_effects(pad, w, 0, 0, bt && ds_haptics_on());   // (Bluetooth: full 0x31 reports from here on)
    }
    pad->wr = w;
    pad->kind = kind;
    Fusion fus;
    auto imu_t = std::chrono::steady_clock::now();
    NinCal cal = {};
    u8 rep[1024];
    DWORD n = 0;
    int nlog = 0;
    u64 last = 0;
    bool first = true;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);   // input latency: decode as soon as a report lands
    u64 rate_n = 0, rate_logs = 0;
    auto rate_t0 = std::chrono::steady_clock::now();
    while (ReadFile(h, rep, sizeof rep, &n, nullptr) && n > 0) {
        if (kind == PadDualSense || kind == PadDS4) sony_decode(rep, kind, bt, pad);
        else nin_decode(rep, kind, &cal, pad);
        if (gyro_on()) {
            if (kind == PadDualSense && ((bt && rep[0] == 0x31) || (!bt && rep[0] == 0x01 && n >= 64))) {
                auto t = std::chrono::steady_clock::now();
                ds_imu(rep, bt, dscal, pad, &fus, std::chrono::duration<double>(t - imu_t).count());
                imu_t = t;
            } else if (kind == PadPro || kind == PadJoyL || kind == PadJoyR) nin_imu(rep, pad, &fus);
        }
        // report rate (input latency, user 2026-10-15): one line per 10 s for the first minute
        rate_n++;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - rate_t0).count();
        if (el >= 10.0 && rate_logs < 6) {
            printf("[hid] slot %d: %.0f reports/s (report %02x, %lu bytes)\n", slot, rate_n / el, rep[0], n);
            rate_logs++; rate_n = 0; rate_t0 = std::chrono::steady_clock::now();
        }
        if (first) {
            first = false;
            printf("[hid] slot %d first report %02x (%lu bytes): sticks %d,%d %d,%d buttons %llx\n", slot, rep[0], n, pad->lx.load(),
                   pad->ly.load(), pad->rx.load(), pad->ry.load(), (unsigned long long)pad->buttons.load());
        }
        pad->live = true;
        u64 b = pad->buttons.load();
        if (b != last && b && nlog < 12) { nlog++; printf("[hid] slot %d buttons %llx (report %02x)\n", slot, (unsigned long long)b, rep[0]); }
        last = b;
    }
    printf("[hid] slot %d disconnected (%lu)\n", slot, GetLastError());
    pad->live = false; pad->has_imu = false; pad->buttons = 0; pad->lx = pad->ly = pad->rx = pad->ry = 0;
    pad->kind = PadNone;
    HANDLE ow = pad->wr.exchange(INVALID_HANDLE_VALUE);
    Sleep(20);   // (rumble_thread may be mid-write on the old handle)
    if (ow != INVALID_HANDLE_VALUE) CloseHandle(ow);
    CloseHandle(h);
    std::lock_guard<std::mutex> l(g_hidpad_mtx);
    g_hidpad_path[slot].clear();
}

static void hidpad_scan_thread() {
    HMODULE hid = LoadLibraryA("hid.dll"), cfg = LoadLibraryA("cfgmgr32.dll");
    if (hid) {
        hidapi.attr = (HidD_GetAttributes_t)GetProcAddress(hid, "HidD_GetAttributes");
        hidapi.prep = (HidD_GetPreparsedData_t)GetProcAddress(hid, "HidD_GetPreparsedData");
        hidapi.freep = (HidD_FreePreparsedData_t)GetProcAddress(hid, "HidD_FreePreparsedData");
        hidapi.caps = (HidP_GetCaps_t)GetProcAddress(hid, "HidP_GetCaps");
        hidapi.getfeat = (HidD_GetFeature_t)GetProcAddress(hid, "HidD_GetFeature");
    }
    if (cfg) {
        hidapi.list_size = (CM_List_Size_t)GetProcAddress(cfg, "CM_Get_Device_Interface_List_SizeW");
        hidapi.list = (CM_List_t)GetProcAddress(cfg, "CM_Get_Device_Interface_ListW");
    }
    if (!hidapi.attr || !hidapi.prep || !hidapi.freep || !hidapi.caps || !hidapi.list_size || !hidapi.list) {
        printf("[hid] raw HID unavailable: no DualSense / DS4 / Pro Controller / Joy-Con support\n");
        return;
    }
    GUID hid_guid = {0x4D1E55B2, 0xF16F, 0x11CF, {0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30}};   // GUID_DEVINTERFACE_HID
    for (;; Sleep(1500)) {
        ULONG len = 0;
        if (hidapi.list_size(&len, &hid_guid, nullptr, 0) != 0 || len <= 1) continue;
        std::vector<wchar_t> buf(len);
        if (hidapi.list(&hid_guid, nullptr, buf.data(), len, 0) != 0) continue;
        for (const wchar_t* path = buf.data(); *path; path += wcslen(path) + 1) {
            int slot = -1;
            {
                std::lock_guard<std::mutex> l(g_hidpad_mtx);
                bool open = false;
                for (int i = 0; i < 4; i++) open |= g_hidpad_path[i] == path;
                for (int i = 0; i < 4 && slot < 0 && !open; i++) if (g_hidpad_path[i].empty()) slot = i;
            }
            if (slot < 0) continue;
            HANDLE q = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (q == INVALID_HANDLE_VALUE) continue;
            struct { ULONG size; USHORT vid, pid, ver; } at = {sizeof at, 0, 0, 0};
            bool ok = hidapi.attr(q, &at);
            CloseHandle(q);
            if (!ok) continue;
            PadKind kind = PadNone;
            if (at.vid == 0x054C) kind = (at.pid == 0x0CE6 || at.pid == 0x0DF2) ? PadDualSense
                                       : (at.pid == 0x05C4 || at.pid == 0x09CC || at.pid == 0x0BA0) ? PadDS4 : PadNone;
            if (at.vid == 0x057E) kind = at.pid == 0x2009 ? PadPro : at.pid == 0x2006 ? PadJoyL : at.pid == 0x2007 ? PadJoyR : PadNone;
            if (kind == PadNone) continue;
            HANDLE h = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h == INVALID_HANDLE_VALUE) h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h == INVALID_HANDLE_VALUE) continue;
            std::wstring lp(path);
            for (auto& ch : lp) ch = towlower(ch);
            bool bt = lp.find(L"00001124-0000-1000-8000-00805f9b34fb") != std::wstring::npos;
            static const char* const names[] = {"", "DualSense", "DualShock 4", "Pro Controller", "Joy-Con (L)", "Joy-Con (R)"};
            printf("[hid] %s connected over %s (pid %04x) -> slot %d\n", names[kind], bt ? "Bluetooth" : "USB", at.pid, slot);
            {
                std::lock_guard<std::mutex> l(g_hidpad_mtx);
                g_hidpad_path[slot] = path;
            }
            std::thread(hidpad_thread, slot, std::wstring(path), kind, bt, h).detach();
        }
    }
}

#else
// ---- Linux: ONE SDL3 gamepad (the first one found; Steam Input presents the Deck's controls as a virtual Xbox 360 / DualSense /
// Switch Pro pad that SDL reads like any other). Buttons / sticks are polled through the SDL gamepad API (state is updated by
// the event pump on the UI thread, device.c ui_run -> shim_sdl3.c totk_wait_event); motion arrives as sensor events on that thread
// (totk_hid_sensor below) and feeds the same fusion as the Windows raw-HID pads; rumble goes out through SDL_RumbleGamepad.
// Face buttons map by POSITION like the Windows XInput path (TOTK_PAD_LABELS=1: by label).
static std::mutex g_gp_mu;
static SDL_Gamepad* g_gp;
static std::atomic<bool> g_gp_rumble{false};
static float g_gp_acc[3] = {0, 0, -1};   // latest accelerometer sample in the Switch frame (g)
static Fusion g_gp_fus;
static Uint64 g_gp_gyro_ts;
static std::atomic<bool> g_gp_has_gyro{false};   // the open SDL gamepad has its own gyro (then the Deck IMU reader stays out)

static void gp_open_first() {   // g_gp_mu held; called at most once a second while no pad is open
    int n = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&n);
    for (int i = 0; i < n && !g_gp; i++) {
        SDL_Gamepad* g = SDL_OpenGamepad(ids[i]);
        if (!g) continue;
        g_gp = g;
        HidPad* pad = &g_hidpad[0];
        bool acc = SDL_GamepadHasSensor(g, SDL_SENSOR_ACCEL), gyr = SDL_GamepadHasSensor(g, SDL_SENSOR_GYRO);
        if (acc) SDL_SetGamepadSensorEnabled(g, SDL_SENSOR_ACCEL, true);
        if (gyr) SDL_SetGamepadSensorEnabled(g, SDL_SENSOR_GYRO, true);
        SDL_PropertiesID pr = SDL_GetGamepadProperties(g);
        g_gp_rumble = SDL_GetBooleanProperty(pr, SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
        { std::lock_guard<std::mutex> l(pad->mmu); pad->m = Motion{}; }
        g_gp_fus = Fusion{}; g_gp_gyro_ts = 0;
        g_gp_has_gyro = gyr;
        pad->has_imu = false; pad->live = true; pad->kind = PadPro;
        printf("[hid] SDL gamepad \"%s\" (type %d, rumble %s, accel %s, gyro %s)\n", SDL_GetGamepadName(g) ? SDL_GetGamepadName(g) : "?",
               (int)SDL_GetGamepadType(g), g_gp_rumble ? "yes" : "no", acc ? "yes" : "no", gyr ? "yes" : "no");
    }
    SDL_free(ids);
}
static void gp_close_locked() {
    if (!g_gp) return;
    printf("[hid] SDL gamepad disconnected\n");
    SDL_CloseGamepad(g_gp);
    g_gp = nullptr; g_gp_rumble = false; g_gp_has_gyro = false;
    HidPad* pad = &g_hidpad[0];
    pad->live = false; pad->has_imu = false; pad->buttons = 0;
}
static void pad_read(HostInput* in) {
    static const bool labels = getenv("TOTK_PAD_LABELS") != nullptr;
    static bool inited, init_ok;
    static u64 next_scan;
    std::lock_guard<std::mutex> l(g_gp_mu);
    if (!inited) {
        inited = true;
        init_ok = SDL_InitSubSystem(SDL_INIT_GAMEPAD);
        if (!init_ok) printf("[hid] SDL gamepad init failed (%s): keyboard only\n", SDL_GetError());
    }
    if (!init_ok) return;
    if (g_gp && !SDL_GamepadConnected(g_gp)) gp_close_locked();
    if (!g_gp) {   // look for a new pad once a second only
        u64 now = GetTickCount64();
        if (now < next_scan) return;
        next_scan = now + 1000;
        gp_open_first();
        if (!g_gp) return;
    }
    SDL_Gamepad* g = g_gp;
    auto btn = [&](SDL_GamepadButton b) { return SDL_GetGamepadButton(g, b); };
    auto lab = [&](SDL_GamepadButton b, u64 pos) -> u64 {   // TOTK_PAD_LABELS: the Switch button with the face button's printed letter
        switch (SDL_GetGamepadButtonLabel(g, b)) {
        case SDL_GAMEPAD_BUTTON_LABEL_A: return BtnA; case SDL_GAMEPAD_BUTTON_LABEL_B: return BtnB;
        case SDL_GAMEPAD_BUTTON_LABEL_X: return BtnX; case SDL_GAMEPAD_BUTTON_LABEL_Y: return BtnY;
        default: return pos;
        }
    };
    struct { SDL_GamepadButton b; u64 pos; } face[] = {{SDL_GAMEPAD_BUTTON_EAST, BtnA}, {SDL_GAMEPAD_BUTTON_SOUTH, BtnB},
                                                        {SDL_GAMEPAD_BUTTON_NORTH, BtnX}, {SDL_GAMEPAD_BUTTON_WEST, BtnY}};
    for (auto& f : face) if (btn(f.b)) in->buttons |= labels ? lab(f.b, f.pos) : f.pos;
    static const struct { SDL_GamepadButton b; u64 sw; } map[] = {
        {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, BtnL}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, BtnR}, {SDL_GAMEPAD_BUTTON_LEFT_STICK, BtnLS},
        {SDL_GAMEPAD_BUTTON_RIGHT_STICK, BtnRS}, {SDL_GAMEPAD_BUTTON_START, BtnPlus}, {SDL_GAMEPAD_BUTTON_BACK, BtnMinus},
        {SDL_GAMEPAD_BUTTON_DPAD_UP, BtnUp}, {SDL_GAMEPAD_BUTTON_DPAD_DOWN, BtnDown}, {SDL_GAMEPAD_BUTTON_DPAD_LEFT, BtnLeft},
        {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, BtnRight}};
    for (auto& m : map) if (btn(m.b)) in->buttons |= m.sw;
    if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 3855) in->buttons |= BtnZL;    // 30 / 255 of the 0..32767 range (XInput threshold)
    if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 3855) in->buttons |= BtnZR;
    auto ax = [&](SDL_GamepadAxis a, bool flip) { int v = SDL_GetGamepadAxis(g, a); if (flip) v = -v; return (SHORT)(v > 32767 ? 32767 : v < -32767 ? -32767 : v); };
    stick(ax(SDL_GAMEPAD_AXIS_LEFTX, false), ax(SDL_GAMEPAD_AXIS_LEFTY, true), 7849, &in->lx, &in->ly);      // SDL y is down, Npad y up
    stick(ax(SDL_GAMEPAD_AXIS_RIGHTX, false), ax(SDL_GAMEPAD_AXIS_RIGHTY, true), 8689, &in->rx, &in->ry);   // (XInput dead zones)
}
// Sensor events from the UI thread (shim_sdl3.c): accelerometer samples are kept, each gyroscope sample is fused with the latest one
// using the sensor's own timestamps. SDL frame (x right, y up, z toward the player; accel m/s^2, gyro rad/s) -> Switch frame as
// Eden/yuzu do: accel (-x, z, -y) / g, gyro (x, -z, y) / 2pi (rev/s).
extern "C" void totk_hid_sensor(int type, const float* d, Uint64 ts_ns) {
    if (!gyro_on()) return;
    if (type == SDL_SENSOR_ACCEL) {
        const float g = SDL_STANDARD_GRAVITY;
        g_gp_acc[0] = -d[0] / g; g_gp_acc[1] = d[2] / g; g_gp_acc[2] = -d[1] / g;
    } else if (type == SDL_SENSOR_GYRO) {
        float dt = g_gp_gyro_ts && ts_ns > g_gp_gyro_ts ? (float)((ts_ns - g_gp_gyro_ts) * 1e-9) : 0.005f;
        g_gp_gyro_ts = ts_ns;
        const float k = 1.0f / 6.2831853f;
        const float gyro[3] = {d[0] * k, -d[2] * k, d[1] * k};
        motion_fuse(&g_hidpad[0], &g_gp_fus, g_gp_acc, gyro, dt);
    }
}
// Steam Deck built-in IMU (user 2026-10-17: "add gyro for the steam deck"). Under Steam Input the game sees only Steam's virtual
// gamepad (SDL: "Steam Deck Controller", accel no, gyro no), but the controller's vendor HID interface (28de:1205, input2) keeps
// streaming its 64-byte deck state reports (01 00 09 40 ..., ~250 Hz) with raw accel / gyro while Steam runs (measured
// 2026-10-17 over SSH: /dev/hidraw3, 124 reports in 0.5 s, accel 388 -1012 16439 lying flat; Steam's uaccess udev rule makes
// the node readable by the logged-in user). But while a game runs Steam switches the IMU off (w3 probe 11:35: 250 reports in
// 1 s, accel / gyro / quaternion all 0) unless the Steam layout uses gyro, so the reader turns it back on with the setting
// Steam and SDL use (feature report 0x87 ID_SET_SETTINGS_VALUES: SETTING_IMU_MODE 48 = SEND_RAW_ACCEL | SEND_RAW_GYRO |
// SEND_ORIENTATION, a runtime mode the controller forgets; SDL_hidapi_steam.c SetSensorsEnabled), again whenever the data
// stays all-zero for 0.5 s (at most every 2 s). TOTK_DECK_IMU=ro: never write (gyro then needs a gyro binding in the Steam
// layout). Scale and axes as SDL's HIDAPI Steam Deck driver (accel +-2 g, gyro +-2000 dps; SDL frame =
// (x, z, -y)) followed by totk_hid_sensor's SDL -> Switch conversion: the raw gyro axes land unchanged on the Switch axes,
// the accelerometer negated. Feeds pad 0 unless the open SDL gamepad has a gyro of its own. TOTK_DECK_IMU=0: off.
static bool deck_imu_enable(int fd) {
    unsigned char buf[65] = {};   // report id 0, then the 64-byte message (USB path of SDL's SetFeatureReport: always 65 bytes)
    buf[1] = 0x87; buf[2] = 3; buf[3] = 48; buf[4] = 0x08 | 0x10 | 0x04; buf[5] = 0;
    return ioctl(fd, HIDIOCSFEATURE(sizeof buf), buf) >= 0;
}
static int deck_imu_open(bool quiet) {
    DIR* d = opendir("/sys/class/hidraw");
    if (!d) return -1;
    int fd = -1;
    while (dirent* e = readdir(d)) {
        if (strncmp(e->d_name, "hidraw", 6) != 0) continue;
        char path[320], buf[1024];
        snprintf(path, sizeof path, "/sys/class/hidraw/%s/device/uevent", e->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[n] = 0;
        if (!strstr(buf, "HID_ID=0003:000028DE:00001205")) continue;
        snprintf(path, sizeof path, "/dev/%s", e->d_name);
        static const bool ro = getenv("TOTK_DECK_IMU") && !strcmp(getenv("TOTK_DECK_IMU"), "ro");
        int h = ro ? -1 : open(path, O_RDWR | O_CLOEXEC);
        if (h < 0) h = open(path, O_RDONLY | O_CLOEXEC);
        if (h < 0) {
            if (!quiet) printf("[hid] Steam Deck IMU: %s: %s\n", path, strerror(errno));
            continue;
        }
        bool ok = false;   // the vendor interface is the one sending deck state reports
        for (int i = 0; i < 20 && !ok; i++) {
            pollfd p = {h, POLLIN, 0};
            if (poll(&p, 1, 50) <= 0) break;
            unsigned char r[64];
            ssize_t k = read(h, r, sizeof r);
            ok = k >= 36 && r[2] == 0x09;
        }
        if (ok) {
            fd = h;
            printf("[hid] Steam Deck IMU: %s (controller HID, read-only)\n", path);
            break;
        }
        close(h);
    }
    closedir(d);
    return fd;
}
static void deck_imu_thread() {
    Fusion fus;
    bool quiet = false;
    for (;;) {
        int fd = deck_imu_open(quiet);
        if (fd < 0) {
            if (!quiet) printf("[hid] Steam Deck IMU: no readable Steam Deck controller interface; looking again every 5 s\n");
            quiet = true;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        quiet = false;
        u64 last = 0, n = 0, zeros = 0, last_on = 0, ons = 0;
        unsigned char r[64];
        for (;;) {
            pollfd p = {fd, POLLIN, 0};
            int pr = poll(&p, 1, 1000);
            if (pr < 0 && errno == EINTR) continue;
            if (pr <= 0 || (p.revents & (POLLERR | POLLHUP | POLLNVAL))) break;   // gone, or silent for 1 s (suspend): look again
            ssize_t k = read(fd, r, sizeof r);
            if (k < 0) { if (errno == EINTR || errno == EAGAIN) continue; break; }
            if (k < 36 || r[2] != 0x09) continue;
            timespec t;
            clock_gettime(CLOCK_MONOTONIC, &t);
            u64 now = (u64)t.tv_sec * 1000000000ull + (u64)t.tv_nsec;
            float dt = last ? (float)((now - last) * 1e-9) : 0.004f;
            last = now;
            if (dt < 0.0005f || dt > 0.02f) dt = 0.004f;   // (reports come every ~4 ms; after a stall use the nominal step)
            if (g_gp_has_gyro) continue;
            s16 v[6];
            memcpy(v, r + 24, sizeof v);   // accel x y z, gyro x y z (int16) after the 4-byte header + packet number + buttons + pads
            if (!(v[0] | v[1] | v[2] | v[3] | v[4] | v[5])) {   // IMU off (a live accelerometer is never exactly 0 on all axes)
                if (++zeros >= 125 && now - last_on > 2000000000ull) {
                    last_on = now; zeros = 0;
                    bool ok = deck_imu_enable(fd);
                    if (++ons <= 3 || !(ons & (ons - 1)))
                        printf("[hid] Steam Deck IMU: data is off, switching it on (%llu): %s\n", (unsigned long long)ons, ok ? "ok" : strerror(errno));
                }
                continue;
            }
            zeros = 0;
            const float ka = 2.0f / 32768.0f, kg = 2000.0f / 32768.0f / 360.0f;
            const float acc[3] = {-v[0] * ka, -v[1] * ka, -v[2] * ka}, gyro[3] = {v[3] * kg, v[4] * kg, v[5] * kg};
            motion_fuse(&g_hidpad[0], &fus, acc, gyro, dt);
            if (++n == 1) printf("[hid] Steam Deck IMU: first sample accel %.2f %.2f %.2f g, gyro %.3f %.3f %.3f rev/s\n", acc[0], acc[1],
                                 acc[2], gyro[0], gyro[1], gyro[2]);
        }
        close(fd);
        printf("[hid] Steam Deck IMU: reports stopped after %llu samples; looking again\n", (unsigned long long)n);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
static void deck_imu_start() {
    const char* e = getenv("TOTK_DECK_IMU");
    if (!gyro_on() || (e && *e == '0')) return;
    if (access("/sys/class/hidraw", F_OK) != 0) return;
    std::thread(deck_imu_thread).detach();
}
#endif
static void rumble_start();   // below the vibration graph
static void hidpad_read(HostInput* in) {
#ifdef _WIN32
    static bool started = [] { std::thread(hidpad_scan_thread).detach(); rumble_start(); return true; }();
#else
    static bool started = [] { rumble_start(); deck_imu_start(); return true; }();
#endif
    (void)started;
    bool lset = in->lx || in->ly, rset = in->rx || in->ry;
    for (auto& p : g_hidpad) {
        if (!p.live) continue;
        in->buttons |= p.buttons.load();
        s32 lx = p.lx, ly = p.ly, rx = p.rx, ry = p.ry;
        if (!lset && (lx || ly)) { in->lx = lx; in->ly = ly; lset = true; }
        if (!rset && (rx || ry)) { in->rx = rx; in->ry = ry; rset = true; }
    }
}

extern "C" volatile int g_debug_frame_request;   // nvn/vulkan/totk_hooks.c (device.c dumps the next frame)
extern "C" volatile int g_skip_opening, g_title_phase, g_in_opening;   // hle_fs.cpp (skip-opening overlay; banner in hle_osd.cpp)
extern "C" { volatile int g_skip_hold_ms = 0; }                          // Minus held in the opening (hle_osd.cpp countdown)
static bool g_f2_down, g_f10_down;
extern "C" bool osd_menu_input(uint64_t buttons, bool key);   // hle_osd.cpp Graphics menu
extern "C" int osd_menu_is_open(void);
extern "C" int vk_precompile_warmup_active(void);   // nvn/vulkan/exec.c (TOTK_WARMUP until "play now")
extern "C" void vk_precompile_play_now(void);
// Restart this exe with TOTK_SKIP_OPENING=1 TOTK_AUTO_NEWGAME=1 (same command line, std handles and other env), then
// end this process at once (the normal exit path is not needed: the new process owns the window from here).
#ifdef _WIN32
static void skip_restart() {
    static bool done;
    if (done) return;
    done = true;
    printf("[hid] in-game skip: restarting into New Game with the opening skipped\n");
    fflush(stdout); fflush(stderr);
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    SetEnvironmentVariableW(L"TOTK_SKIP_OPENING", L"1");
    SetEnvironmentVariableW(L"TOTK_AUTO_NEWGAME", L"1");
    SetEnvironmentVariableW(L"TOTK_PRESS", nullptr);
    STARTUPINFOW si{sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE); si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE); si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe, GetCommandLineW(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        printf("[hid] in-game skip: CreateProcess failed (%lu)\n", GetLastError());
        done = false;
        return;
    }
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    TerminateProcess(GetCurrentProcess(), 0);
}
#else
static void skip_restart() {
    static bool done;
    if (done) return;
    done = true;
    printf("[hid] in-game skip: restarting into New Game with the opening skipped\n");
    fflush(stdout); fflush(stderr);
    // same command line, std handles and other env: exec in place (the window and GPU state die with the old image)
    std::vector<std::string> args;
    if (FILE* f = fopen("/proc/self/cmdline", "rb")) {
        std::string cur; int ch;
        while ((ch = fgetc(f)) != EOF) { if (ch) cur += (char)ch; else { args.push_back(cur); cur.clear(); } }
        fclose(f);
    }
    std::string exe = std::string(rt_posix_exe_dir()) + "/totk";
    char self[4096]; ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) { self[n] = 0; exe = self; }
    setenv("TOTK_SKIP_OPENING", "1", 1);
    setenv("TOTK_AUTO_NEWGAME", "1", 1);
    unsetenv("TOTK_PRESS");
    std::vector<char*> av;
    for (auto& a : args) av.push_back((char*)a.c_str());
    if (av.empty()) av.push_back((char*)exe.c_str());
    av.push_back(nullptr);
    execv(exe.c_str(), av.data());
    printf("[hid] in-game skip: exec failed (%d)\n", errno);
    done = false;
}
#endif
#ifdef _WIN32
static void keyboard_read(HostInput* in) {
    DWORD pid = 0;
    HWND fg = GetForegroundWindow();
    if (!fg || !GetWindowThreadProcessId(fg, &pid) || pid != GetCurrentProcessId()) return;   // game window not in front
    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    static const struct { int vk; u64 b; } keys[] = {
        {VK_SPACE, BtnX}, {VK_SHIFT, BtnB}, {VK_BACK, BtnB}, {'E', BtnA}, {VK_RETURN, BtnA}, {'F', BtnY}, {'Q', BtnL},
        {'R', BtnR}, {'Z', BtnZL}, {'C', BtnZR}, {'V', BtnLS}, {'G', BtnRS}, {VK_ESCAPE, BtnPlus}, {'M', BtnMinus},
        {'1', BtnLeft}, {'2', BtnUp}, {'3', BtnRight}, {'4', BtnDown}};
    for (auto& k : keys) if (down(k.vk)) in->buttons |= k.b;
    // F9: capture the next frame (pass dumps + [draw] log, as LARECOMP_DEBUG_FRAME does), for bugs seen while playing
    static bool f9_was = false;
    bool f9 = down(VK_F9);
    if (f9 && !f9_was) {
        g_debug_frame_request = 1;
        printf("[hid] F9: frame capture requested\n");
    }
    f9_was = f9;
    g_f2_down = down(VK_F2);   // skip-opening toggle (host_input)
    g_f10_down = down(VK_F10);   // Graphics menu (hle_osd.cpp)
    auto axis = [&](int neg, int pos, int neg2, int pos2) { return (down(pos) || down(pos2) ? 1 : 0) - (down(neg) || down(neg2) ? 1 : 0); };
    int mx = axis('A', 'D', 0, 0), my = axis('S', 'W', 0, 0);
    int cx = axis(VK_LEFT, VK_RIGHT, 'J', 'L'), cy = axis(VK_DOWN, VK_UP, 'K', 'I');
    float walk = down(VK_CONTROL) ? 0.45f : 1.0f;
    if (mx || my) {   // keys win over the pad stick while held
        float k = 32767.0f * walk / sqrtf((float)(mx * mx + my * my));
        in->lx = (s32)lrintf(mx * k); in->ly = (s32)lrintf(my * k);
    }
    if (cx || cy) {
        float k = 32767.0f / sqrtf((float)(cx * cx + cy * cy));
        in->rx = (s32)lrintf(cx * k); in->ry = (s32)lrintf(cy * k);
    }
}
#else
// Keyboard (only while the game window has focus): SDL scancodes = physical key positions (QWERTY), same keys as above.
static void keyboard_read(HostInput* in) {
    if (!SDL_GetKeyboardFocus()) return;                        // game window not in front (or no window: headless)
    int nk = 0;
    const bool* ks = SDL_GetKeyboardState(&nk);
    if (!ks) return;
    auto down = [&](SDL_Scancode sc) { return sc != SDL_SCANCODE_UNKNOWN && (int)sc < nk && ks[sc]; };
    const SDL_Scancode U = SDL_SCANCODE_UNKNOWN;
    static const struct { SDL_Scancode a, b; u64 btn; } keys[] = {
        {SDL_SCANCODE_SPACE, SDL_SCANCODE_UNKNOWN, BtnX}, {SDL_SCANCODE_LSHIFT, SDL_SCANCODE_RSHIFT, BtnB}, {SDL_SCANCODE_BACKSPACE, SDL_SCANCODE_UNKNOWN, BtnB},
        {SDL_SCANCODE_E, SDL_SCANCODE_UNKNOWN, BtnA}, {SDL_SCANCODE_RETURN, SDL_SCANCODE_UNKNOWN, BtnA}, {SDL_SCANCODE_F, SDL_SCANCODE_UNKNOWN, BtnY},
        {SDL_SCANCODE_Q, SDL_SCANCODE_UNKNOWN, BtnL}, {SDL_SCANCODE_R, SDL_SCANCODE_UNKNOWN, BtnR}, {SDL_SCANCODE_Z, SDL_SCANCODE_UNKNOWN, BtnZL},
        {SDL_SCANCODE_C, SDL_SCANCODE_UNKNOWN, BtnZR}, {SDL_SCANCODE_V, SDL_SCANCODE_UNKNOWN, BtnLS}, {SDL_SCANCODE_G, SDL_SCANCODE_UNKNOWN, BtnRS},
        {SDL_SCANCODE_ESCAPE, SDL_SCANCODE_UNKNOWN, BtnPlus}, {SDL_SCANCODE_M, SDL_SCANCODE_UNKNOWN, BtnMinus}, {SDL_SCANCODE_1, SDL_SCANCODE_UNKNOWN, BtnLeft},
        {SDL_SCANCODE_2, SDL_SCANCODE_UNKNOWN, BtnUp}, {SDL_SCANCODE_3, SDL_SCANCODE_UNKNOWN, BtnRight}, {SDL_SCANCODE_4, SDL_SCANCODE_UNKNOWN, BtnDown}};
    for (auto& k : keys) if (down(k.a) || down(k.b)) in->buttons |= k.btn;
    static bool f9_was = false;
    bool f9 = down(SDL_SCANCODE_F9);
    if (f9 && !f9_was) {
        g_debug_frame_request = 1;
        printf("[hid] F9: frame capture requested\n");
    }
    f9_was = f9;
    g_f2_down = down(SDL_SCANCODE_F2);   // skip-opening toggle (host_input)
    g_f10_down = down(SDL_SCANCODE_F10);   // Graphics menu (hle_osd.cpp)
    auto axis = [&](SDL_Scancode neg, SDL_Scancode pos, SDL_Scancode neg2, SDL_Scancode pos2) { return (down(pos) || down(pos2) ? 1 : 0) - (down(neg) || down(neg2) ? 1 : 0); };
    int mx = axis(SDL_SCANCODE_A, SDL_SCANCODE_D, U, U), my = axis(SDL_SCANCODE_S, SDL_SCANCODE_W, U, U);
    int cx = axis(SDL_SCANCODE_LEFT, SDL_SCANCODE_RIGHT, SDL_SCANCODE_J, SDL_SCANCODE_L), cy = axis(SDL_SCANCODE_DOWN, SDL_SCANCODE_UP, SDL_SCANCODE_K, SDL_SCANCODE_I);
    float walk = down(SDL_SCANCODE_LCTRL) || down(SDL_SCANCODE_RCTRL) ? 0.45f : 1.0f;
    if (mx || my) {   // keys win over the pad stick while held
        float k = 32767.0f * walk / sqrtf((float)(mx * mx + my * my));
        in->lx = (s32)lrintf(mx * k); in->ly = (s32)lrintf(my * k);
    }
    if (cx || cy) {
        float k = 32767.0f / sqrtf((float)(cx * cx + cy * cy));
        in->rx = (s32)lrintf(cx * k); in->ry = (s32)lrintf(cy * k);
    }
}
#endif

// Several guest threads poll the npad every frame: read the host at most every 1 ms and share the result.
// The refresh clock is steady_clock (QueryPerformanceCounter): GetTickCount64 advances in 15.6 ms steps, so a read
// could get input up to ~15 ms old (input latency, user 2026-10-15). TOTK_INPUT_TICK=1: the old tick clock (A/B).
// "[hid] input age" logs how old the returned sample was at each read (avg / max per 10 s, first minute).
static HostInput host_input() {
    static std::mutex mtx;
    static HostInput cur;
    static u64 last_ms = ~0ull;
    static const bool tick = getenv("TOTK_INPUT_TICK") != nullptr;
    static std::chrono::steady_clock::time_point last_t, log_t0 = std::chrono::steady_clock::now();
    static double age_sum, age_max;
    static u64 age_n, age_logs;
    std::lock_guard<std::mutex> l(mtx);
    u64 now = GetTickCount64();
    auto now_t = std::chrono::steady_clock::now();
    bool refresh = last_ms == ~0ull ||
        (tick ? now - last_ms >= 2 : std::chrono::duration<double, std::milli>(now_t - last_t).count() >= 1.0);
    if (!refresh) {
        double age = std::chrono::duration<double, std::milli>(now_t - last_t).count();
        age_sum += age; age_n++; if (age > age_max) age_max = age;
    } else age_n++;
    if (age_logs < 6 && std::chrono::duration<double>(now_t - log_t0).count() >= 10.0) {
        printf("[hid] input age at read: avg %.2f ms, max %.2f ms (%llu reads, %s clock)\n", age_n ? age_sum / age_n : 0.0, age_max,
               (unsigned long long)age_n, tick ? "tick" : "1 ms");
        age_sum = age_max = 0; age_n = 0; age_logs++; log_t0 = now_t;
    }
    if (refresh) {
        last_ms = now;
        last_t = now_t;
        HostInput in = {};
        pad_read(&in);
        hidpad_read(&in);
        g_f2_down = false;
        keyboard_read(&in);
        // Graphics menu (hle_osd.cpp): L3 + R3 or F10; while it is open the game sees no input
        if (osd_menu_input(in.buttons | scripted_buttons(), g_f10_down)) in = {};
        {   // Shader warm-up screen (TOTK_WARMUP): A / B / + = play now, the warm-up keeps compiling in the background
            // (exec.c vk_precompile_play_now). That press is held back from the game until released, so it doesn't also
            // confirm the title menu.
            static bool held = true, swallow;   // (a button already down when the warm-up starts doesn't count)
            const u64 play = BtnA | BtnB | BtnPlus;
            bool down = ((in.buttons | scripted_buttons()) & play) != 0;
            if (down && !held && vk_precompile_warmup_active()) { vk_precompile_play_now(); swallow = true; }
            if (!down) swallow = false;
            held = down;
            if (swallow) in.buttons &= ~play;
        }
        cur = in;
        // Skip opening toggle while the title screen is up: F2 or Minus (the title menu has no use for Minus)
        static bool was = false;
        bool now_down = g_f2_down || ((in.buttons | scripted_buttons()) & BtnMinus);
        if (now_down && !was && g_title_phase) {
            g_skip_opening = !g_skip_opening;
            printf("[hid] skip opening %s\n", g_skip_opening ? "ON" : "OFF");
        }
        // In-game skip (user 09:5x: "u click a ui button in game and it takes u to when u wake up in the sky island"):
        // inside the opening stage, F2 or Minus held 1.5 s restarts the game into New Game with the skip on
        // (TOTK_AUTO_NEWGAME presses New Game -> Start; the skip overlay then ends the prologue at once).
        static u64 minus_since = 0;
        bool minus = ((in.buttons | scripted_buttons()) & BtnMinus) != 0;
        if (g_in_opening && minus) { if (!minus_since) minus_since = now; } else minus_since = 0;
        g_skip_hold_ms = minus_since ? (int)(now - minus_since) : 0;
        if (g_in_opening && ((g_f2_down && !was) || g_skip_hold_ms >= 1500)) skip_restart();
        was = now_down;
    }
    return cur;
}

// Quick menus at 60 fps (user 2026-10-17: "the wheels still make it all black"): with TOTK_FPS=60 the speed fix scales
// the game's interval ratio (main+0x7a83e4 this+8) to 0.5, and TotK then skips the 3D scene in the frame that captures
// the weapon / shield / material menu background (black menu; w4 log, A/B w5-w9: 30 fps, TOTK_SPEED_FIX=0 and a kept
// ratio all fix it, but the ratio also drives game speed). Those menus open on a held d-pad direction, so while one is
// held hle_vi.cpp runs Switch timing (interval 2, no scaling), and the game sees a new d-pad press only after it has been
// held 40 ms, so that timing is already in effect when the menu opens. TOTK_DPAD_30=0: off.
// Only for presses that can open a quick menu (user 2026-10-19: "it also occurs when i go into the regular menu and
// ultrahand. make sure it only activates in quick menus"): not while a pause menu or the map already holds the picture
// (no 3D scene drawn for a few frames; the game's time multiplier gives no such signal, run601), not R + Left / Right /
// Down (Ultrahand rotation; R + Up = throw materials), and a hold that opened no menu (the scene keeps being drawn)
// drops Switch timing.
extern "C" { volatile int g_dpad_switch_timing = 0; }   // read by hle_vi.cpp (pacing + speed fix)
extern "C" uint64_t vk_scene_idle_frames(void);         // nvn/vulkan/exec.c: presents since the 3D scene was last drawn
static u64 dpad_gate(u64 buttons) {
    static const bool on = !getenv("TOTK_DPAD_30") || strcmp(getenv("TOTK_DPAD_30"), "0");
    static const bool fps60 = getenv("TOTK_FPS") && atoi(getenv("TOTK_FPS")) >= 60;
    if (!on || !fps60) return buttons;
    // (Down = whistle, no quick menu: user 2026-10-19)
    const u64 dmask = buttons & BtnR ? BtnUp : BtnUp | BtnLeft | BtnRight;
    static u64 held_since, released_at;
    static int gated, opened;   // this hold: Switch timing on; a menu took the scene away
    u64 now = GetTickCount64();
    if (buttons & dmask) {
        if (!held_since) {
            held_since = now;
            u64 idle = vk_scene_idle_frames();
            gated = g_dpad_switch_timing || idle <= 4;
            opened = 0;
            printf("[dpad] press %llx: scene idle %llu frames -> %s\n", (unsigned long long)(buttons & dmask),
                   (unsigned long long)idle, gated ? "Switch timing" : "no gate (menu holds the picture)");
        }
        released_at = 0;
        // (gameplay itself shows idle 1 between a present and the next scene draw: run602, every press)
        if (gated && !opened && now - held_since > 40 && vk_scene_idle_frames() >= 3) {
            opened = 1;
            printf("[dpad] menu opened %llu ms after the press\n", (unsigned long long)(now - held_since));
        }
        if (gated && !opened && now - held_since > 400) {
            gated = 0;
            printf("[dpad] no menu 400 ms after the press: Switch timing off for this hold\n");
        }
        g_dpad_switch_timing = gated;
        if (gated && now - held_since < 40) buttons &= ~dmask;
    } else {
        held_since = 0;
        if (!released_at) released_at = now;
        if (now - released_at > 150) g_dpad_switch_timing = 0;   // (taps and quick re-presses keep Switch timing briefly)
    }
    return buttons;
}
static std::atomic<u64> g_cheat_keys{0};   // hle_cheat_vm.inc: the buttons the game saw last (HidNpadButton bits, like hid::GetKeysHeld)
static void npad_fill(NpadState* s, u32 id, bool style_ok) {
    memset(s, 0, sizeof *s);
    s->sampling_number = g_sample++;
    if (!npad_connected(id) || !style_ok) return;
    HostInput in = host_input();
    u64 sb = osd_menu_is_open() ? 0 : scripted_buttons();
    if (sb >> kScriptStick) {
        const s32 M = 32767;
        auto dir = [&](int b) { return (sb >> (kScriptStick + b)) & 1 ? M : 0; };
        if (dir(0) | dir(1) | dir(2) | dir(3)) { in.ly = dir(0) - dir(1); in.lx = dir(3) - dir(2); }
        if (dir(4) | dir(5) | dir(6) | dir(7)) { in.ry = dir(4) - dir(5); in.rx = dir(7) - dir(6); }
        sb &= (1ull << kScriptStick) - 1;
    }
    s->attributes = 1;   // IsConnected
    s->buttons = dpad_gate(in.buttons | sb);
    s->lx = in.lx; s->ly = in.ly; s->rx = in.rx; s->ry = in.ry;
    // stick-as-direction bits 16-23 (Eden NpadButton StickLLeft..StickRDown), set past half deflection
    const s32 h = 16384;
    s->buttons |= (u64)(in.lx < -h) << 16 | (u64)(in.ly > h) << 17 | (u64)(in.lx > h) << 18 | (u64)(in.ly < -h) << 19 |
                  (u64)(in.rx < -h) << 20 | (u64)(in.ry > h) << 21 | (u64)(in.rx > h) << 22 | (u64)(in.ry < -h) << 23;
    g_cheat_keys = s->buttons;
}

HLE_AS(hid_InitializeNpad, "_ZN2nn3hid14InitializeNpadEv") { }
HLE_AS(hid_SetSupportedNpadIdType, "_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm") { }
HLE_AS(hid_SetSupportedNpadStyleSet, "_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") { }
HLE_AS(hid_SetNpadJoyHoldType, "_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE") { }
HLE_AS(hid_DisconnectNpad, "_ZN2nn3hid14DisconnectNpadERKj") { }
HLE_AS(hid_GetNpadStyleSet, "_ZN2nn3hid15GetNpadStyleSetERKj") {
    c->x[0] = npad_connected(*(u32*)c->x[0]) ? StyleFullKey : 0;
}
// GetNpadStates(State* out, int count, const u32& id) -> void in the SDK; fill min(count,1) entries
#define NPAD_STATES(name, mangled, style)                                                           \
    HLE_AS(name, mangled) {                                                                         \
        NpadState* s = (NpadState*)c->x[0]; s32 n = (s32)c->x[1]; u32 id = *(u32*)c->x[2];          \
        for (s32 i = 0; i < n; i++) npad_fill(&s[i], id, (style) == StyleFullKey);                   \
    }
NPAD_STATES(hid_GetNpadStatesFullKey, "_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj", StyleFullKey)
NPAD_STATES(hid_GetNpadStatesJoyDual, "_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj", StyleJoyDual)
NPAD_STATES(hid_GetNpadStatesJoyLeft, "_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj", 0)
NPAD_STATES(hid_GetNpadStatesJoyRight, "_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj", 0)
NPAD_STATES(hid_GetNpadStatesHandheld, "_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj", StyleHandheld)

// handles: u32 {type, npad id, device idx} packed; any stable non-zero value works for our own getters
HLE_AS(hid_GetSixAxisSensorHandles, "_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") {
    u32* h = (u32*)c->x[0]; s32 n = (s32)c->x[1]; u32 id = *(u32*)c->x[2];
    for (s32 i = 0; i < n; i++) h[i] = 0x10000u | (id << 8) | (u32)i;
    c->x[0] = (u64)n;
}
HLE_AS(hid_GetVibrationDeviceHandles, "_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") {
    u32* h = (u32*)c->x[0]; s32 n = (s32)c->x[1]; u32 id = *(u32*)c->x[2];
    for (s32 i = 0; i < n; i++) h[i] = 0x20000u | (id << 8) | (u32)i;
    c->x[0] = (u64)n;
}
HLE_AS(hid_StartSixAxisSensor, "_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE") { }
HLE_AS(hid_InitializeVibrationDevice, "_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE") { }
// ---- rumble (user 2026-10-15: "rumble support for xbox dualsense and pro, with hd rumble on dual and pro")
// The game drives nn::hid's vibration node graph: VibrationPlayer (.bnvib clips), VibrationWriter (direct values),
// VibrationMixer, VibrationTarget (one device handle), joined by VibrationNodeConnection (with a modulation), plus
// SendVibrationValue. The SDK classes are polymorphic and their constructors set an SDK vtable that the game calls
// through later (its connection pool releases connections with a virtual destructor call, slot 0: a no-op constructor
// left the pool's free-list link there -> "indirect call to unknown target" on the first hit rumble, play.log 22:38).
// Each class gets a host vtable in (identity-mapped) guest memory; all node state stays host-side and the graph is
// evaluated by rumble_thread at 200 Hz (wall clock), not by the guest's VibrationNode::Update().
// Value per side (device index 0 = left, 1 = right) -> pads:
//   Xbox (XInput): low band -> left (big) motor, high band -> right motor, Eden/yuzu SDL curve (a + a^0.35) / 2 with
//   their frequency attenuation (research/sweep/w31).
//   DualSense: the same two motor values as classic rumble (output report 0x02 USB / 0x31 Bluetooth). TOTK_DS_HAPTICS=1
//   (Bluetooth only, EXPERIMENTAL: SAxense report 0x32, not confirmed on Windows) streams both bands of each side as a
//   3 kHz waveform to the voice-coil actuators instead (HD rumble).
//   Pro Controller / Joy-Con: HD rumble (output report 0x10) with the game's own frequencies and amplitudes.
// TOTK_RUMBLE=<gain> (default 1, 0 = off). BNVIB samples (observed): 4 bytes [amp lo, freq lo, amp hi, freq hi] at
// 200 Hz after a 12-byte header (04 00 00 00 03 00 c8 00 <u32 data bytes>); amp = b / 255, freq = 10 * 2^(b / 32)
// (0x80 = 160 Hz, 0xA0 = 320 Hz = the SDK defaults; inferred from the data, not documented).
namespace vib {
struct Val { float alo = 0, flo = 160, ahi = 0, fhi = 320; };
enum Cls { Conn, Target, Mixer, Player, Writer, NCls };
static const char* const kClsName[NCls] = {"VibrationNodeConnection", "VibrationTarget", "VibrationMixer", "VibrationPlayer", "VibrationWriter"};
constexpr int kSlots = 12;
static u64 g_vt[NCls][kSlots];
struct Conn_ { u64 src = 0, dst = 0; float mod[4] = {1, 1, 1, 1}; };
struct Play_ { u64 data = 0, size = 0, t0 = 0; bool loop = false, playing = false; s32 pos = 0; float speed = 1.0f; };
static std::mutex g_mu;
static std::unordered_map<u64, int> g_cls;       // node -> class (from its constructor)
static std::unordered_map<u64, Conn_> g_conn;
static std::unordered_map<u64, Play_> g_play;
static std::unordered_map<u64, Val> g_writer;    // VibrationWriter -> last written value
static std::unordered_map<u64, u32> g_target;    // VibrationTarget -> device handle
static Val g_direct[2];                          // SendVibrationValue per side
static u64 g_direct_t[2];
static std::atomic<u64> g_updates{0};

static u64 now_us() { return (u64)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void forget(u64 a) { g_cls.erase(a); g_conn.erase(a); g_play.erase(a); g_writer.erase(a); g_target.erase(a); }

template <int C, int S> static void slot(Ctx* c) {
    static std::atomic<bool> once{false};
    if (!once.exchange(true)) printf("[hid] vibration %s vtable slot %d called (this %llx)\n", kClsName[C], S, (unsigned long long)c->x[0]);
    if (S <= 1) {                                         // D1 / D0 (deleting: the object memory is left to the game's pool/heap)
        std::lock_guard<std::mutex> l(g_mu);
        forget(c->x[0]);
    }
    c->x[0] = 0;
}
template <int C, int... S> static void fill(std::integer_sequence<int, S...>) { ((g_vt[C][S] = rt_host_fn_addr(slot<C, S>)), ...); }
static u64 vtable(int c) {
    static std::once_flag f;
    std::call_once(f, [] {
        fill<Conn>(std::make_integer_sequence<int, kSlots>{}); fill<Target>(std::make_integer_sequence<int, kSlots>{});
        fill<Mixer>(std::make_integer_sequence<int, kSlots>{}); fill<Player>(std::make_integer_sequence<int, kSlots>{});
        fill<Writer>(std::make_integer_sequence<int, kSlots>{});
    });
    return (u64)&g_vt[c][0];
}

struct Clip { const u8* s = nullptr; u32 n = 0, rate = 200; };
static Clip clip(const Play_& p) {
    Clip c;
    if (!p.data || p.size < 12) return c;
    const u8* b = (const u8*)p.data;
    u32 meta = *(const u32*)b;                           // 4 = format + rate (0xC with loop points)
    if (meta < 4 || meta > 64 || 8ull + meta > p.size) return c;
    u16 rate = *(const u16*)(b + 6);
    if (rate >= 50 && rate <= 1000) c.rate = rate;
    u64 bytes = *(const u32*)(b + 4 + meta), avail = p.size - (8 + meta);
    c.s = b + 8 + meta; c.n = (u32)((bytes < avail ? bytes : avail) / 4);
    return c;
}
static double play_pos(const Play_& p, const Clip& k, u64 now) { return p.pos + (double)(now - p.t0) * 1e-6 * k.rate * p.speed; }
static Val player_val(const Play_& p, u64 now) {
    Val v;
    if (!p.playing) return v;
    Clip k = clip(p);
    if (!k.n) return v;
    double i = play_pos(p, k, now);
    if (i < 0) return v;
    u64 n = (u64)i;
    if (n >= k.n) { if (!p.loop) return v; n %= k.n; }
    const u8* q = k.s + 4 * n;
    v.alo = q[0] / 255.0f; v.flo = 10.0f * exp2f(q[1] / 32.0f); v.ahi = q[2] / 255.0f; v.fhi = 10.0f * exp2f(q[3] / 32.0f);
    return v;
}
static void mix(Val& o, const Val& v) {                   // per band: the stronger input wins (amplitude and frequency)
    if (v.alo > o.alo) { o.alo = v.alo; o.flo = v.flo; }
    if (v.ahi > o.ahi) { o.ahi = v.ahi; o.fhi = v.fhi; }
}
static Val eval(u64 node, u64 now, int depth) {
    Val o;
    if (depth > 8) return o;
    auto c = g_cls.find(node);
    int k = c == g_cls.end() ? -1 : c->second;
    if (k == Player) { auto p = g_play.find(node); return p == g_play.end() ? o : player_val(p->second, now); }
    if (k == Writer) { auto w = g_writer.find(node); return w == g_writer.end() ? o : w->second; }
    for (auto& [a, cn] : g_conn) {                        // Mixer / Target: inputs = connections ending here
        if (cn.dst != node || !cn.src || cn.src == node) continue;
        Val v = eval(cn.src, now, depth + 1);
        v.alo *= cn.mod[0]; v.flo *= cn.mod[1]; v.ahi *= cn.mod[2]; v.fhi *= cn.mod[3];
        mix(o, v);
    }
    return o;
}
static void gather(Val out[2], u64 now) {
    std::lock_guard<std::mutex> l(g_mu);
    for (auto& [t, h] : g_target) mix(out[(h & 0xFF) ? 1 : 0], eval(t, now, 0));
    for (int s = 0; s < 2; s++)
        if (g_direct_t[s] && now - g_direct_t[s] < 2000000) mix(out[s], g_direct[s]);   // (a lost "stop" ends after 2 s)
}
static Val read_val(u64 a) {                              // guest VibrationValue {amp lo, freq lo, amp hi, freq hi}
    const float* f = (const float*)a;
    Val v;
    auto ok = [](float x, float hi) { return x >= 0 && x <= hi; };
    if (ok(f[0], 1.5f)) v.alo = f[0];
    if (ok(f[1], 2000)) v.flo = f[1];
    if (ok(f[2], 1.5f)) v.ahi = f[2];
    if (ok(f[3], 2000)) v.fhi = f[3];
    return v;
}
static void logv(const char* what, u64 self, const float* f, std::atomic<int>& n) {
    if (n++ < 6) printf("[hid] %s %llx: %.3f %.1f %.3f %.1f\n", what, (unsigned long long)self, f[0], f[1], f[2], f[3]);
}
}  // namespace vib

HLE_AS(hid_SendVibrationValue, "_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE") {
    static std::atomic<int> n{0};
    vib::logv("SendVibrationValue", *(u32*)c->x[0], (const float*)c->x[1], n);
    int s = (*(u32*)c->x[0] & 0xFF) ? 1 : 0;
    std::lock_guard<std::mutex> l(vib::g_mu);
    vib::g_direct[s] = vib::read_val(c->x[1]); vib::g_direct_t[s] = vib::now_us();
}
#define VIB_CTOR(id, sym, cls) HLE_AS(id, sym) { \
    *(u64*)c->x[0] = vib::vtable(vib::cls); std::lock_guard<std::mutex> l(vib::g_mu); vib::forget(c->x[0]); vib::g_cls[c->x[0]] = vib::cls; }
VIB_CTOR(hid_VibConnC1, "_ZN2nn3hid23VibrationNodeConnectionC1Ev", Conn)
VIB_CTOR(hid_VibTargetC1, "_ZN2nn3hid15VibrationTargetC1Ev", Target)
VIB_CTOR(hid_VibMixerC2, "_ZN2nn3hid14VibrationMixerC2Ev", Mixer)
VIB_CTOR(hid_VibPlayerC2, "_ZN2nn3hid15VibrationPlayerC2Ev", Player)
VIB_CTOR(hid_VibWriterC1, "_ZN2nn3hid15VibrationWriterC1Ev", Writer)
#undef VIB_CTOR
HLE_AS(hid_VibMixerD2, "_ZN2nn3hid14VibrationMixerD2Ev") { std::lock_guard<std::mutex> l(vib::g_mu); vib::forget(c->x[0]); }
HLE_AS(hid_VibPlayerD2, "_ZN2nn3hid15VibrationPlayerD2Ev") { std::lock_guard<std::mutex> l(vib::g_mu); vib::forget(c->x[0]); }
HLE_AS(hid_VibConnConnect, "_ZN2nn3hid23VibrationNodeConnection7ConnectEPNS0_13VibrationNodeES3_") {
    static std::atomic<int> n{0};
    std::lock_guard<std::mutex> l(vib::g_mu);
    auto& cn = vib::g_conn[c->x[0]]; cn.src = c->x[1]; cn.dst = c->x[2];
    if (n++ < 8) {   // measurement: which node classes the game wires, and whether it replaced our vtables (derived classes)
        auto cls = [](u64 a) { auto i = vib::g_cls.find(a); return i == vib::g_cls.end() ? "?" : vib::kClsName[i->second]; };
        auto own = [](u64 a) { auto i = vib::g_cls.find(a); return i != vib::g_cls.end() && *(u64*)a == vib::vtable(i->second); };
        printf("[hid] vibration Connect %llx: %s %llx (%s vtable) -> %s %llx (%s vtable)\n", (unsigned long long)c->x[0], cls(c->x[1]),
               (unsigned long long)c->x[1], own(c->x[1]) ? "our" : "game", cls(c->x[2]), (unsigned long long)c->x[2], own(c->x[2]) ? "our" : "game");
    }
    c->x[0] = 1;
}
HLE_AS(hid_VibConnDisconnect, "_ZN2nn3hid23VibrationNodeConnection10DisconnectEv") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto i = vib::g_conn.find(c->x[0]); if (i != vib::g_conn.end()) i->second.src = i->second.dst = 0;
}
HLE_AS(hid_VibConnSetModulation, "_ZN2nn3hid23VibrationNodeConnection13SetModulationERKNS0_19VibrationModulationE") {
    static std::atomic<int> n{0};
    const float* f = (const float*)c->x[1];
    vib::logv("SetModulation", c->x[0], f, n);
    std::lock_guard<std::mutex> l(vib::g_mu);
    auto& cn = vib::g_conn[c->x[0]];
    for (int i = 0; i < 4; i++) cn.mod[i] = f[i] >= 0 && f[i] <= 16 ? f[i] : 1.0f;
}
HLE_AS(hid_VibNodeUpdate, "_ZN2nn3hid13VibrationNode6UpdateEv") { vib::g_updates++; }
// game classes derived from Mixer / Player call these from their own OnNextSampleRequired: hand them our value
HLE_AS(hid_VibMixerSample, "_ZN2nn3hid14VibrationMixer20OnNextSampleRequiredEPNS0_14VibrationValueEPNS_4util13IntrusiveListINS0_23VibrationNodeConnectionENS4_27IntrusiveListBaseNodeTraitsIS6_S6_EEEE") {
    std::lock_guard<std::mutex> l(vib::g_mu); *(vib::Val*)c->x[1] = vib::eval(c->x[0], vib::now_us(), 0);
}
HLE_AS(hid_VibPlayerSample, "_ZN2nn3hid15VibrationPlayer20OnNextSampleRequiredEPNS0_14VibrationValueEPNS_4util13IntrusiveListINS0_23VibrationNodeConnectionENS4_27IntrusiveListBaseNodeTraitsIS6_S6_EEEE") {
    std::lock_guard<std::mutex> l(vib::g_mu); *(vib::Val*)c->x[1] = vib::eval(c->x[0], vib::now_us(), 0);
}
HLE_AS(hid_VibTargetSetHandle, "_ZN2nn3hid15VibrationTarget24SetVibrationDeviceHandleERKNS0_21VibrationDeviceHandleE") {
    static std::atomic<int> n{0};
    u32 h = *(u32*)c->x[1];
    if (n++ < 4) printf("[hid] VibrationTarget %llx -> handle %x\n", (unsigned long long)c->x[0], h);
    std::lock_guard<std::mutex> l(vib::g_mu); vib::g_target[c->x[0]] = h;
}
HLE_AS(hid_VibTargetUnsetHandle, "_ZN2nn3hid15VibrationTarget26UnsetVibrationDeviceHandleEv") { std::lock_guard<std::mutex> l(vib::g_mu); vib::g_target.erase(c->x[0]); }
HLE_AS(hid_VibWriterWrite, "_ZN2nn3hid15VibrationWriter5WriteERKNS0_14VibrationValueE") {
    static std::atomic<int> n{0};
    vib::logv("VibrationWriter::Write", c->x[0], (const float*)c->x[1], n);
    std::lock_guard<std::mutex> l(vib::g_mu); vib::g_writer[c->x[0]] = vib::read_val(c->x[1]);
}
HLE_AS(hid_VibPlayerLoad, "_ZN2nn3hid15VibrationPlayer4LoadEPKvm") {
    static std::atomic<int> logged{0};
    if (logged++ < 2) {
        const u8* b = (const u8*)c->x[1]; char hex[3 * 32 + 1] = {};
        for (u64 i = 0; i < c->x[2] && i < 32; i++) sprintf(hex + 3 * i, "%02x ", b[i]);
        printf("[hid] VibrationPlayer::Load %llx size %llu: %s\n", (unsigned long long)c->x[0], (unsigned long long)c->x[2], hex);
    }
    std::lock_guard<std::mutex> l(vib::g_mu); auto& p = vib::g_play[c->x[0]]; p.data = c->x[1]; p.size = c->x[2]; p.playing = false; p.pos = 0;
    c->x[0] = 0;                                           // nn::Result success
}
HLE_AS(hid_VibPlayerUnload, "_ZN2nn3hid15VibrationPlayer6UnloadEv") { std::lock_guard<std::mutex> l(vib::g_mu); vib::g_play.erase(c->x[0]); }
HLE_AS(hid_VibPlayerIsLoaded, "_ZNK2nn3hid15VibrationPlayer8IsLoadedEv") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto i = vib::g_play.find(c->x[0]); c->x[0] = i != vib::g_play.end() && i->second.data ? 1 : 0;
}
HLE_AS(hid_VibPlayerPlay, "_ZN2nn3hid15VibrationPlayer4PlayEv") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto& p = vib::g_play[c->x[0]]; p.playing = true; p.t0 = vib::now_us();
}
HLE_AS(hid_VibPlayerStop, "_ZN2nn3hid15VibrationPlayer4StopEv") { std::lock_guard<std::mutex> l(vib::g_mu); vib::g_play[c->x[0]].playing = false; }
HLE_AS(hid_VibPlayerIsPlaying, "_ZNK2nn3hid15VibrationPlayer9IsPlayingEv") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto it = vib::g_play.find(c->x[0]);
    bool on = false;
    if (it != vib::g_play.end() && it->second.playing) {
        vib::Clip k = vib::clip(it->second);
        on = it->second.loop || vib::play_pos(it->second, k, vib::now_us()) < k.n;
    }
    c->x[0] = on ? 1 : 0;
}
HLE_AS(hid_VibPlayerSetLoop, "_ZN2nn3hid15VibrationPlayer7SetLoopEb") { std::lock_guard<std::mutex> l(vib::g_mu); vib::g_play[c->x[0]].loop = (c->x[1] & 1) != 0; }
HLE_AS(hid_VibPlayerIsLoop, "_ZNK2nn3hid15VibrationPlayer6IsLoopEv") { std::lock_guard<std::mutex> l(vib::g_mu); c->x[0] = vib::g_play[c->x[0]].loop ? 1 : 0; }
HLE_AS(hid_VibPlayerSetPos, "_ZN2nn3hid15VibrationPlayer18SetCurrentPositionEi") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto& p = vib::g_play[c->x[0]]; p.pos = (s32)c->x[1]; p.t0 = vib::now_us();
}
HLE_AS(hid_VibPlayerSetSpeed, "_ZN2nn3hid15VibrationPlayer12SetPlaySpeedEf") {
    std::lock_guard<std::mutex> l(vib::g_mu); auto& p = vib::g_play[c->x[0]];
    u64 now = vib::now_us();
    if (p.playing) { p.pos = (s32)vib::play_pos(p, vib::clip(p), now); p.t0 = now; }   // keep the position across the change
    float f = F(c, 0); p.speed = f >= 0 && f <= 16 ? f : 1.0f;
}

// ---- rumble output (200 Hz)
#ifdef _WIN32
typedef struct { WORD l, r; } XVib;                       // XINPUT_VIBRATION
typedef DWORD(WINAPI* XInputSetState_t)(DWORD, XVib*);
static XInputSetState_t xinput_set_load() {
    for (const char* n : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"})
        if (HMODULE m = LoadLibraryA(n))
            if (auto f = (XInputSetState_t)GetProcAddress(m, "XInputSetState")) return f;
    return nullptr;
}
#endif
static float rumble_gain() {
    static const float g = [] { const char* e = getenv("TOTK_RUMBLE"); float v = e && *e ? (float)atof(e) : 1.0f; return v < 0 ? 0.0f : v > 4 ? 4.0f : v; }();
    return g;
}
// Eden/yuzu SDL rumble for 2-motor pads: frequency attenuation, then (a + a^0.35) / 2 (w31)
static float motor_amp(float a, float f, bool high) {
    float att = high ? 1 - (f - 200) / 700 : 1 - (f - 140) / 400;
    att = att < 0.3f ? 0.3f : att > 1 ? 1.0f : att;
    a *= att;
    if (!(a > 0)) return 0;
    a = (a + powf(a, 0.35f)) * 0.5f;
    return a > 1 ? 1.0f : a;
}
// Pro Controller / Joy-Con HD rumble, one actuator (w31: dekuNukem / yuzu encoding). Neutral = 00 01 40 40.
static void pro_encode(u8* b, const vib::Val& v, float gain) {
    float alo = v.alo * gain, ahi = v.ahi * gain, sum = alo + ahi;
    if (sum > 1) { alo /= sum; ahi /= sum; }               // yuzu: keep the two bands from overdriving the actuator
    auto fidx = [](float f) { float i = roundf(log2f(fmaxf(f, 10.0f) / 10.0f) * 32.0f); return i > 255 ? 255 : (int)i; };
    auto aidx = [](float a) -> int {                       // amplitude code 0..100 (101-entry table)
        if (!(a > 0)) return 0;
        float i = a > 0.23f ? log2f(a * 8.7f) * 32 : a > 0.12f ? log2f(a * 17) * 16 : a / 0.12f * 16.46f;
        i = roundf(i);
        return i < 0 ? 0 : i > 100 ? 100 : (int)i;
    };
    int hf = (fidx(v.fhi) - 0x60) * 4, lf = fidx(v.flo) - 0x40;
    hf = hf < 0 ? 0 : hf > 0x1FF ? 0x1FF : hf;
    lf = lf < 0 ? 0 : lf > 0x7F ? 0x7F : lf;
    int ih = aidx(ahi), il = aidx(alo);
    int hf_amp = 2 * ih, lf_amp = ((il & 1) << 15) | (0x40 + (il >> 1));
    b[0] = (u8)(hf & 0xFF); b[1] = (u8)(hf_amp | ((hf >> 8) & 1)); b[2] = (u8)(lf | ((lf_amp >> 8) & 0x80)); b[3] = (u8)(lf_amp & 0xFF);
}
// DualSense Bluetooth haptics (SAxense, egormanga, read 2026-10-15): report 0x32, 142 bytes: [1] tag | seq << 4,
// packet 0x91 len 7 {FE 00 00 00 00 FF n} (n counts reports), packet 0x92 len 64 = 32 stereo int8 frames at 3 kHz
// (one report per 10.67 ms), CRC-32 of 0xA2 + bytes 0..137 at 138..141.
#ifdef _WIN32
struct Haptic { float ph[2][2] = {}; u8 seq = 0, n = 0; u64 next = 0; };
static void ds_haptics(HidPad* p, HANDLE w, const vib::Val v[2], float gain, Haptic* hs) {
    u8 b[142] = {};
    b[0] = 0x32; b[1] = (u8)((hs->seq++ & 15) << 4);
    u8* q = b + 2;
    const u8 p11[9] = {0x91, 7, 0xFE, 0, 0, 0, 0, 0xFF, hs->n++};
    memcpy(q, p11, 9);
    q[9] = 0x92; q[10] = 64;
    s8* smp = (s8*)(q + 11);
    for (int s = 0; s < 2; s++) {
        float alo = v[s].alo * gain, ahi = v[s].ahi * gain, sum = alo + ahi;
        if (sum > 1) { alo /= sum; ahi /= sum; }
        float dlo = 6.2831853f * v[s].flo / 3000.0f, dhi = 6.2831853f * v[s].fhi / 3000.0f;
        for (int i = 0; i < 32; i++) {
            float x = alo * sinf(hs->ph[s][0]) + ahi * sinf(hs->ph[s][1]);
            hs->ph[s][0] = fmodf(hs->ph[s][0] + dlo, 6.2831853f); hs->ph[s][1] = fmodf(hs->ph[s][1] + dhi, 6.2831853f);
            smp[2 * i + s] = (s8)lrintf(fmaxf(-1.0f, fminf(1.0f, x)) * 127.0f);
        }
    }
    u8 t[143]; t[0] = 0xA2; memcpy(t + 1, b, 138);
    u32 c = crc32_buf(t, 139);
    b[138] = (u8)c; b[139] = (u8)(c >> 8); b[140] = (u8)(c >> 16); b[141] = (u8)(c >> 24);
    if (!hid_write(w, b, 142, p->outlen)) {
        static std::atomic<int> told{0};
        if (told++ < 3) printf("[hid] DualSense haptics report 0x32 failed (%lu)\n", GetLastError());
    }
}
#endif
#ifdef _WIN32
static void rumble_thread() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    XInputSetState_t xset = xinput_set_load();
    const float gain = rumble_gain();
    const bool haptics = ds_haptics_on();
    printf("[hid] rumble: gain %.2f, XInput %s, DualSense %s\n", gain, xset ? "yes" : "no", haptics ? "HD haptics over Bluetooth (experimental)" : "classic rumble");
    struct Sent { u8 b[8] = {}; u64 t = 0; bool nz = false; } sent[4], xs_sent;
    Haptic hap[4];
    int first = 0;
    u64 upd_t = vib::now_us(), upd_logs = 0, out_n = 0;
    for (;;) {
        LARGE_INTEGER due; due.QuadPart = -50000;          // 5 ms
        if (timer && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, 20);
        else Sleep(5);
        u64 now = vib::now_us();
        if (upd_logs < 3 && now - upd_t >= 10000000) {     // measurement: how often the game runs VibrationNode::Update
            printf("[hid] rumble: VibrationNode::Update %.1f/s, outputs %.1f/s\n", vib::g_updates.exchange(0) / ((now - upd_t) * 1e-6),
                   out_n / ((now - upd_t) * 1e-6));
            upd_t = now; upd_logs++; out_n = 0;
        }
        vib::Val v[2];
        if (gain > 0) vib::gather(v, now);
        float lo = fminf(1.0f, fmaxf(motor_amp(v[0].alo, v[0].flo, false), motor_amp(v[1].alo, v[1].flo, false)) * gain);
        float hi = fminf(1.0f, fmaxf(motor_amp(v[0].ahi, v[0].fhi, true), motor_amp(v[1].ahi, v[1].fhi, true)) * gain);
        bool any = lo > 0.004f || hi > 0.004f;
        if (any && first < 3) {
            first++;
            printf("[hid] rumble: L %.2f@%.0f/%.2f@%.0f R %.2f@%.0f/%.2f@%.0f -> motors %.2f %.2f\n", v[0].alo, v[0].flo, v[0].ahi, v[0].fhi,
                   v[1].alo, v[1].flo, v[1].ahi, v[1].fhi, lo, hi);
        }
        // Xbox: on change (at most 100/s) and a 1-s refresh while on
        int xs = g_xslot;
        if (xset && xs >= 0) {
            XVib xv = {(WORD)lrintf(lo * 65535), (WORD)lrintf(hi * 65535)};
            bool ch = memcmp(xs_sent.b, &xv, 4) != 0;
            if ((ch && now - xs_sent.t >= 10000) || (any && now - xs_sent.t >= 1000000)) {
                xset((DWORD)xs, &xv); memcpy(xs_sent.b, &xv, 4); xs_sent.t = now; out_n++;
            }
        }
        for (int i = 0; i < 4; i++) {
            HidPad* p = &g_hidpad[i];
            HANDLE w = p->wr;
            int k = p->kind;
            if (!p->live || w == INVALID_HANDLE_VALUE) continue;
            Sent& s = sent[i];
            if (k == PadDualSense && haptics && p->bt) {    // stream continuously (silence = zero samples)
                if (now < hap[i].next) continue;
                hap[i].next = (hap[i].next && now - hap[i].next < 50000 ? hap[i].next : now) + 10667;
                ds_haptics(p, w, v, gain, &hap[i]); out_n++;
            } else if (k == PadDualSense) {
                u8 m[2] = {(u8)lrintf(lo * 255), (u8)lrintf(hi * 255)};
                bool ch = memcmp(s.b, m, 2) != 0;
                if ((ch && now - s.t >= 10000) || (any && now - s.t >= 500000)) {
                    ds_effects(p, w, m[0], m[1], false); memcpy(s.b, m, 2); s.t = now; out_n++;
                }
            } else if (k == PadPro || k == PadJoyL || k == PadJoyR) {
                u8 e[8];
                pro_encode(e, v[0], gain); pro_encode(e + 4, v[1], gain);
                bool nz = (e[1] & 0xFE) || (e[3] != 0x40 && e[3] != 0x00) || (e[2] & 0x80);
                bool ch = memcmp(s.b, e, 8) != 0;
                if ((ch && now - s.t >= 10000) || ((nz || s.nz) && now - s.t >= 40000)) {   // keep-alive while on
                    u8 r[10] = {0x10, (u8)(p->seq++ & 15)};
                    memcpy(r + 2, e, 8);
                    if (!hid_write(w, r, 10, p->outlen)) {
                        static std::atomic<int> told{0};
                        if (told++ < 3) printf("[hid] HD rumble report failed (%lu)\n", GetLastError());
                    }
                    memcpy(s.b, e, 8); s.t = now; s.nz = nz; out_n++;
                }
            }
        }
    }
}
#else
// Linux: the same graph evaluation and motor curve, out through SDL_RumbleGamepad on the open gamepad (two motors; no HD rumble).
static void rumble_thread() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const float gain = rumble_gain();
    printf("[hid] rumble: gain %.2f, SDL_RumbleGamepad\n", gain);
    u64 sent_t = 0; Uint16 sent_l = 0, sent_h = 0;
    int first = 0;
    for (;;) {
        LARGE_INTEGER due; due.QuadPart = -50000;          // 5 ms
        if (timer && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, 20);
        else Sleep(5);
        u64 now = vib::now_us();
        vib::Val v[2];
        if (gain > 0) vib::gather(v, now);
        float lo = fminf(1.0f, fmaxf(motor_amp(v[0].alo, v[0].flo, false), motor_amp(v[1].alo, v[1].flo, false)) * gain);
        float hi = fminf(1.0f, fmaxf(motor_amp(v[0].ahi, v[0].fhi, true), motor_amp(v[1].ahi, v[1].fhi, true)) * gain);
        bool any = lo > 0.004f || hi > 0.004f;
        if (any && first < 3) {
            first++;
            printf("[hid] rumble: L %.2f@%.0f/%.2f@%.0f R %.2f@%.0f/%.2f@%.0f -> motors %.2f %.2f\n", v[0].alo, v[0].flo, v[0].ahi, v[0].fhi,
                   v[1].alo, v[1].flo, v[1].ahi, v[1].fhi, lo, hi);
        }
        // on change (at most 100/s) and a 0.5-s refresh while on (each call runs the motors for 1 s)
        Uint16 l = (Uint16)lrintf(lo * 65535), h = (Uint16)lrintf(hi * 65535);
        bool ch = l != sent_l || h != sent_h;
        if ((ch && now - sent_t >= 10000) || (any && now - sent_t >= 500000)) {
            std::lock_guard<std::mutex> lk(g_gp_mu);
            if (g_gp && g_gp_rumble) SDL_RumbleGamepad(g_gp, l, h, 1000);
            sent_l = l; sent_h = h; sent_t = now;
        }
    }
}
#endif
static void rumble_start() { std::thread(rumble_thread).detach(); }

struct SixAxisState {       // delta_time, sampling_number, accel, gyro, angle, direction (3x3), attributes
    s64 delta_time, sampling_number; float acc[3], gyro[3], angle[3], dir[9]; u32 attributes, reserved;
};
static_assert(sizeof(SixAxisState) == 0x60);
// Motion from the first pad with an IMU (a Joy-Con pair: device index 1 = the right Joy-Con); none = lying still.
static HidPad* imu_pad(u32 handle) {
    if (!gyro_on()) return nullptr;
    int dev = handle & 0xFF;
    HidPad* any = nullptr;
    for (auto& p : g_hidpad) {
        if (!p.live || !p.has_imu) continue;
        int k = p.kind;
        if ((dev == 1 && k == PadJoyR) || (dev == 0 && k == PadJoyL)) return &p;
        if (!any || (k != PadJoyL && k != PadJoyR)) any = &p;
    }
    return any;
}
HLE_AS(hid_GetSixAxisSensorStates, "_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE") {
    SixAxisState* s = (SixAxisState*)c->x[0]; s32 n = (s32)c->x[1];
    HidPad* p = imu_pad(*(u32*)c->x[2]);
    s32 k = n > 0 ? 1 : 0;
    for (s32 i = 0; i < k; i++) {
        memset(&s[i], 0, sizeof s[i]);
        s[i].sampling_number = g_sample++; s[i].delta_time = 5000000;   // 5 ms
        s[i].acc[2] = -1.0f; s[i].dir[0] = s[i].dir[4] = s[i].dir[8] = 1.0f;
        if (p) {
            std::lock_guard<std::mutex> l(p->mmu);
            const Motion& m = p->m;
            s[i].delta_time = m.dt_ns;
            memcpy(s[i].acc, m.acc, sizeof m.acc); memcpy(s[i].gyro, m.gyro, sizeof m.gyro);
            memcpy(s[i].angle, m.angle, sizeof m.angle); memcpy(s[i].dir, m.dir, sizeof m.dir);
        }
        s[i].attributes = 1;                                           // IsConnected
    }
    c->x[0] = (u64)k;
}

HLE_AS(hid_InitializeMouse, "_ZN2nn3hid15InitializeMouseEv") { }
HLE_AS(hid_InitializeKeyboard, "_ZN2nn3hid18InitializeKeyboardEv") { }
HLE_AS(hid_GetMouseStates, "_ZN2nn3hid14GetMouseStatesEPNS0_10MouseStateEi") {   // -> int
    s32 n = (s32)c->x[1]; s32 k = n > 0 ? 1 : 0;
    if (k) { memset((void*)c->x[0], 0, 0x28); *(s64*)c->x[0] = g_sample++; }
    c->x[0] = (u64)k;
}
HLE_AS(hid_GetKeyboardState, "_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE") {
    memset((void*)c->x[0], 0, 0x30); *(s64*)c->x[0] = g_sample++;
}

HLE_AS(hid_ControllerSupportArg_SetDefault, "_ZN2nn3hid20ControllerSupportArg10SetDefaultEv") { }
HLE_AS(hid_ShowControllerSupport, "_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE") {
    if (c->x[0]) { s32* r = (s32*)c->x[0]; r[0] = 1; r[1] = 0; }     // {player_count, selected_id}
    c->x[0] = 0;
}

#include "hle_cheat_vm.inc"

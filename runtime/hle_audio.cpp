// HLE: nn::audio as a real renderer (TotK: 48 kHz, 240-sample frames). Voices (PCM16 / PCM float / DSP-ADPCM) are
// resampled (Catmull-Rom) by sampleRate * pitch, filtered by their two biquads, scaled with per-frame volume ramps and
// mixed into their destination (final mix, sub mix, or each destination of a splitter); sub mixes are processed deepest
// first (effects, then volume x mix matrix into their destination); the final mix goes to the device sink and WASAPI.
// The renderer's system event ticks once per rendered frame, so the game's audio thread runs at the device clock.
// Effects: aux send/return (the game's own DSP; planar per-frame blocks, bypass while the game returns nothing),
// biquad, buffer mixer, delay; reverb / I3DL2 reverb / light limiter pass the dry signal (TODO).
// Struct layouts: research/sweep/w18_nn_audio_api.md (WaveBuffer 0x30, AdpcmParameter u16[16], AdpcmContext 6 B,
// BiquadFilterParameter 0xC, PlayState Play 0 / Stop 1 / Pause 2, SampleFormat PcmInt16 2 / PcmFloat 5 / Adpcm 6).
// Env: TOTK_AUDIO=null (no device, old pacing only), TOTK_AUX=interleaved (default)|planar|bypass, TOTK_AUDIO_LOG=1 (calls),
// TOTK_VOLUME=<0..2> master gain.
// History: run82 OpenAudioRenderer UNIMPL blocked the audio thread forever (event never signalled); run84 a 0-sample
// looping buffer spun under the lock. Both handled below (event ticks per frame; empty loops produce silence).
#include "hle.h"
#define NOMINMAX
#include <windows.h>
#ifdef _WIN32
#include <mmreg.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#else
#include <SDL3/SDL.h>     // Linux: the device sink is an SDL3 audio stream (PipeWire / PulseAudio / ALSA)
#endif
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <array>
#include <deque>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

void hle_sysevent_init(u64 ev, bool autoclear);
void hle_sysevent_signal(u64 ev);

namespace {
constexpr int kMaxCh = 24, kMaxVoiceCh = 6, kFrameMax = 1024;
std::mutex g_mtx;
int g_rate = 48000, g_count = 240;
std::atomic<u64> g_event{0}, g_frames{0}, g_appends{0}, g_releases{0};
std::atomic<bool> g_running{false};
bool g_log = getenv("TOTK_AUDIO_LOG") != nullptr;
float g_master = getenv("TOTK_VOLUME") ? (float)atof(getenv("TOTK_VOLUME")) : 1.0f;
int g_aux_mode = [] { const char* e = getenv("TOTK_AUX"); return !e ? 2 : !strcmp(e, "bypass") ? 0 : !strcmp(e, "planar") ? 1 : 2; }();   // default interleaved (run220)
std::atomic<int> g_logs{0};
#define ALOG(...) do { if (g_log && g_logs++ < 3000) printf(__VA_ARGS__); } while (0)

u32 rd32(u64 a) { return *(const u32*)a; }

// ------------------------------------------------------------------ mixes, splitters, effects
struct Matrix { float v[kMaxCh][kMaxCh] = {}, prev[kMaxCh][kMaxCh] = {}; };
struct Mix {
    bool final = false; int nbuf = 2; float volume = 1, prev_volume = 1;
    u64 dest = 0; bool dest_splitter = false;
    Matrix* mat = nullptr;                 // src buffer -> dest buffer (sub mix with a mix destination)
    std::vector<u64> effects;
    float buf[kMaxCh][kFrameMax];
    int depth = 0;
};
struct SplitDest { u64 mix = 0; Matrix m; };
struct Splitter { std::map<int, SplitDest> dests; };
enum EffKind { E_AUX, E_BUFMIX, E_DELAY, E_REVERB, E_I3DL2, E_BIQUAD, E_LIMITER };
struct Effect {
    EffKind kind; u64 mix = 0; bool enabled = false;
    s8 in[kMaxCh] = {}, out[kMaxCh] = {}; int count = 0;
    std::map<std::string, float> f; std::map<std::string, s64> t; std::map<std::string, int> i;
    u64 buffer = 0, size = 0;
    // aux: host rings of s32 samples
    std::vector<s32> send, ret; size_t send_w = 0, send_r = 0, ret_w = 0, ret_r = 0; bool returned = false;
    // biquad
    s16 num[3] = {16384, 0, 0}, den[2] = {0, 0}; float bq_s[kMaxCh][2] = {};
    // buffer mixer
    float bm_vol[kMaxCh] = {};
    // delay
    std::vector<float> line[kMaxCh]; size_t pos = 0; float lp[kMaxCh] = {};
    // reverb / I3DL2 reverb: Freeverb-style late reverb (8 damped combs + 4 allpasses per channel) behind a predelay
    // line with 4 early-reflection taps; not the DSP's exact algorithm, parameters mapped by meaning
    struct Line { std::vector<float> b; size_t i = 0; float s = 0; };
    std::shared_ptr<std::array<std::array<Line, 13>, kMaxCh>> verb;   // [ch][0..7 combs, 8..11 allpasses, 12 predelay]
};
std::unordered_map<u64, Mix*> g_mixes;
// level meters (peak since the last [audio] line): raw decoded voice, voice after volume/filters, mixes, device out
float g_pk_raw = 0, g_pk_voice = 0, g_pk_sub = 0, g_pk_final = 0, g_pk_out = 0; double g_render_us = 0, g_render_max = 0; u64 g_render_n = 0;
u64 g_starved = 0, g_nodest = 0;
float peak(const float* x, int n) { float p = 0; for (int i = 0; i < n; i++) p = std::max(p, std::fabs(x[i])); return p; }
std::unordered_map<u64, Splitter> g_splitters;
std::unordered_map<u64, Effect> g_effects;
struct Sink { u64 final = 0; s8 in[kMaxCh] = {}; int count = 0; bool used = false; };
std::unordered_map<u64, Sink> g_sinks;

// The SDK's object types (VoiceType, SubMixType, FinalMixType, SplitterType, effect types, sinks, pools) start with a
// pointer to the renderer's info for that object; the game tests it for validity before routing (main 0x29dd380:
// SetSubMixDestination / SetSplitterDestination only run when the mix wrapper reports valid). run216: no handle
// written -> no routing call ever made -> every voice reached nothing. Acquire/Add writes a host block, release clears.
std::mutex g_hmtx;
std::unordered_map<u64, void*> g_handles;
void handle_set(u64 obj) {
    std::lock_guard<std::mutex> l(g_hmtx);
    void*& h = g_handles[obj];
    if (!h) h = calloc(1, 0x400);
    *(u64*)obj = (u64)h;
}
void handle_clear(u64 obj) { *(u64*)obj = 0; }

Mix* mix_get(u64 a, bool final) {
    Mix*& m = g_mixes[a];
    if (!m) { m = new Mix(); m->final = final; }
    return m;
}

// ------------------------------------------------------------------ voices
struct WaveBuf {
    u64 guest; u64 data, size; s32 start, end; bool loop, eos; u64 ctx, ctx_size; bool sent;
};
struct Voice {
    int rate = 48000, nch = 1, fmt = 2, prio = 0;
    int state = 1, pending_state = 1;      // PlayState: 0 play, 1 stop, 2 pause
    float volume = 1, prev_volume = 1, pitch = 1;
    u64 dest = 0; bool dest_splitter = false;
    Matrix mat;                            // voice channel -> destination buffer
    s16 coef[16] = {};
    struct { bool on; s16 num[3], den[2]; float s[kMaxVoiceCh][2]; } bq[2] = {};
    std::deque<WaveBuf> pending, released;
    s32 cur = -1;                          // sample cursor in pending.front() (-1: not started)
    s16 hist1 = 0, hist2 = 0;              // ADPCM history
    std::deque<float> look[kMaxVoiceCh];   // decoded but not yet consumed
    float prevs[kMaxVoiceCh] = {};          // sample before the current position
    double frac = 0;
    u64 played = 0;
    bool started_once = false;
};
std::unordered_map<u64, Voice> g_voices;

void adpcm_load_ctx(Voice& v, const WaveBuf& w) {
    if (w.ctx && w.ctx_size >= 6) { const s16* p = (const s16*)w.ctx; v.hist1 = p[1]; v.hist2 = p[2]; }
}
// one decoded sample frame (all channels) from the wave buffer queue; false when starved
bool next_sample(Voice& v, float* out) {
    for (int guard = 0; guard < 8; guard++) {
        while (!v.pending.empty() && !v.pending.front().sent) return false;
        if (v.pending.empty()) return false;
        WaveBuf& w = v.pending.front();
        if (v.cur < 0) { v.cur = w.start; if (v.fmt == 6) { if (w.ctx) adpcm_load_ctx(v, w); else if (w.start == 0 && !v.started_once) v.hist1 = v.hist2 = 0; } v.started_once = true; }
        if (v.cur < w.end && w.data) {
            s32 s = v.cur++;
            if (v.fmt == 2) { const s16* p = (const s16*)w.data; for (int c = 0; c < v.nch; c++) out[c] = p[(size_t)s * v.nch + c] * (1.0f / 32768.0f); }
            else if (v.fmt == 5) { const float* p = (const float*)w.data; for (int c = 0; c < v.nch; c++) out[c] = p[(size_t)s * v.nch + c]; }
            else if (v.fmt == 6) {
                const u8* fr = (const u8*)w.data + (size_t)(s / 14) * 8;
                if ((size_t)(s / 14) * 8 + 8 > w.size) { out[0] = 0; }
                else {
                    int k = s % 14; u8 hdr = fr[0]; int scale = 1 << (hdr & 0xf), pred = (hdr >> 4) & 7;
                    u8 b = fr[1 + k / 2]; int nib = (k & 1) ? (b & 0xf) : (b >> 4); if (nib >= 8) nib -= 16;
                    s32 c1 = v.coef[pred * 2], c2 = v.coef[pred * 2 + 1];
                    s32 val = ((nib * scale) << 11) + 1024 + c1 * v.hist1 + c2 * v.hist2;
                    val >>= 11; val = val < -32768 ? -32768 : val > 32767 ? 32767 : val;
                    v.hist2 = v.hist1; v.hist1 = (s16)val;
                    out[0] = val * (1.0f / 32768.0f);
                }
            } else { for (int c = 0; c < v.nch; c++) out[c] = 0; }
            v.played++;
            return true;
        }
        // buffer finished
        if (w.loop && w.end > w.start) { v.cur = w.start; if (v.fmt == 6) adpcm_load_ctx(v, w); continue; }
        if (w.loop) return false;              // empty looping buffer: silence (run84)
        if (w.eos) v.played = 0;
        v.released.push_back(w); v.pending.pop_front(); v.cur = -1; g_releases++;
    }
    return false;
}
float crom(float a, float b, float c, float d, float t) {   // Catmull-Rom between b and c
    return b + 0.5f * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}
// renders n output samples of the voice's channels into out[ch][n]
void voice_render(Voice& v, int n, float out[][kFrameMax]) {
    double step = (double)v.rate * (v.pitch > 0 ? v.pitch : 0) / g_rate;
    if (step <= 0) { for (int c = 0; c < v.nch; c++) memset(out[c], 0, n * sizeof(float)); return; }
    if (step > 8) step = 8;
    double endpos = v.frac + (n - 1) * step;
    int need = std::max((int)endpos + 3, (int)(v.frac + n * step) + 1);   // positions 0..need-1 (+ prevs at -1); covers the advance
    static thread_local std::vector<float> T[kMaxVoiceCh];
    float s[kMaxVoiceCh];
    for (int c = 0; c < v.nch; c++) { T[c].resize(need + 1); T[c][0] = v.prevs[c]; }
    int have = 0;
    for (; have < need; have++) {
        bool got;
        if (!v.look[0].empty()) { for (int c = 0; c < v.nch; c++) { s[c] = v.look[c].front(); v.look[c].pop_front(); } got = true; }
        else got = next_sample(v, s);
        if (!got) break;
        for (int c = 0; c < v.nch; c++) T[c][have + 1] = s[c];
    }
    if (have < need && !v.pending.empty()) g_starved++;
    for (int c = 0; c < v.nch; c++) { for (int k = 0; k < have; k++) g_pk_raw = std::max(g_pk_raw, std::fabs(T[c][k + 1])); for (int k = have; k < need; k++) T[c][k + 1] = 0; }
    for (int c = 0; c < v.nch; c++) {
        const float* t = T[c].data();
        for (int i = 0; i < n; i++) {
            double x = v.frac + i * step; int ip = (int)x; float f = (float)(x - ip);
            out[c][i] = crom(t[ip], t[ip + 1], t[ip + 2], t[ip + 3], f);
        }
    }
    double adv_d = v.frac + n * step; int adv = (int)adv_d; v.frac = adv_d - adv;
    // keep decoded-but-unconsumed samples (positions adv..have-1) and the sample before the new position
    for (int c = 0; c < v.nch; c++) {
        v.prevs[c] = adv >= 1 && adv <= have ? T[c][adv] : (adv == 0 ? v.prevs[c] : 0);
        std::deque<float> keep;
        for (int k = adv; k < have; k++) keep.push_back(T[c][k + 1]);
        v.look[c].insert(v.look[c].begin(), keep.begin(), keep.end());
    }
    // biquads (Q14, transposed direct form II with the denominator already negated, as the DSP)
    for (int b = 0; b < 2; b++) {
        if (!v.bq[b].on) continue;
        float b0 = v.bq[b].num[0] / 16384.f, b1 = v.bq[b].num[1] / 16384.f, b2 = v.bq[b].num[2] / 16384.f, a1 = v.bq[b].den[0] / 16384.f, a2 = v.bq[b].den[1] / 16384.f;
        for (int c = 0; c < v.nch; c++) {
            float s0 = v.bq[b].s[c][0], s1 = v.bq[b].s[c][1];
            for (int i = 0; i < n; i++) { float x = out[c][i], y = b0 * x + s0; s0 = s1 + b1 * x + a1 * y; s1 = b2 * x + a2 * y; out[c][i] = y; }
            v.bq[b].s[c][0] = s0; v.bq[b].s[c][1] = s1;
        }
    }
    // volume ramp
    float g0 = v.prev_volume, g1 = v.volume;
    for (int c = 0; c < v.nch; c++) for (int i = 0; i < n; i++) out[c][i] *= g0 + (g1 - g0) * i / n;
    v.prev_volume = v.volume;
}
void accumulate(const float src[][kFrameMax], int nsrc, Mix* dst, Matrix& m, int n, float gain) {
    if (!dst) return;
    for (int s = 0; s < nsrc && s < kMaxCh; s++)
        for (int d = 0; d < dst->nbuf && d < kMaxCh; d++) {
            float a = m.prev[s][d] * gain, b = m.v[s][d] * gain;
            if (a == 0 && b == 0) continue;
            float* o = dst->buf[d]; const float* x = src[s];
            if (a == b) for (int i = 0; i < n; i++) o[i] += x[i] * b;
            else for (int i = 0; i < n; i++) o[i] += x[i] * (a + (b - a) * i / n);
        }
    for (int s = 0; s < nsrc && s < kMaxCh; s++) memcpy(m.prev[s], m.v[s], sizeof m.v[s]);
}
void send_to(u64 dest, bool splitter, const float src[][kFrameMax], int nsrc, Matrix* m, int n, float gain) {
    if (!dest) return;
    if (splitter) {
        auto it = g_splitters.find(dest);
        if (it == g_splitters.end()) return;
        for (auto& [idx, d] : it->second.dests) if (d.mix) { auto mi = g_mixes.find(d.mix); if (mi != g_mixes.end()) accumulate(src, nsrc, mi->second, d.m, n, gain); }
    } else if (m) {
        auto mi = g_mixes.find(dest);
        if (mi != g_mixes.end()) accumulate(src, nsrc, mi->second, *m, n, gain);
    }
}
int mix_depth(Mix* m, int guard = 0) {
    if (!m || m->final || guard > 32 || !m->dest) return 0;
    if (m->dest_splitter) {
        int d = 0; auto it = g_splitters.find(m->dest);
        if (it != g_splitters.end()) for (auto& [k, sd] : it->second.dests) { auto mi = g_mixes.find(sd.mix); if (mi != g_mixes.end()) d = std::max(d, mix_depth(mi->second, guard + 1)); }
        return d + 1;
    }
    auto mi = g_mixes.find(m->dest);
    return mi == g_mixes.end() ? 1 : mix_depth(mi->second, guard + 1) + 1;
}

void effect_run(Effect& e, Mix* m, int n) {
    auto copy_through = [&] { for (int k = 0; k < e.count; k++) { int a = e.in[k], b = e.out[k]; if (a != b && a >= 0 && b >= 0 && a < m->nbuf && b < m->nbuf) memcpy(m->buf[b], m->buf[a], n * sizeof(float)); } };
    if (!e.enabled) { copy_through(); return; }
    switch (e.kind) {
    case E_AUX: {
        if (g_aux_mode == 0 || e.send.empty()) { copy_through(); return; }
        size_t cap = e.send.size();
        for (int k = 0; k < e.count; k++) {                         // send: planar blocks of n samples per channel
            int a = e.in[k]; if (a < 0 || a >= m->nbuf) continue;
            for (int i = 0; i < n; i++) {
                size_t idx = g_aux_mode == 2 ? e.send_w + (size_t)i * e.count + k : e.send_w + (size_t)k * n + i;
                float x = m->buf[a][i]; x = x > 1 ? 1 : x < -1 ? -1 : x;
                e.send[idx % cap] = (s32)(x * 32767.0f);             // 16-bit range in s32, like the DSP mix buffers (yuzu)
            }
        }
        e.send_w += (size_t)e.count * n;
        if (e.send_w - e.send_r > cap) e.send_r = e.send_w - cap;  // reader too slow: drop oldest
        size_t avail = e.ret_w - e.ret_r;
        if (e.returned && avail >= (size_t)e.count * n) {
            size_t rc = e.ret.size();
            for (int k = 0; k < e.count; k++) {
                int b = e.out[k]; if (b < 0 || b >= m->nbuf) continue;
                for (int i = 0; i < n; i++) {
                    size_t idx = g_aux_mode == 2 ? e.ret_r + (size_t)i * e.count + k : e.ret_r + (size_t)k * n + i;
                    m->buf[b][i] = e.ret[idx % rc] * (1.0f / 32768.0f);
                }
            }
            e.ret_r += (size_t)e.count * n;
        } else copy_through();
        return;
    }
    case E_BIQUAD: {
        float b0 = e.num[0] / 16384.f, b1 = e.num[1] / 16384.f, b2 = e.num[2] / 16384.f, a1 = e.den[0] / 16384.f, a2 = e.den[1] / 16384.f;
        for (int k = 0; k < e.count; k++) {
            int a = e.in[k], b = e.out[k]; if (a < 0 || b < 0 || a >= m->nbuf || b >= m->nbuf) continue;
            float s0 = e.bq_s[k][0], s1 = e.bq_s[k][1];
            for (int i = 0; i < n; i++) { float x = m->buf[a][i], y = b0 * x + s0; s0 = s1 + b1 * x + a1 * y; s1 = b2 * x + a2 * y; m->buf[b][i] = y; }
            e.bq_s[k][0] = s0; e.bq_s[k][1] = s1;
        }
        return;
    }
    case E_BUFMIX:
        for (int k = 0; k < e.count; k++) {
            int a = e.in[k], b = e.out[k]; if (a < 0 || b < 0 || a >= m->nbuf || b >= m->nbuf) continue;
            for (int i = 0; i < n; i++) m->buf[b][i] += m->buf[a][i] * e.bm_vol[k];
        }
        return;
    case E_DELAY: {
        s64 dt = e.t.count("time") ? e.t["time"] : 0;
        size_t len = (size_t)std::max<s64>(1, dt / 1000 * g_rate / 1000000);
        float in_g = e.f.count("in") ? e.f["in"] : 1, fb = e.f.count("feedback") ? e.f["feedback"] : 0, dry = e.f.count("dry") ? e.f["dry"] : 1,
              lpa = e.f.count("lowpass") ? e.f["lowpass"] : 0;
        for (int k = 0; k < e.count; k++) {
            int a = e.in[k], b = e.out[k]; if (a < 0 || b < 0 || a >= m->nbuf || b >= m->nbuf) continue;
            auto& L = e.line[k]; if (L.size() != len) L.assign(len, 0.f);
            size_t p = e.pos;
            for (int i = 0; i < n; i++) {
                float x = m->buf[a][i], d = L[p % len];
                e.lp[k] = d * (1 - lpa) + e.lp[k] * lpa;
                L[p % len] = x * in_g + e.lp[k] * fb;
                m->buf[b][i] = x * dry + d;
                p++;
            }
        }
        e.pos += n;
        return;
    }
    case E_REVERB: case E_I3DL2: {
        auto f = [&](const char* k, float d) { auto it = e.f.find(k); return it == e.f.end() ? d : it->second; };
        auto t = [&](const char* k, double d) { auto it = e.t.find(k); return it == e.t.end() ? d : it->second / 1e9; };   // TimeSpan ns -> s
        auto mb = [](float x) { return powf(10.0f, std::clamp(x, -10000.0f, 0.0f) / 2000.0f); };                        // millibels -> linear
        float dry, early, late, t60, ratio; double pre_s, late_s;
        if (e.kind == E_I3DL2) {
            float room = mb(f("room", -1000));
            dry = f("dry", 1); early = room * mb(f("reflections", -1000)); late = room * mb(f("reverb", -1000));
            t60 = (float)t("decay", 1.5); ratio = f("hf_decay", 0.8f); pre_s = t("refl_delay", 0.01); late_s = pre_s + t("late_delay", 0.02);
        } else {
            float out = f("out", 1);
            dry = f("dry", 1); early = f("early", 0.5f) * out; late = f("late", 0.5f) * f("reverb", 0.5f) * out;
            t60 = (float)t("decay", 1.5); ratio = f("hf_decay", 0.8f); pre_s = t("predelay", 0.01); late_s = pre_s + 0.01;
        }
        t60 = std::clamp(t60, 0.1f, 20.0f);
        const float damp = std::clamp(0.9f * (1.0f - ratio) + 0.1f, 0.0f, 0.9f);
        if (!e.verb) printf("[audio] %s on %d ch: dry %.2f early %.3f late %.3f, decay %.2f s, hf ratio %.2f, predelay %.0f ms, late delay %.0f ms\n",
                            e.kind == E_I3DL2 ? "I3DL2 reverb" : "reverb", e.count, dry, early, late, t60, ratio, pre_s * 1000, late_s * 1000);
        const size_t plen = (size_t)g_rate / 2 + 1;
        const size_t ed = std::min<size_t>((size_t)(pre_s * g_rate), plen / 3), ld = std::min<size_t>((size_t)(late_s * g_rate), plen - 1);
        if (!e.verb) {
            static const int cd[8] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617}, ad[4] = {556, 441, 341, 225};
            e.verb = std::make_shared<std::array<std::array<Effect::Line, 13>, kMaxCh>>();
            for (int k = 0; k < kMaxCh; k++) {
                int spread = (k & 1) ? 23 : 0;
                for (int j = 0; j < 8; j++) (*e.verb)[k][j].b.assign((size_t)(cd[j] + spread) * g_rate / 44100, 0.f);
                for (int j = 0; j < 4; j++) (*e.verb)[k][8 + j].b.assign((size_t)(ad[j] + spread) * g_rate / 44100, 0.f);
                (*e.verb)[k][12].b.assign(plen, 0.f);
            }
        }
        static const float tap_t[4] = {1.0f, 1.37f, 1.73f, 2.21f}, tap_g[4] = {0.8f, 0.6f, 0.45f, 0.3f};
        for (int k = 0; k < e.count; k++) {
            int a = e.in[k], b = e.out[k]; if (a < 0 || b < 0 || a >= m->nbuf || b >= m->nbuf) continue;
            auto& L = (*e.verb)[k];
            float g[8];
            for (int j = 0; j < 8; j++) g[j] = std::min(0.98f, powf(10.0f, -3.0f * (float)L[j].b.size() / (t60 * g_rate)));
            auto& P = L[12];
            for (int i = 0; i < n; i++) {
                float x = m->buf[a][i];
                P.b[P.i] = x;
                float er = 0;
                for (int q = 0; q < 4; q++) er += tap_g[q] * P.b[(P.i + plen - std::min(plen - 1, (size_t)(ed * tap_t[q]))) % plen];
                float in = P.b[(P.i + plen - ld) % plen] * 0.015f;
                P.i = (P.i + 1) % plen;
                float acc = 0;
                for (int j = 0; j < 8; j++) {
                    auto& c = L[j]; float y = c.b[c.i];
                    c.s = y * (1 - damp) + c.s * damp;
                    c.b[c.i] = in + c.s * g[j];
                    if (++c.i == c.b.size()) c.i = 0;
                    acc += y;
                }
                for (int j = 8; j < 12; j++) {
                    auto& c = L[j]; float bo = c.b[c.i];
                    float o = -acc + bo; c.b[c.i] = acc + bo * 0.5f;
                    if (++c.i == c.b.size()) c.i = 0;
                    acc = o;
                }
                m->buf[b][i] = dry * x + early * er + late * acc * 3.0f;
            }
        }
        return;
    }
    default: copy_through(); return;           // light limiter: dry signal (the device clamp catches overs)
    }
}

// One renderer frame: n samples per channel into the device sink's channels (out: interleaved, out_ch channels)
bool g_mix_off = getenv("TOTK_AUDIO_MIX") && !strcmp(getenv("TOTK_AUDIO_MIX"), "0");   // A/B: skip mixing (silence)
void render_frame(int n, float* out, int out_ch) {
    if (g_mix_off) { memset(out, 0, sizeof(float) * n * out_ch); return; }
    std::lock_guard<std::mutex> l(g_mtx);
    for (auto& [a, m] : g_mixes) for (int b = 0; b < m->nbuf && b < kMaxCh; b++) memset(m->buf[b], 0, n * sizeof(float));
    static float vbuf[kMaxVoiceCh][kFrameMax];
    for (auto& [a, v] : g_voices) {
        if (v.state != 0) continue;
        voice_render(v, n, vbuf);
        for (int ch = 0; ch < v.nch; ch++) g_pk_voice = std::max(g_pk_voice, peak(vbuf[ch], n));
        if (!v.dest) g_nodest++;
        send_to(v.dest, v.dest_splitter, vbuf, v.nch, &v.mat, n, 1.0f);
    }
    std::vector<Mix*> order;
    for (auto& [a, m] : g_mixes) { m->depth = mix_depth(m); order.push_back(m); }
    std::sort(order.begin(), order.end(), [](Mix* x, Mix* y) { return x->depth > y->depth; });
    Mix* fin = nullptr;
    for (Mix* m : order) {
        for (u64 e : m->effects) { auto it = g_effects.find(e); if (it != g_effects.end()) effect_run(it->second, m, n); }
        for (int b = 0; b < m->nbuf; b++) { float p = peak(m->buf[b], n); if (m->final) g_pk_final = std::max(g_pk_final, p); else g_pk_sub = std::max(g_pk_sub, p); }
        if (m->final) { if (!fin) fin = m; continue; }
        float g0 = m->prev_volume, g1 = m->volume;
        if (g0 != g1) for (int b = 0; b < m->nbuf; b++) for (int i = 0; i < n; i++) m->buf[b][i] *= g0 + (g1 - g0) * i / n;
        else if (g1 != 1) for (int b = 0; b < m->nbuf; b++) for (int i = 0; i < n; i++) m->buf[b][i] *= g1;
        m->prev_volume = m->volume;
        send_to(m->dest, m->dest_splitter, m->buf, m->nbuf, m->mat, n, 1.0f);
    }
    memset(out, 0, sizeof(float) * n * out_ch);
    const Sink* sink = nullptr;
    for (auto& [a, s] : g_sinks) if (s.used) { sink = &s; auto it = g_mixes.find(s.final); if (it != g_mixes.end()) fin = it->second; break; }
    if (!fin) return;
    float g = fin->volume * g_master;
    int cnt = sink ? sink->count : std::min(fin->nbuf, 2);
    auto ch = [&](int k) -> const float* { int b = sink ? sink->in[k] : k; return b >= 0 && b < fin->nbuf ? fin->buf[b] : nullptr; };
    if (cnt >= 6 && out_ch == 2) {                  // 5.1 (FL FR FC LFE RL RR) -> stereo
        const float *fl = ch(0), *fr = ch(1), *fc = ch(2), *rl = ch(4), *rr = ch(5);
        for (int i = 0; i < n; i++) {
            float L = (fl ? fl[i] : 0) + 0.707f * ((fc ? fc[i] : 0) + (rl ? rl[i] : 0));
            float R = (fr ? fr[i] : 0) + 0.707f * ((fc ? fc[i] : 0) + (rr ? rr[i] : 0));
            out[i * 2] = L * g; out[i * 2 + 1] = R * g;
        }
    } else if (cnt == 1) {
        const float* m = ch(0);
        for (int i = 0; i < n; i++) for (int c = 0; c < out_ch; c++) out[i * out_ch + c] = m ? m[i] * g : 0;
    } else {
        for (int c = 0; c < out_ch && c < cnt; c++) { const float* x = ch(c); if (x) for (int i = 0; i < n; i++) out[i * out_ch + c] = x[i] * g; }
    }
    for (int i = 0; i < n * out_ch; i++) out[i] = out[i] > 1 ? 1 : out[i] < -1 ? -1 : out[i];
}

// TOTK_AUDIO_WAV=<file>: raw float32 stereo 48 kHz capture of what goes to the device (tools/f32wav.py -> .wav)
FILE* g_wav = [] { const char* e = getenv("TOTK_AUDIO_WAV"); return e ? fopen(e, "wb") : nullptr; }();
void render_out(int n, float* out, int out_ch) {
    LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
    render_frame(n, out, out_ch);
    QueryPerformanceCounter(&b);
    double us = (b.QuadPart - a.QuadPart) * 1e6 / f.QuadPart; g_render_us += us; g_render_max = std::max(g_render_max, us); g_render_n++;
    g_pk_out = std::max(g_pk_out, peak(out, n * out_ch));
    if (g_wav) { fwrite(out, sizeof(float), (size_t)n * out_ch, g_wav); if (g_frames % 200 == 0) fflush(g_wav); }
}
void frame_done() {
    g_frames++;
    if (g_running.load()) if (u64 ev = g_event.load()) hle_sysevent_signal(ev);
    if (g_frames % 2000 == 0) {
        std::lock_guard<std::mutex> l(g_mtx);
        size_t busy = 0, playing = 0; for (auto& kv : g_voices) { busy += !kv.second.pending.empty(); playing += kv.second.state == 0; }
        printf("[audio] frame %llu: %zu voices (%zu playing, %zu with buffers), %zu mixes, %zu effects, %llu buffers appended, %llu released\n",
               (unsigned long long)g_frames.load(), g_voices.size(), playing, busy, g_mixes.size(), g_effects.size(),
               (unsigned long long)g_appends.load(), (unsigned long long)g_releases.load());
        printf("[audio]   peaks: decoded %.4f, voice %.4f, sub mixes %.4f, final mix %.4f, out %.4f | render %.1f us avg, %.0f max | starved %llu, no destination %llu\n",
               g_pk_raw, g_pk_voice, g_pk_sub, g_pk_final, g_pk_out, g_render_n ? g_render_us / g_render_n : 0.0, g_render_max, (unsigned long long)g_starved, (unsigned long long)g_nodest);
        g_pk_raw = g_pk_voice = g_pk_sub = g_pk_final = g_pk_out = 0; g_render_us = g_render_max = 0; g_render_n = 0; g_starved = g_nodest = 0;
    }
}

// ------------------------------------------------------------------ output thread (WASAPI, event driven, shared mode)
void null_ticker() {
    auto period = std::chrono::nanoseconds((s64)g_count * 1000000000ll / (g_rate ? g_rate : 48000));
    auto next = std::chrono::steady_clock::now();
    std::vector<float> scratch((size_t)kFrameMax * 2);
    for (;;) {
        next += period;
        std::this_thread::sleep_until(next);
        if (g_running.load()) render_out(g_count, scratch.data(), 2);
        frame_done();
    }
}
#ifndef _WIN32
// SDL3 audio stream (float stereo at the renderer's rate). The device pulls: its callback asks for `additional` bytes and gets whole
// renderer frames (g_count samples), so the game's audio thread, woken by frame_done(), runs on the device clock like the WASAPI
// event loop below. SDL_AUDIO_DEVICE_SAMPLE_FRAMES asks for a ~10 ms device buffer; TOTK_AUDIO_FRAMES overrides it.
void SDLCALL sdl_audio_cb(void*, SDL_AudioStream* s, int additional, int) {
    static std::vector<float> frame((size_t)kFrameMax * 2);
    const int bytes = g_count * 2 * (int)sizeof(float);
    do {
        if (g_running.load()) render_out(g_count, frame.data(), 2);
        else memset(frame.data(), 0, (size_t)bytes);
        SDL_PutAudioStreamData(s, frame.data(), bytes);
        additional -= bytes;
        frame_done();
    } while (additional > 0);
}
void output_thread() {
    if (getenv("TOTK_AUDIO") && !strcmp(getenv("TOTK_AUDIO"), "null")) { printf("[audio] TOTK_AUDIO=null: no device\n"); null_ticker(); return; }
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, getenv("TOTK_AUDIO_FRAMES") ? getenv("TOTK_AUDIO_FRAMES") : "512");
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) { printf("[audio] SDL audio init failed (%s): no sound, null pacing\n", SDL_GetError()); null_ticker(); return; }
    SDL_AudioSpec spec = {SDL_AUDIO_F32, 2, g_rate};
    SDL_AudioStream* st = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, sdl_audio_cb, nullptr);
    if (!st) { printf("[audio] SDL audio device failed (%s): no sound, null pacing\n", SDL_GetError()); null_ticker(); return; }
    SDL_AudioSpec dev{}; int dev_frames = 0;
    SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(st), &dev, &dev_frames);
    printf("[audio] SDL3 (%s): %d Hz float stereo -> device %d Hz %d ch, device buffer %d frames (%.1f ms), renderer frame %d samples\n",
           SDL_GetCurrentAudioDriver(), g_rate, dev.freq, dev.channels, dev_frames, dev_frames * 1000.0 / (dev.freq ? dev.freq : 1), g_count);
    SDL_ResumeAudioStreamDevice(st);
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));   // (the callback runs on SDL's audio thread)
}
#else
void output_thread() {
    if (getenv("TOTK_AUDIO") && !strcmp(getenv("TOTK_AUDIO"), "null")) { printf("[audio] TOTK_AUDIO=null: no device\n"); null_ticker(); return; }
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* en = nullptr; IMMDevice* dev = nullptr; IAudioClient* ac = nullptr; IAudioRenderClient* rc = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&en);
    if (SUCCEEDED(hr)) hr = en->GetDefaultAudioEndpoint(eRender, eConsole, &dev);
    if (SUCCEEDED(hr)) hr = dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&ac);
    WAVEFORMATEXTENSIBLE wf{};
    wf.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE; wf.Format.nChannels = 2; wf.Format.nSamplesPerSec = (DWORD)g_rate;
    wf.Format.wBitsPerSample = 32; wf.Format.nBlockAlign = 8; wf.Format.nAvgBytesPerSec = (DWORD)g_rate * 8; wf.Format.cbSize = 22;
    static const GUID kFloat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};   // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
    wf.Samples.wValidBitsPerSample = 32; wf.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT; wf.SubFormat = kFloat;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (SUCCEEDED(hr)) hr = ac->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                           400000 /* 40 ms */, 0, &wf.Format, nullptr);
    if (SUCCEEDED(hr)) hr = ac->SetEventHandle(ev);
    UINT32 bufn = 0;
    if (SUCCEEDED(hr)) hr = ac->GetBufferSize(&bufn);
    if (SUCCEEDED(hr)) hr = ac->GetService(__uuidof(IAudioRenderClient), (void**)&rc);
    if (SUCCEEDED(hr)) hr = ac->Start();
    if (FAILED(hr)) { printf("[audio] WASAPI init failed (hr %08lx): no sound, null pacing\n", (unsigned long)hr); null_ticker(); return; }
    printf("[audio] WASAPI: %d Hz float stereo, device buffer %u frames (%.1f ms), renderer frame %d samples\n", g_rate, bufn, bufn * 1000.0 / g_rate, g_count);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    std::vector<float> frame((size_t)kFrameMax * 2);
    // keep ~20 ms queued: enough for scheduling jitter, small enough for responsive sound
    const UINT32 target = std::min<UINT32>(bufn, (UINT32)(g_rate / 50));
    for (;;) {
        WaitForSingleObject(ev, 20);
        UINT32 pad = 0;
        if (FAILED(ac->GetCurrentPadding(&pad))) { Sleep(5); continue; }
        while (pad + (UINT32)g_count <= target) {
            BYTE* p = nullptr;
            if (FAILED(rc->GetBuffer((UINT32)g_count, &p))) break;
            if (g_running.load()) render_out(g_count, (float*)p, 2);
            else memset(p, 0, (size_t)g_count * 8);
            rc->ReleaseBuffer((UINT32)g_count, 0);
            pad += (UINT32)g_count;
            frame_done();
        }
    }
}
#endif
}  // namespace

// ------------------------------------------------------------------ renderer
// AudioRendererParameter: sampleRate +0, sampleCount +4, mixBufferCount +8, subMixCount +c, voiceCount +10, sinkCount +14,
// effectCount +18, performanceFrameCount +1c, isVoiceDropEnabled +20, splitterCount +24, splitterSendChannelCount +28,
// renderingDevice +2c, executionMode +30, _magic +34
HLE_AS(audio_InitializeAudioRendererParameter, "_ZN2nn5audio32InitializeAudioRendererParameterEPNS0_22AudioRendererParameterE") {
    u32* p = (u32*)c->x[0];
    memset(p, 0, 0x34);
    p[0] = 48000; p[1] = 240; p[2] = 1; p[4] = 24; p[5] = 1;
}
HLE_AS(audio_GetAudioRendererWorkBufferSize, "_ZN2nn5audio30GetAudioRendererWorkBufferSizeERKNS0_22AudioRendererParameterE") { c->x[0] = 0x10000; }
HLE_AS(audio_GetAudioRendererConfigWorkBufferSize, "_ZN2nn5audio36GetAudioRendererConfigWorkBufferSizeERKNS0_22AudioRendererParameterE") { c->x[0] = 0x10000; }
HLE_AS(audio_GetRequiredBufferSizeForPerformanceFrames, "_ZN2nn5audio41GetRequiredBufferSizeForPerformanceFramesERKNS0_22AudioRendererParameterE") { c->x[0] = 0x1000; }
HLE_AS(audio_GetRequiredBufferSizeForAuxSendReturnBuffer, "_ZN2nn5audio43GetRequiredBufferSizeForAuxSendReturnBufferEPKNS0_22AudioRendererParameterEii") {
    const u32* p = (const u32*)c->x[0];
    c->x[0] = ((u64)p[1] * (u32)c->x[1] * (u32)c->x[2] * 4 + 0xfff) & ~0xfffull;   // sampleCount * frames * channels * s32
}
HLE_AS(audio_GetRequiredBufferSizeForDelay, "_ZN2nn5audio29GetRequiredBufferSizeForDelayENS_8TimeSpanEii") {
    u64 n = (u64)((s64)c->x[0] / 1000000) * (u32)c->x[1] / 1000 * (u32)c->x[2] * 4;
    c->x[0] = (n + 0xfff) & ~0xfffull;
}
HLE_AS(audio_GetRequiredBufferSizeForReverb, "_ZN2nn5audio30GetRequiredBufferSizeForReverbEii") { c->x[0] = 0x20000; }
HLE_AS(audio_GetRequiredBufferSizeForI3dl2Reverb, "_ZN2nn5audio35GetRequiredBufferSizeForI3dl2ReverbEii") { c->x[0] = 0x20000; }
HLE_AS(audio_GetRequiredBufferSizeForLightLimiter, "_ZN2nn5audio36GetRequiredBufferSizeForLightLimiterENS_8TimeSpanEii") { c->x[0] = 0x1000; }
HLE_AS(audio_InitializeAudioRendererConfig, "_ZN2nn5audio29InitializeAudioRendererConfigEPNS0_19AudioRendererConfigERKNS0_22AudioRendererParameterEPvm") { c->x[0] = 0; }
HLE_AS(audio_OpenAudioRenderer, "_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm") {
    const u32* p = (const u32*)c->x[2];
    if (p[0]) g_rate = (int)p[0];
    if (p[1]) g_count = std::min((int)p[1], kFrameMax);
    *(u64*)c->x[0] = 1;
    hle_sysevent_init(c->x[1], true);
    g_event = c->x[1];
    static std::once_flag once;
    std::call_once(once, [] { std::thread(output_thread).detach(); });
    printf("[audio] OpenAudioRenderer: %d Hz, %d samples/frame, %u mix buffers, %u sub mixes, %u voices, %u effects, %u splitters, event %llx\n",
           g_rate, g_count, p[2], p[3], p[4], p[6], p[9], (unsigned long long)c->x[1]);
    c->x[0] = 0;
}
HLE_AS(audio_StartAudioRenderer, "_ZN2nn5audio18StartAudioRendererENS0_19AudioRendererHandleE") { g_running = true; c->x[0] = 0; }
HLE_AS(audio_StopAudioRenderer, "_ZN2nn5audio17StopAudioRendererENS0_19AudioRendererHandleE") { g_running = false; c->x[0] = 0; }
HLE_AS(audio_CloseAudioRenderer, "_ZN2nn5audio18CloseAudioRendererENS0_19AudioRendererHandleE") { g_running = false; g_event = 0; }
HLE_AS(audio_GetAudioRendererElapsedFrameCount, "_ZN2nn5audio33GetAudioRendererElapsedFrameCountEPKNS0_19AudioRendererConfigE") { c->x[0] = g_frames.load(); }

// Commit: appended wave buffers become playable and play-state changes take effect (the DSP only sees updated state)
HLE_AS(audio_RequestUpdateAudioRenderer, "_ZN2nn5audio26RequestUpdateAudioRendererENS0_19AudioRendererHandleEPKNS0_19AudioRendererConfigE") {
    std::lock_guard<std::mutex> l(g_mtx);
    for (auto& [a, v] : g_voices) {
        for (auto& w : v.pending) w.sent = true;
        if (v.pending_state != v.state) {
            // play-state log (user 2026-10-15: skipped cooking cutscene keeps its full audio -> which voices stop when)
            static int n_ps = 0;
            if (n_ps++ < 600)
                printf("[audio] frame %llu: voice %llx %s -> %s (%d Hz %d ch fmt %d, volume %.2f, %zu buffers queued, %llu samples played)\n",
                       (unsigned long long)g_frames.load(), (unsigned long long)a, v.state == 0 ? "play" : v.state == 1 ? "stop" : "pause",
                       v.pending_state == 0 ? "play" : v.pending_state == 1 ? "stop" : "pause", v.rate, v.nch, v.fmt, v.volume,
                       v.pending.size(), (unsigned long long)v.played);
            if (v.pending_state == 1) {        // stop: every queued buffer comes back, playback state resets
                for (auto& w : v.pending) { v.released.push_back(w); g_releases++; }
                v.pending.clear(); v.cur = -1; v.played = 0; v.frac = 0; v.hist1 = v.hist2 = 0; v.started_once = false;
                for (int ch = 0; ch < kMaxVoiceCh; ch++) { v.look[ch].clear(); v.prevs[ch] = 0; }
            }
            v.state = v.pending_state;
        }
    }
    c->x[0] = 0;
}

// ------------------------------------------------------------------ voices
HLE_AS(audio_AcquireVoiceSlot, "_ZN2nn5audio16AcquireVoiceSlotEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEiiNS0_12SampleFormatEiPKvmPKNS3_15BehaviorOptionsE") {
    std::lock_guard<std::mutex> l(g_mtx);
    Voice& v = g_voices[c->x[1]];
    v = Voice{};
    v.rate = (int)c->x[2] > 0 ? (int)c->x[2] : 48000;
    v.nch = std::clamp((int)c->x[3], 1, kMaxVoiceCh);
    v.fmt = (int)c->x[4]; v.prio = (int)c->x[5];
    handle_set(c->x[1]);
    if (v.fmt == 6 && c->x[6] && c->x[7] >= 32) memcpy(v.coef, (const void*)c->x[6], 32);
    static std::atomic<int> logged{0};
    if (logged++ < 8) printf("[audio] AcquireVoiceSlot %llx: %d Hz, %d ch, format %d, priority %d, param %llx+%llu\n", (unsigned long long)c->x[1], v.rate, v.nch, v.fmt, v.prio,
                             (unsigned long long)c->x[6], (unsigned long long)c->x[7]);
    c->x[0] = 1;
}
HLE_AS(audio_ReleaseVoiceSlot, "_ZN2nn5audio16ReleaseVoiceSlotEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeE") {
    std::lock_guard<std::mutex> l(g_mtx);
    static int n_rel = 0;
    auto it = g_voices.find(c->x[1]);
    if (it != g_voices.end() && it->second.state == 0 && n_rel++ < 300)
        printf("[audio] frame %llu: voice %llx released while playing (%zu buffers queued)\n", (unsigned long long)g_frames.load(),
               (unsigned long long)c->x[1], it->second.pending.size());
    g_voices.erase(c->x[1]); handle_clear(c->x[1]);
}
HLE_AS(audio_IsVoiceValid, "_ZN2nn5audio12IsVoiceValidEPKNS0_9VoiceTypeE") { std::lock_guard<std::mutex> l(g_mtx); c->x[0] = g_voices.count(c->x[0]) ? 1 : 0; }
HLE_AS(audio_IsVoiceDroppedFlagOn, "_ZN2nn5audio20IsVoiceDroppedFlagOnEPKNS0_9VoiceTypeE") { c->x[0] = 0; }
HLE_AS(audio_AppendWaveBuffer, "_ZN2nn5audio16AppendWaveBufferEPNS0_9VoiceTypeEPKNS0_10WaveBufferE") {
    std::lock_guard<std::mutex> l(g_mtx);
    auto it = g_voices.find(c->x[0]);
    if (it == g_voices.end() || it->second.pending.size() + it->second.released.size() >= 4) { c->x[0] = 0; return; }   // 4 per voice
    u64 wb = c->x[1];
    WaveBuf w{wb, *(const u64*)wb, *(const u64*)(wb + 8), (s32)rd32(wb + 0x10), (s32)rd32(wb + 0x14), *(const u8*)(wb + 0x18) != 0,
              *(const u8*)(wb + 0x19) != 0, *(const u64*)(wb + 0x20), *(const u64*)(wb + 0x28), false};
    it->second.pending.push_back(w); g_appends++;
    if (g_appends <= 12) ALOG("[audio] AppendWaveBuffer %llx: data %llx size %llu samples %d..%d loop %d eos %d ctx %llx+%llu\n", (unsigned long long)c->x[0], (unsigned long long)w.data,
                              (unsigned long long)w.size, w.start, w.end, w.loop, w.eos, (unsigned long long)w.ctx, (unsigned long long)w.ctx_size);
    c->x[0] = 1;
}
HLE_AS(audio_GetReleasedWaveBuffer, "_ZN2nn5audio21GetReleasedWaveBufferEPNS0_9VoiceTypeE") {
    std::lock_guard<std::mutex> l(g_mtx);
    auto it = g_voices.find(c->x[0]);
    if (it == g_voices.end() || it->second.released.empty()) { c->x[0] = 0; return; }
    c->x[0] = it->second.released.front().guest; it->second.released.pop_front();
}
HLE_AS(audio_GetWaveBufferCount, "_ZN2nn5audio18GetWaveBufferCountEPKNS0_9VoiceTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); c->x[0] = it == g_voices.end() ? 0 : it->second.pending.size();
}
HLE_AS(audio_GetVoicePlayedSampleCount, "_ZN2nn5audio25GetVoicePlayedSampleCountEPKNS0_9VoiceTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); c->x[0] = it == g_voices.end() ? 0 : it->second.played;
}
HLE_AS(audio_SetVoiceVolume, "_ZN2nn5audio14SetVoiceVolumeEPNS0_9VoiceTypeEf") { static int n_vv = 0; if (n_vv++ < 40) ALOG("[audio] SetVoiceVolume %llx %.3f\n", (unsigned long long)c->x[0], F(c, 0)); std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); if (it != g_voices.end()) it->second.volume = F(c, 0); }
HLE_AS(audio_SetVoicePitch, "_ZN2nn5audio13SetVoicePitchEPNS0_9VoiceTypeEf") { std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); if (it != g_voices.end()) it->second.pitch = F(c, 0); }
HLE_AS(audio_SetVoicePriority, "_ZN2nn5audio16SetVoicePriorityEPNS0_9VoiceTypeEi") { std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); if (it != g_voices.end()) it->second.prio = (int)c->x[1]; }
HLE_AS(audio_SetVoicePlayState, "_ZN2nn5audio17SetVoicePlayStateEPNS0_9VoiceTypeENS1_9PlayStateE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); if (it != g_voices.end()) it->second.pending_state = (int)c->x[1];
}
HLE_AS(audio_SetVoiceBiquadFilterParameter, "_ZN2nn5audio29SetVoiceBiquadFilterParameterEPNS0_9VoiceTypeEiRKNS0_21BiquadFilterParameterE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); int i = (int)c->x[1];
    if (it == g_voices.end() || i < 0 || i > 1) return;
    const u8* p = (const u8*)c->x[2]; auto& b = it->second.bq[i];
    bool on = p[0] != 0;
    if (on && !b.on) memset(b.s, 0, sizeof b.s);
    b.on = on; memcpy(b.num, p + 2, 6); memcpy(b.den, p + 8, 4);
}
static void voice_dest(Ctx* c, bool splitter) {
    ALOG("[audio] SetVoiceDestination %llx -> %llx%s\n", (unsigned long long)c->x[1], (unsigned long long)c->x[2], splitter ? " (splitter)" : "");
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[1]); if (it == g_voices.end()) return;
    if (it->second.dest != c->x[2]) it->second.mat = Matrix{};
    it->second.dest = c->x[2]; it->second.dest_splitter = splitter;
}
HLE_AS(audio_SetVoiceDestinationSub, "_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_10SubMixTypeE") { voice_dest(c, false); }
HLE_AS(audio_SetVoiceDestinationFinal, "_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_12FinalMixTypeE") { voice_dest(c, false); }
HLE_AS(audio_SetVoiceDestinationSplitter, "_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_12SplitterTypeE") { voice_dest(c, true); }
HLE_AS(audio_ClearVoiceDestination, "_ZN2nn5audio21ClearVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[1]); if (it != g_voices.end()) { it->second.dest = 0; it->second.mat = Matrix{}; }
}
static void voice_mixvol(Ctx* c) {      // (voice, mix, volume, src, dst)
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_voices.find(c->x[0]); int s = (int)c->x[2], d = (int)c->x[3];
    if (it != g_voices.end() && s >= 0 && s < kMaxCh && d >= 0 && d < kMaxCh) it->second.mat.v[s][d] = F(c, 0);
}
HLE_AS(audio_SetVoiceMixVolumeSub, "_ZN2nn5audio17SetVoiceMixVolumeEPNS0_9VoiceTypeEPNS0_10SubMixTypeEfii") { voice_mixvol(c); }
HLE_AS(audio_SetVoiceMixVolumeFinal, "_ZN2nn5audio17SetVoiceMixVolumeEPNS0_9VoiceTypeEPNS0_12FinalMixTypeEfii") { voice_mixvol(c); }

// ------------------------------------------------------------------ mixes and splitters
HLE_AS(audio_AcquireFinalMix, "_ZN2nn5audio15AcquireFinalMixEPNS0_19AudioRendererConfigEPNS0_12FinalMixTypeEi") {
    std::lock_guard<std::mutex> l(g_mtx); Mix* m = mix_get(c->x[1], true); m->nbuf = std::clamp((int)c->x[2], 1, kMaxCh); handle_set(c->x[1]);
    printf("[audio] AcquireFinalMix %llx: %d buffers\n", (unsigned long long)c->x[1], m->nbuf); c->x[0] = 1;
}
HLE_AS(audio_AcquireSubMix, "_ZN2nn5audio13AcquireSubMixEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEii") {
    ALOG("[audio] AcquireSubMix %llx: rate %d, %d buffers\n", (unsigned long long)c->x[1], (int)c->x[2], (int)c->x[3]);
    std::lock_guard<std::mutex> l(g_mtx); Mix* m = mix_get(c->x[1], false); m->nbuf = std::clamp((int)c->x[3], 1, kMaxCh); handle_set(c->x[1]); c->x[0] = 1;
}
static void mix_release(u64 a) { handle_clear(a); std::lock_guard<std::mutex> l(g_mtx); auto it = g_mixes.find(a); if (it != g_mixes.end()) { delete it->second->mat; delete it->second; g_mixes.erase(it); } }
HLE_AS(audio_ReleaseSubMix, "_ZN2nn5audio13ReleaseSubMixEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeE") { mix_release(c->x[1]); }
HLE_AS(audio_ReleaseFinalMix, "_ZN2nn5audio15ReleaseFinalMixEPNS0_19AudioRendererConfigEPNS0_12FinalMixTypeE") { mix_release(c->x[1]); }
HLE_AS(audio_SetFinalMixVolume, "_ZN2nn5audio17SetFinalMixVolumeEPNS0_12FinalMixTypeEf") { ALOG("[audio] SetFinalMixVolume %llx %.3f\n", (unsigned long long)c->x[0], F(c, 0)); std::lock_guard<std::mutex> l(g_mtx); mix_get(c->x[0], true)->volume = F(c, 0); }
HLE_AS(audio_SetSubMixVolume, "_ZN2nn5audio15SetSubMixVolumeEPNS0_10SubMixTypeEf") { ALOG("[audio] SetSubMixVolume %llx %.3f\n", (unsigned long long)c->x[0], F(c, 0)); std::lock_guard<std::mutex> l(g_mtx); mix_get(c->x[0], false)->volume = F(c, 0); }
static void sub_dest(Ctx* c, bool splitter) {
    ALOG("[audio] SetSubMixDestination %llx -> %llx%s\n", (unsigned long long)c->x[1], (unsigned long long)c->x[2], splitter ? " (splitter)" : "");
    std::lock_guard<std::mutex> l(g_mtx); Mix* m = mix_get(c->x[1], false);
    if (m->dest != c->x[2]) { delete m->mat; m->mat = splitter ? nullptr : new Matrix(); }
    m->dest = c->x[2]; m->dest_splitter = splitter;
}
HLE_AS(audio_SetSubMixDestinationSub, "_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeES4_") { sub_dest(c, false); }
HLE_AS(audio_SetSubMixDestinationFinal, "_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEPNS0_12FinalMixTypeE") { sub_dest(c, false); }
HLE_AS(audio_SetSubMixDestinationSplitter, "_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEPNS0_12SplitterTypeE") { sub_dest(c, true); }
HLE_AS(audio_ClearSubMixDestination, "_ZN2nn5audio22ClearSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); Mix* m = mix_get(c->x[1], false); m->dest = 0; delete m->mat; m->mat = nullptr;
}
static void sub_mixvol(Ctx* c) {        // (src mix, dst mix, volume, src, dst)
    ALOG("[audio] SetSubMixMixVolume %llx -> %llx %.3f %d->%d\n", (unsigned long long)c->x[0], (unsigned long long)c->x[1], F(c, 0), (int)c->x[2], (int)c->x[3]);
    std::lock_guard<std::mutex> l(g_mtx); Mix* m = mix_get(c->x[0], false); int s = (int)c->x[2], d = (int)c->x[3];
    if (!m->mat) m->mat = new Matrix();
    if (s >= 0 && s < kMaxCh && d >= 0 && d < kMaxCh) m->mat->v[s][d] = F(c, 0);
}
HLE_AS(audio_SetSubMixMixVolumeSub, "_ZN2nn5audio18SetSubMixMixVolumeEPNS0_10SubMixTypeES2_fii") { sub_mixvol(c); }
HLE_AS(audio_SetSubMixMixVolumeFinal, "_ZN2nn5audio18SetSubMixMixVolumeEPNS0_10SubMixTypeEPNS0_12FinalMixTypeEfii") { sub_mixvol(c); }
HLE_AS(audio_AcquireSplitter, "_ZN2nn5audio15AcquireSplitterEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiii") {
    std::lock_guard<std::mutex> l(g_mtx); g_splitters[c->x[1]] = Splitter{}; handle_set(c->x[1]);
    static std::atomic<int> logged{0};
    if (logged++ < 4) printf("[audio] AcquireSplitter %llx (%d, %d, %d)\n", (unsigned long long)c->x[1], (int)c->x[2], (int)c->x[3], (int)c->x[4]);
    c->x[0] = 1;
}
HLE_AS(audio_ReleaseSplitter, "_ZN2nn5audio15ReleaseSplitterEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeE") { std::lock_guard<std::mutex> l(g_mtx); g_splitters.erase(c->x[1]); handle_clear(c->x[1]); }
static void split_dest(Ctx* c) {        // (config, splitter, index, mix)
    ALOG("[audio] SetSplitterDestination %llx [%d] -> %llx\n", (unsigned long long)c->x[1], (int)c->x[2], (unsigned long long)c->x[3]);
    std::lock_guard<std::mutex> l(g_mtx); SplitDest& d = g_splitters[c->x[1]].dests[(int)c->x[2]];
    if (d.mix != c->x[3]) d.m = Matrix{};
    d.mix = c->x[3];
}
HLE_AS(audio_SetSplitterDestinationSub, "_ZN2nn5audio22SetSplitterDestinationEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiPNS0_10SubMixTypeE") { split_dest(c); }
HLE_AS(audio_SetSplitterDestinationFinal, "_ZN2nn5audio22SetSplitterDestinationEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiPNS0_12FinalMixTypeE") { split_dest(c); }
static void split_vol(Ctx* c) {         // (splitter, index, mix, volume, src, dst)
    ALOG("[audio] SetSplitterMixVolume %llx [%d] mix %llx %.3f %d->%d\n", (unsigned long long)c->x[0], (int)c->x[1], (unsigned long long)c->x[2], F(c, 0), (int)c->x[3], (int)c->x[4]);
    std::lock_guard<std::mutex> l(g_mtx); SplitDest& d = g_splitters[c->x[0]].dests[(int)c->x[1]]; int s = (int)c->x[3], t = (int)c->x[4];
    if (!d.mix) d.mix = c->x[2];
    if (s >= 0 && s < kMaxCh && t >= 0 && t < kMaxCh) d.m.v[s][t] = F(c, 0);
}
HLE_AS(audio_SetSplitterMixVolumeSub, "_ZN2nn5audio20SetSplitterMixVolumeEPNS0_12SplitterTypeEiPNS0_10SubMixTypeEfii") { split_vol(c); }
HLE_AS(audio_SetSplitterMixVolumeFinal, "_ZN2nn5audio20SetSplitterMixVolumeEPNS0_12SplitterTypeEiPNS0_12FinalMixTypeEfii") { split_vol(c); }

// ------------------------------------------------------------------ sinks, pools, device
HLE_AS(audio_AddDeviceSink, "_ZN2nn5audio13AddDeviceSinkEPNS0_19AudioRendererConfigEPNS0_14DeviceSinkTypeEPNS0_12FinalMixTypeEPKaiPKc") {
    std::lock_guard<std::mutex> l(g_mtx); Sink& s = g_sinks[c->x[1]];
    s.final = c->x[2]; s.count = std::clamp((int)c->x[4], 0, kMaxCh); s.used = true; handle_set(c->x[1]);
    memcpy(s.in, (const void*)c->x[3], s.count);
    printf("[audio] AddDeviceSink %s: %d inputs (%d %d ...)\n", c->x[5] ? (const char*)c->x[5] : "?", s.count, s.count > 0 ? s.in[0] : -1, s.count > 1 ? s.in[1] : -1);
    c->x[0] = 0;
}
HLE_AS(audio_RemoveDeviceSink, "_ZN2nn5audio16RemoveDeviceSinkEPNS0_19AudioRendererConfigEPNS0_14DeviceSinkTypeEPNS0_12FinalMixTypeE") { std::lock_guard<std::mutex> l(g_mtx); g_sinks.erase(c->x[1]); handle_clear(c->x[1]); }
std::unordered_map<u64, std::pair<u64, u64>> g_pools;
std::unordered_map<u64, bool> g_pool_attached;
HLE_AS(audio_AcquireMemoryPool, "_ZN2nn5audio17AcquireMemoryPoolEPNS0_19AudioRendererConfigEPNS0_14MemoryPoolTypeEPvm") {
    std::lock_guard<std::mutex> l(g_mtx); g_pools[c->x[1]] = {c->x[2], c->x[3]}; g_pool_attached[c->x[1]] = false; handle_set(c->x[1]); c->x[0] = 1;
}
HLE_AS(audio_ReleaseMemoryPool, "_ZN2nn5audio17ReleaseMemoryPoolEPNS0_19AudioRendererConfigEPNS0_14MemoryPoolTypeE") { std::lock_guard<std::mutex> l(g_mtx); g_pools.erase(c->x[1]); g_pool_attached.erase(c->x[1]); handle_clear(c->x[1]); }
HLE_AS(audio_RequestAttachMemoryPool, "_ZN2nn5audio23RequestAttachMemoryPoolEPNS0_14MemoryPoolTypeE") { std::lock_guard<std::mutex> l(g_mtx); g_pool_attached[c->x[0]] = true; c->x[0] = 1; }
HLE_AS(audio_RequestDetachMemoryPool, "_ZN2nn5audio23RequestDetachMemoryPoolEPNS0_14MemoryPoolTypeE") { std::lock_guard<std::mutex> l(g_mtx); g_pool_attached[c->x[0]] = false; c->x[0] = 1; }
HLE_AS(audio_IsMemoryPoolAttached, "_ZN2nn5audio20IsMemoryPoolAttachedEPKNS0_14MemoryPoolTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_pool_attached.find(c->x[0]); c->x[0] = it != g_pool_attached.end() && it->second;
}
HLE_AS(audio_GetMemoryPoolAddress, "_ZN2nn5audio20GetMemoryPoolAddressEPKNS0_14MemoryPoolTypeE") { std::lock_guard<std::mutex> l(g_mtx); auto it = g_pools.find(c->x[0]); c->x[0] = it == g_pools.end() ? 0 : it->second.first; }
HLE_AS(audio_GetMemoryPoolSize, "_ZN2nn5audio17GetMemoryPoolSizeEPKNS0_14MemoryPoolTypeE") { std::lock_guard<std::mutex> l(g_mtx); auto it = g_pools.find(c->x[0]); c->x[0] = it == g_pools.end() ? 0 : it->second.second; }
HLE_AS(audio_GetSampleByteSize, "_ZN2nn5audio17GetSampleByteSizeENS0_12SampleFormatE") {
    static const u8 sz[] = {0, 1, 2, 3, 4, 4, 0};
    u32 f = (u32)c->x[0]; c->x[0] = f < sizeof sz ? sz[f] : 0;
}
HLE_AS(audio_ListAudioOutputDeviceName, "_ZN2nn5audio25ListAudioOutputDeviceNameEPNS0_15AudioDeviceNameEi") {
    if ((int)c->x[1] < 1) { c->x[0] = 0; return; }
    memset((void*)c->x[0], 0, 256); strcpy((char*)c->x[0], "MainAudioOut"); c->x[0] = 1;
}
HLE_AS(audio_GetActiveAudioOutputDeviceName, "_ZN2nn5audio30GetActiveAudioOutputDeviceNameEPNS0_15AudioDeviceNameE") { memset((void*)c->x[0], 0, 256); strcpy((char*)c->x[0], "MainAudioOut"); c->x[0] = 0; }
HLE_AS(audio_GetActiveAudioDeviceChannelCountForOutput, "_ZN2nn5audio41GetActiveAudioDeviceChannelCountForOutputEv") { c->x[0] = 2; }
HLE_AS(audio_AcquireAudioDeviceNotificationForOutput, "_ZN2nn5audio39AcquireAudioDeviceNotificationForOutputEPNS_2os11SystemEventE") { hle_sysevent_init(c->x[0], true); c->x[0] = 0; }
HLE_AS(audio_SetAudioDeviceOutputVolume, "_ZN2nn5audio26SetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameEf") { c->x[0] = 0; }

// ------------------------------------------------------------------ effects
static Effect& eff(u64 a) { return g_effects[a]; }
static void eff_add(Ctx* c, EffKind k, u64 mix, u64 buf, u64 size) {
    std::lock_guard<std::mutex> l(g_mtx);
    Effect& e = g_effects[c->x[1]]; e = Effect{}; e.kind = k; e.mix = mix; e.buffer = buf; e.size = size; handle_set(c->x[1]);
    Mix* m = g_mixes.count(mix) ? g_mixes[mix] : nullptr;
    if (m) { std::erase(m->effects, c->x[1]); m->effects.push_back(c->x[1]); }
    static std::atomic<int> logged{0};
    if (logged++ < 24) printf("[audio] add effect kind %d %llx on mix %llx (buffer %llx+%llu)\n", (int)k, (unsigned long long)c->x[1], (unsigned long long)mix, (unsigned long long)buf, (unsigned long long)size);
    c->x[0] = 0;
}
static void eff_remove(Ctx* c) {
    std::lock_guard<std::mutex> l(g_mtx);
    auto it = g_effects.find(c->x[1]); u64 buf = 0; handle_clear(c->x[1]);
    if (it != g_effects.end()) { buf = it->second.buffer; auto mi = g_mixes.find(it->second.mix); if (mi != g_mixes.end()) std::erase(mi->second->effects, c->x[1]); g_effects.erase(it); }
    c->x[0] = buf;
}
static void eff_io_set(Ctx* c) {        // (effect, const int8* in, const int8* out, int count)
    std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); e.count = std::clamp((int)c->x[3], 0, kMaxCh);
    memcpy(e.in, (const void*)c->x[1], e.count); memcpy(e.out, (const void*)c->x[2], e.count);
    if (e.kind == E_AUX && e.send.empty()) { size_t cap = std::max<size_t>((size_t)g_count * e.count * 8, 4096); e.send.assign(cap, 0); e.ret.assign(cap, 0); }
}
static void eff_io_get(Ctx* c) {        // (effect, int8* in, int8* out, int* count, int max)
    std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); int n = std::min(e.count, (int)c->x[4]);
    if (c->x[1]) memcpy((void*)c->x[1], e.in, n);
    if (c->x[2]) memcpy((void*)c->x[2], e.out, n);
    if (c->x[3]) *(s32*)c->x[3] = n;
}
#define EFF_ADD(id, sym, kind, bufarg) HLE_AS(id, sym) { eff_add(c, kind, bufarg ? c->x[4] : c->x[2], bufarg ? c->x[2] : 0, bufarg ? c->x[3] : 0); }
#define EFF_ENABLE(id, sym) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); eff(c->x[0]).enabled = (c->x[1] & 1) != 0; }
#define EFF_ISEN(id, sym) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); c->x[0] = eff(c->x[0]).enabled; }
#define EFF_TRUE(id, sym) HLE_AS(id, sym) { c->x[0] = 1; }
#define EFF_SETF(id, sym, key) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); eff(c->x[0]).f[key] = F(c, 0); }
#define EFF_GETF(id, sym, key) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); retf(c, e.f.count(key) ? e.f[key] : 0.f); }
#define EFF_SETT(id, sym, key) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); eff(c->x[0]).t[key] = (s64)c->x[1]; }
#define EFF_GETT(id, sym, key) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); c->x[0] = (u64)(e.t.count(key) ? e.t[key] : 0); }
#define EFF_SETI(id, sym, key) HLE_AS(id, sym) { std::lock_guard<std::mutex> l(g_mtx); eff(c->x[0]).i[key] = (int)c->x[1]; }
#define EFF_REMOVE(id, sym) HLE_AS(id, sym) { eff_remove(c); }
#define EFF_IO(id, sym) HLE_AS(id, sym) { eff_io_set(c); }
#define EFF_IOGET(id, sym) HLE_AS(id, sym) { eff_io_get(c); }

// aux
EFF_ADD(audio_AddAuxSub, "_ZN2nn5audio6AddAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_10SubMixTypeEPvS7_m", E_AUX, false)
EFF_ADD(audio_AddAuxFinal, "_ZN2nn5audio6AddAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_12FinalMixTypeEPvS7_m", E_AUX, false)
EFF_REMOVE(audio_RemoveAuxSub, "_ZN2nn5audio9RemoveAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveAuxFinal, "_ZN2nn5audio9RemoveAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetAuxEnabled, "_ZN2nn5audio13SetAuxEnabledEPNS0_7AuxTypeEb")
EFF_ISEN(audio_IsAuxEnabled, "_ZN2nn5audio12IsAuxEnabledEPKNS0_7AuxTypeE")
EFF_TRUE(audio_IsAuxRemovable, "_ZN2nn5audio14IsAuxRemovableEPKNS0_7AuxTypeE")
EFF_IO(audio_SetAuxInputOutput, "_ZN2nn5audio17SetAuxInputOutputEPNS0_7AuxTypeEPKaS4_i")
HLE_AS(audio_GetAuxSampleCount, "_ZN2nn5audio17GetAuxSampleCountEPKNS0_7AuxTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); c->x[0] = (u64)(e.send_w - e.send_r);
}
// Count convention (main 0x29de0e8, the game's aux pump; measured run220): aux+0x20 = channels (6), aux+0x24 = frames
// (5); blocks = GetAuxSampleCount / (6 * 5), Read/Write count = 6 * blocks (288 of 1440 available) -> counts are ALL
// channels and always a multiple of the channel count = interleaved samples. Treating the count as per channel
// (TOTK_AUX_COUNT=perch) wrote 6x past the game's buffer and crashed (run220).
static bool g_aux_total = !(getenv("TOTK_AUX_COUNT") && !strcmp(getenv("TOTK_AUX_COUNT"), "perch"));
static double rms_s32(const s32* p, size_t n) { double a = 0; for (size_t i = 0; i < n; i++) a += (double)p[i] * p[i]; return n ? sqrt(a / n) : 0; }
HLE_AS(audio_ReadAuxSendBuffer, "_ZN2nn5audio17ReadAuxSendBufferEPNS0_7AuxTypeEPii") {
    std::lock_guard<std::mutex> l(g_mtx); u64 aux = c->x[0]; Effect& e = eff(aux); s32* out = (s32*)c->x[1];
    int count = std::max(0, (int)c->x[2]); size_t mul = g_aux_total ? 1 : (size_t)std::max(1, e.count);
    size_t want = (size_t)count * mul, avail = (size_t)(e.send_w - e.send_r), cap = e.send.size();
    size_t n = cap ? std::min(want, avail) / mul * mul : 0;
    for (size_t i = 0; i < n; i++) out[i] = e.send[(e.send_r + i) % cap];
    e.send_r += n; c->x[0] = n / mul;
    static std::atomic<int> logged{0};
    int k = logged++;
    if ((k < 8 && count) || k % 4000 == 0)
        printf("[audio] ReadAuxSendBuffer %llx: count %d (aux+0x20 %d, +0x24 %d, %d ch) avail %zu -> %zu samples, rms %.0f\n", (unsigned long long)aux, count,
               *(const s32*)(aux + 0x20), *(const s32*)(aux + 0x24), e.count, avail, n, rms_s32(out, n));
}
HLE_AS(audio_WriteAuxReturnBuffer, "_ZN2nn5audio20WriteAuxReturnBufferEPNS0_7AuxTypeEPKii") {
    std::lock_guard<std::mutex> l(g_mtx); u64 aux = c->x[0]; Effect& e = eff(aux); const s32* in = (const s32*)c->x[1];
    int count = std::max(0, (int)c->x[2]); size_t mul = g_aux_total ? 1 : (size_t)std::max(1, e.count);
    size_t n = (size_t)count * mul, cap = e.ret.size();
    if (!cap) { c->x[0] = 0; return; }
    for (size_t i = 0; i < n; i++) e.ret[(e.ret_w + i) % cap] = in[i];
    e.ret_w += n; if (e.ret_w - e.ret_r > cap) e.ret_r = e.ret_w - cap;
    e.returned = true; c->x[0] = (u64)count;
    static std::atomic<int> logged{0};
    int k = logged++;
    if (k < 8 || k % 4000 == 0) printf("[audio] WriteAuxReturnBuffer %llx: count %d -> %zu samples, rms %.0f\n", (unsigned long long)aux, count, n, rms_s32(in, n));
}
// buffer mixer
EFF_ADD(audio_AddBufferMixerSub, "_ZN2nn5audio14AddBufferMixerEPNS0_19AudioRendererConfigEPNS0_15BufferMixerTypeEPNS0_10SubMixTypeE", E_BUFMIX, false)
EFF_ADD(audio_AddBufferMixerFinal, "_ZN2nn5audio14AddBufferMixerEPNS0_19AudioRendererConfigEPNS0_15BufferMixerTypeEPNS0_12FinalMixTypeE", E_BUFMIX, false)
EFF_REMOVE(audio_RemoveBufferMixerSub, "_ZN2nn5audio17RemoveBufferMixerEPNS0_19AudioRendererConfigEPNS0_15BufferMixerTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveBufferMixerFinal, "_ZN2nn5audio17RemoveBufferMixerEPNS0_19AudioRendererConfigEPNS0_15BufferMixerTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetBufferMixerEnabled, "_ZN2nn5audio21SetBufferMixerEnabledEPNS0_15BufferMixerTypeEb")
EFF_ISEN(audio_IsBufferMixerEnabled, "_ZN2nn5audio20IsBufferMixerEnabledEPKNS0_15BufferMixerTypeE")
EFF_TRUE(audio_IsBufferMixerRemovable, "_ZN2nn5audio22IsBufferMixerRemovableEPKNS0_15BufferMixerTypeE")
EFF_IO(audio_SetBufferMixerInputOutput, "_ZN2nn5audio25SetBufferMixerInputOutputEPNS0_15BufferMixerTypeEPKaS4_i")
HLE_AS(audio_SetBufferMixerVolume, "_ZN2nn5audio20SetBufferMixerVolumeEPNS0_15BufferMixerTypeEif") {
    std::lock_guard<std::mutex> l(g_mtx); int i = (int)c->x[1]; if (i >= 0 && i < kMaxCh) eff(c->x[0]).bm_vol[i] = F(c, 0);
}
// biquad filter effect
EFF_ADD(audio_AddBiquadFilterSub, "_ZN2nn5audio15AddBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_10SubMixTypeE", E_BIQUAD, false)
EFF_ADD(audio_AddBiquadFilterFinal, "_ZN2nn5audio15AddBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_12FinalMixTypeE", E_BIQUAD, false)
EFF_REMOVE(audio_RemoveBiquadFilterSub, "_ZN2nn5audio18RemoveBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveBiquadFilterFinal, "_ZN2nn5audio18RemoveBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetBiquadFilterEnabled, "_ZN2nn5audio22SetBiquadFilterEnabledEPNS0_16BiquadFilterTypeEb")
EFF_ISEN(audio_IsBiquadFilterEnabled, "_ZN2nn5audio21IsBiquadFilterEnabledEPKNS0_16BiquadFilterTypeE")
EFF_TRUE(audio_IsBiquadFilterRemovable, "_ZN2nn5audio23IsBiquadFilterRemovableEPKNS0_16BiquadFilterTypeE")
EFF_IO(audio_SetBiquadFilterInputOutput, "_ZN2nn5audio26SetBiquadFilterInputOutputEPNS0_16BiquadFilterTypeEPKaS4_i")
HLE_AS(audio_SetBiquadFilterParameter, "_ZN2nn5audio24SetBiquadFilterParameterEPNS0_16BiquadFilterTypeERKNS0_21BiquadFilterParameterE") {
    std::lock_guard<std::mutex> l(g_mtx); Effect& e = eff(c->x[0]); const u8* p = (const u8*)c->x[1];
    memcpy(e.num, p + 2, 6); memcpy(e.den, p + 8, 4);
}
// delay
HLE_AS(audio_AddDelaySub, "_ZN2nn5audio8AddDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPvmPNS0_10SubMixTypeENS_8TimeSpanEi") { eff_add(c, E_DELAY, c->x[4], c->x[2], c->x[3]); }
HLE_AS(audio_AddDelayFinal, "_ZN2nn5audio8AddDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPvmPNS0_12FinalMixTypeENS_8TimeSpanEi") { eff_add(c, E_DELAY, c->x[4], c->x[2], c->x[3]); }
EFF_REMOVE(audio_RemoveDelaySub, "_ZN2nn5audio11RemoveDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveDelayFinal, "_ZN2nn5audio11RemoveDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetDelayEnabled, "_ZN2nn5audio15SetDelayEnabledEPNS0_9DelayTypeEb")
EFF_ISEN(audio_IsDelayEnabled, "_ZN2nn5audio14IsDelayEnabledEPKNS0_9DelayTypeE")
EFF_TRUE(audio_IsDelayRemovable, "_ZN2nn5audio16IsDelayRemovableEPKNS0_9DelayTypeE")
EFF_IO(audio_SetDelayInputOutput, "_ZN2nn5audio19SetDelayInputOutputEPNS0_9DelayTypeEPKaS4_i")
EFF_SETT(audio_SetDelayTime, "_ZN2nn5audio12SetDelayTimeEPNS0_9DelayTypeENS_8TimeSpanE", "time")
EFF_GETT(audio_GetDelayTime, "_ZN2nn5audio12GetDelayTimeEPKNS0_9DelayTypeE", "time")
EFF_SETF(audio_SetDelayDryGain, "_ZN2nn5audio15SetDelayDryGainEPNS0_9DelayTypeEf", "dry")
EFF_SETF(audio_SetDelayFeedbackGain, "_ZN2nn5audio20SetDelayFeedbackGainEPNS0_9DelayTypeEf", "feedback")
EFF_SETF(audio_SetDelayInGain, "_ZN2nn5audio14SetDelayInGainEPNS0_9DelayTypeEf", "in")
EFF_SETF(audio_SetDelayChannelSpread, "_ZN2nn5audio21SetDelayChannelSpreadEPNS0_9DelayTypeEf", "spread")
EFF_SETF(audio_SetDelayLowPassAmount, "_ZN2nn5audio21SetDelayLowPassAmountEPNS0_9DelayTypeEf", "lowpass")
// reverb (dry only for now)
HLE_AS(audio_AddReverbSub, "_ZN2nn5audio9AddReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPvmPNS0_10SubMixTypeEi") { eff_add(c, E_REVERB, c->x[4], c->x[2], c->x[3]); }
HLE_AS(audio_AddReverbFinal, "_ZN2nn5audio9AddReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPvmPNS0_12FinalMixTypeEi") { eff_add(c, E_REVERB, c->x[4], c->x[2], c->x[3]); }
EFF_REMOVE(audio_RemoveReverbSub, "_ZN2nn5audio12RemoveReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveReverbFinal, "_ZN2nn5audio12RemoveReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetReverbEnabled, "_ZN2nn5audio16SetReverbEnabledEPNS0_10ReverbTypeEb")
EFF_ISEN(audio_IsReverbEnabled, "_ZN2nn5audio15IsReverbEnabledEPKNS0_10ReverbTypeE")
EFF_TRUE(audio_IsReverbRemovable, "_ZN2nn5audio17IsReverbRemovableEPKNS0_10ReverbTypeE")
EFF_IO(audio_SetReverbInputOutput, "_ZN2nn5audio20SetReverbInputOutputEPNS0_10ReverbTypeEPKaS4_i")
EFF_IOGET(audio_GetReverbInputOutput, "_ZN2nn5audio20GetReverbInputOutputEPKNS0_10ReverbTypeEPaS4_Pii")
EFF_SETI(audio_SetReverbEarlyMode, "_ZN2nn5audio18SetReverbEarlyModeEPNS0_10ReverbTypeENS1_9EarlyModeE", "early_mode")
EFF_SETI(audio_SetReverbLateMode, "_ZN2nn5audio17SetReverbLateModeEPNS0_10ReverbTypeENS1_8LateModeE", "late_mode")
EFF_SETF(audio_SetReverbEarlyGain, "_ZN2nn5audio18SetReverbEarlyGainEPNS0_10ReverbTypeEf", "early")
EFF_SETF(audio_SetReverbLateGain, "_ZN2nn5audio17SetReverbLateGainEPNS0_10ReverbTypeEf", "late")
EFF_SETT(audio_SetReverbPredelayTime, "_ZN2nn5audio21SetReverbPredelayTimeEPNS0_10ReverbTypeENS_8TimeSpanE", "predelay")
EFF_SETT(audio_SetReverbDecayTime, "_ZN2nn5audio18SetReverbDecayTimeEPNS0_10ReverbTypeENS_8TimeSpanE", "decay")
EFF_SETF(audio_SetReverbHfDecay, "_ZN2nn5audio32SetReverbHighFrequencyDecayRatioEPNS0_10ReverbTypeEf", "hf_decay")
EFF_SETF(audio_SetReverbColoration, "_ZN2nn5audio19SetReverbColorationEPNS0_10ReverbTypeEf", "coloration")
EFF_SETF(audio_SetReverbReverbGain, "_ZN2nn5audio19SetReverbReverbGainEPNS0_10ReverbTypeEf", "reverb")
EFF_SETF(audio_SetReverbOutGain, "_ZN2nn5audio16SetReverbOutGainEPNS0_10ReverbTypeEf", "out")
EFF_SETF(audio_SetReverbDryGain, "_ZN2nn5audio16SetReverbDryGainEPNS0_10ReverbTypeEf", "dry")
// I3DL2 reverb (dry only for now)
HLE_AS(audio_AddI3dl2ReverbSub, "_ZN2nn5audio14AddI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPvmPNS0_10SubMixTypeEi") { eff_add(c, E_I3DL2, c->x[4], c->x[2], c->x[3]); }
HLE_AS(audio_AddI3dl2ReverbFinal, "_ZN2nn5audio14AddI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPvmPNS0_12FinalMixTypeEi") { eff_add(c, E_I3DL2, c->x[4], c->x[2], c->x[3]); }
EFF_REMOVE(audio_RemoveI3dl2ReverbSub, "_ZN2nn5audio17RemoveI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveI3dl2ReverbFinal, "_ZN2nn5audio17RemoveI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetI3dl2ReverbEnabled, "_ZN2nn5audio21SetI3dl2ReverbEnabledEPNS0_15I3dl2ReverbTypeEb")
EFF_ISEN(audio_IsI3dl2ReverbEnabled, "_ZN2nn5audio20IsI3dl2ReverbEnabledEPKNS0_15I3dl2ReverbTypeE")
EFF_TRUE(audio_IsI3dl2ReverbRemovable, "_ZN2nn5audio22IsI3dl2ReverbRemovableEPKNS0_15I3dl2ReverbTypeE")
EFF_IO(audio_SetI3dl2ReverbInputOutput, "_ZN2nn5audio25SetI3dl2ReverbInputOutputEPNS0_15I3dl2ReverbTypeEPKaS4_i")
EFF_IOGET(audio_GetI3dl2ReverbInputOutput, "_ZN2nn5audio25GetI3dl2ReverbInputOutputEPKNS0_15I3dl2ReverbTypeEPaS4_Pii")
#define I3F(name, mangled_len, key) \
    EFF_SETF(audio_SetI3dl2##name, "_ZN2nn5audio" #mangled_len "SetI3dl2Reverb" #name "EPNS0_15I3dl2ReverbTypeEf", key) \
    EFF_GETF(audio_GetI3dl2##name, "_ZN2nn5audio" #mangled_len "GetI3dl2Reverb" #name "EPKNS0_15I3dl2ReverbTypeE", key)
#define I3T(name, mangled_len, key) \
    EFF_SETT(audio_SetI3dl2##name, "_ZN2nn5audio" #mangled_len "SetI3dl2Reverb" #name "EPNS0_15I3dl2ReverbTypeENS_8TimeSpanE", key) \
    EFF_GETT(audio_GetI3dl2##name, "_ZN2nn5audio" #mangled_len "GetI3dl2Reverb" #name "EPKNS0_15I3dl2ReverbTypeE", key)
I3F(RoomGain, 22, "room") I3F(RoomHfGain, 24, "room_hf") I3F(LateReverbHfDecayRatio, 36, "hf_decay") I3F(ReflectionsGain, 29, "reflections")
I3F(ReverbGain, 24, "reverb") I3F(LateReverbDiffusion, 33, "diffusion") I3F(LateReverbDensity, 31, "density") I3F(HfReference, 25, "hf_ref") I3F(DryGain, 21, "dry")
I3T(LateReverbDecayTime, 33, "decay") I3T(ReflectionDelayTime, 33, "refl_delay") I3T(LateReverbDelayTime, 33, "late_delay")
// light limiter (dry only for now)
HLE_AS(audio_AddLightLimiterSub, "_ZN2nn5audio15AddLightLimiterEPNS0_19AudioRendererConfigEPNS0_16LightLimiterTypeEPvmPNS0_10SubMixTypeENS_8TimeSpanEi") { eff_add(c, E_LIMITER, c->x[4], c->x[2], c->x[3]); }
HLE_AS(audio_AddLightLimiterFinal, "_ZN2nn5audio15AddLightLimiterEPNS0_19AudioRendererConfigEPNS0_16LightLimiterTypeEPvmPNS0_12FinalMixTypeENS_8TimeSpanEi") { eff_add(c, E_LIMITER, c->x[4], c->x[2], c->x[3]); }
EFF_REMOVE(audio_RemoveLightLimiterSub, "_ZN2nn5audio18RemoveLightLimiterEPNS0_19AudioRendererConfigEPNS0_16LightLimiterTypeEPNS0_10SubMixTypeE")
EFF_REMOVE(audio_RemoveLightLimiterFinal, "_ZN2nn5audio18RemoveLightLimiterEPNS0_19AudioRendererConfigEPNS0_16LightLimiterTypeEPNS0_12FinalMixTypeE")
EFF_ENABLE(audio_SetLightLimiterEnabled, "_ZN2nn5audio22SetLightLimiterEnabledEPNS0_16LightLimiterTypeEb")
EFF_ISEN(audio_IsLightLimiterEnabled, "_ZN2nn5audio21IsLightLimiterEnabledEPKNS0_16LightLimiterTypeE")
EFF_TRUE(audio_IsLightLimiterRemovable, "_ZN2nn5audio23IsLightLimiterRemovableEPKNS0_16LightLimiterTypeE")
EFF_IO(audio_SetLightLimiterInputOutput, "_ZN2nn5audio26SetLightLimiterInputOutputEPNS0_16LightLimiterTypeEPKaS4_i")
EFF_IOGET(audio_GetLightLimiterInputOutput, "_ZN2nn5audio26GetLightLimiterInputOutputEPKNS0_16LightLimiterTypeEPaS4_Pii")
EFF_SETT(audio_SetLightLimiterLookAhead, "_ZN2nn5audio28SetLightLimiterLookAheadTimeEPNS0_16LightLimiterTypeENS_8TimeSpanE", "lookahead")
EFF_GETT(audio_GetLightLimiterLookAhead, "_ZN2nn5audio28GetLightLimiterLookAheadTimeEPKNS0_16LightLimiterTypeE", "lookahead")
EFF_SETT(audio_SetLightLimiterAttack, "_ZN2nn5audio25SetLightLimiterAttackTimeEPNS0_16LightLimiterTypeENS_8TimeSpanE", "attack")
EFF_SETT(audio_SetLightLimiterRelease, "_ZN2nn5audio26SetLightLimiterReleaseTimeEPNS0_16LightLimiterTypeENS_8TimeSpanE", "release")
EFF_SETF(audio_SetLightLimiterThreshold, "_ZN2nn5audio24SetLightLimiterThresholdEPNS0_16LightLimiterTypeEf", "threshold")
EFF_SETF(audio_SetLightLimiterInputGain, "_ZN2nn5audio24SetLightLimiterInputGainEPNS0_16LightLimiterTypeEf", "in")
EFF_SETF(audio_SetLightLimiterOutputGain, "_ZN2nn5audio25SetLightLimiterOutputGainEPNS0_16LightLimiterTypeEf", "out")
EFF_SETI(audio_SetLightLimiterMode, "_ZN2nn5audio29SetLightLimiterProcessingModeEPNS0_16LightLimiterTypeENS1_14ProcessingModeE", "mode")

// HLE: NVIDIA multimedia `movie::` library (subsdk0) as used by TotK's pre-rendered cutscenes (Event/Movie/*.webm).
// The game (main 0x240c6c0..0x240e3d4) builds an Extractor (container) and one video Decoder, then runs Android
// MediaCodec-style loops: input slots are filled by Extractor::Read and queued with SendInputBufferForDecode, decoded
// frames come back as output slot indices, and GetOutputBuffer copies one NV12 frame into the game's own buffer, which
// it uploads to a Y (R8) and a UV (RG8) texture. The classes are opaque 0x10-byte objects (vptr + impl); the vtable
// layouts below come from subsdk0's _ZTVN5movie9ExtractorE / _ZTVN5movie7DecoderE (slot = offset from the vptr).
// Demux + VP9 decode: movie_core.h (Windows VP9 MFT; Linux libvpx). Log prefix "[movie]".
#include "hle.h"
#include "movie_core.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

bool hle_fs_host_path(const char* p, std::string* out);   // hle_fs.cpp
void hle_sysevent_signal(u64 ev);                          // hle_os_thread.cpp (any nn::os event)

namespace mv {

constexpr u32 kOk = 0, kErr = 0x1000, kEos = 0x3001, kAgain = 0x3002;

// movie::Buffer (0x40 bytes, subsdk0 127e54..12806c): offset, size, int data, owns, capacity, base, nvn dev, nvn sync
struct GBuffer { s32 offset, size; u64 data; u8 owns, pad[7]; u64 capacity; u64 pad2; u64 base; u64 nvn_dev, nvn_sync; };
static_assert(sizeof(GBuffer) == 0x40);

// movie::MediaData is one pointer to the implementation; ours points at an MD.
struct MD {
    struct Val { int kind = 0; s64 i = 0; std::string s; s32 r[4] = {}; };   // kind 1 int32, 2 int64, 3 string, 4 rect
    std::map<std::string, Val> kv;
    void i32(const char* k, s32 v) { auto& x = kv[k]; x.kind = 1; x.i = v; }
    void i64(const char* k, s64 v) { auto& x = kv[k]; x.kind = 2; x.i = v; }
    void str(const char* k, const std::string& v) { auto& x = kv[k]; x.kind = 3; x.s = v; }
    void rect(const char* k, s32 l, s32 t, s32 r, s32 b) { auto& x = kv[k]; x.kind = 4; x.r[0] = l; x.r[1] = t; x.r[2] = r; x.r[3] = b; }
};
static MD* md_of(u64 g) {
    if (g < 0x10000 || (g & 7)) { static int n = 0; if (n++ < 4) printf("[movie] MediaData pointer %llx rejected\n", (unsigned long long)g); return nullptr; }
    MD*& m = *(MD**)g;
    if (!m) m = new MD();
    return m;
}

static std::atomic<int> g_logs{0};
#define MLOG(...) do { if (mv::g_logs++ < 400) printf(__VA_ARGS__); } while (0)

// ------------------------------------------------------------------ Extractor
struct Ext {
    int type = 0;
    std::string path;
    movie_host::WebM wm;
    std::vector<bool> sel;
    size_t cur = 0;                       // next sample (of a selected track)
    u64 reads = 0;
    void skip() { while (cur < wm.samples.size() && !sel[wm.samples[cur].track]) cur++; }
};
static Ext* ext(Ctx* c) { return *(Ext**)(c->x[0] + 8); }

static void x_dtor(Ctx* c) { Ext* e = ext(c); MLOG("[movie] Extractor %llx destroyed (%llu reads)\n", (unsigned long long)c->x[0], e ? (unsigned long long)e->reads : 0ull); delete e; *(Ext**)(c->x[0] + 8) = nullptr; }
static void x_set_source(Ctx* c) {      // SetDataSource(const char* path)
    Ext* e = ext(c); const char* p = (const char*)c->x[1];
    std::string hp; char err[200] = "";
    if (!hle_fs_host_path(p, &hp)) { MLOG("[movie] SetDataSource %s: unknown mount\n", p); c->x[0] = kErr; return; }
    if (!e->wm.open(hp.c_str(), err, sizeof err)) { MLOG("[movie] SetDataSource %s: %s\n", hp.c_str(), err); c->x[0] = kErr; return; }
    e->path = p; e->sel.assign(e->wm.tracks.size(), false); e->cur = 0;
    MLOG("[movie] SetDataSource %s: %zu tracks, %zu samples, %.1f s\n", p, e->wm.tracks.size(), e->wm.samples.size(),
         e->wm.duration * e->wm.timescale / 1e9);
    c->x[0] = kOk;
}
static void x_set_source_mem(Ctx* c) { MLOG("[movie] SetDataSource(memory %llx, %d) unsupported\n", (unsigned long long)c->x[1], (int)c->x[2]); c->x[0] = kErr; }
static void x_set_source_reader(Ctx* c) { MLOG("[movie] SetDataSource(ClientStreamReader) unsupported\n"); c->x[0] = kErr; }
static void x_track_count(Ctx* c) { Ext* e = ext(c); *(s32*)c->x[1] = (s32)e->wm.tracks.size(); c->x[0] = kOk; }
static void x_track_config(Ctx* c) {    // GetTrackConfiguration(int, MediaData*)
    Ext* e = ext(c); int i = (int)c->x[1]; MD* m = md_of(c->x[2]);
    if (i < 0 || i >= (int)e->wm.tracks.size() || !m) { c->x[0] = kErr; return; }
    const movie_host::Track& t = e->wm.tracks[i];
    std::string mime = t.codec == "V_VP9" ? "video/x-vnd.on2.vp9" : t.codec == "V_VP8" ? "video/x-vnd.on2.vp8" :
                       t.codec == "V_MPEG4/ISO/AVC" ? "video/avc" : t.codec == "A_OPUS" ? "audio/opus" : t.codec == "A_VORBIS" ? "audio/vorbis" : "application/octet-stream";
    m->str("mime", mime);
    s64 dur = (s64)(e->wm.duration * e->wm.timescale / 1000.0);
    m->i64("durationUs", dur);
    if (t.type == 1) {
        m->i32("width", t.width); m->i32("height", t.height);
        m->i32("frame-rate", t.default_dur_ns ? (s32)(1e9 / t.default_dur_ns + 0.5) : 30);
        m->i32("max-input-size", 4 << 20);
    } else if (t.type == 2) {
        m->i32("sample-rate", (s32)t.rate); m->i32("channel-count", t.channels);
    }
    MLOG("[movie] track %d: %s %dx%d %.0f us\n", i, mime.c_str(), t.width, t.height, (double)dur);
    c->x[0] = kOk;
}
static void x_select(Ctx* c) { Ext* e = ext(c); int i = (int)c->x[1]; if (i >= 0 && i < (int)e->sel.size()) { e->sel[i] = true; e->skip(); c->x[0] = kOk; } else c->x[0] = kErr; }
static void x_unselect(Ctx* c) { Ext* e = ext(c); int i = (int)c->x[1]; if (i >= 0 && i < (int)e->sel.size()) { e->sel[i] = false; e->skip(); c->x[0] = kOk; } else c->x[0] = kErr; }
static void x_seek(Ctx* c) {            // SeekTo(us): last key frame at or before
    Ext* e = ext(c); s64 us = (s64)c->x[1]; size_t best = 0;
    for (size_t i = 0; i < e->wm.samples.size(); i++) {
        const auto& s = e->wm.samples[i];
        if (s.pts_us > us) break;
        if (s.key && e->sel[s.track]) best = i;
    }
    e->cur = best; e->skip();
    MLOG("[movie] SeekTo %lld us -> sample %zu\n", (long long)us, e->cur);
    c->x[0] = kOk;
}
static void x_track_for_data(Ctx* c) {  // GetTrackIndexForAvailableData(size_t*)
    Ext* e = ext(c);
    if (e->cur >= e->wm.samples.size()) { c->x[0] = kEos; return; }
    *(u64*)c->x[1] = (u64)e->wm.samples[e->cur].track; c->x[0] = kOk;
}
static void x_read(Ctx* c) {            // Read(Buffer*)
    Ext* e = ext(c); GBuffer* b = (GBuffer*)c->x[1];
    if (e->cur >= e->wm.samples.size()) { b->offset = 0; b->size = 0; c->x[0] = kEos; return; }
    const auto& s = e->wm.samples[e->cur];
    if (!b->base || s.size > b->capacity) { MLOG("[movie] Read: sample %zu (%u bytes) does not fit buffer %llx cap %llu\n", e->cur, s.size, (unsigned long long)b->base, (unsigned long long)b->capacity); c->x[0] = kErr; return; }
    memcpy((void*)b->base, e->wm.d + s.off, s.size);
    b->offset = 0; b->size = (s32)s.size;
    e->reads++;
    c->x[0] = kOk;
}
static void x_advance(Ctx* c) { Ext* e = ext(c); if (e->cur < e->wm.samples.size()) { e->cur++; e->skip(); } c->x[0] = kOk; }
static void x_sample_time(Ctx* c) {
    Ext* e = ext(c);
    if (e->cur >= e->wm.samples.size()) { c->x[0] = kEos; return; }
    *(s64*)c->x[1] = e->wm.samples[e->cur].pts_us; c->x[0] = kOk;
}
static void x_sample_meta(Ctx* c) {
    Ext* e = ext(c); MD* m = md_of(c->x[1]);
    if (e->cur >= e->wm.samples.size() || !m) { c->x[0] = kEos; return; }
    m->i64("timeUs", e->wm.samples[e->cur].pts_us); m->i32("is-sync-frame", e->wm.samples[e->cur].key);
    c->x[0] = kOk;
}
static void x_buffered(Ctx* c) {
    Ext* e = ext(c);
    *(s64*)c->x[1] = 0; *(s64*)c->x[2] = (s64)(e->wm.duration * e->wm.timescale / 1000.0); c->x[0] = kOk;
}
static void x_unused(Ctx* c) { MLOG("[movie] Extractor: unexpected slot\n"); c->x[0] = kErr; }

// ------------------------------------------------------------------ Decoder
constexpr int kIn = 4, kOut = 6;
constexpr u32 kInCap = 4 << 20;
struct Dec {
    int type = 0, fmt = 0, mode = 0;
    u64 ev_in = 0, ev_out = 0, ev_fmt = 0, ev_err = 0;
    int w = 0, h = 0;
    std::mutex m; std::condition_variable cv;
    enum { FREE, OWNED, QUEUED } in_state[kIn] = {};
    u8* in_mem[kIn] = {};
    struct Pkt { int idx; u32 off, size; s64 pts; u32 flags; };
    std::deque<Pkt> pkts;
    enum { OFREE, OFILLED, OGAME } out_state[kOut] = {};
    std::vector<u8> out_frame[kOut];
    s64 out_pts[kOut] = {}; u32 out_flags[kOut] = {};
    std::deque<int> ready;                 // filled output slots in decode order
    std::thread th; bool quit = false, started = false, eos_out = false, fmt_sent = false;
    u64 frames = 0, inputs = 0; double dec_ms = 0;

    Dec() { for (int i = 0; i < kIn; i++) in_mem[i] = (u8*)VirtualAlloc(nullptr, kInCap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }
    ~Dec() { stop(); for (int i = 0; i < kIn; i++) VirtualFree(in_mem[i], 0, MEM_RELEASE); }
    int free_in() const { int n = 0; for (int i = 0; i < kIn; i++) n += in_state[i] == FREE; return n; }

    void run() {
#ifdef _WIN32
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif
        movie_host::Vp9Dec vd; char err[200] = "";
        bool ok = vd.open(w, h, err, sizeof err);
        MLOG("[movie] decoder thread: %s %dx%d%s%s\n", ok ? vd.name : "VP9 decoder FAILED", w, h, ok ? "" : ": ", ok ? "" : err);
        if (!ok && ev_err) hle_sysevent_signal(ev_err);
        auto emit = [&](const u8* nv12, s64 pts) {   // wait for a free output slot, then publish the frame
            std::unique_lock<std::mutex> l(m);
            int slot = -1;
            cv.wait(l, [&] { if (quit) return true; for (int i = 0; i < kOut; i++) if (out_state[i] == OFREE) { slot = i; return true; } return false; });
            if (quit || slot < 0) return;
            out_frame[slot].assign(nv12, nv12 + (size_t)w * h * 3 / 2);
            out_pts[slot] = pts; out_flags[slot] = 0; out_state[slot] = OFILLED; ready.push_back(slot);
            frames++;
            bool first = !fmt_sent; fmt_sent = true;
            l.unlock();
            if (first && ev_fmt) hle_sysevent_signal(ev_fmt);
            if (ev_out) hle_sysevent_signal(ev_out);
        };
        for (;;) {
            Pkt p;
            {
                std::unique_lock<std::mutex> l(m);
                cv.wait(l, [&] { return quit || !pkts.empty(); });
                if (quit) break;
                p = pkts.front(); pkts.pop_front();
            }
            if (ok && p.size) {
                LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a);
                if (!vd.decode(in_mem[p.idx] + p.off, p.size, p.pts, emit)) MLOG("[movie] decode error at input %llu\n", (unsigned long long)inputs);
                QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f); dec_ms += (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
            }
            inputs++;
            bool eos = (p.flags & 4) != 0, send_eos = false;
            if (eos && ok) vd.drain(emit);
            {
                std::lock_guard<std::mutex> l(m);
                in_state[p.idx] = FREE;
                if (eos && !eos_out) send_eos = eos_out = true;
            }
            if (ev_in) hle_sysevent_signal(ev_in);
            if (send_eos) {                    // EOS marker: an empty output slot with flags 4
                std::unique_lock<std::mutex> l(m);
                int slot = -1;
                cv.wait(l, [&] { if (quit) return true; for (int i = 0; i < kOut; i++) if (out_state[i] == OFREE) { slot = i; return true; } return false; });
                if (quit || slot < 0) break;
                out_frame[slot].clear(); out_pts[slot] = 0; out_flags[slot] = 4; out_state[slot] = OFILLED; ready.push_back(slot);
                l.unlock();
                MLOG("[movie] end of stream: %llu frames from %llu inputs, decode %.0f ms (%.2f ms/frame)\n", (unsigned long long)frames,
                     (unsigned long long)inputs, dec_ms, frames ? dec_ms / frames : 0.0);
                if (ev_out) hle_sysevent_signal(ev_out);
            }
        }
        vd.close();
#ifdef _WIN32
        CoUninitialize();
#endif
    }
    void start() {
        { std::lock_guard<std::mutex> l(m); if (started) return; started = true; quit = false; }
        th = std::thread([this] { run(); });
        if (ev_in) hle_sysevent_signal(ev_in);
    }
    void stop() {
        { std::lock_guard<std::mutex> l(m); if (!started) return; quit = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
        std::lock_guard<std::mutex> l(m);
        started = false;
    }
    void clear() {                         // flush: drop queued input and decoded output
        std::lock_guard<std::mutex> l(m);
        pkts.clear(); ready.clear();
        for (int i = 0; i < kIn; i++) in_state[i] = FREE;
        for (int i = 0; i < kOut; i++) out_state[i] = OFREE;
        eos_out = false;
    }
};
static Dec* dec(Ctx* c) { return *(Dec**)(c->x[0] + 8); }

static void d_dtor(Ctx* c) {
    Dec* d = dec(c);
    if (d) MLOG("[movie] Decoder %llx destroyed: %llu frames, decode %.0f ms\n", (unsigned long long)c->x[0], (unsigned long long)d->frames, d->dec_ms);
    delete d; *(Dec**)(c->x[0] + 8) = nullptr;
}
static void d_configure(Ctx* c) {
    Dec* d = dec(c); MD* m = md_of(c->x[1]);
    if (m) {
        auto it = m->kv.find("width"); if (it != m->kv.end()) d->w = (int)it->second.i;
        it = m->kv.find("height"); if (it != m->kv.end()) d->h = (int)it->second.i;
    }
    MLOG("[movie] Decoder::Configure %dx%d (type %d, output format %d, mode %d)\n", d->w, d->h, d->type, d->fmt, d->mode);
    c->x[0] = d->w > 0 && d->h > 0 ? kOk : kErr;
}
static void d_events(Ctx* c) {          // RegisterClientEvents(DecoderEvents*): input, output, format changed, error
    Dec* d = dec(c); const u64* e = (const u64*)c->x[1];
    d->ev_in = e[0]; d->ev_out = e[1]; d->ev_fmt = e[2]; d->ev_err = e[3];
    c->x[0] = kOk;
}
static void fill_format(Dec* d, MD* m) {
    m->str("mime", "video/raw"); m->i32("width", d->w); m->i32("height", d->h); m->i32("stride", d->w); m->i32("slice-height", d->h);
    m->i32("color-format", 21 /* YUV420SemiPlanar */); m->rect("crop", 0, 0, d->w - 1, d->h - 1);
}
static void d_get_format(Ctx* c) { Dec* d = dec(c); MD* m = md_of(c->x[1]); if (m) fill_format(d, m); c->x[0] = kOk; }
static void d_start(Ctx* c) { dec(c)->start(); c->x[0] = kOk; }
static void d_stop(Ctx* c) { dec(c)->stop(); c->x[0] = kOk; }
static void d_reset(Ctx* c) { Dec* d = dec(c); d->stop(); d->clear(); c->x[0] = kOk; }
static void d_flush(Ctx* c) {
    Dec* d = dec(c); bool was = d->started;
    d->stop(); d->clear();
    if (was) d->start();
    c->x[0] = kOk;
}
static void d_acquire_in(Ctx* c) {      // AcquireInputBufferIndex(int* index, int* remaining)
    Dec* d = dec(c); std::lock_guard<std::mutex> l(d->m);
    for (int i = 0; i < kIn; i++)
        if (d->in_state[i] == Dec::FREE) {
            d->in_state[i] = Dec::OWNED; *(s32*)c->x[1] = i;
            if (c->x[2]) *(s32*)c->x[2] = d->free_in();
            c->x[0] = kOk; return;
        }
    if (c->x[2]) *(s32*)c->x[2] = 0;
    c->x[0] = kAgain;
}
static void d_get_in(Ctx* c) {          // GetInputBuffer(int, Buffer*)
    Dec* d = dec(c); int i = (int)c->x[1]; GBuffer* b = (GBuffer*)c->x[2];
    if (i < 0 || i >= kIn) { c->x[0] = kErr; return; }
    b->base = (u64)d->in_mem[i]; b->capacity = kInCap; b->offset = 0; b->size = 0; b->owns = 0;
    c->x[0] = kOk;
}
static void d_send_in(Ctx* c) {         // SendInputBufferForDecode(int index, int offset, int size, long pts, uint32 flags)
    Dec* d = dec(c); int i = (int)c->x[1];
    if (i < 0 || i >= kIn) { c->x[0] = kErr; return; }
    {
        std::lock_guard<std::mutex> l(d->m);
        d->in_state[i] = Dec::QUEUED;
        d->pkts.push_back({i, (u32)c->x[2], (u32)c->x[3], (s64)c->x[4], (u32)c->x[5]});
    }
    d->cv.notify_all();
    c->x[0] = kOk;
}
static void d_acquire_out(Ctx* c) {     // AcquireOutputBufferIndex(int* index, long* pts, uint32* flags, int* remaining)
    Dec* d = dec(c); std::lock_guard<std::mutex> l(d->m);
    if (d->ready.empty()) { if (c->x[4]) *(s32*)c->x[4] = 0; c->x[0] = kAgain; return; }
    int s = d->ready.front(); d->ready.pop_front();
    d->out_state[s] = Dec::OGAME;
    *(s32*)c->x[1] = s;
    if (c->x[2]) *(s64*)c->x[2] = d->out_pts[s];
    if (c->x[3]) *(u32*)c->x[3] = d->out_flags[s];
    if (c->x[4]) *(s32*)c->x[4] = (s32)d->ready.size();
    c->x[0] = kOk;
}
static void get_out(Ctx* c, bool with_md) {   // GetOutputBuffer(int, Buffer*) [slot 0x68] / (int, Buffer*, MediaData*) [slot 0x88]
    Dec* d = dec(c); int s = (int)c->x[1]; GBuffer* b = (GBuffer*)c->x[2];
    if (s < 0 || s >= kOut) { c->x[0] = kErr; return; }
    std::vector<u8>& f = d->out_frame[s];
    static std::atomic<int> logged{0};
    if (logged++ < 2) MLOG("[movie] GetOutputBuffer slot %d: frame %zu bytes -> game buffer %llx capacity %llu offset %d\n", s, f.size(),
                           (unsigned long long)b->base, (unsigned long long)b->capacity, b->offset);
    if (!b->base || f.size() > b->capacity) { MLOG("[movie] GetOutputBuffer: frame %zu bytes does not fit (capacity %llu)\n", f.size(), (unsigned long long)b->capacity); c->x[0] = kErr; return; }
    memcpy((void*)b->base, f.data(), f.size());
    b->offset = 0; b->size = (s32)f.size();
    // the 2-argument form leaves garbage in x3 (play.log 09:21: x3 = 1 -> md_of(1) read address 1 and crashed)
    if (with_md && c->x[3]) { MD* m = md_of(c->x[3]); if (m) fill_format(d, m); }
    c->x[0] = kOk;
}
static void d_get_out(Ctx* c) { get_out(c, false); }
static void d_get_out_md(Ctx* c) { get_out(c, true); }
static void d_out_size(Ctx* c) {        // GetOutputBufferSize(int*, int*)
    Dec* d = dec(c);
    if (c->x[1]) *(s32*)c->x[1] = d->w * d->h * 3 / 2;
    if (c->x[2]) *(s32*)c->x[2] = kOut;
    c->x[0] = kOk;
}
static void d_release_out(Ctx* c) {     // ReleaseOutputBufferIndex(int)
    Dec* d = dec(c); int s = (int)c->x[1];
    if (s >= 0 && s < kOut) { std::lock_guard<std::mutex> l(d->m); d->out_state[s] = Dec::OFREE; }
    d->cv.notify_all();
    c->x[0] = kOk;
}
static void d_last_error(Ctx* c) { c->x[0] = kOk; }
static void d_out_format(Ctx* c) { Dec* d = dec(c); MD* m = md_of(c->x[2]); if (m) fill_format(d, m); c->x[0] = kOk; }

// vtables: slot k at vptr + 8k
static u64 g_ext_vt[16], g_dec_vt[19];
static void init_vtables() {
    static std::once_flag f;
    std::call_once(f, [] {
        GuestFn ex[16] = {x_dtor, x_dtor, x_set_source, x_set_source_mem, x_track_count, x_track_config, x_select, x_unselect,
                          x_seek, x_track_for_data, x_read, x_advance, x_sample_time, x_sample_meta, x_buffered, x_set_source_reader};
        for (int i = 0; i < 16; i++) g_ext_vt[i] = rt_host_fn_addr(ex[i]);
        GuestFn dv[19] = {d_dtor, d_dtor, d_configure, d_events, d_get_format, d_start, d_stop, d_reset, d_flush, d_acquire_in,
                          d_get_in, d_send_in, d_acquire_out, d_get_out, d_out_size, d_release_out, d_last_error, d_get_out_md, d_out_format};
        for (int i = 0; i < 19; i++) g_dec_vt[i] = rt_host_fn_addr(dv[i]);
        (void)x_unused;
    });
}

}  // namespace mv

HLE_AS(movie_SetAllocator, "_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_") { c->x[0] = 0; }
HLE_AS(nv_SetGraphicsAllocator, "_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_") { c->x[0] = 0; }
HLE_AS(nv_SetGraphicsDevtoolsAllocator, "_ZN2nv28SetGraphicsDevtoolsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_") { c->x[0] = 0; }
HLE_AS(nv_InitializeGraphics, "_ZN2nv18InitializeGraphicsEPvm") { c->x[0] = 0; }
HLE_AS(movie_GetThreadPriority, "_ZN5movie17GetThreadPriorityEv") { c->x[0] = 16; }

HLE_AS(movie_ExtractorC1, "_ZN5movie9ExtractorC1ENS_13ContainerTypeE") {
    mv::init_vtables();
    mv::Ext* e = new mv::Ext(); e->type = (int)c->x[1];
    *(u64*)c->x[0] = (u64)&mv::g_ext_vt[0]; *(mv::Ext**)(c->x[0] + 8) = e;
    MLOG("[movie] Extractor %llx (container type %d)\n", (unsigned long long)c->x[0], e->type);
}
HLE_AS(movie_DecoderC1, "_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeEi") {
    mv::init_vtables();
    mv::Dec* d = new mv::Dec(); d->type = (int)c->x[1]; d->fmt = (int)c->x[2]; d->mode = (int)c->x[3];
    *(u64*)c->x[0] = (u64)&mv::g_dec_vt[0]; *(mv::Dec**)(c->x[0] + 8) = d;
    MLOG("[movie] Decoder %llx (type %d, output format %d, mode %d, priority %d)\n", (unsigned long long)c->x[0], d->type, d->fmt, d->mode, (int)c->x[4]);
}

HLE_AS(movie_MediaDataC1, "_ZN5movie9MediaDataC1Ev") { *(mv::MD**)c->x[0] = new mv::MD(); }
HLE_AS(movie_MediaDataD1, "_ZN5movie9MediaDataD1Ev") { delete *(mv::MD**)c->x[0]; *(mv::MD**)c->x[0] = nullptr; }
HLE_AS(movie_MediaDataClear, "_ZN5movie9MediaData5ClearEv") { if (mv::MD* m = *(mv::MD**)c->x[0]) m->kv.clear(); }
HLE_AS(movie_MediaDataFindString, "_ZNK5movie9MediaData10FindStringEPKcPS2_") {
    mv::MD* m = *(mv::MD**)c->x[0]; c->x[0] = 0;
    if (!m) return;
    auto it = m->kv.find((const char*)c->x[1]);
    if (it == m->kv.end() || it->second.kind != 3) return;
    *(u64*)c->x[2] = (u64)it->second.s.c_str(); c->x[0] = 1;
}
HLE_AS(movie_MediaDataFindInt32, "_ZNK5movie9MediaData9FindInt32EPKcPi") {
    mv::MD* m = *(mv::MD**)c->x[0]; c->x[0] = 0;
    if (!m) return;
    auto it = m->kv.find((const char*)c->x[1]);
    if (it == m->kv.end() || (it->second.kind != 1 && it->second.kind != 2)) return;
    *(s32*)c->x[2] = (s32)it->second.i; c->x[0] = 1;
}
HLE_AS(movie_MediaDataFindRect, "_ZNK5movie9MediaData8FindRectEPKcPiS3_S3_S3_") {
    mv::MD* m = *(mv::MD**)c->x[0]; c->x[0] = 0;
    if (!m) return;
    auto it = m->kv.find((const char*)c->x[1]);
    if (it == m->kv.end() || it->second.kind != 4) return;
    for (int i = 0; i < 4; i++) *(s32*)c->x[2 + i] = it->second.r[i];
    c->x[0] = 1;
}

HLE_AS(movie_BufferC1, "_ZN5movie6BufferC1Ev") { memset((void*)c->x[0], 0, sizeof(mv::GBuffer)); }
HLE_AS(movie_BufferD1, "_ZN5movie6BufferD1Ev") { }
HLE_AS(movie_BufferSize, "_ZNK5movie6Buffer4SizeEv") { c->x[0] = (u64)(u32)((mv::GBuffer*)c->x[0])->size; }
HLE_AS(movie_BufferOffset, "_ZNK5movie6Buffer6OffsetEv") { c->x[0] = (u64)(u32)((mv::GBuffer*)c->x[0])->offset; }
HLE_AS(movie_BufferSetData, "_ZN5movie6Buffer18SetDataAndCapacityEPvm") {
    mv::GBuffer* b = (mv::GBuffer*)c->x[0]; b->owns = 0; b->base = c->x[1]; b->capacity = c->x[2];
}
HLE_AS(movie_BufferSetRange, "_ZN5movie6Buffer8SetRangeEii") { mv::GBuffer* b = (mv::GBuffer*)c->x[0]; b->offset = (s32)c->x[1]; b->size = (s32)c->x[2]; }

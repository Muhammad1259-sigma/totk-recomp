// Host-side movie support for hle_movie.cpp and tools/movieprobe.cpp: a WebM (Matroska) demuxer over a mapped file
// and a Media Foundation VP9 decoder that hands out tightly packed NV12 frames.
// TotK's pre-rendered cutscenes (romfs Event/Movie/*.webm) are all VP9 1920x1080, 30 fps, video only (measured 2026-10-15).
// VP9 decoding comes from Windows (the "VP9 Video Extensions" MFT); nothing is bundled.
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#else
// Linux: the file is mmap'd and VP9 is decoded by libvpx (static, apt libvpx-dev / webmproject/libvpx) into the same tight NV12 frames.
#include <windows.h>      // runtime/linux shim (hle_movie.cpp: VirtualAlloc, QueryPerformanceCounter)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vpx/vpx_decoder.h>
#include <vpx/vp8dx.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace movie_host {

struct Track {
    int number = 0, type = 0;              // Matroska track number; type 1 video, 2 audio
    std::string codec;                     // "V_VP9", ...
    std::vector<uint8_t> priv;
    int width = 0, height = 0, channels = 0;
    double rate = 0;
    uint64_t default_dur_ns = 0;
};
struct Sample { int track; int64_t pts_us; uint64_t off; uint32_t size; bool key; };   // track = index into tracks

struct WebM {
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
#endif
    const uint8_t* d = nullptr; uint64_t n = 0;
    uint64_t timescale = 1000000;          // ns per timecode unit
    double duration = 0;                   // in timecode units
    std::vector<Track> tracks;
    std::vector<Sample> samples;
    int laced = 0;                         // laced blocks seen (not supported; logged)

    static bool id_at(const uint8_t* d, uint64_t n, uint64_t p, uint32_t* id, int* len) {
        if (p >= n) return false;
        uint8_t b = d[p]; int l = 1;
        for (uint8_t m = 0x80; l <= 4 && !(b & m); m >>= 1) l++;
        if (l > 4 || p + l > n) return false;
        uint32_t v = 0; for (int i = 0; i < l; i++) v = (v << 8) | d[p + i];
        *id = v; *len = l; return true;
    }
    static bool size_at(const uint8_t* d, uint64_t n, uint64_t p, uint64_t* sz, int* len, bool* unknown) {
        if (p >= n) return false;
        uint8_t b = d[p]; int l = 1; uint8_t m = 0x80;
        for (; l <= 8 && !(b & m); m >>= 1) l++;
        if (l > 8 || p + l > n) return false;
        uint64_t v = b & (m - 1); bool ones = v == (uint64_t)(m - 1);
        for (int i = 1; i < l; i++) { v = (v << 8) | d[p + i]; ones = ones && d[p + i] == 0xff; }
        *sz = v; *len = l; *unknown = ones; return true;
    }
    uint64_t uint_of(uint64_t p, uint64_t sz) const { uint64_t v = 0; for (uint64_t i = 0; i < sz && i < 8; i++) v = (v << 8) | d[p + i]; return v; }
    double float_of(uint64_t p, uint64_t sz) const {
        uint64_t v = uint_of(p, sz);
        if (sz == 4) { uint32_t u = (uint32_t)v; float f; memcpy(&f, &u, 4); return f; }
        double f; memcpy(&f, &v, 8); return f;
    }
    static bool is_master(uint32_t id) {
        switch (id) {
        case 0x18538067: case 0x1654AE6B: case 0xAE: case 0xE0: case 0xE1: case 0x1549A966: case 0x1F43B675: case 0xA0: return true;
        default: return false;
        }
    }
    static bool is_top(uint32_t id) {   // level-1 ids end an unknown-size cluster
        return id == 0x1F43B675 || id == 0x1C53BB6B || id == 0x1254C367 || id == 0x1941A469 || id == 0x1043A770 || id == 0x114D9B74 ||
               id == 0x1549A966 || id == 0x1654AE6B;
    }
    int track_index(int number) const { for (size_t i = 0; i < tracks.size(); i++) if (tracks[i].number == number) return (int)i; return -1; }

    void block(uint64_t p, uint64_t sz, int64_t cluster_tc, bool simple, bool key_hint) {
        uint64_t tn; int l; bool unk;
        if (!size_at(d, n, p, &tn, &l, &unk) || sz < (uint64_t)l + 3) return;
        int16_t rel = (int16_t)((d[p + l] << 8) | d[p + l + 1]);
        uint8_t flags = d[p + l + 2];
        if (flags & 0x06) { laced++; return; }
        int ti = track_index((int)tn);
        if (ti < 0) return;
        Sample s;
        s.track = ti;
        s.pts_us = (int64_t)((double)(cluster_tc + rel) * (double)timescale / 1000.0);
        s.off = p + l + 3; s.size = (uint32_t)(sz - l - 3);
        s.key = simple ? (flags & 0x80) != 0 : key_hint;
        samples.push_back(s);
    }
    void walk(uint64_t p, uint64_t end, uint32_t parent, Track* tr, int64_t* cluster_tc) {
        while (p < end) {
            uint32_t id; int il; uint64_t sz; int sl; bool unk;
            if (!id_at(d, n, p, &id, &il) || !size_at(d, n, p + il, &sz, &sl, &unk)) return;
            uint64_t q = p + il + sl;
            if (unk) {                                  // unknown size: runs to the parent's end or the next level-1 element
                uint64_t e = q;
                if (id == 0x1F43B675) {
                    while (e < end) {
                        uint32_t cid; int cil; uint64_t csz; int csl; bool cunk;
                        if (!id_at(d, n, e, &cid, &cil) || is_top(cid) || !size_at(d, n, e + cil, &csz, &csl, &cunk)) break;
                        e += cil + csl + csz;
                    }
                } else e = end;
                sz = e - q;
            }
            if (q + sz > end) sz = end - q;
            switch (id) {
            case 0x2AD7B1: timescale = uint_of(q, sz); break;
            case 0x4489: duration = float_of(q, sz); break;
            case 0xAE: { tracks.emplace_back(); walk(q, q + sz, id, &tracks.back(), cluster_tc); break; }
            case 0xD7: if (tr) tr->number = (int)uint_of(q, sz); break;
            case 0x83: if (tr) tr->type = (int)uint_of(q, sz); break;
            case 0x86: if (tr) tr->codec.assign((const char*)d + q, (size_t)sz); break;
            case 0x63A2: if (tr) tr->priv.assign(d + q, d + q + sz); break;
            case 0x23E383: if (tr) tr->default_dur_ns = uint_of(q, sz); break;
            case 0xB0: if (tr) tr->width = (int)uint_of(q, sz); break;
            case 0xBA: if (tr) tr->height = (int)uint_of(q, sz); break;
            case 0xB5: if (tr) tr->rate = float_of(q, sz); break;
            case 0x9F: if (tr) tr->channels = (int)uint_of(q, sz); break;
            case 0x1F43B675: { int64_t tc = 0; walk(q, q + sz, id, nullptr, &tc); break; }
            case 0xE7: if (cluster_tc) *cluster_tc = (int64_t)uint_of(q, sz); break;
            case 0xA3: if (cluster_tc) block(q, sz, *cluster_tc, true, false); break;
            case 0xA0: {                                // BlockGroup: key frame unless it has a ReferenceBlock (0xFB)
                bool ref = false; uint64_t bp = 0, bsz = 0, r = q;
                while (r < q + sz) {
                    uint32_t cid; int cil; uint64_t csz; int csl; bool cunk;
                    if (!id_at(d, n, r, &cid, &cil) || !size_at(d, n, r + cil, &csz, &csl, &cunk)) break;
                    if (cid == 0xFB) ref = true;
                    if (cid == 0xA1) { bp = r + cil + csl; bsz = csz; }
                    r += cil + csl + csz;
                }
                if (bp && cluster_tc) block(bp, bsz, *cluster_tc, false, !ref);
                break;
            }
            default:
                if (is_master(id)) walk(q, q + sz, id, tr, cluster_tc);
            }
            p = q + sz;
        }
    }
    bool open(const char* path, char* err, size_t errn) {
#ifdef _WIN32
        int wn = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
        std::wstring w(wn ? wn - 1 : 0, 0); MultiByteToWideChar(CP_UTF8, 0, path, -1, w.data(), wn);
        file = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (file == INVALID_HANDLE_VALUE) { snprintf(err, errn, "open failed (%lu)", GetLastError()); return false; }
        LARGE_INTEGER li; GetFileSizeEx(file, &li); n = (uint64_t)li.QuadPart;
        map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        d = map ? (const uint8_t*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0) : nullptr;
        if (!d) { snprintf(err, errn, "map failed (%lu)", GetLastError()); close(); return false; }
#else
        int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) { snprintf(err, errn, "open failed (%d)", errno); return false; }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) { snprintf(err, errn, "stat failed (%d)", errno); ::close(fd); return false; }
        n = (uint64_t)st.st_size;
        void* m = mmap(nullptr, (size_t)n, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);                                           // (the mapping keeps the file)
        if (m == MAP_FAILED) { snprintf(err, errn, "map failed (%d)", errno); n = 0; return false; }
        d = (const uint8_t*)m;
#endif
        int64_t tc = 0;
        walk(0, n, 0, nullptr, &tc);
        if (tracks.empty()) { snprintf(err, errn, "no tracks"); close(); return false; }
        return true;
    }
    void close() {
#ifdef _WIN32
        if (d) UnmapViewOfFile(d);
        if (map) CloseHandle(map);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        d = nullptr; map = nullptr; file = INVALID_HANDLE_VALUE; n = 0;
#else
        if (d) munmap((void*)d, (size_t)n);
        d = nullptr; n = 0;
#endif
    }
    ~WebM() { close(); }
};

#ifdef _WIN32
// Media Foundation VP9 decoder (synchronous MFT). The calling thread must have called CoInitializeEx.
struct Vp9Dec {
    IMFTransform* t = nullptr;
    IMFSample* out_sample = nullptr;       // when the MFT does not provide samples
    int w = 0, h = 0;                      // display size (tight NV12 output)
    UINT32 ow = 0, oh = 0;                 // decoder buffer size (may be aligned, e.g. 1088)
    LONG stride = 0;
    bool provides = false; DWORD out_cb = 0;
    std::vector<uint8_t> frame;            // tight NV12 w*h*3/2
    char name[128] = "";
    uint64_t frames_out = 0, inputs = 0;

    static void startup() { static bool done = false; if (!done) { done = true; MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET); } }

    bool set_nv12_output(char* err, size_t errn) {
        for (DWORD i = 0;; i++) {
            IMFMediaType* ot = nullptr;
            HRESULT hr = t->GetOutputAvailableType(0, i, &ot);
            if (FAILED(hr)) { snprintf(err, errn, "no NV12 output type (hr %08lx)", (unsigned long)hr); return false; }
            GUID sub{}; ot->GetGUID(MF_MT_SUBTYPE, &sub);
            if (sub == MFVideoFormat_NV12) {
                hr = t->SetOutputType(0, ot, 0);
                if (SUCCEEDED(hr)) {
                    MFGetAttributeSize(ot, MF_MT_FRAME_SIZE, &ow, &oh);
                    UINT32 st = 0;
                    stride = SUCCEEDED(ot->GetUINT32(MF_MT_DEFAULT_STRIDE, &st)) ? (LONG)st : (LONG)ow;
                }
                ot->Release();
                if (FAILED(hr)) { snprintf(err, errn, "SetOutputType NV12 hr %08lx", (unsigned long)hr); return false; }
                MFT_OUTPUT_STREAM_INFO si{};
                t->GetOutputStreamInfo(0, &si);
                provides = (si.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
                out_cb = si.cbSize;
                if (out_sample) { out_sample->Release(); out_sample = nullptr; }
                return true;
            }
            ot->Release();
        }
    }
    bool open(int width, int height, char* err, size_t errn) {
        startup();
        w = width; h = height;
        MFT_REGISTER_TYPE_INFO in{MFMediaType_Video, MFVideoFormat_VP90};
        IMFActivate** acts = nullptr; UINT32 na = 0;
        const UINT32 flag_sets[] = {MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER,
                                    MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER | 0x400 /* UNTRUSTED_STOREMFT */};
        for (UINT32 fl : flag_sets) {
            if (SUCCEEDED(MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER, fl, &in, nullptr, &acts, &na)) && na) break;
            if (acts) CoTaskMemFree(acts);
            acts = nullptr; na = 0;
        }
        if (!na) { snprintf(err, errn, "no VP9 decoder MFT (install the Microsoft VP9 Video Extensions)"); return false; }
        HRESULT hr = E_FAIL;
        for (UINT32 i = 0; i < na && !t; i++) {
            WCHAR* fn = nullptr; UINT32 fl = 0;
            if (SUCCEEDED(acts[i]->GetAllocatedString(MFT_FRIENDLY_NAME_Attribute, &fn, &fl))) {
                WideCharToMultiByte(CP_UTF8, 0, fn, -1, name, sizeof name, nullptr, nullptr); CoTaskMemFree(fn);
            }
            hr = acts[i]->ActivateObject(IID_PPV_ARGS(&t));
        }
        for (UINT32 i = 0; i < na; i++) acts[i]->Release();
        CoTaskMemFree(acts);
        if (!t) { snprintf(err, errn, "ActivateObject hr %08lx", (unsigned long)hr); return false; }
        IMFMediaType* it = nullptr;
        MFCreateMediaType(&it);
        it->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        it->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_VP90);
        MFSetAttributeSize(it, MF_MT_FRAME_SIZE, (UINT32)w, (UINT32)h);
        hr = t->SetInputType(0, it, 0);
        it->Release();
        if (FAILED(hr)) { snprintf(err, errn, "SetInputType VP9 hr %08lx", (unsigned long)hr); close(); return false; }
        if (!set_nv12_output(err, errn)) { close(); return false; }
        t->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        t->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        frame.resize((size_t)w * h * 3 / 2);
        return true;
    }
    // copy one decoded sample into `frame` (tight NV12)
    void copy_out(IMFSample* s) {
        IMFMediaBuffer* mb = nullptr;
        if (FAILED(s->GetBufferByIndex(0, &mb))) return;
        IMF2DBuffer* b2 = nullptr; BYTE* base = nullptr; LONG pitch = 0; DWORD len = 0;
        bool locked2d = SUCCEEDED(mb->QueryInterface(IID_PPV_ARGS(&b2))) && SUCCEEDED(b2->Lock2D(&base, &pitch));
        if (!locked2d) { if (b2) { b2->Release(); b2 = nullptr; } mb->Lock(&base, nullptr, &len); pitch = stride ? stride : (LONG)ow; }
        UINT32 rows = oh ? oh : (UINT32)h;
        const BYTE* y = base; const BYTE* uv = base + (size_t)pitch * rows;
        uint8_t* dy = frame.data(); uint8_t* duv = dy + (size_t)w * h;
        for (int r = 0; r < h; r++) memcpy(dy + (size_t)r * w, y + (size_t)r * pitch, w);
        for (int r = 0; r < h / 2; r++) memcpy(duv + (size_t)r * w, uv + (size_t)r * pitch, w);
        if (locked2d) { b2->Unlock2D(); b2->Release(); } else mb->Unlock();
        mb->Release();
    }
    template <class F> bool pull(F&& cb) {   // drain every ready output; false on a hard error
        for (;;) {
            MFT_OUTPUT_DATA_BUFFER ob{}; DWORD status = 0;
            if (!provides) {
                if (!out_sample) {
                    IMFMediaBuffer* mb = nullptr;
                    DWORD need = out_cb ? out_cb : (DWORD)((size_t)(ow ? ow : w) * (oh ? oh : h) * 3 / 2);
                    MFCreateMemoryBuffer(need, &mb); MFCreateSample(&out_sample); out_sample->AddBuffer(mb); mb->Release();
                }
                ob.pSample = out_sample;
            }
            HRESULT hr = t->ProcessOutput(0, 1, &ob, &status);
            if (ob.pEvents) ob.pEvents->Release();
            if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return true;
            if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
                char err[160];
                if (!set_nv12_output(err, sizeof err)) { printf("[movie] vp9: %s\n", err); return false; }
                continue;
            }
            if (FAILED(hr)) { printf("[movie] vp9: ProcessOutput hr %08lx\n", (unsigned long)hr); return false; }
            LONGLONG ts = 0; ob.pSample->GetSampleTime(&ts);
            copy_out(ob.pSample);
            if (provides) ob.pSample->Release();
            frames_out++;
            cb(frame.data(), (int64_t)(ts / 10));
        }
    }
    template <class F> bool decode(const uint8_t* p, uint32_t n, int64_t pts_us, F&& cb) {
        IMFMediaBuffer* mb = nullptr; IMFSample* s = nullptr; BYTE* dst = nullptr;
        if (FAILED(MFCreateMemoryBuffer(n ? n : 1, &mb))) return false;
        mb->Lock(&dst, nullptr, nullptr); memcpy(dst, p, n); mb->Unlock(); mb->SetCurrentLength(n);
        MFCreateSample(&s); s->AddBuffer(mb); mb->Release();
        s->SetSampleTime(pts_us * 10);
        inputs++;
        HRESULT hr;
        while ((hr = t->ProcessInput(0, s, 0)) == MF_E_NOTACCEPTING)
            if (!pull(cb)) break;
        s->Release();
        if (FAILED(hr)) { printf("[movie] vp9: ProcessInput hr %08lx (input %llu, %u bytes)\n", (unsigned long)hr, (unsigned long long)inputs, n); return false; }
        return pull(cb);
    }
    template <class F> void drain(F&& cb) {
        if (!t) return;
        t->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
        t->ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0);
        pull(cb);
    }
    void flush() { if (t) t->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0); }
    void close() {
        if (out_sample) { out_sample->Release(); out_sample = nullptr; }
        if (t) { t->ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0); t->Release(); t = nullptr; }
    }
    ~Vp9Dec() { close(); }
};
#else
// libvpx VP9 decoder (software, 4 threads) with the same interface as the Media Foundation one above: open / decode / drain / flush /
// close, frames handed to the callback as tight NV12 (w x h Y, then interleaved UV), pts in microseconds. VP9 has no output
// reordering, so a frame comes out with the pts of the input that produced it (a hidden alt-ref input produces none; a
// show-existing-frame input produces one).
struct Vp9Dec {
    vpx_codec_ctx_t ctx{};
    bool ok = false;
    int w = 0, h = 0;                      // display size (tight NV12 output)
    std::vector<uint8_t> frame;            // tight NV12 w*h*3/2
    char name[128] = "";
    uint64_t frames_out = 0, inputs = 0;

    bool open(int width, int height, char* err, size_t errn) {
        w = width; h = height;
        vpx_codec_dec_cfg_t cfg{};
        cfg.threads = 4; cfg.w = (unsigned)w; cfg.h = (unsigned)h;
        vpx_codec_iface_t* iface = vpx_codec_vp9_dx();
        if (vpx_codec_dec_init(&ctx, iface, &cfg, 0) != VPX_CODEC_OK) {
            snprintf(err, errn, "vpx_codec_dec_init: %s", vpx_codec_error(&ctx));
            return false;
        }
        ok = true;
        snprintf(name, sizeof name, "libvpx %s (software, %u threads)", vpx_codec_iface_name(iface), cfg.threads);
        frame.resize((size_t)w * h * 3 / 2);
        return true;
    }
    // copy one decoded image into `frame` (tight NV12)
    void copy_out(const vpx_image_t* im) {
        int cw = im->d_w < (unsigned)w ? (int)im->d_w : w, ch = im->d_h < (unsigned)h ? (int)im->d_h : h;
        uint8_t* dy = frame.data(); uint8_t* duv = dy + (size_t)w * h;
        for (int r = 0; r < ch; r++) memcpy(dy + (size_t)r * w, im->planes[0] + (size_t)r * im->stride[0], cw);
        for (int r = 0; r < ch / 2; r++) {
            const uint8_t* u = im->planes[1] + (size_t)r * im->stride[1]; const uint8_t* v = im->planes[2] + (size_t)r * im->stride[2];
            uint8_t* o = duv + (size_t)r * w;
            for (int x = 0; x < cw / 2; x++) { o[2 * x] = u[x]; o[2 * x + 1] = v[x]; }
        }
    }
    template <class F> bool pull(int64_t pts_us, F&& cb) {
        vpx_codec_iter_t it = nullptr;
        while (const vpx_image_t* im = vpx_codec_get_frame(&ctx, &it)) {
            if (im->fmt != VPX_IMG_FMT_I420) { printf("[movie] vp9: unsupported image format %d (only 8-bit 4:2:0)\n", (int)im->fmt); return false; }
            copy_out(im);
            frames_out++;
            cb(frame.data(), pts_us);
        }
        return true;
    }
    template <class F> bool decode(const uint8_t* p, uint32_t n, int64_t pts_us, F&& cb) {
        if (!ok) return false;
        inputs++;
        if (vpx_codec_decode(&ctx, p, n, nullptr, 0) != VPX_CODEC_OK) {
            printf("[movie] vp9: decode failed at input %llu (%u bytes): %s\n", (unsigned long long)inputs, n, vpx_codec_error(&ctx));
            return false;
        }
        return pull(pts_us, cb);
    }
    template <class F> void drain(F&& cb) {
        if (!ok) return;
        vpx_codec_decode(&ctx, nullptr, 0, nullptr, 0);        // end of stream: flush the decoder
        pull(0, cb);
    }
    void flush() { if (ok) vpx_codec_decode(&ctx, nullptr, 0, nullptr, 0); }
    void close() { if (ok) { vpx_codec_destroy(&ctx); ok = false; } }
    ~Vp9Dec() { close(); }
};
#endif

}  // namespace movie_host

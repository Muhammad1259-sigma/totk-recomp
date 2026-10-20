// HLE: nn::codec hardware Opus decoder -> libopus 1.5.2 (third_party/opus, BSD).
// Packet framing (research/sweep/w18_nn_audio_api.md): 8-byte header {u32 BE size, u32 BE final range} + Opus data;
// the consumed size includes the header; sample counts are per channel. The game walks packet streams with
// GetOpusPacketSize (main 0x21ed11c: advances by the return value, 0 = error), so it returns header + payload.
// TotK: GetHardwareOpusDecoderWorkBufferSize(48000, 1, 0) = mono 48 kHz voices/streams.
#include "hle.h"
#include "../third_party/opus/include/opus/opus.h"   // relative: keeps the global flags (and every TU) unchanged
#include <atomic>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace {
struct Dec { OpusDecoder* d = nullptr; int rate = 48000, ch = 1; };
std::mutex g_mtx;
std::unordered_map<u64, Dec> g_dec;
std::atomic<u64> g_decoded{0}, g_errors{0};
u32 be32(const u8* p) { return (u32)p[0] << 24 | (u32)p[1] << 16 | (u32)p[2] << 8 | p[3]; }
// nn::codec::OpusResult values are unknown beyond 0 = success; any nonzero is a failure for the callers seen
constexpr u32 kOk = 0, kErrInvalid = 1, kErrShort = 6;
}  // namespace

HLE_AS(codec_GetOpusPacketSize, "_ZN2nn5codec17GetOpusPacketSizeEPKhm") {
    const u8* p = (const u8*)c->x[0]; u64 n = c->x[1];
    c->x[0] = (!p || n < 8) ? 0 : 8 + (u64)be32(p);
}
HLE_AS(codec_GetHardwareOpusDecoderWorkBufferSize, "_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii") {
    c->x[0] = 0x8000;   // host state lives in libopus; the guest buffer stays unused
}
HLE_AS(codec_InitializeHardwareOpusDecoder, "_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi") {
    int rate = (int)c->x[1], ch = (int)c->x[2], err = 0;
    OpusDecoder* d = opus_decoder_create(rate, ch, &err);
    static std::atomic<int> logged{0};
    if (logged++ < 8) printf("[opus] InitializeHardwareOpusDecoder %llx: %d Hz, %d ch, work %llx+%llx, option %d -> %s\n", (unsigned long long)c->x[0], rate, ch,
                             (unsigned long long)c->x[3], (unsigned long long)c->x[4], (int)c->x[5], d ? "ok" : opus_strerror(err));
    if (!d) { c->x[0] = kErrInvalid; return; }
    std::lock_guard<std::mutex> l(g_mtx);
    Dec& e = g_dec[c->x[0]];
    if (e.d) opus_decoder_destroy(e.d);
    e = Dec{d, rate, ch};
    c->x[0] = kOk;
}
HLE_AS(codec_FinalizeHardwareOpusDecoder, "_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE") {
    std::lock_guard<std::mutex> l(g_mtx);
    auto it = g_dec.find(c->x[0]);
    if (it != g_dec.end()) { opus_decoder_destroy(it->second.d); g_dec.erase(it); }
}
HLE_AS(codec_GetHardwareOpusDecoderSampleRate, "_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_dec.find(c->x[0]); c->x[0] = it == g_dec.end() ? 0 : it->second.rate;
}
HLE_AS(codec_GetHardwareOpusDecoderChannelCount, "_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_dec.find(c->x[0]); c->x[0] = it == g_dec.end() ? 0 : it->second.ch;
}
HLE_AS(codec_ResetHardwareOpusDecoder, "_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE") {
    std::lock_guard<std::mutex> l(g_mtx); auto it = g_dec.find(c->x[0]);
    if (it != g_dec.end()) opus_decoder_ctl(it->second.d, OPUS_RESET_STATE);
}
// (decoder, size_t* consumed, int* samples per channel, s16* out, size_t out bytes, const void* in, size_t in bytes)
HLE_AS(codec_DecodeOpusInterleavedWithHardware, "_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm") {
    u64* consumed = (u64*)c->x[1]; s32* samples = (s32*)c->x[2]; s16* out = (s16*)c->x[3];
    u64 out_bytes = c->x[4]; const u8* in = (const u8*)c->x[5]; u64 in_bytes = c->x[6];
    Dec d;
    { std::lock_guard<std::mutex> l(g_mtx); auto it = g_dec.find(c->x[0]); if (it == g_dec.end()) { c->x[0] = kErrInvalid; return; } d = it->second; }
    if (consumed) *consumed = 0;
    if (samples) *samples = 0;
    if (!in || in_bytes < 8) { g_errors++; c->x[0] = kErrShort; return; }
    u32 len = be32(in);
    if (8 + (u64)len > in_bytes) { g_errors++; c->x[0] = kErrShort; return; }
    int frames = opus_decode(d.d, in + 8, (opus_int32)len, out, (int)(out_bytes / (2 * (u64)d.ch)), 0);
    if (frames < 0) {
        if (g_errors++ < 8) printf("[opus] decode error %s (packet %u bytes, out %llu bytes)\n", opus_strerror(frames), len, (unsigned long long)out_bytes);
        c->x[0] = kErrInvalid; return;
    }
    if (consumed) *consumed = 8 + (u64)len;
    if (samples) *samples = frames;
    if (g_decoded++ == 0) printf("[opus] first packet: %u bytes -> %d samples/ch (%d ch)\n", len, frames, d.ch);
    c->x[0] = kOk;
}

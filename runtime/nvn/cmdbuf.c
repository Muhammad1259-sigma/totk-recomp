// NVN command buffers: commands are recorded as CmdRec records into a host shadow of the guest-provided command
// memory (nvn.h, CMD_SHADOW_K), so the game's own memory management and out-of-memory callbacks work as designed.
// A command handle is the guest address of a list's first record.
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "nvn.h"

#define JUMP_G cmd_gsize(sizeof(CmdRec))   // guest bytes always kept free for the chaining JUMP

static NvnCmdBuf *cb_of(uint64_t guest) { return HOST(NvnCmdBuf, guest); }

NVN(nvnCommandBufferInitialize) {
    NvnCmdBuf *cb = calloc(1, sizeof *cb);
    cb->guest = A(0);
    SET_HOST(A(0), cb);
    RET(1);
}
NVN(nvnCommandBufferFinalize) { void *p = cb_of(A(0)); SET_HOST(A(0), 0); if (p) backend_async_call(free, p); }
NVN(nvnCommandBufferSetDebugLabel) { (void)c; }

// ---- command memory shadow (see nvn.h). Game threads record, the renderer walks: one spinlock guards both maps.
static volatile char g_cmd_lock;
static void cmd_lock(void) { while (__atomic_test_and_set(&g_cmd_lock, __ATOMIC_ACQUIRE)) {} }
static void cmd_unlock(void) { __atomic_clear(&g_cmd_lock, __ATOMIC_RELEASE); }
typedef struct { uint64_t key; CmdChunk *val; } CmdMapEnt;
typedef struct { CmdMapEnt *e; size_t cap, n; } CmdMap;
static CmdMap g_chunk_by_base, g_chunk_by_handle;
static size_t cmap_slot(const CmdMap *m, uint64_t k) {
    size_t i = (size_t)((k >> 3) * 0x9E3779B97F4A7C15ull >> 20) & (m->cap - 1);
    while (m->e[i].key && m->e[i].key != k) i = (i + 1) & (m->cap - 1);
    return i;
}
static CmdChunk *cmap_get(const CmdMap *m, uint64_t k) {
    if (!m->cap) return NULL;
    size_t i = cmap_slot(m, k);
    return m->e[i].key == k ? m->e[i].val : NULL;
}
static void cmap_put(CmdMap *m, uint64_t k, CmdChunk *v) {
    if ((m->n + 1) * 2 > m->cap) {
        CmdMap o = *m;
        m->cap = o.cap ? o.cap * 2 : 4096; m->n = 0; m->e = calloc(m->cap, sizeof *m->e);
        for (size_t i = 0; i < o.cap; i++) if (o.e[i].key) cmap_put(m, o.e[i].key, o.e[i].val);
        free(o.e);
    }
    size_t i = cmap_slot(m, k);
    if (!m->e[i].key) { m->e[i].key = k; m->n++; }
    m->e[i].val = v;
}
// Host shadow = a linear K-times mirror of guest memory, reserved per 1 GiB guest slab and committed on demand. The
// game re-adds command memory at ever-new bases inside the same blocks; a fresh calloc per base grew without bound
// (run63/64: a 32 MiB calloc failed after ~600 imports). With the mirror, reused guest memory reuses its host bytes,
// like the hardware reuses the memory itself.
#define SLAB_SHIFT 30
// A region spans n slabs: a block straddling a slab boundary gets its own multi-slab region, reused like the others
// (run93: a 4 MiB command block across 0x2548000000 fell back to a fresh 32 MiB calloc per re-add, 2526 times, OOM).
typedef struct { uint64_t slab, n; uint8_t *host; } CmdSlab;
static CmdSlab g_slabs[64];
static int g_nslabs;
// One CmdChunk per slab for cmd_chunk_for: every mirrored chunk of a slab maps guest -> host the same way (host =
// slab host + (guest - slab base) * K), so a handle or JUMP target only needs its slab - no lock, no hash lookup
// (run363: cmd_chunk_for ~4 % of exec; the maps had ~99k chunks after 2 min). Published append-only (g_nslab_pub).
static CmdChunk g_slab_chunk[64];
static volatile long g_nslab_pub;
static volatile long g_mirror_fallback;   // a chunk got a calloc'd shadow instead: slab lookup off (map lookups only)
static uint8_t *cmd_mirror(uint64_t base, uint64_t size) {
    uint64_t lo_s = base >> SLAB_SHIFT, hi_s = (base + size - 1) >> SLAB_SHIFT;
    uint8_t *h = NULL;
    uint64_t first = lo_s;
    for (int i = 0; i < g_nslabs && !h; i++)
        if (g_slabs[i].slab <= lo_s && hi_s < g_slabs[i].slab + g_slabs[i].n) { h = g_slabs[i].host; first = g_slabs[i].slab; }
    if (!h) {
        if (g_nslabs == 64) return NULL;
        uint64_t n = hi_s - lo_s + 1;
        h = VirtualAlloc(NULL, (SIZE_T)(CMD_SHADOW_K * n) << SLAB_SHIFT, MEM_RESERVE, PAGE_READWRITE);
        if (!h) return NULL;
        g_slab_chunk[g_nslabs] = (CmdChunk){lo_s << SLAB_SHIFT, n << SLAB_SHIFT, h};
        g_slabs[g_nslabs++] = (CmdSlab){lo_s, n, h};
        InterlockedExchange(&g_nslab_pub, g_nslabs);
        LOG("[nvn] command shadow slab %llx x%llu -> host %p\n", (unsigned long long)(lo_s << SLAB_SHIFT), (unsigned long long)n, (void *)h);
    }
    uint8_t *p = h + (base - (first << SLAB_SHIFT)) * CMD_SHADOW_K;
    uint64_t lo = (uint64_t)p & ~0xFFFull, hi = ((uint64_t)p + size * CMD_SHADOW_K + 0xFFF) & ~0xFFFull;
    return VirtualAlloc((void *)lo, (SIZE_T)(hi - lo), MEM_COMMIT, PAGE_READWRITE) ? p : NULL;
}
static SRWLOCK g_mirror_mu = SRWLOCK_INIT;   // cmd_mirror's slab table (kept outside the map's spinlock)
static CmdChunk *cmd_chunk_add(uint64_t base, uint64_t size) {
    // The shadow (slab reserve, page commit = syscalls) is made OUTSIDE the map's spinlock: the render thread's
    // cmd_chunk_for spun on it while a recording thread committed pages (run361 hostprof: cmd_chunk_for 11 % of exec).
    cmd_lock();
    CmdChunk *k = cmap_get(&g_chunk_by_base, base);
    cmd_unlock();
    if (k && k->size >= size) return k;
    AcquireSRWLockExclusive(&g_mirror_mu);
    uint8_t *host = cmd_mirror(base, size);
    ReleaseSRWLockExclusive(&g_mirror_mu);
    cmd_lock();
    k = cmap_get(&g_chunk_by_base, base);
    if (!k || k->size < size) {     // a replaced chunk record is kept: lists recorded in it stay valid until overwritten
        k = malloc(sizeof *k);
        k->base = base; k->size = size; k->host = host;
        if (!k->host) { InterlockedExchange(&g_mirror_fallback, 1); k->host = calloc(size, CMD_SHADOW_K); LOG("[nvn] command shadow %llx +%llx: no mirror, calloc\n",
                                                                   (unsigned long long)base, (unsigned long long)size); }
        static uint64_t total, count, next_report = 256ull << 20;
        if (!k->host) FATAL("nvn: no host memory for a %llu-byte command shadow (%llu shadows, %llu MiB so far)\n",
                            (unsigned long long)size * CMD_SHADOW_K, (unsigned long long)count, (unsigned long long)(total >> 20));
        cmap_put(&g_chunk_by_base, base, k);
        total += size * CMD_SHADOW_K; count++;
        static int logged; if (logged++ < 16) LOG("[nvn] command shadow %llx +%llx (host total %llu KiB)\n",
                                                 (unsigned long long)base, (unsigned long long)size, (unsigned long long)(total >> 10));
        if (total >= next_report) { next_report += 256ull << 20;
            LOG("[nvn] command shadows: %llu, %llu MiB host (latest %llx +%llx)\n", (unsigned long long)count,
                (unsigned long long)(total >> 20), (unsigned long long)base, (unsigned long long)size); }
    }
    cmd_unlock();
    return k;
}
const CmdChunk *cmd_chunk_for(uint64_t guest) {
    if (!g_mirror_fallback) {
        long n = InterlockedCompareExchange(&g_nslab_pub, 0, 0);
        for (long i = 0; i < n; i++)
            if (guest - g_slab_chunk[i].base < g_slab_chunk[i].size) return &g_slab_chunk[i];
    }
    cmd_lock();
    CmdChunk *k = cmap_get(&g_chunk_by_handle, guest);
    if (!k) k = cmap_get(&g_chunk_by_base, guest);
    cmd_unlock();
    return k;
}

static CmdRec *raw_rec(const CmdChunk *k, uint64_t at, uint32_t op, uint32_t size) {
    CmdRec *r = cmd_at(k, at);
    memset(r, 0, sizeof *r);
    r->op = op;
    r->size = size;
    return r;
}

NVN(nvnCommandBufferAddCommandMemory) {
    NvnCmdBuf *cb = cb_of(A(0));
    uint64_t mem = HOST(NvnPool, A(1))->storage + A(2), size = A(3);
    static int logged;
    if (logged++ < 24) LOG("[nvn] AddCommandMemory cb %llx off %llx size %llx (prev used %llx/%llx) lr=%llx\n", (unsigned long long)A(0),
                           (unsigned long long)A(2), (unsigned long long)size, (unsigned long long)cb->used,
                           (unsigned long long)cb->size, (unsigned long long)c->x[30]);
    CmdChunk *k = cmd_chunk_add(mem, size);
    if (cb->recording && cb->mem) {
        // chain the list being recorded into the new chunk
        CmdRec *j = raw_rec(cb->chunk, cb->mem + cb->used, CMD_JUMP, sizeof(CmdRec));
        j->x[0] = mem;
    }
    cb->mem = mem;
    cb->size = size;
    cb->used = 0;
    cb->chunk = k;
}
NVN(nvnCommandBufferAddControlMemory) { (void)c; }
NVN(nvnCommandBufferGetCommandMemorySize) { RET(cb_of(A(0))->size); }
NVN(nvnCommandBufferGetCommandMemoryUsed) { RET(cb_of(A(0))->used); }
NVN(nvnCommandBufferGetCommandMemoryFree) {
    NvnCmdBuf *cb = cb_of(A(0));
    RET(cb->size > cb->used + JUMP_G ? cb->size - cb->used - JUMP_G : 0);
}
NVN(nvnCommandBufferGetControlMemorySize) { RET(0x10000); }
NVN(nvnCommandBufferGetControlMemoryUsed) { RET(0); }
NVN(nvnCommandBufferGetControlMemoryFree) { RET(0x10000); }
NVN(nvnCommandBufferSetMemoryCallback) { cb_of(A(0))->callback = A(1); }
NVN(nvnCommandBufferGetMemoryCallback) { RET(cb_of(A(0))->callback); }
NVN(nvnCommandBufferSetMemoryCallbackData) { cb_of(A(0))->callback_data = A(1); }
NVN(nvnCommandBufferGetMemoryCallbackData) { RET(cb_of(A(0))->callback_data); }
NVN(nvnCommandBufferSetCommandMemoryCallbackEnabled) { (void)c; }

// Reserve a record. May call back into the guest for more memory, so callers snapshot their
// arguments first.
static CmdRec *rec(Cpu *c, NvnCmdBuf *cb, uint32_t op, uint32_t extra) {
    uint32_t size = (uint32_t)((sizeof(CmdRec) + extra + 7) & ~7u), g = cmd_gsize(size);
    for (int tries = 0; !cb->mem || cb->used + g + JUMP_G > cb->size; tries++) {
        if (!cb->callback || tries == 2)
            FATAL("nvn: out of command memory (need %u guest B for a %u B record; cb %llx callback %llx tries %d mem %llx used %llu size %llu)\n",
                  g + JUMP_G, size, (unsigned long long)cb->guest, (unsigned long long)cb->callback, tries,
                  (unsigned long long)cb->mem, (unsigned long long)cb->used, (unsigned long long)cb->size);
        uint64_t saved[8];
        memcpy(saved, c->x, sizeof saved);
        c->x[0] = cb->guest;
        c->x[1] = 0;  // NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY
        c->x[2] = g + JUMP_G;
        c->x[3] = cb->callback_data;
        guest_call(c, cb->callback);
        memcpy(c->x, saved, sizeof saved);
    }
    CmdRec *r = raw_rec(cb->chunk, cb->mem + cb->used, op, size);
    cb->used += g;
    return r;
}

NVN(nvnCommandBufferBeginRecording) {
    NvnCmdBuf *cb = cb_of(A(0));
    cb->recording = true;
    CmdRec *r = rec(c, cb, CMD_BARRIER, 0);  // makes sure memory exists; the list starts here
    cb->start = cb->mem + cb->used - cmd_gsize(r->size);
    cb->start_chunk = cb->chunk;
}
NVN(nvnCommandBufferEndRecording) {
    NvnCmdBuf *cb = cb_of(A(0));
    rec(c, cb, CMD_END, 0);
    cb->recording = false;
    cmd_lock();
    cmap_put(&g_chunk_by_handle, cb->start, cb->start_chunk);
    cmd_unlock();
    RET(cb->start);
}
NVN(nvnCommandBufferIsRecording) { RET(cb_of(A(0))->recording); }

// ---- generic recorders
typedef struct { uint64_t x[8]; float f[4]; } Snap;
static Snap snap(Cpu *c) {
    Snap s;
    memcpy(s.x, c->x, sizeof s.x);
    for (int i = 0; i < 4; i++) s.f[i] = *(float *)&c->v[i];
    return s;
}
static CmdRec *record(Cpu *c, uint32_t op, uint32_t extra, Snap *s) {
    CmdRec *r = rec(c, cb_of(s->x[0]), op, extra);
    memcpy(r->x, s->x + 1, sizeof r->x);
    memcpy(r->f, s->f, sizeof r->f);
    return r;
}
// record with a copy of `len` bytes from the guest pointer in argument `arg`
static CmdRec *record_copy(Cpu *c, uint32_t op, int arg, uint32_t len) {
    Snap s = snap(c);
    CmdRec *r = record(c, op, len, &s);
    if (s.x[arg] && len) memcpy(r->extra, (void *)(uintptr_t)s.x[arg], len);
    return r;
}
#define PLAIN(name, op) NVN(name) { Snap s = snap(c); record(c, op, 0, &s); }
// (no-op entry points: the first call is logged, so a game that relies on one shows up in the log)
#define STUB(name) NVN(name) { static int told; if (!told++) LOG("[nvn] %s called: stub, not executed\n", #name); (void)c; }

// ---- state binding
NVN(nvnCommandBufferBindProgram) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_PROGRAM, 0, &s);
    r->x[0] = (uint64_t)(uintptr_t)HOST(NvnProgram, s.x[1]);  // host program
}
NVN(nvnCommandBufferBindBlendState) { record_copy(c, CMD_BIND_BLEND, 1, 8); }
NVN(nvnCommandBufferBindChannelMaskState) { record_copy(c, CMD_BIND_CHANNEL_MASK, 1, 4); }
NVN(nvnCommandBufferBindColorState) { record_copy(c, CMD_BIND_COLOR, 1, 4); }
NVN(nvnCommandBufferBindDepthStencilState) { record_copy(c, CMD_BIND_DEPTH_STENCIL, 1, 8); }
NVN(nvnCommandBufferBindMultisampleState) { record_copy(c, CMD_BIND_MULTISAMPLE, 1, 0x18); }
NVN(nvnCommandBufferBindPolygonState) { record_copy(c, CMD_BIND_POLYGON, 1, 4); }
NVN(nvnCommandBufferBindVertexAttribState) { record_copy(c, CMD_BIND_VERTEX_ATTRIB, 2, (uint32_t)A(1) * 4); }
NVN(nvnCommandBufferBindVertexStreamState) { record_copy(c, CMD_BIND_VERTEX_STREAM, 2, (uint32_t)A(1) * 8); }

// (cmd, first, count, const NVNbufferRange*) and the single-binding forms
NVN(nvnCommandBufferBindVertexBuffers) { record_copy(c, CMD_BIND_VERTEX_BUFFERS, 3, (uint32_t)A(2) * 16); }
NVN(nvnCommandBufferBindVertexBuffer) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_VERTEX_BUFFERS, 16, &s);
    r->x[0] = s.x[1]; r->x[1] = 1;
    memcpy(r->extra, &s.x[2], 16);
}
// (cmd, stage, first, count, const NVNbufferRange*)
NVN(nvnCommandBufferBindUniformBuffers) { record_copy(c, CMD_BIND_UNIFORM_BUFFERS, 4, (uint32_t)A(3) * 16); }
NVN(nvnCommandBufferBindUniformBuffer) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_UNIFORM_BUFFERS, 16, &s);
    r->x[0] = s.x[1]; r->x[1] = s.x[2]; r->x[2] = 1;
    memcpy(r->extra, &s.x[3], 16);
}
NVN(nvnCommandBufferBindStorageBuffers) { record_copy(c, CMD_BIND_STORAGE_BUFFERS, 4, (uint32_t)A(3) * 16); }
NVN(nvnCommandBufferBindStorageBuffer) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_STORAGE_BUFFERS, 16, &s);
    r->x[0] = s.x[1]; r->x[1] = s.x[2]; r->x[2] = 1;
    memcpy(r->extra, &s.x[3], 16);
}
// (cmd, stage, first, count, const handle*)
NVN(nvnCommandBufferBindTextures) { record_copy(c, CMD_BIND_TEXTURES, 4, (uint32_t)A(3) * 8); }
// (cmdBuf, stage, first, count, const NVNtextureHandle *): same arguments as BindTextures (call site 0x706478: x4 =
// handle pointer), but the handles are read when the list executes, so the game can fill them after recording
// (run76: 121,712 calls in 60 s were UNIMPL -> those textures were never bound).
NVN(nvnCommandBufferBindTexturesDeferred) { Snap s = snap(c); record(c, CMD_BIND_TEXTURES_DEFERRED, 0, &s); }
NVN(nvnCommandBufferBindTexture) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_TEXTURES, 8, &s);
    r->x[0] = s.x[1]; r->x[1] = s.x[2]; r->x[2] = 1;
    memcpy(r->extra, &s.x[3], 8);
}
NVN(nvnCommandBufferBindImages) { record_copy(c, CMD_BIND_IMAGES, 4, (uint32_t)A(3) * 8); }
NVN(nvnCommandBufferBindImage) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_IMAGES, 8, &s);
    r->x[0] = s.x[1]; r->x[1] = s.x[2]; r->x[2] = 1;
    memcpy(r->extra, &s.x[3], 8);
}
NVN(nvnCommandBufferBindSeparateTextures) { record_copy(c, CMD_BIND_TEXTURES, 4, (uint32_t)A(3) * 8); }
NVN(nvnCommandBufferBindSeparateTexture) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_BIND_TEXTURES, 8, &s);
    r->x[0] = s.x[1]; r->x[1] = s.x[2]; r->x[2] = 1;
    memcpy(r->extra, &s.x[3], 8);
}
STUB(nvnCommandBufferBindSeparateSamplers)
STUB(nvnCommandBufferBindSeparateSampler)
STUB(nvnCommandBufferBindTransformFeedbackBuffers)
STUB(nvnCommandBufferBindTransformFeedbackBuffer)
STUB(nvnCommandBufferBindCoverageModulationTable)

NVN(nvnCommandBufferSetTexturePool) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_TEXTURE_POOL, 0, &s);
    r->x[0] = (uint64_t)(uintptr_t)HOST(NvnDescPool, s.x[1]);
}
NVN(nvnCommandBufferSetSamplerPool) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_SAMPLER_POOL, 0, &s);
    r->x[0] = (uint64_t)(uintptr_t)HOST(NvnDescPool, s.x[1]);
}

// SetRenderTargets(cmd, n, colors[], colorViews[], depth, depthView)
// extra: { u64 tex[8]; NvnTextureView views[8]; u8 has_view[8]; u64 depth; NvnTextureView dview; u8 has_dview }
typedef struct {
    uint64_t tex[8];
    NvnTextureView views[8];
    uint8_t has_view[8];
    uint64_t depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtRec;
NVN(nvnCommandBufferSetRenderTargets) {
    Snap s = snap(c);
    int n = (int)s.x[1];
    if (n > 8) n = 8;
    RtRec rt = {0};
    const uint64_t *colors = (const uint64_t *)(uintptr_t)s.x[2], *views = (const uint64_t *)(uintptr_t)s.x[3];
    for (int i = 0; i < n; i++) {
        rt.tex[i] = colors && colors[i] ? (uint64_t)(uintptr_t)HOST(NvnTexture, colors[i]) : 0;
        if (views && views[i]) { rt.has_view[i] = 1; memcpy(&rt.views[i], (void *)(uintptr_t)views[i], sizeof(NvnTextureView)); }
    }
    rt.depth = s.x[4] ? (uint64_t)(uintptr_t)HOST(NvnTexture, s.x[4]) : 0;
    if (s.x[5]) { rt.has_dview = 1; memcpy(&rt.dview, (void *)(uintptr_t)s.x[5], sizeof(NvnTextureView)); }
    CmdRec *r = record(c, CMD_SET_RENDER_TARGETS, sizeof rt, &s);
    r->x[0] = (uint64_t)n;
    memcpy(r->extra, &rt, sizeof rt);
}

// viewports/scissors/depth ranges
NVN(nvnCommandBufferSetViewports) { record_copy(c, CMD_SET_VIEWPORTS, 3, (uint32_t)A(2) * 16); }
NVN(nvnCommandBufferSetViewport) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_VIEWPORTS, 16, &s);
    float v[4] = {(float)(int32_t)s.x[1], (float)(int32_t)s.x[2], (float)(int32_t)s.x[3], (float)(int32_t)s.x[4]};
    r->x[0] = 0; r->x[1] = 1;
    memcpy(r->extra, v, 16);
}
NVN(nvnCommandBufferSetScissors) { record_copy(c, CMD_SET_SCISSORS, 3, (uint32_t)A(2) * 16); }
NVN(nvnCommandBufferSetScissor) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_SCISSORS, 16, &s);
    int32_t v[4] = {(int32_t)s.x[1], (int32_t)s.x[2], (int32_t)s.x[3], (int32_t)s.x[4]};
    r->x[0] = 0; r->x[1] = 1;
    memcpy(r->extra, v, 16);
}
NVN(nvnCommandBufferSetDepthRanges) { record_copy(c, CMD_SET_DEPTH_RANGES, 3, (uint32_t)A(2) * 8); }
NVN(nvnCommandBufferSetDepthRange) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_DEPTH_RANGES, 8, &s);
    r->x[0] = 0; r->x[1] = 1;
    memcpy(r->extra, s.f, 8);
}
NVN(nvnCommandBufferSetBlendColor) { record_copy(c, CMD_SET_BLEND_COLOR, 1, 16); }
NVN(nvnCommandBufferSetStencilRef) { Snap s = snap(c); CmdRec *r = record(c, CMD_SET_STENCIL, 0, &s); r->x[6] = 0; }
NVN(nvnCommandBufferSetStencilMask) { Snap s = snap(c); CmdRec *r = record(c, CMD_SET_STENCIL, 0, &s); r->x[6] = 1; }
NVN(nvnCommandBufferSetStencilValueMask) { Snap s = snap(c); CmdRec *r = record(c, CMD_SET_STENCIL, 0, &s); r->x[6] = 2; }
PLAIN(nvnCommandBufferSetPrimitiveRestart, CMD_SET_PRIMITIVE_RESTART)
PLAIN(nvnCommandBufferSetAlphaRef, CMD_SET_ALPHA_REF)
PLAIN(nvnCommandBufferSetDepthBounds, CMD_SET_DEPTH_BOUNDS)
PLAIN(nvnCommandBufferSetPolygonOffsetClamp, CMD_SET_POLYGON_OFFSET)
PLAIN(nvnCommandBufferSetLineWidth, CMD_SET_LINE_WIDTH)
PLAIN(nvnCommandBufferSetPointSize, CMD_SET_POINT_SIZE)
PLAIN(nvnCommandBufferSetSampleMask, CMD_SET_SAMPLE_MASK)
// SetRenderEnable(cmd, bool): x[0] = 0/1. SetRenderEnableConditional(cmd, mode, counters): recorded as the same op with
// x[0] = 2 + mode (NVN mode 0 = render if the two counters at `counters` and `counters + 16` are equal, 1 = not equal),
// x[1] = counters address (exec.c evaluates it)
PLAIN(nvnCommandBufferSetRenderEnable, CMD_SET_RENDER_ENABLE)
NVN(nvnCommandBufferSetRenderEnableConditional) {
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_SET_RENDER_ENABLE, 0, &s);
    r->x[0] = 2 + (s.x[1] & 1);
    r->x[1] = s.x[2];
}
PLAIN(nvnCommandBufferSetDepthClamp, CMD_SET_DEPTH_CLAMP)
NVN(nvnCommandBufferSetRasterizerDiscard) {  // (not executed: logs the enables)
    static uint64_t n_on, n;
    n++;
    if (c->x[1] && (n_on++ < 4 || n_on % 1000 == 0)) LOG("[nvn] nvnCommandBufferSetRasterizerDiscard(%llu) call %llu (enabled %llu times): not executed\n", (unsigned long long)c->x[1], (unsigned long long)n, (unsigned long long)n_on);
}
STUB(nvnCommandBufferSetConservativeRasterEnable)
STUB(nvnCommandBufferSetConservativeRasterDilate)
STUB(nvnCommandBufferSetSubpixelPrecisionBias)
STUB(nvnCommandBufferSetViewportSwizzles)
STUB(nvnCommandBufferSetStencilCullCriteria)
STUB(nvnCommandBufferSetColorReductionEnable)
STUB(nvnCommandBufferSetColorReductionThresholds)
STUB(nvnCommandBufferSetTiledCacheAction)
STUB(nvnCommandBufferSetTiledCacheTileSize)
STUB(nvnCommandBufferSetShaderScratchMemory)
STUB(nvnCommandBufferSetPatchSize)
STUB(nvnCommandBufferSetInnerTessellationLevels)
STUB(nvnCommandBufferSetOuterTessellationLevels)
STUB(nvnCommandBufferSetProgramSubroutines)
STUB(nvnCommandBufferSaveZCullData)
STUB(nvnCommandBufferRestoreZCullData)

static uint64_t copy_row_stride, copy_image_stride;
RT_ON_RESET(cmdbuf_reset) { copy_row_stride = copy_image_stride = 0; }
NVN(nvnCommandBufferSetCopyRowStride) {
    Snap s = snap(c);
    copy_row_stride = s.x[1];
    CmdRec *r = record(c, CMD_SET_COPY_STRIDES, 0, &s);
    r->x[0] = copy_row_stride; r->x[1] = copy_image_stride;
}
NVN(nvnCommandBufferSetCopyImageStride) {
    Snap s = snap(c);
    copy_image_stride = s.x[1];
    CmdRec *r = record(c, CMD_SET_COPY_STRIDES, 0, &s);
    r->x[0] = copy_row_stride; r->x[1] = copy_image_stride;
}
NVN(nvnCommandBufferGetCopyRowStride) { RET(copy_row_stride); }
NVN(nvnCommandBufferGetCopyImageStride) { RET(copy_image_stride); }

// ---- clears
NVN(nvnCommandBufferClearColor) { CmdRec *r = record_copy(c, CMD_CLEAR_COLOR, 2, 16); r->x[5] = 0; }  // x[5] = kind (float)
NVN(nvnCommandBufferClearColori) { CmdRec *r = record_copy(c, CMD_CLEAR_COLOR, 2, 16); r->x[5] = 1; }
NVN(nvnCommandBufferClearColorui) { CmdRec *r = record_copy(c, CMD_CLEAR_COLOR, 2, 16); r->x[5] = 2; }
PLAIN(nvnCommandBufferClearDepthStencil, CMD_CLEAR_DEPTH_STENCIL)
// ClearTexture(cmd, texture, view, region, color, mask)
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } ClearTexRec;
static void clear_texture(Cpu *c, int kind) {
    Snap s = snap(c);
    ClearTexRec ct = {0};
    ct.tex = (uint64_t)(uintptr_t)HOST(NvnTexture, s.x[1]);
    if (s.x[2]) { ct.has_view = 1; memcpy(&ct.view, (void *)(uintptr_t)s.x[2], sizeof ct.view); }
    if (s.x[3]) memcpy(ct.region, (void *)(uintptr_t)s.x[3], sizeof ct.region);
    if (s.x[4]) memcpy(ct.color, (void *)(uintptr_t)s.x[4], 16);
    CmdRec *r = record(c, CMD_CLEAR_TEXTURE, sizeof ct, &s);
    r->x[6] = (uint64_t)kind;
    memcpy(r->extra, &ct, sizeof ct);
}
NVN(nvnCommandBufferClearTexture) { clear_texture(c, 0); }
NVN(nvnCommandBufferClearTexturei) { clear_texture(c, 1); }
NVN(nvnCommandBufferClearTextureui) { clear_texture(c, 2); }
PLAIN(nvnCommandBufferClearBuffer, CMD_CLEAR_BUFFER)
// DiscardColor(cmdbuf, index): the target's contents become undefined. On the Switch a discarded (compressed)
// target reads back cleared; areas the game then leaves undrawn (e.g. a pit after a floor tile falls away) stay
// black there. Treated as a clear to (0,0,0,0) (LARECOMP_KEEP_DISCARD=1: old no-op behaviour).
NVN(nvnCommandBufferDiscardColor) {
    static int keep = -1, logged;
    if (keep < 0) keep = getenv("LARECOMP_KEEP_DISCARD") != NULL;
    if (logged < 5) { logged++; LOG("[nvn] DiscardColor index %llu%s\n", (unsigned long long)c->x[1], keep ? " (kept)" : " -> clear"); }
    if (keep) return;
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_CLEAR_COLOR, 16, &s);
    memset(r->extra, 0, 16);
    r->x[2] = 0xF;  // all channels
    r->x[5] = 0;    // float clear
}
STUB(nvnCommandBufferDiscardDepthStencil)

// ---- draws (all integer arguments fit in x1..x7)
NVN(nvnCommandBufferDrawArrays) { Snap s = snap(c); CmdRec *r = record(c, CMD_DRAW_ARRAYS, 0, &s); r->x[3] = 0; r->x[4] = 1; }
NVN(nvnCommandBufferDrawArraysInstanced) { Snap s = snap(c); record(c, CMD_DRAW_ARRAYS, 0, &s); }
// Debug: LARECOMP_TRACE_COUNT=<n> logs the guest call chain of indexed draws with n indices.
static void trace_draw(Cpu *c, uint64_t count) {
    static long want = -2;
    if (want == -2) want = getenv("LARECOMP_TRACE_COUNT") ? atol(getenv("LARECOMP_TRACE_COUNT")) : -1;
    if (want < 0 || (long)count != want) return;
    static int n;
    if (n++ > 40) return;
    LOG("[trace] draw count %llu lr %llx chain:", (unsigned long long)count, (unsigned long long)c->x[30]);
    uint64_t fp = c->x[29];
    for (int i = 0; i < 14 && fp >= 0x7000000000ull && fp < 0x8000000000ull && (fp & 7) == 0; i++) {
        uint64_t next = *(uint64_t *)(uintptr_t)fp, lr = *(uint64_t *)(uintptr_t)(fp + 8);
        LOG(" %llx", (unsigned long long)lr);
        if (next <= fp) break;
        fp = next;
    }
    LOG("\n");
}

NVN(nvnCommandBufferDrawElements) {
    trace_draw(c, c->x[3]);
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_DRAW_ELEMENTS, 0, &s);
    r->x[4] = 0; r->x[5] = 0; r->x[6] = 1;  // base vertex, base instance, instances
}
NVN(nvnCommandBufferDrawElementsBaseVertex) {
    trace_draw(c, c->x[3]);
    Snap s = snap(c);
    CmdRec *r = record(c, CMD_DRAW_ELEMENTS, 0, &s);
    r->x[5] = 0; r->x[6] = 1;
}
NVN(nvnCommandBufferDrawElementsInstanced) { trace_draw(c, c->x[3]); Snap s = snap(c); record(c, CMD_DRAW_ELEMENTS, 0, &s); }
PLAIN(nvnCommandBufferDrawArraysIndirect, CMD_DRAW_ARRAYS_INDIRECT)
PLAIN(nvnCommandBufferDrawElementsIndirect, CMD_DRAW_ELEMENTS_INDIRECT)
STUB(nvnCommandBufferMultiDrawArraysIndirectCount)
STUB(nvnCommandBufferMultiDrawElementsIndirectCount)
STUB(nvnCommandBufferDrawTransformFeedback)
STUB(nvnCommandBufferDrawTexture)
PLAIN(nvnCommandBufferDispatchCompute, CMD_DISPATCH)
PLAIN(nvnCommandBufferDispatchComputeIndirect, CMD_DISPATCH_INDIRECT)
STUB(nvnCommandBufferBeginTransformFeedback)
STUB(nvnCommandBufferEndTransformFeedback)
STUB(nvnCommandBufferPauseTransformFeedback)
STUB(nvnCommandBufferResumeTransformFeedback)

// ---- copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRef;
static void tex_ref(TexRef *t, uint64_t tex, uint64_t view, uint64_t region) {
    t->tex = tex ? (uint64_t)(uintptr_t)HOST(NvnTexture, tex) : 0;
    if (view) { t->has_view = 1; memcpy(&t->view, (void *)(uintptr_t)view, sizeof t->view); }
    if (region) memcpy(t->region, (void *)(uintptr_t)region, sizeof t->region);
}
NVN(nvnCommandBufferCopyBufferToTexture) {  // (cmd, src, dst, dstView, dstRegion, flags)
    Snap s = snap(c);
    TexRef t = {0};
    tex_ref(&t, s.x[2], s.x[3], s.x[4]);
    CmdRec *r = record(c, CMD_COPY_BUFFER_TO_TEXTURE, sizeof t, &s);
    memcpy(r->extra, &t, sizeof t);
}
NVN(nvnCommandBufferCopyTextureToBuffer) {  // (cmd, src, srcView, srcRegion, dst, flags)
    Snap s = snap(c);
    TexRef t = {0};
    tex_ref(&t, s.x[1], s.x[2], s.x[3]);
    CmdRec *r = record(c, CMD_COPY_TEXTURE_TO_BUFFER, sizeof t, &s);
    memcpy(r->extra, &t, sizeof t);
}
NVN(nvnCommandBufferCopyTextureToTexture) {  // (cmd, src, srcView, srcRegion, dst, dstView, dstRegion, flags)
    Snap s = snap(c);
    TexRef t[2] = {{0}};
    tex_ref(&t[0], s.x[1], s.x[2], s.x[3]);
    tex_ref(&t[1], s.x[4], s.x[5], s.x[6]);
    CmdRec *r = record(c, CMD_COPY_TEXTURE_TO_TEXTURE, sizeof t, &s);
    memcpy(r->extra, t, sizeof t);
}
PLAIN(nvnCommandBufferCopyBufferToBuffer, CMD_COPY_BUFFER_TO_BUFFER)
NVN(nvnCommandBufferDownsample) {
    Snap s = snap(c);
    TexRef t[2] = {{0}};
    tex_ref(&t[0], s.x[1], 0, 0);
    tex_ref(&t[1], s.x[2], 0, 0);
    CmdRec *r = record(c, CMD_DOWNSAMPLE, sizeof t, &s);
    memcpy(r->extra, t, sizeof t);
}
NVN(nvnCommandBufferDownsampleTextureView) {
    Snap s = snap(c);
    TexRef t[2] = {{0}};
    tex_ref(&t[0], s.x[1], s.x[2], 0);
    tex_ref(&t[1], s.x[3], s.x[4], 0);
    CmdRec *r = record(c, CMD_DOWNSAMPLE, sizeof t, &s);
    memcpy(r->extra, t, sizeof t);
}
STUB(nvnCommandBufferTiledDownsample)
STUB(nvnCommandBufferTiledDownsampleTextureView)
STUB(nvnCommandBufferResolveDepthBuffer)
// UpdateUniformBuffer(cmd, buffer, size, offset, dataSize, data)
NVN(nvnCommandBufferUpdateUniformBuffer) { record_copy(c, CMD_UPDATE_UNIFORM_BUFFER, 5, (uint32_t)A(4)); }

// ---- synchronization and misc
NVN(nvnCommandBufferFenceSync) {
    // work executes at submit time, so the sync is satisfied once this list has run
    Snap s = snap(c);
    record(c, CMD_FENCE_SYNC, 0, &s);
}
PLAIN(nvnCommandBufferWaitSync, CMD_WAIT_SYNC)
PLAIN(nvnCommandBufferBarrier, CMD_BARRIER)
PLAIN(nvnCommandBufferSignalEvent, CMD_SIGNAL_EVENT)
PLAIN(nvnCommandBufferWaitEvent, CMD_WAIT_EVENT)
PLAIN(nvnCommandBufferReportCounter, CMD_REPORT_COUNTER)
STUB(nvnCommandBufferReportValue)
STUB(nvnCommandBufferResetCounter)
NVN(nvnCommandBufferCallCommands) { record_copy(c, CMD_CALL, 2, (uint32_t)A(1) * 8); }
NVN(nvnCommandBufferCopyCommands) { record_copy(c, CMD_CALL, 2, (uint32_t)A(1) * 8); }
NVN(nvnCommandBufferInsertDebugMarker) { (void)c; }
NVN(nvnCommandBufferInsertDebugMarkerDynamic) { (void)c; }
NVN(nvnCommandBufferInsertDebugMarkerStatic) { (void)c; }
NVN(nvnCommandBufferPushDebugGroup) { (void)c; }
NVN(nvnCommandBufferPushDebugGroupDynamic) { (void)c; }
NVN(nvnCommandBufferPushDebugGroupStatic) { (void)c; }
NVN(nvnCommandBufferPopDebugGroup) { (void)c; }
NVN(nvnCommandBufferPopDebugGroupId) { (void)c; }

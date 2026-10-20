// NVN implemented on the host. Guest NVN objects hold a pointer to a host object at offset 0;
// builders and small state objects store their fields directly in guest memory.
#pragma once
#include "glue.h"  // TotK: was ../hle.h
#include "tegra.h"

#define NVN(name) HLE_FUNC(#name, hle_##name)

// argument helpers
#define A(i) (c->x[(i)])
#define AP(i) ((void *)(uintptr_t)c->x[(i)])
#define AF(i) (*(float *)&c->v[(i)])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define RETF(f) do { c->v[0] = 0; *(float *)&c->v[0] = (f); } while (0)
#define HOST(T, guest) ((T *)(uintptr_t)(*(uint64_t *)(uintptr_t)(guest)))
#define SET_HOST(guest, p) (*(uint64_t *)(uintptr_t)(guest) = (uint64_t)(uintptr_t)(p))
#define G(T, base, off) (*(T *)((uint8_t *)(uintptr_t)(base) + (off)))

typedef struct NvnPool {
    uint64_t storage, size;
    uint32_t flags;
} NvnPool;

typedef struct NvnTexture {
    TexLayout layout;
    uint32_t target, format, flags, width, height, depth, levels, samples, ds_mode;
    uint8_t swizzle[4];
    uint64_t stride;
    uint64_t addr;  // guest address of texel storage
    NvnPool *pool;
    uint64_t pool_offset;
    void *backend;  // renderer-side texture
    uint64_t debug_id;
    char label[64];
    int pool_refs;  // texture-pool entries naming this object (objects.c register_texture)
    bool zombie;    // finalized by the game while pool entries still named it: kept until the last entry is re-registered
    bool zombie_gpu;  // (zombie) its image was GPU-rendered: only another object's live image at the same storage can stand in
    bool fin_queued, release_queued;  // (async execute) its finalize / zombie release is queued for the render thread
} NvnTexture;

typedef struct {
    uint32_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float lod_min, lod_max, lod_bias, max_aniso;
    float border[4];
    void *backend;
} NvnSamplerDesc;

typedef struct {
    uint8_t setmask;   // which fields are overridden (bits 0-5)
    uint8_t base_level, num_levels;
    uint8_t target : 7, ds_mode : 1;
    uint16_t min_layer, num_layers;
    uint32_t format;
    uint8_t swizzle[4];
} NvnTextureView;  // our packing of the guest NVNtextureView. TotK's is at most 0x10 bytes: f_822ae0 keeps a live
                   // object at view+0x10 (LASRecomp's 0x28 layout zeroed its data pointer -> null store at 0x823414)
_Static_assert(sizeof(NvnTextureView) == 0x10, "guest NVNtextureView is 0x10 bytes");

typedef struct {
    NvnTexture *tex;
    NvnTextureView view;
    bool has_view;
    // (debug: what was last registered here and when; "[draw] texN: none" lines print it)
    uint32_t dbg_w, dbg_h, dbg_fmt, dbg_regs;
    uint64_t dbg_addr, dbg_reg_frame, dbg_fin_frame;  // frames + 1 (0 = never)
    char dbg_label[24];
} NvnTexEntry;

typedef struct NvnDescPool {
    uint64_t addr;
    int count;
    void *entries;  // NvnTexEntry[] or NvnSamplerDesc[]
} NvnDescPool;

// (debug) a texture-pool entry nvnTextureFinalize emptied (objects.c ring of 1024; g_tex_fin_n = total)
typedef struct { int pool; uint32_t tic, w, h, fmt; uint64_t addr, frame; char label[40]; } NvnFinRec;
extern NvnFinRec g_tex_fin[1024];
extern unsigned g_tex_fin_n;
uint64_t renderer_present_count(void);

typedef struct NvnShaderStage {
    uint32_t stage;           // NVNshaderStage
    uint64_t code_addr;       // GPU (== CPU) address of the code
    const uint8_t *control;   // control section
    uint32_t const_offset, const_size;  // compiler constants (bound as c1), within the code blob
    void *backend;
} NvnShaderStage;

typedef struct NvnProgram {
    NvnShaderStage stages[6];
    uint32_t stage_mask;
    void *backend;
} NvnProgram;

typedef struct NvnWindow {
    uint64_t textures[8];  // guest NVNtexture*
    int num, active, interval, next;
    int crop[4];
} NvnWindow;

typedef struct NvnSync {
    volatile int signaled;
    volatile uint64_t seq;  // GPU submission this fence belongs to (renderer_fence_seq), 0 = none
    volatile uint64_t ticket;  // (async execute) queue ticket of its fence op: seq is valid once the queue has passed it
    volatile uint64_t prev_seq;  // LARECOMP_SYNC_LAG=1: seq of this sync's previous fence (objects.c sync_wait_body)
    volatile uint64_t rt_seq;    // (async) seq of its newest fence op the render thread has passed (prev_seq source)
} NvnSync;

// ---- command recording
// TotK sizes command memory for real NVN method encodings (chunks as small as 0x210 B), several times smaller than a
// CmdRec, so records live in a host SHADOW of each guest chunk, CMD_SHADOW_K times its size: a record of h bytes
// takes cmd_gsize(h) bytes of guest command memory (what the game sees as used/free) and sits at
// shadow + (guest offset) * K. Guest addresses stay the currency: a list handle is the guest address of its first
// record and a JUMP targets a chunk base; cmd_chunk_for() maps either back to its chunk. Re-adding the same guest
// chunk reuses its shadow, so overwriting command memory behaves as on hardware.
#define CMD_SHADOW_K 8
typedef struct CmdChunk { uint64_t base, size; uint8_t *host; } CmdChunk;
typedef struct {
    uint64_t mem;    // current command memory chunk (guest address)
    uint64_t size;   // chunk size
    uint64_t used;   // guest bytes used in chunk
    uint64_t start;  // first record of the list being recorded (guest address)
    uint64_t callback, callback_data;
    bool recording;
    uint64_t guest;  // the guest NVNcommandBuffer*
    CmdChunk *chunk, *start_chunk;   // shadows of `mem` and of `start`
} NvnCmdBuf;

// One recorded command: header, the call's integer and float arguments, then copied data.
typedef struct {
    uint32_t op, size;  // size includes the header, multiple of 8 (host bytes in the shadow)
    uint64_t x[7];      // x1..x7 of the call
    float f[4];         // s0..s3 of the call
    uint8_t extra[];
} CmdRec;
static inline uint32_t cmd_gsize(uint32_t hsize) { return ((hsize + CMD_SHADOW_K - 1) / CMD_SHADOW_K + 7) & ~7u; }
static inline CmdRec *cmd_at(const CmdChunk *k, uint64_t p) { return (CmdRec *)(k->host + (p - k->base) * CMD_SHADOW_K); }
const CmdChunk *cmd_chunk_for(uint64_t guest);   // chunk of a list handle or JUMP target; NULL if unknown

enum {
    CMD_END = 0, CMD_JUMP, CMD_CALL,
    CMD_BIND_PROGRAM, CMD_BIND_BLEND, CMD_BIND_CHANNEL_MASK, CMD_BIND_COLOR, CMD_BIND_DEPTH_STENCIL,
    CMD_BIND_MULTISAMPLE, CMD_BIND_POLYGON, CMD_BIND_VERTEX_ATTRIB, CMD_BIND_VERTEX_STREAM,
    CMD_BIND_VERTEX_BUFFERS, CMD_BIND_UNIFORM_BUFFERS, CMD_BIND_STORAGE_BUFFERS, CMD_BIND_TEXTURES,
    CMD_BIND_IMAGES, CMD_SET_TEXTURE_POOL, CMD_SET_SAMPLER_POOL, CMD_SET_RENDER_TARGETS,
    CMD_SET_VIEWPORTS, CMD_SET_SCISSORS, CMD_SET_DEPTH_RANGES, CMD_SET_BLEND_COLOR, CMD_SET_STENCIL,
    CMD_SET_PRIMITIVE_RESTART, CMD_SET_ALPHA_REF, CMD_SET_DEPTH_BOUNDS, CMD_SET_POLYGON_OFFSET,
    CMD_SET_LINE_WIDTH, CMD_SET_POINT_SIZE, CMD_SET_SAMPLE_MASK, CMD_SET_RENDER_ENABLE,
    CMD_CLEAR_COLOR, CMD_CLEAR_DEPTH_STENCIL, CMD_CLEAR_TEXTURE, CMD_CLEAR_BUFFER,
    CMD_DRAW_ARRAYS, CMD_DRAW_ELEMENTS, CMD_DRAW_ARRAYS_INDIRECT, CMD_DRAW_ELEMENTS_INDIRECT,
    CMD_DISPATCH, CMD_DISPATCH_INDIRECT,
    CMD_COPY_BUFFER_TO_BUFFER, CMD_COPY_BUFFER_TO_TEXTURE, CMD_COPY_TEXTURE_TO_BUFFER,
    CMD_COPY_TEXTURE_TO_TEXTURE, CMD_DOWNSAMPLE, CMD_UPDATE_UNIFORM_BUFFER, CMD_FENCE_SYNC, CMD_WAIT_SYNC,
    CMD_BARRIER, CMD_SIGNAL_EVENT, CMD_WAIT_EVENT, CMD_REPORT_COUNTER, CMD_DISCARD_COLOR,
    CMD_DISCARD_DEPTH_STENCIL, CMD_SET_COPY_STRIDES, CMD_DRAW_TEXTURE, CMD_RESOLVE_DEPTH,
    CMD_BIND_TEXTURES_DEFERRED,   // TotK: x[0] stage, x[1] first, x[2] count, x[3] guest handle array read at exec time
    CMD_SET_DEPTH_CLAMP,          // x[0] enable
    CMD_COUNT
};



// renderer backend (runtime/nvn/metal.m)
void backend_init(void);
void backend_execute(uint64_t handle);            // run a recorded command list
// Held while a command list executes. Finalizers that free objects execution reads (textures, programs, samplers,
// pools, syncs) take it too: the game frees textures on a loader thread while its render thread submits
// (user crash 2026-10-15: texture_info re-read a pool entry another thread had just cleared -> NULL + 0x34).
void backend_exec_lock(void);
void backend_exec_unlock(void);
void backend_present(NvnTexture *tex);            // show a texture in the window
// Async execute (exec.c, LARECOMP_ASYNC_EXEC=1): submits, fences, presents and the frees of objects that queued command
// lists name run in order on one render thread; the game's threads only queue them.
bool backend_async_on(void);
void backend_async_call(void (*fn)(void *), void *arg);  // fn(arg) after all queued work, on the render thread under the exec
                                                          // lock (synchronous mode: now, under the lock)
void backend_async_fence(NvnSync *s);
void backend_async_present(NvnTexture *tex);
void backend_async_wait(uint64_t ticket);       // until the queue has run up to ticket (0: everything queued so far)
void backend_texture_destroyed(NvnTexture *tex);
void backend_pool_written(uint64_t addr, uint64_t size);

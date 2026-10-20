// NVN objects, builders, and state objects (everything except command recording).
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glue.h"  // TotK: was ../kernel.h
#include "nvn.h"

void renderer_sync(void);

static uint64_t next_debug_id = 1;

// ================================================================== device
// NVNdeviceInfo -> value
static uint32_t device_integer(uint32_t which) {
    static const uint32_t v[] = {
        [0] = 55, [1] = 15,            // API version (TotK: requires major == 55, minor >= 15; LASRecomp had 53.207)
        [2] = 14, [3] = 65536, [4] = 256, [5] = 8, [6] = 16, [7] = 4, [8] = 16, [9] = 32,
        [10] = 16, [11] = 4, [12] = 4, [13] = 4, [14] = 16, [15] = 32, [16] = 32,
        [17] = 256, [18] = 256,        // reserved texture / sampler descriptors
        [19] = 4, [20] = 8, [21] = 0x10000, [22] = 0x1000,
        [23] = 256, [24] = 256, [25] = 4096, [26] = 0x20000, [27] = 16,
        [28] = 1536, [29] = 1024, [30] = 64, [31] = 1536,
        [32] = 0x7FFFFFFF, [33] = 65535, [34] = 65535,
        [35] = 8, [36] = 1048576, [37] = 4096, [38] = 16, [39] = 0x10000,
        [40] = 1, [41] = 1, [42] = 1,  // min/max filtering, stencil8, ASTC
        [43] = 0x40000, [44] = 15, [45] = 2048,
        [46] = 1, [47] = 5, [48] = 1, [49] = 15,
        // 50-63: values measured on a Switch (open-ead/nnheaders nvn_DeviceConstantsNX.h, "found by running
        // nvnDeviceGetInteger for every value ... within SMO"; research/sweep/w23_nvn_device_info.md). The LASRecomp
        // placeholders here (57 LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT = 0x1000) made TotK pad a 32x32 RGBA8 linear
        // texture's rows to 4096 B and write past its 8 KB pool on the first MainField load (run224-226 crash).
        [50] = 1, [51] = 8, [52] = 8, [53] = 4, [54] = 32, [55] = 128, [56] = 32, [57] = 128,
        [58] = 65536, [59] = 1, [60] = 4, [61] = 16384, [62] = 134217728, [63] = 2048, [64] = 1,
    };
    {   // TOTK_NVN_INT="57=32,4=256": override entries (A/B of single values)
        const char *e = ENV("TOTK_NVN_INT");
        for (const char *p = e; p && *p;) {
            char *q;
            unsigned long k = strtoul(p, &q, 10);
            if (*q != '=') break;
            unsigned long val = strtoul(q + 1, &q, 0);
            if (k == which) return (uint32_t)val;
            p = *q == ',' ? q + 1 : q;
            if (!*q) break;
        }
    }
    if (which >= sizeof v / sizeof *v) {     // TotK asks 66-76, 87, 90 as feature booleans (cmp #0 / cset, main 0x1283f64..)
        uint32_t r = ENV("TOTK_NVN_CAPS1") ? 1 : 0;      // experiment switch until the real values are known (U7)
        LOG("[nvn] DeviceGetInteger(%u) -> %u (beyond LASRecomp table)\n", which, r);
        return r;
    }
    return v[which];
}

NVN(nvnDeviceBuilderSetDefaults) { G(uint32_t, A(0), 0) = 0; }
NVN(nvnDeviceBuilderSetFlags) { G(uint32_t, A(0), 0) = (uint32_t)A(1); }
NVN(nvnDeviceBuilderGetFlags) { RET(G(uint32_t, A(0), 0)); }
NVN(nvnDeviceInitialize) {
    static bool once;
    if (!once) { once = true; backend_init(); }
    LOG("[nvn] device initialized\n");
    RET(1);
}
NVN(nvnDeviceFinalize) { (void)c; }
NVN(nvnDeviceSetDebugLabel) { (void)c; }
NVN(nvnDeviceGetInteger) {
    uint32_t w = (uint32_t)A(1), v = device_integer(w);
    static uint8_t seen[256];
    if (w < 256 && !seen[w]++) LOG("[nvn] DeviceGetInteger(%u) = %u (%#x) lr=%llx\n", w, v, v, (unsigned long long)c->x[30]);
    *(int32_t *)AP(2) = (int32_t)v;
}
NVN(nvnDeviceGetCurrentTimestampInNanoseconds) {
    RET(plat_now_ns());
}
NVN(nvnDeviceGetTimestampInNanoseconds) { RET(G(uint64_t, A(1), 8)); }  // counter.timestamp (ticks == ns here)
NVN(nvnDeviceSetIntermediateShaderCache) { (void)c; }
NVN(nvnDeviceGetTextureHandle) { RET((A(1) & 0xFFFFF) | ((A(2) & 0xFFF) << 20)); }
NVN(nvnDeviceGetTexelFetchHandle) { RET(A(1) & 0xFFFFF); }
NVN(nvnDeviceGetImageHandle) { RET(A(1) & 0xFFFFF); }
NVN(nvnDeviceGetSeparateTextureHandle) { RET(A(1) & 0xFFFFF); }
NVN(nvnDeviceGetSeparateSamplerHandle) { RET((A(1) & 0xFFF) << 20); }
NVN(nvnDeviceInstallDebugCallback) { (void)c; }
NVN(nvnDeviceGenerateDebugDomainId) { RET(1); }
static uint32_t window_origin, depth_mode;
RT_ON_RESET(nvn_device_reset) { window_origin = depth_mode = 0; }  // (the next boot sets its own)
NVN(nvnDeviceSetWindowOriginMode) { window_origin = (uint32_t)A(1); LOG("[nvn] window origin mode %u\n", window_origin); }
uint32_t nvn_window_origin(void) { return window_origin; }
uint32_t nvn_depth_mode(void) { return depth_mode; }
NVN(nvnDeviceGetWindowOriginMode) { RET(window_origin); }
NVN(nvnDeviceSetDepthMode) { depth_mode = (uint32_t)A(1); LOG("[nvn] depth mode %u\n", depth_mode); }
NVN(nvnDeviceGetDepthMode) { RET(depth_mode); }
NVN(nvnDeviceRegisterFastClearColor) { RET(1); }
NVN(nvnDeviceRegisterFastClearColori) { RET(1); }
NVN(nvnDeviceRegisterFastClearColorui) { RET(1); }
NVN(nvnDeviceRegisterFastClearDepth) { RET(1); }
NVN(nvnDeviceApplyDeferredFinalizes) { (void)c; }
NVN(nvnDeviceFinalizeCommandHandle) { (void)c; }
NVN(nvnDeviceIsExternalDebuggerAttached) { RET(0); }
NVN(nvnDeviceWalkDebugDatabase) { (void)c; }

// ================================================================== queue
// QueueBuilder layout: device@0 flags@8 cmd@0xC compute@0x10 control@0x14 mem@0x18 memsize@0x20
//                      flush@0x28 priority@0x2C
NVN(nvnQueueBuilderSetDevice) { G(uint64_t, A(0), 0) = A(1); }
NVN(nvnQueueBuilderSetDefaults) {
    memset(AP(0), 0, 0x30);
    G(uint32_t, A(0), 0x0C) = 0x10000;
    G(uint32_t, A(0), 0x10) = 0x40000;
    G(uint32_t, A(0), 0x14) = 0x4000;
    G(uint32_t, A(0), 0x28) = 0x8000;
    G(uint32_t, A(0), 0x2C) = 1;
}
NVN(nvnQueueBuilderSetFlags) { G(uint32_t, A(0), 8) = (uint32_t)A(1); }
NVN(nvnQueueBuilderSetCommandMemorySize) { G(uint32_t, A(0), 0x0C) = (uint32_t)A(1); }
NVN(nvnQueueBuilderSetComputeMemorySize) { G(uint32_t, A(0), 0x10) = (uint32_t)A(1); }
NVN(nvnQueueBuilderSetControlMemorySize) { G(uint32_t, A(0), 0x14) = (uint32_t)A(1); }
NVN(nvnQueueBuilderSetQueueMemory) { G(uint64_t, A(0), 0x18) = A(1); G(uint64_t, A(0), 0x20) = A(2); }
NVN(nvnQueueBuilderSetCommandFlushThreshold) { G(uint32_t, A(0), 0x28) = (uint32_t)A(1); }
NVN(nvnQueueBuilderSetQueuePriority) { G(uint32_t, A(0), 0x2C) = (uint32_t)A(1); }
NVN(nvnQueueBuilderGetDevice) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnQueueBuilderGetFlags) { RET(G(uint32_t, A(0), 8)); }
NVN(nvnQueueBuilderGetCommandMemorySize) { *(int32_t *)AP(1) = G(int32_t, A(0), 0x0C); RET(1); }
NVN(nvnQueueBuilderGetComputeMemorySize) { *(int32_t *)AP(1) = G(int32_t, A(0), 0x10); RET(1); }
NVN(nvnQueueBuilderGetControlMemorySize) { *(int32_t *)AP(1) = G(int32_t, A(0), 0x14); RET(1); }
NVN(nvnQueueBuilderGetCommandFlushThreshold) { *(int32_t *)AP(1) = G(int32_t, A(0), 0x28); RET(1); }
NVN(nvnQueueBuilderGetQueuePriority) { RET(G(uint32_t, A(0), 0x2C)); }
NVN(nvnQueueBuilderGetMemorySize) { RET(0x10000); }  // required queue memory
NVN(nvnQueueBuilderGetQueueMemorySize) { RET(G(uint64_t, A(0), 0x20)); }
NVN(nvnQueueBuilderGetMemory) { RET(G(uint64_t, A(0), 0x18)); }

NVN(nvnQueueInitialize) { RET(1); }
NVN(nvnQueueFinalize) { (void)c; }
NVN(nvnQueueSetDebugLabel) { (void)c; }
NVN(nvnQueueGetError) { RET(0); }
NVN(nvnQueueGetTotalCommandMemoryUsed) { RET(0); }
NVN(nvnQueueGetTotalControlMemoryUsed) { RET(0); }
NVN(nvnQueueGetTotalComputeMemoryUsed) { RET(0); }
NVN(nvnQueueResetMemoryUsageCounts) { (void)c; }
NVN(nvnQueueSubmitCommands) {
    int n = (int)A(1);
    const uint64_t *handles = AP(2);
    extern bool g_debug_frame;  // which queue submits what, in order (TotK menu capture ordering, 2026-10-15)
    if (g_debug_frame || getenv("LARECOMP_LOG_QUEUE")) {
        static int nq; static uint64_t qs[8];
        int qi = 0;
        while (qi < nq && qs[qi] != A(0)) qi++;
        if (qi == nq && nq < 8) qs[nq++] = A(0);
        LOG("[queue] q%d %llx submit %d command buffers (lr %llx)\n", qi, (unsigned long long)A(0), n, (unsigned long long)c->x[30]);
    }
    for (int i = 0; i < n; i++) backend_execute(handles[i]);
}
NVN(nvnQueueFlush) { (void)c; }
NVN(nvnQueueFinish) { (void)c; backend_async_wait(0); renderer_sync(); }

static NvnSync *sync_of(uint64_t guest) {
    NvnSync *s = HOST(NvnSync, guest);
    if (!s) { s = calloc(1, sizeof *s); SET_HOST(guest, s); }
    return s;
}
uint64_t renderer_fence_seq(void);
void renderer_wait_seq(uint64_t seq);
void renderer_fence_submit(void);
NVN(nvnQueueFenceSync) {
    NvnSync *s = sync_of(A(1));
    if (backend_async_on()) { backend_async_fence(s); return; }   // (seq assigned on the render thread, in order)
    renderer_fence_submit();
    s->prev_seq = s->seq;
    s->signaled = 1; s->seq = renderer_fence_seq();
}
void nvn_sync_set_seq(uint64_t guest, uint64_t seq) { NvnSync *s = sync_of(guest); s->signaled = 1; s->seq = seq; }
NVN(nvnQueueWaitSync) {
    extern bool g_debug_frame;
    if (g_debug_frame || getenv("LARECOMP_LOG_QUEUE"))
        LOG("[queue] %llx waits for sync %llx (signaled %d)\n", (unsigned long long)A(0), (unsigned long long)A(1), sync_of(A(1))->signaled);
    RET(1);
}

// ================================================================== sync / event
NVN(nvnSyncInitialize) { SET_HOST(A(0), calloc(1, sizeof(NvnSync))); RET(1); }
NVN(nvnSyncFinalize) {   // (async: a queued fence op still names it)
    NvnSync *s = HOST(NvnSync, A(0));
    SET_HOST(A(0), 0);
    if (s) backend_async_call(free, s);
}
NVN(nvnSyncSetDebugLabel) { (void)c; }
void renderer_flush(void);
void hle_evlog(char op, uint64_t ev, uint64_t lr);   // (hle_os_thread.cpp wait-chain ring: Y/y around the wait)
static void sync_wait_body(Cpu *c);
NVN(nvnSyncWait) {
    uint64_t sy = A(0), lr = c->x[30];
    hle_evlog('Y', sy, lr);
    sync_wait_body(c);
    hle_evlog('y', sy, lr);
}
static void sync_wait_body(Cpu *c) {  // NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED
    static int flush = -1;
    static uint64_t n;
    if (flush < 0) flush = getenv("LARECOMP_SYNC_FLUSH") ? 1 : 0;
    if (getenv("LARECOMP_LOG_SYNC") && n++ % 60 == 0) LOG("[nvn] SyncWait #%llu (lr %llx)\n", (unsigned long long)n, c->x[30]);
    static int whole = -1;  // LARECOMP_SYNC_ALL=1: old behaviour, wait for all submitted work
    if (whole < 0) whole = getenv("LARECOMP_SYNC_ALL") ? 1 : 0;
    // SyncWait lag (default on since 2026-10-17; LARECOMP_SYNC_LAG=0 = wait for the fence itself): wait for the render thread to
    // pass this fence (the game's command memory is then consumed) but for the GPU only up to this sync's PREVIOUS fence. TotK's
    // frame loop (main 0xa91f98) waits for the fence of the frame it just submitted: on the Deck that wait was 9.6 ms/frame when
    // cloudy (age 1.3 ms), serialising GPU and CPU. d33 (Deck): 2 cloud draws/frame 49.8 fps with lag, 1 cloud draw 35 fps
    // without; frame shots equal. One frame more of GPU latency; relies on the game's CPU-written GPU data being double-buffered.
    static int lag = -1;
    if (lag < 0) { const char *e = getenv("LARECOMP_SYNC_LAG"); lag = !(e && *e == '0'); if (!lag) LOG("[nvn] SyncWait lag off (LARECOMP_SYNC_LAG=0)\n"); }
    if (backend_async_on()) {   // the render thread assigns the fence's submission when the queue reaches it
        NvnSync *s = sync_of(A(0));
        backend_async_wait(flush || whole ? 0 : s->ticket);
        if (flush || whole) renderer_sync(); else renderer_wait_seq(lag && s->prev_seq ? s->prev_seq : s->seq);
        RET(0);
        return;
    }
    if (lag && !flush && !whole) { NvnSync *s = sync_of(A(0)); renderer_wait_seq(s->prev_seq ? s->prev_seq : s->seq); RET(0); return; }
    if (flush) renderer_flush();
    else if (whole) renderer_sync();
    else renderer_wait_seq(sync_of(A(0))->seq);  // only the submission that holds this fence
    RET(0);
}
NVN(nvnSyncCreateGLSync) { RET(0); }
NVN(nvnSyncInitializeFromFencedGLSync) { RET(0); }

// EventBuilder: pool@0 offset@8 ; Event: storage address (u32 value lives in guest memory)
NVN(nvnEventBuilderSetDefaults) { G(uint64_t, A(0), 0) = 0; G(uint64_t, A(0), 8) = 0; }
NVN(nvnEventBuilderSetStorage) { G(uint64_t, A(0), 0) = A(1); G(uint64_t, A(0), 8) = A(2); }
NVN(nvnEventBuilderGetStorage) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnEventBuilderGetMemoryPool) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnEventBuilderGetMemoryOffset) { RET(G(uint64_t, A(0), 8)); }
static uint64_t event_addr(uint64_t ev) { return G(uint64_t, ev, 0); }
NVN(nvnEventInitialize) {
    uint64_t pool = G(uint64_t, A(1), 0), off = G(uint64_t, A(1), 8);
    G(uint64_t, A(0), 0) = HOST(NvnPool, pool)->storage + off;
    G(uint64_t, A(0), 8) = pool;
    G(uint64_t, A(0), 16) = off;
    RET(1);
}
NVN(nvnEventFinalize) { (void)c; }
NVN(nvnEventGetValue) { RET(*(uint32_t *)(uintptr_t)event_addr(A(0))); }
NVN(nvnEventSignal) { *(uint32_t *)(uintptr_t)event_addr(A(0)) = (uint32_t)A(2); }
NVN(nvnEventGetMemoryPool) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnEventGetMemoryOffset) { RET(G(uint64_t, A(0), 16)); }

// ================================================================== memory pools
// Builder: device@0 flags@8 storage@0x10 size@0x18
NVN(nvnMemoryPoolBuilderSetDevice) { G(uint64_t, A(0), 0) = A(1); }
NVN(nvnMemoryPoolBuilderSetDefaults) { memset(AP(0), 0, 0x20); G(uint32_t, A(0), 8) = 0x9; }
NVN(nvnMemoryPoolBuilderSetFlags) { G(uint32_t, A(0), 8) = (uint32_t)A(1); }
NVN(nvnMemoryPoolBuilderSetStorage) { G(uint64_t, A(0), 0x10) = A(1); G(uint64_t, A(0), 0x18) = A(2); }
NVN(nvnMemoryPoolBuilderGetDevice) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnMemoryPoolBuilderGetFlags) { RET(G(uint32_t, A(0), 8)); }
NVN(nvnMemoryPoolBuilderGetMemory) { RET(G(uint64_t, A(0), 0x10)); }
NVN(nvnMemoryPoolBuilderGetSize) { RET(G(uint64_t, A(0), 0x18)); }

NVN(nvnMemoryPoolInitialize) {
    NvnPool *p = calloc(1, sizeof *p);
    p->storage = G(uint64_t, A(1), 0x10);
    p->size = G(uint64_t, A(1), 0x18);
    p->flags = G(uint32_t, A(1), 8);
    SET_HOST(A(0), p);
    LOG("[nvn] pool init %llx flags %x storage %llx size %llx\n", (unsigned long long)A(0), p->flags,
        (unsigned long long)p->storage, (unsigned long long)p->size);
    RET(1);
}
NVN(nvnMemoryPoolFinalize) { void *p = HOST(NvnPool, A(0)); SET_HOST(A(0), 0); if (p) backend_async_call(free, p); }
NVN(nvnMemoryPoolSetDebugLabel) { (void)c; }
NVN(nvnMemoryPoolMap) { RET(HOST(NvnPool, A(0))->storage); }
NVN(nvnMemoryPoolGetBufferAddress) { RET(HOST(NvnPool, A(0))->storage); }  // GPU address == CPU address
NVN(nvnMemoryPoolGetSize) { RET(HOST(NvnPool, A(0))->size); }
NVN(nvnMemoryPoolGetFlags) {
    uintptr_t h = (uintptr_t)HOST(NvnPool, A(0));
    if (h < 0x10000 || h >> 47) {
        // run224/225: the game's 3-byte stores into a CPU grid (f_a51fe0 at 0xa56c64: base + y * pitch + x * bpp, cell
        // (x, y) from a u16) ran past pool 1077812060's 0x2000-byte storage into the pool object behind it, then its
        // flush helper f_c5cca4(buf, offset, size) asked for this pool's flags. Print the grid's numbers once (frames:
        // f_c5cca4 saved f_a51fe0's x29/x20 in its frame), answer "not CPU-cached" (no flush) instead of crashing.
        static int n;
        if (n++ < 3) {
            LOG("[nvn] MemoryPoolGetFlags: pool %llx handle %llx corrupted; buf x19 %llx off x22 %llx size x20 %llx [buf] %llx\n",
                (unsigned long long)A(0), (unsigned long long)h, (unsigned long long)c->x[19], (unsigned long long)c->x[22],
                (unsigned long long)c->x[20], (unsigned long long)c->x[23]);
            uint64_t fp = c->x[29];
            for (int i = 0; i < 6 && fp > 0x10000 && fp < 0x7fffffffffffull; i++) {
                uint64_t lr = *(uint64_t *)(fp + 8);
                LOG("[nvn]   frame fp %llx lr %llx\n", (unsigned long long)fp, (unsigned long long)lr);
                if (lr == 0x7100a56c7cull) {   // fp = f_c5cca4's frame: [fp] = f_a51fe0's x29, [fp+0x30] = its x20
                    uint64_t fa = *(uint64_t *)fp, t = *(uint64_t *)(fp + 0x30), cell = *(uint64_t *)(fa - 0x148);
                    uint16_t xy = cell > 0x10000 ? *(uint16_t *)cell : 0;
                    LOG("[nvn]   grid this %llx: pitch %u bpp %u base off %u | buf %llx | cell ptr %llx = %04x (x %u y %u)\n",
                        (unsigned long long)t, *(uint32_t *)(t + 0x3458), *(uint32_t *)(t + 0x345c), *(uint32_t *)(t + 0x3120),
                        (unsigned long long)*(uint64_t *)(t + 0x3138), (unsigned long long)cell, xy, xy & 0xff, xy >> 8);
                    for (int k = 0x3100; k < 0x3470; k += 0x20)
                        LOG("[nvn]     +%x: %08x %08x %08x %08x %08x %08x %08x %08x\n", k, *(uint32_t *)(t + k), *(uint32_t *)(t + k + 4),
                            *(uint32_t *)(t + k + 8), *(uint32_t *)(t + k + 12), *(uint32_t *)(t + k + 16), *(uint32_t *)(t + k + 20),
                            *(uint32_t *)(t + k + 24), *(uint32_t *)(t + k + 28));
                }
                fp = *(uint64_t *)fp;
            }
        }
        RET(0);
        return;
    }
    RET(((NvnPool *)h)->flags);
}
NVN(nvnMemoryPoolFlushMappedRange) {
    NvnPool *p = HOST(NvnPool, A(0));
    backend_pool_written(p->storage + A(1), A(2));
}
NVN(nvnMemoryPoolInvalidateMappedRange) { (void)c; }
NVN(nvnMemoryPoolMapVirtual) {     // (virtualPool, count, requests): TotK maps physical pools into virtual ones
    static int n;
    if (n++ < 16) {
        LOG("[nvn] MapVirtual pool %llx count %d:", (unsigned long long)A(0), (int)A(1));
        for (int i = 0; i < (int)A(1) && i < 4; i++) {
            const uint64_t *r = (const uint64_t *)(uintptr_t)(A(2) + 0x28ull * i);
            LOG(" {%llx %llx %llx %llx %llx}", (unsigned long long)r[0], (unsigned long long)r[1], (unsigned long long)r[2],
                (unsigned long long)r[3], (unsigned long long)r[4]);
        }
        LOG("\n");
    }
    NvnPool *vp = HOST(NvnPool, A(0));
    for (int i = 0; vp && i < (int)A(1); i++) {
        const uint64_t *r = (const uint64_t *)(uintptr_t)(A(2) + 0x28ull * i);   // {physPool, physOff, virtOff, size, class}
        NvnPool *pp = r[0] ? HOST(NvnPool, r[0]) : NULL;
        if (!pp) continue;   // unmap request
        uint64_t base = pp->storage + r[1] - r[2];
        if (!vp->storage) vp->storage = base;
        else if (vp->storage != base) LOG("[nvn] MapVirtual: non-linear mapping into pool %llx (needs real aliasing)\n", (unsigned long long)A(0));
    }
    RET(1);
}

// ================================================================== buffers
// Builder: device@0 pool@8 offset@0x10 size@0x18. Buffer: address@0 pool@8 offset@0x10 size@0x18 id@0x20
NVN(nvnBufferBuilderSetDevice) { G(uint64_t, A(0), 0) = A(1); }
NVN(nvnBufferBuilderSetDefaults) { memset(AP(0) + 8, 0, 0x18); }
NVN(nvnBufferBuilderSetStorage) {
    G(uint64_t, A(0), 8) = A(1); G(uint64_t, A(0), 0x10) = A(2); G(uint64_t, A(0), 0x18) = A(3);
}
NVN(nvnBufferBuilderGetDevice) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnBufferBuilderGetMemoryPool) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnBufferBuilderGetMemoryOffset) { RET(G(uint64_t, A(0), 0x10)); }
NVN(nvnBufferBuilderGetSize) { RET(G(uint64_t, A(0), 0x18)); }
NVN(nvnBufferInitialize) {
    uint64_t pool = G(uint64_t, A(1), 8), off = G(uint64_t, A(1), 0x10);
    if (!pool || !HOST(NvnPool, pool))
        FATAL("[nvn] BufferInitialize: pool %llx not initialized (off %llx size %llx; pool words %llx %llx %llx %llx)\n",
              (unsigned long long)pool, (unsigned long long)off, (unsigned long long)G(uint64_t, A(1), 0x18),
              pool ? (unsigned long long)G(uint64_t, pool, 0) : 0ull, pool ? (unsigned long long)G(uint64_t, pool, 8) : 0ull,
              pool ? (unsigned long long)G(uint64_t, pool, 0x10) : 0ull, pool ? (unsigned long long)G(uint64_t, pool, 0x18) : 0ull);
    G(uint64_t, A(0), 0) = HOST(NvnPool, pool)->storage + off;
    G(uint64_t, A(0), 8) = pool;
    G(uint64_t, A(0), 0x10) = off;
    G(uint64_t, A(0), 0x18) = G(uint64_t, A(1), 0x18);
    G(uint64_t, A(0), 0x20) = next_debug_id++;
    RET(1);
}
NVN(nvnBufferFinalize) { (void)c; }
NVN(nvnBufferSetDebugLabel) { (void)c; }
NVN(nvnBufferMap) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnBufferGetAddress) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnBufferGetMemoryPool) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnBufferGetMemoryOffset) { RET(G(uint64_t, A(0), 0x10)); }
NVN(nvnBufferGetSize) { RET(G(uint64_t, A(0), 0x18)); }
NVN(nvnBufferGetDebugID) { RET(G(uint64_t, A(0), 0x20)); }
NVN(nvnBufferFlushMappedRange) { backend_pool_written(G(uint64_t, A(0), 0) + A(1), A(2)); }
NVN(nvnBufferInvalidateMappedRange) { (void)c; }

// ================================================================== textures
// TextureBuilder: device@0 flags@8 target@0xC width@0x10 height@0x14 depth@0x18 levels@0x1C
//   format@0x20 samples@0x24 swizzle@0x28 dsmode@0x2C stride@0x30 pool@0x38 offset@0x40
//   packaged@0x48 layout@0x50 glname@0x58
#define TB(T, off) G(T, A(0), off)
NVN(nvnTextureBuilderSetDevice) { TB(uint64_t, 0) = A(1); }
NVN(nvnTextureBuilderSetDefaults) {
    memset(AP(0) + 8, 0, 0x58);
    TB(uint32_t, 0xC) = 1;  // 2D
    TB(uint32_t, 0x10) = TB(uint32_t, 0x14) = TB(uint32_t, 0x18) = 1;
    TB(uint32_t, 0x1C) = 1;
    TB(uint8_t, 0x28) = 2; TB(uint8_t, 0x29) = 3; TB(uint8_t, 0x2A) = 4; TB(uint8_t, 0x2B) = 5;
}
NVN(nvnTextureBuilderSetFlags) { TB(uint32_t, 8) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetTarget) { TB(uint32_t, 0xC) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetWidth) { TB(uint32_t, 0x10) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetHeight) { TB(uint32_t, 0x14) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetDepth) { TB(uint32_t, 0x18) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetSize1D) { TB(uint32_t, 0x10) = (uint32_t)A(1); TB(uint32_t, 0x14) = 1; TB(uint32_t, 0x18) = 1; }
NVN(nvnTextureBuilderSetSize2D) { TB(uint32_t, 0x10) = (uint32_t)A(1); TB(uint32_t, 0x14) = (uint32_t)A(2); TB(uint32_t, 0x18) = 1; }
NVN(nvnTextureBuilderSetSize3D) {
    TB(uint32_t, 0x10) = (uint32_t)A(1); TB(uint32_t, 0x14) = (uint32_t)A(2); TB(uint32_t, 0x18) = (uint32_t)A(3);
}
NVN(nvnTextureBuilderSetLevels) { TB(uint32_t, 0x1C) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetFormat) { TB(uint32_t, 0x20) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetSamples) { TB(uint32_t, 0x24) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetSwizzle) {
    TB(uint8_t, 0x28) = (uint8_t)A(1); TB(uint8_t, 0x29) = (uint8_t)A(2);
    TB(uint8_t, 0x2A) = (uint8_t)A(3); TB(uint8_t, 0x2B) = (uint8_t)A(4);
}
NVN(nvnTextureBuilderSetDepthStencilMode) { TB(uint32_t, 0x2C) = (uint32_t)A(1); }
NVN(nvnTextureBuilderSetStride) { TB(uint64_t, 0x30) = A(1); }
NVN(nvnTextureBuilderSetStorage) { TB(uint64_t, 0x38) = A(1); TB(uint64_t, 0x40) = A(2); }
NVN(nvnTextureBuilderSetPackagedTextureData) { TB(uint64_t, 0x48) = A(1); }
NVN(nvnTextureBuilderSetPackagedTextureLayout) { TB(uint64_t, 0x50) = A(1) ? *(uint64_t *)AP(1) : 0; }
NVN(nvnTextureBuilderSetGLTextureName) { TB(uint32_t, 0x58) = (uint32_t)A(1); }
NVN(nvnTextureBuilderGetDevice) { RET(TB(uint64_t, 0)); }
NVN(nvnTextureBuilderGetFlags) { RET(TB(uint32_t, 8)); }
NVN(nvnTextureBuilderGetTarget) { RET(TB(uint32_t, 0xC)); }
NVN(nvnTextureBuilderGetWidth) { RET(TB(uint32_t, 0x10)); }
NVN(nvnTextureBuilderGetHeight) { RET(TB(uint32_t, 0x14)); }
NVN(nvnTextureBuilderGetDepth) { RET(TB(uint32_t, 0x18)); }
NVN(nvnTextureBuilderGetLevels) { RET(TB(uint32_t, 0x1C)); }
NVN(nvnTextureBuilderGetFormat) { RET(TB(uint32_t, 0x20)); }
NVN(nvnTextureBuilderGetSamples) { RET(TB(uint32_t, 0x24)); }
NVN(nvnTextureBuilderGetSwizzle) {
    for (int i = 0; i < 4; i++) *(uint32_t *)AP(1 + i) = TB(uint8_t, 0x28 + i);
}
NVN(nvnTextureBuilderGetDepthStencilMode) { RET(TB(uint32_t, 0x2C)); }
NVN(nvnTextureBuilderGetStride) { RET(TB(uint64_t, 0x30)); }
NVN(nvnTextureBuilderGetMemoryPool) { RET(TB(uint64_t, 0x38)); }
NVN(nvnTextureBuilderGetMemoryOffset) { RET(TB(uint64_t, 0x40)); }
NVN(nvnTextureBuilderGetPackagedTextureData) { RET(TB(uint64_t, 0x48)); }
NVN(nvnTextureBuilderGetPackagedTextureLayout) { RET(0); }
NVN(nvnTextureBuilderGetGLTextureName) { RET(TB(uint32_t, 0x58)); }
NVN(nvnTextureBuilderGetSparseTileLayout) { RET(0); }
NVN(nvnTextureBuilderGetZCullStorageSize) { RET(0); }
NVN(nvnTextureBuilderGetStorageClass) { RET(0); }
NVN(nvnTextureBuilderGetRawStorageClass) { RET(0); }

static bool target_is_array(uint32_t t) { return t == 3 || t == 4 || t == 6 || t == 8 || t == 9; }

static void builder_layout(uint64_t b, TexLayout *t) {
    uint32_t target = G(uint32_t, b, 0xC);
    t->fmt = nvn_format_info(G(uint32_t, b, 0x20));
    t->width = G(uint32_t, b, 0x10);
    t->height = target == 0 || target == 3 ? 1 : G(uint32_t, b, 0x14);
    if (target == 3) {  // 1D array: height is the layer count
        t->layers = G(uint32_t, b, 0x14);
    }
    t->depth = target == 2 ? G(uint32_t, b, 0x18) : 1;
    if (target_is_array(target) && target != 3) t->layers = G(uint32_t, b, 0x18);
    if (target == 8) t->layers = 6;
    if (!t->layers) t->layers = 1;
    t->levels = G(uint32_t, b, 0x1C) ? G(uint32_t, b, 0x1C) : 1;
    tegra_layout_init(t);
    // Packaged textures (BNTX) were swizzled offline with their own block height, which can differ
    // from the driver default: NVNpackagedTextureLayout's low word is log2(GOBs per block).
    // low word: bits 0-3 block height log2, bits 4-7 block depth log2 (3D; run293: a 64x64x64 BC4 volume has 0x40)
    uint64_t pl = G(uint64_t, b, 0x50);
    if (pl && ((uint32_t)pl & 0xF) <= 5) t->block_height_log2 = (uint32_t)pl & 0xF;
    if (pl && t->depth > 1 && (((uint32_t)pl >> 4) & 0xF) <= 5) t->block_depth_log2 = ((uint32_t)pl >> 4) & 0xF;
    if (nvn_flags_linear(G(uint32_t, b, 8)) && target != 10) {   // pitch-linear: sizes / offsets follow the stride
        t->pitch = G(uint64_t, b, 0x30);
        if (!t->pitch) t->pitch = (uint64_t)((t->width + t->fmt->bw - 1) / t->fmt->bw) * t->fmt->bpb;
    }
    if (pl && getenv("LARECOMP_LOG_PACKAGED")) {
        static int n;
        if (n++ < 4000) LOG("[nvn] packaged layout %016llx fmt %#x %ux%u lv %u -> default bh %u\n", pl, G(uint32_t, b, 0x20), t->width, t->height, t->levels, t->block_height_log2);
    }
}

static bool is_linear(uint64_t b) { return nvn_flags_linear(G(uint32_t, b, 8)) || G(uint32_t, b, 0xC) == 10; }

static uint64_t builder_storage_size(uint64_t b) {
    if (G(uint32_t, b, 0xC) == 10) {  // buffer texture
        return (uint64_t)G(uint32_t, b, 0x10) * nvn_format_info(G(uint32_t, b, 0x20))->bpb;
    }
    if (is_linear(b)) return G(uint64_t, b, 0x30) * G(uint32_t, b, 0x14);
    TexLayout t = {0};
    builder_layout(b, &t);
    return tegra_total_size(&t);
}
NVN(nvnTextureBuilderGetStorageSize) { RET(builder_storage_size(A(0))); }
NVN(nvnTextureBuilderGetStorageAlignment) { RET(is_linear(A(0)) ? 32 : 512); }

static NvnTexture *texture_from_builder(uint64_t b) {
    NvnTexture *t = calloc(1, sizeof *t);
    t->target = G(uint32_t, b, 0xC);
    t->flags = G(uint32_t, b, 8);
    t->width = G(uint32_t, b, 0x10);
    t->height = G(uint32_t, b, 0x14);
    t->depth = G(uint32_t, b, 0x18);
    t->levels = G(uint32_t, b, 0x1C) ? G(uint32_t, b, 0x1C) : 1;
    t->format = G(uint32_t, b, 0x20);
    t->samples = G(uint32_t, b, 0x24);
    memcpy(t->swizzle, (void *)(uintptr_t)(b + 0x28), 4);
    t->ds_mode = G(uint32_t, b, 0x2C);
    t->stride = G(uint64_t, b, 0x30);
    uint64_t pool = G(uint64_t, b, 0x38);
    t->pool = pool ? HOST(NvnPool, pool) : NULL;
    t->pool_offset = G(uint64_t, b, 0x40);
    t->addr = t->pool ? t->pool->storage + t->pool_offset : 0;
    builder_layout(b, &t->layout);
    t->debug_id = next_debug_id++;
    return t;
}

NVN(nvnTextureInitialize) {
    NvnTexture *t = texture_from_builder(A(1));
    if (t->flags & 2)
        LOG("[nvn] video texture %p: %ux%u fmt %#x flags %#x target %u addr %llx stride %llu\n", t, t->width, t->height,
            t->format, t->flags, t->target, t->addr, t->stride);
    {   // LARECOMP_TEX_WATCH_FMT=<fmt hex>x<w>x<h>[,...]: watch every texture of that format and size like
        // LARECOMP_TEX_WATCH does by address (addresses move between runs; 2026-10-15 terrain heightmaps 6x260x260)
        static char filt[256];
        static int init;
        if (!init) { init = 1; const char *e = getenv("LARECOMP_TEX_WATCH_FMT"); if (e) snprintf(filt, sizeof filt, "%s", e); }
        if (filt[0]) {
            char key[48];
            snprintf(key, sizeof key, "%xx%ux%u", t->format, t->width, t->height);
            for (const char *q = filt; *q;) {
                const char *c = strchr(q, ',');
                size_t n = c ? (size_t)(c - q) : strlen(q);
                if (n == strlen(key) && !strncmp(q, key, n)) {
                    extern void tex_watch_add(uint64_t addr, uint64_t size);
                    tex_watch_add(t->addr, tegra_total_size(&t->layout));
                    LOG("[texw] init %s addr %llx flags %#x target %u levels %u stride %llu bh_log2 %u bd_log2 %u layers %u size %llu label '%s'\n",
                        key, (unsigned long long)t->addr, t->flags, t->target, t->levels, (unsigned long long)t->stride,
                        t->layout.block_height_log2, t->layout.block_depth_log2, t->layout.layers,
                        (unsigned long long)tegra_total_size(&t->layout), t->label);
                    break;
                }
                q += n + (c ? 1 : 0);
            }
        }
    }
    SET_HOST(A(0), t);
    RET(1);
}
extern NvnDescPool *g_texture_pools[64];
// (debug) pool entries emptied by nvnTextureFinalize: what the slot named, and when. "[draw] texN: none" lines look the
// slot up here (2026-10-15 minimap black square: its source slots were empty at execution time)
NvnFinRec g_tex_fin[1024];
unsigned g_tex_fin_n;
// Textures finalized while texture-pool entries still name them. Real NVN copies the descriptor into the pool at
// RegisterTexture, so the slot keeps working after the object is finalized; TotK relies on that: the map / minimap
// icons (96x96, 192x192, 72x72 UI textures) are finalized at frame 1909 and still drawn through their slots at 2400
// (run320) -> "tex none" -> solid black squares at fixed map positions (user 2026-10-15). Our entries hold the object
// pointer, so the object stays (zombie, backend kept) until the last entry naming it is re-registered.
// LARECOMP_NO_TEX_ZOMBIE=1: empty the entries at finalize (the old behaviour).
static int g_zombies;
static uint64_t g_zombie_bytes;
static void zombie_release(NvnTexture *t) {  // (exec lock held)
    g_zombies--;
    g_zombie_bytes -= tegra_total_size(&t->layout);
    backend_texture_destroyed(t);
    free(t);
}
static void zombie_release_cb(void *t) { zombie_release((NvnTexture *)t); }
uint32_t g_pool_gen;   // bumped by every texture / sampler pool entry write (exec.c stage_set memo key)
void nvn_pool_entry_set(NvnTexEntry *e, NvnTexture *nt) {  // every write of NvnTexEntry.tex goes through here
    NvnTexture *old = e->tex;
    if (old == nt) return;
    __atomic_add_fetch(&g_pool_gen, 1, __ATOMIC_RELAXED);
    if (nt) __atomic_add_fetch(&nt->pool_refs, 1, __ATOMIC_RELAXED);
    e->tex = nt;
    // (async: this runs on a game thread while the render thread may be in texture_finalize_locked for `old`. Both
    // sides store their half (refs / zombie) seq-cst before reading the other's, so at least one sees both; the
    // exchange on release_queued lets exactly one release it. run383: c0000374 heap corruption on the render thread
    // with plain reads = both released it.)
    if (old && __atomic_sub_fetch(&old->pool_refs, 1, __ATOMIC_SEQ_CST) <= 0 && __atomic_load_n(&old->zombie, __ATOMIC_SEQ_CST) &&
        !__atomic_exchange_n(&old->release_queued, true, __ATOMIC_SEQ_CST))
        backend_async_call(zombie_release_cb, old);   // (async: after the queued lists that may still sample it)
}
// The finalize itself, with the exec lock held: now (synchronous execute) or on the render thread after every list
// queued before it (async execute: queued lists name the object by host pointer).
static void texture_finalize_locked(void *arg) {
    NvnTexture *t = (NvnTexture *)arg;
    static int nozombie = -1;
    if (nozombie < 0) nozombie = getenv("LARECOMP_NO_TEX_ZOMBIE") != NULL;
    if (__atomic_load_n(&t->pool_refs, __ATOMIC_ACQUIRE) > 0 && !nozombie) {
        // The object stays, its image goes (recycled): TotK finalizes ~100 transient render targets per second while
        // their slots stay registered (run321: 6000 kept images = 6.2 GB of VRAM in 30 s). Sampled later, a kept
        // object reads another object's live rendering at the same storage (textures.c alias_writer) or, if it was a
        // CPU-written texture (the map icons), is uploaded again from its storage (exec.c shader_texture_object).
        extern bool renderer_texture_gpu_owned(NvnTexture *t);
        t->zombie_gpu = renderer_texture_gpu_owned(t);
        backend_texture_destroyed(t);
        __atomic_store_n(&t->zombie, true, __ATOMIC_SEQ_CST);   // (before re-reading pool_refs below: nvn_pool_entry_set)
        g_zombies++;
        g_zombie_bytes += tegra_total_size(&t->layout);
        static int nlog;
        static uint64_t next_mb = 64;  // (and whenever the kept total doubles past 64 MB: VRAM watch)
        bool big = g_zombie_bytes >= next_mb << 20;
        if (big) next_mb *= 2;
        if (nlog++ < 10 || !(nlog % 1000) || big)
            LOG("[nvn] texture %p %ux%u f%#x addr %llx%s finalized while %d pool slots name it: object kept (%d kept, %.1f MB of storage)\n",
                (void *)t, t->width, t->height, t->format, (unsigned long long)t->addr, t->zombie_gpu ? " (rendered)" : "", t->pool_refs,
                g_zombies, g_zombie_bytes / 1048576.0);
        if (__atomic_load_n(&t->pool_refs, __ATOMIC_SEQ_CST) <= 0 &&   // (async: the last slot was re-registered meanwhile)
            !__atomic_exchange_n(&t->release_queued, true, __ATOMIC_SEQ_CST))
            zombie_release(t);
        return;
    }
    // pool entries hold the object pointer (real NVN copies the descriptor); drop the ones still naming it, or
    // a later draw reads freed memory (run92: crash in draw_, entry->tex->backend == 1). Every entry write goes through
    // nvn_pool_entry_set, so pool_refs counts them: at 0 nothing names it and the whole-pool scan (~100 finalizes/s, render
    // thread 0.9 % of all Deck samples, d45) is skipped. LARECOMP_TEX_FIN_SCAN=1: always scan.
    static int fin_scan = -1;
    if (fin_scan < 0) fin_scan = getenv("LARECOMP_TEX_FIN_SCAN") != NULL;
    if (fin_scan || __atomic_load_n(&t->pool_refs, __ATOMIC_ACQUIRE) > 0)
    for (int p = 0; p < 64 && g_texture_pools[p]; p++) {
        NvnTexEntry *e = (NvnTexEntry *)g_texture_pools[p]->entries;
        for (int i = 0; i < g_texture_pools[p]->count; i++) if (e[i].tex == t) {
            e[i].tex = NULL;
            t->pool_refs--;
            e[i].dbg_fin_frame = renderer_present_count() + 1;
            NvnFinRec *r = &g_tex_fin[g_tex_fin_n++ & 1023];
            r->pool = p; r->tic = (uint32_t)i; r->w = t->width; r->h = t->height; r->fmt = t->format; r->addr = t->addr;
            r->frame = renderer_present_count();
            snprintf(r->label, sizeof r->label, "%s", t->label);
        }
    }
    backend_texture_destroyed(t); free(t);
}
NVN(nvnTextureFinalize) {
    NvnTexture *t = HOST(NvnTexture, A(0));
    SET_HOST(A(0), 0);
    if (!t || t->zombie || t->fin_queued) return;  // (a second finalize, through a copy of the guest object)
    t->fin_queued = true;
    backend_async_call(texture_finalize_locked, t);
}
NVN(nvnTextureSetDebugLabel) {
    NvnTexture *t = HOST(NvnTexture, A(0));
    if (t && A(1)) strncpy(t->label, (const char *)AP(1), sizeof t->label - 1);
}
#define TEX HOST(NvnTexture, A(0))
NVN(nvnTextureGetFlags) { RET(TEX->flags); }
NVN(nvnTextureGetTarget) { RET(TEX->target); }
NVN(nvnTextureGetWidth) { RET(TEX->width); }
NVN(nvnTextureGetHeight) { RET(TEX->height); }
NVN(nvnTextureGetDepth) { RET(TEX->depth); }
NVN(nvnTextureGetLevels) { RET(TEX->levels); }
NVN(nvnTextureGetFormat) { RET(TEX->format); }
NVN(nvnTextureGetSamples) { RET(TEX->samples); }
NVN(nvnTextureGetSwizzle) { for (int i = 0; i < 4; i++) *(uint32_t *)AP(1 + i) = TEX->swizzle[i]; }
NVN(nvnTextureGetDepthStencilMode) { RET(TEX->ds_mode); }
NVN(nvnTextureGetStride) { RET(TEX->stride); }
NVN(nvnTextureGetTextureAddress) { RET(TEX->addr); }
NVN(nvnTextureGetMemoryPool) { RET(0); }
NVN(nvnTextureGetMemoryOffset) { RET(TEX->pool_offset); }
NVN(nvnTextureGetStorageSize) { RET(tegra_total_size(&TEX->layout)); }
NVN(nvnTextureGetDebugID) { RET(TEX->debug_id); }
NVN(nvnTextureGetViewOffset) {   // byte offset of the view's first level/layer in the texture storage (U11)
    const NvnTextureView *v = A(1) ? (const NvnTextureView *)AP(1) : NULL;
    uint32_t level = v && (v->setmask & 1) ? v->base_level : 0, layer = v && (v->setmask & 2) ? v->min_layer : 0;
    const TexLayout *t = &TEX->layout;
    static int zero = -1;
    if (zero < 0) zero = ENV("TOTK_VIEWOFF0") ? 1 : 0;   // A/B switch: old behaviour (0) while U12 is open
    uint64_t off = (uint64_t)layer * tegra_layer_size(t) + (level < t->levels ? tegra_level(t, level).offset : 0);
    static int logged;
    if (logged++ < 16) LOG("[nvn] GetViewOffset level %u layer %u -> %#llx (storage %#llx, %u levels %u layers)\n", level, layer,
                           (unsigned long long)off, (unsigned long long)tegra_total_size(t), t->levels, t->layers);
    RET(zero ? 0 : off);
}
NVN(nvnTextureGetStorageClass) { RET(0); }
NVN(nvnTextureGetRawStorageClass) { RET(0); }
NVN(nvnTextureGetSparseTileLayout) { RET(0); }
NVN(nvnTextureGetZCullStorageSize) { RET(0); }
NVN(nvnTextureCompare) { RET(A(0) == A(1)); }
NVN(nvnTextureFlushTexels) { (void)c; }
NVN(nvnTextureInvalidateTexels) { (void)c; }

// Texture views (NvnTextureView in nvn.h: our 0x10-byte packing of the guest struct)
#define TVP NvnTextureView *v = AP(0)
NVN(nvnTextureViewSetDefaults) { memset(AP(0), 0, sizeof(NvnTextureView)); }
NVN(nvnTextureViewSetLevels) { TVP; v->setmask |= 1; v->base_level = (uint8_t)A(1); v->num_levels = (uint8_t)A(2); }
NVN(nvnTextureViewSetLayers) { TVP; v->setmask |= 2; v->min_layer = (uint16_t)A(1); v->num_layers = (uint16_t)A(2); }
NVN(nvnTextureViewSetFormat) { TVP; v->setmask |= 4; v->format = (uint32_t)A(1); }
NVN(nvnTextureViewSetSwizzle) {
    TVP;
    v->setmask |= 8;
    v->swizzle[0] = (uint8_t)A(1); v->swizzle[1] = (uint8_t)A(2); v->swizzle[2] = (uint8_t)A(3); v->swizzle[3] = (uint8_t)A(4);
}
NVN(nvnTextureViewSetDepthStencilMode) { TVP; v->setmask |= 16; v->ds_mode = (uint8_t)A(1) & 1; }
NVN(nvnTextureViewSetTarget) { TVP; v->setmask |= 32; v->target = (uint8_t)A(1) & 0x7f; }
NVN(nvnTextureViewGetLevels) { TVP; *(int32_t *)AP(1) = v->base_level; *(int32_t *)AP(2) = v->num_levels; RET(!!(v->setmask & 1)); }
NVN(nvnTextureViewGetLayers) { TVP; *(int32_t *)AP(1) = v->min_layer; *(int32_t *)AP(2) = v->num_layers; RET(!!(v->setmask & 2)); }
NVN(nvnTextureViewGetFormat) { TVP; *(uint32_t *)AP(1) = v->format; RET(!!(v->setmask & 4)); }
NVN(nvnTextureViewGetSwizzle) { TVP; for (int i = 0; i < 4; i++) *(uint32_t *)AP(1 + i) = v->swizzle[i]; RET(!!(v->setmask & 8)); }
NVN(nvnTextureViewGetDepthStencilMode) { TVP; *(uint32_t *)AP(1) = v->ds_mode; RET(!!(v->setmask & 16)); }
NVN(nvnTextureViewGetTarget) { TVP; *(uint32_t *)AP(1) = v->target; RET(!!(v->setmask & 32)); }
NVN(nvnTextureViewCompare) { RET(memcmp(AP(0), AP(1), sizeof(NvnTextureView)) == 0); }

// (texture, view, region, data[, row stride, image stride]): CPU texel copies between tightly packed (or strided)
// linear data and the texture's storage. region = x, y, z offsets, width, height, depth in texels; z selects the
// layer of array / cube textures and the slice of 3D ones; the view picks the level. Block-linear storage: the
// level (one layer, or the whole 3D level) is deswizzled, patched or read, and swizzled back on writes.
void backend_texture_cpu_written(NvnTexture *t);  // (vulkan/textures.c: re-upload at the next use)
static void texels_rw(NvnTexture *t, const NvnTextureView *v, const int32_t *r, uint8_t *data, uint64_t rs, uint64_t is, bool write) {
    const FormatInfo *fi = t->layout.fmt;
    if (!t->addr || !r || !data || !fi || !fi->bpb || r[3] <= 0 || r[4] <= 0) return;
    uint32_t lv = v && (v->setmask & 1) ? v->base_level : 0;
    LevelLayout l = tegra_level(&t->layout, lv);
    bool is3d = t->target == 2, linear = nvn_flags_linear(t->flags);
    uint32_t bpb = fi->bpb;
    uint32_t bx = (uint32_t)r[0] / fi->bw, by = (uint32_t)r[1] / fi->bh;
    uint32_t nbx = ((uint32_t)r[3] + fi->bw - 1) / fi->bw, nby = ((uint32_t)r[4] + fi->bh - 1) / fi->bh;
    uint32_t z0 = r[2] > 0 ? (uint32_t)r[2] : 0, nz = r[5] > 0 ? (uint32_t)r[5] : 1;
    extern bool tex_watched(uint64_t addr);
    if (tex_watched(t->addr)) {
        static int nlog;
        if (nlog++ < 40)
            printf("[texw] %s texels addr %llx level %u region %d,%d,%d %dx%dx%d row stride %llu image stride %llu\n",
                   write ? "WriteTexels" : "ReadTexels", (unsigned long long)t->addr, lv, r[0], r[1], r[2], r[3], r[4], r[5],
                   (unsigned long long)rs, (unsigned long long)is);
    }
    if (bx >= l.wb || by >= l.hb) return;
    if (bx + nbx > l.wb) nbx = l.wb - bx;
    if (by + nby > l.hb) nby = l.hb - by;
    uint64_t row = (uint64_t)nbx * bpb;
    if (!rs) rs = row;
    if (!is) is = rs * nby;
    uint64_t lrow = (uint64_t)l.wb * bpb, lslice = lrow * l.hb;
    uint32_t slabs = is3d ? 1 : nz;   // 3D: one slab (the level, all slices); arrays: one slab per layer
    uint32_t dd = is3d ? l.d : 1;
    uint8_t *tmp = linear ? NULL : malloc(lslice * dd);
    if (!linear && !tmp) return;
    for (uint32_t s = 0; s < slabs; s++) {
        uint8_t *base = (uint8_t *)(uintptr_t)t->addr + (is3d ? 0 : (uint64_t)(z0 + s) * tegra_layer_size(&t->layout)) + l.offset;
        if (!linear) tegra_deswizzle(base, tmp, l.wb, l.hb, dd, bpb, l.block_height_log2, l.block_depth_log2);
        uint32_t zs = is3d ? z0 : 0, zn = is3d ? nz : 1;
        for (uint32_t z = 0; z < zn && zs + z < dd; z++)
            for (uint32_t y = 0; y < nby; y++) {
                uint8_t *img = linear ? base + (uint64_t)(by + y) * t->stride + (uint64_t)bx * bpb
                                      : tmp + (uint64_t)(zs + z) * lslice + (uint64_t)(by + y) * lrow + (uint64_t)bx * bpb;
                uint8_t *lin = data + (uint64_t)(s + z) * is + (uint64_t)y * rs;
                if (write) memcpy(img, lin, row); else memcpy(lin, img, row);
            }
        if (write && !linear) tegra_swizzle(tmp, base, l.wb, l.hb, dd, bpb, l.block_height_log2, l.block_depth_log2);
    }
    free(tmp);
    if (write) backend_texture_cpu_written(t);
}
NVN(nvnTextureWriteTexels) { texels_rw(HOST(NvnTexture, A(0)), AP(1), AP(2), AP(3), 0, 0, true); }
NVN(nvnTextureWriteTexelsStrided) { texels_rw(HOST(NvnTexture, A(0)), AP(1), AP(2), AP(3), A(4), A(5), true); }
NVN(nvnTextureReadTexels) { texels_rw(HOST(NvnTexture, A(0)), AP(1), AP(2), AP(3), 0, 0, false); }
NVN(nvnTextureReadTexelsStrided) { texels_rw(HOST(NvnTexture, A(0)), AP(1), AP(2), AP(3), A(4), A(5), false); }

// ================================================================== samplers
// SamplerBuilder: device@0 then a packed SbStore at 8. TotK: NVNsamplerBuilder is at most 0x40 bytes (measured: f_30f78
// keeps one at sp+0 with its saved x29/x30 at sp+0x40); LASRecomp's NvnSamplerDesc at +8 (80 bytes in all) wrote the
// border colour over them. SbStore + device = 56 bytes.
typedef struct {
    uint8_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float lod_min, lod_max, lod_bias, max_aniso;
    float border[4];
} SbStore;
_Static_assert(8 + sizeof(SbStore) <= 0x40, "must fit the guest NVNsamplerBuilder");
#define SD ((SbStore *)(AP(0) + 8))
static NvnSamplerDesc sb_desc(uint64_t builder) {
    const SbStore *b = (const SbStore *)(uintptr_t)(builder + 8);
    NvnSamplerDesc d = {.min_filter = b->min_filter, .mag_filter = b->mag_filter, .wrap = {b->wrap[0], b->wrap[1], b->wrap[2]},
                        .compare_mode = b->compare_mode, .compare_func = b->compare_func, .reduction = b->reduction,
                        .lod_min = b->lod_min, .lod_max = b->lod_max, .lod_bias = b->lod_bias, .max_aniso = b->max_aniso};
    memcpy(d.border, b->border, sizeof d.border);
    return d;
}
NVN(nvnSamplerBuilderSetDevice) { G(uint64_t, A(0), 0) = A(1); }
NVN(nvnSamplerBuilderSetDefaults) {
    SbStore *d = SD;
    memset(d, 0, sizeof *d);
    d->min_filter = 0; d->mag_filter = 0;
    d->wrap[0] = d->wrap[1] = d->wrap[2] = 1;  // REPEAT
    d->lod_min = 0; d->lod_max = 1000.0f; d->max_aniso = 1.0f;
    d->compare_func = 4;
}
NVN(nvnSamplerBuilderSetMinMagFilter) { SD->min_filter = (uint32_t)A(1); SD->mag_filter = (uint32_t)A(2); }
NVN(nvnSamplerBuilderSetWrapMode) { SD->wrap[0] = (uint32_t)A(1); SD->wrap[1] = (uint32_t)A(2); SD->wrap[2] = (uint32_t)A(3); }
NVN(nvnSamplerBuilderSetLodClamp) { SD->lod_min = AF(0); SD->lod_max = AF(1); }
NVN(nvnSamplerBuilderSetLodBias) { SD->lod_bias = AF(0); }
NVN(nvnSamplerBuilderSetCompare) { SD->compare_mode = (uint32_t)A(1); SD->compare_func = (uint32_t)A(2); }
NVN(nvnSamplerBuilderSetBorderColor) { memcpy(SD->border, AP(1), 16); }
NVN(nvnSamplerBuilderSetBorderColori) { memcpy(SD->border, AP(1), 16); }
NVN(nvnSamplerBuilderSetBorderColorui) { memcpy(SD->border, AP(1), 16); }
NVN(nvnSamplerBuilderSetMaxAnisotropy) { SD->max_aniso = AF(0); }
NVN(nvnSamplerBuilderSetReductionFilter) { SD->reduction = (uint32_t)A(1); }
NVN(nvnSamplerBuilderSetLodSnap) { (void)c; }
NVN(nvnSamplerBuilderGetDevice) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnSamplerBuilderGetMinMagFilter) { *(uint32_t *)AP(1) = SD->min_filter; *(uint32_t *)AP(2) = SD->mag_filter; }
NVN(nvnSamplerBuilderGetWrapMode) { for (int i = 0; i < 3; i++) *(uint32_t *)AP(1 + i) = SD->wrap[i]; }
NVN(nvnSamplerBuilderGetLodClamp) { *(float *)AP(1) = SD->lod_min; *(float *)AP(2) = SD->lod_max; }
NVN(nvnSamplerBuilderGetLodBias) { RETF(SD->lod_bias); }
NVN(nvnSamplerBuilderGetCompare) { *(uint32_t *)AP(1) = SD->compare_mode; *(uint32_t *)AP(2) = SD->compare_func; }
NVN(nvnSamplerBuilderGetBorderColor) { memcpy(AP(1), SD->border, 16); }
NVN(nvnSamplerBuilderGetBorderColori) { memcpy(AP(1), SD->border, 16); }
NVN(nvnSamplerBuilderGetBorderColorui) { memcpy(AP(1), SD->border, 16); }
NVN(nvnSamplerBuilderGetMaxAnisotropy) { RETF(SD->max_aniso); }
NVN(nvnSamplerBuilderGetReductionFilter) { RET(SD->reduction); }
NVN(nvnSamplerBuilderGetLodSnap) { RETF(0); }

NVN(nvnSamplerInitialize) {
    NvnSamplerDesc *d = malloc(sizeof *d);
    *d = sb_desc(A(1));
    SET_HOST(A(0), d);
    RET(1);
}
NVN(nvnSamplerFinalize) { void *p = HOST(NvnSamplerDesc, A(0)); SET_HOST(A(0), 0); if (p) backend_async_call(free, p); }
NVN(nvnSamplerSetDebugLabel) { (void)c; }
#define SMP HOST(NvnSamplerDesc, A(0))
NVN(nvnSamplerGetMinMagFilter) { *(uint32_t *)AP(1) = SMP->min_filter; *(uint32_t *)AP(2) = SMP->mag_filter; }
NVN(nvnSamplerGetWrapMode) { for (int i = 0; i < 3; i++) *(uint32_t *)AP(1 + i) = SMP->wrap[i]; }
NVN(nvnSamplerGetLodClamp) { *(float *)AP(1) = SMP->lod_min; *(float *)AP(2) = SMP->lod_max; }
NVN(nvnSamplerGetLodBias) { RETF(SMP->lod_bias); }
NVN(nvnSamplerGetCompare) { *(uint32_t *)AP(1) = SMP->compare_mode; *(uint32_t *)AP(2) = SMP->compare_func; }
NVN(nvnSamplerGetBorderColor) { memcpy(AP(1), SMP->border, 16); }
NVN(nvnSamplerGetBorderColori) { memcpy(AP(1), SMP->border, 16); }
NVN(nvnSamplerGetBorderColorui) { memcpy(AP(1), SMP->border, 16); }
NVN(nvnSamplerGetMaxAnisotropy) { RETF(SMP->max_aniso); }
NVN(nvnSamplerGetReductionFilter) { RET(SMP->reduction); }
NVN(nvnSamplerCompare) { RET(memcmp(HOST(NvnSamplerDesc, A(0)), HOST(NvnSamplerDesc, A(1)), sizeof(NvnSamplerDesc)) == 0); }
NVN(nvnSamplerGetDebugID) { RET(A(0)); }

// ================================================================== descriptor pools
NvnDescPool *g_texture_pools[64], *g_sampler_pools[64];
RT_ON_RESET(desc_pools_reset) { memset(g_texture_pools, 0, sizeof g_texture_pools); memset(g_sampler_pools, 0, sizeof g_sampler_pools); }

static NvnDescPool *desc_pool_init(uint64_t guest, uint64_t pool, uint64_t off, int count, size_t entry, NvnDescPool **reg) {
    NvnDescPool *d = calloc(1, sizeof *d);
    d->addr = HOST(NvnPool, pool)->storage + off;
    d->count = count;
    d->entries = calloc(count, entry);
    SET_HOST(guest, d);
    G(uint64_t, guest, 8) = pool;
    G(uint64_t, guest, 16) = off;
    G(uint32_t, guest, 24) = (uint32_t)count;
    for (int i = 0; i < 64; i++) if (!reg[i]) { reg[i] = d; break; }
    return d;
}

NVN(nvnTexturePoolInitialize) {
    desc_pool_init(A(0), A(1), A(2), (int)A(3), sizeof(NvnTexEntry), g_texture_pools);
    RET(1);
}
NVN(nvnTexturePoolFinalize) { (void)c; }
NVN(nvnTexturePoolSetDebugLabel) { (void)c; }
NVN(nvnTexturePoolGetMemoryPool) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnTexturePoolGetMemoryOffset) { RET(G(uint64_t, A(0), 16)); }
NVN(nvnTexturePoolGetSize) { RET(G(uint32_t, A(0), 24)); }
static void register_texture(Cpu *c) {
    NvnDescPool *d = HOST(NvnDescPool, A(0));
    int id = (int)A(1);
    if (id < 0 || id >= d->count) return;
    NvnTexEntry *e = &((NvnTexEntry *)d->entries)[id];
    nvn_pool_entry_set(e, A(2) ? HOST(NvnTexture, A(2)) : NULL);   // (bumps g_pool_gen when the texture changes)
    bool hv = A(3) != 0;
    if (hv != e->has_view || (hv && memcmp(&e->view, AP(3), sizeof e->view))) __atomic_add_fetch(&g_pool_gen, 1, __ATOMIC_RELAXED);
    e->has_view = hv;
    if (e->has_view) memcpy(&e->view, AP(3), sizeof e->view);
    e->dbg_regs++;
    e->dbg_reg_frame = renderer_present_count() + 1;
    if (e->tex) {
        e->dbg_w = e->tex->width; e->dbg_h = e->tex->height; e->dbg_fmt = e->tex->format; e->dbg_addr = e->tex->addr;
        snprintf(e->dbg_label, sizeof e->dbg_label, "%s", e->tex->label);
    } else {
        e->dbg_w = e->dbg_h = e->dbg_fmt = 0; e->dbg_addr = A(2);
        static int nlog;
        if (nlog++ < 20) LOG("[nvn] RegisterTexture id %d: texture object %llx has no host data\n", id, (unsigned long long)A(2));
    }
}
NVN(nvnTexturePoolRegisterTexture) { register_texture(c); }
NVN(nvnTexturePoolRegisterImage) { register_texture(c); }

NVN(nvnSamplerPoolInitialize) {
    desc_pool_init(A(0), A(1), A(2), (int)A(3), sizeof(NvnSamplerDesc), g_sampler_pools);
    RET(1);
}
NVN(nvnSamplerPoolFinalize) { (void)c; }
NVN(nvnSamplerPoolSetDebugLabel) { (void)c; }
NVN(nvnSamplerPoolGetMemoryPool) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnSamplerPoolGetMemoryOffset) { RET(G(uint64_t, A(0), 16)); }
NVN(nvnSamplerPoolGetSize) { RET(G(uint32_t, A(0), 24)); }
NVN(nvnSamplerPoolRegisterSampler) {
    NvnDescPool *d = HOST(NvnDescPool, A(0));
    int id = (int)A(1);
    if (id >= 0 && id < d->count && memcmp(&((NvnSamplerDesc *)d->entries)[id], HOST(NvnSamplerDesc, A(2)), sizeof(NvnSamplerDesc))) {
        ((NvnSamplerDesc *)d->entries)[id] = *HOST(NvnSamplerDesc, A(2));
        __atomic_add_fetch(&g_pool_gen, 1, __ATOMIC_RELAXED);
    }
}
NVN(nvnSamplerPoolRegisterSamplerBuilder) {
    NvnDescPool *d = HOST(NvnDescPool, A(0));
    int id = (int)A(1);
    if (id >= 0 && id < d->count) {
        NvnSamplerDesc nd = sb_desc(A(2));
        if (memcmp(&((NvnSamplerDesc *)d->entries)[id], &nd, sizeof nd)) {
            ((NvnSamplerDesc *)d->entries)[id] = nd;
            __atomic_add_fetch(&g_pool_gen, 1, __ATOMIC_RELAXED);
        }
    }
}

NvnDescPool *nvn_find_pool(NvnDescPool **reg, uint64_t guest_pool) {
    return guest_pool ? HOST(NvnDescPool, guest_pool) : NULL;
}

// ================================================================== programs
NVN(nvnProgramInitialize) { SET_HOST(A(0), calloc(1, sizeof(NvnProgram))); RET(1); }
NVN(nvnProgramFinalize) { void *p = HOST(NvnProgram, A(0)); SET_HOST(A(0), 0); if (p) backend_async_call(free, p); }
NVN(nvnProgramSetDebugLabel) { (void)c; }
NVN(nvnProgramSetSampleShading) { (void)c; }
NVN(nvnProgramSetSubroutineLinkage) { RET(1); }
static void set_shaders(Cpu *c, int stride) {
    NvnProgram *p = HOST(NvnProgram, A(0));
    int n = (int)A(1);
    const uint8_t *data = AP(2);  // NVNshaderData[n] = {u64 data (GPU addr), const void* control}
    memset(p->stages, 0, sizeof p->stages);
    p->stage_mask = 0;
    p->backend = NULL;
    for (int i = 0; i < n; i++) {
        uint64_t code = *(uint64_t *)(data + i * stride);
        const uint8_t *control = (const uint8_t *)(uintptr_t) * (uint64_t *)(data + i * stride + 8);
        // stage from the Maxwell shader program header (after the 0x30-byte NVN header)
        uint32_t sph0 = *(uint32_t *)(uintptr_t)(code + 0x30);
        uint32_t stage;
        switch ((sph0 >> 10) & 0xF) {
        case 1: stage = 0; break;  // vertex
        case 5: stage = 1; break;  // fragment
        case 4: stage = 2; break;  // geometry
        case 2: stage = 3; break;  // tess control
        case 3: stage = 4; break;  // tess eval
        default: stage = 5; break; // compute (no SPH)
        }
        if (*(uint32_t *)(uintptr_t)code != 0x12345678) stage = 5;
        p->stages[stage].stage = stage;
        p->stages[stage].code_addr = code;
        p->stages[stage].control = control;
        p->stages[stage].const_offset = *(const uint32_t *)(control + 0x700);
        p->stages[stage].const_size = *(const uint32_t *)(control + 0x6FC);
        p->stage_mask |= 1u << stage;
    }
    RET(1);
}

NVN(nvnProgramSetShaders) { set_shaders(c, 16); }    // NVNshaderData {data, control}
NVN(nvnProgramSetShadersExt) { set_shaders(c, 24); } // NVNshaderDataExt {data, control, debug hash}

// ================================================================== window
// WindowBuilder: device@0 native@8 num@0x10 textures@0x18 interval@0x20 active@0x24
NVN(nvnWindowBuilderSetDevice) { G(uint64_t, A(0), 0) = A(1); }
NVN(nvnWindowBuilderSetDefaults) { memset(AP(0) + 8, 0, 0x20); G(uint32_t, A(0), 0x20) = 1; }
NVN(nvnWindowBuilderSetNativeWindow) { G(uint64_t, A(0), 8) = A(1); }
NVN(nvnWindowBuilderSetTextures) { G(uint32_t, A(0), 0x10) = (uint32_t)A(1); G(uint64_t, A(0), 0x18) = A(2); }
NVN(nvnWindowBuilderSetPresentInterval) { G(uint32_t, A(0), 0x20) = (uint32_t)A(1); }
NVN(nvnWindowBuilderSetNumActiveTextures) { G(uint32_t, A(0), 0x24) = (uint32_t)A(1); }
NVN(nvnWindowBuilderGetDevice) { RET(G(uint64_t, A(0), 0)); }
NVN(nvnWindowBuilderGetNativeWindow) { RET(G(uint64_t, A(0), 8)); }
NVN(nvnWindowBuilderGetNumTextures) { RET(G(uint32_t, A(0), 0x10)); }
NVN(nvnWindowBuilderGetTexture) {
    int i = (int)A(1);
    RET(i < (int)G(uint32_t, A(0), 0x10) ? ((uint64_t *)(uintptr_t)G(uint64_t, A(0), 0x18))[i] : 0);
}
NVN(nvnWindowBuilderGetPresentInterval) { RET(G(uint32_t, A(0), 0x20)); }
NVN(nvnWindowBuilderGetNumActiveTextures) { RET(G(uint32_t, A(0), 0x24)); }

NVN(nvnWindowInitialize) {
    NvnWindow *w = calloc(1, sizeof *w);
    w->num = (int)G(uint32_t, A(1), 0x10);
    if (w->num > 8) w->num = 8;
    memcpy(w->textures, (void *)(uintptr_t)G(uint64_t, A(1), 0x18), w->num * 8);
    w->interval = (int)G(uint32_t, A(1), 0x20);
    w->active = G(uint32_t, A(1), 0x24) ? (int)G(uint32_t, A(1), 0x24) : w->num;
    SET_HOST(A(0), w);
    LOG("[nvn] window with %d textures\n", w->num);
    RET(1);
}
NVN(nvnWindowFinalize) { (void)c; }
NVN(nvnWindowSetDebugLabel) { (void)c; }
#define WIN HOST(NvnWindow, A(0))
NVN(nvnWindowGetNativeWindow) { RET(0); }
NVN(nvnWindowGetNumTextures) { RET(WIN->num); }
NVN(nvnWindowGetNumActiveTextures) { RET(WIN->active); }
NVN(nvnWindowSetNumActiveTextures) { WIN->active = (int)A(1); }
NVN(nvnWindowGetPresentInterval) { RET(WIN->interval); }
NVN(nvnWindowSetPresentInterval) { WIN->interval = (int)A(1); }
NVN(nvnWindowSetCrop) {
    int *cr = WIN->crop, n[4] = {(int)A(1), (int)A(2), (int)A(3), (int)A(4)};
    if (memcmp(cr, n, sizeof n)) LOG("[nvn] window crop %d,%d %dx%d\n", n[0], n[1], n[2], n[3]);
    memcpy(cr, n, sizeof n);
}
int g_present_crop[4];   // the presenting window's crop rect (device.c backend_present shows only that part of the texture)
NVN(nvnWindowGetCrop) { memcpy(AP(1), WIN->crop, 16); }
NVN(nvnWindowAcquireTexture) {
    NvnWindow *w = WIN;
    if (A(1)) sync_of(A(1))->signaled = 1;
    *(int32_t *)AP(2) = w->next;
    w->next = (w->next + 1) % (w->active ? w->active : 1);
    RET(0);  // NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS
}
NVN(nvnQueueAcquireTexture) {
    NvnWindow *w = HOST(NvnWindow, A(1));
    *(int32_t *)AP(2) = w->next;
    w->next = (w->next + 1) % (w->active ? w->active : 1);
    RET(0);
}
void vi_present_pace(int interval);   // hle_vi.cpp: blocks for the window's present interval (60 Hz vsync count)
static volatile uint64_t g_presents;   // TotK: frames presented by the game (scripted input keys on it, hle_hid.cpp)
uint64_t nvn_present_count(void) { return g_presents; }
NVN(nvnQueuePresentTexture) {
    NvnWindow *w = HOST(NvnWindow, A(1));
    int i = (int)A(2);
    if (i >= 0 && i < w->num) {
        memcpy(g_present_crop, w->crop, sizeof g_present_crop);
        if (backend_async_on()) backend_async_present(HOST(NvnTexture, w->textures[i]));
        else backend_present(HOST(NvnTexture, w->textures[i]));
    }
    g_presents++;
    vi_present_pace(w->interval);
    static uint64_t n, n_last; static double t_last;    // frame pacing: one line per ~2 s of wall time
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    double t = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (!n++) { t_last = t; LOG("[nvn] first present (texture %d)\n", i); }
    else if (t - t_last >= 2.0) { LOG("[nvn] present #%llu: %.1f fps\n", (unsigned long long)n, (n - n_last) / (t - t_last)); n_last = n; t_last = t; }
}

// ================================================================== small state objects
// BlendState (8): target, srcC, dstC, srcA, dstA, eqC, eqA, advanced
NVN(nvnBlendStateSetDefaults) {
    uint8_t *b = AP(0);
    b[0] = 0; b[1] = 2; b[2] = 1; b[3] = 2; b[4] = 1; b[5] = 1; b[6] = 1; b[7] = 0;
}
NVN(nvnBlendStateSetBlendTarget) { G(uint8_t, A(0), 0) = (uint8_t)A(1); }
NVN(nvnBlendStateSetBlendFunc) {
    uint8_t *b = AP(0);
    b[1] = (uint8_t)A(1); b[2] = (uint8_t)A(2); b[3] = (uint8_t)A(3); b[4] = (uint8_t)A(4);
}
NVN(nvnBlendStateSetBlendEquation) { G(uint8_t, A(0), 5) = (uint8_t)A(1); G(uint8_t, A(0), 6) = (uint8_t)A(2); }
NVN(nvnBlendStateSetAdvancedMode) { G(uint8_t, A(0), 7) = (uint8_t)A(1); }
NVN(nvnBlendStateSetAdvancedOverlap) { (void)c; }
NVN(nvnBlendStateSetAdvancedPremultipliedSrc) { (void)c; }
NVN(nvnBlendStateSetAdvancedNormalizedDst) { (void)c; }
NVN(nvnBlendStateGetBlendTarget) { RET(G(uint8_t, A(0), 0)); }
NVN(nvnBlendStateGetBlendFunc) {
    uint8_t *b = AP(0);
    for (int i = 0; i < 4; i++) *(uint32_t *)AP(1 + i) = b[1 + i];
}
NVN(nvnBlendStateGetBlendEquation) { *(uint32_t *)AP(1) = G(uint8_t, A(0), 5); *(uint32_t *)AP(2) = G(uint8_t, A(0), 6); }
NVN(nvnBlendStateGetAdvancedMode) { RET(G(uint8_t, A(0), 7)); }
NVN(nvnBlendStateGetAdvancedOverlap) { RET(0); }
NVN(nvnBlendStateGetAdvancedPremultipliedSrc) { RET(1); }
NVN(nvnBlendStateGetAdvancedNormalizedDst) { RET(1); }

// ChannelMaskState (4): 4 bits per target
NVN(nvnChannelMaskStateSetDefaults) { G(uint32_t, A(0), 0) = 0xFFFFFFFF; }
NVN(nvnChannelMaskStateSetChannelMask) {
    int t = (int)A(1);
    uint32_t m = (A(2) ? 1 : 0) | (A(3) ? 2 : 0) | (A(4) ? 4 : 0) | (A(5) ? 8 : 0);
    G(uint32_t, A(0), 0) = (G(uint32_t, A(0), 0) & ~(0xFu << (t * 4))) | (m << (t * 4));
}
NVN(nvnChannelMaskStateGetChannelMask) {
    uint32_t m = G(uint32_t, A(0), 0) >> ((int)A(1) * 4);
    for (int i = 0; i < 4; i++) *(uint8_t *)AP(2 + i) = (m >> i) & 1;
}

// ColorState (4): blend enable mask, logic op, alpha test
NVN(nvnColorStateSetDefaults) { G(uint8_t, A(0), 0) = 0; G(uint8_t, A(0), 1) = 3; G(uint8_t, A(0), 2) = 8; }
NVN(nvnColorStateSetBlendEnable) {
    uint8_t m = G(uint8_t, A(0), 0), bit = (uint8_t)(1u << A(1));
    G(uint8_t, A(0), 0) = A(2) ? (m | bit) : (m & ~bit);
}
NVN(nvnColorStateSetLogicOp) { G(uint8_t, A(0), 1) = (uint8_t)A(1); }
NVN(nvnColorStateSetAlphaTest) { G(uint8_t, A(0), 2) = (uint8_t)A(1); }
NVN(nvnColorStateGetBlendEnable) { RET((G(uint8_t, A(0), 0) >> A(1)) & 1); }
NVN(nvnColorStateGetLogicOp) { RET(G(uint8_t, A(0), 1)); }
NVN(nvnColorStateGetAlphaTest) { RET(G(uint8_t, A(0), 2)); }

// DepthStencilState (8): [0]=depthTest [1]=depthWrite [2]=depthFunc [3]=stencilTest
//   [4]=front func|back func<<4 [5..7]=ops (front fail,zfail,zpass / back) in 4-bit fields
NVN(nvnDepthStencilStateSetDefaults) {
    uint8_t *d = AP(0);
    d[0] = 0; d[1] = 1; d[2] = 2; d[3] = 0; d[4] = 0x88;
    d[5] = 0x11; d[6] = 0x11; d[7] = 0x11;
}
NVN(nvnDepthStencilStateSetDepthTestEnable) { G(uint8_t, A(0), 0) = (uint8_t)A(1); }
NVN(nvnDepthStencilStateSetDepthWriteEnable) { G(uint8_t, A(0), 1) = (uint8_t)A(1); }
NVN(nvnDepthStencilStateSetDepthFunc) { G(uint8_t, A(0), 2) = (uint8_t)A(1); }
NVN(nvnDepthStencilStateSetStencilTestEnable) { G(uint8_t, A(0), 3) = (uint8_t)A(1); }
NVN(nvnDepthStencilStateSetStencilFunc) {
    uint8_t *d = AP(0);
    int face = (int)A(1);  // 1 front, 2 back, 3 both
    if (face & 1) d[4] = (d[4] & 0xF0) | (A(2) & 0xF);
    if (face & 2) d[4] = (d[4] & 0x0F) | (uint8_t)((A(2) & 0xF) << 4);
}
NVN(nvnDepthStencilStateSetStencilOp) {
    uint8_t *d = AP(0);
    int face = (int)A(1);
    uint32_t ops = (uint32_t)((A(2) & 0xF) | (A(3) & 0xF) << 4 | (A(4) & 0xF) << 8);
    uint32_t packed = d[5] | d[6] << 8 | d[7] << 16;
    if (face & 1) packed = (packed & ~0xFFFu) | ops;
    if (face & 2) packed = (packed & 0xFFF) | ops << 12;
    d[5] = (uint8_t)packed; d[6] = (uint8_t)(packed >> 8); d[7] = (uint8_t)(packed >> 16);
}
NVN(nvnDepthStencilStateGetDepthTestEnable) { RET(G(uint8_t, A(0), 0)); }
NVN(nvnDepthStencilStateGetDepthWriteEnable) { RET(G(uint8_t, A(0), 1)); }
NVN(nvnDepthStencilStateGetDepthFunc) { RET(G(uint8_t, A(0), 2)); }
NVN(nvnDepthStencilStateGetStencilTestEnable) { RET(G(uint8_t, A(0), 3)); }
NVN(nvnDepthStencilStateGetStencilFunc) { RET((A(1) & 2) ? G(uint8_t, A(0), 4) >> 4 : G(uint8_t, A(0), 4) & 0xF); }
NVN(nvnDepthStencilStateGetStencilOp) {
    uint8_t *d = AP(0);
    uint32_t packed = d[5] | d[6] << 8 | d[7] << 16;
    if (A(1) & 2) packed >>= 12;
    for (int i = 0; i < 3; i++) *(uint32_t *)AP(2 + i) = (packed >> (4 * i)) & 0xF;
}

// MultisampleState (0x18): enable@0 samples@4 raster@8 a2c@0xC a2cdither@0xD cov2color@0xE output@0xF
NVN(nvnMultisampleStateSetDefaults) { memset(AP(0), 0, 0x18); G(uint8_t, A(0), 0) = 1; G(uint8_t, A(0), 0xD) = 1; }
NVN(nvnMultisampleStateSetMultisampleEnable) { G(uint8_t, A(0), 0) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetSamples) { G(uint32_t, A(0), 4) = (uint32_t)A(1); }
NVN(nvnMultisampleStateSetRasterSamples) { G(uint32_t, A(0), 8) = (uint32_t)A(1); }
NVN(nvnMultisampleStateSetAlphaToCoverageEnable) { G(uint8_t, A(0), 0xC) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetAlphaToCoverageDither) { G(uint8_t, A(0), 0xD) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetCoverageToColorEnable) { G(uint8_t, A(0), 0xE) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetCoverageToColorOutput) { G(uint8_t, A(0), 0xF) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetCoverageModulationMode) { G(uint8_t, A(0), 0x10) = (uint8_t)A(1); }
NVN(nvnMultisampleStateSetSampleLocations) { (void)c; }
NVN(nvnMultisampleStateSetSampleLocationsEnable) { (void)c; }
NVN(nvnMultisampleStateSetSampleLocationsGridEnable) { (void)c; }
NVN(nvnMultisampleStateGetMultisampleEnable) { RET(G(uint8_t, A(0), 0)); }
NVN(nvnMultisampleStateGetSamples) { RET(G(uint32_t, A(0), 4)); }
NVN(nvnMultisampleStateGetRasterSamples) { RET(G(uint32_t, A(0), 8)); }
NVN(nvnMultisampleStateGetAlphaToCoverageEnable) { RET(G(uint8_t, A(0), 0xC)); }
NVN(nvnMultisampleStateGetAlphaToCoverageDither) { RET(G(uint8_t, A(0), 0xD)); }
NVN(nvnMultisampleStateGetCoverageToColorEnable) { RET(G(uint8_t, A(0), 0xE)); }
NVN(nvnMultisampleStateGetCoverageToColorOutput) { RET(G(uint8_t, A(0), 0xF)); }
NVN(nvnMultisampleStateGetCoverageModulationMode) { RET(G(uint8_t, A(0), 0x10)); }
NVN(nvnMultisampleStateGetSampleLocationsEnable) { RET(0); }
NVN(nvnMultisampleStateGetSampleLocationsGrid) { (void)c; }
NVN(nvnMultisampleStateGetSampleLocationsGridEnable) { RET(0); }

// PolygonState (4): cull, front face, polygon mode, offset enables
NVN(nvnPolygonStateSetDefaults) { uint8_t *p = AP(0); p[0] = 0; p[1] = 1; p[2] = 2; p[3] = 0; }
NVN(nvnPolygonStateSetCullFace) { G(uint8_t, A(0), 0) = (uint8_t)A(1); }
NVN(nvnPolygonStateSetFrontFace) { G(uint8_t, A(0), 1) = (uint8_t)A(1); }
NVN(nvnPolygonStateSetPolygonMode) { G(uint8_t, A(0), 2) = (uint8_t)A(1); }
NVN(nvnPolygonStateSetPolygonOffsetEnables) { G(uint8_t, A(0), 3) = (uint8_t)A(1); }
NVN(nvnPolygonStateGetCullFace) { RET(G(uint8_t, A(0), 0)); }
NVN(nvnPolygonStateGetFrontFace) { RET(G(uint8_t, A(0), 1)); }
NVN(nvnPolygonStateGetPolygonMode) { RET(G(uint8_t, A(0), 2)); }
NVN(nvnPolygonStateGetPolygonOffsetEnables) { RET(G(uint8_t, A(0), 3)); }

// VertexAttribState (4): format bits 0-7, stream bits 8-12, relative offset bits 13-31
NVN(nvnVertexAttribStateSetDefaults) { G(uint32_t, A(0), 0) = 0; }
NVN(nvnVertexAttribStateSetFormat) {
    uint32_t v = G(uint32_t, A(0), 0);
    G(uint32_t, A(0), 0) = (v & 0x1F00) | (uint32_t)(A(1) & 0xFF) | (uint32_t)(A(2) << 13);
}
NVN(nvnVertexAttribStateSetStreamIndex) {
    G(uint32_t, A(0), 0) = (G(uint32_t, A(0), 0) & ~0x1F00u) | (uint32_t)((A(1) & 0x1F) << 8);
}
NVN(nvnVertexAttribStateGetFormat) {
    uint32_t v = G(uint32_t, A(0), 0);
    *(uint32_t *)AP(1) = v & 0xFF;
    *(int64_t *)AP(2) = v >> 13;
}
NVN(nvnVertexAttribStateGetStreamIndex) { RET((G(uint32_t, A(0), 0) >> 8) & 0x1F); }

// VertexStreamState (8): stride u32, divisor u32
NVN(nvnVertexStreamStateSetDefaults) { G(uint64_t, A(0), 0) = 0; }
NVN(nvnVertexStreamStateSetStride) { G(uint32_t, A(0), 0) = (uint32_t)A(1); }
NVN(nvnVertexStreamStateSetDivisor) { G(uint32_t, A(0), 4) = (uint32_t)A(1); }
NVN(nvnVertexStreamStateGetStride) { RET(G(uint32_t, A(0), 0)); }
NVN(nvnVertexStreamStateGetDivisor) { RET(G(uint32_t, A(0), 4)); }

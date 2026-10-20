// vkq_impl.h: deferred command recording (vkq.h), included once at the end of exec.c. The real Vulkan functions are
// called as "(vkCmdX)(...)", which the function-like macros of vkq.h leave alone.
// Ring: variable-size records (16-byte multiples) in one 32 MB buffer, monotonic byte counters q_w (published by the
// producer = whoever holds the backend lock and records, as before) and q_r (the record thread). Order is kept exactly,
// so the command buffers come out as if recorded inline. vkq_drain() waits for q_r to reach the producer's position.
// ring size: LARECOMP_VKQ_RING_MB (power of two, default 32): a ring larger than the caches makes every record a cold
// line on both threads
static uint64_t q_size = 32ull << 20;
#define Q_SIZE q_size
enum {
    QOP_WRAP, QOP_BEGIN, QOP_END, QOP_DRAW, QOP_DRAWIDX, QOP_DISPATCH, QOP_PIPE, QOP_SETS, QOP_PUSH, QOP_VBUF, QOP_IBUF,
    QOP_VIEWPORT, QOP_SCISSOR, QOP_BLENDC, QOP_DBIAS, QOP_DBIAS_EN, QOP_DTEST, QOP_DWRITE, QOP_DCMP, QOP_STEST, QOP_SOP, QOP_SCMASK,
    QOP_SWMASK, QOP_SREF, QOP_CULL, QOP_FRONT, QOP_TOPO, QOP_RESTART, QOP_BEGINR, QOP_ENDR, QOP_BARRIER, QOP_CPBUF, QOP_CPIMG,
    QOP_CPB2I, QOP_CPI2B, QOP_BLIT, QOP_RESOLVE, QOP_CLRCOL, QOP_CLRDS, QOP_CLRATT, QOP_QRESET, QOP_TS, QOP_FBL
};
typedef struct { uint32_t op, size; VkCommandBuffer cb; } QHdr;   // size: whole record, header included
static uint8_t *q_buf;
static _Atomic uint64_t q_w, q_r;
static uint64_t q_wl;            // producer's write position (q_w lags it until the record is complete)
static QHdr *q_cur;
static _Atomic int q_sleeping, q_drain_wait;
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER, q_dmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER, q_dcv = PTHREAD_COND_INITIALIZER;
static int q_state = -1;         // -1 unknown, 0 off, 1 on
static uint64_t q_stat_recs, q_stat_drains, q_stat_drain_ns, q_stat_full_ns, q_stat_wakes;
static int q_spin = -1, q_batch = -1;   // LARECOMP_VKQ_SPIN (pause loops before the record thread sleeps, default 0),
                                        // LARECOMP_VKQ_BATCH (records per wake-up, default 64): d42_q1 spun 4000 = 62 % of a
                                        // core for 28 % of work, which costs clock under the Deck's 15 W cap
static _Atomic uint64_t q_stat_busy_ns;   // (TSC ticks: two vDSO clock reads per batch were ~0.9 % of all Deck samples, d50_perf;
                                          //  converted to ns in vkq_stats with the TSC rate measured since q_main started)
static uint64_t q_tsc0, q_ns0;

static void *q_main(void *arg);
int vkq_enabled(void) {
    if (q_state < 0) {
        const char *e = getenv("LARECOMP_VKQ");
        q_state = !e || *e == '1';   // (default on since 2026-10-19: run588 PC 116.0 -> 119.2 fps, standing 111 -> 119 with SET_MEMO; =0 off)
        e = getenv("LARECOMP_VKQ_SPIN");
        q_spin = e ? atoi(e) : 0;   // (d51/d52: 0 vs 200 = same fps (58.1-58.8 both); 0 keeps the core idle when the frame is capped)
        e = getenv("LARECOMP_VKQ_BATCH");
        q_batch = e && atoi(e) > 0 ? atoi(e) : 64;
        if (q_state) {
            e = getenv("LARECOMP_VKQ_RING_MB");
            if (e && atoi(e) > 0) { q_size = 1ull << 20; while (q_size < (uint64_t)atoi(e) << 20) q_size <<= 1; }
            q_buf = malloc(Q_SIZE);
            if (!q_buf) q_state = 0;
            else {
                pthread_t t;
                pthread_create(&t, NULL, q_main, NULL);
                pthread_detach(t);
                LOG("[vkq] deferred command recording on (record thread, ring %llu MB)\n", (unsigned long long)(q_size >> 20));
            }
        }
    }
    return q_state;
}
static inline void q_pause(void) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}
// room for n more bytes after q_wl
static void q_wait_space(uint64_t n) {
    if (q_wl + n - atomic_load_explicit(&q_r, memory_order_acquire) <= Q_SIZE) return;
    uint64_t t0 = SDL_GetTicksNS();
    while (q_wl + n - atomic_load_explicit(&q_r, memory_order_acquire) > Q_SIZE) q_pause();
    q_stat_full_ns += SDL_GetTicksNS() - t0;
}
static void *q_alloc(uint32_t op, VkCommandBuffer cb, size_t payload) {
    uint64_t size = (sizeof(QHdr) + payload + 15) & ~15ull;
    uint64_t pos = q_wl & (Q_SIZE - 1);
    if (pos + size > Q_SIZE) {   // no room before the end: a wrap record, then from 0
        q_wait_space(Q_SIZE - pos + size);
        QHdr *w = (QHdr *)(q_buf + pos);
        w->op = QOP_WRAP;
        w->size = (uint32_t)(Q_SIZE - pos);
        q_wl += Q_SIZE - pos;
        pos = 0;
    } else q_wait_space(size);
    q_cur = (QHdr *)(q_buf + pos);
    q_cur->op = op;
    q_cur->size = (uint32_t)size;
    q_cur->cb = cb;
    return q_cur + 1;
}
static void q_wake(void) {
    q_stat_wakes++;
    pthread_mutex_lock(&q_mu);
    pthread_cond_signal(&q_cv);
    pthread_mutex_unlock(&q_mu);
}
static void q_commit(void) {
    q_wl += q_cur->size;
    q_stat_recs++;
    // (release, not seq-cst: an xchg per record. A lost wake-up - the store passing the q_sleeping load - only delays the
    // replay to the next commit or drain; vkq_drain fences before its q_sleeping check, so it always wakes it)
    atomic_store_explicit(&q_w, q_wl, memory_order_release);
    // (the record thread sleeps only with the ring empty, so q_wl - q_r = what was queued since it slept)
    if (atomic_load(&q_sleeping) && q_wl - atomic_load_explicit(&q_r, memory_order_relaxed) >= (uint64_t)q_batch * 32) q_wake();
}
void vkq_drain(void) {
    if (q_state <= 0) return;
    uint64_t target = q_wl;
    if (atomic_load(&q_r) == target) return;
    // (fence: the q_w release store must be visible before q_sleeping is read, or the record thread can fall asleep
    // between them with work queued and both threads wait forever: the user's 2026-10-19 16:23 freeze, minidump)
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load(&q_sleeping)) q_wake();
    uint64_t t0 = SDL_GetTicksNS();
    for (int i = 0; i < 2000 && atomic_load(&q_r) != target; i++) q_pause();
    if (atomic_load(&q_r) != target) {
        pthread_mutex_lock(&q_dmu);
        atomic_store(&q_drain_wait, 1);
        while (atomic_load(&q_r) != target) {
            if (atomic_load(&q_sleeping)) q_wake();   // (a record thread asleep with work queued would never broadcast)
            pthread_cond_wait(&q_dcv, &q_dmu);
        }
        atomic_store(&q_drain_wait, 0);
        pthread_mutex_unlock(&q_dmu);
    }
    q_stat_drains++;
    q_stat_drain_ns += SDL_GetTicksNS() - t0;
}
// stats for the "[vk]" line: records, drains + their wait, record-thread busy time; then reset
void vkq_stats(char *out, size_t n) {
    if (q_state <= 0) { if (n) out[0] = 0; return; }
    uint64_t tn = __rdtsc(), nn = SDL_GetTicksNS();
    double q_ns_per_tick = q_tsc0 && tn > q_tsc0 ? (double)(nn - q_ns0) / (double)(tn - q_tsc0) : 0;
    snprintf(out, n, " | vkq %llu recs, %llu wakes, %llu drains %.0f ms, full %.0f ms, rec busy %.0f ms", (unsigned long long)q_stat_recs,
             (unsigned long long)q_stat_wakes, (unsigned long long)q_stat_drains, q_stat_drain_ns / 1e6, q_stat_full_ns / 1e6,
             atomic_exchange(&q_stat_busy_ns, 0) * q_ns_per_tick / 1e6);
    q_stat_recs = q_stat_drains = q_stat_drain_ns = q_stat_full_ns = q_stat_wakes = 0;
}

// ---- records
#define QREC(OP, CB, T) T *a = (T *)q_alloc(OP, CB, sizeof(T))
#define QEXTRA(OP, CB, T, EXTRA) T *a = (T *)q_alloc(OP, CB, sizeof(T) + (EXTRA)); uint8_t *x_ = (uint8_t *)(a + 1); (void)x_
#define QCOPY(DST, SRC, N) do { if ((N) && (SRC)) { memcpy(x_, (SRC), (N)); } (DST) = (void *)x_; x_ += ((N) + 7) & ~(size_t)7; } while (0)

typedef struct { VkCommandBufferUsageFlags flags; } QBegin;
VkResult q_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) {
    if (!vkq_enabled() || bi->pNext || bi->pInheritanceInfo) { vkq_drain(); return (vkBeginCommandBuffer)(cb, bi); }
    QREC(QOP_BEGIN, cb, QBegin); a->flags = bi->flags; q_commit();
    return VK_SUCCESS;
}
VkResult q_vkEndCommandBuffer(VkCommandBuffer cb) {
    if (!vkq_enabled()) return (vkEndCommandBuffer)(cb);
    q_alloc(QOP_END, cb, 0); q_commit();
    return VK_SUCCESS;
}
typedef struct { uint32_t a, b, c, d; int32_t e; uint32_t f; } QU6;
void q_vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    if (!vkq_enabled()) { (vkCmdDraw)(cb, vc, ic, fv, fi); return; }
    QREC(QOP_DRAW, cb, QU6); a->a = vc; a->b = ic; a->c = fv; a->d = fi; q_commit();
}
void q_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t finst) {
    if (!vkq_enabled()) { (vkCmdDrawIndexed)(cb, ic, inst, fi, vo, finst); return; }
    QREC(QOP_DRAWIDX, cb, QU6); a->a = ic; a->b = inst; a->c = fi; a->e = vo; a->f = finst; q_commit();
}
void q_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    if (!vkq_enabled()) { (vkCmdDispatch)(cb, x, y, z); return; }
    QREC(QOP_DISPATCH, cb, QU6); a->a = x; a->b = y; a->c = z; q_commit();
}
typedef struct { VkPipelineBindPoint bp; VkPipeline p; } QPipe;
void q_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    if (!vkq_enabled()) { (vkCmdBindPipeline)(cb, bp, p); return; }
    QREC(QOP_PIPE, cb, QPipe); a->bp = bp; a->p = p; q_commit();
}
typedef struct { VkPipelineBindPoint bp; VkPipelineLayout l; uint32_t first, n, ndyn; const VkDescriptorSet *sets; const uint32_t *dyn; } QSets;
void q_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n,
                               const VkDescriptorSet *sets, uint32_t ndyn, const uint32_t *dyn) {
    if (!vkq_enabled()) { (vkCmdBindDescriptorSets)(cb, bp, l, first, n, sets, ndyn, dyn); return; }
    QEXTRA(QOP_SETS, cb, QSets, n * sizeof *sets + ndyn * sizeof *dyn + 16);
    a->bp = bp; a->l = l; a->first = first; a->n = n; a->ndyn = ndyn;
    QCOPY(a->sets, sets, n * sizeof *sets);
    QCOPY(a->dyn, dyn, ndyn * sizeof *dyn);
    q_commit();
}
typedef struct { VkPipelineLayout l; VkShaderStageFlags st; uint32_t off, size; const void *data; } QPush;
void q_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size, const void *data) {
    if (!vkq_enabled()) { (vkCmdPushConstants)(cb, l, st, off, size, data); return; }
    QEXTRA(QOP_PUSH, cb, QPush, size + 8);
    a->l = l; a->st = st; a->off = off; a->size = size;
    QCOPY(a->data, data, size);
    q_commit();
}
typedef struct { uint32_t first, n; const VkBuffer *b; const VkDeviceSize *o; } QVbuf;
void q_vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) {
    if (!vkq_enabled()) { (vkCmdBindVertexBuffers)(cb, first, n, b, o); return; }
    QEXTRA(QOP_VBUF, cb, QVbuf, n * (sizeof *b + sizeof *o) + 16);
    a->first = first; a->n = n;
    QCOPY(a->b, b, n * sizeof *b);
    QCOPY(a->o, o, n * sizeof *o);
    q_commit();
}
typedef struct { VkBuffer b; VkDeviceSize o; VkIndexType t; } QIbuf;
void q_vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    if (!vkq_enabled()) { (vkCmdBindIndexBuffer)(cb, b, o, t); return; }
    QREC(QOP_IBUF, cb, QIbuf); a->b = b; a->o = o; a->t = t; q_commit();
}
typedef struct { uint32_t first, n; const void *p; } QArr;
void q_vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport *v) {
    if (!vkq_enabled()) { (vkCmdSetViewport)(cb, first, n, v); return; }
    QEXTRA(QOP_VIEWPORT, cb, QArr, n * sizeof *v + 8); a->first = first; a->n = n; QCOPY(a->p, v, n * sizeof *v); q_commit();
}
void q_vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D *r) {
    if (!vkq_enabled()) { (vkCmdSetScissor)(cb, first, n, r); return; }
    QEXTRA(QOP_SCISSOR, cb, QArr, n * sizeof *r + 8); a->first = first; a->n = n; QCOPY(a->p, r, n * sizeof *r); q_commit();
}
typedef struct { float f[4]; } QF4;
void q_vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    if (!vkq_enabled()) { (vkCmdSetBlendConstants)(cb, c); return; }
    QREC(QOP_BLENDC, cb, QF4); memcpy(a->f, c, 16); q_commit();
}
void q_vkCmdSetDepthBias(VkCommandBuffer cb, float c, float cl, float s) {
    if (!vkq_enabled()) { (vkCmdSetDepthBias)(cb, c, cl, s); return; }
    QREC(QOP_DBIAS, cb, QF4); a->f[0] = c; a->f[1] = cl; a->f[2] = s; q_commit();
}
typedef struct { uint32_t v[5]; } QV;   // small enum / flag / bool arguments
#define QSIMPLE1(NAME, OP, T) \
    void q_##NAME(VkCommandBuffer cb, T v0) { \
        if (!vkq_enabled()) { (NAME)(cb, v0); return; } \
        QREC(OP, cb, QV); a->v[0] = (uint32_t)v0; q_commit(); \
    }
QSIMPLE1(vkCmdSetDepthBiasEnable, QOP_DBIAS_EN, VkBool32)
QSIMPLE1(vkCmdSetDepthTestEnable, QOP_DTEST, VkBool32)
QSIMPLE1(vkCmdSetDepthWriteEnable, QOP_DWRITE, VkBool32)
QSIMPLE1(vkCmdSetDepthCompareOp, QOP_DCMP, VkCompareOp)
QSIMPLE1(vkCmdSetStencilTestEnable, QOP_STEST, VkBool32)
QSIMPLE1(vkCmdSetCullMode, QOP_CULL, VkCullModeFlags)
QSIMPLE1(vkCmdSetFrontFace, QOP_FRONT, VkFrontFace)
QSIMPLE1(vkCmdSetPrimitiveTopology, QOP_TOPO, VkPrimitiveTopology)
QSIMPLE1(vkCmdSetPrimitiveRestartEnable, QOP_RESTART, VkBool32)
void q_vkCmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags f, VkStencilOp fail, VkStencilOp pass, VkStencilOp dfail, VkCompareOp cmp) {
    if (!vkq_enabled()) { (vkCmdSetStencilOp)(cb, f, fail, pass, dfail, cmp); return; }
    QREC(QOP_SOP, cb, QV); a->v[0] = f; a->v[1] = fail; a->v[2] = pass; a->v[3] = dfail; a->v[4] = cmp; q_commit();
}
#define QSIMPLE2(NAME, OP) \
    void q_##NAME(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v) { \
        if (!vkq_enabled()) { (NAME)(cb, f, v); return; } \
        QREC(OP, cb, QV); a->v[0] = f; a->v[1] = v; q_commit(); \
    }
QSIMPLE2(vkCmdSetStencilCompareMask, QOP_SCMASK)
QSIMPLE2(vkCmdSetStencilWriteMask, QOP_SWMASK)
QSIMPLE2(vkCmdSetStencilReference, QOP_SREF)
typedef struct { VkRenderingInfo ri; } QBeginR;
// (attachment pNext = exec.c's static fb_loop_info only: that pointer stays valid, so it is queued as is. NVIDIA has the
// feedback-loop extension: every pass drained the ring before, run583 45k drains / 2 s = 10 % slower than without VKQ)
#define Q_CHAIN(pn) ((pn) && (pn) != (const void *)&fb_loop_info)
void q_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    bool chain = ri->pNext || (ri->pDepthAttachment && Q_CHAIN(ri->pDepthAttachment->pNext)) || (ri->pStencilAttachment && Q_CHAIN(ri->pStencilAttachment->pNext));
    for (uint32_t i = 0; i < ri->colorAttachmentCount && !chain; i++) chain = Q_CHAIN(ri->pColorAttachments[i].pNext);
    if (!vkq_enabled() || chain) { vkq_drain(); (vkCmdBeginRendering)(cb, ri); return; }
    size_t na = ri->colorAttachmentCount * sizeof(VkRenderingAttachmentInfo);
    QEXTRA(QOP_BEGINR, cb, QBeginR, na + 2 * sizeof(VkRenderingAttachmentInfo) + 24);
    a->ri = *ri;
    QCOPY(a->ri.pColorAttachments, ri->pColorAttachments, na);
    if (ri->pDepthAttachment) QCOPY(a->ri.pDepthAttachment, ri->pDepthAttachment, sizeof(VkRenderingAttachmentInfo));
    if (ri->pStencilAttachment) QCOPY(a->ri.pStencilAttachment, ri->pStencilAttachment, sizeof(VkRenderingAttachmentInfo));
    q_commit();
}
typedef struct { PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT fn; VkImageAspectFlags m; } QFbl;
void q_set_fbl(PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT fn, VkCommandBuffer cb, VkImageAspectFlags m) {
    if (!vkq_enabled()) { fn(cb, m); return; }
    QREC(QOP_FBL, cb, QFbl); a->fn = fn; a->m = m; q_commit();
}
void q_vkCmdEndRendering(VkCommandBuffer cb) {
    if (!vkq_enabled()) { (vkCmdEndRendering)(cb); return; }
    q_alloc(QOP_ENDR, cb, 0); q_commit();
}
typedef struct { VkDependencyInfo di; } QBar;
void q_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    bool chain = di->pNext != NULL;
    for (uint32_t i = 0; i < di->memoryBarrierCount && !chain; i++) chain = di->pMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount && !chain; i++) chain = di->pBufferMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount && !chain; i++) chain = di->pImageMemoryBarriers[i].pNext != NULL;
    if (!vkq_enabled() || chain) { vkq_drain(); (vkCmdPipelineBarrier2)(cb, di); return; }
    size_t nm = di->memoryBarrierCount * sizeof(VkMemoryBarrier2), nb = di->bufferMemoryBarrierCount * sizeof(VkBufferMemoryBarrier2),
           ni = di->imageMemoryBarrierCount * sizeof(VkImageMemoryBarrier2);
    QEXTRA(QOP_BARRIER, cb, QBar, nm + nb + ni + 24);
    a->di = *di;
    QCOPY(a->di.pMemoryBarriers, di->pMemoryBarriers, nm);
    QCOPY(a->di.pBufferMemoryBarriers, di->pBufferMemoryBarriers, nb);
    QCOPY(a->di.pImageMemoryBarriers, di->pImageMemoryBarriers, ni);
    q_commit();
}
// copies / blits / clears: src, dst handles + layouts, region array
typedef struct { uint64_t src, dst; VkImageLayout sl, dl; uint32_t n, filter; const void *r; } QCopy;
void q_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {
    if (!vkq_enabled()) { (vkCmdCopyBuffer)(cb, s, d, n, r); return; }
    QEXTRA(QOP_CPBUF, cb, QCopy, n * sizeof *r + 8); a->src = (uint64_t)s; a->dst = (uint64_t)d; a->n = n; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy *r) {
    if (!vkq_enabled()) { (vkCmdCopyImage)(cb, s, sl, d, dl, n, r); return; }
    QEXTRA(QOP_CPIMG, cb, QCopy, n * sizeof *r + 8);
    a->src = (uint64_t)s; a->dst = (uint64_t)d; a->sl = sl; a->dl = dl; a->n = n; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl, uint32_t n, const VkBufferImageCopy *r) {
    if (!vkq_enabled()) { (vkCmdCopyBufferToImage)(cb, s, d, dl, n, r); return; }
    QEXTRA(QOP_CPB2I, cb, QCopy, n * sizeof *r + 8);
    a->src = (uint64_t)s; a->dst = (uint64_t)d; a->dl = dl; a->n = n; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d, uint32_t n, const VkBufferImageCopy *r) {
    if (!vkq_enabled()) { (vkCmdCopyImageToBuffer)(cb, s, sl, d, n, r); return; }
    QEXTRA(QOP_CPI2B, cb, QCopy, n * sizeof *r + 8);
    a->src = (uint64_t)s; a->dst = (uint64_t)d; a->sl = sl; a->n = n; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageBlit *r, VkFilter f) {
    if (!vkq_enabled()) { (vkCmdBlitImage)(cb, s, sl, d, dl, n, r, f); return; }
    QEXTRA(QOP_BLIT, cb, QCopy, n * sizeof *r + 8);
    a->src = (uint64_t)s; a->dst = (uint64_t)d; a->sl = sl; a->dl = dl; a->n = n; a->filter = f; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageResolve *r) {
    if (!vkq_enabled()) { (vkCmdResolveImage)(cb, s, sl, d, dl, n, r); return; }
    QEXTRA(QOP_RESOLVE, cb, QCopy, n * sizeof *r + 8);
    a->src = (uint64_t)s; a->dst = (uint64_t)d; a->sl = sl; a->dl = dl; a->n = n; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
typedef struct { uint64_t img; VkImageLayout l; uint32_t n; VkClearValue v; const VkImageSubresourceRange *r; } QClr;
void q_vkCmdClearColorImage(VkCommandBuffer cb, VkImage img, VkImageLayout l, const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *r) {
    if (!vkq_enabled()) { (vkCmdClearColorImage)(cb, img, l, c, n, r); return; }
    QEXTRA(QOP_CLRCOL, cb, QClr, n * sizeof *r + 8);
    a->img = (uint64_t)img; a->l = l; a->n = n; a->v.color = *c; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
void q_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage img, VkImageLayout l, const VkClearDepthStencilValue *c, uint32_t n,
                                   const VkImageSubresourceRange *r) {
    if (!vkq_enabled()) { (vkCmdClearDepthStencilImage)(cb, img, l, c, n, r); return; }
    QEXTRA(QOP_CLRDS, cb, QClr, n * sizeof *r + 8);
    a->img = (uint64_t)img; a->l = l; a->n = n; a->v.depthStencil = *c; QCOPY(a->r, r, n * sizeof *r); q_commit();
}
typedef struct { uint32_t n, nr; const VkClearAttachment *at; const VkClearRect *r; } QClrAtt;
void q_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment *at, uint32_t nr, const VkClearRect *r) {
    if (!vkq_enabled()) { (vkCmdClearAttachments)(cb, n, at, nr, r); return; }
    QEXTRA(QOP_CLRATT, cb, QClrAtt, n * sizeof *at + nr * sizeof *r + 16);
    a->n = n; a->nr = nr; QCOPY(a->at, at, n * sizeof *at); QCOPY(a->r, r, nr * sizeof *r); q_commit();
}
typedef struct { VkQueryPool p; uint32_t a, b; VkPipelineStageFlags2 st; } QQuery;
void q_vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool p, uint32_t first, uint32_t n) {
    if (!vkq_enabled()) { (vkCmdResetQueryPool)(cb, p, first, n); return; }
    QREC(QOP_QRESET, cb, QQuery); a->p = p; a->a = first; a->b = n; q_commit();
}
void q_vkCmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 st, VkQueryPool p, uint32_t q) {
    if (!vkq_enabled()) { (vkCmdWriteTimestamp2)(cb, st, p, q); return; }
    QREC(QOP_TS, cb, QQuery); a->p = p; a->a = q; a->st = st; q_commit();
}

// ---- record thread
static void q_exec(const QHdr *h) {
    VkCommandBuffer cb = h->cb;
    const void *p = h + 1;
    switch (h->op) {
    case QOP_BEGIN: {
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = ((const QBegin *)p)->flags};
        VK_CHECK((vkBeginCommandBuffer)(cb, &bi));
        break;
    }
    case QOP_END: VK_CHECK((vkEndCommandBuffer)(cb)); break;
    case QOP_DRAW: { const QU6 *a = p; (vkCmdDraw)(cb, a->a, a->b, a->c, a->d); break; }
    case QOP_DRAWIDX: { const QU6 *a = p; (vkCmdDrawIndexed)(cb, a->a, a->b, a->c, a->e, a->f); break; }
    case QOP_DISPATCH: { const QU6 *a = p; (vkCmdDispatch)(cb, a->a, a->b, a->c); break; }
    case QOP_PIPE: { const QPipe *a = p; (vkCmdBindPipeline)(cb, a->bp, a->p); break; }
    case QOP_SETS: { const QSets *a = p; (vkCmdBindDescriptorSets)(cb, a->bp, a->l, a->first, a->n, a->sets, a->ndyn, a->ndyn ? a->dyn : NULL); break; }
    case QOP_PUSH: { const QPush *a = p; (vkCmdPushConstants)(cb, a->l, a->st, a->off, a->size, a->data); break; }
    case QOP_VBUF: { const QVbuf *a = p; (vkCmdBindVertexBuffers)(cb, a->first, a->n, a->b, a->o); break; }
    case QOP_IBUF: { const QIbuf *a = p; (vkCmdBindIndexBuffer)(cb, a->b, a->o, a->t); break; }
    case QOP_VIEWPORT: { const QArr *a = p; (vkCmdSetViewport)(cb, a->first, a->n, (const VkViewport *)a->p); break; }
    case QOP_SCISSOR: { const QArr *a = p; (vkCmdSetScissor)(cb, a->first, a->n, (const VkRect2D *)a->p); break; }
    case QOP_BLENDC: { const QF4 *a = p; (vkCmdSetBlendConstants)(cb, a->f); break; }
    case QOP_DBIAS: { const QF4 *a = p; (vkCmdSetDepthBias)(cb, a->f[0], a->f[1], a->f[2]); break; }
    case QOP_DBIAS_EN: (vkCmdSetDepthBiasEnable)(cb, ((const QV *)p)->v[0]); break;
    case QOP_DTEST: (vkCmdSetDepthTestEnable)(cb, ((const QV *)p)->v[0]); break;
    case QOP_DWRITE: (vkCmdSetDepthWriteEnable)(cb, ((const QV *)p)->v[0]); break;
    case QOP_DCMP: (vkCmdSetDepthCompareOp)(cb, (VkCompareOp)((const QV *)p)->v[0]); break;
    case QOP_STEST: (vkCmdSetStencilTestEnable)(cb, ((const QV *)p)->v[0]); break;
    case QOP_CULL: (vkCmdSetCullMode)(cb, ((const QV *)p)->v[0]); break;
    case QOP_FRONT: (vkCmdSetFrontFace)(cb, (VkFrontFace)((const QV *)p)->v[0]); break;
    case QOP_TOPO: (vkCmdSetPrimitiveTopology)(cb, (VkPrimitiveTopology)((const QV *)p)->v[0]); break;
    case QOP_RESTART: (vkCmdSetPrimitiveRestartEnable)(cb, ((const QV *)p)->v[0]); break;
    case QOP_SOP: {
        const QV *a = p;
        (vkCmdSetStencilOp)(cb, a->v[0], (VkStencilOp)a->v[1], (VkStencilOp)a->v[2], (VkStencilOp)a->v[3], (VkCompareOp)a->v[4]);
        break;
    }
    case QOP_SCMASK: (vkCmdSetStencilCompareMask)(cb, ((const QV *)p)->v[0], ((const QV *)p)->v[1]); break;
    case QOP_SWMASK: (vkCmdSetStencilWriteMask)(cb, ((const QV *)p)->v[0], ((const QV *)p)->v[1]); break;
    case QOP_SREF: (vkCmdSetStencilReference)(cb, ((const QV *)p)->v[0], ((const QV *)p)->v[1]); break;
    case QOP_BEGINR: (vkCmdBeginRendering)(cb, &((const QBeginR *)p)->ri); break;
    case QOP_ENDR: (vkCmdEndRendering)(cb); break;
    case QOP_BARRIER: (vkCmdPipelineBarrier2)(cb, &((const QBar *)p)->di); break;
    case QOP_CPBUF: { const QCopy *a = p; (vkCmdCopyBuffer)(cb, (VkBuffer)a->src, (VkBuffer)a->dst, a->n, (const VkBufferCopy *)a->r); break; }
    case QOP_CPIMG: { const QCopy *a = p; (vkCmdCopyImage)(cb, (VkImage)a->src, a->sl, (VkImage)a->dst, a->dl, a->n, (const VkImageCopy *)a->r); break; }
    case QOP_CPB2I: { const QCopy *a = p; (vkCmdCopyBufferToImage)(cb, (VkBuffer)a->src, (VkImage)a->dst, a->dl, a->n, (const VkBufferImageCopy *)a->r); break; }
    case QOP_CPI2B: { const QCopy *a = p; (vkCmdCopyImageToBuffer)(cb, (VkImage)a->src, a->sl, (VkBuffer)a->dst, a->n, (const VkBufferImageCopy *)a->r); break; }
    case QOP_BLIT: {
        const QCopy *a = p;
        (vkCmdBlitImage)(cb, (VkImage)a->src, a->sl, (VkImage)a->dst, a->dl, a->n, (const VkImageBlit *)a->r, (VkFilter)a->filter);
        break;
    }
    case QOP_RESOLVE: { const QCopy *a = p; (vkCmdResolveImage)(cb, (VkImage)a->src, a->sl, (VkImage)a->dst, a->dl, a->n, (const VkImageResolve *)a->r); break; }
    case QOP_CLRCOL: { const QClr *a = p; (vkCmdClearColorImage)(cb, (VkImage)a->img, a->l, &a->v.color, a->n, a->r); break; }
    case QOP_CLRDS: { const QClr *a = p; (vkCmdClearDepthStencilImage)(cb, (VkImage)a->img, a->l, &a->v.depthStencil, a->n, a->r); break; }
    case QOP_CLRATT: { const QClrAtt *a = p; (vkCmdClearAttachments)(cb, a->n, a->at, a->nr, a->r); break; }
    case QOP_QRESET: { const QQuery *a = p; (vkCmdResetQueryPool)(cb, a->p, a->a, a->b); break; }
    case QOP_TS: { const QQuery *a = p; (vkCmdWriteTimestamp2)(cb, a->st, a->p, a->a); break; }
    case QOP_FBL: { const QFbl *a = p; a->fn(cb, a->m); break; }
    default: FATAL("[vkq] bad record op %u\n", h->op);
    }
}
static void *q_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#ifndef _WIN32
    extern int pthread_setname_np(pthread_t, const char *);
    pthread_setname_np(pthread_self(), "vk-record");   // ([cpu] line)
    { extern void host_affinity(int kind, unsigned long long mask); host_affinity(1, 0); }   // (TOTK_AFFINITY)
#endif
    uint64_t r = 0;
    q_ns0 = SDL_GetTicksNS(); q_tsc0 = __rdtsc();
    for (;;) {
        uint64_t w = atomic_load(&q_w);
        if (r == w) {
            atomic_thread_fence(memory_order_seq_cst);   // (the q_r release store before the q_drain_wait load: Dekker with vkq_drain)
            if (atomic_load(&q_drain_wait)) {
                pthread_mutex_lock(&q_dmu);
                pthread_cond_broadcast(&q_dcv);
                pthread_mutex_unlock(&q_dmu);
            }
            for (int i = 0; i < q_spin && atomic_load(&q_w) == r; i++) q_pause();
            if (atomic_load(&q_w) == r) {
                pthread_mutex_lock(&q_mu);
                atomic_store(&q_sleeping, 1);
                while (atomic_load(&q_w) == r) pthread_cond_wait(&q_cv, &q_mu);
                atomic_store(&q_sleeping, 0);
                pthread_mutex_unlock(&q_mu);
            }
            continue;
        }
        uint64_t t0 = __rdtsc();
        while (r != w) {
            const QHdr *h = (const QHdr *)(q_buf + (r & (Q_SIZE - 1)));
            if (h->op != QOP_WRAP) q_exec(h);
            r += h->size;
            atomic_store_explicit(&q_r, r, memory_order_release);
        }
        atomic_fetch_add(&q_stat_busy_ns, __rdtsc() - t0);
    }
    return NULL;
}

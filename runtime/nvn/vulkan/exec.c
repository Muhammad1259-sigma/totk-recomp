// Executes recorded NVN command lists with Vulkan (port of renderer.m; same structure, same behaviour).
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_timer.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
#include <time.h>
#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 1   // (Linux; the header hides it without _GNU_SOURCE)
#endif
#endif

#include "../shim/las.h"  // TotK: was ../../kernel.h
#include "shaders.h"
#include "vk.h"
#include "../shim/las.h"  // TotK: was ../renderer.h
#include "../shim/las.h"  // TotK: was ../../settings.h

uint64_t g_skip_prog_frame, g_skip_pipe_frame, g_skip_comp_frame, g_draws_frame, g_skip_full_frame;  // per presented frame: skips of work still being built (device.c holds such frames)
static bool g_render_off;   // SetRenderEnableConditional evaluated false (LARECOMP_COND_RENDER=1)
static uint64_t g_cond_evals, g_cond_off, g_cond_skipped, g_cond_flips;
static uint64_t prof_bake_waits, prof_bake_wait_ns;  // bake draws (3D / BC targets) that waited for their program
static bool g_draw_bake;
static uint64_t prof_pipe_inline, prof_pipe_waited, prof_pipe_wait_ns;  // (LARECOMP_PIPE_INLINE) draws that built / waited for their pipeline
// why draws were skipped (stats line "skip why"): 0 no program bound, 1 program not built, 2 LARECOMP_SKIP_FSHASH / LAS hash,
// 3 unsupported topology, 4 pipeline not built, 5 LAS item cursor, 6 compute not built
static uint64_t stat_skip_why[8];
static int pbn_cause = 2;                 // who asked for the next pass barrier: the vk_end_pass line, 1 = game barrier, 2 = start
static uint32_t stat_pbn_cause[8192];     // pass barriers per cause in the stats window (3 = LARECOMP_BARRIERS=pass forced it)
static bool pass_barrier_needed = true;   // (begin_pass: a new command buffer starts with a full barrier, begin_cb)
static bool pass_unbarriered;             // the open pass skipped its barrier
static bool game_bar_since_fb = true;      // a game CMD_BARRIER since the last feedback barrier (LARECOMP_FEEDBACK_GAMEBAR)
static uint64_t stat_fb_elided;
static uint64_t stat_bar_bits[512], stat_bar_inpass;   // game CMD_BARRIER: count per bits value (x[0] & 0x1ff), inside a pass
static uint64_t stat_count_fs;            // LARECOMP_COUNT_FSHASH=<fs hash>: that shader's draws per stats window (scene tag)
static bool pass_barrier_fresh;           // a full barrier was the last thing recorded (resolve_feedback): the pass needs no other
static uint32_t bar_epoch = 1;            // full barriers recorded in the main command buffers so far (vk_full_barrier)
static uint64_t stat_hazard_barriers;     // LARECOMP_BARRIERS=tracked: barriers for a texture / storage read after a write
static uint64_t stat_pass_barriers, stat_pass_nobarrier, stat_game_barriers, stat_game_barrier_breaks;
static uint64_t stat_draws, stat_skipped, stat_passes, stat_dispatches, stat_wait_ns, stat_flushes, stat_feedback_copies, stat_feedback_live, stat_depth_war, stat_fb_selfw, stat_fb_alias;
static uint64_t stat_upload_bytes;
int vk_up_kind;                   // (stats) what the vk_transient bytes being allocated are for: 0 other, 1 uniform (guest range),
                                  // 2 vertex, 3 index, 4 texture, 5 uniform driver constants (c0), 6 shader constants (c1), 7 zeros
static uint64_t stat_up_kind[8];  // upload bytes per kind in the stats window ("upload N MB (u/v/i/t/o MB)")
// LARECOMP_VK_PROF=1: render-thread time per category in the fps log (exec = whole command lists, draw = draw(),
// sets = descriptor sets incl. cbuf copies, vtx = index scan + vertex copies, submit = vkQueueSubmit2)
static int prof = -1;
static uint64_t prof_set_alloc, prof_set_cbuf, prof_set_fill, prof_set_update, prof_set_writes;  // (stage_set zones)
static uint64_t prof_set_calls[6], prof_set_same[6], prof_set_samedyn[6];  // (VK_PROF) stage_set calls per stage, same set / + same dynamic offsets as the previous call of that stage
extern uint64_t prof_memo_hit[6], prof_memo_bad[6], prof_memo_why[6][9];
// (VK_PROF) where the shader constants (c1) come from: zero-copy, heap mirror, or copies and why; guest_copy outcomes
static int hm_why;   // (VK_PROF) why heap_mirror_bind refused: 1 shader-written page, 2 past the mirror, 3 straddles, 4 no block, 5 off
static uint64_t prof_hmw[6], prof_c1[8], prof_c1_b[8], prof_gc_hit, prof_gc_new, prof_gc_new_b, prof_gc_evict, prof_gc_evict_b;   // (stage_set inputs memo, defined before stage_set)
static uint64_t prof_set_reused, prof_dyn_set, prof_dyn_skip, prof_pipe_bind, prof_pipe_skip;  // (stage_set reuse, draw_ dynamic state / pipeline cache)
static uint64_t prof_exec, prof_draw, prof_sets, prof_vtx, prof_submit, prof_pre, prof_key, prof_dyn, prof_tex, prof_up, prof_gpu;
uint64_t prof_vx_copy_n, prof_vx_copy_bytes, prof_vx_low_bytes, prof_vx_low_region, prof_vx_straddle, prof_vx_inflight;  // vertex streams copied instead of bound from the heap import
static uint64_t gmem_size;
static int ubo_zc;                            // (tentative: defined with cbuf_info, LARECOMP_UBO_ZEROCOPY)
typedef struct { uint64_t lo, hi; } ZcRange;
static ZcRange zc_excl[64];                   // (see cbuf_info: buffers the game updates inline)
static int zc_nexcl;
static uint64_t stat_ubo_zc, stat_ubo_copy;
static VkQueryPool prof_qp;
static uint64_t cum_exec, cum_wait, cum_draws, cum_submits, cum_tex_up;  // never reset: hitch log
static uint64_t cum_exec_cpu, cum_exec_ivcsw;   // (Linux) render thread on-CPU ns and involuntary switches inside exec: hitch log
static _Atomic int cum_prog_builds;
// (VK_PROF) draw_'s "pre" zone split (user's PC hitches 2026-10-18: 4.1 of 4.5 s of draw time in "pre", no compiles):
// program lookup, fs hash + debug, feedback, snapshots + begin_pass, topology; gps_ns = get_program_slow's zones in this
// draw: content key, claim (pc_mu), record (cache/code files), translate
static uint64_t prof_pz[5], gps_ns[4];
static uint32_t prof_end_line[4096];  // passes ended, by the exec.c line that ended them (0: submit / present)
uint64_t prof_views_created;  // textures.c vk_image_view cache misses
uint64_t prof_up_ns = ~0ull;  // textures.c upload checks (~0: profiling off)
// Profiling clock: the TSC scaled to SDL_GetTicksNS's epoch (calibrated in vk_exec_init). The ~35 zone reads per draw through
// SDL_GetTicksNS (vDSO clock_gettime) were 24 % of the Deck render thread's perf samples with VK_PROF on (d5, 2026-10-16).
static uint64_t prof_tsc0, prof_ns0;
static double prof_ns_per_tick = 1;
static inline uint64_t prof_now(void) { return prof_ns0 + (uint64_t)((double)(__rdtsc() - prof_tsc0) * prof_ns_per_tick); }
#define PROF_T0() uint64_t prof_t0_ = prof ? prof_now() : 0
#define PROF_ADD(v) do { if (prof) v += prof_now() - prof_t0_; } while (0)
static _Atomic int stat_compiles;
static uint64_t op_counts[CMD_COUNT];
static uint64_t op_ns[CMD_COUNT], op_t0, op_n2[CMD_COUNT];  // (LARECOMP_VK_PROF) render-thread time per command kind
static int op_last;
static uint64_t prof_binds_skipped, prof_binds_done;  // (LARECOMP_VK_PROF) vertex/index binds skipped as redundant
static uint64_t prof_bp, prof_bp_view, prof_bp_end, prof_bp_n, prof_bpz[4];  // bpz: vk_cmd, barrier, dl_note, BeginRendering  // (LARECOMP_VK_PROF) begin_pass: total, vk_view, previous pass end
bool g_debug_frame;
static char stats_text[2048];   // (640 cut off the tail of the VK_PROF line)
void vk_collect_dead(void);

// ================================================================== submissions
// A submission = optional upload command buffer + the main one, signalling the timeline with its number.
#define SUBS 16   // (8 -> 16 for chunked submission, LARECOMP_CHUNK_DRAWS)
typedef struct {
    VkCommandBuffer main, upload;
    bool main_begun, upload_begun, ts;  // ts: GPU timestamps written (profiling)
    uint64_t seq;
    VkDescriptorPool pools[16];
    int npools, cur_pool;
} Sub;
static Sub subs[SUBS];
static int cur_sub;
static uint64_t submit_seq;  // last submitted
// Chunked submission (LARECOMP_CHUNK_DRAWS=<n>, default 300, 0 = off; run269 vs run270 on Sky Island at the 60 cap:
// median 50.8 -> 59.9 fps, game fence wait 6.0 -> 1.1 ms, gdev traffic 163 -> 546 MB per 2 s): TotK hands over ~1 big command list per frame and the
// GPU got it only at the end of backend_execute (~14 ms of translation on Sky Island, run267/268), so the game's fence
// wait (nvnSyncWait, ~4.8 ms per frame) covered translation + GPU serially. With chunks, begin_pass submits what was
// recorded so far once n draws/dispatches are in it, and the GPU runs while the rest is translated.
static uint32_t chunk_draws;
static uint64_t prof_chunks;
static long chunk_ab = -1;
static bool chunk_ab_on = true;
// A chunk boundary is ours, not the game's: nothing guest-visible waits on it. Device-window regions stay in VRAM
// across the chunks of one list (no copy-back + re-upload per chunk: gdev traffic 129 -> 616 MB per 2 s in run300)
// and are copied back by the next real submission. A submission holding a FenceSync always copies back.
// LARECOMP_CHUNK_NO_CARRY=1: copy back at every chunk (run300 behaviour).
static bool g_chunk_submit;
static uint64_t g_fence_seq_mark;
static bool gdev_carry_ok(void);
static void gdev_carry(uint64_t from, uint64_t to);
static int chunk_limit(void) {
    static int n = -1;
    // Default ON (300) since 2026-10-15 17:38: the GPU faults of runs 270-272 (nvlddmkm 153, PC froze) stopped once
    // every command buffer starts with a full barrier (begin_cb); runs 300/302/303/304 (~10 min chunked incl. a 5-min
    // soak) clean; look-down A/B run303 30.9 -> 42.1 fps. LARECOMP_CHUNK_DRAWS=0: off.
    if (n < 0) { const char *e = getenv("LARECOMP_CHUNK_DRAWS"); n = e && *e ? atoi(e) : 300; if (n < 0) n = 0; }
    // LARECOMP_CHUNK_AB=<frames>: chunking on and off in alternating blocks of that many presented frames (same scene
    // A/B in one run; scripted camera moves differ between runs). Off blocks log "[chunk] off" in the stats window.
    // The decision is latched per guest list (backend_execute) so a list never switches mode half-way.
    if (chunk_ab < 0) { const char *e = getenv("LARECOMP_CHUNK_AB"); chunk_ab = e ? atol(e) : 0; }
    if (n && chunk_ab > 0 && !chunk_ab_on) return 0;
    return n;
}
// Alias mirror epoch: one guest command list (backend_execute). The guest finished its CPU writes for a list before
// submitting it, so the VRAM mirror is synced once per list, not once per Vulkan submission (a sync costs ~1.3 ms:
// run267 111 ms / 85 syncs). Outside an execute: once per submission, as before.
static uint64_t alias_epoch = 1, alias_sync_epoch;
static uint64_t exec_start_seq;  // pending submission when the current guest list started
static bool in_execute;
// nvnSyncWait timing (stats "| syncwait"): how long before the wait the waited submission went to the GPU, and how long
// the wait took. Age ~0 + a long wait = the game waits on work the GPU only just received (chunking helps).
static uint64_t sub_t[64];   // submit time by seq % 64
static uint64_t sw_n, sw_age_ns, sw_wait_ns;
static pthread_mutex_t seq_lock = PTHREAD_MUTEX_INITIALIZER;
static bool in_pass;

// Device window for the storage buffers shaders write (gdev_* below); LARECOMP_VK_NO_GDEV=1: off.
#define GDEV_SIZE (64ull << 20)
static uint64_t gdev_window_size(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GDEV") != NULL;
    return off || !g_vk.host_import ? 0 : GDEV_SIZE;
}
static void gdev_writeback(VkCommandBuffer cb);
static void gdev_check_prev(void);  // (LARECOMP_GDEV_CHECK)
static void gdev_shadow_sync(VkCommandBuffer up);
static uint64_t chk_words, chk_clob;
static bool gdev_chk_on(void);
static VkBuffer chk_buf[2];
static uint64_t gdev_map(uint64_t addr, uint64_t size, bool write);
static uint64_t gdev_stat_in, gdev_stat_regions, gdev_stat_snap, gdev_stat_host;  // (stats line)
static int gdev_n;

uint64_t vk_completed_seq(void) {
    static _Atomic uint64_t last;   // (after a device loss the query fails: keep the last value it gave)
    uint64_t v = 0;
    if (vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v) == VK_SUCCESS) last = v;
    return last;
}

// GPU wait time per call site (file:line), printed in the 2-s stats as "| waits site ms/count ..."
#define WSITES 32
static struct { const char *file; int line; _Atomic uint64_t ns, n; } wsite[WSITES];
static pthread_mutex_t wsite_mu = PTHREAD_MUTEX_INITIALIZER;
static void wsite_add(const char *file, int line, uint64_t ns) {
    int i = 0;
    for (; i < WSITES && wsite[i].file; i++)
        if (wsite[i].line == line && wsite[i].file == file) break;
    if (i == WSITES) return;
    if (!wsite[i].file) {
        pthread_mutex_lock(&wsite_mu);
        for (i = 0; i < WSITES && wsite[i].file && !(wsite[i].line == line && wsite[i].file == file); i++) {}
        if (i < WSITES && !wsite[i].file) { wsite[i].line = line; wsite[i].file = file; }
        pthread_mutex_unlock(&wsite_mu);
        if (i == WSITES) return;
    }
    wsite[i].ns += ns;
    wsite[i].n++;
}

static void hang_report(uint64_t hung);   // (GPU hang breadcrumbs, below)
static void hang_checkpoints(void);
void vk_wait_seq_at(uint64_t seq, const char *file, int line) {
    if (!seq) return;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    // A GPU that has not finished a submission after 3 s is hung (no TotK GPU job takes that long): close the process
    // at once instead of piling more work on it (user 2026-10-15: tests must close themselves when the PC starts to
    // freeze; run270 waited 10 s and carried on). LARECOMP_GPU_HANG_MS=<ms> (0 = old 10 s wait, keep running).
    static long hang_ms = -1;
    if (hang_ms < 0) { const char *e = getenv("LARECOMP_GPU_HANG_MS"); hang_ms = e ? atol(e) : 3000; }
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, (hang_ms > 0 ? (uint64_t)hang_ms : 10000ull) * 1000000ull);
    uint64_t dt = SDL_GetTicksNS() - t0;
    stat_wait_ns += dt;
    cum_wait += dt;
    wsite_add(file, line, dt);
    if (r == VK_TIMEOUT && hang_ms > 0) {
        // The stuck work is in the first unfinished submission, not necessarily the one waited for (2026-10-17 17:00:
        // the wait was for an empty flush, "0 draws, 0 dispatches", so the report named nothing)
        uint64_t done = vk_completed_seq();
        LOG("[vk] GPU HANG: submission %llu not finished after %ld ms (wait at %s:%d); GPU done up to %llu\n", (unsigned long long)seq,
            hang_ms, file, line, (unsigned long long)done);
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        hang_report(done < seq ? done + 1 : seq);
        // Closing at once does not free the GPU (2026-10-17: the process lived on until the driver reset the GPU
        // 97 s later, Windows TdrDelay = 100 s here), so wait: if the GPU finishes, the game carries on; if the driver
        // resets it, the checkpoints say which operation it was stuck on. LARECOMP_GPU_HANG_WAIT_S=<s> (default 150).
        static long wait_s = -1;
        if (wait_s < 0) { const char *e = getenv("LARECOMP_GPU_HANG_WAIT_S"); wait_s = e ? atol(e) : 150; }
        LOG("[hang] waiting up to %ld s for the GPU to finish or be reset\n", wait_s);
        fflush(stdout);
        uint64_t h0 = SDL_GetTicksNS();
        // LARECOMP_HANG_IDLE=1 (diagnostic): wait with vkDeviceWaitIdle instead, so a layer that watches for device lost
        // there (LunarG crash diagnostic layer) writes its dump
        if (getenv("LARECOMP_HANG_IDLE")) r = vkDeviceWaitIdle(g_vk.dev);
        else
            for (long i = 0; i < wait_s; i++) {
                r = vkWaitSemaphores(g_vk.dev, &wi, 1000000000ull);
                if (r != VK_TIMEOUT) break;
            }
        double hs = (SDL_GetTicksNS() - h0) / 1e9 + hang_ms / 1e3;
        if (r == VK_SUCCESS) {
            // A driver reset (TDR, ~100 s here) can signal the semaphore too: 2026-10-17 17:00 "finished after 102.5 s",
            // then the next submit failed with device lost and no checkpoints were printed. Ask whether the device is lost.
            uint64_t v = 0;
            VkResult lr = vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v);
            if (lr == VK_ERROR_DEVICE_LOST) {
                LOG("[hang] after %.1f s the semaphore is signalled but the device is lost (driver reset); closing the process\n", hs);
                hang_checkpoints();
                vk_device_fault_report();
                fflush(stdout);
                fflush(stderr);
                _exit(5);
            }
            LOG("[hang] the GPU finished submission %llu after %.1f s: carrying on\n", (unsigned long long)seq, hs);
            fflush(stdout);
            return;
        }
        LOG("[hang] after %.1f s: %s; closing the process\n", hs, r == VK_ERROR_DEVICE_LOST ? "the driver reset the GPU (device lost)" : "still stuck");
        hang_checkpoints();
        vk_device_fault_report();
        fflush(stdout);
        fflush(stderr);
        _exit(5);
    }
    if (r == VK_TIMEOUT) LOG("[vk] GPU did not finish submission %llu within 10 s (device lost?)\n", (unsigned long long)seq);
    else if (r != VK_SUCCESS) {
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        // (2026-10-19: after a reboot the driver resets a hung GPU in ~2 s, before the 3-s hang report above fires)
        uint64_t done = vk_completed_seq();
        hang_report(done < seq ? done + 1 : seq);
        hang_checkpoints();
        vk_device_fault_report();
        FATAL("vkWaitSemaphores: %d (device lost)\n", r);
    }
}

uint64_t vk_pending_seq(void) { return submit_seq + 1; }

static Sub *sub(void) {
    Sub *s = &subs[cur_sub];
    if (s->seq != submit_seq + 1) {  // slot reused: its previous submission must be done
        vk_wait_seq(s->seq);
        if (s->ts) {  // (profiling) GPU time of that submission
            uint64_t t[2];
            if (vkGetQueryPoolResults(g_vk.dev, prof_qp, (uint32_t)cur_sub * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0])
                prof_gpu += (uint64_t)((double)(t[1] - t[0]) * g_vk.props.limits.timestampPeriod);
            s->ts = false;
        }
        for (int i = 0; i < s->npools; i++) vkResetDescriptorPool(g_vk.dev, s->pools[i], 0);
        s->cur_pool = 0;
        s->seq = submit_seq + 1;
        s->main_begun = s->upload_begun = false;
    }
    return s;
}

static void begin_cb(VkCommandBuffer *cb, bool *begun) {
    if (*begun) return;
    if (!*cb) {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        vkq_drain();   // (the record thread uses the same command pool)
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, cb));
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(*cb, &bi));
    *begun = true;
    // Order against everything submitted before (a barrier's first scope covers earlier submissions on the queue). The
    // upload buffer used to start with its copies at once: texture uploads and guest -> device-window copies could run
    // while the PREVIOUS submission's shaders still read / wrote the same images and window regions. Rare when one
    // submission covers a whole frame; every time with chunked submission (suspect for the GPU faults of runs 270-272).
    // LARECOMP_NO_CB_BARRIER=1: off (old).
    static int nob = -1;
    if (nob < 0) nob = getenv("LARECOMP_NO_CB_BARRIER") != NULL;
    if (!nob) vk_full_barrier(*cb);
}

static void gp_mark(VkCommandBuffer cb, const char *fmt, ...);
VkCommandBuffer vk_cmd(void) {
    Sub *s = sub();
    if (!s->main_begun) {
        begin_cb(&s->main, &s->main_begun);
        gp_mark(s->main, "(submission start)");
        if (prof && prof_qp) {
            vkCmdResetQueryPool(s->main, prof_qp, (uint32_t)cur_sub * 2, 2);
            vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_NONE, prof_qp, (uint32_t)cur_sub * 2);
            s->ts = true;
        }
    }
    return s->main;
}

VkCommandBuffer vk_upload_cmd(void) {
    Sub *s = sub();
    begin_cb(&s->upload, &s->upload_begun);
    return s->upload;
}

void vk_full_barrier(VkCommandBuffer cb) {
    if (cb == sub()->main) bar_epoch++;
    // LARECOMP_BARRIER_ACCESS=narrow (A/B, 2026-10-18: uncapped + GPU-bound on the Deck, no barriers = +5 % but tracked
    // = -1.5 %): the device-side write -> read kinds only instead of MEMORY_WRITE -> MEMORY_READ|WRITE (cache flush cost?)
    static int narrow = -1;
    if (narrow < 0) { const char *e = getenv("LARECOMP_BARRIER_ACCESS"); narrow = e && !strcmp(e, "narrow"); }
    const VkAccessFlags2 wr = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = narrow ? wr : VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = narrow ? wr | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                                         VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                                         VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                                                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                         VK_ACCESS_2_TRANSFER_READ_BIT
                                                  : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void ring_retire(void);

// ---- LARECOMP_GPUPROF_AT=<frame>: GPU time per pass / dispatch / copy of that frame, logged as "[gpuprof]".
// A timestamp (ALL_COMMANDS: after everything before it) at each mark; a row = the work since the previous mark.
bool g_gpuprof_frame;
volatile int g_gpuprof_request;   // (vk_prof_frame, LARECOMP_GPUPROF_SLOW: profile the next frame)
static VkQueryPool gp_qp;
static int gp_n;
#define GP_MAX 16384  // (512 until 2026-10-16: the Deck profile of a gameplay frame stopped at 511 marks; 4096 until 2026-10-17: d24 stopped at 10.9 of ~25 ms)
static char gp_label[GP_MAX][112];
static uint32_t gp_draws[GP_MAX];
// LARECOMP_VK_DEVLOST_TRACE=1: the last 64 GPU operations, dumped on device loss (with LARECOMP_SYNC_SUBMIT=1 the
// dump ends with the command list that faulted).
static char dl_ring[64][128];
static unsigned dl_n;
static int dl_on = -1;
static void dl_note(const char *fmt, ...) {
    if (dl_on < 0) dl_on = getenv("LARECOMP_VK_DEVLOST_TRACE") ? atoi(getenv("LARECOMP_VK_DEVLOST_TRACE")) : 0;
    if (!dl_on) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dl_ring[dl_n++ & 63], sizeof dl_ring[0], fmt, ap);
    va_end(ap);
}
void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv) { dl_note("  upload %p %ux%u lv%u", img, w, h, lv); }
void vk_devlost_dump(void) {
    if (!dl_on) return;
    LOG("[vk] last GPU operations before the device loss (oldest first):\n");
    for (unsigned i = dl_n > 64 ? dl_n - 64 : 0; i < dl_n; i++) LOG("[vk]   %s\n", dl_ring[i & 63]);
}
// Hang breadcrumbs (2026-10-17: the user's play sessions froze the PC three times with "[vk] GPU HANG" and nothing
// else, because the trace above is off in play): every draw / dispatch / pass / copy goes into a ring (a few stores)
// with its submission number, and with VK_NV_device_diagnostic_checkpoints it is also a checkpoint whose marker is
// its ring position, so after a hang the driver names the last operation each pipeline stage reached. hang_report
// prints both. (With LARECOMP_VKQ=1 commands are recorded later on another thread: ring only, no checkpoints.)
typedef struct { uint64_t seq, a, b; uint32_t kind, x, y, z; } HangOp;
enum { HOP_DRAW = 1, HOP_DISPATCH, HOP_PASS, HOP_COPY, HOP_SUBMIT };
// draw: a = fs hash, b = vs code, x = vertices/indices, y = instances; dispatch: a = cs code, b = threads, x y z = groups;
// pass: x = colour targets, y z = size; copy: a = source, b = destination, y z = size; submit: a = seq, x = waits
enum { HANG_OPS = 1 << 16 };
static HangOp hang_ops[HANG_OPS];
static uint64_t hang_n;   // render thread
static int hang_ready;
static PFN_vkCmdSetCheckpointNV hang_cp;
static inline void hang_op(VkCommandBuffer cb, uint32_t kind, uint64_t a, uint64_t b, uint32_t x, uint32_t y, uint32_t z) {
    if (!hang_ready) {
        hang_ready = 1;
        if (g_vk.checkpoints) hang_cp = (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(g_vk.dev, "vkCmdSetCheckpointNV");
    }
    HangOp *o = &hang_ops[hang_n & (HANG_OPS - 1)];
    o->seq = submit_seq + 1, o->a = a, o->b = b, o->kind = kind, o->x = x, o->y = y, o->z = z;
    hang_n++;
    if (hang_cp && cb && !vkq_enabled()) hang_cp(cb, (void *)(uintptr_t)hang_n);   // (marker = ring position + 1)
}
static void hang_op_log(const char *tag, uint64_t pos) {   // pos = ring position
    if (pos >= hang_n || hang_n - pos > HANG_OPS) { LOG("[hang] %s: #%llu (no longer in the ring)\n", tag, (unsigned long long)pos); return; }
    const HangOp *o = &hang_ops[pos & (HANG_OPS - 1)];
    switch (o->kind) {
    case HOP_DRAW: LOG("[hang] %s: #%llu submission %llu draw fs %016llx vs %llx, %u vertices x %u instances | fb live %u snap %u%s%s%s%s%s\n", tag,
                       (unsigned long long)pos, (unsigned long long)o->seq, (unsigned long long)o->a, (unsigned long long)o->b, o->x, o->y,
                       o->z & 15, o->z >> 4 & 15, o->z >> 8 & 1 ? " +break" : "", o->z >> 9 & 1 ? " +selfwrite" : "",
                       o->z >> 10 & 1 ? " +depth" : "", o->z >> 11 & 1 ? " +copied" : "", o->z >> 12 & 1 ? " +hazard" : ""); break;
    case HOP_DISPATCH: LOG("[hang] %s: #%llu submission %llu dispatch cs %llx %ux%ux%u groups (%llu threads)\n", tag,
                           (unsigned long long)pos, (unsigned long long)o->seq, (unsigned long long)o->a, o->x, o->y, o->z, (unsigned long long)o->b); break;
    case HOP_PASS: LOG("[hang] %s: #%llu submission %llu pass begin (%u colour targets, %ux%u)\n", tag, (unsigned long long)pos,
                       (unsigned long long)o->seq, o->x, o->y, o->z); break;
    case HOP_COPY: LOG("[hang] %s: #%llu submission %llu copy/blit %llx -> %llx %ux%u\n", tag, (unsigned long long)pos,
                       (unsigned long long)o->seq, (unsigned long long)o->a, (unsigned long long)o->b, o->y, o->z); break;
    case HOP_SUBMIT: LOG("[hang] %s: #%llu submit %llu (wait semaphores %u)\n", tag, (unsigned long long)pos, (unsigned long long)o->a, o->x); break;
    default: LOG("[hang] %s: #%llu kind %u\n", tag, (unsigned long long)pos, o->kind);
    }
}
// What the stuck submissions held: per submission the counts, the biggest draw, every dispatch and the fragment
// shaders used (fs hash = the name tools/shadertool and LARECOMP_SKIP_FSHASH use).
static void hang_report(uint64_t hung) {
    uint64_t from = hang_n > HANG_OPS ? hang_n - HANG_OPS : 0, newest = submit_seq + 1;
    LOG("[hang] stuck on submission %llu; newest %llu (open), %llu operations recorded\n", (unsigned long long)hung,
        (unsigned long long)newest, (unsigned long long)hang_n);
    for (uint64_t s = hung; s <= newest && s < hung + 6; s++) {
        unsigned nd = 0, nc = 0, np = 0, ncp = 0, nfs = 0, nall = 0;
        uint64_t big = 0, bigpos = 0, fs[24];
        unsigned fsn[24];
        for (uint64_t i = from; i < hang_n; i++) {
            const HangOp *o = &hang_ops[i & (HANG_OPS - 1)];
            if (o->seq != s) continue;
            nall++;
            if (o->kind == HOP_DRAW) {
                nd++;
                if ((uint64_t)o->x * o->y >= big) big = (uint64_t)o->x * o->y, bigpos = i;
                unsigned k = 0;
                while (k < nfs && fs[k] != o->a) k++;
                if (k < nfs) fsn[k]++;
                else if (nfs < 24) fs[nfs] = o->a, fsn[nfs++] = 1;
            } else if (o->kind == HOP_DISPATCH) {
                if (nc++ < 16) hang_op_log("  dispatch", i);
            } else if (o->kind == HOP_PASS) np++;
            else if (o->kind == HOP_COPY) ncp++;
        }
        double age = s <= submit_seq && submit_seq - s < 64 ? (SDL_GetTicksNS() - sub_t[s & 63]) / 1e6 : -1;
        LOG("[hang] submission %llu: %u draws, %u dispatches, %u passes, %u copies (%u ring entries; submitted %.0f ms ago)\n",
            (unsigned long long)s, nd, nc, np, ncp, nall, age);
        if (nd) hang_op_log("  biggest draw", bigpos);
        char line[24 * 24 + 8];
        int w = 0;
        for (unsigned k = 0; k < nfs && w < (int)sizeof line - 24; k++) w += snprintf(line + w, sizeof line - w, " %016llx x%u", (unsigned long long)fs[k], fsn[k]);
        if (nfs) LOG("[hang]   fragment shaders:%s%s\n", line, nfs == 24 ? " ..." : "");
    }
    fflush(stdout);
}
// After the device is lost: the driver's last checkpoint per pipeline stage (= ring positions) + what was around them.
static void hang_checkpoints(void) {
    if (!hang_cp) return;
    PFN_vkGetQueueCheckpointDataNV get = (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(g_vk.dev, "vkGetQueueCheckpointDataNV");
    if (!get) return;
    uint32_t n = 0;
    get(g_vk.queue, &n, NULL);
    VkCheckpointDataNV cd[16];
    if (n > 16) n = 16;
    for (uint32_t i = 0; i < n; i++) cd[i] = (VkCheckpointDataNV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV};
    get(g_vk.queue, &n, cd);
    LOG("[hang] driver checkpoints: %u\n", n);
    uint64_t lo = UINT64_MAX, hi = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t m = (uint64_t)(uintptr_t)cd[i].pCheckpointMarker;
        char tag[64];
        snprintf(tag, sizeof tag, "last reached stage %#x", (unsigned)cd[i].stage);
        if (!m) { LOG("[hang] %s: none\n", tag); continue; }
        hang_op_log(tag, m - 1);
        for (uint64_t p = m - 1 > 3 ? m - 4 : 0; p < m - 1; p++) hang_op_log("  before", p);
        for (uint64_t p = m; p < m + 3 && p < hang_n; p++) hang_op_log("  after", p);
        if (m - 1 < lo) lo = m - 1;
        if (m - 1 > hi) hi = m - 1;
    }
    // The marker goes in before its operation, so the culprit is between the oldest stage's mark (bottom of pipe: all
    // work before it finished) and the newest (top of pipe: all work before it started). 10-18: 33 operations.
    if (lo < hi && hi - lo <= 512)
        for (uint64_t p = lo; p <= hi; p++) hang_op_log("  in flight", p);
    fflush(stdout);
}
static void gp_mark(VkCommandBuffer cb, const char *fmt, ...) {
    if (!g_gpuprof_frame || gp_n >= GP_MAX - 1) return;
    if (!gp_qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = GP_MAX};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gp_qp) != VK_SUCCESS) { g_gpuprof_frame = false; return; }
    }
    if (!gp_n) vkCmdResetQueryPool(cb, gp_qp, 0, GP_MAX);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gp_qp, (uint32_t)gp_n);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gp_label[gp_n], sizeof gp_label[0], fmt, ap);
    va_end(ap);
    gp_draws[gp_n] = 0;
    gp_n++;
}
// LARECOMP_GPUPROF_DRAWS=1: the profiled frame also gets a timestamp before every draw, and the report adds the GPU time
// per fragment shader hash (sum over the frame's draws, top 30). Timestamps between draws keep the GPU from overlapping
// them, so the per-draw times are each draw's cost on its own (the frame total grows); rank by them, don't add them up.
static bool gp_per_draw(void) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_GPUPROF_DRAWS") && strcmp(getenv("LARECOMP_GPUPROF_DRAWS"), "0");
    return on;
}
void renderer_flush(void);
void vk_gpuprof_report(void) {
    if (!gp_n) return;
    renderer_flush();
    static uint64_t t[GP_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, gp_qp, 0, (uint32_t)gp_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod / 1e6, total = (t[gp_n - 1] - t[0]) * per;
        LOG("[gpuprof] %d marks, %.2f ms first to last (rows: the work after each mark, until the next)\n", gp_n, total);
        for (int i = 0; i + 1 < gp_n; i++) {
            double ms = (t[i + 1] - t[i]) * per;
            LOG("[gpuprof] %6.3f ms %5.1f%%  %s%s", ms, 100 * ms / (total > 0 ? total : 1), gp_label[i], gp_draws[i] ? "" : "\n");
            if (gp_draws[i]) LOG(", %u draws\n", gp_draws[i]);
        }
        if (gp_per_draw()) {   // per fragment shader: draws, GPU ms (the rows that start with "draw fs")
            typedef struct { uint64_t h; double ms, max; uint32_t n; } GpAgg;
            static GpAgg agg[1024];
            int na = 0;
            double all = 0;
            for (int i = 0; i + 1 < gp_n; i++) {
                unsigned long long h;
                if (sscanf(gp_label[i], "draw fs %llx", &h) != 1) continue;
                double ms = (t[i + 1] - t[i]) * per;
                all += ms;
                int k = 0;
                while (k < na && agg[k].h != h) k++;
                if (k == na) { if (na == 1024) continue; agg[na].h = h; agg[na].ms = agg[na].max = 0; agg[na].n = 0; na++; }
                agg[k].ms += ms; agg[k].n++;
                if (ms > agg[k].max) agg[k].max = ms;
            }
            for (int a = 0; a < na; a++)   // (sort by total, descending)
                for (int b = a + 1; b < na; b++)
                    if (agg[b].ms > agg[a].ms) { GpAgg x = agg[a]; agg[a] = agg[b]; agg[b] = x; }
            LOG("[gpuprof] per fs (draws timed one by one: %.2f ms over %d shaders):\n", all, na);
            for (int a = 0; a < na && a < 30; a++)
                LOG("[gpuprof]   fs %016llx %7.3f ms %5.1f%% %5u draws, max %.3f ms\n", (unsigned long long)agg[a].h, agg[a].ms,
                    100 * agg[a].ms / (all > 0 ? all : 1), agg[a].n, agg[a].max);
        }
    }
    gp_n = 0;
}

// TOTK_TEST_DEVLOST=<present#>[,<present#>...] (device-lost recovery tests): at that presented frame the submission ends
// with one compute dispatch that never finishes (a loop on a push constant that stays 0) = a real GPU hang / TDR.
uint64_t renderer_present_count(void);
static void devlost_test(VkCommandBuffer cb) {
    static uint64_t at[8];
    static int n = -1, next;
    if (n < 0) {
        n = 0;
        for (const char *e = getenv("TOTK_TEST_DEVLOST"); e && *e && n < 8;) {
            char *end;
            at[n++] = strtoull(e, &end, 10);
            if (*end != ',') break;
            e = end + 1;
        }
    }
    if (next >= n || renderer_present_count() < at[next]) return;
    next++;
    static VkPipeline pipe;
    static VkPipelineLayout layout;
    if (!pipe) {
        static const char *src = "#version 450\nlayout(local_size_x = 32) in;\nlayout(push_constant) uniform P { uint go; } p;\n"
                                 "shared uint s;\nvoid main() { uint i = 0u; while (p.go == 0u) { atomicAdd(s, 1u); if (++i == 0xFFFFFFF0u) i = 0u; } }\n";
        size_t nw;
        uint32_t *spv = vk_glsl_to_spirv(src, STAGE_COMPUTE, "devlost_test", &nw);
        if (!spv) return;
        VkShaderModule m = vk_module(spv, nw);
        free(spv);
        VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
        VkPipelineLayoutCreateInfo pl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &pl, NULL, &layout));
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                    .module = m, .pName = "main"},
                                          .layout = layout};
        VK_CHECK(vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe));
        vkDestroyShaderModule(g_vk.dev, m, NULL);
    }
    uint32_t go = 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &go);
    vkCmdDispatch(cb, 1, 1, 1);
    LOG("[vk] TOTK_TEST_DEVLOST: endless dispatch recorded at present %llu (submission %llu)\n",
        (unsigned long long)renderer_present_count(), (unsigned long long)submit_seq + 1);
}

// Submit the pending work (if any). wait/signal: binary semaphores for presentation (may be null).
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra) {
    (void)extra;
    vk_end_pass();
    gdev_check_prev();
    Sub *s = sub();
    if (!s->main_begun && !s->upload_begun && !wait && !signal) return;
    VkCommandBufferSubmitInfo cbs[2];
    int n = 0;
    if (s->upload_begun) {
        gdev_shadow_sync(s->upload);
        vk_full_barrier(s->upload);  // uploads complete before anything in the main buffer reads them
        VK_CHECK(vkEndCommandBuffer(s->upload));
        cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->upload};
    }
    vk_cmd();  // (an empty main buffer still carries the semaphores)
    uint64_t this_seq = submit_seq + 1;
    bool carry = g_chunk_submit && g_fence_seq_mark != this_seq && gdev_carry_ok();
    // (GPU profile: the row of a submission's last draw used to run on to the next submission's start, so GPU idle + the
    // next submission's uploads (its upload buffer runs first) counted as that draw: d9 "38 ms" draws of fs 2a84561f)
    devlost_test(s->main);
    gp_mark(s->main, "(submission end: gdev write-back)");
    if (!carry) gdev_writeback(s->main);
    gp_mark(s->main, "(after the submission: GPU idle + the next one's uploads, until its start)");
    if (s->ts) vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 2 + 1);
    VK_CHECK(vkEndCommandBuffer(s->main));
    cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->main};
    VkSemaphoreSubmitInfo sig[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_vk.timeline,
                                     .value = submit_seq + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = signal,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = wait,
                                .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = (uint32_t)n, .pCommandBufferInfos = cbs,
                        .signalSemaphoreInfoCount = signal ? 2 : 1, .pSignalSemaphoreInfos = sig,
                        .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &wt};
    {
        PROF_T0();
        dl_note("submit %llu", (unsigned long long)submit_seq + 1);
        hang_op(NULL, HOP_SUBMIT, submit_seq + 1, 0, wait ? 1u : 0u, 0, 0);
        extern const char *volatile g_rt_op;
        extern volatile uint64_t g_rt_op_ns;
        vkq_drain();   // (LARECOMP_VKQ: the record thread finishes both command buffers first)
        g_rt_op = "vkQueueSubmit2", g_rt_op_ns = SDL_GetTicksNS();
        VkResult sr = vkQueueSubmit2(g_vk.queue, 1, &si, VK_NULL_HANDLE);
        g_rt_op = NULL;
        if (sr == VK_ERROR_DEVICE_LOST) {   // (the hang path above can miss a reset: name the stuck operation here too)
            vk_devlost_dump();
            hang_report(vk_completed_seq() + 1);
            hang_checkpoints();   // (VK_CHECK below adds the fault report)
        }
        VK_CHECK(sr);
        PROF_ADD(prof_submit);
    }
    pthread_mutex_lock(&seq_lock);
    submit_seq++;
    sub_t[submit_seq & 63] = SDL_GetTicksNS();
    pthread_mutex_unlock(&seq_lock);
    if (carry) gdev_carry(this_seq, submit_seq + 1);
    cur_sub = (cur_sub + 1) % SUBS;
    chunk_draws = 0;
    stat_flushes++;
    cum_submits++;
    ring_retire();
    vk_collect_dead();
}

void renderer_submit(void) { vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE); }
// At a game fence (nvnQueueFenceSync): end the open submission. The game rewrites its buffers once a SyncWait on the fence
// returns, and with SYNC_LAG that could happen while the fence's submission was still open: its later lists then shared the
// per-submission state made for the earlier ones (guest copies, device-window regions, mirror uploads) = UI text drawn with
// the previous layout for single frames (flickcheck run532 444 minimap-edge flips, SYNC_LAG=0 run535 0). With this, one
// submission only holds data the game cannot have rewritten yet. LARECOMP_FENCE_SUBMIT=0: old.
void renderer_fence_submit(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_FENCE_SUBMIT"); on = !(e && *e == '0'); }
    if (!on) return;
    backend_exec_lock();
    renderer_submit();
    backend_exec_unlock();
}

void renderer_flush(void) {
    renderer_submit();
    vk_wait_seq(submit_seq);
}

void renderer_sync(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    vk_wait_seq(s);
}

uint64_t renderer_fence_seq(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t q = submit_seq + (subs[cur_sub].main_begun || subs[cur_sub].upload_begun ? 1 : 0);
    pthread_mutex_unlock(&seq_lock);
    return q;
}

static uint64_t stat_sync_early;  // SyncWaits on a fence whose submission was still open (returned before it ran)

// LARECOMP_RACECHECK=1 (2026-10-20, Deck HUD chaos and rare 3D pops): guest memory the GPU reads in place (uniforms with
// UBO_ZEROCOPY, heap vertex / index data on an iGPU, the alias window without its mirror) is hashed when its draw is
// recorded and again when a SyncWait has seen its submission done, before the game is released. A change in between = the
// game rewrote data the GPU may still have been reading (the SyncWait lag returns one fence early). GPU writes into guest
// memory by later submissions also count, so compare with LARECOMP_SYNC_LAG=0 (the floor). Logs "[race]" lines.
// "cpu" = changes no shader write of a submission in between covers (gw_overlap): the guest CPU rewrote it in flight.
enum { RC_N = 1 << 16, RC_DEDUP = 4096, RC_K = 4 };
static int rc_on = -1;
static struct { uint64_t addr, size, seq, h, d0; int kind; } rc_e[RC_N];
static uint64_t rc_head, rc_tail, rc_n[RC_K], rc_chg[RC_K], rc_chg_b[RC_K], rc_cpu[RC_K], rc_drop, rc_waits;
static pthread_mutex_t rc_lock = PTHREAD_MUTEX_INITIALIZER;
static const char *const rc_kind[RC_K] = {"uniform", "vertex/index", "alias window", "mirror GPU-copied page"};
static int gw_overlap(uint64_t addr, uint64_t size, uint64_t lo, uint64_t hi);
static uint64_t rc_hash(uint64_t addr, uint64_t size) {   // every word up to 16 KB, else 2048 spread over the range
    const uint8_t *p = (const uint8_t *)(uintptr_t)addr;
    uint64_t h = size * 0x9E3779B97F4A7C15ull, w, n = size / 8, m = n < 2048 ? n : 2048;
    for (uint64_t k = 0; k < m; k++) {
        memcpy(&w, p + (n <= 2048 ? k : k * (n - 1) / (m - 1)) * 8, 8);
        h = (h ^ w) * 0x100000001B3ull;
    }
    return h;
}
static void rc_record(int kind, uint64_t addr, uint64_t size) {   // (render thread)
    if (rc_on < 0) {
        const char *e = getenv("LARECOMP_RACECHECK");
        rc_on = e && strcmp(e, "0");
        if (rc_on) LOG("[race] in-place guest data checked at each SyncWait (LARECOMP_RACECHECK)\n");
    }
    if (!rc_on || size < 8) return;
    static struct { uint64_t addr, size, seq; } dd[RC_DEDUP];   // once per range and submission
    uint64_t seq = vk_pending_seq(), k = addr * 0x9E3779B97F4A7C15ull ^ size;
    uint32_t sl = (uint32_t)(k ^ k >> 29) & (RC_DEDUP - 1);
    if (dd[sl].addr == addr && dd[sl].size == size && dd[sl].seq == seq) return;
    dd[sl].addr = addr; dd[sl].size = size; dd[sl].seq = seq;
    uint64_t h = rc_hash(addr, size), d0 = vk_completed_seq();
    pthread_mutex_lock(&rc_lock);
    if (rc_head - rc_tail == RC_N) rc_drop++;
    else rc_e[rc_head++ % RC_N] = (typeof(rc_e[0])){addr, size, seq, h, d0, kind};
    pthread_mutex_unlock(&rc_lock);
}
static void rc_check(uint64_t done) {   // (game thread inside SyncWait: submissions <= done have finished on the GPU)
    if (rc_on <= 0) return;
    static int nlog;
    pthread_mutex_lock(&rc_lock);
    while (rc_tail < rc_head && rc_e[rc_tail % RC_N].seq <= done) {
        typeof(rc_e[0]) *e = &rc_e[rc_tail++ % RC_N];
        rc_n[e->kind]++;
        if (rc_hash(e->addr, e->size) == e->h) continue;
        rc_chg[e->kind]++;
        rc_chg_b[e->kind] += e->size;
        int gw = gw_overlap(e->addr, e->size, e->d0, done);   // (record thread's ring read from here: diagnostic only)
        if (!gw) rc_cpu[e->kind]++;
        if (nlog++ < 60)
            LOG("[race] frame %llu: %s %llx +%llx (submission %llu) changed before the GPU had finished it (%s)\n",
                (unsigned long long)renderer_present_count(), rc_kind[e->kind], (unsigned long long)e->addr,
                (unsigned long long)e->size, (unsigned long long)e->seq, gw > 0 ? "shader-written bytes" : gw ? "unknown" : "cpu");
    }
    if (++rc_waits % 600 == 0) {
        LOG("[race] frame %llu, last 600 SyncWaits: uniform %llu changed (cpu %llu) of %llu (%llu KB) | vertex/index %llu (cpu %llu) "
            "of %llu (%llu KB) | alias %llu (cpu %llu) of %llu (%llu KB) | mirror gpu page %llu (cpu %llu) of %llu (%llu KB) | dropped %llu\n",
            (unsigned long long)renderer_present_count(),
            (unsigned long long)rc_chg[0], (unsigned long long)rc_cpu[0], (unsigned long long)rc_n[0], (unsigned long long)(rc_chg_b[0] >> 10),
            (unsigned long long)rc_chg[1], (unsigned long long)rc_cpu[1], (unsigned long long)rc_n[1], (unsigned long long)(rc_chg_b[1] >> 10),
            (unsigned long long)rc_chg[2], (unsigned long long)rc_cpu[2], (unsigned long long)rc_n[2], (unsigned long long)(rc_chg_b[2] >> 10),
            (unsigned long long)rc_chg[3], (unsigned long long)rc_cpu[3], (unsigned long long)rc_n[3], (unsigned long long)(rc_chg_b[3] >> 10),
            (unsigned long long)rc_drop);
        memset(rc_n, 0, sizeof rc_n); memset(rc_chg, 0, sizeof rc_chg); memset(rc_chg_b, 0, sizeof rc_chg_b);
        memset(rc_cpu, 0, sizeof rc_cpu); rc_drop = 0;
    }
    pthread_mutex_unlock(&rc_lock);
}

void renderer_wait_seq(uint64_t seq) {
    if (!seq) return;
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    if (seq > s) {
        stat_sync_early++;
        static int nlog;
        if (nlog++ < 12) LOG("[vk] SyncWait on open submission %llu (submitted %llu): returns early\n", (unsigned long long)seq,
                             (unsigned long long)s);
    }
    uint64_t w = seq > s ? s : seq, t0 = SDL_GetTicksNS();
    // LARECOMP_LOG_SYNCWAIT=<frame>: from that presented frame on, 40 lines "[syncwait] frame F seq W (newest submitted S,
    // GPU done C) age A ms wait B ms" = which submission the game waits for and how far behind the GPU is.
    static int64_t lsw_from = -2;
    static int lsw_n;
    if (lsw_from == -2) { const char *e = getenv("LARECOMP_LOG_SYNCWAIT"); lsw_from = e ? atoll(e) : -1; }
    uint64_t done0 = lsw_from >= 0 ? vk_completed_seq() : 0;
    if (w && vk_completed_seq() < w) {
        sw_n++;
        sw_age_ns += s - w < 64 ? t0 - sub_t[w & 63] : 0;
        vk_wait_seq(w);  // not submitted yet (other thread): everything submitted so far
        sw_wait_ns += SDL_GetTicksNS() - t0;
    }
    if (lsw_from >= 0 && renderer_present_count() >= (uint64_t)lsw_from && lsw_n < 40) {
        lsw_n++;
        LOG("[syncwait] frame %llu seq %llu (asked %llu, newest submitted %llu, GPU done %llu) age %.2f ms wait %.2f ms\n",
            (unsigned long long)renderer_present_count(), (unsigned long long)w, (unsigned long long)seq, (unsigned long long)s,
            (unsigned long long)done0, s - w < 64 ? (t0 - sub_t[w & 63]) / 1e6 : -1.0, (SDL_GetTicksNS() - t0) / 1e6);
    }
    rc_check(w);
}

VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) {
    Sub *s = sub();
    for (;;) {
        if (s->cur_pool == s->npools) {
            if (s->npools == 16) FATAL("descriptor pools exhausted\n");
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768},
                                            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
                                            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 32768}};
            VkDescriptorPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8192,
                                             .poolSizeCount = 6, .pPoolSizes = sizes};
            VK_CHECK(vkCreateDescriptorPool(g_vk.dev, &ci, NULL, &s->pools[s->npools++]));
        }
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = s->pools[s->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &layout};
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(g_vk.dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        s->cur_pool++;
    }
}

// ================================================================== transient memory ring
// 64 MB host-visible chunks (device-local when the GPU exposes resizable BAR); a chunk is reused once every
// submission that allocated from it has completed.
#define CHUNK_SIZE (64ull << 20)
typedef struct { VkBuffer buf; VmaAllocation mem; uint8_t *ptr; uint64_t size, used, seq; } Chunk;
static Chunk *chunks;
static int nchunks, active = -1;

static int new_chunk(uint64_t size) {
    chunks = realloc(chunks, (nchunks + 1) * sizeof *chunks);
    Chunk *c = &chunks[nchunks];
    memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    // Integrated GPU (Steam Deck): host memory (GTT, write-combined). d17 2026-10-16: the ring grew to ~18 chunks (1.1 GB) in the
    // "device-local" heap, which there is a 1 GB VRAM carve-out + GTT placed first come first served (VRAM asked 3.1 GB, 985 MB
    // resident), so one-shot upload data competed with render targets for VRAM. LARECOMP_VK_CHUNK_DEVICE=1: device-local as before.
    static int chunk_host = -1;
    if (chunk_host < 0)
        chunk_host = g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && !getenv("LARECOMP_VK_CHUNK_DEVICE");
    VmaAllocationCreateInfo ai = {.usage = chunk_host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info));
    if (nchunks == 0) LOG("[vk] transient ring chunks: memory type %u (%s)\n", info.memoryType, chunk_host ? "host, integrated GPU" : "device-local preferred");
    c->ptr = info.pMappedData;
    c->size = size;
    return nchunks++;
}

static void ring_retire(void) {}  // chunks are recycled lazily in vk_transient

VkSpan vk_transient(uint64_t size, uint64_t align) {
    if (align < 16) align = 16;
    uint64_t pending = vk_pending_seq();
    if (active >= 0) {
        Chunk *c = &chunks[active];
        uint64_t off = (c->used + align - 1) & ~(align - 1);
        if (off + size <= c->size) {
            c->used = off + size;
            c->seq = pending;
            stat_upload_bytes += size;
            stat_up_kind[vk_up_kind] += size;
            if (vk_up_kind == 4) cum_tex_up += size;
            return (VkSpan){c->buf, off, c->ptr + off};
        }
    }
    uint64_t done = vk_completed_seq();
    int pick = -1;
    bool keep_list = in_execute && chunk_limit();  // (guest_copy shares copies across the chunks of the list)
    for (int i = 0; i < nchunks; i++)
        if (i != active && chunks[i].seq <= done && chunks[i].size >= size && !(keep_list && chunks[i].seq >= exec_start_seq)) { pick = i; break; }
    if (pick < 0) pick = new_chunk(size > CHUNK_SIZE ? size : CHUNK_SIZE);
    active = pick;
    chunks[pick].used = size;
    chunks[pick].seq = pending;
    stat_upload_bytes += size;
    stat_up_kind[vk_up_kind] += size;
    if (vk_up_kind == 4) cum_tex_up += size;
    return (VkSpan){chunks[pick].buf, 0, chunks[pick].ptr};
}

// GPU -> CPU copies: the ring is write-combined (often device-local over the BAR), where CPU reads crawl (a
// 1280x720 screenshot took 7 s). Readbacks use one host-cached buffer instead. Every user waits for the GPU and
// reads the data before the next readback (render thread), so a single grow-only buffer suffices.
VkSpan vk_readback(uint64_t size) {
    static VkBuffer buf;
    static VmaAllocation mem;
    static uint8_t *ptr;
    static uint64_t cap;
    if (size > cap) {
        if (buf) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, buf, mem); }
        cap = (size + (16 << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &buf, &mem, &info));
        ptr = info.pMappedData;
    }
    return (VkSpan){buf, 0, ptr};
}

// (VK_PROF) who allocates "other" (kind 0) upload bytes: caller return address, as an offset from vk_transient_copy
static struct { uintptr_t ra; uint64_t bytes, n; } up0_site[16];
static void up0_note(uintptr_t ra, uint64_t size) {
    int i = 0;
    for (; i < 16 && up0_site[i].ra && up0_site[i].ra != ra; i++) {}
    if (i == 16) return;
    up0_site[i].ra = ra; up0_site[i].bytes += size; up0_site[i].n++;
}
VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align) {
    if (prof > 0 && vk_up_kind == 0) up0_note((uintptr_t)__builtin_return_address(0), size);
    VkSpan s = vk_transient(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}

// Per-draw constants that usually repeat the previous draw's (scale table, tint, driver constants): reuse that copy
// while it's identical, within one submission. slot: one per kind of data.
enum { DEDUPE_RSCALE_V, DEDUPE_RSCALE_F, DEDUPE_TINT, DEDUPE_DRV_V, DEDUPE_DRV_F, DEDUPE_DRV_C, DEDUPE_SLOTS };
static VkSpan copy_dedupe(int slot, const void *data, uint64_t size, uint64_t align) {
    static struct { uint64_t seq, size; uint8_t data[0x800]; VkSpan span; } last[DEDUPE_SLOTS];
    uint64_t seq = vk_pending_seq();
    if (size > sizeof last[0].data) return vk_transient_copy(data, size, align);
    if (last[slot].seq == seq && last[slot].size == size && !memcmp(last[slot].data, data, size)) return last[slot].span;
    VkSpan s = vk_transient_copy(data, size, align);
    last[slot].seq = seq; last[slot].size = size; memcpy(last[slot].data, data, size); last[slot].span = s;
    return s;
}

// LARECOMP_VX_STATS=1 (with VK_PROF): what the vertex copies are, per 4 KB heap page - copied already in this submission,
// unchanged since an earlier submission copied it (a VRAM mirror would skip it), or changed / new. "Unchanged" is sampled
// (32 words per page), so it can over-count pages changed only between the samples.
uint64_t prof_vxp_dup, prof_vxp_same, prof_vxp_new;
static void vx_page_stats(uint64_t addr, uint64_t size, uint64_t pend) {
    static int on = -1;
    static uint64_t *pseq, *phash, npages;
    if (on < 0) {
        on = getenv("LARECOMP_VX_STATS") != NULL;
        npages = (4ull << 30) >> 12;  // (heap import <= 4 GB)
        if (on) { pseq = calloc(npages, 8); phash = calloc(npages, 8); on = pseq && phash; }
    }
    if (!on || addr < HEAP_BASE) return;
    for (uint64_t a = addr, end = addr + size; a < end;) {
        uint64_t p = (a - HEAP_BASE) >> 12, pe = HEAP_BASE + ((p + 1) << 12), n = (pe < end ? pe : end) - a;
        if (p >= npages) return;
        if (pseq[p] == pend) prof_vxp_dup += n;
        else {
            const uint64_t *w = (const uint64_t *)(uintptr_t)(HEAP_BASE + (p << 12));
            uint64_t h = 0;
            for (int i = 0; i < 512; i += 16) h = (h ^ w[i]) * 0x9E3779B97F4A7C15ull;
            if (pseq[p] && phash[p] == h) prof_vxp_same += n; else prof_vxp_new += n;
            pseq[p] = pend; phash[p] = h;
        }
        a += n;
    }
}

// Guest memory copied once per submission: draws binding the same range share the copy.
#define GCOPY_TAB 4096
static struct { uint64_t addr, size, seq, used; VkSpan span; } gcopy[GCOPY_TAB];
static uint64_t gcopy_gen;  // bumped to drop every shared copy (guest memory written mid-list)
static void guest_copy_flush(void) { gcopy_gen++; }
// With chunked submission the copies are shared by every chunk of one guest list (nothing writes guest memory within a
// list, gcopy_gen otherwise): the key is the list, and a reuse in a later chunk keeps the ring chunk alive until then.
static VkSpan guest_copy(uint64_t addr, uint64_t size) {
    uint64_t pend = vk_pending_seq();
    // An inline UpdateUniformBuffer bumps gcopy_gen. In zero-copy mode every buffer it ever wrote is in zc_excl, so copies of
    // other ranges cannot go stale and keep being shared (d55_dbg: ~281 inline updates per frame re-copied a 64 KB block of
    // shader constants 132 times = 8 MB per frame).
    uint64_t gen = gcopy_gen;
    if (ubo_zc > 0) {
        gen = 0;
        for (int i = 0; i < zc_nexcl; i++) if (addr < zc_excl[i].hi && addr + size > zc_excl[i].lo) { gen = gcopy_gen; break; }
    }
    // Shared per guest list (LARECOMP_GCOPY_PER_SUB=1: per submission, old): a submission can hold several lists, and the
    // game rewrites its buffers between them once a SyncWait returns (SYNC_LAG: it can return while that submission is
    // still open) - a later list then drew with the earlier list's copy (minimap text with other strings' layout, run538).
    static int per_sub = -1;
    if (per_sub < 0) per_sub = getenv("LARECOMP_GCOPY_PER_SUB") != NULL;
    uint64_t seq = (in_execute && (!per_sub || chunk_limit()) ? (alias_epoch | (1ull << 39)) : pend) + (gen << 40),
             h = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (GCOPY_TAB - 1);
    if (gcopy[slot].seq == seq && gcopy[slot].addr == addr && gcopy[slot].size == size) {
        if (prof) prof_gc_hit++;
        if (gcopy[slot].used != pend) {
            gcopy[slot].used = pend;
            for (int i = 0; i < nchunks; i++)
                if (chunks[i].buf == gcopy[slot].span.buf && chunks[i].seq < pend) { chunks[i].seq = pend; break; }
        }
        return gcopy[slot].span;
    }
    if (prof) {   // (a slot already holding this list's copy of another range = a hash collision evicting it)
        if (gcopy[slot].seq == seq) prof_gc_evict++, prof_gc_evict_b += size; else prof_gc_new++, prof_gc_new_b += size;
    }
    VkSpan s = vk_transient_copy((const void *)(uintptr_t)addr, size, 256);
    gcopy[slot].addr = addr; gcopy[slot].size = size; gcopy[slot].seq = seq; gcopy[slot].span = s; gcopy[slot].used = pend;
    if (prof && vk_up_kind == 2) vx_page_stats(addr, size, pend);
    return s;
}

// ================================================================== worker threads (shader / pipeline builds)
// First-use compiles run off the render thread; draws needing them are skipped until ready (as on Metal).
// LARECOMP_SYNC_SHADERS=1 builds on first use, blocking (deterministic tests).
typedef struct Job { void (*fn)(void *); void *arg; struct Job *next; } Job;
// The game's own translations / builds queued or running: the precompile workers pause while any are (pc_yield), so a
// shader the game is waiting for never queues behind the backlog (Deck run 2026-10-16: 2 precompile threads on 4 cores,
// "program not built" draws for minutes, frames held = 0.5 fps on screen).
static _Atomic int g_jobs_pending;
static Job *jobs_head, *jobs_tail;
static pthread_mutex_t jobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jobs_cond = PTHREAD_COND_INITIALIZER;

static void *worker(void *a) {
    (void)a;
#ifndef _WIN32
    { extern int pthread_setname_np(pthread_t, const char *); pthread_setname_np(pthread_self(), "vk-pipeline"); }   // ([cpu] line)
#endif
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!jobs_head) pthread_cond_wait(&jobs_cond, &jobs_lock);
        Job *j = jobs_head;
        jobs_head = j->next;
        if (!jobs_head) jobs_tail = NULL;
        pthread_mutex_unlock(&jobs_lock);
        j->fn(j->arg);
        free(j);
        atomic_fetch_sub(&g_jobs_pending, 1);
    }
    return NULL;
}

static bool g_bake_sync;  // draw_(): the draw renders into a 3D / BC target (one-shot bake): its program builds now
static bool sync_shaders(void) {
    static int v = -1;
    if (v < 0) v = getenv("LARECOMP_SYNC_SHADERS") != NULL;
    return v || g_bake_sync;
}

// Shader translation: one dedicated thread (the translator isn't re-entrant - translate.c's RSC2 string ring - and
// translating on the game thread cost up to ~90 ms per frame on entering an area). Compiles go on to the workers.
static pthread_mutex_t translate_lock = PTHREAD_MUTEX_INITIALIZER;  // (compute shaders translate on the game thread)
// maxwell_translate (under translate_lock) with the stage's compiler constants (c1, csize bytes after the code) in view:
// translate.c reads BRX jump tables from them. csize 0 (archive programs have only their code): every slot a possible target.
static ShaderInfo *translate_stage(uint64_t code, uint64_t cmax, uint64_t csize, int stage, const ShaderKey *key) {
    maxwell_const_data = csize ? (const uint8_t *)(uintptr_t)(code + cmax) : NULL;
    maxwell_const_size = (size_t)csize;
    ShaderInfo *si = maxwell_translate((const uint8_t *)(uintptr_t)code, (size_t)cmax, stage, key);
    maxwell_const_data = NULL;
    maxwell_const_size = 0;
    return si;
}
static Job *tjobs_head, *tjobs_tail;
static pthread_mutex_t tjobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tjobs_cond = PTHREAD_COND_INITIALIZER;
static void *translator(void *a) {
    (void)a;
#ifndef _WIN32
    { extern int pthread_setname_np(pthread_t, const char *); pthread_setname_np(pthread_self(), "vk-translate"); }   // ([cpu] line)
#endif
    for (;;) {
        pthread_mutex_lock(&tjobs_lock);
        while (!tjobs_head) pthread_cond_wait(&tjobs_cond, &tjobs_lock);
        Job *j = tjobs_head;
        tjobs_head = j->next;
        if (!tjobs_head) tjobs_tail = NULL;
        pthread_mutex_unlock(&tjobs_lock);
        j->fn(j->arg);
        free(j);
        atomic_fetch_sub(&g_jobs_pending, 1);
    }
    return NULL;
}
static bool sync_shaders(void);
static void run_translate(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) { started = true; pthread_t t; pthread_create(&t, NULL, translator, NULL); }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    atomic_fetch_add(&g_jobs_pending, 1);
    pthread_mutex_lock(&tjobs_lock);
    if (tjobs_tail) tjobs_tail->next = j; else tjobs_head = j;
    tjobs_tail = j;
    pthread_cond_signal(&tjobs_cond);
    pthread_mutex_unlock(&tjobs_lock);
}

static void run_job(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) {
        started = true;
        // first entry into an area builds dozens of pipelines at once: use most of the CPU (draws wait for them)
        int n = SDL_GetNumLogicalCPUCores() - 4;
        if (n < 4) n = 4;
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) { pthread_t t; pthread_create(&t, NULL, worker, NULL); }
    }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    atomic_fetch_add(&g_jobs_pending, 1);
    pthread_mutex_lock(&jobs_lock);
    if (jobs_tail) jobs_tail->next = j; else jobs_head = j;
    jobs_tail = j;
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}

// ================================================================== state
typedef struct {
    NvnTexture *color[8];
    NvnTextureView cview[8];
    uint8_t has_cview[8];
    int ncolor;
    NvnTexture *depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtState;

typedef struct { uint64_t addr, size; } Range;

static struct {
    RtState rt;
    bool rt_dirty;
    NvnProgram *program;
    uint8_t blend[8][8];
    uint32_t channel_mask;
    uint8_t color_state[4], ds[8], ms[0x18], poly[4];
    float poly_offset[3];
    uint8_t depth_clamp;  // nvnCommandBufferSetDepthClamp
    uint32_t attrib[16];
    int nattrib;
    uint32_t stream[16][2];
    int nstream;
    Range vbuf[16], ubuf[6][18], sbuf[6][16];
    uint64_t tex[6][32], img[6][8];
    NvnDescPool *tpool, *spool;
    float viewport[16][4];
    int32_t scissor[16][4];
    float depth_range[16][2];
    float blend_color[4];
    uint32_t stencil_ref[2], stencil_mask[2], stencil_vmask[2];
    bool prim_restart;
    uint32_t restart_index;
    uint64_t copy_row_stride, copy_image_stride;
} st;

uint32_t nvn_window_origin(void);
uint32_t nvn_depth_mode(void);
bool renderer_origin_lower_left(void) { return nvn_window_origin() == 0; }

// Called once per presented frame: frames over 50 ms are logged with where their time went (always on since 2026-10-18,
// user: stutters in play "w the new shader precompile ... or other stuff like texture streaming"; was LARECOMP_VK_PROF only).
// present_ns: time in backend_present's acquire/blit/present. "other" = the game's own threads (game file reads = streaming).
static uint32_t g_drs_w, g_drs_h;   // this frame's G-buffer size (begin_pass: a pass with >= 4 colour targets)
void vk_prof_frame(uint64_t frame, uint64_t present_ns) {
    static uint64_t last_t, l_exec, l_wait, l_draws, l_subs, l_tex, l_fs, l_img, l_img_ns, l_op[CMD_COUNT], l_opc[CMD_COUNT], l_cpu, l_ivcsw;
    static int l_progs;
    extern uint64_t cum_img_n, cum_img_ns;   // (textures.c create_image)
    extern unsigned long long hle_fs_read_bytes(void);
    uint64_t now = SDL_GetTicksNS(), fs = hle_fs_read_bytes();
    int progs = atomic_load(&cum_prog_builds);
    static uint64_t hitch_ns;  // LARECOMP_HITCH_MS: log threshold (default 50 ms; 20 shows the late frames of a 60 fps run)
    if (!hitch_ns) hitch_ns = (getenv("LARECOMP_HITCH_MS") ? (uint64_t)atoi(getenv("LARECOMP_HITCH_MS")) : 50) * 1000000ull;
    if (last_t && now - last_t > hitch_ns) {
        uint64_t dt = now - last_t, ex = cum_exec - l_exec, wt = cum_wait - l_wait;
        LOG("[hitch] frame %llu: %.0f ms = backend exec %.1f (of which GPU waits %.1f) + present %.1f + other %.1f; %llu draws, "
            "%llu submits, compiles %d pending, new programs %d, texture upload %.1f MB, game file reads %.1f MB, images %llu (%.1f ms)\n",
            (unsigned long long)frame, dt / 1e6, ex / 1e6, wt / 1e6, present_ns / 1e6,
            ((double)dt - ex - present_ns) / 1e6, (unsigned long long)(cum_draws - l_draws), (unsigned long long)(cum_submits - l_subs),
            atomic_load(&stat_compiles), progs - l_progs, (cum_tex_up - l_tex) / 1048576.0, (fs - l_fs) / 1048576.0,
            (unsigned long long)(cum_img_n - l_img), (cum_img_ns - l_img_ns) / 1e6);
#ifndef _WIN32
        LOG("[hitch] frame %llu exec on-CPU %.1f of %.1f ms, %llu involuntary switches\n", (unsigned long long)frame,
            (cum_exec_cpu - l_cpu) / 1e6, ex / 1e6, (unsigned long long)(cum_exec_ivcsw - l_ivcsw));
#else
        if (prof) LOG("[hitch] frame %llu exec on-CPU %.1f of %.1f ms\n", (unsigned long long)frame, (cum_exec_cpu - l_cpu) / 1e6, ex / 1e6);
#endif
        if (prof) {   // this frame's heaviest command kinds (nvn.h CMD_* index), for the periodic exec-side hitches (d84)
            uint64_t d[CMD_COUNT];
            for (int i = 0; i < CMD_COUNT; i++) d[i] = op_ns[i] >= l_op[i] ? op_ns[i] - l_op[i] : op_ns[i];   // op_ns resets every 2 s
            LOG("[hitch] frame %llu ops (kind:ms/count):", (unsigned long long)frame);
            for (int rank = 0; rank < 5; rank++) {
                int best = -1;
                for (int i = 0; i < CMD_COUNT; i++) if (d[i] && (best < 0 || d[i] > d[best])) best = i;
                if (best < 0) break;
                LOG(" %d:%.1f/%llu", best, d[best] / 1e6, (unsigned long long)(op_counts[best] - l_opc[best]));
                d[best] = 0;
            }
            LOG("\n");
        }
    }
    if (prof) { memcpy(l_op, op_ns, sizeof l_op); memcpy(l_opc, op_counts, sizeof l_opc); }
    l_cpu = cum_exec_cpu; l_ivcsw = cum_exec_ivcsw;
    l_tex = cum_tex_up; l_fs = fs; l_progs = progs; l_img = cum_img_n; l_img_ns = cum_img_ns;
    {   // "[drs]": the 3D resolution (G-buffer size, begin_pass) whenever it changes, for the game's dynamic resolution
        static uint32_t lw, lh;
        static int n;
        if (g_drs_w && (g_drs_w != lw || g_drs_h != lh) && n < 500) {
            LOG("[drs] frame %llu: 3D %ux%u\n", (unsigned long long)frame, g_drs_w, g_drs_h);
            n++; lw = g_drs_w; lh = g_drs_h;
        }
        g_drs_w = g_drs_h = 0;
    }
    // LARECOMP_GPUPROF_SLOW=<ms>: GPU-profile the frame after one slower than <ms> (up to 4 times, >= 120 frames apart;
    // device.c also saves the image and a [mem] report). For the Deck's intermittent ~200 ms GPU frames (d1/d2 2026-10-16).
    static long slow_ms = -1;
    static int slow_n;
    static uint64_t slow_last;
    if (slow_ms < 0) slow_ms = getenv("LARECOMP_GPUPROF_SLOW") ? atol(getenv("LARECOMP_GPUPROF_SLOW")) : 0;
    if (slow_ms > 0 && last_t && now - last_t > (uint64_t)slow_ms * 1000000ull && slow_n < 4 && (!slow_n || frame > slow_last + 120)) {
        extern volatile int g_gpuprof_request;
        slow_n++;
        slow_last = frame;
        g_gpuprof_request = 1;
        LOG("[gpuprof] frame %llu took %.0f ms: profiling the next frame (slow trigger %d of 4)\n", (unsigned long long)frame,
            (now - last_t) / 1e6, slow_n);
    }
    last_t = now; l_exec = cum_exec; l_wait = cum_wait; l_draws = cum_draws; l_subs = cum_submits;
}

static void fbc_report(void);
const char *renderer_stats(void) {
    static int dumps;
    if (dumps++ < 3) {
        LOG("[vk] command counts:");
        for (int i = 0; i < CMD_COUNT; i++) if (op_counts[i]) LOG(" %d:%llu", i, (unsigned long long)op_counts[i]);
        LOG("\n");
    }
    snprintf(stats_text, sizeof stats_text,
             "%llu draws, %llu skipped, %llu passes, %llu dispatches, %llu flushes, wait %.0fms, compile %d, fbcopy %llu (live barriers %llu, depth WAR %llu, self-write %llu, alias %llu), upload %.1fMB (u %.0f c0 %.0f c1 %.0f z %.0f v %.0f i %.0f t %.0f o %.0f), gdev %d regions %.1fMB in (snapshot %.1f, host fallbacks %llu)",
             (unsigned long long)stat_draws, (unsigned long long)stat_skipped, (unsigned long long)stat_passes,
             (unsigned long long)stat_dispatches, (unsigned long long)stat_flushes, (double)stat_wait_ns / 1e6,
             atomic_exchange(&stat_compiles, 0), (unsigned long long)stat_feedback_copies, (unsigned long long)stat_feedback_live, (unsigned long long)stat_depth_war, (unsigned long long)stat_fb_selfw, (unsigned long long)stat_fb_alias, (double)stat_upload_bytes / 1048576.0, stat_up_kind[1] / 1048576.0, stat_up_kind[5] / 1048576.0, stat_up_kind[6] / 1048576.0,
             stat_up_kind[7] / 1048576.0, stat_up_kind[2] / 1048576.0, stat_up_kind[3] / 1048576.0,
             stat_up_kind[4] / 1048576.0, stat_up_kind[0] / 1048576.0,
             gdev_n, (double)gdev_stat_in / 1048576.0, (double)gdev_stat_snap / 1048576.0, (unsigned long long)gdev_stat_host);
    stat_feedback_copies = stat_feedback_live = stat_depth_war = stat_fb_selfw = stat_fb_alias = stat_upload_bytes = 0;
    memset(stat_up_kind, 0, sizeof stat_up_kind);
    gdev_stat_in = gdev_stat_snap = gdev_stat_host = 0;
    {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, ", pass barriers %llu (skipped %llu; game barriers %llu, %llu mid-pass breaks; feedback elided %llu; hazard barriers %llu)",
                 (unsigned long long)stat_pass_barriers, (unsigned long long)stat_pass_nobarrier, (unsigned long long)stat_game_barriers,
                 (unsigned long long)stat_game_barrier_breaks, (unsigned long long)stat_fb_elided, (unsigned long long)stat_hazard_barriers);
        stat_pass_barriers = stat_pass_nobarrier = stat_game_barriers = stat_game_barrier_breaks = stat_fb_elided = stat_hazard_barriers = 0;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | game barrier bits (in pass %llu):", (unsigned long long)stat_bar_inpass);
        stat_bar_inpass = 0;
        for (int k = 0; k < 4; k++) {   // (top 4 values)
            int best = -1;
            for (int b = 0; b < 512; b++) if (stat_bar_bits[b] && (best < 0 || stat_bar_bits[b] > stat_bar_bits[best])) best = b;
            if (best < 0) break;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " %#x x%llu", best, (unsigned long long)stat_bar_bits[best]);
            stat_bar_bits[best] = 0;
        }
        memset(stat_bar_bits, 0, sizeof stat_bar_bits);
        n = strlen(stats_text);
        { extern void vkq_stats(char *out, size_t n); vkq_stats(stats_text + n, sizeof stats_text - n); n = strlen(stats_text); }
        if (getenv("LARECOMP_COUNT_FSHASH")) {
            snprintf(stats_text + n, sizeof stats_text - n, " | counted fs draws %llu", (unsigned long long)stat_count_fs);
            stat_count_fs = 0;
            n = strlen(stats_text);
        }
        snprintf(stats_text + n, sizeof stats_text - n, " | pass barrier causes:");
        for (int k = 0; k < 6; k++) {   // (top 6: exec.c line of the vk_end_pass, game = game barrier, start, forced)
            int best = -1;
            for (int b = 0; b < 8192; b++) if (stat_pbn_cause[b] && (best < 0 || stat_pbn_cause[b] > stat_pbn_cause[best])) best = b;
            if (best < 0) break;
            n = strlen(stats_text);
            if (best == 1) snprintf(stats_text + n, sizeof stats_text - n, " game x%u", stat_pbn_cause[best]);
            else if (best == 2) snprintf(stats_text + n, sizeof stats_text - n, " start x%u", stat_pbn_cause[best]);
            else if (best == 3) snprintf(stats_text + n, sizeof stats_text - n, " forced x%u", stat_pbn_cause[best]);
            else snprintf(stats_text + n, sizeof stats_text - n, " L%d x%u", best, stat_pbn_cause[best]);
            stat_pbn_cause[best] = 0;
        }
        memset(stat_pbn_cause, 0, sizeof stat_pbn_cause);
    }
    if (stat_sync_early) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, ", early syncwaits %llu", (unsigned long long)stat_sync_early);
        stat_sync_early = 0;
    }
    {   // programs the game asked for that no precompile had built: translate / SPIR-V + modules time (first-sight cost)
        extern _Atomic int stat_prog_builds;
        extern _Atomic uint64_t stat_prog_translate_ns, stat_prog_compile_ns;
        int nb = atomic_exchange(&stat_prog_builds, 0);
        uint64_t tn = atomic_exchange(&stat_prog_translate_ns, 0), cn = atomic_exchange(&stat_prog_compile_ns, 0);
        if (nb) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, ", new programs %d (translate %.0f ms, spirv+modules+libs %.0f ms)", nb, tn / 1e6, cn / 1e6);
        }
        extern _Atomic int stat_gpl_links;
        extern _Atomic uint64_t stat_gpl_link_ns;
        int nl = atomic_exchange(&stat_gpl_links, 0);
        uint64_t ln = atomic_exchange(&stat_gpl_link_ns, 0);
        if (nl) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, ", linked %d pipelines (%.1f ms)", nl, ln / 1e6);
        }
    }
    if (chk_words) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, ", gdev-check %llu words changed / %llu overwritten", (unsigned long long)chk_words,
                 (unsigned long long)chk_clob);
        chk_words = chk_clob = 0;
    }
    if (prof) {
        fbc_report();
        LOG("[prof] stage sets: alloc %.0f cbuf %.0f fill %.0f update %.0f ms, %llu writes, %llu sets reused | dynamic state %llu set, %llu skipped\n",
            prof_set_alloc / 1e6, prof_set_cbuf / 1e6, prof_set_fill / 1e6, prof_set_update / 1e6, (unsigned long long)prof_set_writes,
            (unsigned long long)prof_set_reused, (unsigned long long)prof_dyn_set, (unsigned long long)prof_dyn_skip);
        LOG("[prof] pipelines: %llu bound, %llu rebinds skipped\n", (unsigned long long)prof_pipe_bind, (unsigned long long)prof_pipe_skip);
        LOG("[prof] stage_set same as the stage's previous call: vs %llu/%llu (+dyn %llu), fs %llu/%llu (+dyn %llu), cs %llu/%llu (+dyn %llu)\n",
            (unsigned long long)prof_set_same[0], (unsigned long long)prof_set_calls[0], (unsigned long long)prof_set_samedyn[0],
            (unsigned long long)prof_set_same[1], (unsigned long long)prof_set_calls[1], (unsigned long long)prof_set_samedyn[1],
            (unsigned long long)prof_set_same[5], (unsigned long long)prof_set_calls[5], (unsigned long long)prof_set_samedyn[5]);
        LOG("[prof] stage_set inputs memo (would skip / of which built a different set): vs %llu/%llu, fs %llu/%llu, cs %llu/%llu\n",
            (unsigned long long)prof_memo_hit[0], (unsigned long long)prof_memo_bad[0], (unsigned long long)prof_memo_hit[1],
            (unsigned long long)prof_memo_bad[1], (unsigned long long)prof_memo_hit[5], (unsigned long long)prof_memo_bad[5]);
        LOG("[prof] c1 constants: alias mirror %llu (%.1f MB), zero-copy %llu, heap mirror %llu (%.1f MB), copied: below heap %llu (%.1f MB), misaligned %llu (%.1f MB), "
            "mirror refused %llu (%.1f MB), mirror off %llu | guest_copy: hits %llu, new %llu (%.1f MB), collisions %llu (%.1f MB)\n",
            (unsigned long long)prof_c1[6], prof_c1_b[6] / 1048576.0, (unsigned long long)prof_c1[0], (unsigned long long)prof_c1[1], prof_c1_b[1] / 1048576.0, (unsigned long long)prof_c1[2], prof_c1_b[2] / 1048576.0,
            (unsigned long long)prof_c1[3], prof_c1_b[3] / 1048576.0, (unsigned long long)prof_c1[4], prof_c1_b[4] / 1048576.0, (unsigned long long)prof_c1[5],
            (unsigned long long)prof_gc_hit, (unsigned long long)prof_gc_new, prof_gc_new_b / 1048576.0, (unsigned long long)prof_gc_evict, prof_gc_evict_b / 1048576.0);
        LOG("[prof] c1 mirror refusals: shader-written page %llu, past the mirror %llu, straddle %llu, no block %llu, mirror off %llu\n",
            (unsigned long long)prof_hmw[1], (unsigned long long)prof_hmw[2], (unsigned long long)prof_hmw[3], (unsigned long long)prof_hmw[4], (unsigned long long)prof_hmw[5]);
        memset(prof_hmw, 0, sizeof prof_hmw);
        memset(prof_c1, 0, sizeof prof_c1); memset(prof_c1_b, 0, sizeof prof_c1_b);
        prof_gc_hit = prof_gc_new = prof_gc_new_b = prof_gc_evict = prof_gc_evict_b = 0;
        for (int sg = 0; sg < 2; sg++) {
            uint64_t *y = prof_memo_why[sg];
            LOG("[prof] stage_set memo misses with the same set, %s: gen %llu, seq/frame %llu, views/pool entries %llu, prog/pools %llu, tex %llu, ubuf %llu, sbuf %llu, pre %llu, other %llu\n",
                sg ? "fs" : "vs", (unsigned long long)y[0], (unsigned long long)y[1], (unsigned long long)y[2], (unsigned long long)y[3], (unsigned long long)y[4],
                (unsigned long long)y[5], (unsigned long long)y[6], (unsigned long long)y[7], (unsigned long long)y[8]);
        }
        memset(prof_memo_why, 0, sizeof prof_memo_why);
        memset(prof_memo_hit, 0, sizeof prof_memo_hit); memset(prof_memo_bad, 0, sizeof prof_memo_bad);
        memset(prof_set_calls, 0, sizeof prof_set_calls); memset(prof_set_same, 0, sizeof prof_set_same); memset(prof_set_samedyn, 0, sizeof prof_set_samedyn);
        prof_pipe_bind = prof_pipe_skip = 0;
        prof_set_alloc = prof_set_cbuf = prof_set_fill = prof_set_update = prof_set_writes = prof_set_reused = 0;
        prof_dyn_set = prof_dyn_skip = 0;
        prof_up = prof_up_ns;
        prof_up_ns = 0;
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | gpu %.0f cpu exec %.0f draw %.0f (pre %.0f key %.0f dyn %.0f sets %.0f tex %.0f (upload %.0f) vtx %.0f) submit %.0f ms",
                 prof_gpu / 1e6, prof_exec / 1e6, prof_draw / 1e6, prof_pre / 1e6, prof_key / 1e6, prof_dyn / 1e6, prof_sets / 1e6,
                 prof_tex / 1e6, prof_up / 1e6, prof_vtx / 1e6, prof_submit / 1e6);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pre: program %.0f fs/debug %.0f feedback %.0f begin_pass %.0f topology %.0f ms",
                 prof_pz[0] / 1e6, prof_pz[1] / 1e6, prof_pz[2] / 1e6, prof_pz[3] / 1e6, prof_pz[4] / 1e6);
        memset(prof_pz, 0, sizeof prof_pz);
        extern uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn, prof_tex_ww_clean;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | vtx copies %llu (%.1f MB, below heap %.1f MB, region %llx0000000, gmem %.0f MB, window straddles %llu)",
                 (unsigned long long)prof_vx_copy_n, prof_vx_copy_bytes / 1e6, prof_vx_low_bytes / 1e6,
                 (unsigned long long)prof_vx_low_region, gmem_size / 1e6, (unsigned long long)prof_vx_straddle);
        prof_vx_copy_n = prof_vx_copy_bytes = prof_vx_low_bytes = prof_vx_straddle = 0;
        for (int i = 0; i < 16 && up0_site[i].ra; i++) {
            if (up0_site[i].bytes < (1u << 20)) continue;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | up0 vk_transient_copy%+lld: %.0f MB / %llu",
                     (long long)(up0_site[i].ra - (uintptr_t)&vk_transient_copy), up0_site[i].bytes / 1e6, (unsigned long long)up0_site[i].n);
        }
        memset(up0_site, 0, sizeof up0_site);
        if (prof_vx_inflight) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | vtx in place (written in flight) %llu", (unsigned long long)prof_vx_inflight);
            prof_vx_inflight = 0;
        }
        if (prof_vxp_dup + prof_vxp_same + prof_vxp_new) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | vx pages: dup in submission %.0f MB, unchanged %.0f MB, changed/new %.0f MB",
                     prof_vxp_dup / 1e6, prof_vxp_same / 1e6, prof_vxp_new / 1e6);
            prof_vxp_dup = prof_vxp_same = prof_vxp_new = 0;
        }
        extern uint64_t prof_alias_syncs, prof_alias_bytes, prof_alias_ranges, prof_alias_ns, prof_hm_syncs, prof_hm_bytes, prof_hm_ranges, prof_hm_ns, prof_hm_vol;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | alias mirror syncs %llu, %.1f MB in %llu ranges, %.1f ms",
                 (unsigned long long)prof_alias_syncs, prof_alias_bytes / 1e6, (unsigned long long)prof_alias_ranges, prof_alias_ns / 1e6);
        prof_alias_syncs = prof_alias_bytes = prof_alias_ranges = prof_alias_ns = 0;
        if (prof_hm_syncs) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | heap mirror syncs %llu, %.1f MB in %llu ranges, %.1f ms, %llu pages rewritten within a submission",
                     (unsigned long long)prof_hm_syncs, prof_hm_bytes / 1e6, (unsigned long long)prof_hm_ranges, prof_hm_ns / 1e6,
                     (unsigned long long)prof_hm_vol);
            prof_hm_syncs = prof_hm_bytes = prof_hm_ranges = prof_hm_ns = prof_hm_vol = 0;
        }
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pipeline waits: built inline %llu, waited %llu, %.1f ms",
                 (unsigned long long)prof_pipe_inline, (unsigned long long)prof_pipe_waited, prof_pipe_wait_ns / 1e6);
        prof_pipe_inline = prof_pipe_waited = prof_pipe_wait_ns = 0;
        if (prof_bake_waits) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | bake program waits %llu, %.1f ms", (unsigned long long)prof_bake_waits, prof_bake_wait_ns / 1e6);
            prof_bake_waits = prof_bake_wait_ns = 0;
        }
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | cond render evals %llu, false %llu, skipped cmds %llu, flips %llu",
                 (unsigned long long)g_cond_evals, (unsigned long long)g_cond_off, (unsigned long long)g_cond_skipped,
                 (unsigned long long)g_cond_flips);
        g_cond_evals = g_cond_off = g_cond_skipped = g_cond_flips = 0;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | chunks %llu | syncwait %llu blocking, age %.1f ms, wait %.1f ms avg | waits",
                 (unsigned long long)prof_chunks, (unsigned long long)sw_n, sw_n ? sw_age_ns / 1e6 / sw_n : 0.0,
                 sw_n ? sw_wait_ns / 1e6 / sw_n : 0.0);
        prof_chunks = sw_n = sw_age_ns = sw_wait_ns = 0;
        if (ubo_zc > 0 || stat_ubo_zc) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | ubo zero-copy %llu, copied %llu", (unsigned long long)stat_ubo_zc,
                     (unsigned long long)stat_ubo_copy);
            stat_ubo_zc = stat_ubo_copy = 0;
        }
        for (int i = 0; i < WSITES && wsite[i].file; i++) {
            uint64_t ns = atomic_exchange(&wsite[i].ns, 0), cnt = atomic_exchange(&wsite[i].n, 0);
            if (ns < 1000000) continue;
            const char *f = wsite[i].file, *b1 = strrchr(f, '/'), *b2 = strrchr(f, '\\');
            if (b2 > b1) b1 = b2;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " %s:%d %.0fms/%llu", b1 ? b1 + 1 : f, wsite[i].line, ns / 1e6,
                     (unsigned long long)cnt);
        }
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | tex checks %llu samples %llu full %llu (%.1f MB, dynamic %llu) ww clean %llu",
                 (unsigned long long)prof_tex_checks, (unsigned long long)prof_tex_samples, (unsigned long long)prof_tex_full,
                 prof_tex_full_bytes / 1048576.0, (unsigned long long)prof_tex_dyn, (unsigned long long)prof_tex_ww_clean);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pass ends:");
        for (int l = 0; l < 4096; l++)
            if (prof_end_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_end_line[l]); }
        memset(prof_end_line, 0, sizeof prof_end_line);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | views created %llu", (unsigned long long)prof_views_created);
        prof_views_created = 0;
        {   // command kinds by render-thread self time (from the record to the next one), with their counts
            LOG("[prof] begin_pass %llu: %.0f ms (prev end %.0f, views %.0f)\n", (unsigned long long)prof_bp_n, prof_bp / 1e6, prof_bp_end / 1e6,
                prof_bp_view / 1e6);
            LOG("[prof] begin_pass zones: vk_cmd %.0f barrier %.0f dl_note %.0f BeginRendering %.0f ms\n", prof_bpz[0] / 1e6,
                prof_bpz[1] / 1e6, prof_bpz[2] / 1e6, prof_bpz[3] / 1e6);
            memset(prof_bpz, 0, sizeof prof_bpz);
            prof_bp = prof_bp_view = prof_bp_end = prof_bp_n = 0;
            extern uint64_t prof_img_n, prof_img_ns;
            extern uint64_t prof_img_reused, prof_img_same_addr;
            LOG("[prof] images created %llu: %.0f ms, reused %llu (same storage %llu)\n", (unsigned long long)prof_img_n, prof_img_ns / 1e6,
                (unsigned long long)prof_img_reused, (unsigned long long)prof_img_same_addr);
            prof_img_n = prof_img_ns = prof_img_reused = prof_img_same_addr = 0;
            LOG("[prof] vertex/index binds: %llu done, %llu skipped (same as bound)\n", (unsigned long long)prof_binds_done,
                (unsigned long long)prof_binds_skipped);
            prof_binds_done = prof_binds_skipped = 0;
            LOG("[prof] ops (kind:ms/count):");
            for (int rank = 0; rank < 12; rank++) {
                int best = -1;
                for (int i = 0; i < CMD_COUNT; i++) if (op_ns[i] && (best < 0 || op_ns[i] > op_ns[best])) best = i;
                if (best < 0) break;
                LOG(" %d:%.0f/%llu", best, op_ns[best] / 1e6, (unsigned long long)(op_counts[best] - op_n2[best]));
                op_ns[best] = 0;
            }
            LOG("\n");
            memset(op_ns, 0, sizeof op_ns);
            memcpy(op_n2, op_counts, sizeof op_n2);
        }
        prof_tex_checks = prof_tex_samples = prof_tex_full = prof_tex_full_bytes = prof_tex_dyn = prof_tex_ww_clean = 0;
        prof_exec = prof_draw = prof_sets = prof_vtx = prof_submit = prof_pre = prof_key = prof_dyn = prof_tex = prof_up = prof_gpu = 0;
    }
    if (stat_skipped) {
        LOG("[vk] skip why: no program %llu, program not built %llu, hash %llu, topology %llu, pipeline not built %llu, cursor %llu, compute %llu\n",
            (unsigned long long)stat_skip_why[0], (unsigned long long)stat_skip_why[1], (unsigned long long)stat_skip_why[2],
            (unsigned long long)stat_skip_why[3], (unsigned long long)stat_skip_why[4], (unsigned long long)stat_skip_why[5],
            (unsigned long long)stat_skip_why[6]);
        memset(stat_skip_why, 0, sizeof stat_skip_why);
    }
    stat_draws = stat_skipped = stat_passes = stat_dispatches = stat_flushes = stat_wait_ns = 0;
    return stats_text;
}

static void st_defaults(void) {
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 16; i++) {
        st.viewport[i][2] = 1280; st.viewport[i][3] = 720;
        st.scissor[i][2] = 16384; st.scissor[i][3] = 16384;
        st.depth_range[i][1] = 1.0f;
    }
    st.channel_mask = 0xFFFFFFFF;
    st.stencil_mask[0] = st.stencil_mask[1] = 0xFF;
    st.stencil_vmask[0] = st.stencil_vmask[1] = 0xFF;
    st.ds[1] = 1; st.ds[2] = 2;
    st.poly[1] = 1; st.poly[2] = 2;
    for (int i = 0; i < 8; i++) { st.blend[i][1] = 2; st.blend[i][2] = 1; st.blend[i][3] = 2; st.blend[i][4] = 1; st.blend[i][5] = 1; st.blend[i][6] = 1; }
}

void vk_exec_init(void) {
    prof = getenv("LARECOMP_VK_PROF") && strcmp(getenv("LARECOMP_VK_PROF"), "0");   // (=0: off; run.sh defaults it to 1)
    if (prof) {   // (prof_now: TSC ticks per ns over ~20 ms)
        uint64_t n0 = SDL_GetTicksNS(), c0 = __rdtsc(), n1;
        while ((n1 = SDL_GetTicksNS()) - n0 < 20000000ull) {}
        uint64_t c1 = __rdtsc();
        prof_ns_per_tick = (double)(n1 - n0) / (double)(c1 - c0);
        prof_tsc0 = c1; prof_ns0 = n1;
    }
    if (prof) {
        prof_up_ns = 0;
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SUBS * 2};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &prof_qp) != VK_SUCCESS) prof_qp = VK_NULL_HANDLE;
    }
    st_defaults();
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy);
void renderer_texture_sample_scale(NvnTexture *t, float *sx, float *sy);
void renderer_texture_promote(NvnTexture *t);
bool renderer_texture_presented(NvnTexture *t);
bool renderer_texture_first_write(NvnTexture *t);
bool renderer_texture_level_fresh(NvnTexture *t);
bool renderer_texture_frame_fresh(NvnTexture *t);
void renderer_texture_mark_presented(NvnTexture *t);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);
static float pass_sx = 1, pass_sy = 1;
static float vp_sx = 1, vp_sy = 1, vp_ox = 0, vp_oy = 0;
static float tex_sc[2][64][2];
static uint32_t tex_wh[2][64][2];

// ================================================================== render passes
static VkFormat pass_color[8], pass_depth, pass_stencil;
static uint32_t pass_ncolor, pass_samples = 1, pass_w, pass_h;
static int debug_pass_no;

// Passes into block shadows (textures.c block_shadow): after the pass, the rendered level / slice is copied into the BC
// image (size-compatible copy: extent in the shadow's texels = blocks of the BC image).
typedef struct { TexBackend *bc, *shadow; uint32_t level, z; } BlkCopy;
static BlkCopy pass_blk[8];
static int pass_nblk;
static uint64_t stat_blk_copies;

static void blk_copy_back(VkCommandBuffer cb) {
    if (!pass_nblk) return;
    vk_full_barrier(cb);
    for (int i = 0; i < pass_nblk; i++) {
        const BlkCopy *c = &pass_blk[i];
        TexBackend *bc = c->bc, *s = c->shadow;
        if (!bc || !s || !bc->image || !s->image || c->level >= s->levels || c->level >= bc->levels) continue;
        bool is3d = s->base_type == VK_IMAGE_VIEW_TYPE_3D;
        uint32_t sw = s->width >> c->level, sh = s->height >> c->level, sd = is3d ? s->depth >> c->level : s->layers;
        uint32_t bw = bc->width >> c->level, bh = bc->height >> c->level;
        if (!sw) sw = 1;
        if (!sh) sh = 1;
        if (!sd) sd = 1;
        // the BC level must hold whole blocks (levels below 4x4 are partial blocks: not copied, logged once)
        if (bw < 4 || bh < 4 || bw != sw * 4 || bh != sh * 4 || c->z >= sd) {
            static int nlog;
            if (nlog++ < 10) LOG("[vk] block shadow copy skipped: level %u %ux%u (blocks %ux%u) slice %u of %u\n", c->level, bw, bh, sw, sh, c->z, sd);
            continue;
        }
        VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c->level, is3d ? 0 : c->z, 1},
                          .srcOffset = {0, 0, is3d ? (int32_t)c->z : 0},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c->level, is3d ? 0 : c->z, 1},
                          .dstOffset = {0, 0, is3d ? (int32_t)c->z : 0}, .extent = {sw, sh, 1}};
        vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, bc->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        if (++stat_blk_copies <= 4 || !(stat_blk_copies & 255))
            LOG("[vk] block shadow copy #%llu: level %u slice %u, %ux%u blocks -> BC %ux%u\n", (unsigned long long)stat_blk_copies,
                c->level, c->z, sw, sh, bw, bh);
    }
    pass_nblk = 0;
    vk_full_barrier(cb);
}

static void vk_end_pass_at(int line);
// LARECOMP_PASS_IMAGE_BARRIERS=1 (A/B, 2026-10-19 NVIDIA decal hang): like Eden's Scheduler::EndRenderPass, an image
// memory barrier (GENERAL -> GENERAL, attachment writes -> all reads/writes) on every attachment image of a pass right
// after it ends, besides our global memory barriers (equivalent by the spec; drivers may treat them differently).
static VkImage pass_img[9];
static VkImageAspectFlags pass_img_aspect[9];
static int n_pass_img;
static bool depth_live_in_pass;   // a draw of the open pass sampled the bound depth attachment live (resolve_feedback)
// Attachments of game textures are declared as possible feedback loops (device.c g_vk.feedback_loop; textures.c gives them the
// usage bit): TotK's decals sample the targets they render to while bound.
static const VkAttachmentFeedbackLoopInfoEXT fb_loop_info = {.sType = VK_STRUCTURE_TYPE_ATTACHMENT_FEEDBACK_LOOP_INFO_EXT, .feedbackLoopEnable = VK_TRUE};
static void debug_save_depth(TexBackend *b, const char *path);
static uint32_t dump_layer;   // pass dumps: the array layer the target view renders to (read by raw_view)
void vk_end_pass(void) { vk_end_pass_at(0); }
#define vk_end_pass() vk_end_pass_at(__LINE__)
static void vk_end_pass_at(int line) {
    if (line >= 0) { pass_barrier_needed = true; pbn_cause = line < 8192 ? line : 0; }   // (our own work outside a pass follows: the next pass waits for it)
    if (!in_pass) return;
    if (prof && line < 4096) prof_end_line[line]++;
    vkCmdEndRendering(vk_cmd());
    in_pass = false;
    depth_live_in_pass = false;
    static int imgbar = -1;
    if (imgbar < 0) { const char *e = getenv("LARECOMP_PASS_IMAGE_BARRIERS"); imgbar = e && *e == '1'; }
    if (imgbar && n_pass_img) {
        VkImageMemoryBarrier2 ib[9];
        for (int i = 0; i < n_pass_img; i++)
            ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = pass_img[i], .subresourceRange = {pass_img_aspect[i], 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)n_pass_img, .pImageMemoryBarriers = ib};
        vkCmdPipelineBarrier2(vk_cmd(), &di);
    }
    n_pass_img = 0;
    blk_copy_back(vk_cmd());
    static int nopng = -1;   // LARECOMP_DEBUG_NOPNG=1: debug frames log only (no pass dumps; long LARECOMP_DEBUG_FRAMES runs)
    if (nopng < 0) nopng = getenv("LARECOMP_DEBUG_NOPNG") != NULL;
    if (g_debug_frame && !nopng && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {
        char path[256];
        static uint64_t debug_pass_frame = ~0ull;
        static int multi = -1;
        if (multi < 0) multi = getenv("LARECOMP_DEBUG_FRAMES") != NULL;
        if (multi && renderer_present_count() != debug_pass_frame) { debug_pass_frame = renderer_present_count(); debug_pass_no = 0; }
        if (multi) snprintf(path, sizeof path, "build/pass_%llu_%03d.png", (unsigned long long)debug_pass_frame, debug_pass_no++);
        else snprintf(path, sizeof path, "build/pass_%03d.png", debug_pass_no++);
        RtState rts = st.rt;   // (saving ends the pass and may touch st)
        // each target's own array layer (an 800x450x2 cave-ambient array is written as layer 0 + layer 1 of one image;
        // dumping layer 0 for both showed two identical images, F9 frame 65510 pass 135)
#define RT_LAYER(i) (rts.has_cview[i] && (rts.cview[i].setmask & 2) ? rts.cview[i].min_layer : 0u)
        LOG("[vk] pass dump %s: rt0 %p addr %llx layer %u\n", path, (void *)st.rt.color[0], (unsigned long long)st.rt.color[0]->addr, RT_LAYER(0));
        dump_layer = RT_LAYER(0);
        renderer_save_texture_tex(rts.color[0]->backend, path);
        static int allrt = -1;
        if (allrt < 0) allrt = getenv("LARECOMP_DEBUG_ALLRT") != NULL;   // also color targets 1..7: pass_NNN_c<i>.png
        for (int i = 1; allrt && i < rts.ncolor; i++)
            if (rts.color[i] && rts.color[i]->backend) {
                char p2[300];
                snprintf(p2, sizeof p2, "%.*s_c%d.png", (int)(strlen(path) - 4), path, i);
                dump_layer = RT_LAYER(i);
                if (dump_layer) LOG("[vk] pass dump %s: rt%d layer %u\n", p2, i, dump_layer);
                renderer_save_texture_tex(rts.color[i]->backend, p2);
            }
        dump_layer = 0;
#undef RT_LAYER
    }
    // LARECOMP_DEBUG_DEPTH=1 (debug frame): depth-only passes (shadow maps) too -> build/pass_<frame>_NNN_depth.pgm
    static int ddepth = -1;
    if (ddepth < 0) ddepth = getenv("LARECOMP_DEBUG_DEPTH") != NULL;
    if (ddepth && g_debug_frame && !st.rt.ncolor && st.rt.depth && st.rt.depth->backend) {
        char path[256];
        snprintf(path, sizeof path, "build/pass_%llu_%03d_depth.pgm", (unsigned long long)renderer_present_count(), debug_pass_no++);
        LOG("[vk] depth pass dump %s: depth %p addr %llx f%#x\n", path, (void *)st.rt.depth, (unsigned long long)st.rt.depth->addr,
            st.rt.depth->format);
        debug_save_depth(st.rt.depth->backend, path);
    }
}

static uint32_t level_extent(uint32_t v, uint32_t lv) { v >>= lv; return v ? v : 1; }

// Pass-begin barriers (2026-10-17, Deck d24 standing frame: ~400 passes per frame, each began with a full ALL_COMMANDS
// barrier = a GPU drain; TotK's 132 independent 64x64 cube-filter passes (fs 544d1de3, 4 taps per pixel) took 16-25 us each,
// 2.1 ms per frame, mostly drain + ramp). NVN orders render-to-texture -> sampling only through the game's own
// nvnCommandBufferBarrier (Maxwell does too), so a pass now starts with the barrier only when something since the last one
// needs it: a game CMD_BARRIER, or any of our own work outside a pass (copies, dispatches, snapshots, present: everything
// that ends the pass through vk_end_pass, see vk_end_pass_at). Consecutive passes with only draws between them run without
// one (attachment write-after-write across passes stays in submission order on the GPU). LARECOMP_PASS_BARRIER=1: every pass
// (old).
// (pass_barrier_needed, pass_unbarriered, stat_pass_*: declared at the top with the other stats)
// d25 OBSERVED: TotK issues ~1.2 barriers per draw (331k per 2 s, 267k draws), so the game-barrier gating ("game") skipped
// only 2772 of 20163 pass barriers: no gain. "tracked" (2026-10-17) ignores the game's barriers and orders by hazard, like
// DXVK: the guest storage ranges of the attachments written since the last full barrier are kept (wr_mark at pass begin);
// a draw that samples a texture overlapping one of them, or reads storage after a storage / image write in this epoch, gets a
// full barrier first (resolve_feedback); everything else (pass changes, write-after-write on an attachment) runs without one.
// Our own work outside passes still barriers before itself and before the next pass (vk_end_pass_at).
// LARECOMP_BARRIERS = pass (default for now: a full barrier before every pass, the old way) | game | tracked | none
// (none: only our own work's barriers, no feedback barriers: a measurement of the upper bound, wrong pictures expected).
// LARECOMP_PASS_BARRIER=1 = pass.
enum { BAR_PASS, BAR_GAME, BAR_TRACKED, BAR_NONE };
static int barrier_mode(void) {
    static int m = -1;
    if (m < 0) {
        const char *e = getenv("LARECOMP_BARRIERS"), *o = getenv("LARECOMP_PASS_BARRIER");
        m = !e ? BAR_PASS : !strcmp(e, "game") ? BAR_GAME : !strcmp(e, "tracked") ? BAR_TRACKED : !strcmp(e, "none") ? BAR_NONE : BAR_PASS;
        if (o && *o == '1') m = BAR_PASS;
        LOG("[vk] barrier mode: %s\n", m == BAR_PASS ? "pass" : m == BAR_GAME ? "game" : m == BAR_TRACKED ? "tracked" : "none");
    }
    return m;
}
// Written-in-this-epoch marks: the attachment's backend image (wr_epoch) and its guest storage address in a small table
// (textures that alias the same storage through another object / backend start at the same address). d28 OBSERVED with a
// 64-entry range list: it overflowed (hundreds of 64x64 passes per epoch) -> every sampled texture counted as written:
// 29581 hazard barriers per 2 s, 46036 passes (twice the old count).
#define WR_TAB 4096
static struct { uint64_t addr; uint32_t epoch; } wr_tab[WR_TAB];
static uint32_t gmem_wr_epoch;                       // a draw wrote storage / images in that epoch
static void wr_mark(NvnTexture *t) {
    if (!t) return;
    TexBackend *b = t->backend;
    if (b) b->wr_epoch = bar_epoch;
    TexBackend *rb = vk_render_backend(t);
    if (rb && rb != b) rb->wr_epoch = bar_epoch;
    if (!t->addr) return;
    uint32_t h = (uint32_t)((t->addr >> 8) * 0x9E3779B1u) & (WR_TAB - 1);
    for (int k = 0; k < 8; k++) {   // (linear probe; a slot from an older epoch is free)
        uint32_t i = (h + (uint32_t)k) & (WR_TAB - 1);
        if (wr_tab[i].addr == t->addr || wr_tab[i].epoch != bar_epoch) { wr_tab[i].addr = t->addr; wr_tab[i].epoch = bar_epoch; return; }
    }
    wr_tab[h].addr = t->addr;   // (table full around h: overwrite; a lost mark could miss a hazard, 8 probes make it rare)
    wr_tab[h].epoch = bar_epoch;
}
static bool wr_hazard(NvnTexture *t) {
    if (!t) return false;
    TexBackend *b = t->backend;
    if (b && b->wr_epoch == bar_epoch) return true;
    if (!t->addr) return false;
    uint32_t h = (uint32_t)((t->addr >> 8) * 0x9E3779B1u) & (WR_TAB - 1);
    for (int k = 0; k < 8; k++) {
        uint32_t i = (h + (uint32_t)k) & (WR_TAB - 1);
        if (wr_tab[i].addr == t->addr && wr_tab[i].epoch == bar_epoch) return true;
    }
    return false;
}

static uint64_t g_cur_fs_hash;   // (tentative; defined with the program state below)

static void begin_pass(void) {
    if (in_pass && !st.rt_dirty) return;
    uint64_t bp_t0 = prof ? prof_now() : 0, bp_tv = 0;
    vk_end_pass_at(-1);   // (-1: this pass change alone does not ask for a barrier)
    if (prof) { prof_bp_end += prof_now() - bp_t0; prof_bp_n++; }
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each pass its own submission)
    else if (chunk_limit() && chunk_draws >= (uint32_t)chunk_limit()) { prof_chunks++; g_chunk_submit = true; renderer_submit(); g_chunk_submit = false; }
    st.rt_dirty = false;
    if (st.rt.ncolor >= 4 && st.rt.color[0] && st.rt.color[0]->width > g_drs_w) { g_drs_w = st.rt.color[0]->width; g_drs_h = st.rt.color[0]->height; }
    uint64_t z0 = prof ? prof_now() : 0;
    VkCommandBuffer cb = vk_cmd();
    uint64_t z1 = prof ? prof_now() : 0;
    if (pass_barrier_needed || barrier_mode() == BAR_PASS) {
        stat_pbn_cause[pass_barrier_needed ? pbn_cause : 3]++;
        vk_full_barrier(cb);  // passes, copies and dispatches see each other's results
        pass_barrier_needed = false;
        pass_unbarriered = false;
        stat_pass_barriers++;
    } else {
        pass_unbarriered = !pass_barrier_fresh;
        stat_pass_nobarrier++;
    }
    pass_barrier_fresh = false;
    uint64_t z2 = prof ? prof_now() : 0;
    {
        NvnTexture *t0n = st.rt.ncolor ? st.rt.color[0] : NULL;
        dl_note("pass %d color %ux%u fmt %#x depth %p", st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0, t0n ? t0n->format : 0,
                (void *)st.rt.depth);
        hang_op(cb, HOP_PASS, 0, 0, (uint32_t)st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0);
    }
    if (prof) { prof_bpz[0] += z1 - z0; prof_bpz[1] += z2 - z1; prof_bpz[2] += prof_now() - z2; }
    if (g_gpuprof_frame) {
        NvnTexture *t0 = st.rt.ncolor ? st.rt.color[0] : NULL;
        gp_mark(cb, "pass: %d color %ux%u fmt %#x, depth %s", st.rt.ncolor, t0 ? t0->width : 0, t0 ? t0->height : 0,
                t0 ? t0->format : 0, st.rt.depth ? "yes" : "no");
    }
    {   // (textures.c) a promoted target that kept its unscaled backend: reallocated at the scene's scale before use
        extern bool renderer_texture_apply_promotion(NvnTexture *t);
        for (int i = 0; i <= st.rt.ncolor; i++) {
            NvnTexture *t = i < st.rt.ncolor ? st.rt.color[i] : st.rt.depth;
            if (t && t->backend) renderer_texture_apply_promotion(t);
        }
    }
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    uint32_t w = 0, minw = UINT32_MAX, minh = UINT32_MAX;  // w: first attachment seen
    pass_ncolor = (uint32_t)st.rt.ncolor;
    pass_samples = 1;
    pass_nblk = 0;
    n_pass_img = 0;
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        pass_color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        renderer_mark_gpu_owned(t);
        uint64_t tv0 = prof ? prof_now() : 0;
        VkImageView v = vk_view(t, &st.rt.cview[i], st.rt.has_cview[i], true, TEXTYPE_2D);
        if (prof) bp_tv += prof_now() - tv0;
        TexBackend *b = vk_render_backend(t);   // (a BC target draws into its block shadow: textures.c block_shadow)
        if (!v) continue;
        if (b != t->backend && pass_nblk < 8) {
            uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
            uint32_t z = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 2) ? st.rt.cview[i].min_layer : 0;
            pass_blk[pass_nblk++] = (BlkCopy){t->backend, b, lv, z};
        }
        ca[i].imageView = v;
        pass_img[n_pass_img] = b->image; pass_img_aspect[n_pass_img++] = VK_IMAGE_ASPECT_COLOR_BIT;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (b->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) ca[i].pNext = &fb_loop_info;   // (draws may sample it)
        bool fresh = (int)renderer_texture_level_fresh(t) | (int)renderer_texture_frame_fresh(t);  // both: each marks its state
        if (renderer_texture_first_write(t) || fresh) {
            ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ca[i].clearValue.color.float32[3] = fresh ? 0.0f : 1.0f;
            if (g_debug_frame) LOG("[pass] implicit clear rt%d %p addr %llx %ux%u (%s)\n", i, (void *)t, (unsigned long long)t->addr,
                                   t->width, t->height, fresh ? "fresh" : "first write");
        }
        pass_color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == b->format
                      ? vk_format(st.rt.cview[i].format) : b->format;
        uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
        uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
        if (!w) { renderer_texture_scale(t, &pass_sx, &pass_sy); w = ew; }
        if (ew < minw) minw = ew;
        if (eh < minh) minh = eh;
        pass_samples = b->samples;
    }
    pass_depth = pass_stencil = VK_FORMAT_UNDEFINED;
    bool has_d = false, has_s = false;
    if (st.rt.depth) {
        renderer_mark_gpu_owned(st.rt.depth);
        uint64_t tv0 = prof ? prof_now() : 0;
        VkImageView v = vk_view(st.rt.depth, &st.rt.dview, st.rt.has_dview, true, TEXTYPE_2D);
        if (prof) bp_tv += prof_now() - tv0;
        TexBackend *b = st.rt.depth->backend;
        if (v && renderer_format_is_depth(st.rt.depth->format)) {
            VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            if (b->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) a.pNext = &fb_loop_info;
            if (b->zero_on_write) {   // recycled depth image (textures.c recycle_get): zeroed like a new one
                a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                a.clearValue.depthStencil = (VkClearDepthStencilValue){0.0f, 0};
                b->zero_on_write = false;
                if (g_debug_frame) LOG("[pass] implicit depth clear %p addr %llx %ux%u (recycled)\n", (void *)st.rt.depth,
                                       (unsigned long long)st.rt.depth->addr, st.rt.depth->width, st.rt.depth->height);
            }
            pass_img[n_pass_img] = b->image; pass_img_aspect[n_pass_img++] = b->aspect;
            if (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da = a; has_d = true; pass_depth = b->format; }
            if (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa = a; has_s = true; pass_stencil = b->format; }
            uint32_t lv = st.rt.has_dview && (st.rt.dview.setmask & 1) ? st.rt.dview.base_level : 0;
            uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
            if (!w) { renderer_texture_scale(st.rt.depth, &pass_sx, &pass_sy); w = ew; }
            if (ew < minw) minw = ew;
            if (eh < minh) minh = eh;
            if (!pass_ncolor) pass_samples = b->samples;
        }
    }
    if (!w) { pass_sx = pass_sy = 1; minw = minh = 16; }
    {   // attachments at different scales: the render area is the smallest one, so the scaled viewport gets clipped
        // (logged at the 1st, 2nd, 4th, 8th ... pass of each fs hash + attachment size, so later frames show up too)
        static struct { uint64_t key, n; } mixed_seen[32];
        for (int i = 0; i <= pass_ncolor; i++) {
            NvnTexture *t = i < pass_ncolor ? st.rt.color[i] : st.rt.depth;
            float sx = 1, sy = 1;
            if (!t || !t->backend) continue;
            renderer_texture_scale(t, &sx, &sy);
            if (sx == pass_sx && sy == pass_sy) continue;
            uint64_t key = g_cur_fs_hash ^ ((uint64_t)t->width << 40 | (uint64_t)t->height << 20 | (uint64_t)i), n = 0;
            for (int k = 0; k < 32; k++)
                if (mixed_seen[k].key == key || !mixed_seen[k].n) { mixed_seen[k].key = key; n = ++mixed_seen[k].n; break; }
            if (n && !(n & (n - 1))) {
                NvnTexture *t0 = pass_ncolor ? st.rt.color[0] : st.rt.depth;   // (the attachment that set the pass scale)
                LOG("[pass] mixed scales: attachment %s%d %ux%u fmt 0x%x tgt %u flags 0x%x at %.2f, pass at %.2f -> render area %ux%u "
                    "(rt0 %ux%u fmt 0x%x flags 0x%x addr %llx, fs %016llx, frame %llu, %d colour)\n",
                    i < pass_ncolor ? "rt" : "depth", i, t->width, t->height, t->format, t->target, t->flags, sx, pass_sx, minw, minh,
                    t0 ? t0->width : 0, t0 ? t0->height : 0, t0 ? t0->format : 0, t0 ? t0->flags : 0,
                    t0 ? (unsigned long long)t0->addr : 0ull, (unsigned long long)g_cur_fs_hash,
                    (unsigned long long)renderer_present_count(), pass_ncolor);
                LOG("[pass]   (mixed pass #%llu of this kind)\n", (unsigned long long)n);
            }
        }
        // (textures.c renderer_texture_promote) an unscaled 2D attachment of the same guest size as one at the scene's
        // scale renders at that scale from its next backend
        extern void renderer_get_scale(float *sx, float *sy);
        float rsx, rsy;
        renderer_get_scale(&rsx, &rsy);
        NvnTexture *big = NULL;
        for (int i = 0; i <= pass_ncolor && (rsx != 1 || rsy != 1); i++) {
            NvnTexture *t = i < pass_ncolor ? st.rt.color[i] : st.rt.depth;
            float sx = 1, sy = 1;
            if (!t || !t->backend) continue;
            renderer_texture_scale(t, &sx, &sy);
            if (sx == rsx && sy == rsy && t->target == 1) { big = t; break; }
        }
        for (int i = 0; big && i <= pass_ncolor; i++) {
            NvnTexture *t = i < pass_ncolor ? st.rt.color[i] : st.rt.depth;
            float sx = 1, sy = 1;
            if (!t || !t->backend || t == big) continue;
            renderer_texture_scale(t, &sx, &sy);
            if (sx == 1 && sy == 1 && t->width == big->width && t->height == big->height) renderer_texture_promote(t);
        }
    }
    pass_w = minw; pass_h = minh;
    if (barrier_mode() == BAR_TRACKED) {   // (after the barrier above: these are written in the current epoch)
        for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) wr_mark(st.rt.color[i]);
        if (st.rt.depth) wr_mark(st.rt.depth);
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {minw, minh}}, .layerCount = 1,
                          .colorAttachmentCount = pass_ncolor, .pColorAttachments = ca,
                          .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    uint64_t z3 = prof ? prof_now() : 0;
    vkCmdBeginRendering(cb, &ri);
    in_pass = true;
    if (prof) { prof_bpz[3] += prof_now() - z3; prof_bp += prof_now() - bp_t0; prof_bp_view += bp_tv; }
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    stat_passes++;
}

static void pass_size(int *w, int *h) {
    *w = *h = 0;
    for (int i = 0; i < st.rt.ncolor && !*w; i++)
        if (st.rt.color[i]) { *w = (int)st.rt.color[i]->width; *h = (int)st.rt.color[i]->height; }
    if (!*w && st.rt.depth) { *w = (int)st.rt.depth->width; *h = (int)st.rt.depth->height; }
    *w = (int)lroundf(*w * pass_sx);
    *h = (int)lroundf(*h * pass_sy);
}

static VkRect2D scissor_rect(void) {
    int w, h;
    pass_size(&w, &h);
    if (!w) return (VkRect2D){{0, 0}, {pass_w, pass_h}};
    int x = st.scissor[0][0], y = st.scissor[0][1], sw = st.scissor[0][2], sh = st.scissor[0][3];
    if (vp_sx != 1 || vp_sy != 1 || vp_ox != 0 || vp_oy != 0) {
        double x0 = floor(x * (double)vp_sx + vp_ox), y0 = floor(y * (double)vp_sy + vp_oy);
        double x1 = ceil(((double)x + sw) * vp_sx + vp_ox), y1 = ceil(((double)y + sh) * vp_sy + vp_oy);
        if (sw >= 16384) x1 = 1e9;
        if (sh >= 16384) y1 = 1e9;
        if (x1 > 1e9) x1 = 1e9;
        if (y1 > 1e9) y1 = 1e9;
        x = (int)x0; y = (int)y0; sw = (int)(x1 - x0); sh = (int)(y1 - y0);
    }
    if (x < 0) { sw += x; x = 0; }
    if (y < 0) { sh += y; y = 0; }
    if (w > (int)pass_w) w = (int)pass_w;
    if (h > (int)pass_h) h = (int)pass_h;
    if (x >= w || y >= h) { x = y = 0; sw = sh = 0; }
    if (x + sw > w) sw = w - x;
    if (y + sh > h) sh = h - y;
    if (sw < 0) sw = 0;
    if (sh < 0) sh = 0;
    return (VkRect2D){{x, y}, {(uint32_t)sw, (uint32_t)sh}};
}

// Viewport / scissor as last set by draw_ in this command buffer (dyn_cache_vs: the same command buffer + submission;
// dyn_invalidate clears them): a draw with the same values skips the call (run382: every draw set both).
static struct { VkCommandBuffer cb; uint64_t seq; bool vp_ok, sc_ok; VkViewport vp; VkRect2D sc; } vsc;
static bool vsc_same_cb(VkCommandBuffer cb) {
    uint64_t q = vk_pending_seq();
    if (vsc.cb != cb || vsc.seq != q) { vsc.cb = cb; vsc.seq = q; vsc.vp_ok = vsc.sc_ok = false; }
    return true;
}
static uint64_t g_cur_fs_hash;   // (tentative; defined with the program state below)
static void apply_scissor(void) {
    VkRect2D r = scissor_rect();
    // LARECOMP_TINY_SCISSOR_FS=<fs hash> (measurement): that shader's draws get a 1x1 scissor, so its vertex work still runs
    // but almost no fragments do. A/B with the GPU profile: cost gone = fragment-bound, cost kept = vertex / setup-bound.
    // "all" = every draw (Deck power split: GPU fragment watts vs render-thread CPU, same CPU work).
    static uint64_t tiny = 1;
    if (tiny == 1) {
        const char *e = getenv("LARECOMP_TINY_SCISSOR_FS");
        tiny = !e ? 0 : !strcmp(e, "all") ? 2 : strtoull(e, NULL, 16);
    }
    if (tiny && (tiny == 2 || g_cur_fs_hash == tiny)) r = (VkRect2D){{0, 0}, {1, 1}};
    VkCommandBuffer cb = vk_cmd();
    vsc_same_cb(cb);
    if (vsc.sc_ok && !memcmp(&vsc.sc, &r, sizeof r)) return;
    vkCmdSetScissor(cb, 0, 1, &r);
    vsc.sc = r; vsc.sc_ok = true;
}

// Metal convention (y up in NDC, origin top-left) through a negative-height viewport.
static void set_viewport(float x, float y, float w, float h, float zn, float zf) {
    if (w <= 0) w = 1;
    if (h == 0) h = 1;
    VkViewport vp = {x, y + h, w, -h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)};
    VkCommandBuffer cb = vk_cmd();
    vsc_same_cb(cb);
    if (vsc.vp_ok && !memcmp(&vsc.vp, &vp, sizeof vp)) return;
    vkCmdSetViewport(cb, 0, 1, &vp);
    vsc.vp = vp; vsc.vp_ok = true;
}

// ================================================================== clears
static VkPipelineLayout pc_layout(void) {  // push constants only (clears)
    static VkPipelineLayout l;
    if (!l) {
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    }
    return l;
}

static const VkDynamicState DYN[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT};
#define NDYN (sizeof DYN / sizeof DYN[0] - (g_vk.feedback_loop ? 0 : 1))   // (the last one only with device.c's feedback loops)

// Dynamic state draw_ used to set in full on every draw (run362: ~42 % of the driver's samples in exec were under
// draw_, mostly these vkCmdSet* calls): a value equal to the one the command buffer already holds is skipped. Keyed
// like bind_check (command buffer + submission); dyn_defaults and the blit / clear pipeline binds call dyn_invalidate()
// (the blit pipelines have only viewport / scissor dynamic, so binding one leaves the rest undefined).
// LARECOMP_VK_DYN_CACHE=0: everything every draw, as before.
static struct {
    VkCommandBuffer cb;
    uint64_t seq;
    bool valid, sten_valid;
    bool dt, dw, sten, bias_on, restart;
    VkCompareOp dop;
    VkCullModeFlags cull;
    VkFrontFace front;
    VkPrimitiveTopology topo;
    float blend[4], bias[3];
    uint32_t sop[2], cmask[2], wmask[2], ref[2];
    VkPipeline pipe;   // graphics pipeline bound last by draw_ (VK_NULL_HANDLE: unknown)
    VkImageAspectFlags fbl;   // attachment feedback loop aspects (g_vk.feedback_loop)
    int8_t clamp;   // dynamic depth clamp (vk_dyn_clamp); -1 after a pipeline with a static clamp was bound
} dyn;
static void dyn_invalidate(void) { dyn.valid = false; dyn.pipe = VK_NULL_HANDLE; vsc.vp_ok = vsc.sc_ok = false; }
void vk_dyn_invalidate(void) { dyn_invalidate(); }   // (device.c: its own viewport / scissor / pipeline in this command buffer)
static bool dyn_begin(VkCommandBuffer cb) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_DYN_CACHE"); on = !e || strcmp(e, "0"); }
    uint64_t q = vk_pending_seq();
    if (!on || dyn.cb != cb || dyn.seq != q) { dyn.cb = cb; dyn.seq = q; dyn.valid = false; dyn.pipe = VK_NULL_HANDLE; }
    if (!dyn.valid) dyn.sten_valid = false;
    return dyn.valid;
}
static void dyn_end(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_DYN_CACHE"); on = !e || strcmp(e, "0"); }
    dyn.valid = on;
}

// Fixed-function state that isn't part of a pipeline is set explicitly on every bind (it persists across pipelines).
static void dyn_defaults(VkCommandBuffer cb) {
    dyn_invalidate();
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
}

typedef struct { uint64_t key; VkPipeline pipe; } PipeEnt;
#define CLEAR_TAB 256
static PipeEnt clear_tab[CLEAR_TAB];

static VkPipeline clear_pipeline(int index, int kind, uint32_t mask, bool depth, bool stencil) {
    uint64_t key = 0xcbf29ce484222325ull;
#define MIX(v) do { uint64_t v_ = (uint64_t)(v); for (int b_ = 0; b_ < 8; b_++) { key ^= (v_ >> (b_ * 8)) & 0xFF; key *= 0x100000001B3ull; } } while (0)
    MIX(index); MIX(kind); MIX(mask); MIX(depth); MIX(stencil); MIX(pass_ncolor); MIX(pass_depth); MIX(pass_stencil); MIX(pass_samples);
    for (uint32_t i = 0; i < pass_ncolor; i++) MIX(pass_color[i]);
    uint32_t slot = (uint32_t)key & (CLEAR_TAB - 1);
    for (int p = 0; p < CLEAR_TAB; p++, slot = (slot + 1) & (CLEAR_TAB - 1)) {
        if (clear_tab[slot].key == key) return clear_tab[slot].pipe;
        if (!clear_tab[slot].key) break;
    }
    char fsname[32];
    if (index >= 0) snprintf(fsname, sizeof fsname, "clear.frag.%c%d", kind == 1 ? 'i' : kind == 2 ? 'u' : 'f', index);
    else snprintf(fsname, sizeof fsname, "clear.frag.none");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("clear.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fsname), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < pass_ncolor; i++)
        if ((int)i == index) cba[i].colorWriteMask = mask & 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = pass_ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo rinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = pass_ncolor,
                                           .pColorAttachmentFormats = pass_color, .depthAttachmentFormat = pass_depth,
                                           .stencilAttachmentFormat = pass_stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rinfo, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pc_layout()};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p) != VK_SUCCESS) LOG("[vk] clear pipeline failed\n");
    clear_tab[slot].key = key;
    clear_tab[slot].pipe = p;
    return p;
}

static void clear(int index, int kind, const void *color, uint32_t mask, bool depth, float dval, bool stencil, uint32_t sval,
                  uint32_t smask) {
    begin_pass();
    VkCommandBuffer cb = vk_cmd();
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    VkRect2D rect = scissor_rect();
    if (!rect.extent.width || !rect.extent.height) return;
    if (index >= 0 && (index >= (int)pass_ncolor || pass_color[index] == VK_FORMAT_UNDEFINED)) return;
    bool simple = (index < 0 || (mask & 0xF) == 0xF) && (!stencil || (smask & 0xFF) == 0xFF);
    if (simple) {
        VkClearAttachment a[2];
        uint32_t n = 0;
        if (index >= 0) {
            a[n] = (VkClearAttachment){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)index};
            memcpy(a[n].clearValue.color.float32, color, 16);
            n++;
        }
        VkImageAspectFlags dsa = (depth && pass_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil && pass_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (dsa) a[n++] = (VkClearAttachment){.aspectMask = dsa, .clearValue.depthStencil = {dval, sval}};
        VkClearRect cr = {rect, 0, 1};
        if (n) vkCmdClearAttachments(cb, n, a, 1, &cr);
        return;
    }
    VkPipeline p = clear_pipeline(index, kind, mask, depth, stencil);
    if (!p) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    dyn_defaults(cb);
    if (depth) { vkCmdSetDepthTestEnable(cb, VK_TRUE); vkCmdSetDepthWriteEnable(cb, VK_TRUE); }
    if (stencil) {
        vkCmdSetStencilTestEnable(cb, VK_TRUE);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                          VK_COMPARE_OP_ALWAYS);
        vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, smask);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sval);
    }
    int w, h;
    pass_size(&w, &h);
    VkViewport vp = {0, 0, (float)(w ? w : (int)pass_w), (float)(h ? h : (int)pass_h), 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &rect);
    float pc[8] = {0};
    if (color) memcpy(pc, color, 16);
    pc[4] = dval;
    vkCmdPushConstants(cb, pc_layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}

// ================================================================== programs
#define MAX_BIND 96
typedef struct {
    ShaderInfo *vi, *fi;
    uint64_t code[2], cmax[2];  // translation inputs (copied: translated on the translator thread)
    uint64_t csize[2];          // compiler constants (c1) after the code, bytes (0: none / archive program; translate_stage)
    ShaderKey vkey, fkey;
    VkShaderModule vs, fs;
    VkBindingInfo binds[MAX_BIND];
    int nbinds;
    VkDescriptorSetLayout set_layout[2];
    VkPipelineLayout layout;
    VkPipelineLayout lib_layout[2];   // GPL shared libraries: {set 0} for the VS library, {-, set 1} for the FS library
    bool vs_uses_vid;
    _Atomic int state;  // 0 building, 1 ready, 2 failed
    uint64_t ckey;        // content key (code hashes + keys) for the shader precompile; 0 = not recorded
    bool inline_compile;  // precompile thread: compile on the calling thread, not on the job pool
    bool sync_try;        // game's first sight within the sync budget: translated on the drawing thread, compiled there too if its SPIR-V is cached
    bool glsl_saved;      // debug_draw saved its GLSL under this session's code addresses
    VkPipeline lib_pre, lib_frag;   // GPL: pre-rasterization (VS, fill, no depth clamp) / fragment shader (1 sample) libraries
    _Atomic int lib_state;          // 0 none, 2 ready, 3 failed (g_vk.gpl)
    bool modules_dropped;           // vs / fs destroyed after the libraries: reload from the SPIR-V cache by spv_key
    uint64_t spv_key[2];
} Prog;

// Uniform buffers are bound as UNIFORM_BUFFER_DYNAMIC (the span's offset is given at vkCmdBindDescriptorSets): a stage's
// set then depends on the buffers, ranges, views and samplers only, so draws of one material with per-object constants
// share it (stage_set's reuse table) instead of vkAllocateDescriptorSets + vkUpdateDescriptorSets per draw (run382:
// 12 % of the render thread in the driver there). Game cbufs (bindings 3..17) first, then the per-draw driver data
// (0, 18, 19), constants (1) last, up to maxDescriptorSetUniformBuffersDynamic per layout; the rest stay plain.
// LARECOMP_VK_DYN_UBO=0: all plain (old).
static int dyn_ubo_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_DYN_UBO"); on = !e || strcmp(e, "0"); }
    return on;
}
static void make_dynamic_ubos(Prog *p) {
    if (!dyn_ubo_on()) return;
    uint32_t lim = g_vk.props.limits.maxDescriptorSetUniformBuffersDynamic, n = 0, total = 0;
    static const int pass_of[20] = {1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};  // 0 game, 1 driver, 2 constants
    for (int pass = 0; pass < 3; pass++)
        for (int i = 0; i < p->nbinds; i++) {
            VkBindingInfo *b = &p->binds[i];
            if (b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) continue;
            int k = b->binding < 20 ? pass_of[b->binding] : 3;
            if (pass == 0) total++;
            if (k != pass || n >= lim) continue;
            b->type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            n++;
        }
    static int nlog;
    if (n < total && nlog++ < 8) LOG("[vk] program with %u uniform buffers: %u dynamic (limit %u), the rest plain\n", total, n, lim);
}
// Shared stage libraries (2026-10-17, user: "speed up shader compiling while keeping the same amount of shaders"): the
// archive's 18325 programs hold only 13346 distinct vertex and 8320 distinct fragment shaders, but each program's VS / FS
// library was created with the program's whole layout (VS set + FS set), so the driver compiled a VS again for every FS
// it was paired with (run403: GPL libraries 240 ms of ~280 ms per program). With independent sets (DXVK's layout) each
// library sees only its own stage's set, so the same shader + set gives the same library and the driver's pipeline cache
// returns it instead of compiling (up to 36650 -> 21666 compiles; FS GLSL also depends on the VS outputs). Linked and
// monolithic pipelines use the full layout, which carries the flag too (descriptor binds must match it).
// LARECOMP_GPL_SHARED=0: the old per-program layouts.
static bool gpl_shared(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("LARECOMP_GPL_SHARED"); v = g_vk.gpl && !(e && *e == '0'); }
    return v;
}
static VkPipelineLayout make_layout(Prog *p, int nsets, VkShaderStageFlags s0, VkShaderStageFlags s1) {
    make_dynamic_ubos(p);
    p->set_layout[0] = vk_set_layout(p->binds, p->nbinds, 0, s0);
    if (nsets > 1) p->set_layout[1] = vk_set_layout(p->binds, p->nbinds, 1, s1);
    bool ind = nsets == 2 && gpl_shared();
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = (uint32_t)nsets,
                                     .pSetLayouts = p->set_layout,
                                     .flags = ind ? VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT : 0};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    if (ind) {
        VkDescriptorSetLayout fs_only[2] = {VK_NULL_HANDLE, p->set_layout[1]};
        ci.setLayoutCount = 1;
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &p->lib_layout[0]));
        ci.setLayoutCount = 2;
        ci.pSetLayouts = fs_only;
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &p->lib_layout[1]));
    }
    return l;
}

static void save_glsl(const char *src, uint64_t addr, const char *suffix) {
    // LARECOMP_SAVE_GLSL=1: save only (no pc annotations, so the SPIR-V cache still hits; the launcher sets it for F9)
    if (!getenv("LARECOMP_SAVE_MSL") && !getenv("LARECOMP_SAVE_GLSL")) return;
    mkdir("build/glsl", 0755);
    char path[256];
    snprintf(path, sizeof path, "build/glsl/%llx_%s.glsl", (unsigned long long)addr, suffix);
    FILE *f = fopen(path, "w");
    if (f) { fputs(src, f); fclose(f); }
}

// ================================================================== graphics pipeline libraries (g_vk.gpl)
// DXVK's answer to first-draw stutter. Each program's shaders are compiled once, when the program is built (by the
// game's request or by the precompile), into a pre-rasterization library (VS) and a fragment-shader library (FS). A new
// draw state then only fast-links them with small vertex-input / fragment-output libraries (cached by state) on the
// drawing thread, instead of a full pipeline compile on a worker while the draw is skipped and the frame held.
// Pipelines with line fill, multisampling or alpha-to-coverage keep the monolithic path; depth clamp too unless it is
// dynamic in the pre-rasterization library (vk_dyn_clamp, device.c; draw_ sets it per linked pipeline).
// LARECOMP_NO_GPL=1: off (device.c). Stats line: "linked N pipelines (ms)".
extern bool vk_dyn_clamp;
static __thread bool pa_tm_on;            // (archive warm-up stage timers: see compile_program)
static __thread bool translate_unlocked;   // pa_job with LARECOMP_PARALLEL_TRANSLATE=1: no translate_lock (state is thread-local)
static __thread uint64_t pa_tm[8];
static _Atomic uint64_t pa_tm_sum[8], pa_tm_n;
_Atomic int stat_gpl_links;
_Atomic uint64_t stat_gpl_link_ns;
static VkPipeline gpl_create(VkGraphicsPipelineCreateInfo *pi, VkGraphicsPipelineLibraryFlagsEXT what) {
    VkGraphicsPipelineLibraryCreateInfoEXT li = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                 .pNext = pi->pNext, .flags = what};
    pi->pNext = &li;
    VkPipelineCreationFeedback fb = {0};   // (warm-up stage timers: was it a pipeline-cache hit?)
    VkPipelineCreationFeedbackCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                .pNext = pi->pNext, .pPipelineCreationFeedback = &fb};
    if (pa_tm_on) pi->pNext = &fci;
    pi->flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, pi, NULL, &p);
    if (pa_tm_on && r == VK_SUCCESS) {
        pa_tm[7]++;
        if ((fb.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) &&
            (fb.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)) pa_tm[6]++;
    }
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline library %#x failed: %d\n", (unsigned)what, r);
        return VK_NULL_HANDLE;
    }
    return p;
}
static void gpl_program_libs(Prog *p) {
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineShaderStageCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                          .module = p->vs, .pName = "main"};
    VkPipelineShaderStageCreateInfo fs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                          .module = p->fs, .pName = "main"};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkDynamicState dyn_pre[sizeof DYN / sizeof DYN[0] + 1];
    memcpy(dyn_pre, DYN, sizeof DYN);
    VkPipelineDynamicStateCreateInfo dyp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                            .dynamicStateCount = NDYN, .pDynamicStates = dyn_pre};
    if (vk_dyn_clamp) dyn_pre[dyp.dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT;
    VkGraphicsPipelineCreateInfo a = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 1, .pStages = &vs,
                                      .pViewportState = &vp, .pRasterizationState = &rs, .pDynamicState = &dyp,
                                      .layout = p->lib_layout[0] ? p->lib_layout[0] : p->layout};
    p->lib_pre = gpl_create(&a, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT);
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo b = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 1, .pStages = &fs,
                                      .pMultisampleState = &ms, .pDepthStencilState = &ds, .pDynamicState = &dy,
                                      .layout = p->lib_layout[1] ? p->lib_layout[1] : p->layout};
    p->lib_frag = p->lib_pre ? gpl_create(&b, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) : VK_NULL_HANDLE;
    atomic_store(&p->lib_state, p->lib_pre && p->lib_frag ? 2 : 3);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n);
// A stage's code hash as draw_ computes it (0x200 bytes after the 0x30 header), but never past code + constants: recorded
// and archived programs are exact-size host copies (U22, d71_imm: SIGSEGV at boot reading the page after one).
static uint64_t prog_code_hash(const Prog *p, int i) {
    uint64_t n = p->cmax[i] + p->csize[i];
    return hmix(1469598103934665603ull, (const void *)(uintptr_t)(p->code[i] + 0x30), n >= 0x230 ? 0x200 : n > 0x30 ? n - 0x30 : 0);
}
_Atomic int stat_prog_builds;   // (game-requested programs only; the precompile's own builds are not counted)
_Atomic uint64_t stat_prog_translate_ns, stat_prog_compile_ns;
// Archive warm-up stage timers (2026-10-17, user: "any way u can speed up shader compiling while keeping the same amount of
// shaders"): per program, thread time in each stage, summed over the warm-up workers and logged with each warm-up record
// ("[precompile] warm-up stages"). 0 lock wait, 1 translate (under translate_lock), 2 GLSL -> SPIR-V (disk cache or
// shaderc), 3 modules + reflection + layout, 4 GPL libraries (the driver's compile); 5 = stages whose SPIR-V was cached;
// 6 = libraries the driver reported as pipeline-cache hits (creation feedback), 7 = libraries created (declared above
// gpl_create).
// LARECOMP_GLSL_SWAP=<dir> (debug): <dir>/<fs hash>_vs.glsl / _fs.glsl replace that program's translated GLSL; the
// translated text goes to <dir>/<fs hash>_<stage>.orig.glsl. Descriptors follow the swapped SPIR-V's reflection.
// (2026-10-18: which stage of fs d4201cbc hangs the NVIDIA driver)
static void glsl_swap(char **msl, const char *fs, const char *stage) {
    const char *dir = getenv("LARECOMP_GLSL_SWAP");
    if (!dir || !*msl) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s_%s.glsl", dir, fs, stage);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc(n + 1);
    s[fread(s, 1, n, f)] = 0;
    fclose(f);
    snprintf(path, sizeof path, "%s/%s_%s.orig.glsl", dir, fs, stage);
    if ((f = fopen(path, "wb"))) { fputs(*msl, f); fclose(f); }
    free(*msl);
    *msl = s;
    LOG("[shader] fs %s: %s GLSL swapped (LARECOMP_GLSL_SWAP)\n", fs, stage);
}
static void compile_program(void *arg) {
    Prog *p = arg;
    uint64_t t0 = SDL_GetTicksNS();
    if (pa_tm_on) pa_tm[5] += vk_spirv_cached(p->vi->msl, STAGE_VERTEX) + vk_spirv_cached(p->fi->msl, STAGE_FRAGMENT);
    size_t nv = 0, nf = 0;
    char nv_[40], nf_[40];  // names carry the guest code hash (fs hashes as in draw_; SPIR-V cache miss log)
    snprintf(nv_, sizeof nv_, "vs_%016llx", (unsigned long long)prog_code_hash(p, 0));
    snprintf(nf_, sizeof nf_, "fs_%016llx", (unsigned long long)prog_code_hash(p, 1));
    glsl_swap(&p->vi->msl, nf_ + 3, "vs");
    glsl_swap(&p->fi->msl, nf_ + 3, "fs");
    uint32_t *sv = vk_glsl_to_spirv(p->vi->msl, STAGE_VERTEX, nv_, &nv);
    uint32_t *sf = sv ? vk_glsl_to_spirv(p->fi->msl, STAGE_FRAGMENT, nf_, &nf) : NULL;
    if (!sv || !sf) { free(sv); free(sf); atomic_store(&p->state, 2); return; }
    uint64_t t1 = SDL_GetTicksNS();
    p->nbinds = vk_reflect(sv, nv, p->binds, MAX_BIND);
    p->nbinds += vk_reflect(sf, nf, p->binds + p->nbinds, MAX_BIND - p->nbinds);
    p->vs = vk_module(sv, nv);
    p->fs = vk_module(sf, nf);
    free(sv);
    free(sf);
    p->layout = make_layout(p, 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    uint64_t t2 = SDL_GetTicksNS();
    if (g_vk.gpl && p->vs && p->fs) gpl_program_libs(p);
    if (pa_tm_on) { uint64_t t3 = SDL_GetTicksNS(); pa_tm[2] += t1 - t0; pa_tm[3] += t2 - t1; pa_tm[4] += t3 - t2; }
    bool ok = p->vs && p->fs;
    // With libraries, the modules and the GLSL are only needed again for a monolithic pipeline (line fill, depth clamp,
    // MSAA, A2C): drop them (~0.5 MB per program; ~10k programs are precompiled) and reload the SPIR-V from the disk
    // cache then (build_pipeline_body). LARECOMP_KEEP_MODULES=1: keep them (old).
    static int keep = -1;
    // (LARECOMP_SAVE_GLSL, set by the launcher for F9, no longer keeps them: the F9 frame re-translates dropped programs)
    if (keep < 0) keep = getenv("LARECOMP_KEEP_MODULES") || getenv("LARECOMP_SAVE_MSL") || getenv("LARECOMP_VK_NO_SPIRV_CACHE");
    if (ok && !keep && atomic_load(&p->lib_state) == 2 && vk_spirv_cached(p->vi->msl, STAGE_VERTEX) &&
        vk_spirv_cached(p->fi->msl, STAGE_FRAGMENT)) {
        p->spv_key[0] = vk_spirv_key(p->vi->msl, STAGE_VERTEX);
        p->spv_key[1] = vk_spirv_key(p->fi->msl, STAGE_FRAGMENT);
        vkDestroyShaderModule(g_vk.dev, p->vs, NULL);
        vkDestroyShaderModule(g_vk.dev, p->fs, NULL);
        p->vs = p->fs = VK_NULL_HANDLE;
        p->modules_dropped = true;
        free(p->vi->msl);
        free(p->fi->msl);
        p->vi->msl = p->fi->msl = NULL;
    }
    atomic_fetch_add(&stat_compiles, 1);
    if (!p->inline_compile) atomic_fetch_add(&stat_prog_compile_ns, SDL_GetTicksNS() - t0);
    atomic_store(&p->state, ok ? 1 : 2);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n);
// First sight of a program (2026-10-17, user: "i still experience shader lag"; their 10-15 PC session held frames 36 times
// / 100 frames for programs still being built): a draw whose program was not built is skipped and the frame held. Most
// first-sight programs now have their SPIR-V in the disk cache and their pipeline libraries in the driver cache (archive
// warm-up, recordings), so the drawing thread translates and builds them itself while the time spent so in the last
// 50 ms stays under LARECOMP_PROG_SYNC_MS (default 8; 0 = off: the old skip + hold); a SPIR-V cache miss (a GLSL compile)
// still goes to the workers. "[newprog]" lines: each game-requested program's cost and path (first 300).
static uint64_t g_psync_win_t, g_psync_win_ns;   // (drawing thread)
static bool prog_sync_ok(void) {
    static long budget_ms = -1;
    if (budget_ms < 0) { const char *e = getenv("LARECOMP_PROG_SYNC_MS"); budget_ms = e ? atol(e) : 8; }
    if (!budget_ms || sync_shaders()) return false;
    uint64_t now = SDL_GetTicksNS();
    if (now - g_psync_win_t > 50000000ull) { g_psync_win_t = now; g_psync_win_ns = 0; }
    return g_psync_win_ns < (uint64_t)budget_ms * 1000000ull;
}
static void prog_sync_note(const Prog *p, uint64_t t0, bool built) {
    uint64_t dt = SDL_GetTicksNS() - t0;
    g_psync_win_ns += dt;
    static int n;
    if (n++ < 300)
        LOG("[newprog] vs %llx fs %llx: %s in %.1f ms\n", (unsigned long long)p->code[0], (unsigned long long)p->code[1],
            built ? "built on the drawing thread (SPIR-V cached)" : "translated; SPIR-V not cached -> workers", dt / 1e6);
}
// TotK's upscale pass (fs 32845c90c668fdbb, FSR 1 EASU 1600x900 -> 1920x1080) replaced when the scene or the canvas is
// scaled: one resample from the scene's real pixels to the canvas's (TOTK_RES_SCALE 2 + TOTK_OUT_SCALE at 1440p = 3200x1800
// -> 2560x1440), instead of point samples on the 1600x900 grid. Catmull-Rom, widened to the source footprint when it
// shrinks; averaged through x / (1 + max(x)) so HDR highlights do not ring. Same bindings as the translated shader:
// cb3[0].xy = EASU con0 (input viewport / output size, guest pixels), rscale[0] = target scale + offset, rscale[1] = t0's.
static const char upscale_resample_glsl[] =
    "#version 460\n"
    "layout(std140, set = 1, binding = 3) uniform CB3 { uvec4 cb3[16]; };\n"
    "layout(std140, set = 1, binding = 18) uniform RScale { vec4 rscale[65]; };\n"
    "layout(set = 1, binding = 32) uniform sampler2D t0;\n"
    "layout(location = 0) out vec4 c0;\n"
    "float cr(float x) {\n"
    "  x = abs(x);\n"
    "  return x < 1.0f ? (1.5f * x - 2.5f) * x * x + 1.0f : x < 2.0f ? ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f : 0.0f;\n"
    "}\n"
    "void main() {\n"
    "  vec2 ratio = uintBitsToFloat(cb3[0].xy);\n"
    "  vec2 g = (gl_FragCoord.xy - rscale[0].zw) / rscale[0].xy;\n"
    "  vec2 p = g * ratio * rscale[1].xy;\n"
    "  vec2 f = max(ratio * rscale[1].xy / rscale[0].xy, vec2(1.0f));\n"
    "  ivec2 mx = textureSize(t0, 0) - 1;\n"
    "  ivec2 a = ivec2(floor(p - 2.0f * f)), b = min(ivec2(ceil(p + 2.0f * f)), a + 16);\n"
    "  vec4 acc = vec4(0.0f); float ws = 0.0f;\n"
    "  for (int y = a.y; y <= b.y; y++) {\n"
    "    float wy = cr((float(y) + 0.5f - p.y) / f.y);\n"
    "    if (wy == 0.0f) continue;\n"
    "    for (int x = a.x; x <= b.x; x++) {\n"
    "      float w = wy * cr((float(x) + 0.5f - p.x) / f.x);\n"
    "      if (w == 0.0f) continue;\n"
    "      vec4 v = texelFetch(t0, clamp(ivec2(x, y), ivec2(0), mx), 0);\n"
    "      v.rgb = max(v.rgb, vec3(0.0f));\n"
    "      v.rgb /= 1.0f + max(max(v.r, v.g), v.b);\n"
    "      acc += w * v; ws += w;\n"
    "    }\n"
    "  }\n"
    "  acc /= max(ws, 1e-6f);\n"
    "  acc.rgb = clamp(acc.rgb, vec3(0.0f), vec3(0.999f));\n"
    "  acc.rgb /= 1.0f - max(max(acc.r, acc.g), acc.b);\n"
    "  c0 = vec4(clamp(acc.rgb, vec3(0.0f), vec3(1.0f)), 1.0f);\n"
    "}\n";

static void translate_program(void *arg) {
    Prog *p = arg;
    bool upscale_resample = false;
    uint64_t t0 = SDL_GetTicksNS();
    if (!translate_unlocked) pthread_mutex_lock(&translate_lock);
    uint64_t tl = SDL_GetTicksNS();
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    maxwell_gdev_size = gdev_window_size();
    {   // LARECOMP_DEBUG_VS=<fs hash hex>[,vid]: the vertex shader of the programs with that fragment shader prints the
        // inputs + clip position of every vertex far off screen (or w <= 0) as "[vtrace] ..." lines, and with vid every
        // register write of gl_VertexIndex == vid ("[trace] pc r<n>=hex"). Needs the same validation printf env as
        // LARECOMP_DEBUG_SHADER. (2026-10-15: vertical streaks from the terrain grass shader, fs 657fb1ab0f780c67)
        static uint64_t vh = 1;
        static int vvid = -1;
        if (vh == 1) {
            const char *e = getenv("LARECOMP_DEBUG_VS");
            char *q = NULL;
            vh = e ? strtoull(e, &q, 16) : 0;
            if (q && *q == ',') vvid = (int)strtol(q + 1, NULL, 10);
        }
        if (vh && p->code[1] && prog_code_hash(p, 1) == vh) {
            extern MAXWELL_TLS bool maxwell_vtrace_next;
            extern MAXWELL_TLS int maxwell_vtrace_vid;
            maxwell_vtrace_next = true; maxwell_vtrace_vid = vvid;
        }
    }
    p->vi = translate_stage(p->code[0], p->cmax[0], p->csize[0], STAGE_VERTEX, &p->vkey);
    if (p->vi->msl) {
        p->fkey.vs_out_mask = p->vi->out_mask;
        {   // LARECOMP_DEBUG_SHADER=<fs hash hex>,x,y: that fragment shader prints every register write at pixel (x, y)
            // (needs LARECOMP_VK_VALIDATION=1 + VK_LAYER_PRINTF_ONLY_PRESET=1 VK_LAYER_PRINTF_TO_STDOUT=1)
            static uint64_t th = 1;
            static int tx, ty, tx2 = -1, ty2 = -1;   // optional second pixel: <hash>,x,y,x2,y2
            if (th == 1) {
                const char *e = getenv("LARECOMP_DEBUG_SHADER");
                th = 0;
                if (e) {
                    char *q; th = strtoull(e, &q, 16);
                    if (*q == ',') { tx = (int)strtol(q + 1, &q, 10); if (*q == ',') ty = (int)strtol(q + 1, &q, 10); }
                    if (*q == ',') { tx2 = (int)strtol(q + 1, &q, 10); if (*q == ',') ty2 = (int)strtol(q + 1, &q, 10); }
                }
            }
            if (th && p->code[1] && prog_code_hash(p, 1) == th) {
                extern MAXWELL_TLS int maxwell_trace_x2, maxwell_trace_y2;
                maxwell_trace_next = true; maxwell_trace_x = tx; maxwell_trace_y = ty; maxwell_trace_x2 = tx2; maxwell_trace_y2 = ty2;
            }
            static uint64_t sh = 1;  // LARECOMP_DEBUG_SOLID=<fs hash hex>: that fragment shader writes magenta, never discards
            if (sh == 1) { const char *e = getenv("LARECOMP_DEBUG_SOLID"); sh = e ? strtoull(e, NULL, 16) : 0; }
            if (sh && p->code[1] && prog_code_hash(p, 1) == sh) {
                extern MAXWELL_TLS bool maxwell_solid_next;
                maxwell_solid_next = true;
                LOG("[shader] fs hash %016llx: solid magenta (LARECOMP_DEBUG_SOLID)\n", (unsigned long long)sh);
            }
            static uint64_t nd = 1;  // LARECOMP_NO_DISCARD=<fs hash hex>: that fragment shader never discards (A/B)
            if (nd == 1) { const char *e = getenv("LARECOMP_NO_DISCARD"); nd = e && strcmp(e, "1") ? strtoull(e, NULL, 16) : 0; }
            if (nd && p->code[1] && prog_code_hash(p, 1) == nd) {
                extern MAXWELL_TLS bool maxwell_nodiscard_next;
                maxwell_nodiscard_next = true;
                LOG("[shader] fs hash %016llx: discard disabled (LARECOMP_NO_DISCARD)\n", (unsigned long long)nd);
            }
            // TotK's upscale pass (FSR 1 EASU, scene 1600x900 -> 1920x1080) reads its 12 taps as single texels at guest
            // coordinates, so with TOTK_RES_SCALE only one texel of each scaled block reached the screen: the same
            // stair-steps at every scale (2026-10-17 run420, fs 32845c90c668fdbb texelFetch). Its fetches average the
            // whole block instead = supersampling. With TOTK_RES_SCALE != 1 or TOTK_OUT_SCALE the whole pass becomes one
            // resample to the canvas (upscale_resample_glsl). TOTK_SSAA=fetch: the averaged fetches only; 0: the old pass.
            static int ssaa = -1;
            if (ssaa < 0) {
                const char *e = getenv("TOTK_SSAA"), *rs = getenv("TOTK_RES_SCALE"), *os = getenv("TOTK_OUT_SCALE");
                bool scaled = (rs && *rs && strcmp(rs, "1") && strcmp(rs, "1.0")) || (os && *os && strcmp(os, "0"));
                ssaa = e && !strcmp(e, "0") ? 0 : (e && !strcmp(e, "fetch")) || !scaled ? 1 : 2;
            }
            if (ssaa && p->code[1] && prog_code_hash(p, 1) == 0x32845c90c668fdbbull) {
                extern MAXWELL_TLS bool maxwell_ssfetch_next;
                maxwell_ssfetch_next = true;
                upscale_resample = ssaa == 2;
                LOG("[shader] fs hash 32845c90c668fdbb (upscale): %s\n", upscale_resample ? "one resample to the canvas" : "supersampled fetches");
            }
        }
        p->fi = translate_stage(p->code[1], p->cmax[1], p->csize[1], STAGE_FRAGMENT, &p->fkey);
        // (the translated shader's bindings stay: cb3, rscale and t0 are declared there too)
        if (upscale_resample && p->fi->msl && p->fi->ntex >= 1 && p->fi->uses_rscale && (p->fi->cbuf_mask & (1u << 3))) {
            free(p->fi->msl);
            p->fi->msl = strdup(upscale_resample_glsl);
        }
    }
    if (!translate_unlocked) pthread_mutex_unlock(&translate_lock);
    if (pa_tm_on) { pa_tm[0] += tl - t0; pa_tm[1] += SDL_GetTicksNS() - tl; }
    if (!p->inline_compile) {
        atomic_fetch_add(&stat_prog_builds, 1);
        atomic_fetch_add(&cum_prog_builds, 1);
        atomic_fetch_add(&stat_prog_translate_ns, SDL_GetTicksNS() - t0);   // (includes waiting for translate_lock)
    }
    if (!p->vi->msl) {
        LOG("[shader] VS translate failed: %s (code %llx)\n", p->vi->error, (unsigned long long)p->code[0]);
        atomic_store(&p->state, 2);
        return;
    }
    if (!p->fi->msl) {
        LOG("[shader] FS translate failed: %s\n", p->fi->error);
        atomic_store(&p->state, 2);
        return;
    }
    p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
    if (!p->inline_compile) {   // not the precompile's ~10k programs (the launcher's SAVE_GLSL wrote ~20k files at boot)
        save_glsl(p->vi->msl, p->code[0], "vs");
        save_glsl(p->fi->msl, p->code[1], "fs");
    }
    if (p->inline_compile) compile_program(p);
    else if (p->sync_try && vk_spirv_cached(p->vi->msl, STAGE_VERTEX) && vk_spirv_cached(p->fi->msl, STAGE_FRAGMENT)) {
        compile_program(p);   // (SPIR-V + driver cache hits after the archive warm-up: no skipped draw, no held frame)
        prog_sync_note(p, t0, true);
    } else {
        if (p->sync_try) prog_sync_note(p, t0, false);
        run_job(compile_program, p);
    }
}

#define PROG_TAB 16384
static struct { uint64_t key; Prog *prog; } prog_tab[PROG_TAB];

// shader precompile (below the pipeline code)
static bool pc_on(void);
static void pc_start(void);
static uint64_t pc_prog_key(const NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey, uint64_t code_hash[2], uint64_t cmax[2]);
static Prog *pc_claim_prog(uint64_t ckey, Prog *fresh);
static void pc_record_prog(const Prog *p, const uint64_t code_hash[2]);

// Per-draw memos (render thread only): consecutive draws mostly repeat the previous draw's program and pipeline
// (run385: 253k of 340k draws rebound nothing), but each one re-hashed the shader keys byte by byte and the ~700-byte
// pipeline key (pre 0.70 + key 0.50 us a draw). The memo compares the same inputs by value, so a hit returns what the
// table lookup would. vk_exec_reset bumps g_memo_gen (tables cleared).
static uint32_t g_memo_gen;
static Prog *get_program_slow(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey);
static Prog *get_program(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    if (!(np->stage_mask & 1) || !(np->stage_mask & 2)) return NULL;
    static struct { NvnProgram *np; uint64_t code[2]; uint32_t mask, gen; ShaderKey vkey; uint8_t rt[8]; Prog *p; } memo;
    if (memo.p && memo.gen == g_memo_gen && memo.np == np && memo.code[0] == np->stages[0].code_addr &&
        memo.code[1] == np->stages[1].code_addr && memo.mask == np->stage_mask && !memcmp(&memo.vkey, vkey, sizeof *vkey) &&
        !memcmp(memo.rt, fkey->rt_kind, 8))
        return memo.p;
    Prog *res = get_program_slow(np, vkey, fkey);
    memo.np = np; memo.code[0] = np->stages[0].code_addr; memo.code[1] = np->stages[1].code_addr; memo.mask = np->stage_mask;
    memo.gen = g_memo_gen; memo.vkey = *vkey; memcpy(memo.rt, fkey->rt_kind, 8); memo.p = res;
    return res;
}
static Prog *get_program_slow(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 2; i++) { uint64_t a = np->stages[i].code_addr; for (int k = 0; k < 8; k++) { h ^= (a >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; } }
    const uint8_t *vb = (const uint8_t *)vkey;
    for (size_t i = 0; i < sizeof *vkey; i++) { h ^= vb[i]; h *= 0x100000001B3ull; }
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    if (!h) h = 1;
    uint32_t slot = (uint32_t)(h ^ (h >> 31)) & (PROG_TAB - 1);
    while (prog_tab[slot].key && prog_tab[slot].key != h) slot = (slot + 1) & (PROG_TAB - 1);
    if (prog_tab[slot].key) return prog_tab[slot].prog;
    uint64_t gz = prof ? prof_now() : 0;
#define GPS_ZONE(i) do { if (prof) { uint64_t n_ = prof_now(); gps_ns[i] += n_ - gz; gz = n_; } } while (0)
    // same code + keys under another address, or built ahead by the shader precompile: share that program
    uint64_t chash[2], cmax[2], ck = 0;
    if (pc_on()) { pc_start(); ck = pc_prog_key(np, vkey, fkey, chash, cmax); }
    GPS_ZONE(0);
    Prog *p = calloc(1, sizeof *p);
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
        p->csize[i] = np->stages[i].const_offset ? (np->stages[i].const_size < 0x10000 ? np->stages[i].const_size : 0x10000) : 0;
    }
    p->vkey = *vkey;
    p->fkey = *fkey;
    p->ckey = ck;
    if (ck) {
        Prog *q = pc_claim_prog(ck, p);
        GPS_ZONE(1);
        if (q != p) {
            // q from a PC_PROG record (no compiler constants): record this one with them for the next session; if q is
            // built and a BRX table was unreadable (every instruction a switch case: Deck d40 cloud VS, 5.5 ms a draw),
            // use this program instead, translated with its constants
            bool q_c1 = q->csize[0] || q->csize[1], p_c1 = p->csize[0] || p->csize[1];
            if (!q_c1 && p_c1) pc_record_prog(p, chash);
            bool blind = !q_c1 && p_c1 && atomic_load(&q->state) == 1 && ((q->vi && q->vi->brx_blind) || (q->fi && q->fi->brx_blind));
            if (!blind) { free(p); prog_tab[slot].key = h; prog_tab[slot].prog = q; return q; }
            LOG("[precompile] program %016llx: precompiled without constants, BRX blind -> translated again\n", (unsigned long long)ck);
        } else pc_record_prog(p, chash);
        GPS_ZONE(2);
    }
    prog_tab[slot].key = h;
    prog_tab[slot].prog = p;
    if (prog_sync_ok()) { p->sync_try = true; translate_program(p); }
    else run_translate(translate_program, p);
    GPS_ZONE(3);
#undef GPS_ZONE
    return p;
}

// ================================================================== vertex formats / fixed function tables
static int attr_kind(uint32_t fmt) {
    switch (fmt) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23:
    case 0x27: case 0x2C: case 0x2F: case 0x3E:
        return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24:
    case 0x28: case 0x2D: case 0x30:
        return 1;
    default: return 0;
    }
}

static VkFormat vertex_format(uint32_t f) {
    switch (f) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0A: return VK_FORMAT_R32_SFLOAT;
    case 0x0B: return VK_FORMAT_R32_UINT;
    case 0x0C: return VK_FORMAT_R32_SINT;
    case 0x0D: return VK_FORMAT_R8G8_UNORM;
    case 0x0E: return VK_FORMAT_R8G8_SNORM;
    case 0x0F: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1A: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1B: return VK_FORMAT_R8G8B8_UINT;
    case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1E: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1F: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2D: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2F: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    // integer-to-float formats (NVN *_UI2F / *_I2F pairs, formats.c sizes): the vertex fetch converts the integer to
    // float = Vulkan USCALED / SSCALED. 0x64 RGBA8_I2F fell to the RGBA32F default before (16-byte reads of a 4-byte
    // attribute: TotK falling leaves / drips drawn as long vertical spikes). No 32-bit scaled formats in Vulkan.
    case 0x51: return VK_FORMAT_R8_USCALED;
    case 0x52: return VK_FORMAT_R8_SSCALED;
    case 0x53: return VK_FORMAT_R16_USCALED;
    case 0x54: return VK_FORMAT_R16_SSCALED;
    case 0x57: return VK_FORMAT_R8G8_USCALED;
    case 0x58: return VK_FORMAT_R8G8_SSCALED;
    case 0x59: return VK_FORMAT_R16G16_USCALED;
    case 0x5A: return VK_FORMAT_R16G16_SSCALED;
    case 0x5D: return VK_FORMAT_R8G8B8_USCALED;
    case 0x5E: return VK_FORMAT_R8G8B8_SSCALED;
    case 0x5F: return VK_FORMAT_R16G16B16_USCALED;
    case 0x60: return VK_FORMAT_R16G16B16_SSCALED;
    case 0x63: return VK_FORMAT_R8G8B8A8_USCALED;
    case 0x64: return VK_FORMAT_R8G8B8A8_SSCALED;
    case 0x65: return VK_FORMAT_R16G16B16A16_USCALED;
    case 0x66: return VK_FORMAT_R16G16B16A16_SSCALED;
    default: { static uint8_t told[256]; if (!told[f & 0xFF]) { told[f & 0xFF] = 1; LOG("[vk] vertex format %#x unknown: read as RGBA32F\n", f); } }
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

// 3-component 8/16-bit vertex formats aren't universally supported: read 4 components instead.
static VkFormat vertex_format_supported(VkFormat f) {
    static uint8_t known[256];  // 1 ok, 2 not
    uint32_t i = f < 256 ? f : 0;
    if (!known[i]) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
        known[i] = (p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 2;
    }
    if (known[i] == 1) return f;
    { static uint8_t told[256]; if (!told[i]++) LOG("[vk] vertex format %d unsupported: read as 4 components\n", f); }
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case VK_FORMAT_R8G8B8_USCALED: return VK_FORMAT_R8G8B8A8_USCALED;
    case VK_FORMAT_R8G8B8_SSCALED: return VK_FORMAT_R8G8B8A8_SSCALED;
    case VK_FORMAT_R16G16B16_USCALED: return VK_FORMAT_R16G16B16A16_USCALED;
    case VK_FORMAT_R16G16B16_SSCALED: return VK_FORMAT_R16G16B16A16_SSCALED;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

static VkBlendFactor blend_factor(uint8_t f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: {
        static uint8_t warned[256];
        if (!warned[f]++) LOG("[vk] unknown blend factor %#x\n", f);
        return VK_BLEND_FACTOR_ONE;
    }
    }
}

static VkBlendOp blend_op(uint8_t e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkCompareOp compare_fn(uint8_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkStencilOp stencil_op(uint8_t o) {
    switch (o) {
    case 2: return VK_STENCIL_OP_ZERO;
    case 3: return VK_STENCIL_OP_REPLACE;
    case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_INVERT;
    case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}

// colour target i's NVN format as the pass draws it: the view's format when the view sets one (BC volumes are drawn
// through RGBA16UI views: integer outputs, no blending)
static uint32_t rt_view_format(int i) {
    return st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) ? st.rt.cview[i].format : st.rt.color[i]->format;
}

static int rt_kind(uint32_t f) {
    switch (f) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x27: case 0x2C: case 0x2F: case 0x3E: return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x28: case 0x2D: case 0x30: return 1;
    default: return 0;
    }
}

// ================================================================== graphics pipelines
// Everything a pipeline bakes in, captured by value so it can be built on a worker thread.
#define ZERO_BINDING 16
typedef struct {
    Prog *prog;
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, samples;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t attr_mask;
    uint32_t stride[17], divisor[17], used_streams;
    uint8_t blend[8][8];
    uint8_t blend_on, a2c, topo_class, poly_fill, depth_clamp;
    uint32_t write_mask[8];
} PipeKey;

// Colour-format slots past ncolor are not part of the pipeline (VkPipelineRenderingCreateInfo takes ncolor formats) but
// pass_color[] keeps stale formats there from earlier passes: identical pipelines got new keys and were rebuilt (61 % of
// cache/precompile.bin's 7019 pipelines, 2026-10-15; each first-sight build held a frame = the user's micro stutters).
static void pipekey_norm(PipeKey *k) {
    for (uint32_t i = k->ncolor; i < 8; i++) k->color[i] = VK_FORMAT_UNDEFINED;
}
typedef struct { PipeKey k; uint64_t hash; VkPipeline pipe; _Atomic int state, claimed; uint64_t skip_t0; uint32_t nskip; bool linked; } Pipe;  // state: 0 building, 1 ready, 2 failed; claimed: a builder took it; skip_t0/nskip: [pipeskip] (drawing thread); linked: GPL fast link (dynamic depth clamp)
#define PIPE_TAB 32768
static struct { uint64_t key; Pipe *pipe; } pipe_tab[PIPE_TAB];

// 8 bytes a step (PipeKey is ~700 bytes, hashed per draw); in-memory tables only - hmix values are compared with
// stored constants (fragment-shader hashes) and must not change.
static uint64_t hash_words(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (; n >= 8; n -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; n; n--, b++) { h = (h ^ *b) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

static VkPrimitiveTopology topo_class_base(uint8_t c) {
    return c == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : c == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

static void build_pipeline_body(Pipe *pp);
static void build_pipeline(void *arg) {  // (worker job; a draw may have claimed and built it inline already)
    Pipe *pp = arg;
    int unclaimed = 0;
    if (atomic_compare_exchange_strong(&pp->claimed, &unclaimed, 1)) build_pipeline_body(pp);
}
// ---- GPL link (see gpl_program_libs): vertex-input and fragment-output libraries cached by their state
static bool gpl_linkable(const PipeKey *k) {
    return g_vk.gpl && k->prog && atomic_load(&k->prog->lib_state) == 2 && k->poly_fill && (!k->depth_clamp || vk_dyn_clamp) &&
           k->samples == VK_SAMPLE_COUNT_1_BIT && !k->a2c;
}
typedef struct {
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32], attr_mask, stride[17], divisor[17], used_streams, topo_class;
} GplVi;
typedef struct {
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, write_mask[8];
    uint8_t blend[8][8], blend_on;
} GplFo;
#define GPL_TAB 8192
static struct { uint64_t key; VkPipeline lib; } gpl_vi_tab[GPL_TAB], gpl_fo_tab[GPL_TAB];
static pthread_mutex_t gpl_mu = PTHREAD_MUTEX_INITIALIZER;
static VkPipeline gpl_cached(uint64_t h, bool fo, VkPipeline (*make)(const PipeKey *), const PipeKey *k) {
    if (!h) h = 1;
    __typeof__(gpl_vi_tab[0]) *t = fo ? gpl_fo_tab : gpl_vi_tab;
    pthread_mutex_lock(&gpl_mu);
    uint32_t i = (uint32_t)(h ^ (h >> 29)) & (GPL_TAB - 1);
    int probes = 0;
    while (t[i].key && t[i].key != h && probes++ < GPL_TAB) i = (i + 1) & (GPL_TAB - 1);
    VkPipeline r = VK_NULL_HANDLE;
    if (t[i].key == h) r = t[i].lib;
    else if (!t[i].key) { r = make(k); if (r) { t[i].key = h; t[i].lib = r; } }   // (cheap: no shaders; made under the lock)
    pthread_mutex_unlock(&gpl_mu);
    return r;
}
static VkPipeline gpl_make_vi(const PipeKey *k) {
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pDynamicState = &dy};
    return gpl_create(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT);
}
static VkPipeline gpl_make_fo(const PipeKey *k) {
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = k->ncolor,
                                              .pAttachments = cba};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .pColorBlendState = &cb,
                                       .pMultisampleState = &ms, .pDynamicState = &dy};
    return gpl_create(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT);
}
static bool gpl_link(Pipe *pp) {
    const PipeKey *k = &pp->k;
    GplVi v;
    memset(&v, 0, sizeof v);
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) { v.attr_fmt[i] = k->attr_fmt[i]; v.attr_off[i] = k->attr_off[i]; v.attr_bind[i] = k->attr_bind[i]; }
    for (int s = 0; s < 17; s++)
        if (k->used_streams & (1u << s)) { v.stride[s] = k->stride[s]; v.divisor[s] = k->divisor[s]; }
    v.attr_mask = k->attr_mask;
    v.used_streams = k->used_streams;
    v.topo_class = k->topo_class;
    GplFo f;
    memset(&f, 0, sizeof f);
    f.ncolor = k->ncolor;
    for (uint32_t i = 0; i < k->ncolor && i < 8; i++) {
        f.color[i] = k->color[i];
        f.write_mask[i] = k->write_mask[i];
        if ((k->blend_on >> i) & 1) memcpy(f.blend[i], k->blend[i], 8);
    }
    f.blend_on = k->blend_on;
    f.depth = k->depth;
    f.stencil = k->stencil;
    VkPipeline libs[4] = {gpl_cached(hash_words(0x243f6a8885a308d3ull, &v, sizeof v), false, gpl_make_vi, k), k->prog->lib_pre,
                          k->prog->lib_frag, gpl_cached(hash_words(0x13198a2e03707344ull, &f, sizeof f), true, gpl_make_fo, k)};
    if (!libs[0] || !libs[3]) return false;
    VkPipelineLibraryCreateInfoKHR lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4, .pLibraries = libs};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &lci, .layout = k->prog->layout};
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &pi, NULL, &pp->pipe);   // fast link (no LTO)
    atomic_fetch_add(&stat_gpl_links, 1);
    atomic_fetch_add(&stat_gpl_link_ns, SDL_GetTicksNS() - t0);
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline link failed: %d (monolithic build instead)\n", r);
        return false;
    }
    pp->linked = true;
    atomic_store(&pp->state, 1);
    return true;
}

static void build_pipeline_body(Pipe *pp) {
    PipeKey *k = &pp->k;
    if (gpl_linkable(k) && gpl_link(pp)) return;
    VkShaderModule mod[2] = {k->prog->vs, k->prog->fs}, tmp[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (k->prog->modules_dropped) {   // (compile_program dropped them after building the libraries)
        for (int s = 0; s < 2; s++) {
            size_t nw = 0;
            uint32_t *spv = vk_spirv_load(k->prog->spv_key[s], &nw);
            if (spv) { tmp[s] = vk_module(spv, nw); free(spv); }
            mod[s] = tmp[s];
        }
        if (!mod[0] || !mod[1]) {
            LOG("[vk] pipeline: program modules not in the SPIR-V cache any more (%016llx / %016llx)\n",
                (unsigned long long)k->prog->spv_key[0], (unsigned long long)k->prog->spv_key[1]);
            for (int s = 0; s < 2; s++) if (tmp[s]) vkDestroyShaderModule(g_vk.dev, tmp[s], NULL);
            atomic_store(&pp->state, 2);
            return;
        }
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mod[0], .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mod[1], .pName = "main"}};
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .depthClampEnable = k->depth_clamp,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = k->ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = k->prog->layout};
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pp->pipe);
    for (int s = 0; s < 2; s++) if (tmp[s]) vkDestroyShaderModule(g_vk.dev, tmp[s], NULL);
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline creation failed: %d\n", r);
    }
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&pp->state, r == VK_SUCCESS ? 1 : 2);
}

// ================================================================== shader precompile
// (user 2026-10-15: "make a thing to precompile all shaders for a smooth experience")
// Every graphics program the game uses (its Maxwell code, saved by content hash as cache/code/<hash>.bin) and every
// pipeline state it draws with are appended to cache/precompile.bin. At the next launch, background threads translate
// those programs and build their pipelines (from the warm VkPipelineCache) before the game asks for them; a draw finds
// them by content (code hashes + shader keys), not by guest address. Pipelines never seen before still build on first
// sight (the frame is held). TOTK_PRECOMPILE=0: off (no recording, no precompile). Progress: vk_precompile_progress
// (title bar + banner), "[precompile]" log lines. Compute programs too (PC_COMP, since 2026-10-17).
#ifdef _WIN32
__declspec(dllimport) void *__stdcall GetCurrentThread(void);
__declspec(dllimport) int __stdcall SetThreadPriority(void *thread, int priority);
#endif
enum { PC_MAGIC = 0x31435054u, PC_HEADER = 1, PC_PROG = 2, PC_PIPE = 3, PC_PROG2 = 4, PC_COMP = 5 };
typedef struct { uint32_t magic, kind, size, pad; } PcHdr;   // then `size` payload bytes
typedef struct { uint64_t heap_base, gdev_size, prog_size, pipe_size; } PcInfo;
typedef struct { uint64_t ckey, code_hash[2], cmax[2]; ShaderKey vkey, fkey; } PcProg;
typedef struct { uint64_t ckey; PipeKey k; } PcPipe;          // k.prog = NULL
// PC_PROG2 (2026-10-17): + each stage's compiler constants size; the bytes are in cache/code/<code hash>.c1. Without
// them translate.c cannot read BRX jump tables and makes every instruction a switch case: Deck d40, the cloud VS
// 7f..3f1e00 precompiled from a PC_PROG record had 2534 cases (34 when the game built it) = 5.5 ms per cloud draw.
// Programs known only from PC_PROG records are recorded again as PC_PROG2 when the game uses them.
typedef struct { PcProg p; uint64_t csize[2]; } PcProg2;
#define PC_SEEN_C1 0x9e3779b97f4a7c15ull   // (pc_seen key of a PC_PROG2 record: ckey ^ this)
// PC_COMP (2026-10-17): a compute program (code in cache/code/<hash>.bin + .c1, workgroup size). The user's PC session of
// 10-15 held frames 20 times for nothing but a compute shader's first dispatch: compute was never recorded.
typedef struct { uint64_t ckey, code_hash, cmax, csize; uint32_t local[3], pad; } PcComp;
#define PC_SEEN_CS 0xc0335a3e5ull          // (pc_seen key of a PC_COMP record: ckey ^ this)
static PcComp *pc_comps;
static int pc_ncomps;
static _Atomic int pc_next_comp, pc_comps_done;
static void pc_comp_build(const PcComp *r);   // (compute section)

static pthread_mutex_t pc_mu = PTHREAD_MUTEX_INITIALIZER;
#define PC_CTAB 16384
static struct { uint64_t key; Prog *prog; } pc_ctab[PC_CTAB];   // content key -> program (pc_mu)
#define PC_PTAB 65536
static struct { uint64_t key; Pipe *pipe; } pc_ptab[PC_PTAB];   // draw hash -> precompiled pipeline (pc_mu)
#define PC_SEEN 131072
static uint64_t pc_seen[PC_SEEN];   // recorded program content keys / pipeline recipe keys (game thread after pc_start)
static int pc_nseen;
static FILE *pc_out;
static PcProg2 *pc_progs;   // (PC_PROG records: csize 0)
static PcPipe *pc_pipes;
static int pc_nprogs, pc_npipes;
static _Atomic int pc_next_prog, pc_progs_done, pc_next_pipe, pc_pipes_done, pc_running;
static int pc_nthreads;
static uint64_t pc_t0;

static bool pc_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("TOTK_PRECOMPILE"); on = !(e && !strcmp(e, "0")); }
    return on;
}
static bool pc_seen_add(uint64_t k) {   // true if new
    if (!k) k = 1;
    uint32_t i = (uint32_t)(k ^ (k >> 31)) & (PC_SEEN - 1);
    while (pc_seen[i] && pc_seen[i] != k) i = (i + 1) & (PC_SEEN - 1);
    if (pc_seen[i]) return false;
    if (pc_nseen > PC_SEEN * 3 / 4) return false;   // (full: stop recording)
    pc_seen[i] = k;
    pc_nseen++;
    return true;
}
static uint64_t pc_info_heap(void) { return HEAP_BASE; }

// the whole range is committed, readable host memory (guest heap, alias regions: run266 recorded only 16 programs while
// the check allowed the heap alone)
static bool pc_readable(uint64_t a, uint64_t n) {
    uint64_t lo = HEAP_BASE, hi = HEAP_BASE + kernel_heap_size();
    if (lo && a >= lo && a + n <= hi) return true;
#ifdef _WIN32
    // ReadProcessMemory probe, cost ~ n like the POSIX one. VirtualQuery walked the whole committed region to compute
    // RegionSize: 3.7 ms per GB, 12 ms per GB with MEM_WRITE_WATCH (scratchpad vq.c / rpm.c 2026-10-18: 2 GB 24.5 ms vs
    // 0.011 ms for a 64 KB probe) = ~45 ms per new program on the user's PC, the open-world hitches every ~2 s.
    char probe[8192];   // (windows.h comes in through pthread.h)
    while (n) {
        SIZE_T k = n < sizeof probe ? (SIZE_T)n : sizeof probe, got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), (LPCVOID)(uintptr_t)a, probe, k, &got) || got != k) return false;
        a += k;
        n -= k;
    }
    return true;
#else
    extern int rt_posix_readable(unsigned long long a, unsigned long long n);   // runtime/linux/plat_posix.cpp (write(2) probe into a pipe: EFAULT, not a signal)
    return rt_posix_readable(a, n) != 0;
#endif
}
static _Atomic int pc_rejects;

static uint64_t pc_ckey(const uint64_t code_hash[2], const ShaderKey *vkey, const ShaderKey *fkey);
// content key: code bytes of both stages (committed host memory only) + every shader key field
static uint64_t pc_prog_key(const NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey, uint64_t code_hash[2], uint64_t cmax[2]) {
    for (int i = 0; i < 2; i++) {
        uint64_t a = np->stages[i].code_addr;
        cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
        if (cmax[i] > 0x100000 || !pc_readable(a, cmax[i])) {
            if (pc_rejects++ < 8) LOG("[precompile] not recorded: stage %d code %llx size %llx (heap %llx +%llx)\n", i,
                                      (unsigned long long)a, (unsigned long long)cmax[i], (unsigned long long)HEAP_BASE, (unsigned long long)kernel_heap_size());
            return 0;
        }
        code_hash[i] = hash_words(0x6a09e667f3bcc908ull ^ cmax[i], (const void *)(uintptr_t)a, (size_t)cmax[i]);
    }
    return pc_ckey(code_hash, vkey, fkey);
}
static uint64_t pc_ckey(const uint64_t code_hash[2], const ShaderKey *vkey, const ShaderKey *fkey) {
    const ShaderKey *ks[2] = {vkey, fkey};
    uint64_t h = hash_words(0x510e527fade682d1ull, code_hash, 16);
    for (int i = 0; i < 2; i++) {
        h = hash_words(h, ks[i]->attr_kind, 32);
        h = hash_words(h, ks[i]->rt_kind, 8);
        uint32_t f[4] = {ks[i]->depth_minus_one, ks[i]->flip_y, i == 0 ? ks[i]->vs_out_mask : 0, ks[i]->glsl};
        h = hash_words(h, f, 16);
    }
    return h ? h : 1;
}
static Prog *pc_claim_prog(uint64_t ckey, Prog *fresh) {   // the program already under this key, or `fresh` (now registered)
    pthread_mutex_lock(&pc_mu);
    uint32_t i = (uint32_t)(ckey ^ (ckey >> 29)) & (PC_CTAB - 1);
    int probes = 0;
    while (pc_ctab[i].key && pc_ctab[i].key != ckey && probes++ < PC_CTAB) i = (i + 1) & (PC_CTAB - 1);
    Prog *r = fresh;
    if (pc_ctab[i].key == ckey) r = pc_ctab[i].prog;
    else if (!pc_ctab[i].key) { pc_ctab[i].key = ckey; pc_ctab[i].prog = fresh; }
    pthread_mutex_unlock(&pc_mu);
    return r;
}
static Prog *pc_find_prog(uint64_t ckey) {
    pthread_mutex_lock(&pc_mu);
    uint32_t i = (uint32_t)(ckey ^ (ckey >> 29)) & (PC_CTAB - 1);
    int probes = 0;
    while (pc_ctab[i].key && pc_ctab[i].key != ckey && probes++ < PC_CTAB) i = (i + 1) & (PC_CTAB - 1);
    Prog *r = pc_ctab[i].key == ckey ? pc_ctab[i].prog : NULL;
    pthread_mutex_unlock(&pc_mu);
    return r;
}
static Pipe *pc_take_pipe(uint64_t ph) {
    if (!pc_on()) return NULL;
    pthread_mutex_lock(&pc_mu);
    uint32_t i = (uint32_t)(ph ^ (ph >> 29)) & (PC_PTAB - 1);
    int probes = 0;
    while (pc_ptab[i].key && pc_ptab[i].key != ph && probes++ < PC_PTAB) i = (i + 1) & (PC_PTAB - 1);
    Pipe *r = pc_ptab[i].key == ph ? pc_ptab[i].pipe : NULL;
    pthread_mutex_unlock(&pc_mu);
    return r;
}
static void pc_put_pipe(uint64_t ph, Pipe *pp) {
    pthread_mutex_lock(&pc_mu);
    uint32_t i = (uint32_t)(ph ^ (ph >> 29)) & (PC_PTAB - 1);
    int probes = 0;
    while (pc_ptab[i].key && pc_ptab[i].key != ph && probes++ < PC_PTAB) i = (i + 1) & (PC_PTAB - 1);
    if (!pc_ptab[i].key) { pc_ptab[i].key = ph; pc_ptab[i].pipe = pp; }
    pthread_mutex_unlock(&pc_mu);
}

static void pc_write(uint32_t kind, const void *payload, uint32_t size) {
    if (!pc_out) {
        mkdir("cache", 0755);
        pc_out = fopen("cache/precompile.bin", "ab");
        if (!pc_out) return;
        if (ftell(pc_out) == 0) {
            PcInfo in = {pc_info_heap(), gdev_window_size(), sizeof(PcProg), sizeof(PipeKey)};
            PcHdr h = {PC_MAGIC, PC_HEADER, sizeof in, 0};
            fwrite(&h, sizeof h, 1, pc_out);
            fwrite(&in, sizeof in, 1, pc_out);
        }
    }
    PcHdr h = {PC_MAGIC, kind, size, 0};
    fwrite(&h, sizeof h, 1, pc_out);
    fwrite(payload, size, 1, pc_out);
    fflush(pc_out);   // (the process ends with TerminateProcess: no buffered tail)
}
static void pc_record_prog(const Prog *p, const uint64_t code_hash[2]) {
    if (!p->ckey || !pc_seen_add(p->ckey ^ PC_SEEN_C1)) return;
    pc_seen_add(p->ckey);
    mkdir("cache/code", 0755);
    for (int i = 0; i < 2; i++) {
        char path[64];
        snprintf(path, sizeof path, "cache/code/%016llx.bin", (unsigned long long)code_hash[i]);
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); continue; }
        f = fopen(path, "wb");
        if (!f) return;
        bool ok = fwrite((const void *)(uintptr_t)p->code[i], 1, (size_t)p->cmax[i], f) == p->cmax[i];
        fclose(f);
        if (!ok) { remove(path); return; }
    }
    PcProg2 r;
    memset(&r, 0, sizeof r);
    for (int i = 0; i < 2; i++) {   // the compiler constants (c1) right after the code
        uint64_t n = p->csize[i];
        if (!n || !pc_readable(p->code[i] + p->cmax[i], n)) continue;
        char path[64];
        snprintf(path, sizeof path, "cache/code/%016llx.c1", (unsigned long long)code_hash[i]);
        FILE *f = fopen(path, "rb");
        if (f) { fseek(f, 0, SEEK_END); long have = ftell(f); fclose(f); if ((uint64_t)have == n) { r.csize[i] = n; continue; } }
        f = fopen(path, "wb");
        if (!f) continue;
        bool ok = fwrite((const void *)(uintptr_t)(p->code[i] + p->cmax[i]), 1, (size_t)n, f) == n;
        fclose(f);
        if (ok) r.csize[i] = n;
        else remove(path);
    }
    r.p.ckey = p->ckey;
    for (int i = 0; i < 2; i++) { r.p.code_hash[i] = code_hash[i]; r.p.cmax[i] = p->cmax[i]; }
    r.p.vkey = p->vkey;
    r.p.fkey = p->fkey;
    pc_write(PC_PROG2, &r, sizeof r);
}
static void pc_record_pipe(uint64_t ckey, const PipeKey *k) {
    PcPipe r;
    memset(&r, 0, sizeof r);
    r.ckey = ckey;
    r.k = *k;
    r.k.prog = NULL;
    uint64_t rk = hash_words(ckey, &r.k, sizeof r.k);
    if (!pc_seen_add(rk ^ 0x5bd1e995ull)) return;
    pc_write(PC_PIPE, &r, sizeof r);
}

// TOTK_PRECOMPILE_SEED=<dir> (2026-10-17): a read-only precompile.bin + code/ shipped with the build (programs, pipelines
// and compute programs recorded on other machines and sessions), loaded on every start next to cache/precompile.bin.
// Default: cache-seed/ next to cache/ when it has a precompile.bin (the PC build dir); TOTK_PRECOMPILE_SEED= (empty) = none.
static const char *pc_seed_dir(void) {
    static const char *d = (const char *)1;
    if (d == (const char *)1) {
        d = getenv("TOTK_PRECOMPILE_SEED");
        struct stat sb;
        if (!d && stat("cache-seed/precompile.bin", &sb) == 0) d = "cache-seed";
        if (d && !*d) d = NULL;
    }
    return d;
}
static FILE *pc_code_open(uint64_t hash, const char *ext) {   // cache/code/<hash>.<ext>, else the seed's
    char path[600];
    snprintf(path, sizeof path, "cache/code/%016llx.%s", (unsigned long long)hash, ext);
    FILE *f = fopen(path, "rb");
    if (!f && pc_seed_dir()) {
        snprintf(path, sizeof path, "%s/code/%016llx.%s", pc_seed_dir(), (unsigned long long)hash, ext);
        f = fopen(path, "rb");
    }
    return f;
}
static uint8_t *pc_load_code(uint64_t hash, uint64_t size) {
    FILE *f = pc_code_open(hash, "bin");
    if (!f) return NULL;
    uint8_t *b = malloc((size_t)size);
    bool ok = b && fread(b, 1, (size_t)size, f) == size;
    fclose(f);
    if (!ok) { free(b); return NULL; }
    return b;   // (kept: the program's code pointer)
}
// code (size bytes) followed by its compiler constants (cache/code/<hash>.c1, *csize bytes; *csize = 0 when missing)
static uint8_t *pc_load_code_c1(uint64_t hash, uint64_t size, uint64_t *csize) {
    if (!*csize) return pc_load_code(hash, size);
    FILE *c = pc_code_open(hash, "bin"), *k = pc_code_open(hash, "c1");
    uint8_t *b = c ? malloc((size_t)(size + *csize)) : NULL;
    bool ok = b && fread(b, 1, (size_t)size, c) == size;
    if (ok && !(k && fread(b + size, 1, (size_t)*csize, k) == *csize)) *csize = 0;   // (no constants: code only)
    if (c) fclose(c);
    if (k) fclose(k);
    if (!ok) { free(b); return NULL; }
    return b;
}

// ---- archive precompile (user 2026-10-15: "at the start of every playthrough, compiles every major shader in the game,
// it can run in the background during the opening"). tools/shader_pack.py packs every VS+FS program of the game's
// shader archives (romfs/Shader/*.bfsha: 8712 programs, 9131 stage blobs; material = 8412) into
// cache/archive_shaders.bin. After the recorded programs and pipelines, the workers translate each one with the key
// the game will most likely use and compile its SPIR-V into the disk cache (only what is missing). No modules are kept
// (~17k stages x ~140 KB SPIR-V). A program the game asks for later then costs translation + a cache read instead of
// a GLSL compile. Measured 2026-10-15: 864 of 1813 recorded programs come from these archives (analysis/bfsha/
// match_runtime_output.txt); the rest (post effects, particles, UI) are covered by the recording only.
// TOTK_PRECOMPILE_ARCHIVE=0: off.
static uint8_t *pa_file;
static uint32_t pa_nblobs, pa_nprogs;
static const uint32_t *pa_blob;   // pa_nblobs x {offset, size}
static const uint8_t *pa_prog;    // pa_nprogs x {u32 vs, u32 fs, u8 attr_kind[32], u32 model}
static const uint8_t *pa_code;
static uint64_t *pa_known;        // sorted content keys of the recorded programs (built in full by the program phase)
static int pa_nknown;
static _Atomic int pa_next, pa_done, pa_compiled, pa_cached, pa_recorded, pa_failed;
static _Atomic uint64_t pa_translate_ns, pa_spirv_ns;
static uint64_t pa_t0;
static bool pa_full;   // TOTK_PRECOMPILE_ARCHIVE=full: complete programs (modules + GPL libraries), not just SPIR-V
static bool pa_warm;   // default (GPL on): libraries built once into the pipeline cache, then destroyed; =spirv: SPIR-V only
enum { PA_PROG = 44 };

static int pa_cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static void pa_load(void) {
    const char *e = getenv("TOTK_PRECOMPILE_ARCHIVE");
    if (e && !strcmp(e, "0")) return;
    pa_full = e && !strcmp(e, "full");
    pa_warm = !(e && !strcmp(e, "spirv"));
    FILE *f = fopen("cache/archive_shaders.bin", "rb");
    if (!f && pc_seed_dir()) {   // (shipped with the build: the Deck package had none, so its archive warm-up never ran)
        char path[600];
        snprintf(path, sizeof path, "%s/archive_shaders.bin", pc_seed_dir());
        f = fopen(path, "rb");
    }
    if (!f) { LOG("[precompile] no cache/archive_shaders.bin (tools/shader_pack.py): archive precompile off\n"); return; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = n > 16 ? malloc((size_t)n) : NULL;
    bool ok = d && fread(d, 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    uint32_t h[4];
    if (ok) memcpy(h, d, 16);
    ok = ok && h[0] == 0x50534b54u && h[1] == 1;   // 'TKSP' v1
    uint64_t o = 16 + (uint64_t)h[2] * 8 + (uint64_t)h[3] * PA_PROG;
    uint32_t nm = 0;
    if (ok && o + 4 <= (uint64_t)n) { memcpy(&nm, d + o, 4); o += 4 + (uint64_t)nm * 64; }
    ok = ok && o <= (uint64_t)n;
    for (uint32_t i = 0; ok && i < h[2]; i++) {   // every blob inside the file
        uint32_t bo, bs;
        memcpy(&bo, d + 16 + i * 8, 4);
        memcpy(&bs, d + 16 + i * 8 + 4, 4);
        ok = bs >= 0x80 && o + bo + bs <= (uint64_t)n;
    }
    if (!ok) { free(d); LOG("[precompile] cache/archive_shaders.bin: bad file, archive precompile off\n"); return; }
    pa_file = d;
    pa_nblobs = h[2];
    pa_nprogs = h[3];
    pa_blob = (const uint32_t *)(d + 16);
    pa_prog = d + 16 + (size_t)pa_nblobs * 8;
    pa_code = d + o;
    pa_known = malloc(sizeof *pa_known * (pc_nprogs + 1));
    for (int i = 0; i < pc_nprogs; i++) pa_known[pa_nknown++] = pc_progs[i].p.ckey;
    qsort(pa_known, pa_nknown, sizeof *pa_known, pa_cmp_u64);
    LOG("[precompile] archive: %u programs, %u stage blobs (%.1f MB)\n", pa_nprogs, pa_nblobs, (n - (long)o) / 1048576.0);
    // TOTK_WARMUP_START=<index> (compile-speed A/B runs): start at that program, so each run compiles programs no earlier
    // run put into the driver's caches
    const char *st = getenv("TOTK_WARMUP_START");
    if (st && atoi(st) > 0 && atoi(st) < (int)pa_nprogs) atomic_store(&pa_next, atoi(st));
}
static void pa_warm_load(void);
static uint32_t pa_nwarmed;   // (archive programs in cache/archive_warm.bin, below)
static void pa_load_all(void) {
    pa_load();
    if (pa_nprogs) {
        pa_warm_load();
        LOG("[precompile] archive warm-up record: %u programs already warmed (skipped)\n", pa_nwarmed);
    }
}
// Warm-up record (user 2026-10-15: "stop clearing the cache for every new build"). The GPL warm-up rebuilt all 16354
// archive programs every launch (the user's 22:16 session: 280 s on 5 threads, pipeline cache +242 KB = nearly all
// driver-cache hits, "0 already cached" because the warm path never checked). cache/archive_warm.bin lists the content
// keys already warmed into cache/vk_pipelines.bin; it is valid only with a loaded pipeline cache. Keys are committed after
// the pipeline cache that contains them is on disk (device.c vk_pipeline_cache_save -> vk_archive_warm_commit).
// TOTK_PRECOMPILE_REWARM=1 ignores the record. The record used to be tied to translate.c's build time, so every rebuild
// re-warmed all 16k programs (user 22:59: "its compiling AGAIN"). Now it survives rebuilds: a translator fix changes the
// GLSL of a few programs, which then compile at first use (SPIR-V cache miss) as before the warm-up existed; programs that
// failed to translate are never recorded, so they are retried every launch. Bump PA_WARM_EPOCH only for a change that
// alters the GLSL of most programs (shared prologue, SPIR-V options); the float-controls mode is part of the stamp.
static uint64_t *pa_warmed;   // sorted, from the file (pa_nwarmed entries)
static uint64_t *pa_wnew;     // warmed this session, in completion order
static uint32_t pa_nwnew, pa_capwnew, pa_ncommitted;
static pthread_mutex_t pa_wlock = PTHREAD_MUTEX_INITIALIZER;
enum { PA_WARM_STAMP = 32 };
#define PA_WARM_EPOCH "warm epoch 4"   // (2: 2026-10-19, drops the stale shader generations from the pipeline cache once;
                                       // 3: 2026-10-20, translate.c builtin bit casts change nearly every program's GLSL;
                                       // 4: 2026-10-20, one precise pf_, fmin/fmax/saturate as selects, integer attributes)
static void pa_warm_stamp(char *s) {
    extern int vk_spirv_float_preserve(void);
    snprintf(s, PA_WARM_STAMP, "%s%s%s%s", PA_WARM_EPOCH, vk_spirv_float_preserve() ? " fp" : "", gpl_shared() ? " gs" : "",
             g_vk.feedback_loop ? " fl" : "");   // (gs: shared stage libraries)
}
static void pa_warm_load(void) {
    extern size_t g_vk_pcache_loaded;
    if (getenv("TOTK_PRECOMPILE_REWARM") || !g_vk_pcache_loaded) return;
    FILE *f = fopen("cache/archive_warm.bin", "rb");
    if (!f) return;
    char hdr[8 + PA_WARM_STAMP], want[PA_WARM_STAMP] = {0};
    uint32_t n = 0;
    pa_warm_stamp(want);
    bool ok = fread(hdr, 1, sizeof hdr, f) == sizeof hdr && !memcmp(hdr, "TKWM\1\0\0\0", 8) && fread(&n, 4, 1, f) == 1;
    if (ok && memcmp(hdr + 8, want, PA_WARM_STAMP)) {
        LOG("[precompile] archive warm-up record stamp \"%.32s\", want \"%s\": warming again\n", hdr + 8, want);
        ok = false;
    }
    if (ok && n) {
        pa_warmed = malloc(sizeof *pa_warmed * n);
        ok = fread(pa_warmed, 8, n, f) == n;
        if (ok) pa_nwarmed = n;
        else { free(pa_warmed); pa_warmed = NULL; }
    }
    fclose(f);
}
// device.c, before loading the pipeline cache: a record with another stamp means every archive program is rebuilt in a
// new form, so the cached pipelines are a previous generation that only costs RAM (2026-10-19: a 5.3 GB file of 3
// generations = ~7.5 GB of driver memory plus a 5.3 GB copy per save -> low-RAM watchdog kills).
bool vk_archive_warm_stale(void) {
    FILE *f = fopen("cache/archive_warm.bin", "rb");
    if (!f) return false;
    char hdr[8 + PA_WARM_STAMP], want[PA_WARM_STAMP] = {0};
    pa_warm_stamp(want);
    bool stale = fread(hdr, 1, sizeof hdr, f) == sizeof hdr && !memcmp(hdr, "TKWM\1\0\0\0", 8) && memcmp(hdr + 8, want, PA_WARM_STAMP);
    fclose(f);
    return stale;
}
// device.c, right after a stale record dropped the old pipeline cache: an empty record with the current stamp. Without it the
// old stamp stayed until a session committed its first archive program (first save 3 min into the warm-up), so every
// shorter session after an update threw away the cache the previous one saved (Deck 2026-10-20: five test runs in a row
// "previous shader generation, starting empty").
void vk_archive_warm_reset(void) {
    char hdr[8 + PA_WARM_STAMP] = "TKWM\1\0\0\0";
    uint32_t zero = 0;
    pa_warm_stamp(hdr + 8);
    FILE *f = fopen("cache/archive_warm.bin.tmp", "wb");
    bool ok = f && fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr && fwrite(&zero, 4, 1, f) == 1;
    if (f) fclose(f);
    if (ok) { remove("cache/archive_warm.bin"); ok = rename("cache/archive_warm.bin.tmp", "cache/archive_warm.bin") == 0; }
    LOG("[precompile] archive warm-up record reset to the current shader generation%s\n", ok ? "" : " (write FAILED)");
}
static bool pa_is_warmed(uint64_t ck) { return pa_nwarmed && bsearch(&ck, pa_warmed, pa_nwarmed, sizeof ck, pa_cmp_u64); }
static void pa_warm_add(uint64_t ck) {
    pthread_mutex_lock(&pa_wlock);
    if (pa_nwnew == pa_capwnew) { pa_capwnew = pa_capwnew ? pa_capwnew * 2 : 4096; pa_wnew = realloc(pa_wnew, sizeof *pa_wnew * pa_capwnew); }
    pa_wnew[pa_nwnew++] = ck;
    pthread_mutex_unlock(&pa_wlock);
}
// "[precompile] warm-up stages": per built program, ms of thread time in each stage (see pa_tm), and translate_lock's busy
// share of the wall time since the warm-up began (near 100 % = the single translator limits the warm-up)
static void pa_tm_log(void) {
    uint64_t n = atomic_load(&pa_tm_n);
    if (!n) return;
    double w = (SDL_GetTicksNS() - (pa_t0 ? pa_t0 : pc_t0)) / 1e9, d = (double)n * 1e6, s[8];
    for (int k = 0; k < 8; k++) s[k] = (double)atomic_load(&pa_tm_sum[k]);
    if (w <= 0) w = 1;
    LOG("[precompile] warm-up stages: %llu programs in %.0f s (%.1f/s): ms per program: lock wait %.1f, translate %.1f, "
        "GLSL->SPIR-V %.1f (%.0f %% of stages cached), modules %.1f, GPL libraries %.1f (%.0f %% of %.0f cache hits%s); "
        "translate_lock busy %.0f %%\n",
        (unsigned long long)n, w, n / w, s[0] / d, s[1] / d, s[2] / d, s[5] * 50.0 / n, s[3] / d, s[4] / d,
        s[7] ? s[6] * 100.0 / s[7] : 0.0, s[7], gpl_shared() ? ", shared" : "", s[1] / 1e9 / w * 100.0);
}
// device.c: before reading the pipeline cache data (keys warmed so far are inside it) ...
uint32_t vk_archive_warm_snapshot(void) {
    pthread_mutex_lock(&pa_wlock);
    uint32_t n = pa_nwnew;
    pthread_mutex_unlock(&pa_wlock);
    return n;
}
// ... and after that data is on disk (or the cache did not change since the last file): record them.
void vk_archive_warm_commit(uint32_t n) {
    pthread_mutex_lock(&pa_wlock);
    if (n > pa_ncommitted) {
        uint32_t total = pa_nwarmed + n;
        uint64_t *all = malloc(sizeof *all * total);
        if (pa_nwarmed) memcpy(all, pa_warmed, sizeof *all * pa_nwarmed);
        memcpy(all + pa_nwarmed, pa_wnew, sizeof *all * n);
        qsort(all, total, sizeof *all, pa_cmp_u64);
        char hdr[8 + PA_WARM_STAMP] = "TKWM\1\0\0\0";
        pa_warm_stamp(hdr + 8);
        FILE *f = fopen("cache/archive_warm.bin.tmp", "wb");
        bool ok = f && fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr && fwrite(&total, 4, 1, f) == 1 &&
                  fwrite(all, 8, total, f) == total;
        if (f) fclose(f);
        if (ok) { remove("cache/archive_warm.bin"); ok = rename("cache/archive_warm.bin.tmp", "cache/archive_warm.bin") == 0; }
        if (ok) pa_ncommitted = n;
        LOG("[precompile] archive warm-up record: %u programs%s\n", total, ok ? "" : " (write FAILED)");
        pa_tm_log();
        free(all);
    }
    pthread_mutex_unlock(&pa_wlock);
}

static void pa_job(int i) {
    const uint8_t *r = pa_prog + (size_t)i * PA_PROG;
    uint32_t ix[2];
    memcpy(ix, r, 8);
    if (ix[0] >= pa_nblobs || ix[1] >= pa_nblobs) { atomic_fetch_add(&pa_failed, 1); return; }
    ShaderKey vkey = {0}, fkey = {0};   // as draw_() builds them
    memcpy(vkey.attr_kind, r + 8, 32);
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    const uint8_t *code[2];
    uint64_t cmax[2], chash[2];
    for (int s = 0; s < 2; s++) {
        code[s] = pa_code + pa_blob[ix[s] * 2];
        cmax[s] = pa_blob[ix[s] * 2 + 1];
        chash[s] = hash_words(0x6a09e667f3bcc908ull ^ cmax[s], code[s], (size_t)cmax[s]);
    }
    uint64_t ck = pc_ckey(chash, &vkey, &fkey);
    if (bsearch(&ck, pa_known, pa_nknown, sizeof ck, pa_cmp_u64)) { atomic_fetch_add(&pa_recorded, 1); return; }
    if (!pa_full && pa_warm && g_vk.gpl && pa_is_warmed(ck)) { atomic_fetch_add(&pa_cached, 1); return; }
    if (pa_full || (pa_warm && g_vk.gpl)) {
        // full: a complete program (modules -> pipeline libraries) the game's draws pick up by content key (~1.3 MB of
        // driver memory per program: run345 vs 347, so ~10 GB for the archive - opt-in only).
        // warm (default with GPL): the same build, then everything is destroyed again; the compiled libraries stay in
        // the driver's pipeline cache, so the game's own build of the program later is a cache hit.
        Prog *p = calloc(1, sizeof *p);
        p->vkey = vkey;
        p->fkey = fkey;
        p->ckey = ck;
        p->inline_compile = true;
        for (int s = 0; s < 2; s++) { p->code[s] = (uint64_t)(uintptr_t)code[s]; p->cmax[s] = cmax[s]; }   // (pa_file stays)
        if (pa_full && pc_claim_prog(ck, p) != p) { free(p); atomic_fetch_add(&pa_recorded, 1); return; }
        uint64_t t0 = SDL_GetTicksNS();
        static int par = -1;
        if (par < 0) { const char *e = getenv("LARECOMP_PARALLEL_TRANSLATE"); par = e && *e == '1'; }
        memset(pa_tm, 0, sizeof pa_tm);
        pa_tm_on = true;
        translate_unlocked = par;
        translate_program(p);
        translate_unlocked = false;
        pa_tm_on = false;
        for (int k = 0; k < 8; k++) atomic_fetch_add(&pa_tm_sum[k], pa_tm[k]);
        atomic_fetch_add(&pa_tm_n, 1);
        atomic_fetch_add(&pa_spirv_ns, SDL_GetTicksNS() - t0);
        atomic_fetch_add(atomic_load(&p->state) == 1 ? &pa_compiled : &pa_failed, 1);
        if (!pa_full && atomic_load(&p->state) == 1 && p->lib_pre && p->lib_frag) pa_warm_add(ck);
        if (!pa_full) {
            if (p->lib_pre) vkDestroyPipeline(g_vk.dev, p->lib_pre, NULL);
            if (p->lib_frag) vkDestroyPipeline(g_vk.dev, p->lib_frag, NULL);
            if (p->vs) vkDestroyShaderModule(g_vk.dev, p->vs, NULL);
            if (p->fs) vkDestroyShaderModule(g_vk.dev, p->fs, NULL);
            if (p->layout) vkDestroyPipelineLayout(g_vk.dev, p->layout, NULL);
            for (int s = 0; s < 2; s++) if (p->lib_layout[s]) vkDestroyPipelineLayout(g_vk.dev, p->lib_layout[s], NULL);
            for (int s = 0; s < 2; s++) if (p->set_layout[s]) vkDestroyDescriptorSetLayout(g_vk.dev, p->set_layout[s], NULL);
            maxwell_free(p->vi);
            maxwell_free(p->fi);
            free(p);
        }
        return;
    }
    uint64_t t0 = SDL_GetTicksNS();
    pthread_mutex_lock(&translate_lock);
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    maxwell_gdev_size = gdev_window_size();
    ShaderInfo *vi = maxwell_translate(code[0], (size_t)cmax[0], STAGE_VERTEX, &vkey), *fi = NULL;
    if (vi->msl) {
        fkey.vs_out_mask = vi->out_mask;
        fi = maxwell_translate(code[1], (size_t)cmax[1], STAGE_FRAGMENT, &fkey);
    }
    pthread_mutex_unlock(&translate_lock);
    uint64_t t1 = SDL_GetTicksNS();
    atomic_fetch_add(&pa_translate_ns, t1 - t0);
    if (!vi->msl || !fi || !fi->msl) {
        if (atomic_fetch_add(&pa_failed, 1) < 5)
            LOG("[precompile] archive program %d (blobs %u/%u): translate failed: %s\n", i, ix[0], ix[1], !vi->msl ? vi->error : fi ? fi->error : "?");
    } else if (vk_spirv_cached(vi->msl, STAGE_VERTEX) && vk_spirv_cached(fi->msl, STAGE_FRAGMENT)) {
        atomic_fetch_add(&pa_cached, 1);
    } else {
        size_t nw;
        uint32_t *sv = vk_glsl_to_spirv(vi->msl, STAGE_VERTEX, "archive_vs", &nw);
        uint32_t *sf = vk_glsl_to_spirv(fi->msl, STAGE_FRAGMENT, "archive_fs", &nw);
        atomic_fetch_add(sv && sf ? &pa_compiled : &pa_failed, 1);
        free(sv);
        free(sf);
    }
    atomic_fetch_add(&pa_spirv_ns, SDL_GetTicksNS() - t1);
    maxwell_free(vi);
    maxwell_free(fi);
}

// TOTK_WARMUP=1 (2026-10-17): warm-up only. The archive warm-up runs on every CPU but one, unthrottled, then the pipeline
// cache is saved and the process exits ("[precompile] warm-up finished"). Deck s1_seed: the warm-up in the background of
// play took 90 % of a core and the Deck fell from a locked 60 to 55-57 fps, so on <= 8 CPUs the in-play warm-up now
// runs at TOTK_PRECOMPILE_ARCHIVE_DUTY percent of one core (default 20 there, 100 elsewhere) and the warm-up mode does the
// bulk once (tools/linux/warmup.sh).
// "Play now" (user 2026-10-20: "add an option to start game right now and the compilation will keep going in the
// background"): A on the warm-up screen (hle_hid.cpp) releases the warm-up mode in place. Normal pacing returns, nothing
// exits at the end, and TOTK_WARMUP_PLAY_THREADS (default 1) workers keep compiling the archive at
// TOTK_WARMUP_PLAY_DUTY percent (default 100) at the lowest priority; the other workers stop after their current program.
static _Atomic int pa_released;
static bool pa_warmup_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("TOTK_WARMUP"); m = e && *e && strcmp(e, "0"); }
    return m && !atomic_load(&pa_released);
}
int vk_precompile_warmup_active(void) { return pa_warmup_mode(); }   // (hle_vi.cpp pacing, hle_hid.cpp "play now")
void vk_precompile_play_now(void) {
    if (!pa_warmup_mode()) return;
    atomic_store(&pa_released, 1);
    LOG("[precompile] play now: warm-up released at %u of %u archive programs, compiling on in the background\n",
        (unsigned)atomic_load(&pa_done), (unsigned)pa_nprogs);
}
static void pc_yield(void) {   // (pause while the game's own shader jobs are queued or running)
    while (atomic_load(&g_jobs_pending) > 0) SDL_Delay(3);
}
static void *pc_worker(void *arg) {
    const int widx = (int)(intptr_t)arg;   // (worker index: "play now" keeps the first TOTK_WARMUP_PLAY_THREADS)
#ifndef _WIN32
    { extern int pthread_setname_np(pthread_t, const char *); pthread_setname_np(pthread_self(), "vk-precompile"); }   // ([cpu] line)
#endif
    // THREAD_PRIORITY_BELOW_NORMAL: the game's own builds go first (Linux: runtime/linux shim = nice +5 for this thread;
    // matters on the Steam Deck's 4 cores, where these workers otherwise compete with the game's threads)
    SetThreadPriority(GetCurrentThread(), -1);
    for (;;) {   // compute programs first (few; each one's first dispatch otherwise holds a frame)
        pc_yield();
        int i = atomic_fetch_add(&pc_next_comp, 1);
        if (i >= pc_ncomps) break;
        pc_comp_build(&pc_comps[i]);
        atomic_fetch_add(&pc_comps_done, 1);
    }
    for (;;) {   // programs: translated + compiled on this thread (translation itself is serialised by translate_lock)
        pc_yield();
        int i = atomic_fetch_add(&pc_next_prog, 1);
        if (i >= pc_nprogs) break;
        const PcProg *r = &pc_progs[i].p;
        Prog *p = calloc(1, sizeof *p);
        p->vkey = r->vkey;
        p->fkey = r->fkey;
        p->ckey = r->ckey;
        p->inline_compile = true;
        bool ok = true;
        for (int s = 0; s < 2; s++) {
            uint64_t cs = pc_progs[i].csize[s];
            uint8_t *b = pc_load_code_c1(r->code_hash[s], r->cmax[s], &cs);
            if (!b) { ok = false; break; }
            p->code[s] = (uint64_t)(uintptr_t)b;
            p->cmax[s] = r->cmax[s];
            p->csize[s] = cs;
        }
        if (ok && pc_claim_prog(r->ckey, p) == p) translate_program(p);
        else free(p);   // (the game got there first; its program serves the pipelines below)
        atomic_fetch_add(&pc_progs_done, 1);
    }
    while (atomic_load(&pc_progs_done) < pc_nprogs) SDL_Delay(5);
    for (;;) {   // pipelines, keyed exactly as draw() keys them
        pc_yield();
        int i = atomic_fetch_add(&pc_next_pipe, 1);
        if (i >= pc_npipes) break;
        const PcPipe *r = &pc_pipes[i];
        Prog *p = pc_find_prog(r->ckey);
        uint64_t t0 = SDL_GetTicksNS();
        while (p && atomic_load(&p->state) == 0 && SDL_GetTicksNS() - t0 < 10000000000ull) SDL_Delay(2);
        if (p && atomic_load(&p->state) == 1) {
            Pipe *pp = calloc(1, sizeof *pp);
            pp->k = r->k;
            pp->k.prog = p;
            uint64_t ph = hash_words(0xcbf29ce484222325ull, &pp->k, sizeof pp->k);
            if (!ph) ph = 1;
            pp->claimed = 1;
            if (pc_take_pipe(ph)) free(pp);
            else { pc_put_pipe(ph, pp); build_pipeline_body(pp); }
        }
        atomic_fetch_add(&pc_pipes_done, 1);
    }
    static _Atomic int phase1_done;
    if (atomic_fetch_add(&phase1_done, 1) == pc_nthreads - 1) {
        LOG("[precompile] done: %d programs, %d pipelines in %.1f s\n", pc_nprogs, pc_npipes, (SDL_GetTicksNS() - pc_t0) / 1e9);
        pa_t0 = SDL_GetTicksNS();
    }
    if (pa_nprogs) {   // archive programs last, at the lowest priority
        SetThreadPriority(GetCurrentThread(), -2);   // THREAD_PRIORITY_LOWEST (Linux shim: nice +10)
        static long duty = -1;
        if (duty < 0) {
            const char *e = getenv("TOTK_PRECOMPILE_ARCHIVE_DUTY");
            duty = e ? atol(e) : SDL_GetNumLogicalCPUCores() <= 8 ? 20 : 100;
            if (duty < 1) duty = 1;
            if (pa_warmup_mode() || duty > 100) duty = 100;
        }
        static long play_thr = -1, play_duty = -1;
        if (play_thr < 0) {
            const char *e = getenv("TOTK_WARMUP_PLAY_THREADS"), *d = getenv("TOTK_WARMUP_PLAY_DUTY");
            play_thr = e ? atol(e) : 1;
            play_duty = d ? atol(d) : 100;
            if (play_duty < 1) play_duty = 1;
            if (play_duty > 100) play_duty = 100;
        }
        for (;;) {
            pc_yield();
            bool released = atomic_load(&pa_released);
            if (released && widx >= play_thr) break;   // ("play now": this worker steps back)
            int i = atomic_fetch_add(&pa_next, 1);
            if (i >= (int)pa_nprogs) break;
            uint64_t t0 = SDL_GetTicksNS();
            long duty_now = released ? play_duty : duty;
            pa_job(i);
            atomic_fetch_add(&pa_done, 1);
            {   // the stage timers every 30 s too (a short test run can end before any warm-up record is written)
                static _Atomic uint64_t last_log;
                uint64_t now = SDL_GetTicksNS(), l = atomic_load(&last_log);
                if (!l) atomic_compare_exchange_strong(&last_log, &l, now);
                else if (now - l > 30000000000ull && atomic_compare_exchange_strong(&last_log, &l, now)) pa_tm_log();
            }
            if (duty_now < 100) {   // (idle (100 - duty) / duty times the job's own time)
                uint64_t ms = (SDL_GetTicksNS() - t0) / 1000000ull * (uint64_t)(100 - duty_now) / (uint64_t)duty_now;
                if (ms) SDL_Delay((uint32_t)(ms < 2000 ? ms : 2000));
            }
        }
    }
    bool last = atomic_fetch_sub(&pc_running, 1) == 1;
    if (last && pa_nprogs)
        LOG("[precompile] archive done: %u programs in %.1f s: %d compiled to SPIR-V, %d already cached, %d recorded (built in "
            "full above), %d failed; thread time translate %.1f s, SPIR-V %.1f s\n", pa_nprogs,
            (SDL_GetTicksNS() - (pa_t0 ? pa_t0 : pc_t0)) / 1e9, atomic_load(&pa_compiled), atomic_load(&pa_cached),
            atomic_load(&pa_recorded), atomic_load(&pa_failed), atomic_load(&pa_translate_ns) / 1e9, atomic_load(&pa_spirv_ns) / 1e9);
    if (last) pa_tm_log();
    if (last && pa_warmup_mode()) {
        extern void vk_pipeline_cache_save(void);
        vk_pipeline_cache_save();   // (commits the warm-up record too)
        LOG("[precompile] warm-up finished: exiting (TOTK_WARMUP)\n");
        fflush(stdout); fflush(stderr);
        _exit(0);
    }
    return NULL;
}

static void pc_start_workers(void);
static int pc_load_file(const char *path);
// Loads cache/precompile.bin (first program request: the heap base the translator bakes in is known by then) and
// starts the workers. A file written for another heap base / gdev window / record layout is set aside as .old.
static void pc_start(void) {
    static bool started;
    if (started) return;
    started = true;
    pc_t0 = SDL_GetTicksNS();
    int own = pc_load_file("cache/precompile.bin");
    if (own < 0) LOG("[precompile] no cache/precompile.bin yet: recording this session\n");
    if (own == 0) {
        LOG("[precompile] cache/precompile.bin was written for another heap base / layout: set aside, recording anew\n");
        remove("cache/precompile.bin.old");
        rename("cache/precompile.bin", "cache/precompile.bin.old");
    }
    if (pc_seed_dir()) {
        char path[600];
        snprintf(path, sizeof path, "%s/precompile.bin", pc_seed_dir());
        int np = pc_nprogs, nl = pc_npipes, nc = pc_ncomps;
        int r = pc_load_file(path);
        LOG("[precompile] seed %s: %s, +%d programs, +%d pipelines, +%d compute programs\n", path,
            r < 0 ? "missing" : r ? "loaded" : "another heap base / layout, ignored", pc_nprogs - np, pc_npipes - nl, pc_ncomps - nc);
    }
    pc_start_workers();
}
// one precompile.bin into pc_progs / pc_pipes / pc_comps (records seen before are skipped): 1 loaded, 0 written for
// another heap base / gdev window / record layout, -1 missing or unreadable
static int pc_load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = n > 0 ? malloc((size_t)n) : NULL;
    bool ok = d && fread(d, 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    if (!ok) { free(d); return -1; }
    PcInfo want = {pc_info_heap(), gdev_window_size(), sizeof(PcProg), sizeof(PipeKey)};
    static int cap_p, cap_l, cap_c;
    bool good = false;
    for (int pass = 0; pass < 2; pass++)   // PC_PROG2 records + pipelines first; then PC_PROG ones no PC_PROG2 replaced
    for (long o = 0; o + (long)sizeof(PcHdr) <= n;) {
        PcHdr h;
        memcpy(&h, d + o, sizeof h);
        if (h.magic != PC_MAGIC || o + (long)sizeof h + (long)h.size > n) break;   // (torn tail)
        const uint8_t *pl = d + o + sizeof h;
        if (h.kind == PC_HEADER && h.size == sizeof(PcInfo)) good = !memcmp(pl, &want, sizeof want);
        else if (good && ((pass == 0 && h.kind == PC_PROG2 && h.size == sizeof(PcProg2)) ||
                          (pass == 1 && h.kind == PC_PROG && h.size == sizeof(PcProg)))) {
            if (pc_nprogs == cap_p) { cap_p = cap_p ? cap_p * 2 : 1024; pc_progs = realloc(pc_progs, cap_p * sizeof *pc_progs); }
            memset(&pc_progs[pc_nprogs], 0, sizeof *pc_progs);
            memcpy(&pc_progs[pc_nprogs], pl, h.size);
            uint64_t ck = pc_progs[pc_nprogs].p.ckey;
            if (pc_seen_add(ck)) { if (pass == 0) pc_seen_add(ck ^ PC_SEEN_C1); pc_nprogs++; }
        } else if (good && pass == 0 && h.kind == PC_PIPE && h.size == sizeof(PcPipe)) {
            if (pc_npipes == cap_l) { cap_l = cap_l ? cap_l * 2 : 4096; pc_pipes = realloc(pc_pipes, cap_l * sizeof *pc_pipes); }
            memcpy(&pc_pipes[pc_npipes], pl, sizeof(PcPipe));
            pipekey_norm(&pc_pipes[pc_npipes].k);   // (records written before the fix carry stale slots: dedupe them)
            uint64_t rk = hash_words(pc_pipes[pc_npipes].ckey, &pc_pipes[pc_npipes].k, sizeof(PipeKey));
            if (pc_seen_add(rk ^ 0x5bd1e995ull)) pc_npipes++;
        } else if (good && pass == 0 && h.kind == PC_COMP && h.size == sizeof(PcComp)) {
            if (pc_ncomps == cap_c) { cap_c = cap_c ? cap_c * 2 : 256; pc_comps = realloc(pc_comps, cap_c * sizeof *pc_comps); }
            memcpy(&pc_comps[pc_ncomps], pl, sizeof(PcComp));
            if (pc_seen_add(pc_comps[pc_ncomps].ckey ^ PC_SEEN_CS)) pc_ncomps++;
        }
        o += sizeof h + h.size;
    }
    free(d);
    return good;
}
static void pc_start_workers(void) {
    pa_load_all();
    int ncpu = SDL_GetNumLogicalCPUCores(), nt = ncpu / 4;
    if (nt < 2) nt = 2;
    if (nt > 6) nt = 6;
    if (ncpu <= 8) nt = 1;   // (Steam Deck: 4 cores / 8 threads, the game's own threads need them)
    if (pa_warmup_mode()) nt = ncpu > 2 ? ncpu - 1 : 1;
    LOG("[precompile] %d programs, %d pipelines, %d compute programs from earlier sessions, %u archive programs: building on %d threads\n",
        pc_nprogs, pc_npipes, pc_ncomps, pa_nprogs, nt);
    if (!pc_nprogs && !pc_npipes && !pa_nprogs && !pc_ncomps) return;
    pc_nthreads = nt;
    atomic_store(&pc_running, nt);
    for (int i = 0; i < nt; i++) { pthread_t t; pthread_create(&t, NULL, pc_worker, (void *)(intptr_t)i); }
}

// 1 while the precompile runs; *done / *total count programs + pipelines + archive programs (device.c title bar,
// hle_osd.cpp banner)
int vk_precompile_progress(int *done, int *total) {
    *total = pc_ncomps + pc_nprogs + pc_npipes + (int)pa_nprogs;
    *done = atomic_load(&pc_comps_done) + atomic_load(&pc_progs_done) + atomic_load(&pc_pipes_done) + atomic_load(&pa_done);
    return atomic_load(&pc_running) > 0;
}
// the recorded programs + pipelines only (the part a hitch-free start needs; mostly cache loads): the in-game banner.
// The archive warm-up after it runs silently in the background (title bar only).
int vk_precompile_recorded_progress(int *done, int *total) {
    *total = pc_ncomps + pc_nprogs + pc_npipes;
    *done = atomic_load(&pc_comps_done) + atomic_load(&pc_progs_done) + atomic_load(&pc_pipes_done);
    return atomic_load(&pc_running) > 0 && *done < *total;
}
// TOTK_WARMUP (tools/linux/warmup.sh): the archive warm-up's progress for its own banner (hle_osd.cpp)
int vk_precompile_warmup_progress(int *done, int *total) {
    if (!pa_warmup_mode() || !pa_nprogs) return 0;
    *total = (int)pa_nprogs;
    *done = atomic_load(&pa_done);
    return atomic_load(&pc_running) > 0;
}

// ================================================================== feedback snapshots (renderer.m resolve_feedback)
static NvnTexture *feedback_tex[4];
static VkImageView feedback_view[4];
static int nfeedback;
// This draw's feedback handling, for the hang ring (2026-10-19 NVIDIA decal hang: which way the stuck draws sampled
// attachments): bits 0-3 live attachments, 4-7 snapshots bound, 8 pass break + barrier for live sampling, 9 a live
// colour target the draw also writes, 10 live depth, 11 snapshot copied now, 12 hazard barrier.
static uint32_t fb_bits;
static uint64_t g_cur_fs_hash;
static NvnTexture *snap_tex[8];
static bool snap_ok[8];
static const char *snap_why[8];
static const char *snap_reason = "?";
static void snapshots_invalidate(NvnTexture *t) {
    for (int i = 0; i < 8; i++) if ((!t || snap_tex[i] == t) && snap_ok[i]) { snap_ok[i] = false; snap_why[i] = snap_reason; }
}

// LARECOMP_VK_PROF: feedback copies per (fragment shader, copied image, what invalidated the last snapshot), printed
// as "[fbcopy]" with each stats window (looking down at the Sky Island cloud sea: ~1170 copies per frame, run295).
typedef struct { uint64_t fs; uint32_t w, h, fmt, levels; const char *why; bool depth; uint64_t n; } FbcKey;
static FbcKey fbc[48];
static int fbc_n;
static uint64_t fbc_other;
static void fbc_note(uint64_t fs, TexBackend *src, const char *why, bool depth) {
    for (int i = 0; i < fbc_n; i++)
        if (fbc[i].fs == fs && fbc[i].w == src->width && fbc[i].h == src->height && fbc[i].fmt == (uint32_t)src->format &&
            fbc[i].why == why && fbc[i].depth == depth) { fbc[i].n++; return; }
    if (fbc_n == 48) { fbc_other++; return; }
    fbc[fbc_n++] = (FbcKey){fs, src->width, src->height, (uint32_t)src->format, src->levels, why, depth, 1};
}
static void fbc_report(void) {
    if (!fbc_n) return;
    for (int i = 1; i < fbc_n; i++)
        for (int j = i; j > 0 && fbc[j].n > fbc[j - 1].n; j--) { FbcKey t = fbc[j]; fbc[j] = fbc[j - 1]; fbc[j - 1] = t; }
    LOG("[fbcopy] top:");
    for (int i = 0; i < fbc_n && i < 8; i++)
        LOG(" fs %016llx %ux%u fmt %u lv %u %s (%s) x%llu;", (unsigned long long)fbc[i].fs, fbc[i].w, fbc[i].h, fbc[i].fmt, fbc[i].levels,
            fbc[i].depth ? "depth" : "color", fbc[i].why ? fbc[i].why : "?", (unsigned long long)fbc[i].n);
    LOG(" (%d keys, %llu untracked)\n", fbc_n, (unsigned long long)fbc_other);
    fbc_n = 0;
    fbc_other = 0;
}

typedef struct { NvnTexture *key; TexBackend copy; } Snap;
static Snap snaps[16];
static int nsnaps;

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    NvnTexture *t = ((NvnTexEntry *)st.tpool->entries)[tic].tex;
    if (t && t->zombie && !t->backend) {
        // finalized while registered (objects.c nvnTextureFinalize): usable only through another object's live
        // rendering at its storage, or (CPU-written texture) by uploading its storage again if that is still readable
        extern bool renderer_texture_has_writer(NvnTexture *t);
        if (!renderer_texture_has_writer(t) && (t->zombie_gpu || !pc_readable(t->addr, tegra_total_size(&t->layout)))) return NULL;
    }
    return t;
}

static bool is_attachment(NvnTexture *t) {
    if (!t) return false;
    if (t == st.rt.depth) return true;
    for (int i = 0; i < st.rt.ncolor; i++) if (t == st.rt.color[i]) return true;
    return false;
}

// Mip levels a feedback snapshot keeps. TotK's bloom renders each level of one 320x180 texture while sampling the level
// next to it (textureLod); a level-0-only copy made every step read level 0 = a blocky box of glow around bright lights
// (torch, user 2026-10-15). LARECOMP_SNAP_ONE_LEVEL=1: level 0 only (the old behaviour).
static uint32_t snap_levels(TexBackend *src) {
    static int one = -1;
    if (one < 0) one = getenv("LARECOMP_SNAP_ONE_LEVEL") != NULL;
    return one || src->samples > 1 || src->levels < 1 ? 1 : src->levels;
}

static TexBackend *snapshot_image(NvnTexture *t, TexBackend *src) {
    Snap *s = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) s = &snaps[i];
    if (!s) {
        if (nsnaps == 16) { static int rr; s = &snaps[rr++ & 15]; } else s = &snaps[nsnaps++];
        s->key = t;
    }
    TexBackend *d = &s->copy;
    uint32_t nlv = snap_levels(src);
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format || d->levels != nlv)) {
        vk_views_forget(d);  // (resolution change)
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = nlv, .arrayLayers = 1,
                                .samples = (VkSampleCountFlagBits)src->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = nlv;
        d->samples = src->samples; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// ================================================================== pause-screen capture (TotK workaround)
// User 2026-10-15: "the frozen ui still makes everything black. also happens whjen switching weapons or items". When a
// pause UI opens (ability wheel, weapon / item switch), the game draws NO 3D scene in that first frame (run339 frame
// 2317: 108 draws: the canvas 1920x1080 R11G11B10F is cleared, the HUD drawn) and then copies the canvas (fs 8d0bd2a0,
// cond-gated, into a canvas-sized RGB10A2 texture) as the frozen background it blurs and shows behind the menu. On
// Switch that copy shows the world; here it was the HUD over black. WHY the game skips the scene in that frame on our
// side is unknown (not pacing: run340 at 30 fps; not cross-queue order: one queue, run341). Workaround: the canvas is
// copied just before its first clear of each frame (= the previous frame, scene + HUD), and a canvas-sized copy OF the
// canvas into a texture that is not presented, in a frame with no scene yet (< 1000 draws), samples that copy instead.
// "[pause] ..." logs each use. LARECOMP_NO_PAUSE_KEEP=1: off.
static NvnTexture *pk_canvas;          // what the present copy (fs 8d0bd2a0 into a presented texture) reads
static TexBackend pk_keep;
// (2026-10-17, user: "the background goes back to that exact snapshot ... a lot darker ... eventually its just black";
// run430: the kept canvas was black: the frames before a capture can hold no scene, and the wheel's own blurred
// background, a scalable texture, counted as "scene", so dimmed wheel frames were kept and captured again.) pk_keep is
// now the canvas of the last frame the scene reached it, kept at the next clear. Scene = the docked upscale (fs 32845c90)
// or any draw into the canvas after >= 1000 draws in the frame: handheld mode (the Deck) composites its 1280x720 scene
// with another shader (fs 0dcc1cad), and at scale 1 nothing is scalable (why the Deck and PC 1x were always black).
// Deck d_wheel_fix 2 s windows: scene frames ~3400 draws each, weapon-menu frames ~180, wheel / capture frames fewer.
static bool pk_scene_since_clear;       // the scene reached the canvas since its last clear
static bool pk_valid;                   // pk_keep holds such a canvas
static uint64_t pk_scene_frame = ~0ull; // present count of the last scene draw into the canvas (see pk_draw)
static uint64_t pk_cap_frame = ~0ull;   // present count of the first frame with a capture copy since the scene left (~0: none yet)
static NvnTexture *pk_subst_tex;        // (this draw only) sample pk_keep instead of this texture
static VkImageView pk_subst_view;
static bool pk_on(void) { static int on = -1; if (on < 0) on = getenv("LARECOMP_NO_PAUSE_KEEP") == NULL; return on; }

static void pk_take(NvnTexture *t) {   // (CMD_CLEAR_COLOR, before the clear)
    if (!pk_on() || t != pk_canvas) return;
    bool scene = pk_scene_since_clear;
    pk_scene_since_clear = false;
    if (!scene) return;
    TexBackend *src = vk_tex(t, true);
    if (!src || !src->image || src->samples > 1 || (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    TexBackend *d = &pk_keep;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageCopy rg = {.srcSubresource = {src->aspect, 0, 0, 1}, .dstSubresource = {src->aspect, 0, 0, 1}, .extent = {src->width, src->height, 1}};
    vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    vk_full_barrier(cb);   // (the clear that follows writes the canvas)
    st.rt_dirty = true;
    pk_valid = true;
    static int plog = -1;
    if (plog < 0) plog = getenv("LARECOMP_PAUSE_DUMP") != NULL;
    if (plog && plog++ < 400) {
        extern uint64_t g_draws_frame;
        LOG("[pause] take: frame %llu, canvas %ux%u kept after %llu draws\n", (unsigned long long)renderer_present_count(), src->width, src->height,
            (unsigned long long)g_draws_frame);
    }
}

// Frames since the scene last reached the canvas (render thread; hle_osd.cpp shows the Graphics menu hint while a pause /
// System menu holds the picture); 0 = not known yet.
uint64_t vk_scene_idle_frames(void) { return pk_scene_frame == ~0ull ? 0 : renderer_present_count() - pk_scene_frame; }

static void pk_draw(Prog *pr) {   // (draw_, before the fragment textures resolve)
    pk_subst_tex = NULL;
    if (!pk_on() || !st.rt.ncolor || !st.rt.color[0]) return;
    extern uint64_t g_draws_frame;
    if (pk_canvas && st.rt.color[0] == pk_canvas && (g_cur_fs_hash == 0x32845c90c668fdbbull || g_draws_frame >= 1000)) {
        pk_scene_frame = renderer_present_count();   // (the scene arriving)
        pk_scene_since_clear = true;
        pk_cap_frame = ~0ull;
    }
    if (g_cur_fs_hash != 0x8d0bd2a0046fd06aull || pr->fi->ntex < 1) return;
    NvnTexture *src = shader_texture_object(1, &pr->fi->tex[0]), *dst = st.rt.color[0];
    // (the capture frame itself is presented black at 144 fps: run443/444, one frame per wheel opening. Presenting the kept
    // canvas from this copy did not change it (run444: still black), so something later in that frame writes the image.)
    if (renderer_texture_presented(dst)) { pk_canvas = src; return; }   // the present copy: learn the canvas
    if (!src || src != pk_canvas || dst == src) return;
    // the capture copy: a 16:9 target at least as large as the canvas (docked: 1920x1080 = the canvas; handheld, Deck
    // d_wheel_fix2: the 1280x720 canvas into a 1920x1080 capture texture). Not the title screen's 520x40 strips.
    float da = dst->height ? (float)dst->width / (float)dst->height : 0;
    bool capture = da > 1.74f && da < 1.81f && dst->width >= src->width && dst->height >= src->height;
    // only in the first frame WITH A CAPTURE COPY since the scene left (all of its capture copies: 2 in the user's frame
    // 3824). Menus that copy the canvas every frame (user's 144 fps session 16:5x: frames 5142+ copies into f0x3d + f0x3f,
    // 37 / 355 draws) keep their own copies after that, as before the 2026-10-17 marker change. (Not "the first frame
    // without a scene": at 144 fps the wheel opens with a 2-draw frame (72x72 copy) and captures in the next one, run441
    // frames 5560/5561: not substituted, black.)
    uint64_t now = renderer_present_count();
    // (the wheel captures <= 2 frames after the scene left; later first captures are titles / loads: Deck d_menu_fix5 frame 393)
    if (capture && !pk_scene_since_clear && pk_cap_frame == ~0ull && now - pk_scene_frame <= 3) pk_cap_frame = now;
    bool first = pk_cap_frame == now;
    if (!capture || !first || !pk_keep.image || !pk_valid || pk_scene_since_clear) {
        static uint32_t seen[16];   // (a canvas copy in a frame with no scene that is not substituted: once per target size)
        static int nseen;
        uint32_t key = dst->width << 16 | dst->height;
        bool known = false;
        for (int i = 0; i < nseen; i++) if (seen[i] == key) known = true;
        if (!pk_scene_since_clear && first && !known && nseen < 16) {
            seen[nseen++] = key;
            LOG("[pause] frame %llu: canvas %ux%u copy into %ux%u f%#x after %llu draws NOT substituted (%s)\n",
                (unsigned long long)renderer_present_count(), src->width, src->height, dst->width, dst->height, dst->format,
                (unsigned long long)g_draws_frame, !capture ? "not a capture-sized target" : "no kept scene canvas yet");
        }
        return;
    }
    static const VkComponentMapping id = {0};
    pk_subst_tex = src;
    pk_subst_view = vk_image_view(&pk_keep, VK_IMAGE_VIEW_TYPE_2D, pk_keep.format, 0, 1, 0, 1, id, pk_keep.aspect);
    static int nl;
    if (nl++ < 40)
        LOG("[pause] frame %llu: canvas copy into %ux%u f%#x after %llu draws (no scene this frame) -> uses the previous frame\n",
            (unsigned long long)renderer_present_count(), dst->width, dst->height, dst->format, (unsigned long long)g_draws_frame);
    static int pdump = -1;   // LARECOMP_PAUSE_DUMP=1: the kept canvas of the first 4 substitutions -> build/pk_keep_<frame>.png
    if (pdump < 0) pdump = getenv("LARECOMP_PAUSE_DUMP") != NULL;   // (run431: the debug frame it also requested stalled the run)
    if (pdump && pdump++ <= 4) {
        extern void renderer_save_texture_tex(TexBackend *b, const char *path);
        char path[64];
        snprintf(path, sizeof path, "build/pk_keep_%llu.png", (unsigned long long)renderer_present_count());
        renderer_save_texture_tex(&pk_keep, path);
        st.rt_dirty = true;
    }
}

// (debug frames) a depth image's depth values as a greyscale PGM, stretched over its own min..max
static void debug_save_depth(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    bool d16 = b->format == VK_FORMAT_D16_UNORM || b->format == VK_FORMAT_D16_UNORM_S8_UINT;
    bool d24 = b->format == VK_FORMAT_X8_D24_UNORM_PACK32 || b->format == VK_FORMAT_D24_UNORM_S8_UINT;
    if (!d16 && !d24 && b->format != VK_FORMAT_D32_SFLOAT_S8_UINT && b->format != VK_FORMAT_D32_SFLOAT) return;
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    float *d = out.ptr;   // (depth-aspect copies: D16 = 2 bytes per texel, D24 = low 24 bits of 4, D32 = float)
    if (d16) for (uint64_t i = n; i-- > 0;) d[i] = ((const uint16_t *)out.ptr)[i] / 65535.0f;
    else if (d24) for (uint64_t i = 0; i < n; i++) d[i] = (((const uint32_t *)out.ptr)[i] & 0xFFFFFF) / 16777215.0f;
    float lo = 1e30f, hi = -1e30f;
    for (uint64_t i = 0; i < n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    for (uint64_t i = 0; i < n; i++) fputc(hi > lo ? (int)((d[i] - lo) / (hi - lo) * 255.0f) : 0, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u depth %.6f..%.6f)\n", path, b->width, b->height, lo, hi);
}

static VkImageAspectFlags fb_live_aspects;   // aspects of the bound attachments this draw samples live (draw_: feedback loop state)
static void resolve_feedback(Prog *pr) {
    nfeedback = 0;
    fb_bits = 0;
    fb_live_aspects = 0;
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_FEEDBACK") != NULL;
    if (off) return;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    int bm = barrier_mode();
    bool fb_done = false;   // a full barrier was recorded for this draw already (one covers every sampled image)
    if (bm == BAR_TRACKED) {
        bool hz = gmem_wr_epoch == bar_epoch && (pr->vi->uses_gmem || pr->fi->uses_gmem);
        for (int sg = 0; sg < 2 && !hz; sg++)
            for (int i = 0; i < infos[sg]->ntex && !hz; i++) {
                if (infos[sg]->tex[i].is_image) { hz = gmem_wr_epoch == bar_epoch; continue; }
                NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
                if (t && !is_attachment(t) && wr_hazard(t)) hz = true;
            }
        if (hz) {
            vk_end_pass();
            VkCommandBuffer cb = vk_cmd();
            gp_mark(cb, "hazard barrier");
            vk_full_barrier(cb);
            pass_barrier_needed = false;
            pass_barrier_fresh = true;
            st.rt_dirty = true;
            fb_done = true;
            stat_hazard_barriers++;
            fb_bits |= 1u << 12;
        }
        bool wr = pr->vi->writes_gmem || pr->fi->writes_gmem;
        for (int sg = 0; sg < 2 && !wr; sg++)
            for (int i = 0; i < infos[sg]->ntex; i++) if (infos[sg]->tex[i].is_image) wr = true;
        if (wr) gmem_wr_epoch = bar_epoch;   // (this draw runs in the current epoch)
    }
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++) {
            NvnTexture *t0 = shader_texture_object(sg, &infos[sg]->tex[i]);
            // t0 may sample another object's image over the same storage (textures.c alias: TotK samples its G-buffer
            // through second texture objects); that image's attachment object is what this pass writes. Missed before
            // 2026-10-19: those reads got no barrier while the pass wrote the image (NVIDIA decal-pass hangs, CURRENT.md).
            extern NvnTexture *renderer_texture_alias_writer(NvnTexture *t);
            NvnTexture *t = t0 && !is_attachment(t0) ? renderer_texture_alias_writer(t0) : t0;
            if (!is_attachment(t)) continue;
            bool alias = t != t0;
            stat_fb_alias += alias;
            bool have = false;
            for (int k = 0; k < nfeedback; k++) have |= feedback_tex[k] == t0;
            if (have || nfeedback == 4) continue;
            static int nosnap = -1;
            if (nosnap < 0) nosnap = getenv("LARECOMP_NO_SNAPSHOT_REUSE") != NULL;
            int si = -1;
            for (int k = 0; k < 8; k++) if (snap_tex[k] == t) si = k;
            bool self_write = t == st.rt.depth && st.ds[0] && st.ds[1];
            // A colour target this draw also writes: same-draw read + write is a data race in Vulkan even in a declared
            // feedback loop. LARECOMP_FEEDBACK_SELFCOPY=1 samples a copy for those (A/B, 2026-10-19 NVIDIA decal hang).
            bool self_write_c = false;
            for (int k = 0; k < st.rt.ncolor; k++)
                if (st.rt.color[k] == t && ((pr->fi->out_mask & st.channel_mask) >> (k * 4) & 0xF)) self_write_c = true;
            stat_fb_selfw += self_write_c;
            static int selfcopy = -1;
            if (selfcopy < 0) selfcopy = getenv("LARECOMP_FEEDBACK_SELFCOPY") != NULL;
            TexBackend *src = vk_tex(t, true);
            Snap *existing = NULL;
            for (int k = 0; k < nsnaps; k++) if (snaps[k].key == t && snaps[k].copy.image) existing = &snaps[k];
            VkImageAspectFlags sa = src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : src->aspect;
            static const VkComponentMapping id = {0};
            // Colour attachments are sampled LIVE, like Eden (BarrierFeedbackLoop: end the render pass, barrier, sample the
            // attachment itself) instead of copying the whole image per draw. TotK's G-buffer decals read the targets they
            // write: looking down at the Sky Island = ~575 copies of 1600x900 images per frame, 58 % of the GPU frame
            // (run296 gpuprof). The pass break is needed only after a write since the last one (snap_ok).
            // LARECOMP_FEEDBACK_COPY=1: copy into a snapshot (the old way; depth attachments always do).
            static int fcopy = -1;
            if (fcopy < 0) fcopy = getenv("LARECOMP_FEEDBACK_COPY") != NULL;
            // Depth attachments a draw samples WITHOUT writing depth (2026-10-17, Deck d24: 6 copies of the 1280x720 D32 per
            // frame at 150 us each for fs dd9edc34 / 8c7a04f0) are sampled live too; a draw that writes the depth it samples
            // still copies. LARECOMP_DEPTH_FEEDBACK_COPY=1: always copy depth (old).
            static int dcopy = -1;
            if (dcopy < 0) dcopy = getenv("LARECOMP_DEPTH_FEEDBACK_COPY") != NULL;
            // LARECOMP_COLOR_FEEDBACK_COPY=1: colour targets always copy, depth as above (A/B, 2026-10-19 decal hang)
            static int ccopy = -1;
            if (ccopy < 0) ccopy = getenv("LARECOMP_COLOR_FEEDBACK_COPY") != NULL;
            if ((alias && vk_format(t0->format) != src->format) || (!fcopy && !(ccopy && !(src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) && !(selfcopy && self_write_c) && (!(src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || (!dcopy && !self_write)))) {   // (an alias read in another format stays live: its view comes from vk_view)
                // LARECOMP_FEEDBACK_GAMEBAR=1 (experiment, 2026-10-17): Maxwell's texture cache does not see ROP writes
                // without the game's own nvnCommandBufferBarrier either, so a sampled attachment re-written since the last
                // feedback barrier gets a new one only when the game issued a barrier since (else it reads the image as is).
                static int fb_gamebar = -1;
                if (fb_gamebar < 0) { const char *e = getenv("LARECOMP_FEEDBACK_GAMEBAR"); fb_gamebar = e && *e == '1'; }
                bool need = si < 0 || !snap_ok[si];
                if (need && fb_gamebar && si >= 0 && !game_bar_since_fb) { need = false; stat_fb_elided++; }
                if (need && bm == BAR_NONE) { need = false; stat_fb_elided++; }
                if (need) {
                    game_bar_since_fb = false;
                    if (!fb_done) {
                        stat_feedback_live++;
                        fb_bits |= 1u << 8;
                        vk_end_pass();
                        VkCommandBuffer cb = vk_cmd();
                        gp_mark(cb, "feedback barrier %ux%u fmt %d", src->width, src->height, src->format);
                        vk_full_barrier(cb);
                        pass_barrier_needed = false;   // (just issued, nothing recorded before the pass begins again)
                        pass_barrier_fresh = true;
                        st.rt_dirty = true;
                        fb_done = true;
                    }
                    if (si < 0) { static int rr2; si = rr2++ & 7; snap_tex[si] = t; }
                    snap_ok[si] = true;
                }
                if (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) depth_live_in_pass = true;
                fb_live_aspects |= src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                feedback_tex[nfeedback] = t0;
                feedback_view[nfeedback++] = alias ? VK_NULL_HANDLE : vk_image_view(src, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, src->levels, 0, 1, id, sa);
                fb_bits += 1;
                if (self_write_c) fb_bits |= 1u << 9;
                if (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) fb_bits |= 1u << 10;
                continue;
            }
            if (!nosnap && si >= 0 && (snap_ok[si] || self_write) && existing) {
                feedback_tex[nfeedback] = t0;
                feedback_view[nfeedback++] = vk_image_view(&existing->copy, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, existing->copy.levels, 0, 1, id, sa);
                fb_bits += 1u << 4;
                continue;
            }
            stat_feedback_copies++;
            fb_bits += 1u << 4;
            fb_bits |= 1u << 11;
            fbc_note(g_cur_fs_hash, src, si < 0 || !existing ? "new" : snap_why[si], t == st.rt.depth);
            vk_end_pass();
            TexBackend *dst = snapshot_image(t, src);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            gp_mark(cb, "feedback snapshot %ux%u fmt %d", src->width, src->height, src->format);
            VkImageCopy rg[16];
            uint32_t nlv = dst->levels < 16 ? dst->levels : 16;
            for (uint32_t l = 0; l < nlv; l++) {
                uint32_t w = src->width >> l, h = src->height >> l;
                rg[l] = (VkImageCopy){.srcSubresource = {src->aspect, l, 0, 1}, .dstSubresource = {src->aspect, l, 0, 1},
                                      .extent = {w ? w : 1, h ? h : 1, 1}};
            }
            vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, nlv, rg);
            if (g_debug_frame && (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                static int nd;
                char path[128];
                snprintf(path, sizeof path, "build/fbdepth_%03d_src.pgm", nd);
                debug_save_depth(src, path);
                snprintf(path, sizeof path, "build/fbdepth_%03d_snap.pgm", nd++);
                debug_save_depth(dst, path);
                cb = vk_cmd();
            }
            feedback_tex[nfeedback] = t0;
            feedback_view[nfeedback++] = vk_image_view(dst, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, dst->levels, 0, 1, id, sa);
            if (si < 0) { static int rr; si = rr++ & 7; snap_tex[si] = t; }
            snap_ok[si] = true;
            st.rt_dirty = true;
        }
}

// ================================================================== resource binding
static const void *ubo_last_ptr;   // (LARECOMP_CBDUMP: the host copy behind the last span)
static VkDescriptorBufferInfo ubo_span(VkSpan s, uint64_t size) {
    uint64_t max = g_vk.props.limits.maxUniformBufferRange;
    ubo_last_ptr = s.ptr;
    return (VkDescriptorBufferInfo){s.buf, s.off, size > max ? max : size};
}

// LARECOMP_CBDUMP=<dir> with LARECOMP_DRAWLOG_FS (debug, 2026-10-19 NVIDIA hang): the first logged draw's uniform buffers go
// to <dir>/vs_cb<binding>.bin / fs_cb<binding>.bin exactly as bound, and its draw record + indices to <dir>/draw.txt,
// so scratchpad drawh.exe can replay the draw outside the game.
static int cbdump_n;
static void cbdump_write(int stage, int k, const void *p, uint64_t size) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s_cb%d.bin", getenv("LARECOMP_CBDUMP"), stage == 1 ? "fs" : "vs", k);
    FILE *f = p ? fopen(path, "wb") : NULL;
    if (f) { fwrite(p, 1, size, f); fclose(f); }
    LOG("[cbdump] st%d binding %d: %llu bytes %s\n", stage, k, (unsigned long long)size, f ? "written" : "NOT written");
}

// LARECOMP_UBO_ZEROCOPY=1 (2026-10-17, Deck: the render thread is the limit with SyncWait lag, UBO snapshots = cbuf ~1.9 ms of
// its ~18.6 ms per frame): the game's uniform buffers are bound straight from the imported guest heap (gmem_buf, dynamic offset
// = address - heap base) instead of copied per submission. The GPU then reads them when it runs, like the Switch does; an
// inline UpdateUniformBuffer (CMD_UPDATE_UNIFORM_BUFFER, writes guest memory mid-list) turns it off for the rest of the run.
static VkBuffer gmem_buf;
static uint64_t gmem_size;
static VkDescriptorBufferInfo gmem_info(void);
static int ubo_zc = -1;
static uint64_t stat_ubo_zc, stat_ubo_copy;
// Uniform buffers the game updates inline (d36: two nvnCommandBufferUpdateUniformBuffer at boot, buffer 100e588000 +64 KB):
// those stay snapshots; the rest keeps zero-copy. More than 64 such buffers: zero-copy off.
// Constant buffer k of a stage (renderer.m bind_cbufs), copied into transient memory.
// Uniform data bound straight from the imported guest heap instead of a per-submission copy (the GPU reads guest memory
// at execution time, as Maxwell does; the game keeps it alive until its fences). OFF by default everywhere since 2026-10-20:
// with the SyncWait lag the game rewrites its UI uniforms before the GPU reads them, so on the Deck HUD elements (hearts,
// sword icon, panels) were drawn screen-sized for single frames outdoors (user video; flick_score2 run B 2037 flash frames
// in 159 s with it, run C 0 with copies; the user watched C: no UI flicker). Its gain was ~1 % (Deck d53/d54 58.9 / 58.2
// fps vs 58.2 / 57.8 copied). LARECOMP_UBO_ZEROCOPY=1 / 2 turns it on. Buffers the game updates inline (UpdateUniformBuffer)
// stay snapshots (zc_excl).
static bool heap_mirror_bind(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off);
static bool alias_mirror_bind(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off);
// LARECOMP_UBO_ZEROCOPY=2 (discrete GPU, off by default): bound from the VRAM heap mirror instead (run518: uniform +
// shader constant copies were 740 MB per 2 s; run519 with it +3 % fps, and the user saw flicker in runs 520 / 522).
static bool ubo_zc_span(uint64_t addr, uint64_t size, VkSpan *out) {
    if (ubo_zc < 0) {
        const char *e = getenv("LARECOMP_UBO_ZEROCOPY");
        ubo_zc = !g_vk.host_import ? 0 : e ? atoi(e) : 0;
        if (ubo_zc) LOG("[vk] uniform buffers + shader constants bound from the guest heap%s (LARECOMP_UBO_ZEROCOPY=0: copies)\n",
                        ubo_zc == 2 ? "'s VRAM mirror" : "");
    }
    if (ubo_zc <= 0 || !addr) return false;
    if (!gmem_buf) gmem_info();
    uint64_t al = g_vk.props.limits.minUniformBufferOffsetAlignment ? g_vk.props.limits.minUniformBufferOffsetAlignment : 256;
    for (int i = 0; i < zc_nexcl; i++) if (addr < zc_excl[i].hi && addr + size > zc_excl[i].lo) return false;
    if (!gmem_buf || addr < HEAP_BASE || addr + size > HEAP_BASE + gmem_size || (addr - HEAP_BASE) % al) return false;
    if (ubo_zc == 2) {
        VkBuffer b;
        VkDeviceSize o;
        if (!heap_mirror_bind(addr, size, &b, &o)) return false;
        *out = (VkSpan){b, o, NULL};
        return true;
    }
    *out = (VkSpan){gmem_buf, addr - HEAP_BASE, NULL};
    rc_record(0, addr, size);
    return true;
}
static VkDescriptorBufferInfo cbuf_info_(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz);
static VkDescriptorBufferInfo cbuf_info(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    ubo_last_ptr = NULL;
    vk_up_kind = 1; VkDescriptorBufferInfo r = cbuf_info_(stage, si, k, gx, gy, gz); vk_up_kind = 0;
    if (cbdump_n) cbdump_write(stage, k, ubo_last_ptr, r.range);
    return r;
}
static uint64_t g_sb0_bound;   // (LARECOMP_COUNT_DUMP) VS storage buffer #0's driver-table address in the last draw
static VkDescriptorBufferInfo cbuf_info_(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    static const uint8_t zeros[0x10000];
    if (k == 0) vk_up_kind = 5; else if (k == 1) vk_up_kind = 6;
    if (k == 0 && stage != 5 && !si->uses_gmem) {
        // (graphics, no global memory: the contents are just st.sbuf[stage]'s 16 ranges at a per-stage offset. Same
        // ranges and size as the previous call for this stage in this submission: the same copy, without zeroing and
        // comparing the 2 KB block; run385 had cbuf at 0.56 us a draw)
        static struct { uint64_t seq; uint64_t size; Range sb[16]; VkDescriptorBufferInfo bi; } dm[2];
        int d = stage == 1;
        uint64_t size = si->cbuf_size[0] > 0x800 ? 0x800 : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        uint64_t q = vk_pending_seq();
        if (dm[d].seq == q && dm[d].size == size && !memcmp(dm[d].sb, st.sbuf[stage], sizeof dm[d].sb) && !cbdump_n) return dm[d].bi;
        uint32_t drv[0x800 / 4] = {0};
        uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
        for (int i = 0; i < 16; i++) {
            uint32_t *e = &drv[(base + 16 * i) / 4];
            e[0] = (uint32_t)st.sbuf[stage][i].addr; e[1] = (uint32_t)(st.sbuf[stage][i].addr >> 32); e[2] = (uint32_t)st.sbuf[stage][i].size;
        }
        dm[d].bi = ubo_span(copy_dedupe(stage == 1 ? DEDUPE_DRV_F : DEDUPE_DRV_V, drv, size, 256), size);
        dm[d].seq = q; dm[d].size = size; memcpy(dm[d].sb, st.sbuf[stage], sizeof dm[d].sb);
        return dm[d].bi;
    }
    if (k == 0) {
        uint32_t drv[0x800 / 4] = {0};
        if (stage == 5) {
            drv[0x200 / 4] = gx; drv[0x204 / 4] = gy; drv[0x208 / 4] = gz;
            for (int i = 0; i < 16; i++) {
                uint32_t *u = &drv[(0x210 + 16 * i) / 4];
                u[0] = (uint32_t)st.ubuf[5][i].addr; u[1] = (uint32_t)(st.ubuf[5][i].addr >> 32); u[2] = (uint32_t)st.ubuf[5][i].size;
                uint32_t *e = &drv[(0x310 + 16 * i) / 4];
                uint64_t a = si->uses_gmem ? gdev_map(st.sbuf[5][i].addr, st.sbuf[5][i].size, si->writes_gmem) : st.sbuf[5][i].addr;
                e[0] = (uint32_t)a; e[1] = (uint32_t)(a >> 32); e[2] = (uint32_t)st.sbuf[5][i].size;
                if (g_debug_frame && si->uses_gmem && st.sbuf[5][i].addr)
                    LOG("[ssbo] cs #%d %llx +%llx %s -> %s\n", i, (unsigned long long)st.sbuf[5][i].addr, (unsigned long long)st.sbuf[5][i].size,
                        si->writes_gmem ? "rw" : "ro", a != st.sbuf[5][i].addr ? "window" : "host");
            }
        } else {
            uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
            for (int i = 0; i < 16; i++) {
                uint32_t *e = &drv[(base + 16 * i) / 4];
                uint64_t a = si->uses_gmem ? gdev_map(st.sbuf[stage][i].addr, st.sbuf[stage][i].size, si->writes_gmem) : st.sbuf[stage][i].addr;
                if (stage == 0 && i == 0) g_sb0_bound = a;
                e[0] = (uint32_t)a;
                e[1] = (uint32_t)(a >> 32);
                e[2] = (uint32_t)st.sbuf[stage][i].size;
                if (g_debug_frame && si->uses_gmem && st.sbuf[stage][i].addr)
                    LOG("[ssbo] st%d #%d %llx +%llx %s -> %s\n", stage, i, (unsigned long long)st.sbuf[stage][i].addr, (unsigned long long)st.sbuf[stage][i].size,
                        si->writes_gmem ? "rw" : "ro", a != st.sbuf[stage][i].addr ? "window" : "host");
            }
        }
        uint64_t size = si->cbuf_size[0] > sizeof drv ? sizeof drv : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        if (size < 0x500 && stage == 5) size = 0x500;
        return ubo_span(copy_dedupe(stage == 5 ? DEDUPE_DRV_C : stage == 1 ? DEDUPE_DRV_F : DEDUPE_DRV_V, drv, size, 256), size);
    }
    if (k == 1) {  // compiler constants stored after the shader code
        NvnShaderStage *ss = &st.program->stages[stage];
        uint64_t size = ss->const_size ? ss->const_size : 16;
        if (size < si->cbuf_size[1]) size = si->cbuf_size[1];
        VkSpan zs;   // (d54: re-copied every submission = 340 MB per 2 s on the Deck; the code memory does not change)
        if (ubo_zc_span(ss->code_addr + ss->const_offset, size, &zs)) { if (prof) prof_c1[0]++, prof_c1_b[0] += size; return ubo_span(zs, size); }
        // Discrete GPU: from the heap's VRAM mirror (LARECOMP_C1_MIRROR=0: copies). Unlike per-frame uniforms these never
        // change after the shader loads, so the mirror's sync timing can't matter (run523: c1 copies 598 MB per 2 s).
        static int c1m = -1;
        if (c1m < 0) { const char *e = getenv("LARECOMP_C1_MIRROR"); c1m = e ? *e == '1' : g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU; }
        uint64_t ca = ss->code_addr + ss->const_offset, al = g_vk.props.limits.minUniformBufferOffsetAlignment ? g_vk.props.limits.minUniformBufferOffsetAlignment : 256;
        VkBuffer mb;
        VkDeviceSize mo;
        if (c1m && ca >= HEAP_BASE && (ca - HEAP_BASE) % al == 0 && heap_mirror_bind(ca, size, &mb, &mo)) { if (prof) prof_c1[1]++, prof_c1_b[1] += size; return ubo_span((VkSpan){mb, mo, NULL}, size); }
        // LARECOMP_C1_ALIAS=1: constants in the alias window from its VRAM mirror (+1-2 %, run587). OFF by default until it
        // is cleared for the 2026-10-19 15:35 GPU hang (run595: the only change of the day in what the GPU reads)
        static int c1a = -1;
        if (c1a < 0) { const char *e = getenv("LARECOMP_C1_ALIAS"); c1a = e && *e == '1'; }
        if (c1m && c1a && ca % al == 0 && alias_mirror_bind(ca, size, &mb, &mo)) { if (prof) prof_c1[6]++, prof_c1_b[6] += size; return ubo_span((VkSpan){mb, mo, NULL}, size); }
        if (prof) { int why = !c1m ? 5 : ca < HEAP_BASE ? 2 : (ca - HEAP_BASE) % al ? 3 : 4; prof_c1[why]++; prof_c1_b[why] += size; if (why == 4) prof_hmw[hm_why]++; }
        if (g_debug_frame) LOG("[upcopy] c1 st%d %llx +%llu\n", stage, (unsigned long long)(ss->code_addr + ss->const_offset), (unsigned long long)size);
        return ubo_span(guest_copy(ss->code_addr + ss->const_offset, size), size);  // (constant: shared per submission)
    }
    int idx = k - 3;  // cbuf_base
    Range r = (idx >= 0 && idx < 18) ? st.ubuf[stage][idx] : (Range){0, 0};
    uint64_t size = si->cbuf_size[k] ? si->cbuf_size[k] : 256;
    if (!r.addr) { vk_up_kind = 7; return ubo_span(vk_transient_copy(zeros, size, 256), size); }
    if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;
    if (size > r.size && r.size) size = r.size;
    if (size < 16) size = 16;
    VkSpan zs;
    if (ubo_zc_span(r.addr, size, &zs)) { stat_ubo_zc++; return ubo_span(zs, size); }
    if (ubo_zc > 0) stat_ubo_copy++;
    if (g_debug_frame) LOG("[upcopy] cb%d st%d %llx +%llu%s\n", idx, stage, (unsigned long long)r.addr, (unsigned long long)size,
                           (si->cbuf_dynamic & (1u << k)) ? " dynamic" : "");
    // snapshot: later CPU writes must not leak in. Shared by the draws of one submission binding the same range:
    // within a command list nothing writes guest memory (no inline UpdateUniformBuffer: that writes through below
    // and drops the shared copies).
    return ubo_span(guest_copy(r.addr, size), size);
}

// Global memory (shader LDG/STG, maxwell_gmem_base = HEAP_BASE): the guest heap imported as a GPU buffer
// (VK_EXT_external_memory_host), zero-copy like metal.m's g_guest_buffer, so compute results land in guest
// memory. The import pins its pages, so it covers only the heap the game set up (SetHeapSize), made on first use.
// LARECOMP_VK_NO_GMEM=1: no import, reads return zero.
static VkBuffer gmem_buf;
static VkDeviceMemory gmem_mem;
static uint64_t gmem_size;

#ifndef _WIN32
#include <unistd.h>
int rt_gpu_bo_find(unsigned long long addr, unsigned long long size, unsigned long long *block_size);
#endif
// Linux / amdgpu: a range runtime/linux/plat_posix.cpp put in a driver-owned GTT buffer (rt_gpu_bo_map: the guest heap, the alias
// window) is imported by dma-buf, whole: 1 imported (*got = the block size), 0 not such a range, -1 the import failed.
// A host-pointer import of the same range would be a userptr buffer (MMU notifier: Deck v2's 6 s kernel stall in vkQueueSubmit2).
static int dmabuf_import(uint64_t base, uint64_t size, uint64_t *got, VkBuffer *pbuf, VkDeviceMemory *pmem) {
#ifdef _WIN32
    (void)base, (void)size, (void)got, (void)pbuf, (void)pmem;
    return 0;
#else
    unsigned long long bsize = 0;
    int fd = rt_gpu_bo_find(base, size, &bsize);
    if (fd < 0) return 0;
    PFN_vkGetMemoryFdPropertiesKHR fdprops =
        g_vk.dmabuf_import ? (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryFdPropertiesKHR") : NULL;
    VkMemoryFdPropertiesKHR mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (!fdprops || fdprops(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &mp) != VK_SUCCESS || !mp.memoryTypeBits) {
        LOG("[vk] dma-buf import of %llx + %llu MB: not supported by the driver\n", (unsigned long long)base, bsize >> 20);
        return -1;
    }
    int dfd = dup(fd);  // (a successful import owns the fd it was given)
    VkImportMemoryFdInfoKHR imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                   .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = dfd};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = bsize,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(mp.memoryTypeBits)};
    VkDeviceMemory mem;
    VkResult r = dfd < 0 ? VK_ERROR_TOO_MANY_OBJECTS : vkAllocateMemory(g_vk.dev, &ai, NULL, &mem);
    if (r != VK_SUCCESS) {
        if (dfd >= 0) close(dfd);
        LOG("[vk] dma-buf import of %llx + %llu MB failed (%d)\n", (unsigned long long)base, bsize >> 20, (int)r);
        return -1;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = bsize,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VkBuffer buf;
    VkMemoryRequirements req = {0};
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, mem, NULL); return -1; }
    vkGetBufferMemoryRequirements(g_vk.dev, buf, &req);
    if (!(req.memoryTypeBits & (1u << ai.memoryTypeIndex)) || req.size > bsize || vkBindBufferMemory(g_vk.dev, buf, mem, 0) != VK_SUCCESS) {
        LOG("[vk] dma-buf import of %llx + %llu MB: buffer does not fit the memory (types %x, need %llu MB)\n", (unsigned long long)base,
            bsize >> 20, req.memoryTypeBits, (unsigned long long)(req.size >> 20));
        vkDestroyBuffer(g_vk.dev, buf, NULL);
        vkFreeMemory(g_vk.dev, mem, NULL);
        return -1;
    }
    *pbuf = buf;
    *pmem = mem;
    if (got) *got = bsize;
    LOG("[vk] %llx + %llu MB imported by dma-buf (driver-owned GTT buffer, memory type %u)\n", (unsigned long long)base, bsize >> 20,
        ai.memoryTypeIndex);
    return 1;
#endif
}

// Imports [base, base + size) of committed host memory as one buffer (storage / vertex / index / transfer); *got (may be NULL)
// = the buffer's size (larger than size for a driver-owned block, imported whole).
static bool host_import(uint64_t base, uint64_t size, uint64_t *got, VkBuffer *pbuf, VkDeviceMemory *pmem) {
    int d = dmabuf_import(base, size, got, pbuf, pmem);
    if (d) return d > 0;  // (a userptr import of a driver-owned mapping cannot work)
    if (got) *got = size;
    PFN_vkGetMemoryHostPointerPropertiesEXT props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!props || props(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)base, &mp) != VK_SUCCESS ||
        !mp.memoryTypeBits)
        return false;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)base};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = size,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(mp.memoryTypeBits)};
    VkDeviceMemory mem;
    if (vkAllocateMemory(g_vk.dev, &ai, NULL, &mem) != VK_SUCCESS) return false;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, mem, NULL); return false; }
    vkBindBufferMemory(g_vk.dev, buf, mem, 0);
    *pbuf = buf;
    *pmem = mem;
    return true;
}

static bool gmem_import(uint64_t size) {
    VkBuffer buf;
    VkDeviceMemory mem;
    if (!host_import(HEAP_BASE, size, &size, &buf, &mem)) return false;
    if (gmem_buf) {  // the heap grew: the old import may still be in use by submitted work
        vk_wait_seq(submit_seq);
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
    }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    LOG("[vk] guest heap imported for shader global memory: %llu MB\n", (unsigned long long)(size >> 20));
    return true;
}

static uint64_t gmem_tried;  // heap size the import was last attempted for (gmem_info imports / re-imports)

static VkDescriptorBufferInfo gmem_info(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import;
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (want > HEAP_MAX) want = HEAP_MAX;
    if (!off && want > gmem_size && !gmem_import(want)) {
        LOG("[vk] guest heap import failed (%llu MB): shader global memory reads return zero\n", (unsigned long long)(want >> 20));
        off = 1;
    }
    if (gmem_buf) return (VkDescriptorBufferInfo){gmem_buf, 0, gmem_size};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// ---- Device window: the storage buffers shaders write live in VRAM while a submission runs. run95 measured RED
// atomics into the host-imported heap (sysmem over PCIe) at 69 + 8 ms of a 113 ms GPU frame (37 ms without them).
// The driver cbuf's storage buffer table points such a buffer at HEAP_BASE + HEAP_MAX + offset, which translated
// shaders route to gdev_ (maxwell_gdev_size). Host memory stays the truth between submissions: a region's first use
// in a submission copies host -> VRAM in the upload command buffer (runs before the main one), and the end of the
// submission copies it back (gdev_writeback), so CPU reads after the fence and CPU writes before submit both hold.
#define GDEV_REGION_MAX (16ull << 20)
typedef struct { uint64_t addr, size, off, seq, up; } GdevRegion;  // seq: the submission it was copied in for (or
// carried into, chunked submission); up: the submission whose upload buffer copied it in
static GdevRegion gdev_reg[256];
static int gdev_n;
static uint64_t gdev_top, gdev_upload_seq, gdev_stat_in, gdev_stat_regions, gdev_stat_snap, gdev_stat_host;
static int gdev_half;               // the window half new regions go into
static uint64_t gdev_half_seq[2];   // per half: the last submission that copied a region into it
static VkBuffer gdev_buf;
static VmaAllocation gdev_alloc;

static bool gdev_ready(void) {
    static bool failed;
    if (failed || !maxwell_gdev_size || !gmem_buf) return false;
    if (!gdev_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GDEV_SIZE,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &gdev_buf, &gdev_alloc, NULL) != VK_SUCCESS) {
            gdev_buf = VK_NULL_HANDLE;
            failed = true;
            LOG("[vk] device window: no %llu MB of VRAM, written storage buffers stay in host memory\n", GDEV_SIZE >> 20);
            return false;
        }
        LOG("[vk] device window for written storage buffers: %llu MB VRAM\n", GDEV_SIZE >> 20);
    }
    return true;
}

static VkDescriptorBufferInfo gdev_info(void) {
    if (gdev_ready()) return (VkDescriptorBufferInfo){gdev_buf, 0, GDEV_SIZE};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// Diff copy-back (default; LARECOMP_GDEV_BLIND=1 = the old whole-region copy). The guest CPU keeps writing host memory
// while a submission runs: run139 (LARECOMP_GDEV_CHECK) caught per-frame float values in 0x30-0x80 byte regions written
// by the CPU mid-submission and reverted by the blind copy-back (with the window off the far torch stays lit, run136).
// A VRAM shadow of each region as uploaded (window -> shadow at the end of the upload buffer) lets the copy-back
// compute pass write only the words the GPU changed; host words the GPU left alone keep what the CPU wrote meanwhile.
static VkBuffer gdev_shadow, wb_set_gmem;
static VmaAllocation gdev_shadow_alloc;
static VkPipeline wb_pipe;
static VkPipelineLayout wb_layout;
static VkDescriptorSetLayout wb_dsl;
static VkDescriptorPool wb_pool;
static VkDescriptorSet wb_set;
static int wb_mode = -1;  // 1 diff, 0 blind

static bool wb_init(void) {
    if (wb_mode >= 0) return wb_mode == 1;
    wb_mode = 0;
    static int blind = -1;
    if (blind < 0) blind = getenv("LARECOMP_GDEV_BLIND") != NULL;
    if (blind) return false;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GDEV_SIZE,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &gdev_shadow, &gdev_shadow_alloc, NULL) != VK_SUCCESS) {
        gdev_shadow = VK_NULL_HANDLE;
        LOG("[vk] device window: no VRAM for the copy-back shadow, blind copy-back\n");
        return false;
    }
    static const char *src =
        "#version 450\n"
        "layout(local_size_x = 256) in;\n"
        "layout(std430, set = 0, binding = 0) readonly buffer Win { uint w[]; } win;\n"
        "layout(std430, set = 0, binding = 1) readonly buffer Sh { uint w[]; } sh;\n"
        "layout(std430, set = 0, binding = 2) buffer Host { uint w[]; } host;\n"
        "layout(push_constant) uniform P { uint win_off, host_off, words; } p;  // (in words)\n"
        "void main() {\n"
        "    uint i = gl_GlobalInvocationID.x;\n"
        "    if (i >= p.words) return;\n"
        "    uint v = win.w[p.win_off + i];\n"
        "    if (v != sh.w[p.win_off + i]) host.w[p.host_off + i] = v;\n"
        "}\n";
    size_t nw;
    uint32_t *spv = vk_glsl_to_spirv(src, STAGE_COMPUTE, "gdev_copy_back", &nw);
    if (!spv) return false;
    VkShaderModule m = vk_module(spv, nw);
    free(spv);
    VkDescriptorSetLayoutBinding b[3];
    for (uint32_t i = 0; i < 3; i++) b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dl = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &dl, NULL, &wb_dsl));
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &wb_dsl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &pl, NULL, &wb_layout));
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .module = m, .pName = "main"},
                                      .layout = wb_layout};
    VK_CHECK(vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &wb_pipe));
    vkDestroyShaderModule(g_vk.dev, m, NULL);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 64};
    VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 64, .poolSizeCount = 1, .pPoolSizes = &ps};
    VK_CHECK(vkCreateDescriptorPool(g_vk.dev, &dp, NULL, &wb_pool));
    wb_mode = 1;
    LOG("[vk] device window: diff copy-back (only the words the GPU changed go back to host memory)\n");
    return true;
}

// The copy-back set for the current heap import (a new one when the import is replaced; old sets stay valid for the
// submissions still using them).
static bool wb_set_ready(void) {
    if (wb_set && wb_set_gmem == gmem_buf) return true;
    VkDescriptorSetAllocateInfo a = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = wb_pool, .descriptorSetCount = 1,
                                     .pSetLayouts = &wb_dsl};
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(g_vk.dev, &a, &set) != VK_SUCCESS) {
        static int told;
        if (!told++) LOG("[vk] device window: copy-back descriptor pool exhausted, blind copy-back\n");
        return false;
    }
    VkDescriptorBufferInfo bi[3] = {{gdev_buf, 0, GDEV_SIZE}, {gdev_shadow, 0, GDEV_SIZE}, {gmem_buf, 0, gmem_size}};
    VkWriteDescriptorSet w[3];
    for (uint32_t i = 0; i < 3; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i, .descriptorCount = 1,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(g_vk.dev, 3, w, 0, NULL);
    wb_set = set;
    wb_set_gmem = gmem_buf;
    return true;
}

// End of the upload buffer: the shadow takes the regions as uploaded (window -> shadow, VRAM to VRAM, so a CPU write
// racing the upload can't make the two copies differ).
static void gdev_shadow_sync(VkCommandBuffer up) {
    if (wb_mode != 1) return;
    uint64_t cur = submit_seq + 1;
    VkBufferCopy bc[256];
    uint32_t n = 0;
    for (int i = 0; i < gdev_n; i++)  // (carried regions keep the shadow of their first upload: the final diff covers every chunk)
        if (gdev_reg[i].seq == cur && gdev_reg[i].up == cur) bc[n++] = (VkBufferCopy){gdev_reg[i].off, gdev_reg[i].off, gdev_reg[i].size};
    if (!n) return;
    vk_full_barrier(up);
    vkCmdCopyBuffer(up, gdev_buf, gdev_shadow, n, bc);
}

// Regions used by the open submission (copies of their placement), for guest_direct's per-draw lookups: the region
// table keeps up to 256 stale regions, and scanning it for every index / vertex binding cost ~1 us per draw.
static struct { uint64_t addr, size, off; } gdev_lv[256];
static int gdev_lv_n;
static uint64_t gdev_lv_seq;
static GdevRegion *gdev_find(uint64_t addr, uint64_t size) {  // the region containing [addr, addr + size)
    for (int i = 0; i < gdev_n; i++)
        if (addr >= gdev_reg[i].addr && addr + size <= gdev_reg[i].addr + gdev_reg[i].size) return &gdev_reg[i];
    return NULL;
}

// The address a storage buffer binding gets in the driver table: its window alias when a shader writes it or it
// lies in a region already in the window, else the guest address unchanged.
// LARECOMP_GDEV_HOST=<hex sizes,...> (bisecting tool): storage buffer bindings of those sizes, and every binding that
// overlaps memory such a binding covered, stay in host memory (consistently, so readers and writers agree).
static bool gdev_host_rule(uint64_t addr, uint64_t size) {
    static uint64_t sizes[32], ex_a[512], ex_s[512];
    static int nsizes = -1, nex;
    if (nsizes < 0) {
        nsizes = 0;
        for (const char *p = getenv("LARECOMP_GDEV_HOST"); p && *p && nsizes < 32;) {
            sizes[nsizes++] = strtoull(p, (char **)&p, 16);
            while (*p == ',') p++;
        }
        if (nsizes) LOG("[vk] device window: %d binding sizes kept in host memory (LARECOMP_GDEV_HOST)\n", nsizes);
    }
    if (!nsizes) return false;
    static long from = -1;  // LARECOMP_GDEV_HOST_AT=<present>: only from then on (a slow rule would desync scripted input)
    if (from < 0) { const char *e = getenv("LARECOMP_GDEV_HOST_AT"); from = e ? atol(e) : 0; }
    extern uint64_t renderer_present_count(void);
    if (from && renderer_present_count() < (uint64_t)from) return false;
    for (int i = 0; i < nex; i++) if (addr < ex_a[i] + ex_s[i] && ex_a[i] < addr + size) return true;
    for (int i = 0; i < nsizes; i++)
        if (size == sizes[i]) {
            if (nex < 512) { ex_a[nex] = addr; ex_s[nex] = size; nex++; }
            return true;
        }
    return false;
}

// Per 64 KB heap block: the last submission that bound it as a written storage buffer. guest_direct's heap copies read
// host memory at record time, so a range a submission still in flight writes (window copy-back or direct) is bound in place.
// Shader-written heap memory, per 4 KB page (submission seq + 1 of the last write). 64 KB blocks swept the game's
// CPU-written UI data in with the shader-written buffers beside it: read from host memory on the GPU timeline, after
// the game had already rewritten it for its next frame (UI flicker, flickcheck run532 444 minimap-edge flips vs 0 with
// LARECOMP_SYNC_LAG=0, run535).
#define GDEV_SH 12
#define GDEV_BLKS ((4ull << 30) >> GDEV_SH)
static uint64_t gdev_blk[GDEV_BLKS];
// The exact storage buffer ranges bound for writing shaders, with the submission whose end puts them in host memory (carried
// with chunked submissions, gdev_carry). gdev_blk's 4 KB pages swept the game's CPU-written dynamic vertex data in with small
// shader-written buffers sharing its pages (Deck RACECHECK R4: 1010ef2e20 +2c0 bound in place, ~20 changes per 600
// SyncWaits, beside written 1010eb2x80 +10): the in-flight tests below use these ranges (LARECOMP_LIVE_PAGES=1: pages only).
enum { GW_N = 8192 };
static struct { uint64_t addr, size, seq; } gw_e[GW_N];
static uint64_t gw_head, gw_lost;   // gw_lost: the newest submission among entries the ring overwrote
static void gw_push(uint64_t addr, uint64_t size, uint64_t seq) {
    for (uint64_t k = gw_head; k > 0 && k + 16 > gw_head; k--) {   // (bindings alternate among a few buffers)
        const typeof(gw_e[0]) *p = &gw_e[(k - 1) % GW_N];
        if (p->addr == addr && p->size == size && p->seq == seq) return;
    }
    if (gw_head >= GW_N && gw_e[gw_head % GW_N].seq > gw_lost) gw_lost = gw_e[gw_head % GW_N].seq;
    gw_e[gw_head++ % GW_N] = (typeof(gw_e[0])){addr, size, seq};
}
static int gw_overlap(uint64_t addr, uint64_t size, uint64_t lo, uint64_t hi) {   // 1: a submission in (lo, hi] writes a byte of
    if (gw_lost > lo) return -1;                                                    // the range, 0: none, -1: unknown (ring lost it)
    for (uint64_t k = gw_head; k > 0 && k + GW_N > gw_head; k--) {
        const typeof(gw_e[0]) *p = &gw_e[(k - 1) % GW_N];
        if (p->seq <= lo) break;   // (submissions never decrease along the ring)
        if (p->seq <= hi && addr < p->addr + p->size && p->addr < addr + size) return 1;
    }
    return 0;
}
static bool live_pages(void) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_LIVE_PAGES") != NULL;
    return on;
}
static uint64_t gdev_map(uint64_t addr, uint64_t size, bool write) {
    if (write && addr >= HEAP_BASE && size) {
        static uint64_t la, ls, lseq;
        if (la != addr || ls != size || lseq != submit_seq + 1) {
            la = addr; ls = size; lseq = submit_seq + 1;
            for (uint64_t b = (addr - HEAP_BASE) >> GDEV_SH, be = (addr + size - 1 - HEAP_BASE) >> GDEV_SH; b <= be && b < GDEV_BLKS; b++) gdev_blk[b] = lseq;
            gw_push(addr, size, lseq);
        }
    }
    if (!addr || !size || addr < HEAP_BASE || addr + size > HEAP_BASE + gmem_size || !gdev_ready()) return addr;
    if (gdev_host_rule(addr, size)) return addr;
    uint64_t cur = submit_seq + 1;
    GdevRegion *r = gdev_find(addr, size);
    const uint64_t half = GDEV_SIZE / 2;
    static int oldfull = -1;   // LARECOMP_GDEV_OLDFULL=1: the old window-full rule (A/B)
    if (oldfull < 0) oldfull = getenv("LARECOMP_GDEV_OLDFULL") != NULL;
    // an older region in the other half is re-made in this one: a half is then only pinned by submissions that copied into it
    // while it was the current half, so switching back to it waits for nothing recent
    if (r && !oldfull && r->seq != cur && (int)(r->off >= half) != gdev_half) { *r = gdev_reg[--gdev_n]; r = NULL; }
    if (!r) {
        // Storage buffers shaders only read go through the window too (default since 2026-10-19; LARECOMP_GDEV_READS=0 = read
        // in place): filled from a record-time snapshot below, where in place they were read on the GPU timeline, after the
        // game could have rewritten them (UI flicker). Eden keeps every storage buffer in VRAM; the diff copy-back leaves
        // their host words alone. (run516 with them: +12 MB window uploads per 2 s, 1 window reset in 718 s.)
        static int reads = -1;
        if (reads < 0) { const char *e = getenv("LARECOMP_GDEV_READS"); reads = !(e && *e == '0') && wb_init(); }   // (needs the diff copy-back)
        if ((!write && !reads) || size > GDEV_REGION_MAX) return addr;
        for (int i = 0; i < gdev_n; i++)
            if (addr < gdev_reg[i].addr + gdev_reg[i].size && gdev_reg[i].addr < addr + size) {
                if (gdev_reg[i].seq == cur && !write) {  // read-only use: its own snapshot copy is safe unless a shader of
                    // this submission writes the overlap (the UI's ring buffer slices overlap: left in host memory they were read
                    // on the GPU timeline after the game rewrote them = minimap text flicker, run544 probe of fs a8f8133c)
                    uint64_t lo = addr > gdev_reg[i].addr ? addr : gdev_reg[i].addr, hi = addr + size < gdev_reg[i].addr + gdev_reg[i].size ? addr + size : gdev_reg[i].addr + gdev_reg[i].size;
                    bool wr = false;
                    for (uint64_t b = (lo - HEAP_BASE) >> GDEV_SH, be = (hi - 1 - HEAP_BASE) >> GDEV_SH; b <= be && b < GDEV_BLKS && !wr; b++) wr = gdev_blk[b] == cur;
                    if (!wr) continue;
                }
                if (gdev_reg[i].seq == cur) {  // overlaps one in use this submission: two copies would race
                    static int nlog;
                    if (nlog++ < 8 || g_debug_frame)
                        LOG("[vk] device window: %llx +%llx overlaps live region %llx +%llx, left in host memory\n", (unsigned long long)addr,
                            (unsigned long long)size, (unsigned long long)gdev_reg[i].addr, (unsigned long long)gdev_reg[i].size);
                    gdev_stat_host++;
                    return addr;
                }
                gdev_reg[i--] = gdev_reg[--gdev_n];  // stale (copied back already): drop it
            }
        // Two halves used in turn. The old "full: reset" needed no region of the current submission in use, which mid-frame
        // never held: from the first fill on every new region silently stayed in host memory, read on the GPU timeline after
        // the game had rewritten it (UI ring slices: minimap text flicker with SyncWait lag; run557 half the text draws).
        if (gdev_n == 256 && !oldfull)   // table full: older regions are only a lookup cache (their window space is guarded per half)
            for (int i = 0; i < gdev_n; i++) if (gdev_reg[i].seq != cur) gdev_reg[i--] = gdev_reg[--gdev_n];
        uint64_t off = (gdev_top + 255) & ~255ull;
        if (oldfull && (off + size > GDEV_SIZE || gdev_n == 256)) {
            for (int i = 0; i < gdev_n; i++) if (gdev_reg[i].seq == cur) { gdev_stat_host++; return addr; }
            vk_wait_seq(submit_seq);
            gdev_n = 0;
            off = 0;
        } else if (off + size > (uint64_t)(gdev_half + 1) * half) {  // this half is full: switch to the other one
            int h = !gdev_half;
            if (gdev_half_seq[h] == cur) off = ~0ull;   // (one submission filled both halves)
            else {
                if (gdev_half_seq[h] > vk_completed_seq()) vk_wait_seq(gdev_half_seq[h]);
                for (int i = 0; i < gdev_n; i++) if ((int)(gdev_reg[i].off >= half) == h) gdev_reg[i--] = gdev_reg[--gdev_n];
                gdev_half = h;
                off = (uint64_t)h * half;
            }
        }
        if (off == ~0ull || gdev_n == 256 || size > half) {   // (rare: counted in the stats line)
            static int nlog;
            if (nlog++ < 8) LOG("[vk] device window: %llx +%llx left in host memory (%d regions, half %d)\n", (unsigned long long)addr,
                                (unsigned long long)size, gdev_n, gdev_half);
            gdev_stat_host++;
            return addr;
        }
        r = &gdev_reg[gdev_n++];
        *r = (GdevRegion){addr, size, off, 0, 0};
        gdev_top = off + size;
        gdev_stat_regions++;
    }
    if (r->seq != cur) {  // first use in this submission: host -> VRAM, after the previous submission's copy-back
        VkCommandBuffer up = vk_upload_cmd();
        if (gdev_upload_seq != cur) { vk_full_barrier(up); gdev_upload_seq = cur; }
        VkBufferCopy bc = {r->addr - HEAP_BASE, r->off, r->size};
        // Read-only use: snapshot the CPU's data now, as the vertex mirror does (copied on the GPU timeline it can already
        // hold the game's next frame: SyncWait lets the game run one fence ahead of the GPU). Pages a shader wrote in a
        // submission still in flight reach host memory by its copy-back on the GPU timeline: those stay GPU copies.
        bool snap = !write;
        static int snap_all = -1;  // LARECOMP_GDEV_SNAP_ALL=1 (diagnostic, can drop in-flight GPU writes): snapshot every read-only use
        if (snap_all < 0) snap_all = getenv("LARECOMP_GDEV_SNAP_ALL") != NULL;
        uint64_t done = snap ? (snap_all ? ~0ull : vk_completed_seq()) : 0;
        for (uint64_t b = (r->addr - HEAP_BASE) >> GDEV_SH, be = (r->addr + r->size - 1 - HEAP_BASE) >> GDEV_SH; snap && b <= be && b < GDEV_BLKS; b++)
            if (gdev_blk[b] > done) snap = false;
        if (snap) {
            int k0 = vk_up_kind;
            vk_up_kind = 2;
            VkSpan sp = vk_transient(r->size, 256);
            vk_up_kind = k0;
            memcpy(sp.ptr, (const void *)(uintptr_t)r->addr, r->size);
            bc.srcOffset = sp.off;
            vkCmdCopyBuffer(up, sp.buf, gdev_buf, 1, &bc);
            gdev_stat_snap += r->size;
        } else
            vkCmdCopyBuffer(up, gmem_buf, gdev_buf, 1, &bc);
        if (gdev_chk_on()) vkCmdCopyBuffer(up, gmem_buf, chk_buf[0], 1, &bc);  // (LARECOMP_GDEV_CHECK: host at the upload)
        wb_init();  // (first use: the copy-back pipeline + shadow; gdev_shadow_sync fills the shadow at the end of the upload buffer)
        r->seq = cur;
        r->up = cur;
        gdev_half_seq[r->off >= GDEV_SIZE / 2] = cur;
        if (gdev_lv_seq != cur) { gdev_lv_seq = cur; gdev_lv_n = 0; }
        if (gdev_lv_n < 256) { gdev_lv[gdev_lv_n].addr = r->addr; gdev_lv[gdev_lv_n].size = r->size; gdev_lv[gdev_lv_n].off = r->off; gdev_lv_n++; }
        gdev_stat_in += r->size;
    }
    return HEAP_BASE + HEAP_MAX + r->off + (addr - r->addr);
}

// LARECOMP_GDEV_CHECK=1 (diagnostic): does host memory of a region change while the GPU runs the submission that has
// it in the window (guest CPU writes, or GPU writes that bypass the window)? The blind copy-back below would overwrite
// them. Host contents are snapshotted on the GPU timeline at the upload (chk 0) and just before the copy-back (chk 1);
// the next submit waits for that submission and compares ("[gdev-check]" lines).
static int gdev_chk = -1;
static VkBuffer chk_buf[2];
static VmaAllocation chk_mem[2];
static uint8_t *chk_ptr[2];
static struct { uint64_t addr, size, off; } chk_pend[256];
static int chk_n;
static uint64_t chk_seq, chk_words, chk_clob;
static bool gdev_chk_on(void) {
    if (gdev_chk < 0) {
        gdev_chk = getenv("LARECOMP_GDEV_CHECK") != NULL;
        for (int i = 0; gdev_chk && i < 2; i++) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GDEV_SIZE, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
            VmaAllocationInfo info;
            VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &chk_buf[i], &chk_mem[i], &info));
            chk_ptr[i] = info.pMappedData;
        }
    }
    return gdev_chk;
}
static void gdev_check_prev(void) {
    if (!chk_n) return;
    vk_wait_seq(chk_seq);
    static int nlog;
    for (int k = 0; k < chk_n; k++) {
        const uint32_t *a = (const uint32_t *)(chk_ptr[0] + chk_pend[k].off), *b = (const uint32_t *)(chk_ptr[1] + chk_pend[k].off);
        const uint32_t *now = (const uint32_t *)(uintptr_t)chk_pend[k].addr;
        uint64_t words = chk_pend[k].size / 4, diff = 0, clob = 0, first = ~0ull;
        for (uint64_t i = 0; i < words; i++)
            if (a[i] != b[i]) { diff++; if (now[i] != b[i]) { clob++; if (first == ~0ull) first = i; } }
        chk_words += diff;
        chk_clob += clob;
        if (diff && (nlog++ < 400 || g_debug_frame))
            LOG("[gdev-check] seq %llu region %llx +%llx: host changed in %llu words while in the window, %llu overwritten by the copy-back"
                "%s (first +%llx: start %08x changed %08x final %08x)\n", (unsigned long long)chk_seq, (unsigned long long)chk_pend[k].addr,
                (unsigned long long)chk_pend[k].size, (unsigned long long)diff, (unsigned long long)clob, "",
                (unsigned long long)(first == ~0ull ? 0 : first * 4), first == ~0ull ? 0 : a[first], first == ~0ull ? 0 : b[first],
                first == ~0ull ? 0 : now[first]);
    }
    chk_n = 0;
}

// End of a submission's main command buffer (no pass open): the regions it used go back to host memory.
static bool gdev_carry_ok(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_CHUNK_NO_CARRY") != NULL;
    return !off && wb_mode == 1 && !gdev_chk_on();
}
static void gdev_carry(uint64_t from, uint64_t to) {
    for (int i = 0; i < gdev_n; i++) if (gdev_reg[i].seq == from) gdev_reg[i].seq = to;
    for (uint64_t k = gw_head; k > 0 && k + GW_N > gw_head && gw_e[(k - 1) % GW_N].seq >= from; k--)   // (newest entries)
        if (gw_e[(k - 1) % GW_N].seq == from) gw_e[(k - 1) % GW_N].seq = to;
    if (gdev_lv_seq == from) gdev_lv_seq = to;
}

static void gdev_writeback(VkCommandBuffer cb) {
    uint64_t cur = submit_seq + 1;
    VkBufferCopy bc[256];
    uint32_t n = 0;
    for (int i = 0; i < gdev_n; i++)
        if (gdev_reg[i].seq == cur) bc[n++] = (VkBufferCopy){gdev_reg[i].off, gdev_reg[i].addr - HEAP_BASE, gdev_reg[i].size};
    if (!n) return;
    vk_full_barrier(cb);
    if (gdev_chk_on()) {  // host as it is right before the copy-back
        VkBufferCopy hc[256];
        for (uint32_t i = 0; i < n; i++) {
            hc[i] = (VkBufferCopy){bc[i].dstOffset, bc[i].srcOffset, bc[i].size};
            chk_pend[i].addr = bc[i].dstOffset + HEAP_BASE; chk_pend[i].size = bc[i].size; chk_pend[i].off = bc[i].srcOffset;
        }
        vkCmdCopyBuffer(cb, gmem_buf, chk_buf[1], n, hc);
        vk_full_barrier(cb);
        chk_n = (int)n;
        chk_seq = cur;
    }
    if (wb_mode == 1 && wb_set_ready()) {  // diff copy-back: one dispatch per region; unaligned regions copied whole
        VkBufferCopy whole[256];
        uint32_t nw = 0;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, wb_pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, wb_layout, 0, 1, &wb_set, 0, NULL);
        for (uint32_t i = 0; i < n; i++) {
            if (((bc[i].dstOffset | bc[i].size) & 3) || bc[i].dstOffset / 4 > 0xFFFFFFFFull) { whole[nw++] = bc[i]; continue; }
            uint32_t pc[3] = {(uint32_t)(bc[i].srcOffset / 4), (uint32_t)(bc[i].dstOffset / 4), (uint32_t)(bc[i].size / 4)};
            vkCmdPushConstants(cb, wb_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            vkCmdDispatch(cb, (pc[2] + 255) / 256, 1, 1);
        }
        if (nw) vkCmdCopyBuffer(cb, gdev_buf, gmem_buf, nw, whole);
    } else
        vkCmdCopyBuffer(cb, gdev_buf, gmem_buf, n, bc);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// Vertex / index data read in place from the imported guest heap (renderer.m bind_guest does the same with
// g_guest_buffer): no per-draw copy, no index scan for the copy size. The game fences its CPU writes as on the
// Switch. Falls back to copies outside the heap or without the import; LARECOMP_VK_COPY_VERTICES=1 forces copies.
// The alias region's import window (hle_os_mem.cpp, TOTK_ALIAS_IMPORT_MB): sead PageHeap memory where TotK keeps much
// of its vertex / index data (run187: ~100k copies, ~5 GB per 2 s from there). Imported once, on first use; shaders
// never write it through the device window (gdev covers the heap only), so it is bound in place without that check.
// run189/190 A/B: bound in place, the GPU read ~40 MB of vertex data per frame over PCIe (GPU busy 575 -> 1250 ms per
// 2 s, fps unchanged). So draws read a VRAM mirror of the window instead; the pages the guest wrote since the previous
// submission (Windows write watch, alias_dirty_take) are copied sysmem -> VRAM at the head of each submission that binds
// alias data (LARECOMP_VK_ALIAS_MIRROR=0: bind the sysmem import directly). Off by default on an integrated GPU (Steam
// Deck APU, 2026-10-16): its "VRAM" is the same RAM the import already sits in, so the mirror would only cost 2 GB of
// the 16 shared and a copy per submission (LARECOMP_VK_ALIAS_MIRROR=1 forces it on).
extern uint64_t g_alias_import_base, g_alias_import_size;
extern int alias_dirty_take(uint64_t *offs, uint64_t *lens, int max, uint64_t *bytes);
extern int alias_dirty_take_range(uint64_t off, uint64_t len, uint64_t *offs, uint64_t *lens, int max, uint64_t *bytes);
static VkBuffer alias_buf, alias_dev;   // (alias_dev: non-null = the VRAM mirror is on; its memory is in al_blk blocks)
static VkDeviceMemory alias_mem;
// The VRAM mirrors are made of 16 MB blocks, each allocated when a draw first binds a range in it (run575: of the
// 2168 MB heap and 2048 MB alias mirrors only 41 + 149 MB were ever bound; whole-size mirrors pushed VRAM past the OS
// budget during the walk = WDDM paging, 3-14 fps in run571/575). A range that straddles two blocks is copied per draw.
#define MBLK_SH 24
static uint64_t g_mirror_blk_bytes[2];   // allocated block bytes: heap, alias
static VkBuffer mirror_block(VkBuffer *bl, VmaAllocation *bm, uint64_t i, uint64_t len, VkBufferUsageFlags usage, int which) {
    if (!bl[i]) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = len, .usage = usage};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &bl[i], &bm[i], NULL) != VK_SUCCESS) bl[i] = VK_NULL_HANDLE;
        else g_mirror_blk_bytes[which] += len;
    }
    return bl[i];
}
static VkBuffer *al_blk;
static VmaAllocation *al_blk_mem;
static uint64_t al_nblk;
static uint64_t alias_sync_seq;
static uint64_t *alias_chunk_key, alias_chunk_n;  // lazy sync: per 1 MB chunk, the list / submission key it was last synced for
// Lazy sync snapshots the dirty pages at the sync (CPU copy into upload staging), as the heap mirror does: copied from the
// import when the GPU ran the upload, they could already hold the game's next frame (SyncWait lets the game run a fence
// ahead of the GPU): the minimap's coordinate text was drawn with another string's glyph vertices for single frames
// (flickcheck crops run538). Pages rewritten between two lists of one submission stay out of the mirror for the rest of it
// (draws binding them copy) and catch up at the next submission (al_pg_*: as hm_pg_* in heap_mirror).
static uint64_t *al_pg_sub, *al_pg_vol, *al_chunk_vol;
static uint8_t *al_pg_stale;
uint64_t prof_alias_syncs, prof_alias_bytes, prof_alias_ranges, prof_alias_ns;
static int alias_state;  // 0 untried, 1 imported, -1 failed / off
static bool alias_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    uint64_t base = g_alias_import_base, wsize = g_alias_import_size;
    if (!wsize || addr < base || addr + size > base + wsize || alias_state < 0) return false;
    if (!alias_state) {
        alias_state = g_vk.host_import && !getenv("LARECOMP_VK_NO_ALIAS_IMPORT") && host_import(base, wsize, NULL, &alias_buf, &alias_mem) ? 1 : -1;
        LOG("[vk] alias window import %llx + %llu MB: %s\n", (unsigned long long)base, (unsigned long long)(wsize >> 20),
            alias_state > 0 ? "vertex / index data there bound in place" : "off or failed, copied");
        if (alias_state < 0) return false;
        const char *m = getenv("LARECOMP_VK_ALIAS_MIRROR");
        uint64_t probe;
        bool uma = g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if ((m ? strcmp(m, "0") != 0 : !uma) && alias_dirty_take(NULL, NULL, 0, &probe) >= 0) {  // (first take: everything written so far)
            // (blocks: the probe take above reset the watch, so each block copies its whole span when it is made)
            al_nblk = (wsize + (1ull << MBLK_SH) - 1) >> MBLK_SH;
            al_blk = calloc(al_nblk, sizeof *al_blk);
            al_blk_mem = calloc(al_nblk, sizeof *al_blk_mem);
            alias_dev = al_blk && al_blk_mem ? alias_buf : VK_NULL_HANDLE;
            LOG("[vk] alias window VRAM mirror (%llu MB in %llu blocks made on first use): %s\n", (unsigned long long)(wsize >> 20),
                (unsigned long long)al_nblk, alias_dev ? "on" : "off, sysmem");
            alias_sync_seq = submit_seq + 1;
            alias_sync_epoch = alias_epoch;
        }
    }
    // Lazy sync (default; LARECOMP_VK_ALIAS_LAZY=0: the whole span at the first alias bind of each list): only the 1 MB
    // chunks this range covers, each once per list (or per submission, outside one). run359: the whole-span scan was
    // ~1.1 ms per list (98 syncs, 107 ms per 2 s) and GetWriteWatch the top non-driver cost of exec in run362's profile.
    // Chunks never bound keep their write-watch bits and are copied when a draw first binds them.
    static int lazy = -1;
    if (lazy < 0) { const char *e = getenv("LARECOMP_VK_ALIAS_LAZY"); lazy = !e || strcmp(e, "0"); }
    uint64_t ab = (addr - base) >> MBLK_SH, abase = ab << MBLK_SH;
    VkBuffer mb = VK_NULL_HANDLE;
    if (alias_dev) {
        if (size && ((addr - base + size - 1) >> MBLK_SH) != ab) return false;  // straddles two mirror blocks: copied
        if (!al_blk[ab]) {
            uint64_t blen = wsize - abase < (1ull << MBLK_SH) ? wsize - abase : (1ull << MBLK_SH);
            if (!mirror_block(al_blk, al_blk_mem, ab, blen, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 1)) return false;
            VkCommandBuffer up = vk_upload_cmd();   // the block's whole span once (pages written since: the syncs below, after it)
            vk_full_barrier(up);
            VkBufferCopy bc = {abase, 0, blen};
            vkCmdCopyBuffer(up, alias_buf, al_blk[ab], 1, &bc);
            vk_full_barrier(up);
        }
        mb = al_blk[ab];
    }
    if (alias_dev && lazy) {
        static uint64_t barrier_seq;
        uint64_t *chunk_key = alias_chunk_key;
        if (!chunk_key) {
            chunk_key = alias_chunk_key = calloc(alias_chunk_n = (wsize + (1u << 20) - 1) >> 20, sizeof *chunk_key);
            al_chunk_vol = calloc(alias_chunk_n, 8);
            al_pg_sub = calloc(alias_chunk_n << 8, 8);
            al_pg_vol = calloc(alias_chunk_n << 8, 8);
            al_pg_stale = calloc(alias_chunk_n << 8, 1);
            if (!chunk_key || !al_chunk_vol || !al_pg_sub || !al_pg_vol || !al_pg_stale) { alias_chunk_key = NULL; return false; }
        }
        uint64_t cur = submit_seq + 1, key = in_execute ? alias_epoch : cur | (1ull << 62);
        for (uint64_t ch = (addr - base) >> 20, ce = (addr - base + (size ? size - 1 : 0)) >> 20; ch <= ce; ch++) {
            if (chunk_key[ch] == key) continue;
            chunk_key[ch] = key;
            static uint64_t offs[256], lens[256];
            uint64_t t0 = prof ? prof_now() : 0, bytes = 0, cb = ch << 20, clen = wsize - cb < (1u << 20) ? wsize - cb : (1u << 20);
            int n = alias_dirty_take_range(cb, clen, offs, lens, 256, &bytes);
            uint8_t pg[256] = {0};
            for (int i = 0; i < n; i++)
                for (uint64_t o = offs[i]; o < offs[i] + lens[i]; o += 4096) pg[(o - cb) >> 12] = 1;
            uint64_t snap = 0, np = (clen + 4095) >> 12;
            for (uint64_t i = 0; i < np; i++) {
                uint64_t q = (cb >> 12) + i;
                if (!pg[i] && al_pg_stale[q] && al_pg_sub[q] != cur) pg[i] = 1;  // left out last submission: catch up
                if (pg[i] && al_pg_sub[q] == cur) { pg[i] = 0; al_pg_stale[q] = 1; al_pg_vol[q] = cur; al_chunk_vol[ch] = cur; continue; }
                if (pg[i]) { al_pg_sub[q] = cur; al_pg_stale[q] = 0; snap += 4096; }
            }
            int nr = 0;
            if (snap) {
                VkBufferCopy bc[256];
                int k0 = vk_up_kind;
                vk_up_kind = 2;
                VkSpan sp = vk_transient(snap, 256);
                vk_up_kind = k0;
                uint64_t so = 0;
                for (uint64_t i = 0; i < np; i++) {
                    if (!pg[i]) continue;
                    uint64_t o = cb + (i << 12), l = o + 4096 > wsize ? wsize - o : 4096;
                    memcpy((uint8_t *)sp.ptr + so, (const void *)(uintptr_t)(base + o), l);
                    if (nr && bc[nr - 1].dstOffset + bc[nr - 1].size == o - abase) bc[nr - 1].size += l;
                    else bc[nr++] = (VkBufferCopy){sp.off + so, o - abase, l};
                    so += l;
                }
                VkCommandBuffer up = vk_upload_cmd();
                if (barrier_seq != cur) {  // once per submission: after the earlier submissions' reads of the mirror
                    vk_full_barrier(up);
                    barrier_seq = cur;
                }
                vkCmdCopyBuffer(up, sp.buf, mb, (uint32_t)nr, bc);
            }
            prof_alias_syncs++;
            prof_alias_bytes += bytes;
            prof_alias_ranges += nr;
            if (prof) prof_alias_ns += prof_now() - t0;
        }
        for (uint64_t ch = (addr - base) >> 20, ce = (addr - base + (size ? size - 1 : 0)) >> 20; ch <= ce; ch++) {
            if (al_chunk_vol[ch] != cur) continue;
            for (uint64_t q = (addr - base) >> 12, qe = (addr - base + (size ? size - 1 : 0)) >> 12; q <= qe; q++)
                if (al_pg_vol[q] == cur) return false;  // rewritten since the mirror got it this submission: copied instead
            break;
        }
    } else if (alias_dev && (in_execute ? alias_sync_epoch != alias_epoch : alias_sync_seq != submit_seq + 1)) {  // first alias bind of
        // this command list (or submission, outside one): guest writes -> VRAM
        // shortcut: copied on the GPU timeline (not snapshotted like the lazy path); only the LARECOMP_VK_ALIAS_LAZY=0 opt-out
        static uint64_t offs[4096], lens[4096];
        uint64_t t0 = prof ? prof_now() : 0, bytes = 0;
        int n = alias_dirty_take(offs, lens, 4096, &bytes);
        if (n > 0) {
            VkCommandBuffer up = vk_upload_cmd();
            vk_full_barrier(up);  // after the earlier submissions' reads of the mirror
            for (int i = 0; i < n; i++)
                for (uint64_t o = offs[i], e = offs[i] + lens[i]; o < e;) {
                    uint64_t k = o >> MBLK_SH, ke = ((k + 1) << MBLK_SH) < e ? (k + 1) << MBLK_SH : e;
                    if (al_blk[k]) vkCmdCopyBuffer(up, alias_buf, al_blk[k], 1, &(VkBufferCopy){o, o - (k << MBLK_SH), ke - o});
                    o = ke;
                }
        }
        alias_sync_seq = submit_seq + 1;
        alias_sync_epoch = alias_epoch;
        prof_alias_syncs++;
        prof_alias_bytes += bytes;
        prof_alias_ranges += n > 0 ? n : 0;
        if (prof) prof_alias_ns += prof_now() - t0;
    }
    *buf = alias_dev ? mb : alias_buf;
    *off = addr - base - (alias_dev ? abase : 0);
    if (!alias_dev) rc_record(2, addr, size);
    return true;
}
// The alias window's VRAM mirror only (never the in-place sysmem import): shader constants live there (run586: 547k c1
// binds / 2 s were above the heap = in the alias window, all copied, ~430 MB / 2 s).
static bool alias_mirror_bind(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    VkBuffer b;
    VkDeviceSize o;
    if (!alias_direct(addr, size, &b, &o) || !alias_dev || b == alias_buf) return false;
    *buf = b; *off = o;
    return true;
}

// VRAM mirror of the heap for vertex / index data (discrete GPU; LARECOMP_VK_NO_HEAP_MIRROR=1: per-draw copies). run517
// (VX_STATS, 2 s): the copies were 11 GB of pages already copied in that submission + 1.1 GB unchanged since an earlier
// one + ~5 MB really changed, and vertex copies + index scans took 582 ms of the 1926 ms busy render thread. Each 1 MB
// chunk a draw binds is synced once per submission, in the upload buffer: the pages the CPU wrote since the mirror copied
// them (heap write watch) plus the pages a shader wrote since (gdev_blk; GPU writes skip the write watch).
extern int heap_mirror_take(uint64_t addr, uint64_t len, uint64_t *offs, uint64_t *lens, int max, uint64_t *bytes);
uint64_t prof_hm_syncs, prof_hm_bytes, prof_hm_ranges, prof_hm_ns;
static int hm_state;  // 0 untried, 1 on, -1 off
static VkBuffer *hm_blk;   // 16 MB blocks made on first use (see mirror_block)
static VmaAllocation *hm_blk_mem;
static uint64_t hm_size, *hm_chunk_key, hm_blk_seq[GDEV_BLKS];
// One mirror copy serves a whole submission (its uploads run before it), but the game can rewrite a page between two
// command lists of one submission. The later version then stays out of the mirror (the earlier lists' draws read the
// mirror), the page is volatile for the rest of the submission (draws binding it copy, as before the mirror) and stale
// (snapshotted at its chunk's first sync of a later submission).
static uint64_t *hm_pg_sub, *hm_pg_vol, *hm_chunk_vol;  // per page: submission that last wrote it into the mirror / made it volatile
static uint64_t *hm_pg_gpu;   // (LARECOMP_RACECHECK) per page: submission whose upload copied all of it on the GPU timeline
static uint8_t *hm_pg_stale;
uint64_t prof_hm_vol;
static bool heap_mirror(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (hm_state < 0) { hm_why = 5; return false; }
    if (!hm_state) {
        uint64_t probe;
        hm_state = -1;
        if (!getenv("LARECOMP_VK_NO_HEAP_MIRROR") && gmem_buf && heap_mirror_take(HEAP_BASE, 0, NULL, NULL, 0, &probe) >= 0) {
            uint64_t nblk = (gmem_size + (1ull << MBLK_SH) - 1) >> MBLK_SH;
            hm_blk = calloc(nblk, sizeof *hm_blk);
            hm_blk_mem = calloc(nblk, sizeof *hm_blk_mem);
            if (hm_blk && hm_blk_mem) {
                hm_size = gmem_size;
                uint64_t nch = (hm_size + (1u << 20) - 1) >> 20, npg = nch << 8;
                hm_chunk_key = calloc(nch, 8); hm_chunk_vol = calloc(nch, 8);
                hm_pg_sub = calloc(npg, 8); hm_pg_vol = calloc(npg, 8); hm_pg_stale = calloc(npg, 1);
                if (getenv("LARECOMP_RACECHECK")) hm_pg_gpu = calloc(npg, 8);
                hm_state = hm_chunk_key && hm_chunk_vol && hm_pg_sub && hm_pg_vol && hm_pg_stale ? 1 : -1;
            }
        }
        LOG("[vk] heap VRAM mirror (%llu MB): %s\n", (unsigned long long)(gmem_size >> 20), hm_state > 0 ? "on" : "off, per-draw copies");
        if (hm_state < 0) return false;
    }
    if (addr + size > HEAP_BASE + hm_size) { hm_why = 2; return false; }
    uint64_t hb = (addr - HEAP_BASE) >> MBLK_SH, hbase = hb << MBLK_SH;
    if (size && ((addr - HEAP_BASE + size - 1) >> MBLK_SH) != hb) { hm_why = 3; return false; }  // straddles two mirror blocks: copied
    VkBuffer mb = mirror_block(hm_blk, hm_blk_mem, hb, hm_size - hbase < (1ull << MBLK_SH) ? hm_size - hbase : (1ull << MBLK_SH),
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 0);
    if (!mb) { hm_why = 4; return false; }
    // once per command list (as the alias mirror): the guest CPU can write between the lists of one submission
    uint64_t cur = submit_seq + 1, key = in_execute ? alias_epoch : cur | (1ull << 62);
    static uint64_t barrier_seq;
    for (uint64_t ch = (addr - HEAP_BASE) >> 20, ce = (addr - HEAP_BASE + (size ? size - 1 : 0)) >> 20; ch <= ce; ch++) {
        if (hm_chunk_key[ch] == key) continue;
        hm_chunk_key[ch] = key;
        uint64_t t0 = prof ? prof_now() : 0, bytes = 0, base = ch << 20, len = hm_size - base < (1u << 20) ? hm_size - base : (1u << 20);
        static uint64_t offs[128], lens[128];
        int n = heap_mirror_take(HEAP_BASE + base, len, offs, lens, 128, &bytes);
        if (n < 0) { offs[0] = base; lens[0] = len; n = 1; bytes = len; }  // (outside the committed heap / failed: all of it)
        // dirty pages of the chunk: 1 = written by the CPU, 2 = in a block a shader wrote (host memory gets that data from
        // the window copy-back on the GPU timeline, so it is copied there). CPU-written pages are snapshotted NOW, as the
        // per-draw copies did: copied from host memory when the GPU runs the upload, they could already hold the game's next
        // frame (the render thread runs behind the game; user 2026-10-19, run520/522: flicker with the GPU-time copies).
        uint8_t pg[256] = {0};
        for (int i = 0; i < n; i++)
            for (uint64_t o = offs[i]; o < offs[i] + lens[i]; o += 4096) pg[(o - base) >> 12] = 1;
        uint64_t done = vk_completed_seq();
        for (uint64_t bk = base >> GDEV_SH, be = (base + len - 1) >> GDEV_SH; bk <= be; bk++) {
            // (also CPU-written, shader's submission done: host memory holds both (diff copy-back) -> snapshot it)
            uint64_t i = ((bk << GDEV_SH) - base) >> 12;
            if (gdev_blk[bk] && gdev_blk[bk] >= hm_blk_seq[bk] && !(pg[i] == 1 && gdev_blk[bk] <= done)) {
                // The page's CPU bytes are snapshotted now, as above; only the ranges an earlier submission still in flight writes
                // are copied from host memory on the GPU timeline, after the snapshot (3). The whole page that way (2) read the
                // game's dynamic vertex data on it late (LARECOMP_LIVE_PAGES=1, or the ring lost an entry).
                int gw = live_pages() ? -1 : gw_overlap(HEAP_BASE + (bk << GDEV_SH), 1u << GDEV_SH, done, cur - 1);
                pg[i] = gw < 0 ? 2 : gw ? 3 : 1;
                bytes += 1u << GDEV_SH;
            }
            hm_blk_seq[bk] = cur;
        }
        VkBufferCopy bc[512], sc[128];
        int nb = 0, ns = 0, n3 = 0;
        uint64_t snap = 0, np = (len + 4095) >> 12;
        for (uint64_t i = 0; i < np; i++) {
            uint64_t q = (base >> 12) + i;
            if (!pg[i] && hm_pg_stale[q] && hm_pg_sub[q] != cur) pg[i] = 1;  // left out last submission: catch up
            if (pg[i] & 1 && hm_pg_sub[q] == cur) { pg[i] = 0; hm_pg_stale[q] = 1; hm_pg_vol[q] = cur; hm_chunk_vol[ch] = cur; prof_hm_vol++; continue; }
            if (pg[i]) { hm_pg_sub[q] = cur; hm_pg_stale[q] = 0; }
            if (pg[i] == 2 && hm_pg_gpu) hm_pg_gpu[q] = cur;
        }
        for (uint64_t i = 0; i < np; i++) if (pg[i] & 1) snap += 4096;
        VkSpan sp = {0};
        if (snap) { int k0 = vk_up_kind; vk_up_kind = 2; sp = vk_transient(snap, 256); vk_up_kind = k0; }
        uint64_t so = 0;
        for (uint64_t i = 0; i < np; i++) {
            if (!pg[i]) continue;
            uint64_t o = base + (i << 12), l = o + 4096 > hm_size ? hm_size - o : 4096;
            if (pg[i] == 2) {
                if (nb && bc[nb - 1].srcOffset + bc[nb - 1].size == o) bc[nb - 1].size += l;
                else bc[nb++] = (VkBufferCopy){o, o - hbase, l};
                continue;
            }
            memcpy((uint8_t *)sp.ptr + so, (const void *)(uintptr_t)(HEAP_BASE + o), l);
            if (ns && sc[ns - 1].dstOffset + sc[ns - 1].size == o - hbase) sc[ns - 1].size += l;
            else sc[ns++] = (VkBufferCopy){sp.off + so, o - hbase, l};
            so += l;
            if (pg[i] != 3) continue;
            uint64_t iv[8][2];   // the page's in-flight written bytes (merged: one copy per byte)
            int ni = 0;
            for (uint64_t k = gw_head; k > 0 && k + GW_N > gw_head && gw_e[(k - 1) % GW_N].seq > done; k--) {
                const typeof(gw_e[0]) *p = &gw_e[(k - 1) % GW_N];
                uint64_t a = p->addr - HEAP_BASE > o ? p->addr - HEAP_BASE : o, e = p->addr - HEAP_BASE + p->size < o + l ? p->addr - HEAP_BASE + p->size : o + l;
                if (p->seq >= cur || p->addr < HEAP_BASE || a >= e) continue;
                int j = 0;
                for (; j < ni; j++)
                    if (a <= iv[j][1] && iv[j][0] <= e) { iv[j][0] = a < iv[j][0] ? a : iv[j][0]; iv[j][1] = e > iv[j][1] ? e : iv[j][1]; break; }
                if (j == ni && ni < 8) { iv[ni][0] = a; iv[ni][1] = e; ni++; }
                else if (j == ni) { ni = 1; iv[0][0] = o; iv[0][1] = o + l; break; }   // (too many: the whole page, after the snapshot)
            }
            for (int j = 0; j < ni; j++)
                for (int m = j + 1; m < ni; m++)   // (merges made above can now touch a later interval)
                    if (iv[m][0] <= iv[j][1] && iv[j][0] <= iv[m][1]) {
                        iv[j][0] = iv[m][0] < iv[j][0] ? iv[m][0] : iv[j][0]; iv[j][1] = iv[m][1] > iv[j][1] ? iv[m][1] : iv[j][1];
                        iv[m][0] = iv[--ni][0]; iv[m][1] = iv[ni][1]; m = j;
                    }
            if (nb + ni + (int)(np - i - 1) > 512) { bc[nb++] = (VkBufferCopy){o, o - hbase, l}; n3++; continue; }   // (room: 1 per page left)
            for (int j = 0; j < ni; j++) bc[nb++] = (VkBufferCopy){iv[j][0], iv[j][0] - hbase, iv[j][1] - iv[j][0]};
            n3 += ni;
        }
        if (nb || ns) {
            VkCommandBuffer up = vk_upload_cmd();
            if (barrier_seq != cur) { vk_full_barrier(up); barrier_seq = cur; }  // after the earlier submissions' reads of the mirror
            if (ns) vkCmdCopyBuffer(up, sp.buf, mb, (uint32_t)ns, sc);
            if (ns && n3) vk_full_barrier(up);   // (written ranges over their page's snapshot)
            if (nb) vkCmdCopyBuffer(up, gmem_buf, mb, (uint32_t)nb, bc);
        }
        prof_hm_syncs++;
        prof_hm_bytes += bytes;
        prof_hm_ranges += nb + ns;
        if (prof) prof_hm_ns += prof_now() - t0;
    }
    for (uint64_t ch = (addr - HEAP_BASE) >> 20, ce = (addr - HEAP_BASE + (size ? size - 1 : 0)) >> 20; ch <= ce; ch++) {
        if (hm_chunk_vol[ch] != cur) continue;
        for (uint64_t q = (addr - HEAP_BASE) >> 12, qe = (addr - HEAP_BASE + (size ? size - 1 : 0)) >> 12; q <= qe; q++)
            if (hm_pg_vol[q] == cur) return false;  // rewritten since the mirror got it this submission: copied instead
        break;
    }
    for (uint64_t q = (addr - HEAP_BASE) >> 12, qe = (addr - HEAP_BASE + (size ? size - 1 : 0)) >> 12; hm_pg_gpu && q <= qe; q++)
        if (hm_pg_gpu[q] == cur) { rc_record(3, addr, size); break; }   // (read from host memory at GPU time)
    *buf = mb;
    *off = addr - HEAP_BASE - hbase;
    return true;
}
// (device.c "[vram]": the VRAM mirrors' sizes; used[0..3] = heap / alias 1 MB chunks ever bound, heap / alias 64 MB blocks
// holding one)
uint64_t vk_mirror_bytes(uint64_t *alias, int used[4]) {
    if (alias) *alias = g_mirror_blk_bytes[1];
    uint64_t hb = 0, ab = 0;
    used[0] = used[1] = 0;
    for (uint64_t c = 0; hm_chunk_key && c < ((hm_size + (1u << 20) - 1) >> 20); c++) if (hm_chunk_key[c]) { used[0]++; hb |= 1ull << ((c >> 6) & 63); }
    for (uint64_t c = 0; alias_chunk_key && c < alias_chunk_n; c++) if (alias_chunk_key[c]) { used[1]++; ab |= 1ull << ((c >> 6) & 63); }
    used[2] = __builtin_popcountll(hb);
    used[3] = __builtin_popcountll(ab);
    return g_mirror_blk_bytes[0];
}

// The mirror, unless a shader of this submission writes the range (host memory / the window is newer then).
static bool heap_mirror_bind(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (addr < HEAP_BASE || !size) return false;
    hm_why = 1;
    for (uint64_t bk = (addr - HEAP_BASE) >> GDEV_SH, be = (addr + size - 1 - HEAP_BASE) >> GDEV_SH; bk <= be && bk < GDEV_BLKS; bk++)
        if (gdev_blk[bk] == submit_seq + 1) return false;
    hm_why = 0;
    return heap_mirror(addr, size, buf, off);
}

static bool guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    if (copy) return false;
    if (alias_direct(addr, size, buf, off)) return true;
    // Heap vertex / index data copied (dirty pages through the heap mirror; LARECOMP_VK_COPY_HEAP_VERTICES=0: bound in
    // place). Discrete GPU: run514 (RTX 5070, 2x) mean fps 74 -> 108, CPU wait on the GPU 765 -> 8 ms per 2 s - in place,
    // every vertex / instance fetch crossed PCIe. (Not the 2026-10-19 decal hang: run515 hung with it.) iGPU too since
    // 2026-10-20: in place, the game rewrote small dynamic vertex buffers (0x2c0-0x560 at 0x1010f1xxxx) while the GPU could
    // still read them (SyncWait lag): LARECOMP_RACECHECK Deck R1 580-6234 ranges changed in flight per 600 SyncWaits, R4
    // copied 14-26 (R3 SYNC_LAG=0 floor 0-98); Deck fps with copies within noise (D 62.2 vs C 61.3 / E 64.1).
    static int copy_heap = -1;
    if (copy_heap < 0) {
        const char *e = getenv("LARECOMP_VK_COPY_HEAP_VERTICES");
        copy_heap = e ? strcmp(e, "0") != 0 : 1;
    }
    if (addr < HEAP_BASE) return false;
    if (addr + size > HEAP_BASE + gmem_size && kernel_heap_size() != gmem_tried) { gmem_tried = kernel_heap_size(); gmem_info(); }
    if (!gmem_buf || addr + size > HEAP_BASE + gmem_size) {
        static int nlog;
        if (prof && nlog++ < 12)
            LOG("[vk] direct bind refused: range %llx +%llx ends past the heap import (%llx +%llx)\n", (unsigned long long)addr,
                (unsigned long long)size, (unsigned long long)HEAP_BASE, (unsigned long long)gmem_size);
        return false;
    }
    for (int i = 0; gdev_lv_seq == submit_seq + 1 && i < gdev_lv_n; i++) {  // written by a shader this submission: in the device window
        const typeof(gdev_lv[0]) *r = &gdev_lv[i];
        if (addr >= r->addr + r->size || r->addr >= addr + size) continue;
        if (addr < r->addr || addr + size > r->addr + r->size) {  // straddles it: (stale) copy
            static int nlog;
            prof_vx_straddle++;
            if (nlog++ < 20 || g_debug_frame)
                LOG("[vk] vertex/index range %llx +%llx straddles live window region %llx +%llx: copied from host (stale)\n",
                    (unsigned long long)addr, (unsigned long long)size, (unsigned long long)r->addr, (unsigned long long)r->size);
            return false;
        }
        if (g_debug_frame) LOG("[vk] vertex/index range %llx +%llx read from the window\n", (unsigned long long)addr, (unsigned long long)size);
        *buf = gdev_buf;
        *off = r->off + (addr - r->addr);
        return true;
    }
    if (copy_heap && heap_mirror_bind(addr, size, buf, off)) return true;
    if (copy_heap) {  // copied unless a submission still in flight writes it (the copy would read host memory too early)
        uint64_t done = 0;
        bool live = false;
        for (uint64_t bk = (addr - HEAP_BASE) >> GDEV_SH, be = (addr + size - 1 - HEAP_BASE) >> GDEV_SH; bk <= be && bk < GDEV_BLKS && !live; bk++)
            if (gdev_blk[bk] && gdev_blk[bk] > (done ? done : (done = vk_completed_seq()))) live = true;
        if (live && !live_pages() && !gw_overlap(addr, size, done, ~0ull)) live = false;   // CPU bytes on a shader-written page
        if (!live) return false;
        prof_vx_inflight++;
    }
    *buf = gmem_buf;
    *off = addr - HEAP_BASE;
    rc_record(1, addr, size);
    return true;
}

static int g_cs_n;  // (compute program table below)

// Vertex / index buffer bindings are command buffer state (they survive pipeline binds and passes): skip a bind equal
// to the current one. Keyed by (command buffer, submission): the ring reuses command buffer handles.
static VkCommandBuffer bind_cb;
static uint64_t bind_seq;
static VkBuffer bind_vb[17], bind_ib;
static VkDeviceSize bind_vo[17], bind_io;
static VkIndexType bind_it;
static void bind_check(VkCommandBuffer cb) {
    uint64_t q = vk_pending_seq();
    if (bind_cb != cb || bind_seq != q) { bind_cb = cb; bind_seq = q; memset(bind_vb, 0, sizeof bind_vb); bind_ib = VK_NULL_HANDLE; }
}
static void bind_vertex(VkCommandBuffer cb, uint32_t s, VkBuffer b, VkDeviceSize o) {
    bind_check(cb);
    if (bind_vb[s] == b && bind_vo[s] == o) { prof_binds_skipped++; return; }
    bind_vb[s] = b; bind_vo[s] = o;
    prof_binds_done++;
    vkCmdBindVertexBuffers(cb, s, 1, &b, &o);
}
static void bind_index(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    bind_check(cb);
    if (bind_ib == b && bind_io == o && bind_it == t) { prof_binds_skipped++; return; }
    bind_ib = b; bind_io = o; bind_it = t;
    prof_binds_done++;
    vkCmdBindIndexBuffer(cb, b, o, t);
}

// In-place reboot (device.c backend_reset; GPU idle): drop what is keyed by guest addresses - programs by code
// address (the old Prog / Pipe objects are leaked, so pipeline keys holding their pointers can't collide), compute
// programs, shared copies - and the heap import: its pages are replaced (re-imported on first use).
void vk_exec_reset(void) {
    memset(prog_tab, 0, sizeof prog_tab);
    memset(pipe_tab, 0, sizeof pipe_tab);
    g_memo_gen++;   // (draw memos: get_program, the pipeline key)
    g_cs_n = 0;
    gcopy_gen++;
    if (gmem_buf) {
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
        gmem_buf = VK_NULL_HANDLE;
        gmem_mem = VK_NULL_HANDLE;
    }
    gmem_size = gmem_tried = 0;
    if (alias_buf) {
        vkDestroyBuffer(g_vk.dev, alias_buf, NULL);
        vkFreeMemory(g_vk.dev, alias_mem, NULL);
        alias_buf = VK_NULL_HANDLE;
        alias_mem = VK_NULL_HANDLE;
    }
    if (alias_dev) {
        for (uint64_t i = 0; i < al_nblk; i++) if (al_blk[i]) vmaDestroyBuffer(g_vk.vma, al_blk[i], al_blk_mem[i]);
        free(al_blk);
        free(al_blk_mem);
        al_blk = NULL;
        al_blk_mem = NULL;
        al_nblk = 0;
        g_mirror_blk_bytes[1] = 0;
        alias_dev = VK_NULL_HANDLE;
    }
    alias_sync_seq = 0;
    alias_sync_epoch = 0;
    if (alias_chunk_key) memset(alias_chunk_key, 0, alias_chunk_n * sizeof *alias_chunk_key);
    alias_state = 0;
    gdev_n = 0;  // (GPU idle: the window's contents went back to the old heap)
    gdev_top = 0;
    gdev_half = 0;
    gdev_half_seq[0] = gdev_half_seq[1] = 0;
    gdev_lv_n = 0;
    gdev_lv_seq = 0;
    st_defaults();
}

static void texture_info(int stage, ShaderInfo *si, int i, VkDescriptorImageInfo *out, float sc[2], uint32_t wh[2]) {
    ShaderTexture *st_ = &si->tex[i];
    int slot = ((int)st_->handle_offset - 0x20) / 8;  // 64-bit handles from offset 0x20
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler smp = VK_NULL_HANDLE;
    uint32_t tic = h & 0xFFFFF, tsc = (uint32_t)(h >> 20) & 0xFFF;
    sc[0] = sc[1] = 1;
    wh[0] = wh[1] = 0;
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        NvnTexture *tex = e->tex;  // read once (finalize clears entries; nvn.h backend_exec_lock)
        if (tex) {
            // Per-frame memo: ~2400 lookups per overworld frame recomputed the view (format / range math, cache
            // probe). The first lookup of a frame still runs vk_view (upload check); a view that may have been
            // destroyed (eviction, image re-created) bumps vk_view_gen. LARECOMP_VK_NO_TEX_MEMO=1 disables.
            extern uint32_t vk_view_gen;
            extern uint64_t vk_frame_counter(void);
            static int nomemo = -1;
            if (nomemo < 0) nomemo = getenv("LARECOMP_VK_NO_TEX_MEMO") != NULL;
            static struct { NvnTexture *tex; uint64_t vh, frame; uint32_t gen; uint8_t type, has_view; VkImageView view; } memo[1024];
            uint64_t vh = e->has_view ? hash_words(0, &e->view, sizeof e->view) : 0, frame = vk_frame_counter();
            uint32_t ms = (uint32_t)((((uintptr_t)tex >> 4) ^ vh ^ (uint64_t)st_->type * 0x9E3779B97F4A7C15ull) & 1023);
            if (!nomemo && memo[ms].tex == tex && memo[ms].vh == vh && memo[ms].frame == frame && memo[ms].gen == vk_view_gen &&
                memo[ms].type == st_->type && memo[ms].has_view == e->has_view) {
                view = memo[ms].view;
            } else {
                view = vk_view(tex, &e->view, e->has_view, false, st_->type);
                memo[ms].tex = tex; memo[ms].vh = vh; memo[ms].frame = frame; memo[ms].gen = vk_view_gen;
                memo[ms].type = (uint8_t)st_->type; memo[ms].has_view = e->has_view; memo[ms].view = view;
            }
            renderer_texture_sample_scale(tex, &sc[0], &sc[1]);
            wh[0] = tex->width; wh[1] = tex->height;
            for (int k = 0; k < nfeedback; k++)
                if (tex == feedback_tex[k] && feedback_view[k] && (st_->type == TEXTYPE_2D || st_->type == TEXTYPE_2D_ARRAY)) {
                    view = feedback_view[k];
                    if (st_->type == TEXTYPE_2D_ARRAY) view = vk_dummy_view(TEXTYPE_2D_ARRAY);  // (2D snapshot as array: rare)
                }
            if (pk_subst_tex && tex == pk_subst_tex && st_->type == TEXTYPE_2D) view = pk_subst_view;   // (pause capture)
        }
    }
    if (h && st.spool && tsc < (uint32_t)st.spool->count) smp = vk_sampler(&((NvnSamplerDesc *)st.spool->entries)[tsc]);
    if (!view) view = vk_dummy_view(st_->type);
    if (!smp) smp = vk_default_sampler(true);
    *out = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL};
}

// stage_set reuse table: written descriptor sets of the pending submission by content hash (sets live until their
// submission's descriptor pools are reset, sub()).
#define SET_TAB 4096
typedef struct { uint64_t h, seq; VkDescriptorSetLayout layout; VkDescriptorSet set; } SetEnt;
static SetEnt set_tab[SET_TAB];

// Writes the descriptor set of one stage. gx..gz: compute group counts (driver constants).
// pre_ii: texture descriptors already resolved by the caller (draw: fragment textures, needed earlier for the
// viewport decision), or NULL.
// dyn / ndyn: the set's dynamic uniform-buffer offsets in binding order (for vkCmdBindDescriptorSets), appended at
// dyn[*ndyn]; *ndyn is advanced.
#define UBO_DYN(t) ((t) == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || (t) == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
// Per-stage memo of stage_set's INPUTS (2026-10-19, run579: 74-78 % of calls build the same set as the stage's previous
// call, ~390 ms of fill + cbuf per 2 s spent finding that out). g_set_gen counts the commands that write the state it
// reads (exec_list). LARECOMP_SET_MEMO=count (default for now): only counts would-skips and violations (would-skip but
// the built set / offsets differ) in the "[prof] stage_set" line.
static uint64_t g_set_gen;
uint64_t prof_memo_hit[6], prof_memo_bad[6];
typedef struct {
    uint64_t gen, q, epoch, frame, h;
    uint32_t view_gen, pool_gen, gxyz[3], nd, d[MAX_BIND];
    VkDescriptorSet ds;   // the set the call returned
    Prog *p; VkDescriptorSetLayout layout; uint32_t set; ShaderInfo *si;
    float rs0[4], tint[8]; bool has_tint;
    VkDescriptorImageInfo pre[64]; int npre;
    Range ubuf[18], sbuf[16]; uint64_t tex[32]; void *prog, *tpool, *spool;   // the stage's bound state (contents)
    uint64_t wk[MAX_BIND][3]; int nw;   // the writes built last time (binding|type, view|buffer^range, sampler|offset)
} StageMemo;
// why a call that built the same set as last time missed the memo (count mode): one counter per first differing input
enum { MW_GEN, MW_SEQ, MW_VIEW, MW_PROG, MW_TEX, MW_UBUF, MW_SBUF, MW_PRE, MW_OTHER, MW_N };
uint64_t prof_memo_why[6][MW_N];
static StageMemo g_smemo[6];
static VkDescriptorSet stage_set(Prog *p, VkDescriptorSetLayout layout, uint32_t set, int stage, ShaderInfo *si, const float *rscale_row0,
                                 const float tint[8], uint32_t gx, uint32_t gy, uint32_t gz, const VkDescriptorImageInfo *pre_ii,
                                 uint32_t *dyn, uint32_t *ndyn) {
    uint64_t z0 = prof ? prof_now() : 0;
    VkDescriptorSet ds = VK_NULL_HANDLE;   // (allocated below, once the writes are known: an identical set may exist)
    uint64_t z1 = z0, zc = 0;
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    int n = 0;
    int sg = stage == 1 ? 1 : 0;
    // (rscale: only rows 0..ntex are filled, copied and bound: the shader reads rscale[0] and rscale[1 + slot], slot < ntex.
    // Was the whole 65-row table, zeroed, compared and copied twice per draw, 2026-10-17)
    float rs[65][4];
    rs[0][0] = 1; rs[0][1] = 1; rs[0][2] = 0; rs[0][3] = 0;
    if (rscale_row0) memcpy(rs[0], rscale_row0, 16);
    StageMemo *m = stage >= 0 && stage < 6 ? &g_smemo[stage] : NULL;
    extern uint32_t vk_view_gen;
    extern uint64_t vk_frame_counter(void);
    uint64_t m_epoch = in_execute ? alias_epoch : 0, m_frame = vk_frame_counter(), m_q = vk_pending_seq();
    int npre = pre_ii ? (si->ntex < 64 ? si->ntex : 64) : 0;
    extern uint32_t g_pool_gen;
    uint32_t m_pool_gen = __atomic_load_n(&g_pool_gen, __ATOMIC_RELAXED);
    bool memo_hit = m && !nfeedback && !pk_subst_tex && !si->uses_gmem && m->ds && m->pool_gen == m_pool_gen && m->gen == g_set_gen && m->q == m_q && m->epoch == m_epoch &&
                    m->frame == m_frame && m->view_gen == vk_view_gen && m->p == p && m->layout == layout && m->set == set && m->si == si &&
                    m->gxyz[0] == gx && m->gxyz[1] == gy && m->gxyz[2] == gz && !memcmp(m->rs0, rs[0], 16) &&
                    m->has_tint == (tint != NULL) && (!tint || !memcmp(m->tint, tint, 32)) && m->npre == npre &&
                    (!npre || !memcmp(m->pre, pre_ii, (size_t)npre * sizeof *pre_ii)) && m->prog == (void *)st.program &&
                    m->tpool == (void *)st.tpool && m->spool == (void *)st.spool && !memcmp(m->tex, st.tex[stage], sizeof m->tex) &&
                    !memcmp(m->ubuf, st.ubuf[stage], sizeof m->ubuf) && !memcmp(m->sbuf, st.sbuf[stage], sizeof m->sbuf);
    // Skip (default; LARECOMP_SET_MEMO=0 off): same inputs as this stage's previous call in this submission and list -> its set and
    // offsets. run582: the only differences such calls ever built were new copies of the same guest ranges (dyn offsets).
    // Not for shaders using global memory (gdev_map's per-call window bookkeeping) or feedback / capture substitutions.
    static int memo_skip = -1;
    if (memo_skip < 0) { const char *e = getenv("LARECOMP_SET_MEMO"); memo_skip = !e || *e == '1'; }   // (default on; =0 off)
    if (memo_skip && memo_hit) {
        memcpy(dyn + *ndyn, m->d, (size_t)m->nd * 4);
        *ndyn += m->nd;
        prof_memo_hit[stage]++;
        if (prof) prof_set_fill += prof_now() - z0;
        return m->ds;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            float sc[2]; uint32_t wh[2];
            if (ti < si->ntex && pre_ii && ti < 64) { ii[n] = pre_ii[ti]; sc[0] = tex_sc[sg][ti][0]; sc[1] = tex_sc[sg][ti][1]; wh[0] = tex_wh[sg][ti][0]; wh[1] = tex_wh[sg][ti][1]; }
            else if (ti < si->ntex) texture_info(stage, si, ti, &ii[n], sc, wh);
            else { ii[n] = (VkDescriptorImageInfo){vk_default_sampler(true), vk_dummy_view(TEXTYPE_2D), VK_IMAGE_LAYOUT_GENERAL}; sc[0] = sc[1] = 1; wh[0] = wh[1] = 0; }
            if (ti < 64) {
                tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1];
                tex_wh[sg][ti][0] = wh[0]; tex_wh[sg][ti][1] = wh[1];
            }
            w[n].pImageInfo = &ii[n];
        } else if (UBO_DYN(b->type) && b->binding < 17) {
            uint64_t c0 = prof ? prof_now() : 0;
            bi[n] = cbuf_info(stage, si, (int)b->binding, gx, gy, gz);
            if (prof) zc += prof_now() - c0;
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            bi[n] = b->binding == GDEV_BUFFER_INDEX ? gdev_info() : gmem_info();
            w[n].pBufferInfo = &bi[n];
        } else {
            continue;  // rscale / tint: written below, once the textures (and their scales) are known
        }
        n++;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set || !UBO_DYN(b->type) || (b->binding != 18 && b->binding != 19)) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->binding == 18) {
            int nt = si->ntex < 64 ? si->ntex : 64;
            for (int t = 0; t < nt; t++) { rs[1 + t][0] = tex_sc[sg][t][0]; rs[1 + t][1] = tex_sc[sg][t][1]; rs[1 + t][2] = rs[1 + t][3] = 0; }
            uint64_t rsz = (uint64_t)(1 + nt) * 16;
            bi[n] = ubo_span(copy_dedupe(sg ? DEDUPE_RSCALE_F : DEDUPE_RSCALE_V, rs, rsz, 256), rsz);
        } else {
            float zero[8] = {0};
            bi[n] = ubo_span(copy_dedupe(DEDUPE_TINT, tint ? tint : zero, 32, 256), 32);
        }
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    {   // dynamic uniform buffers: the offset goes to the bind (binding order), the descriptor holds offset 0
        uint32_t nd = 0, bnd[MAX_BIND];
        for (int i = 0; i < n; i++) {
            if (w[i].descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) continue;
            uint32_t j = nd++;
            while (j > 0 && bnd[j - 1] > w[i].dstBinding) { bnd[j] = bnd[j - 1]; dyn[*ndyn + j] = dyn[*ndyn + j - 1]; j--; }
            bnd[j] = w[i].dstBinding;
            dyn[*ndyn + j] = (uint32_t)bi[i].offset;
            bi[i].offset = 0;
        }
        *ndyn += nd;
    }
    uint64_t z2 = prof ? prof_now() : 0;
    // Identical set already written in this submission (same layout, same buffers / ranges / views / samplers): bind
    // that one; skips vkAllocateDescriptorSets + vkUpdateDescriptorSets (run362: ~13 % of exec in the driver under
    // vk_alloc_set + stage_set). UBO spans are content- or address-deduped per submission (copy_dedupe, guest_copy), so
    // draws sharing a material share a set. LARECOMP_VK_SET_REUSE=0: a new set every time.
    static int reuse = -1;
    if (reuse < 0) { const char *e = getenv("LARECOMP_VK_SET_REUSE"); reuse = !e || strcmp(e, "0"); }
    uint64_t h = 0, q = vk_pending_seq();
    SetEnt *se = NULL;
    if (reuse) {
        h = (uint64_t)(uintptr_t)layout * 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
        uint64_t wk[MAX_BIND][3];
        for (int i = 0; i < n; i++) {
            uint64_t a0 = (uint64_t)w[i].dstBinding | (uint64_t)w[i].descriptorType << 32, a1, a2;
            if (w[i].pImageInfo) { a1 = (uint64_t)(uintptr_t)w[i].pImageInfo->imageView; a2 = (uint64_t)(uintptr_t)w[i].pImageInfo->sampler ^ (uint64_t)w[i].pImageInfo->imageLayout << 56; }
            else { a1 = (uint64_t)(uintptr_t)w[i].pBufferInfo->buffer ^ w[i].pBufferInfo->range << 40; a2 = w[i].pBufferInfo->offset; }
            wk[i][0] = a0; wk[i][1] = a1; wk[i][2] = a2;
            h = (h ^ a0) * 0xFF51AFD7ED558CCDull;
            h = (h ^ a1) * 0xC4CEB9FE1A85EC53ull;
            h = (h ^ a2) * 0x9E3779B97F4A7C15ull;
            h ^= h >> 29;
        }
        se = &set_tab[(h ^ h >> 32) & (SET_TAB - 1)];
        if (prof && stage >= 0 && stage < 6) {   // (how often a per-stage memo would hit)
            static struct { uint64_t h, q; VkDescriptorSetLayout layout; uint32_t nd, d[MAX_BIND]; } last[6];
            uint32_t nd = 0;
            for (int i = 0; i < n; i++) if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) nd++;
            const uint32_t *d = dyn + *ndyn - nd;
            prof_set_calls[stage]++;
            if (last[stage].h == h && last[stage].q == q && last[stage].layout == layout) {
                prof_set_same[stage]++;
                if (last[stage].nd == nd && !memcmp(last[stage].d, d, nd * 4)) prof_set_samedyn[stage]++;
            }
            last[stage].h = h; last[stage].q = q; last[stage].layout = layout; last[stage].nd = nd; memcpy(last[stage].d, d, nd * 4);
        }
        if (m) {   // inputs memo (count mode): a would-skip must have built the same set and offsets as last time
            uint32_t nd = 0;
            for (int i = 0; i < n; i++) if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) nd++;
            const uint32_t *d = dyn + *ndyn - nd;
            bool same = m->h == h && m->nd == nd && !memcmp(m->d, d, nd * 4);
            if (memo_hit) {
                prof_memo_hit[stage]++;
                if (!same) {
                    prof_memo_bad[stage]++;
                    static int nlog;   // the first violations: which write differs (a missing input in the key)
                    if (nlog < 60) {
                        nlog++;
                        LOG("[memo] violation st%d n %d/%d nd %u/%u:", stage, m->nw, n, m->nd, nd);
                        for (int i = 0; i < n && i < m->nw; i++)
                            if (memcmp(m->wk[i], wk[i], sizeof wk[i]))
                                LOG(" [b%u t%u %llx/%llx -> %llx/%llx]", (unsigned)wk[i][0], (unsigned)(wk[i][0] >> 32), (unsigned long long)m->wk[i][1],
                                    (unsigned long long)m->wk[i][2], (unsigned long long)wk[i][1], (unsigned long long)wk[i][2]);
                        for (uint32_t i = 0; i < nd && i < m->nd; i++)
                            if (m->d[i] != d[i]) LOG(" [dyn%u %x -> %x]", i, m->d[i], d[i]);
                        LOG("\n");
                    }
                }
            } else if (same && m->p == p && m->layout == layout) {
                int why = m->gen != g_set_gen ? MW_GEN : (m->q != m_q || m->epoch != m_epoch || m->frame != m_frame) ? MW_SEQ :
                          (m->view_gen != vk_view_gen || m->pool_gen != m_pool_gen) ? MW_VIEW : (m->prog != (void *)st.program || m->tpool != (void *)st.tpool || m->spool != (void *)st.spool) ? MW_PROG :
                          memcmp(m->tex, st.tex[stage], sizeof m->tex) ? MW_TEX : memcmp(m->ubuf, st.ubuf[stage], sizeof m->ubuf) ? MW_UBUF :
                          memcmp(m->sbuf, st.sbuf[stage], sizeof m->sbuf) ? MW_SBUF : (m->npre != npre || (npre && memcmp(m->pre, pre_ii, (size_t)npre * sizeof *pre_ii))) ? MW_PRE : MW_OTHER;
                prof_memo_why[stage][why]++;
            }
            m->gen = g_set_gen; m->q = m_q; m->epoch = m_epoch; m->frame = m_frame; m->h = h; m->view_gen = vk_view_gen; m->pool_gen = m_pool_gen;
            m->gxyz[0] = gx; m->gxyz[1] = gy; m->gxyz[2] = gz; m->nd = nd; memcpy(m->d, d, nd * 4);
            m->p = p; m->layout = layout; m->set = set; m->si = si; memcpy(m->rs0, rs[0], 16);
            m->has_tint = tint != NULL; if (tint) memcpy(m->tint, tint, 32);
            m->npre = npre; if (npre) memcpy(m->pre, pre_ii, (size_t)npre * sizeof *pre_ii);
            m->prog = st.program; m->tpool = st.tpool; m->spool = st.spool;
            memcpy(m->tex, st.tex[stage], sizeof m->tex); memcpy(m->ubuf, st.ubuf[stage], sizeof m->ubuf); memcpy(m->sbuf, st.sbuf[stage], sizeof m->sbuf);
            m->nw = n; memcpy(m->wk, wk, (size_t)n * sizeof wk[0]);
        }
        if (se->seq == q && se->h == h && se->layout == layout) {
            prof_set_reused++;
            if (prof) prof_set_fill += z2 - z1 - zc, prof_set_cbuf += zc;
            if (m) m->ds = se->set;
            return se->set;
        }
    }
    ds = vk_alloc_set(layout);
    for (int i = 0; i < n; i++) w[i].dstSet = ds;
    uint64_t z2b = prof ? prof_now() : 0;
    if (n) vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    if (se) *se = (SetEnt){h, q, layout, ds};
    if (m && reuse) m->ds = ds;
    if (prof) {
        uint64_t z3 = prof_now();
        prof_set_alloc += z2b - z2; prof_set_cbuf += zc; prof_set_fill += z2 - z1 - zc; prof_set_update += z3 - z2b;
        prof_set_writes += (uint64_t)n;
    }
    return ds;
}

// ================================================================== compute
typedef struct { Prog p; ShaderInfo *si; VkPipeline pipe; uint32_t local[3]; } CsProg;

static void build_compute(void *arg) {
    CsProg *cp = arg;
    size_t n = 0;
    uint32_t *spv = vk_glsl_to_spirv(cp->si->msl, STAGE_COMPUTE, "cs", &n);
    if (!spv) { atomic_store(&cp->p.state, 2); return; }
    cp->p.nbinds = vk_reflect(spv, n, cp->p.binds, MAX_BIND);
    VkShaderModule m = vk_module(spv, n);
    free(spv);
    cp->p.layout = make_layout(&cp->p, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo spec = {3, me, 12, cp->local};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = &spec},
                                      .layout = cp->p.layout};
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &cp->pipe);
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&cp->p.state, r == VK_SUCCESS ? 1 : 2);
    if (r != VK_SUCCESS) LOG("[compute] pipeline failed: %d\n", r);
}

static struct CsEnt { uint64_t addr; CsProg *cp; } g_cs_tab[256];  // compute programs by code address

// compute programs by content (code hash + workgroup size): built ahead by the shader precompile, or the same code at
// another address
#define PC_CSTAB 1024
static struct { uint64_t key; CsProg *cp; } pc_cstab[PC_CSTAB];   // (pc_mu)
static CsProg *pc_cs_get(uint64_t ck, CsProg *put) {   // the program under ck; else registers `put` (if given) and returns it
    pthread_mutex_lock(&pc_mu);
    uint32_t i = (uint32_t)(ck ^ (ck >> 29)) & (PC_CSTAB - 1);
    int probes = 0;
    while (pc_cstab[i].key && pc_cstab[i].key != ck && probes++ < PC_CSTAB) i = (i + 1) & (PC_CSTAB - 1);
    CsProg *r = pc_cstab[i].key == ck ? pc_cstab[i].cp : NULL;
    if (!r && put && !pc_cstab[i].key) { pc_cstab[i].key = ck; pc_cstab[i].cp = put; r = put; }
    pthread_mutex_unlock(&pc_mu);
    return r;
}
static uint64_t pc_comp_key(uint64_t code_hash, const uint32_t local[3]) {
    uint64_t k = hash_words(code_hash ^ PC_SEEN_CS, local, 12);
    return k ? k : 1;
}
// code + compiler constants into cache/code (as pc_record_prog does for graphics programs), then the PC_COMP record
static void pc_record_comp(uint64_t code, uint64_t cmax, uint64_t csize, uint64_t chash, uint64_t ck, const uint32_t local[3]) {
    if (!pc_seen_add(ck ^ PC_SEEN_CS)) return;
    mkdir("cache/code", 0755);
    char path[64];
    snprintf(path, sizeof path, "cache/code/%016llx.bin", (unsigned long long)chash);
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    else {
        if (!(f = fopen(path, "wb"))) return;
        bool ok = fwrite((const void *)(uintptr_t)code, 1, (size_t)cmax, f) == cmax;
        fclose(f);
        if (!ok) { remove(path); return; }
    }
    PcComp r;
    memset(&r, 0, sizeof r);
    if (csize && pc_readable(code + cmax, csize)) {
        snprintf(path, sizeof path, "cache/code/%016llx.c1", (unsigned long long)chash);
        long have = -1;
        if ((f = fopen(path, "rb"))) { fseek(f, 0, SEEK_END); have = ftell(f); fclose(f); }
        if ((uint64_t)have == csize) r.csize = csize;
        else if ((f = fopen(path, "wb"))) {
            bool ok = fwrite((const void *)(uintptr_t)(code + cmax), 1, (size_t)csize, f) == csize;
            fclose(f);
            if (ok) r.csize = csize;
            else remove(path);
        }
    }
    r.ckey = ck;
    r.code_hash = chash;
    r.cmax = cmax;
    memcpy(r.local, local, sizeof r.local);
    pc_write(PC_COMP, &r, sizeof r);
}
// precompile worker: translate + build a recorded compute program
static void pc_comp_build(const PcComp *r) {
    if (pc_cs_get(r->ckey, NULL)) return;
    uint64_t cs = r->csize;
    uint8_t *b = pc_load_code_c1(r->code_hash, r->cmax, &cs);
    if (!b) return;
    CsProg *cp = calloc(1, sizeof *cp);
    memcpy(cp->local, r->local, sizeof cp->local);
    ShaderKey k = {0};
    k.glsl = 1;
    pthread_mutex_lock(&translate_lock);
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    maxwell_gdev_size = gdev_window_size();
    cp->si = translate_stage((uint64_t)(uintptr_t)b, r->cmax, cs, STAGE_COMPUTE, &k);
    pthread_mutex_unlock(&translate_lock);
    free(b);
    cp->p.vi = cp->si;
    if (!cp->si->msl || pc_cs_get(r->ckey, cp) != cp) { maxwell_free(cp->si); free(cp); return; }   // (failed, or the game was first)
    build_compute(cp);
}

static CsProg *compute_program(NvnProgram *np) {
    NvnShaderStage *ss = &np->stages[5];
    if (!ss->code_addr) return NULL;
    struct CsEnt *tab = g_cs_tab;
    int n = g_cs_n;
    for (int i = 0; i < n; i++) if (tab[i].addr == ss->code_addr) return tab[i].cp;
    uint32_t local[3];
    for (int i = 0; i < 3; i++) {
        local[i] = ss->control ? *(const uint32_t *)(ss->control + 0x748 + 4 * i) : 1;
        if (!local[i]) local[i] = 1;
    }
    uint64_t cmax = ss->const_offset ? ss->const_offset : 0x10000;
    uint64_t csize = ss->const_offset ? (ss->const_size < 0x10000 ? ss->const_size : 0x10000) : 0;
    uint64_t chash = 0, ck = 0;
    if (pc_on() && cmax <= 0x100000 && pc_readable(ss->code_addr, cmax)) {
        pc_start();
        chash = hash_words(0x6a09e667f3bcc908ull ^ cmax, (const void *)(uintptr_t)ss->code_addr, (size_t)cmax);
        ck = pc_comp_key(chash, local);
        CsProg *q = pc_cs_get(ck, NULL);
        if (q && atomic_load(&q->p.state) != 2) {   // (built or being built; a failed one is translated again below)
            if (n < 256) { tab[n].addr = ss->code_addr; tab[n].cp = q; g_cs_n = n + 1; }
            return q;
        }
    }
    CsProg *cp = calloc(1, sizeof *cp);
    if (n < 256) { tab[n].addr = ss->code_addr; tab[n].cp = cp; g_cs_n = n + 1; }
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    maxwell_gdev_size = gdev_window_size();
    ShaderKey k = {0};
    k.glsl = 1;
    pthread_mutex_lock(&translate_lock);
    cp->si = translate_stage(ss->code_addr, cmax, csize, STAGE_COMPUTE, &k);
    pthread_mutex_unlock(&translate_lock);
    cp->p.vi = cp->si;
    if (!cp->si->msl) {
        LOG("[compute] translate failed: %s (code %llx)\n", cp->si->error, (unsigned long long)ss->code_addr);
        static int dumps;   // TotK: what is at the code address (run76: all-zero first instruction)
        if (dumps++ < 3) {
            const uint32_t *w = (const uint32_t *)(uintptr_t)ss->code_addr;
            for (int r = 0; r < 8; r++)
                LOG("  code+%02x: %08x %08x %08x %08x\n", r * 16, w[r * 4], w[r * 4 + 1], w[r * 4 + 2], w[r * 4 + 3]);
            if (ss->control) {
                const uint32_t *k = (const uint32_t *)ss->control;
                LOG("  control %p: %08x %08x %08x %08x | +6f0: %08x %08x %08x %08x %08x\n", (void *)ss->control, k[0], k[1], k[2], k[3],
                    k[0x6f0 / 4], k[0x6f4 / 4], k[0x6f8 / 4], k[0x6fc / 4], k[0x700 / 4]);
            }
            uint64_t nz = 0; const uint64_t *q = (const uint64_t *)(uintptr_t)ss->code_addr;
            for (int i = 1; i < 0x2000 / 8; i++) if (q[i]) { nz = (uint64_t)i * 8; break; }
            LOG("  first nonzero qword at code+%llx\n", (unsigned long long)nz);
        }
        atomic_store(&cp->p.state, 2);
        return cp;
    }
    memcpy(cp->local, local, sizeof cp->local);
    save_glsl(cp->si->msl, ss->code_addr, "cs");
    if (ck) { pc_cs_get(ck, cp); pc_record_comp(ss->code_addr, cmax, csize, chash, ck, local); }
    // SPIR-V already in the disk cache: build it here (module + pipeline-cache hit, about a millisecond) instead of
    // skipping the dispatch, which holds the frame (the user's 10-15 PC session: 20 holds for a compute shader alone)
    if (vk_spirv_cached(cp->si->msl, STAGE_COMPUTE)) build_compute(cp);
    else run_job(build_compute, cp);
    return cp;
}

static void dispatch_compute(uint32_t gx, uint32_t gy, uint32_t gz) {
    snap_reason = "compute"; snapshots_invalidate(NULL);
    if (!st.program || !gx || !gy || !gz) return;
    CsProg *cp = compute_program(st.program);
    if (!cp || atomic_load(&cp->p.state) != 1) { stat_skipped++; stat_skip_why[6]++; if (cp && atomic_load(&cp->p.state) == 0) g_skip_comp_frame++; return; }
    // more than 2^28 threads in one dispatch is a garbage size, not TotK: skipped instead of hanging the GPU
    uint64_t threads = (uint64_t)gx * gy * gz * cp->local[0] * cp->local[1] * cp->local[2];
    if (threads > (1ull << 28)) {
        static int nlog;
        if (nlog++ < 20) LOG("[vk] dispatch skipped: %ux%ux%u groups of %ux%ux%u (cs %llx)\n", gx, gy, gz, cp->local[0], cp->local[1],
                             cp->local[2], (unsigned long long)st.program->stages[5].code_addr);
        stat_skipped++;
        return;
    }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    hang_op(cb, HOP_DISPATCH, st.program->stages[5].code_addr, threads, gx, gy, gz);
    vk_full_barrier(cb);
    gp_mark(cb, "dispatch %ux%ux%u (local %ux%ux%u)", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2]);
    dl_note("dispatch %ux%ux%u cs %llx gmem %d", gx, gy, gz, (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    nfeedback = 0;
    uint32_t cdyn[MAX_BIND], ncdyn = 0;
    VkDescriptorSet ds = stage_set(&cp->p, cp->p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL, cdyn, &ncdyn);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, 0, 1, &ds, ncdyn, cdyn);
    vkCmdDispatch(cb, gx, gy, gz);
    stat_dispatches++; chunk_draws++;
    if (g_debug_frame)   // (with LARECOMP_SAVE_MSL=1 its GLSL is build/glsl/<cs>_cs.glsl)
        LOG("[dispatch] %ux%ux%u local %ux%ux%u cs %llx gmem %d ntex %d\n", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2],
            (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem, cp->si->ntex);
}

// ================================================================== draws
static void debug_draw(Prog *pr, CmdRec *r, bool indexed);
static void debug_probe(Prog *pr);
static void debug_probe_row(Prog *pr);
static bool probe_dt, probe_dw;    // (debug_probe: the depth state draw() set for the probed draw)
static VkCompareOp probe_dop;

// Quads / fans / polygons as a 32-bit triangle index list.
static uint32_t *expand_indices(uint32_t mode, uint32_t count, const void *indices, int itype, uint32_t first, uint32_t *out_count) {
    uint32_t tris = mode == 7 ? count / 4 * 2 : count >= 3 ? count - 2 : 0;
    uint32_t *out = malloc(tris * 3 * 4 + 4);
    uint32_t n = 0;
#define IDX(i) (indices ? (itype == 0 ? ((const uint8_t *)indices)[i] : itype == 1 ? ((const uint16_t *)indices)[i] : ((const uint32_t *)indices)[i]) : first + (i))
    if (mode == 7) {
        for (uint32_t q = 0; q + 3 < count; q += 4) {
            out[n++] = IDX(q); out[n++] = IDX(q + 1); out[n++] = IDX(q + 2);
            out[n++] = IDX(q); out[n++] = IDX(q + 2); out[n++] = IDX(q + 3);
        }
    } else {
        for (uint32_t i = 1; i + 1 < count; i++) { out[n++] = IDX(0); out[n++] = IDX(i); out[n++] = IDX(i + 1); }
    }
#undef IDX
    *out_count = n;
    return out;
}

static bool topology(uint32_t mode, VkPrimitiveTopology *t, uint8_t *cls) {
    switch (mode) {
    case 0: *t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; *cls = 0; return true;
    case 1: *t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; *cls = 1; return true;
    case 3: *t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; *cls = 1; return true;
    case 4: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return true;
    case 5: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; *cls = 2; return true;
    default: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return false;
    }
}

static void draw_(CmdRec *r, bool indexed);
static int flame_probe_x = -1, flame_probe_y;  // (LARECOMP_FLAME_AUTO: the emitter pixel the probe reads)
static bool flame_nodepth;  // (LARECOMP_FLAME_AUTO=...,ab second capture: the torch flame draw without depth test)
static void draw(CmdRec *r, bool indexed) {
    // LARECOMP_NULL_DRAWS=1 (measurement, 2026-10-17): every draw dropped before any work -> the fps the guest CPU alone
    // allows (render thread nearly idle, GPU near idle so the CPU gets the Deck's power budget)
    static int null_draws = -1;
    if (null_draws < 0) null_draws = getenv("LARECOMP_NULL_DRAWS") != NULL;
    if (null_draws) { stat_skipped++; return; }
    PROF_T0();
    draw_(r, indexed);
    PROF_ADD(prof_draw);
}
static uint64_t prof_tp_;
static void draw_(CmdRec *r, bool indexed) {
    if (prof) { prof_tp_ = prof_now(); memset(gps_ns, 0, sizeof gps_ns); }
    if (!st.program) { stat_skipped++; stat_skip_why[0]++; return; }
    ShaderKey vkey = {0}, fkey = {0};
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    {   // only attributes the vertex shader reads (SPH imap: 4 bits per generic attribute at SPH + 0x18) take part in the
        // key: the translator declares just those, and the archive precompile (tools/shader_pack.py) keys them the same way
        const uint8_t *sph = st.program->stages[0].code_addr && (st.program->stage_mask & 1)
                                 ? (const uint8_t *)(uintptr_t)(st.program->stages[0].code_addr + 0x30) : NULL;
        for (int i = 0; i < st.nattrib && i < 32; i++)
            if (!sph || ((sph[0x18 + i / 2] >> ((i & 1) * 4)) & 0xF)) vkey.attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    }
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(rt_view_format(i));
    // One-shot bakes (3D volumes, BC volumes the game encodes on the GPU at load: run289 skipped them all while their new
    // program compiled, and nothing redraws them) never skip: program and pipeline are built / waited for (<= 3 s).
    // Screen passes keep the async skip + held frame. LARECOMP_BAKE_ASYNC=1: bakes skip like the rest (old).
    static int bake_async = -1;
    if (bake_async < 0) bake_async = getenv("LARECOMP_BAKE_ASYNC") != NULL;
    g_draw_bake = false;
    for (int i = 0; i < st.rt.ncolor && !bake_async; i++) {
        NvnTexture *t = st.rt.color[i];
        if (t && (t->target == 2 || (t->format >= 0x41 && t->format <= 0x50))) g_draw_bake = true;
    }
    g_bake_sync = g_draw_bake;
    Prog *pr = get_program(st.program, &vkey, &fkey);
    g_bake_sync = false;
    if (g_draw_bake && pr && atomic_load(&pr->state) == 0) {   // (queued earlier, or by the precompile: wait for it)
        uint64_t t0 = SDL_GetTicksNS();
        while (atomic_load(&pr->state) == 0 && SDL_GetTicksNS() - t0 < 3000000000ull) SDL_Delay(1);
        prof_bake_waits++;
        prof_bake_wait_ns += SDL_GetTicksNS() - t0;
    }
    uint64_t pz1 = prof ? prof_now() : 0;
    if (!pr || atomic_load(&pr->state) != 1) { stat_skipped++; stat_skip_why[1]++; if (pr && atomic_load(&pr->state) == 0) g_skip_prog_frame++; return; }
    {   // fragment-shader hash: interior backdrop detection (widescreen), skip lists, item twinkle (renderer.m)
        static uint64_t last_fs, last_hash;
        uint64_t fs = st.program->stages[1].code_addr;
        if (fs != last_fs) {
            // (cached per code address: the fragment shader changes on most draws and the 512-byte byte-wise hash was
            // ~9 % of draw_ on the Deck, d47 perf annotate. Two code words check that the address still holds the same
            // shader; the hash itself must not change: hmix values are compared with stored constants.)
            static struct { uint64_t fs, w0, w1, h; uint32_t gen; } fsc[512];
            last_fs = fs;
            if (fs) {
                const uint64_t *cw = (const uint64_t *)(uintptr_t)(fs + 0x30);
                uint32_t i = (uint32_t)(fs >> 4 ^ fs >> 13) & 511;
                if (fsc[i].fs != fs || fsc[i].gen != g_memo_gen || fsc[i].w0 != cw[10] || fsc[i].w1 != cw[63])
                    fsc[i] = (__typeof__(fsc[0])){fs, cw[10], cw[63], hmix(1469598103934665603ull, cw, 0x200), g_memo_gen};
                last_hash = fsc[i].h;
            } else last_hash = 0;
        }
        static uint64_t skip[16];
        static int nskip = -1;
        if (nskip < 0) {
            nskip = 0;
            for (const char *p = getenv("LARECOMP_SKIP_FSHASH"); p && *p && nskip < 16;) {
                skip[nskip++] = strtoull(p, (char **)&p, 16);
                while (*p == ',') p++;
            }
        }
        for (int i = 0; i < nskip; i++) if (last_hash == skip[i]) { stat_skipped++; stat_skip_why[2]++; return; }
        extern bool pot_level_active(void);
        if (last_hash == 0xa2177dbb72197bd9ull && pot_level_active()) { stat_skipped++; stat_skip_why[2]++; return; }
        if (g_debug_frame && getenv("LARECOMP_LOG_FSHASH")) LOG("[fshash] fs=%llx hash %016llx\n", (unsigned long long)fs, (unsigned long long)last_hash);
        g_cur_fs_hash = last_hash;
        extern void widescreen_backdrop_seen(uint64_t hash);
        if (last_hash) widescreen_backdrop_seen(last_hash);
        extern void widescreen_scene_ubo(const float *u, uint64_t size);  // (the camera's view-projection)
        if (st.ubuf[0][5].addr) widescreen_scene_ubo((const float *)(uintptr_t)st.ubuf[0][5].addr, st.ubuf[0][5].size);
    }
    flame_nodepth = false;
    if (g_cur_fs_hash == 0x85b366a0e972a9bcull && st.ubuf[0][7].addr && st.ubuf[0][5].size >= 8 * 16) {
        // TotK torch flame: the emitter's screen pixel and distance (cb10 [4..7] = emitter matrix, translation in .w;
        // VS cb8 rows 0-2 = view, rows 4-7 = projection, as the flame VS uses them). LARECOMP_FLAME_AUTO=<metres>[,ab]:
        // the first time the flame is drawn at least that far in front of the camera and on screen, dump the next frame
        // and probe the emitter's pixel; ",ab" adds a second capture (>= 90 frames later) with the flame's depth test off.
        const float *e = (const float *)(uintptr_t)st.ubuf[0][7].addr, *m = (const float *)(uintptr_t)st.ubuf[0][5].addr;
        float v[4] = {0, 0, 0, 1}, c[4];
        for (int i = 0; i < 3; i++) v[i] = m[i * 4] * e[19] + m[i * 4 + 1] * e[23] + m[i * 4 + 2] * e[27] + m[i * 4 + 3];
        for (int i = 0; i < 4; i++) c[i] = m[(4 + i) * 4] * v[0] + m[(4 + i) * 4 + 1] * v[1] + m[(4 + i) * 4 + 2] * v[2] + m[(4 + i) * 4 + 3];
        float nx = c[3] ? c[0] / c[3] : 0, ny = c[3] ? c[1] / c[3] : 0, dist = -v[2];
        int px = (int)(st.viewport[0][0] + (nx * 0.5f + 0.5f) * st.viewport[0][2]);
        int py = (int)(st.viewport[0][1] + (0.5f - 0.5f * ny) * st.viewport[0][3]);
        static float auto_min = -1, auto_dy, auto_near;
        static long auto_after;
        static int auto_ab, captures;
        static uint64_t last_capture;
        if (auto_min < 0) {
            const char *a = getenv("LARECOMP_FLAME_AUTO");
            auto_min = a ? (float)atof(a) : 0;
            auto_ab = a && strstr(a, "ab") != NULL;
            const char *d = a ? strstr(a, "dy=") : NULL;  // ",dy=<metres>": probe the world point that far above/below the emitter
            auto_dy = d ? (float)atof(d + 3) : 0;
            const char *nr = a ? strstr(a, "near=") : NULL;  // ",near=<m>": capture 1 = flame closer than that, capture 2 = far
            auto_near = nr ? (float)atof(nr + 5) : 0;
            const char *af = a ? strstr(a, "after=") : NULL;  // ",after=<present>": no capture before that present
            auto_after = af ? atol(af + 6) : 0;
        }
        int qx = px, qy = py > 15 ? py - 15 : 0;  // (default: above the torch head, where the flame burns)
        if (auto_dy) {
            float w[4] = {0, 0, 0, 1}, q[4];
            for (int i = 0; i < 3; i++) w[i] = m[i * 4] * e[19] + m[i * 4 + 1] * (e[23] + auto_dy) + m[i * 4 + 2] * e[27] + m[i * 4 + 3];
            for (int i = 0; i < 4; i++) q[i] = m[(4 + i) * 4] * w[0] + m[(4 + i) * 4 + 1] * w[1] + m[(4 + i) * 4 + 2] * w[2] + m[(4 + i) * 4 + 3];
            if (q[3] > 0) {
                qx = (int)(st.viewport[0][0] + (q[0] / q[3] * 0.5f + 0.5f) * st.viewport[0][2]);
                qy = (int)(st.viewport[0][1] + (0.5f - 0.5f * q[1] / q[3]) * st.viewport[0][3]);
            }
        }
        bool on_screen = c[3] > 0 && px >= 0 && py >= 0 && px < (int)st.viewport[0][2] && py < (int)st.viewport[0][3];
        {   // game-speed gauge: the effect clock (cb10[2][0]) against wall time, once per 60 presents
            static uint64_t last_p;
            uint64_t p = renderer_present_count();
            if (p / 60 != last_p / 60) {
                last_p = p;
                const float *cm = st.ubuf[0][5].size >= 30 * 16 ? m + 116 : NULL;  // camera position (VS cb8[29])
                LOG("[speed] present %llu wall %.3f s effect time %g cam %g %g %g\n", (unsigned long long)p, SDL_GetTicksNS() / 1e9,
                    e[8], cm ? cm[0] : 0.f, cm ? cm[1] : 0.f, cm ? cm[2] : 0.f);
            }
        }
        if (g_debug_frame) {
            LOG("[flame] emitter view %g %g %g clip %g %g %g %g -> pixel %d,%d dist %g%s\n", v[0], v[1], v[2], c[0], c[1], c[2], c[3], px,
                py, dist, on_screen ? "" : " (off screen)");
            flame_nodepth = auto_ab && captures == 2;
            if (flame_nodepth) LOG("[flame] capture 2: depth test off for this draw\n");
            if (st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {  // its colour target before the draw (after: below)
                char path[96];
                snprintf(path, sizeof path, "build/pass_flame_%llu_before.png", (unsigned long long)renderer_present_count());
                renderer_save_texture_tex((TexBackend *)st.rt.color[0]->backend, path);
            }
        } else if (auto_min > 0 && on_screen && (long)renderer_present_count() >= auto_after &&
                   (auto_near && !captures ? dist <= auto_near : dist >= auto_min) &&
                   captures < (auto_ab || auto_near ? 2 : 1) &&
                   (!captures || renderer_present_count() >= last_capture + 90)) {
            extern volatile int g_debug_frame_request;
            g_debug_frame_request = 1;
            flame_probe_x = qx; flame_probe_y = qy;
            last_capture = renderer_present_count();
            captures++;
            LOG("[flame] auto capture %d at present %llu: emitter %g m away at pixel %d,%d -> dumping + probing pixel %d,%d of the next frame\n",
                captures, (unsigned long long)last_capture, dist, px, py, qx, qy);
        }
    }
    if (g_debug_frame && g_cur_fs_hash == 0x85b366a0e972a9bcull && st.ubuf[0][7].addr) {  // TotK torch flame: VS cb10 (emitter
        const float *e = (const float *)(uintptr_t)st.ubuf[0][7].addr;                     // constants; cb10[3][0] = alpha)
        LOG("[flame] vs cb10 %llx +%llx:", (unsigned long long)st.ubuf[0][7].addr, (unsigned long long)st.ubuf[0][7].size);
        for (int i = 0; i < 32 && (uint64_t)i * 4 < st.ubuf[0][7].size; i++) LOG(" %g", e[i]);
        LOG("\n");
        if (st.ubuf[1][5].size >= 31 * 16 && st.ubuf[1][6].size >= 141 * 16) {  // FS soft-particle depth fade inputs
            const float *c8 = (const float *)(uintptr_t)st.ubuf[1][5].addr + 30 * 4, *c9 = (const float *)(uintptr_t)st.ubuf[1][6].addr;
            LOG("[flame] fs cb8[30] %g %g %g %g | cb9[139] %g %g %g %g | cb9[140] %g %g %g %g\n", c8[0], c8[1], c8[2], c8[3],
                c9[139 * 4], c9[139 * 4 + 1], c9[139 * 4 + 2], c9[139 * 4 + 3], c9[140 * 4], c9[140 * 4 + 1], c9[140 * 4 + 2], c9[140 * 4 + 3]);
        }
        if (st.ubuf[0][5].size >= 30 * 16) {  // VS cb8[28] (camera forward, used in the behind-camera test) and cb8[29] (camera pos)
            const float *c = (const float *)(uintptr_t)st.ubuf[0][5].addr;
            float dx = e[19] - c[29 * 4], dy = e[23] - c[29 * 4 + 1], dz = e[27] - c[29 * 4 + 2];
            LOG("[flame] vs cb8[28] %g %g %g %g | cb8[29] %g %g %g %g | emitter-camera %g %g %g, along cb8[28] %g\n", c[112], c[113],
                c[114], c[115], c[116], c[117], c[118], c[119], dx, dy, dz, dx * c[112] + dy * c[113] + dz * c[114]);
        }
        LOG("[flame] draw x0..x6 %llx %llx %llx %llx %llx %llx %llx\n", (unsigned long long)r->x[0], (unsigned long long)r->x[1],
            (unsigned long long)r->x[2], (unsigned long long)r->x[3], (unsigned long long)r->x[4], (unsigned long long)r->x[5],
            (unsigned long long)r->x[6]);
        for (int i = 0; i < 16; i++) {  // vertex attributes (a4.w / a5.w = particle lifetime / birth time): first 12 elements as floats
            if (!(pr->vi->attr_mask & (1u << i))) continue;
            uint32_t a = i < st.nattrib ? st.attrib[i] : 0, s = (a >> 8) & 0x1F, off = a >> 13;
            uint32_t stride = s < (uint32_t)st.nstream ? st.stream[s][0] : 0, div = s < (uint32_t)st.nstream ? st.stream[s][1] : 0;
            LOG("[flame] a%d fmt %#x stream %u off %u stride %u div %u vbuf %llx +%llx:", i, a & 0xFF, s, off, stride, div,
                (unsigned long long)st.vbuf[s & 15].addr, (unsigned long long)st.vbuf[s & 15].size);
            for (uint32_t k = 0; k < 12 && st.vbuf[s & 15].addr; k++) {
                uint64_t o = (uint64_t)k * stride + off;
                if (o + 16 > st.vbuf[s & 15].size) break;
                const float *f = (const float *)(uintptr_t)(st.vbuf[s & 15].addr + o);
                LOG(" (%g %g %g %g)", f[0], f[1], f[2], f[3]);
                if (!stride) break;
            }
            LOG("\n");
        }
    }
    if (g_debug_frame && g_cur_fs_hash == 0x18ef69743ba92eabull) {  // TotK point-light volume (torch light): FS cb3[10..15], cb4[0..7]
        for (int k = 0; k < 2; k++) {                                // (cb4[5] = light position, cb4[0].y = 1/r^2, cb4[3..7] = colour/params)
            const Range *ub = &st.ubuf[1][k];
            int from = k ? 0 : 10, to = k ? 8 : 16;
            LOG("[light] fs cb%d %llx +%llx:", k + 3, (unsigned long long)ub->addr, (unsigned long long)ub->size);
            for (int i = from; i < to && ub->addr && (uint64_t)(i + 1) * 16 <= ub->size; i++) {
                const float *f = (const float *)(uintptr_t)(ub->addr + i * 16);
                LOG(" [%d] %g %g %g %g", i, f[0], f[1], f[2], f[3]);
            }
            LOG("\n");
        }
    }
    if (g_debug_frame && g_cur_fs_hash == 0x3a703939f30c38beull) {  // TotK tile lighting: the per-tile words its VS reads
        static int dumped;   // (vertex storage buffer 4 + 0x41c0 + 16 * tile; 25 x 15 tiles of 64 px; all work so far flushed first)
        if (!dumped++ && st.sbuf[0][4].addr) {
            renderer_flush();
            // +0x1c0: the grid the tile-classification pass (fs 728ac122) reads (w1 == 1: tile processed) and writes (w0, w2, w3);
            // +0x41c0: the grid it writes for the lighting VS
            for (int g = 0; g < 2; g++) {
                const uint32_t *t = (const uint32_t *)(uintptr_t)(st.sbuf[0][4].addr + (g ? 0x41c0 : 0x1c0));
                LOG("[tiles] buffer %llx +%llx grid +%x, words w0 w1 w2 w3 per tile (row: 25 tiles of 64 px)\n",
                    (unsigned long long)st.sbuf[0][4].addr, (unsigned long long)st.sbuf[0][4].size, g ? 0x41c0 : 0x1c0);
                for (int y = 0; y < 15; y++) {
                    char line[25 * 5 + 1], *o = line;
                    for (int x = 0; x < 25; x++, o += 5) {
                        const uint32_t *w = t + (y * 25 + x) * 4;
                        snprintf(o, 6, "%x%x%x%x ", w[0] & 15, w[1] & 15, w[2] & 15, w[3] & 15);
                    }
                    LOG("[tiles] g%d y%02d %s\n", g, y, line);
                }
            }
        }
    }
    if (g_debug_frame) debug_draw(pr, r, indexed);
    else {   // LARECOMP_DRAWLOG_FS=<fs hash hex>: the debug-frame draw line for that shader's first 8 draws (2026-10-18 hang)
        static uint64_t lh = 1;
        static int ln;
        if (lh == 1) { const char *e = getenv("LARECOMP_DRAWLOG_FS"); lh = e ? strtoull(e, NULL, 16) : 0; }
        if (lh && g_cur_fs_hash == lh && ln < 8) {
            ln++;
            debug_draw(pr, r, indexed);
            const char *cd = getenv("LARECOMP_CBDUMP");
            char path[512];
            FILE *f;
            if (ln == 1 && cd && (snprintf(path, sizeof path, "%s/draw.txt", cd), f = fopen(path, "w"))) {
                fprintf(f, "indexed %d x", indexed);
                for (int i = 0; i < 8; i++) fprintf(f, " %llx", (unsigned long long)r->x[i]);
                fprintf(f, "\nindices");
                uint32_t isz = r->x[1] == 1 ? 2 : r->x[1] == 0 ? 1 : 4;   // (x[1]: 0 = u8, 1 = u16, 2 = u32)
                for (uint32_t i = 0; indexed && r->x[3] && i < (uint32_t)r->x[2] && i < 64; i++) {
                    const uint8_t *s = (const uint8_t *)(uintptr_t)r->x[3] + (size_t)i * isz;
                    fprintf(f, " %u", isz == 1 ? *s : isz == 2 ? *(const uint16_t *)s : *(const uint32_t *)s);
                }
                fprintf(f, "\n");
                fclose(f);
                cbdump_n = 1;   // (cleared once both stages' sets are built below)
            }
        }
    }
    uint64_t pz2 = prof ? prof_now() : 0;
    // A depth write after draws of this pass sampled that depth attachment live: end the pass + barrier first, so their
    // texture reads finish before the depth writes start. Without it the reads and writes overlap inside one render
    // pass instance (undefined; NVIDIA 617.14 hung the GPU in the decal pass, 2026-10-19 runs 479-492, CURRENT.md).
    if (depth_live_in_pass && in_pass && st.rt.depth && st.ds[0] && st.ds[1]) {
        vk_end_pass();
        VkCommandBuffer wcb = vk_cmd();
        gp_mark(wcb, "depth write after live sampling");
        vk_full_barrier(wcb);
        pass_barrier_needed = false;
        pass_barrier_fresh = true;
        st.rt_dirty = true;
        stat_depth_war++;
    }
    resolve_feedback(pr);
    uint64_t pz3 = prof ? prof_now() : 0;
    snap_reason = "draw-depthwrite";
    if (st.rt.depth && st.ds[0] && st.ds[1]) snapshots_invalidate(st.rt.depth);
    snap_reason = "draw-colorwrite";
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i] && ((st.channel_mask >> (i * 4)) & 0xF)) snapshots_invalidate(st.rt.color[i]);
    begin_pass();
    VkCommandBuffer cb = vk_cmd();
    uint64_t pz4 = prof ? prof_now() : 0;

    uint32_t mode = (uint32_t)r->x[0];
    VkPrimitiveTopology topo;
    uint8_t cls;
    bool native = topology(mode, &topo, &cls);
    if (!native && mode != 6 && mode != 7 && mode != 9) { stat_skipped++; stat_skip_why[3]++; static int nl; if (nl++ < 20) LOG("[skip] topology mode %u (fs hash %016llx, count %llu, rt0 %ux%u f%#x)\n", mode, (unsigned long long)g_cur_fs_hash, (unsigned long long)r->x[2], st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->width : 0, st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->height : 0, st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->format : 0); return; }

    uint64_t prof_tk_ = prof ? prof_now() : 0;
    if (prof) {
        prof_pre += prof_tk_ - prof_tp_;
        prof_pz[0] += pz1 - prof_tp_; prof_pz[1] += pz2 - pz1; prof_pz[2] += pz3 - pz2; prof_pz[3] += pz4 - pz3; prof_pz[4] += prof_tk_ - pz4;
        static int nslow;
        if (prof_tk_ - prof_tp_ > 20000000ull && nslow++ < 300)
            LOG("[slowdraw] frame %llu: pre %.1f ms = program %.1f (new: key %.1f claim %.1f record %.1f translate %.1f) + fs/debug %.1f "
                "+ feedback %.1f + snapshots/begin_pass %.1f + topology %.1f; vs %llx fs %llx, textures %d+%d, rt0 %ux%u f%#x\n",
                (unsigned long long)renderer_present_count(), (prof_tk_ - prof_tp_) / 1e6, (pz1 - prof_tp_) / 1e6, gps_ns[0] / 1e6,
                gps_ns[1] / 1e6, gps_ns[2] / 1e6, gps_ns[3] / 1e6, (pz2 - pz1) / 1e6, (pz3 - pz2) / 1e6, (pz4 - pz3) / 1e6,
                (prof_tk_ - pz4) / 1e6, (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
                pr->vi->ntex, pr->fi->ntex, st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->width : 0,
                st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->height : 0, st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->format : 0);
    }
    // ---- pipeline
    PipeKey k;
    memset(&k, 0, sizeof k);
    k.prog = pr;
    k.ncolor = pass_ncolor;
    memcpy(k.color, pass_color, sizeof k.color);
    pipekey_norm(&k);
    k.depth = pass_depth;
    k.stencil = pass_stencil;
    k.samples = pass_samples;
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        k.attr_mask |= 1u << i;
        uint32_t a = i < st.nattrib ? st.attrib[i] : 0x2E;
        uint32_t fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
        if (i >= st.nattrib || fmt == 0 || stream >= 16) {  // unbound attribute: (0, 0, 0, 1) from a constant stream
            k.attr_fmt[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
            k.attr_bind[i] = ZERO_BINDING;
            k.used_streams |= 1u << ZERO_BINDING;
            continue;
        }
        k.attr_fmt[i] = vertex_format_supported(vertex_format(fmt));
        k.attr_off[i] = off;
        k.attr_bind[i] = stream;
        k.used_streams |= 1u << stream;
    }
    for (int s = 0; s < 16; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        k.stride[s] = s < st.nstream ? st.stream[s][0] : 0;
        k.divisor[s] = s < st.nstream && k.stride[s] ? st.stream[s][1] : 0;
        if (k.divisor[s] > 1 && !g_vk.divisor) k.divisor[s] = 1;
    }
    for (uint32_t i = 0; i < k.ncolor && i < 8; i++) {
        if (!st.rt.color[i] || k.color[i] == VK_FORMAT_UNDEFINED) continue;
        // Targets the fragment shader doesn't output keep their contents on Metal, but are undefined in Vulkan
        // (NVIDIA writes black): mask them. Foliage depth pre-passes left black silhouettes (user 2026-10-08).
        k.write_mask[i] = ((pr->fi->out_mask >> (i * 4)) & 0xF) ? (st.channel_mask >> (i * 4)) & 0xF : 0;
        if (((st.color_state[0] >> i) & 1) && !rt_kind(rt_view_format(i))) {
            k.blend_on |= (uint8_t)(1u << i);
            memcpy(k.blend[i], st.blend[i], 8);
        }
    }
    k.a2c = st.ms[0xC] != 0;
    k.topo_class = cls;
    k.poly_fill = st.poly[2] != 1;
    k.depth_clamp = st.depth_clamp;
    static struct { PipeKey k; Pipe *pp; uint32_t gen; } pk_memo;   // (k is memset first: padding compares equal)
    Pipe *pp;
    if (pk_memo.pp && pk_memo.gen == g_memo_gen && !memcmp(&k, &pk_memo.k, sizeof k)) pp = pk_memo.pp;
    else {
    uint64_t ph = hash_words(0xcbf29ce484222325ull, &k, sizeof k);
    if (!ph) ph = 1;
    uint32_t pslot = (uint32_t)(ph ^ (ph >> 29)) & (PIPE_TAB - 1);
    while (pipe_tab[pslot].key && pipe_tab[pslot].key != ph) pslot = (pslot + 1) & (PIPE_TAB - 1);
    pp = pipe_tab[pslot].pipe;
    if (!pp) {
        pp = pc_take_pipe(ph);   // built (or being built) by the shader precompile
        bool fresh = !pp;
        if (fresh) { pp = calloc(1, sizeof *pp); pp->k = k; }
        pipe_tab[pslot].key = ph;
        pipe_tab[pslot].pipe = pp;
        if (fresh && gpl_linkable(&k)) { pp->claimed = 1; build_pipeline_body(pp); }   // fast link here: no skip
        else if (fresh) run_job(build_pipeline, pp);
        if (pr->ckey) pc_record_pipe(pr->ckey, &k);
    }
    pk_memo.k = k; pk_memo.pp = pp; pk_memo.gen = g_memo_gen;
    }
    // A draw whose pipeline is still being built is skipped; device.c then does not show that frame (a skipped final pass
    // made whole frames black: run199). LARECOMP_PIPE_INLINE=1: the draw builds the pipeline itself or waits for the
    // worker (<= 100 ms) instead; that stalled 1.9-2.7 s on bursts of new pipelines (run201 / run202, ~50 ms each even
    // with a warm pipeline cache). "[vk] pipeline waits" in the stats line.
    static int pipe_inline = -1;
    if (pipe_inline < 0) pipe_inline = getenv("LARECOMP_PIPE_INLINE") != NULL;
    if (atomic_load(&pp->state) == 0 && (pipe_inline || g_draw_bake)) {
        uint64_t t0 = SDL_GetTicksNS();
        int unclaimed = 0;
        if (atomic_compare_exchange_strong(&pp->claimed, &unclaimed, 1)) {
            build_pipeline_body(pp);
            prof_pipe_inline++;
        } else {
            uint64_t limit = g_draw_bake ? 3000000000ull : 100000000ull;
            while (atomic_load(&pp->state) == 0 && SDL_GetTicksNS() - t0 < limit) SDL_Delay(1);
            prof_pipe_waited++;
        }
        prof_pipe_wait_ns += SDL_GetTicksNS() - t0;
    }
    // (g_skip_full_frame: the skipped draw has <= 6 vertices = a fullscreen / tile pass; device.c always holds those)
    // "[pipeskip]" (first 300): why a skipped pipeline was not fast-linked, then how long its draws were missing. Frames with
    // <= LARECOMP_HOLD_MIN such skips are shown (user's free play 2026-10-20 07:31: 39 shown frames, "buildings disappear").
    static int ps_log;
    if (atomic_load(&pp->state) != 1) {
        stat_skipped++; stat_skip_why[4]++;
        if (atomic_load(&pp->state) == 0) {
            g_skip_pipe_frame++; g_skip_full_frame += r->x[2] <= 6;
            if (!pp->nskip++ && ps_log < 300) {
                ps_log++;
                pp->skip_t0 = SDL_GetTicksNS();
                const Prog *pg = pp->k.prog;
                LOG("[pipeskip] frame %llu fs %llx: libs %d, depth clamp %d, line fill %d, samples %d, a2c %d, verts %llu\n",
                    (unsigned long long)renderer_present_count(), pg ? (unsigned long long)pg->code[1] : 0ull,
                    pg ? atomic_load(&pg->lib_state) : -1, pp->k.depth_clamp, !pp->k.poly_fill, (int)pp->k.samples, pp->k.a2c,
                    (unsigned long long)r->x[2]);
            }
        }
        return;
    }
    if (pp->skip_t0) {
        LOG("[pipeskip] frame %llu fs %llx ready after %.1f ms, %u draws skipped\n", (unsigned long long)renderer_present_count(),
            pp->k.prog ? (unsigned long long)pp->k.prog->code[1] : 0ull, (SDL_GetTicksNS() - pp->skip_t0) / 1e6, pp->nskip);
        pp->skip_t0 = 0;
    }
    // Same pipeline as the one bound last in this command buffer: no rebind (run363: vkCmdBindPipeline here was ~16 %
    // of exec in the NVIDIA driver; every draw rebound). Tracked with the dynamic-state cache (dyn_begin keys it).
    dyn_begin(cb);
    if (dyn.pipe != pp->pipe) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp->pipe);
        dyn.pipe = pp->pipe;
        if (!pp->linked) dyn.clamp = -1;   // (a static clamp leaves the dynamic value undefined)
        prof_pipe_bind++;
    } else prof_pipe_skip++;
    uint64_t prof_td_ = prof ? prof_now() : 0;
    if (prof) prof_key += prof_td_ - prof_tk_;

    // ---- depth / stencil (dynamic)
    static int nodt = -1;
    if (nodt < 0) nodt = getenv("LARECOMP_NO_DEPTH_TEST") != NULL;
    static uint64_t nodt_hash = 1;   // LARECOMP_NO_DEPTH_FS=<fs hash>: depth test off for that fragment shader only (A/B)
    if (nodt_hash == 1) { const char *e = getenv("LARECOMP_NO_DEPTH_FS"); nodt_hash = e ? strtoull(e, NULL, 16) : 0; }
    bool dt = st.ds[0] && !nodt && pass_depth && !flame_nodepth && !(nodt_hash && g_cur_fs_hash == nodt_hash);
    bool dv = dyn_begin(cb);
#define DYN_SET(cond, assign, call) do { if (!dv || (cond)) { assign; call; prof_dyn_set++; } else prof_dyn_skip++; } while (0)
    bool dw = dt && st.ds[1];
    DYN_SET(dyn.dt != dt, dyn.dt = dt, vkCmdSetDepthTestEnable(cb, dt));
    DYN_SET(dyn.dw != dw, dyn.dw = dw, vkCmdSetDepthWriteEnable(cb, dw));
    if (vk_dyn_clamp && pp->linked) {   // (same signature as the feedback-loop setter: queued in order through q_set_fbl)
        static PFN_vkCmdSetDepthClampEnableEXT set_clamp;
        if (!set_clamp) set_clamp = (PFN_vkCmdSetDepthClampEnableEXT)vkGetDeviceProcAddr(g_vk.dev, "vkCmdSetDepthClampEnableEXT");
        int8_t c = pp->k.depth_clamp;
        DYN_SET(dyn.clamp != c, dyn.clamp = c, q_set_fbl((PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT)set_clamp, cb, c));
    }
    if (g_vk.feedback_loop) {   // (declares this draw's live attachment sampling: device.c / begin_pass fb_loop_info)
        static PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT set_fbl;
        if (!set_fbl) set_fbl = (PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT)vkGetDeviceProcAddr(g_vk.dev, "vkCmdSetAttachmentFeedbackLoopEnableEXT");
        DYN_SET(dyn.fbl != fb_live_aspects, dyn.fbl = fb_live_aspects, q_set_fbl(set_fbl, cb, fb_live_aspects));
    }
    VkCompareOp dop = dt ? compare_fn(st.ds[2]) : VK_COMPARE_OP_ALWAYS;
    // EQUAL against a depth pre-pass written by a different vertex shader (foliage, trees) needs both shaders to land
    // on identical depths. They do since the translator stopped using unpackHalf2x16 (NVIDIA ran the half math after
    // it in fp16, differently per shader: black slashes in foliage). LARECOMP_VK_EQUAL_TOL=1 brings back the old
    // workaround: LESS_OR_EQUAL with a few ulps of bias towards the camera, which also let walls paint over the
    // plants stuck flat on them (beach) - so it is off.
    static int tol = -1;
    if (tol < 0) tol = getenv("LARECOMP_VK_EQUAL_TOL") != NULL;
    bool equal_tol = dop == VK_COMPARE_OP_EQUAL && tol;
    if (equal_tol) dop = VK_COMPARE_OP_LESS_OR_EQUAL;
    DYN_SET(dyn.dop != dop, dyn.dop = dop, vkCmdSetDepthCompareOp(cb, dop));
    probe_dt = dt; probe_dw = dw; probe_dop = dop;
    bool sten = st.ds[3] && pass_stencil;
    DYN_SET(dyn.sten != sten, dyn.sten = sten, vkCmdSetStencilTestEnable(cb, sten));
    if (sten) {
        uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;
        bool sv = dv && dyn.sten_valid;
        for (int f = 0; f < 2; f++) {
            uint32_t o = f ? ops >> 12 : ops;
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            uint32_t sop = (o & 0xFFF) | (uint32_t)(f ? st.ds[4] >> 4 : st.ds[4] & 0xF) << 12;
            uint32_t cm = st.stencil_vmask[f] & 0xFF, wm = st.stencil_mask[f] & 0xFF, ref = st.stencil_ref[f];
            if (!sv || dyn.sop[f] != sop) {
                vkCmdSetStencilOp(cb, face, stencil_op(o & 0xF), stencil_op((o >> 8) & 0xF), stencil_op((o >> 4) & 0xF),
                                  compare_fn(f ? st.ds[4] >> 4 : st.ds[4] & 0xF));
                dyn.sop[f] = sop;
            }
            if (!sv || dyn.cmask[f] != cm) { vkCmdSetStencilCompareMask(cb, face, cm); dyn.cmask[f] = cm; }
            if (!sv || dyn.wmask[f] != wm) { vkCmdSetStencilWriteMask(cb, face, wm); dyn.wmask[f] = wm; }
            if (!sv || dyn.ref[f] != ref) { vkCmdSetStencilReference(cb, face, ref); dyn.ref[f] = ref; }
        }
        dyn.sten_valid = true;
    }
    DYN_SET(memcmp(dyn.blend, st.blend_color, sizeof dyn.blend), memcpy(dyn.blend, st.blend_color, sizeof dyn.blend),
            vkCmdSetBlendConstants(cb, st.blend_color));

    // ---- rasterizer
    uint8_t cull = st.poly[0];
    VkCullModeFlags cm = cull == 1 ? VK_CULL_MODE_FRONT_BIT : cull == 2 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    DYN_SET(dyn.cull != cm, dyn.cull = cm, vkCmdSetCullMode(cb, cm));
    bool ccw = st.poly[1] == 1;
    static int flipw = -1;  // LARECOMP_VK_FLIP_WINDING: invert the front-face mapping (A/B against Metal)
    if (flipw < 0) flipw = getenv("LARECOMP_VK_FLIP_WINDING") != NULL;
    bool front_ccw = (ccw != vkey.flip_y) != (flipw != 0);
    VkFrontFace ff = front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    DYN_SET(dyn.front != ff, dyn.front = ff, vkCmdSetFrontFace(cb, ff));
    VkPrimitiveTopology tp = native ? topo : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    DYN_SET(dyn.topo != tp, dyn.topo = tp, vkCmdSetPrimitiveTopology(cb, tp));
    bool restart = st.prim_restart && native && (mode == 3 || mode == 5);
    DYN_SET(dyn.restart != restart, dyn.restart = restart, vkCmdSetPrimitiveRestartEnable(cb, restart));
    {
        float f[3] = {0, 0, 0};
        static int off = -1;
        if (off < 0) off = getenv("LARECOMP_NO_DEPTH_BIAS") != NULL;
        bool on = (st.poly[3] & 4) && !off;
        if (on) memcpy(f, st.poly_offset, sizeof f);
        if (equal_tol) { f[1] -= 8; on = true; }  // (see the EQUAL comment above: a few depth ulps nearer)
        DYN_SET(dyn.bias_on != on, dyn.bias_on = on, vkCmdSetDepthBiasEnable(cb, on));
        DYN_SET(memcmp(dyn.bias, f, sizeof f), memcpy(dyn.bias, f, sizeof f), vkCmdSetDepthBias(cb, f[1], f[2], f[0]));
    }
#undef DYN_SET
    dyn_end();

    // ---- textures first (their scales feed the widescreen decision and the rscale table), then the viewport
    float *vp = st.viewport[0];
    float zn = st.depth_range[0][0], zf = st.depth_range[0][1];
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    {   // the Items page's cursor brackets while the cursor is on the companion card (runtime/hle/pets.c)
        extern bool pets_hide_item_cursor(void);
        extern bool texpatch_is_item_cursor(int id);
        if (pets_hide_item_cursor())
            for (int i = 0; i < pr->fi->ntex; i++) {
                NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
                const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
                if (b && b->tp > 0 && texpatch_is_item_cursor(b->tp - 1)) { stat_skipped++; stat_skip_why[5]++; return; }
            }
    }
    float tint[8] = {0};
    if (pr->fi->uses_tint) {
        extern bool pot_tint(const float *m, float tint[8], uint64_t mesh);
        extern bool pot_glow_active(void);
        if (st.ubuf[0][2].addr && st.ubuf[0][2].size >= 48 && pot_glow_active())
            pot_tint((const float *)(uintptr_t)st.ubuf[0][2].addr, tint, st.vbuf[0].addr);
        // the gold Escape Bottle (runtime/hle/escape.c): draws sampling the bottle's item icon / model map
        // and the HUD hearts (heart colour, Cosmetics tab)
        extern int texpatch_bottle_kind(int id);
        extern int texpatch_heart_kind(int id);
        extern void texpatch_heart_tint(int kind, float tint[8]);
        int bottle = 0, heart = 0;
        for (int i = 0; i < pr->fi->ntex; i++) {
            NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
            const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
            if (b && b->tp > 0 && !bottle) bottle = texpatch_bottle_kind(b->tp - 1);
            if (b && b->tp > 0 && !heart) heart = texpatch_heart_kind(b->tp - 1);
        }
        if (heart) texpatch_heart_tint(heart, tint);
        if (bottle) {
            extern void escape_bottle_tint(int kind, const float *ui, uint32_t ui_size, float tint[8]);
            int u = bottle == 3 ? 3 : 0;  // (the fairy model: its bone matrices, for where it stands - souls.c)
            escape_bottle_tint(bottle, (const float *)(uintptr_t)st.ubuf[0][u].addr, (uint32_t)st.ubuf[0][u].size, tint);
        }
    }
    // fragment textures are bound before deciding the viewport (tex_sc / tex_wh), rscale depends on the decision:
    // build the vertex set now and the fragment set after the decision (stage_set fills tex_sc for stage 1).
    if (prof) prof_dyn += prof_now() - prof_td_;
    PROF_T0();
    uint32_t dyn[2 * MAX_BIND], ndyn = 0;
    VkDescriptorSet vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL, dyn, &ndyn);
    PROF_ADD(prof_sets);
    VkDescriptorImageInfo fii[64];
    uint64_t prof_tx_ = prof ? prof_now() : 0;
    pk_draw(pr);
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
        float sc[2]; uint32_t wh[2];
        texture_info(1, pr->fi, i, &fii[i], sc, wh);
        tex_sc[1][i][0] = sc[0]; tex_sc[1][i][1] = sc[1]; tex_wh[1][i][0] = wh[0]; tex_wh[1][i][1] = wh[1];
    }
    pk_subst_tex = NULL;
    if (prof) prof_tex += prof_now() - prof_tx_;
    if (fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && renderer_texture_presented(st.rt.color[0])) {
        // Draws into the swapchain image (see renderer.m): composites fill the window, UI goes in a centred 16:9 box.
        bool reads_scene = false;
        for (int i = 0; i < pr->fi->ntex && i < 64; i++)
            if ((tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) && tex_wh[1][i][0] == st.rt.color[0]->width &&
                tex_wh[1][i][1] == st.rt.color[0]->height)
                reads_scene = true;
        bool fullscreen_quad = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
                               st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height;
        bool fullscreen_fill = pr->fi->ntex == 0 && fullscreen_quad;
        if (fullscreen_quad && !reads_scene && pr->fi->ntex == 1 && (tex_sc[1][0][0] != 1 || tex_sc[1][0][1] != 1)) reads_scene = true;
        static uint64_t wide_frame = ~0ull;
        if (reads_scene) wide_frame = renderer_present_count();
        bool fade_fill = false;
        if (fullscreen_fill && st.ubuf[1][0].addr && st.ubuf[1][0].size >= 12) {
            const float *c = (const float *)(uintptr_t)st.ubuf[1][0].addr;
            bool black = c[0] < 0.01f && c[1] < 0.01f && c[2] < 0.01f, white = c[0] > 0.99f && c[1] > 0.99f && c[2] > 0.99f;
            fade_fill = black || white;
        }
        bool composite = reads_scene || fade_fill || (fullscreen_fill && wide_frame == renderer_present_count());
        if (!composite) wide_frame = ~0ull;
        if (!composite) {
            int pw, ph;
            pass_size(&pw, &ph);
            float u = fminf(pass_sx, pass_sy);
            vp_sx = vp_sy = u;
            vp_ox = (pw - st.rt.color[0]->width * u) * 0.5f;
            vp_oy = (ph - st.rt.color[0]->height * u) * 0.5f;
        }
    }
    {   // camera-attached effects sized for 16:9 are widened with the view (runtime/hle/widescreen.c)
        extern float widescreen_effect_stretch(uint64_t fs_hash);
        float k = widescreen_effect_stretch(g_cur_fs_hash);
        float vx = vp[0] * vp_sx + vp_ox, vw = vp[2] * vp_sx;
        if (k > 1.0f) { vx -= vw * (k - 1) / 2; vw *= k; }
        set_viewport(vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, zn, zf);
        if (g_debug_frame) {
            VkRect2D sr = scissor_rect();
            LOG("[draw]   vp %.2f %.2f %.2f %.2f (stretch %.3f, fs hash %016llx) scissor %d %d %u %u | guest sc %d %d %d %d | pass %ux%u s %.4f %.4f\n",
                vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, k, (unsigned long long)g_cur_fs_hash, sr.offset.x, sr.offset.y,
                sr.extent.width, sr.extent.height, st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], pass_w, pass_h,
                vp_sx, vp_sy);
        }
    }
    apply_scissor();
    float rs0[4] = {vp_sx, vp_sy, vp_ox, vp_oy};
    uint64_t prof_t1_ = prof ? prof_now() : 0;
    uint32_t ndyn_v = ndyn;
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, fii, dyn, &ndyn);
    cbdump_n = 0;
    if (prof) prof_sets += prof_now() - prof_t1_;
    VkDescriptorSet sets[2] = {vset, fset};
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 0, 2, sets, ndyn, dyn);

    uint64_t prof_t2_ = prof ? prof_now() : 0;
    // ---- indices (copied; the largest index bounds the vertex data copied below)
    uint32_t first = 0, count = (uint32_t)r->x[2], inst, base_inst, max_index = 0;
    int32_t base_vertex = 0;
    VkSpan ib = {0};
    VkIndexType itype_vk = VK_INDEX_TYPE_UINT32;
    uint32_t icount = 0;
    bool use_index = indexed || !native;
    const void *scan_src = NULL;  // deferred index scan (direct index buffer)
    uint32_t scan_isz = 0, scan_restart = 0;
    vk_up_kind = 3;
    if (!indexed) {
        first = (uint32_t)r->x[1]; base_inst = (uint32_t)r->x[3]; inst = (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 1;
        max_index = first + count;
        if (!native) {
            uint32_t *e = expand_indices(mode, count, NULL, 0, first, &icount);
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
    } else {
        uint32_t itype = (uint32_t)r->x[1];
        uint64_t iaddr = r->x[3];
        base_vertex = (int32_t)r->x[4];
        base_inst = (uint32_t)r->x[5];
        inst = (uint32_t)r->x[6] ? (uint32_t)r->x[6] : 1;
        const void *src = (const void *)(uintptr_t)iaddr;
        uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu;
        if (native && itype != 0) {
            uint32_t isz = itype == 1 ? 2 : 4;
            itype_vk = itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            icount = count;
            if (iaddr % isz == 0 && guest_direct(iaddr, (uint64_t)count * isz, &ib.buf, &ib.off)) {
                scan_src = src; scan_isz = isz; scan_restart = restart;  // max_index only if a stream gets copied
            } else {
                ib = vk_transient_copy(src, (uint64_t)count * isz + 4, 4);
                if (isz == 2) { const uint16_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != (uint16_t)restart) max_index = s[i]; }
                else { const uint32_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != restart) max_index = s[i]; }
            }
        } else if (native) {  // 8-bit indices: widen to 16
            ib = vk_transient((uint64_t)count * 2 + 4, 4);
            uint16_t *w = ib.ptr;
            const uint8_t *s = src;
            for (uint32_t i = 0; i < count; i++) { w[i] = s[i] == 0xFF && st.prim_restart ? 0xFFFF : s[i]; if (s[i] > max_index && w[i] != 0xFFFF) max_index = s[i]; }
            itype_vk = VK_INDEX_TYPE_UINT16;
            icount = count;
        } else {
            uint32_t *e = expand_indices(mode, count, src, (int)itype, 0, &icount);
            for (uint32_t i = 0; i < icount; i++) if (e[i] > max_index) max_index = e[i];
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
        if (!scan_src) {
            max_index += 1;
            if (base_vertex > 0) max_index += (uint32_t)base_vertex;
        }
    }
    vk_up_kind = 0;
    if (use_index) bind_index(cb, ib.buf, ib.off, itype_vk);

    // ---- vertex streams: the range this draw can read, copied once per submission
    for (int s = 0; s < 17; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        VkSpan sp;
        if (s == ZERO_BINDING) {
            static const float zero[4] = {0, 0, 0, 1};
            sp = vk_transient_copy(zero, 16, 16);
        } else {
            Range v = st.vbuf[s];
            VkBuffer dbuf;
            VkDeviceSize doff;
            if (v.addr && v.size && guest_direct(v.addr, v.size, &dbuf, &doff)) {
                bind_vertex(cb, (uint32_t)s, dbuf, doff);
                continue;
            }
            if (scan_src) {  // copying after all: the index scan bounds the copy
                if (scan_isz == 2) { const uint16_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != (uint16_t)scan_restart) max_index = q[i]; }
                else { const uint32_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != scan_restart) max_index = q[i]; }
                max_index += 1;
                if (base_vertex > 0) max_index += (uint32_t)base_vertex;
                scan_src = NULL;
            }
            uint64_t stride = k.stride[s], need;
            if (!stride) need = 256;
            else if (k.divisor[s]) need = ((uint64_t)base_inst + inst + k.divisor[s] - 1) / k.divisor[s] * stride + 256;
            else need = (uint64_t)max_index * stride + 256;
            if (v.size && need > v.size) need = v.size;
            if (prof && v.addr) {  // why the direct (zero-copy) bind was refused: below the heap / past the import / window
                extern uint64_t prof_vx_copy_n, prof_vx_copy_bytes, prof_vx_low_bytes, prof_vx_low_region;
                prof_vx_copy_n++; prof_vx_copy_bytes += need;
                if (v.addr < HEAP_BASE) { prof_vx_low_bytes += need; prof_vx_low_region = v.addr >> 28; }
            }
            vk_up_kind = 2;
            if (!v.addr) { static const uint8_t zero[256]; sp = vk_transient_copy(zero, 256, 16); }
            else sp = guest_copy(v.addr, need ? need : 16);
            vk_up_kind = 0;
        }
        VkDeviceSize off = sp.off;
        bind_vertex(cb, (uint32_t)s, sp.buf, off);
    }

    if (prof) prof_vtx += prof_now() - prof_t2_;
    // ---- draw
    // Instanced draws with item spots among their instances (bushsanity: bushes are drawn in batches, each instance's
    // model matrix in vertex storage buffer 0, indexed by instance id): drawn as runs of instances, gold runs tinted
    // (as renderer.m; gl_InstanceIndex includes the base instance, so a run [b + k, b + k + n) draws exactly those).
    enum { MAX_RUNS = 64 };
    uint32_t run_at[MAX_RUNS], run_n[MAX_RUNS];
    bool run_gold[MAX_RUNS];
    int nruns = 0;
    if (pr->fi->uses_tint && inst > 1 && st.sbuf[0][0].addr) {
        extern bool pot_glow_active(void);
        extern bool pot_instance_gold(const float *m);
        uint64_t recs = st.sbuf[0][0].size / 64;
        if (pot_glow_active() && (uint64_t)base_inst + inst <= recs) {
            const float *m = (const float *)(uintptr_t)st.sbuf[0][0].addr;
            for (uint32_t i = 0; i < inst; i++) {
                bool g = pot_instance_gold(m + (size_t)(base_inst + i) * 16);
                if (nruns && run_gold[nruns - 1] == g) { run_n[nruns - 1]++; continue; }
                if (nruns == MAX_RUNS) { nruns = 0; break; }  // (too fragmented: draw as is)
                run_at[nruns] = i; run_n[nruns] = 1; run_gold[nruns] = g; nruns++;
            }
            if (nruns == 1 && !run_gold[0]) nruns = 0;
        }
    }
    if (!nruns) { run_at[0] = 0; run_n[0] = inst; nruns = 1; run_gold[0] = false; }
    // more than 2^28 vertices in one draw is a garbage count, not TotK: skipped instead of hanging the GPU
    bool huge = (uint64_t)(use_index ? icount : count) * inst > (1ull << 28);
    if (huge) {
        static int nlog;
        if (nlog++ < 20) LOG("[vk] draw skipped: %u vertices x %u instances (fs %016llx)\n", use_index ? icount : count, inst,
                             (unsigned long long)g_cur_fs_hash);
        nruns = 0;
    }
    hang_op(cb, HOP_DRAW, g_cur_fs_hash, st.program->stages[0].code_addr, use_index ? icount : count, inst, fb_bits);
    if (g_gpuprof_frame && gp_per_draw())   // (LARECOMP_GPUPROF_DRAWS=1: a timestamp before every draw; see vk_gpuprof_report)
        gp_mark(cb, "draw fs %016llx n %u inst %u", (unsigned long long)g_cur_fs_hash, use_index ? icount : count, inst);
    for (int ri = 0; ri < nruns; ri++) {
        if (nruns > 1 || run_gold[0]) {  // this run's tint: a fragment set of its own
            float rtint[8] = {0};
            extern void pot_gold_tint(float tint[8]);
            if (run_gold[ri]) pot_gold_tint(rtint);
            uint32_t rdyn[MAX_BIND], nrdyn = 0;
            VkDescriptorSet rset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, rtint, 0, 0, 0, fii, rdyn, &nrdyn);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 1, 1, &rset, nrdyn, rdyn);
        }
        if (!use_index) vkCmdDraw(cb, count, run_n[ri], first, base_inst + run_at[ri]);
        else vkCmdDrawIndexed(cb, icount, run_n[ri], 0, base_vertex, base_inst + run_at[ri]);
    }
    stat_draws++; g_draws_frame++; chunk_draws++;
    {
        static uint64_t cnt_hash = 1;
        if (cnt_hash == 1) { const char *e = getenv("LARECOMP_COUNT_FSHASH"); cnt_hash = e ? strtoull(e, NULL, 16) : 0; }
        if (cnt_hash && g_cur_fs_hash == cnt_hash) {
            stat_count_fs++;
            // LARECOMP_COUNT_TRIGGER=gpuprof|debug: the first frame (from present 1500 on) that draws it 3+ times gets the next
            // frame GPU-profiled (+ a slow_N.png shot) / debug-dumped once (TotK's weather decides when the clouds are up)
            static uint64_t cf_frame, cf_n;
            static int trig = -1, fired;
            if (trig < 0) { const char *e = getenv("LARECOMP_COUNT_TRIGGER"); trig = !e ? 0 : !strcmp(e, "gpuprof") ? 1 : !strcmp(e, "debug") ? 2 : 0; }
            uint64_t p = renderer_present_count();
            // LARECOMP_COUNT_LOG=<file>: per frame "frame draws inst_sum vertex_sum" of that shader (game-side input check)
            static FILE *cl; static int cl_init; static uint64_t cl_inst, cl_cnt;
            if (!cl_init) { cl_init = 1; const char *e = getenv("LARECOMP_COUNT_LOG"); if (e) cl = fopen(e, "w"); }
            // 5th column "late": draws whose VS storage buffer #0 bytes changed between record time and this shader's first draw of
            // the next frame (the game double-buffers the UI ring: a change within one frame = written after it submitted)
            extern uint64_t g_op_push_ns;
            static uint64_t lw_a[32], lw_s[32], lw_t[32], lw_lag[32], cl_maxlag; static uint8_t lw_d[32][512]; static int lw_n, lw_log;
            static uint64_t late_addr = 1;   // LARECOMP_LATE_ADDR=<hex>: log every late write of the draw at that address
            if (late_addr == 1) { const char *e = getenv("LARECOMP_LATE_ADDR"); late_addr = e ? strtoull(e, NULL, 16) : 0; }
            if (p != cf_frame && cl && cf_n) {
                int late = 0;
                for (int i = 0; i < lw_n; i++)
                    if (memcmp(lw_d[i], (const void *)(uintptr_t)lw_a[i], lw_s[i])) {
                        late++;
                        if (lw_log < 40 || lw_a[i] == late_addr) {
                            lw_log++;
                            const uint32_t *o = (const uint32_t *)lw_d[i], *nw = (const uint32_t *)(uintptr_t)lw_a[i];
                            int w = 0, nd = 0;
                            while (w < (int)(lw_s[i] / 4) - 1 && o[w] == nw[w]) w++;
                            for (int j = 0; j < (int)(lw_s[i] / 4); j++) nd += o[j] != nw[j];
                            LOG("[late] frame %llu draw %d %llx +%llu: word %d %08x -> %08x (%d words differ) snap t %llu check t %llu queue lag %llu us\n",
                                (unsigned long long)cf_frame, i, (unsigned long long)lw_a[i], (unsigned long long)lw_s[i], w, o[w], nw[w], nd,
                                (unsigned long long)lw_t[i], (unsigned long long)(SDL_GetTicksNS() / 1000), (unsigned long long)lw_lag[i]);
                        }
                    }
                fprintf(cl, "%llu %llu %llu %llu %d %llu\n", (unsigned long long)cf_frame, (unsigned long long)cf_n,
                        (unsigned long long)cl_inst, (unsigned long long)cl_cnt, late, (unsigned long long)cl_maxlag);
            }
            if (p != cf_frame) { cf_frame = p; cf_n = 0; cl_inst = cl_cnt = 0; lw_n = 0; cl_maxlag = 0; }
            if (cl && lw_n < 32 && st.sbuf[0][0].addr) {
                lw_a[lw_n] = st.sbuf[0][0].addr;
                lw_s[lw_n] = st.sbuf[0][0].size < 512 ? st.sbuf[0][0].size : 512;
                memcpy(lw_d[lw_n], (const void *)(uintptr_t)lw_a[lw_n], lw_s[lw_n]);
                lw_t[lw_n] = SDL_GetTicksNS() / 1000;
                lw_lag[lw_n] = g_op_push_ns ? lw_t[lw_n] - g_op_push_ns / 1000 : 0;
                if (lw_lag[lw_n] > cl_maxlag) cl_maxlag = lw_lag[lw_n];
                lw_n++;
            }
            cl_inst += inst; cl_cnt += use_index ? icount : count;
            // LARECOMP_COUNT_DUMP=<file>: per draw u64 frame, u32 inst, u32 bound-in-host-memory, u64 sbuf0 addr, then the first
            // 512 bytes of VS storage buffer #0 as host memory holds them now (record time)
            // (kept in memory and written once at present LARECOMP_COUNT_DUMP_AT, default 11000: file writes per draw changed the timing)
            static FILE *cd; static int cd_init; static uint8_t *cd_buf; static size_t cd_len; static uint64_t cd_at;
            if (!cd_init) {
                cd_init = 1;
                const char *e = getenv("LARECOMP_COUNT_DUMP"), *at = getenv("LARECOMP_COUNT_DUMP_AT");
                if (e && (cd_buf = malloc(64u << 20))) cd = fopen(e, "wb");
                cd_at = at ? strtoull(at, NULL, 10) : 11000;
            }
            if (cd && st.sbuf[0][0].addr) {
                struct { uint64_t f; uint32_t inst, host; uint64_t a; } h = {p, inst, g_sb0_bound == st.sbuf[0][0].addr, st.sbuf[0][0].addr};
                if (cd_len + sizeof h + 512 <= (64u << 20) && p < cd_at) {
                    memcpy(cd_buf + cd_len, &h, sizeof h);
                    memset(cd_buf + cd_len + sizeof h, 0, 512);
                    memcpy(cd_buf + cd_len + sizeof h, (const void *)(uintptr_t)st.sbuf[0][0].addr, st.sbuf[0][0].size < 512 ? st.sbuf[0][0].size : 512);
                    cd_len += sizeof h + 512;
                } else {
                    fwrite(cd_buf, 1, cd_len, cd);
                    fclose(cd);
                    cd = NULL;
                    LOG("[count] storage buffer dump written: %zu bytes\n", cd_len);
                }
            }
            if (++cf_n == 3 && trig && !fired && p >= 1500) {
                fired = 1;
                if (trig == 1) { extern volatile int g_gpuprof_request; g_gpuprof_request = 1; }
                else { extern volatile int g_debug_frame_request; g_debug_frame_request = 1; }
                LOG("[count] fs %016llx drawn 3 times in frame %llu -> %s the next frame\n", (unsigned long long)cnt_hash,
                    (unsigned long long)p, trig == 1 ? "GPU profile" : "debug dump of");
            }
        }
    }
    cum_draws++;
    if (g_debug_frame) { debug_probe(pr); debug_probe_row(pr); }
    if (g_debug_frame && g_cur_fs_hash == 0x85b366a0e972a9bcull && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {
        char path[96];  // (LARECOMP_FLAME_AUTO / torch flame: its colour target after the draw)
        snprintf(path, sizeof path, "build/pass_flame_%llu_after.png", (unsigned long long)renderer_present_count());
        renderer_save_texture_tex((TexBackend *)st.rt.color[0]->backend, path);
    }
    if (g_gpuprof_frame && gp_n) {
        if (!gp_draws[gp_n - 1]) {  // first draw of the row: its fragment shader hash
            size_t l = strlen(gp_label[gp_n - 1]);
            snprintf(gp_label[gp_n - 1] + l, sizeof gp_label[0] - l, ", first fs %016llx", (unsigned long long)g_cur_fs_hash);
        }
        gp_draws[gp_n - 1]++;
    }
}

// LARECOMP_DEBUG_PROBE=x,y (debug frame): after each draw, texel (x, y) of every colour target big enough (its own
// allocation's pixels) -> "[probe] #<draw> fs <hash> vs/fs <addr> rt<i> <w>x<h> f<fmt>: <old hex> -> <new hex>" when it
// changed. Finds the draw that writes a given pixel. Ends the open pass without a dump (the next draw begins it again,
// LOAD) and waits for the GPU after every draw: debug frames only.
static int probe_auto_x = -1, probe_auto_y;  // LARECOMP_DEBUG_PROBE=auto: first NaN/Inf pixel a pass dump found
// (flame_probe_x / _y: set by LARECOMP_FLAME_AUTO, used when LARECOMP_DEBUG_PROBE is not set)
static uint64_t probe_auto_frame;
static void debug_probe(Prog *pr) {
    static int px = -2, py, ndraw, automode, prel;
    if (px == -2) {  // "x,yr": x, y in 1600x900 scene units, scaled to each target (half-res targets too)
        const char *e = getenv("LARECOMP_DEBUG_PROBE");
        px = -1;
        automode = e && !strcmp(e, "auto");
        if (e && !automode) { px = atoi(e); const char *cm = strchr(e, ','); py = cm ? atoi(cm + 1) : 0; prel = strchr(e, 'r') != NULL; }
    }
    if (automode) {  // (with LARECOMP_DEBUG_FRAMES=2: frame N's dumps find the pixel, frame N + 1 probes it)
        if (probe_auto_x < 0 || renderer_present_count() == probe_auto_frame) return;
        px = probe_auto_x; py = probe_auto_y;
    }
    int qx = px, qy = py;
    bool rel = prel;  // (LARECOMP_FLAME_AUTO pixels are in 1600x900 scene units: scaled to each target's own size)
    if (qx < 0 && flame_probe_x >= 0) { qx = flame_probe_x; qy = flame_probe_y; rel = true; }
    if (qx < 0 || !st.rt.ncolor) return;
    ndraw++;
    if (in_pass) { vkCmdEndRendering(vk_cmd()); in_pass = false; blk_copy_back(vk_cmd()); }
    RtState rts = st.rt;
    VkSpan s = vk_readback(16 * 9);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    bool any = false;
    for (int i = 0; i < rts.ncolor && i < 8; i++) {
        TexBackend *b = rts.color[i] ? (TexBackend *)rts.color[i]->backend : NULL;
        if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) continue;
        int tx = rel ? (int)((int64_t)qx * b->width / 1600) : qx, ty = rel ? (int)((int64_t)qy * b->height / 900) : qy;
        if ((uint32_t)tx >= b->width || (uint32_t)ty >= b->height) continue;
        VkBufferImageCopy rg = {.bufferOffset = s.off + (VkDeviceSize)i * 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageOffset = {tx, ty, 0}, .imageExtent = {1, 1, 1}};
        vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, s.buf, 1, &rg);
        any = true;
    }
    // the depth attachment's texel too (depth aspect only; slot 8): material-class depth buffers (TotK lighting quads use
    // depth EQUAL against a class depth written by gl_FragDepth)
    TexBackend *db = rts.depth ? (TexBackend *)rts.depth->backend : NULL;
    int dtx = 0, dty = 0;
    bool dok = db && db->image && (db->aspect & VK_IMAGE_ASPECT_DEPTH_BIT);
    if (dok) {
        dtx = rel ? (int)((int64_t)qx * db->width / 1600) : qx; dty = rel ? (int)((int64_t)qy * db->height / 900) : qy;
        dok = (uint32_t)dtx < db->width && (uint32_t)dty < db->height;
    }
    if (dok) {
        VkBufferImageCopy rg = {.bufferOffset = s.off + 8 * 16, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                                .imageOffset = {dtx, dty, 0}, .imageExtent = {1, 1, 1}};
        vkCmdCopyImageToBuffer(cb, db->image, VK_IMAGE_LAYOUT_GENERAL, s.buf, 1, &rg);
        any = true;
    }
    if (!any) return;
    renderer_flush();
    if (dok) {
        static const void *dlast_t; static uint32_t dlast;
        uint32_t dv = 0;
        memcpy(&dv, (const uint8_t *)s.ptr + 8 * 16, rts.depth->format == 0x32 && db->format == VK_FORMAT_D16_UNORM ? 2 : 4);
        if (dlast_t != rts.depth || dlast != dv) {
            double df = db->format == VK_FORMAT_D16_UNORM ? dv / 65535.0 : (double)*(float *)&dv;
            LOG("[probe] #%d fs %016llx depth f%#x %ux%u: %08x (%.9f)\n", ndraw, (unsigned long long)g_cur_fs_hash,
                rts.depth->format, db->width, db->height, dv, df);
            dlast_t = rts.depth; dlast = dv;
        }
    }
    static struct { const void *t; uint8_t v[16]; } last[64];
    for (int i = 0; i < rts.ncolor && i < 8; i++) {
        NvnTexture *t = rts.color[i];
        TexBackend *b = t ? (TexBackend *)t->backend : NULL;
        if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) continue;
        int tx = rel ? (int)((int64_t)qx * b->width / 1600) : qx, ty = rel ? (int)((int64_t)qy * b->height / 900) : qy;
        if ((uint32_t)tx >= b->width || (uint32_t)ty >= b->height) continue;
        const FormatInfo *fi = nvn_format_info(t->format);
        int n = fi && fi->bpb && fi->bpb <= 16 ? fi->bpb : 4;
        const uint8_t *v = (const uint8_t *)s.ptr + i * 16;
        int k = 0;
        while (k < 64 && last[k].t && last[k].t != t) k++;
        if (k == 64) k = 63;
        bool same = last[k].t == t && !memcmp(last[k].v, v, n);
        if (!same) {
            static int state_logged = -1;
            if (state_logged != ndraw) {
                state_logged = ndraw;
                LOG("[probe] #%d fs %016llx state: depth test %d write %d op %d (ds %02x %02x %02x) pass depth fmt %d viewport z %.6f..%.6f\n",
                    ndraw, (unsigned long long)g_cur_fs_hash, probe_dt, probe_dw, (int)probe_dop, st.ds[0], st.ds[1], st.ds[2],
                    (int)pass_depth, st.depth_range[0][0], st.depth_range[0][1]);
            }
            char o[40] = "", nw[40] = "";
            for (int j = 0; j < n; j++) { sprintf(o + j * 2, "%02x", last[k].t == t ? last[k].v[j] : 0); sprintf(nw + j * 2, "%02x", v[j]); }
            LOG("[probe] #%d fs %016llx vs %llx fs %llx rt%d %ux%u f%#x: %s -> %s\n", ndraw, (unsigned long long)g_cur_fs_hash,
                (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr, i, b->width,
                b->height, t->format, last[k].t == t ? o : "(first)", nw);
            last[k].t = t;
            memcpy(last[k].v, v, n);
        }
    }
    (void)pr;
}

// LARECOMP_DEBUG_PROBE_ROW=<y>[,<y>...] (debug frame, up to 4 rows; y in 1600x900 scene units, scaled to each target):
// after each draw, the whole rows y of every colour target -> "[rowprobe] #<draw> fs <hash> vs/fs <addr> rt<i> <w>x<h>
// f<fmt> y <y>: <n> px in <k> runs: a-b ..." for each target row the draw changed. Finds the draws that write thin
// features (the vertical streaks, user 2026-10-15: the same thin x range on rows far apart) without knowing in advance
// where one lands. Waits for the GPU after every draw: debug frames only.
static void debug_probe_row(Prog *pr) {
    static int nrows = -1, ry[4], ndraw;
    if (nrows < 0) {
        nrows = 0;
        const char *e = getenv("LARECOMP_DEBUG_PROBE_ROW");
        for (const char *p = e; p && *p && nrows < 4;) {
            ry[nrows++] = (int)strtol(p, (char **)&p, 10);
            if (*p == ',') p++; else break;
        }
    }
    if (!nrows || !st.rt.ncolor) return;
    ndraw++;
    if (in_pass) { vkCmdEndRendering(vk_cmd()); in_pass = false; blk_copy_back(vk_cmd()); }
    RtState rts = st.rt;
    enum { MAXW = 4096, STRIDE = MAXW * 16 };
    VkSpan s = vk_readback((uint64_t)STRIDE * 8 * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    int tw[8] = {0}, th[8] = {0}, bpp[8] = {0};
    bool any = false;
    for (int i = 0; i < rts.ncolor && i < 8; i++) {
        TexBackend *b = rts.color[i] ? (TexBackend *)rts.color[i]->backend : NULL;
        if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) || b->width > MAXW) continue;
        const FormatInfo *fi = nvn_format_info(rts.color[i]->format);
        bpp[i] = fi && fi->bpb && fi->bpb <= 16 ? fi->bpb : 4;
        for (int r = 0; r < nrows; r++) {
            int ty = (int)((int64_t)ry[r] * b->height / 900);
            if ((uint32_t)ty >= b->height) continue;
            VkBufferImageCopy rg = {.bufferOffset = s.off + (VkDeviceSize)(r * 8 + i) * STRIDE,
                                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageOffset = {0, ty, 0},
                                    .imageExtent = {b->width, 1, 1}};
            vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, s.buf, 1, &rg);
        }
        tw[i] = (int)b->width; th[i] = (int)b->height;
        any = true;
    }
    if (!any) return;
    renderer_flush();
    static struct { const void *t; int row; uint8_t *data; int w; } last[1024];
    for (int i = 0; i < rts.ncolor && i < 8; i++) {
        if (!tw[i]) continue;
        NvnTexture *t = rts.color[i];
        for (int r = 0; r < nrows; r++) {
            int ty = (int)((int64_t)ry[r] * th[i] / 900);
            if (ty >= th[i]) continue;
            int k = 0;
            while (k < 1024 && last[k].t && !(last[k].t == t && last[k].row == r)) k++;
            if (k == 1024) continue;   // (was: reuse slot 63 -> compares unrelated rows, 2026-10-15 run309 garbage repeats)
            if (!last[k].data) last[k].data = malloc(STRIDE);
            const uint8_t *v = (const uint8_t *)s.ptr + (size_t)(r * 8 + i) * STRIDE;
            int w = tw[i], n = bpp[i];
            if (last[k].t == t && last[k].w == w) {
                int nchg = 0, nruns = 0, start = -1, rl = 0;
                char runs[600] = "";
                for (int x = 0; x <= w; x++) {
                    bool c = x < w && memcmp(v + (size_t)x * n, last[k].data + (size_t)x * n, n) != 0;
                    if (c) { nchg++; if (start < 0) start = x; }
                    else if (start >= 0) {
                        nruns++;
                        if (rl < (int)sizeof runs - 24) rl += snprintf(runs + rl, sizeof runs - rl, " %d-%d", start, x - 1);
                        start = -1;
                    }
                }
                if (nchg)
                    LOG("[rowprobe] #%d fs %016llx vs %llx fs %llx rt%d %dx%d f%#x y %d: %d px in %d runs:%s\n", ndraw,
                        (unsigned long long)g_cur_fs_hash, (unsigned long long)st.program->stages[0].code_addr,
                        (unsigned long long)st.program->stages[1].code_addr, i, w, th[i], t->format, ty, nchg, nruns, runs);
            }
            last[k].t = t; last[k].row = r; last[k].w = w;
            memcpy(last[k].data, v, (size_t)w * n);
        }
    }
    (void)pr;
}

static void debug_draw(Prog *pr, CmdRec *r, bool indexed) {
    // LARECOMP_SAVE_GLSL: also every program this debug frame draws (programs built by the precompile replay or an
    // earlier translation were saved under that session's code addresses, not this one's)
    if (getenv("LARECOMP_SAVE_GLSL") && pr->vi && pr->fi && !pr->glsl_saved) {
        pr->glsl_saved = true;
        if (pr->vi->msl && pr->fi->msl) {
            save_glsl(pr->vi->msl, st.program->stages[0].code_addr, "vs");
            save_glsl(pr->fi->msl, st.program->stages[1].code_addr, "fs");
        } else {   // GLSL dropped after the pipeline libraries were built (compile_program): translate again to save it
            pthread_mutex_lock(&translate_lock);
            maxwell_gmem_base = HEAP_BASE;
            maxwell_gmem_size = HEAP_MAX;
            maxwell_gdev_size = gdev_window_size();
            ShaderInfo *vi = translate_stage(pr->code[0], pr->cmax[0], pr->csize[0], STAGE_VERTEX, &pr->vkey);
            ShaderInfo *fi = translate_stage(pr->code[1], pr->cmax[1], pr->csize[1], STAGE_FRAGMENT, &pr->fkey);
            pthread_mutex_unlock(&translate_lock);
            if (vi->msl) save_glsl(vi->msl, st.program->stages[0].code_addr, "vs");
            if (fi->msl) save_glsl(fi->msl, st.program->stages[1].code_addr, "fs");
            maxwell_free(vi);
            maxwell_free(fi);
        }
    }
    LOG("[draw] %s inst %llu mode %llu count %llu prog vs=%llx fs=%llx | vs cb %#x fs cb %#x | rt %d", indexed ? "e" : "a",
        (unsigned long long)(indexed ? r->x[6] : r->x[4]), (unsigned long long)r->x[0], (unsigned long long)r->x[2],
        (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
        pr->vi->cbuf_mask, pr->fi->cbuf_mask, st.rt.ncolor);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" %ux%u/f%#x", st.rt.color[i]->width, st.rt.color[i]->height, st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d%#x %p", st.rt.depth->format, (void *)st.rt.depth);
    LOG(" | vp %.0f %.0f %.0f %.0f | attrs %#x blend %#x ds %02x%02x%02x poly %02x%02x chmask %08x%s\n", st.viewport[0][0], st.viewport[0][1],
        st.viewport[0][2], st.viewport[0][3], pr->vi->attr_mask, st.color_state[0], st.ds[0], st.ds[1], st.ds[2], st.poly[0], st.poly[1],
        st.channel_mask, st.depth_clamp ? " clamp" : "");
    // LARECOMP_DEBUG_CB=v11.25,v4.147,f3.7: vec4 <index> of shader cbuf <n> of the vertex (v) / fragment (f) stage, per draw
    if (getenv("LARECOMP_DEBUG_CB")) {
        const char *e = getenv("LARECOMP_DEBUG_CB");
        while (*e) {
            char sg = *e++;
            int c = (int)strtol(e, (char **)&e, 10), idx = 0;
            if (*e == '.') idx = (int)strtol(e + 1, (char **)&e, 10);
            while (*e == ',' || *e == ' ') e++;
            int stg = sg == 'f' ? 1 : 0;
            ShaderInfo *si = stg ? pr->fi : pr->vi;
            if (c == 1 && si && st.program) {  // compiler constants after the code (cbuf_info k == 1)
                const NvnShaderStage *ss = &st.program->stages[stg];
                const uint32_t *u = (const uint32_t *)(uintptr_t)(ss->code_addr + ss->const_offset + (uint64_t)idx * 16);
                LOG("[draw]   cb %c1[%d] (code %llx const off %#x size %#x, shader reads %#x) = %08x %08x %08x %08x\n", sg, idx,
                    (unsigned long long)ss->code_addr, ss->const_offset, ss->const_size, si->cbuf_size[1], u[0], u[1], u[2], u[3]);
                continue;
            }
            if (c < 3 || c >= 18 || !si || !((si->cbuf_mask >> c) & 1)) continue;
            const Range *ub = &st.ubuf[stg][c - 3];
            if (!ub->addr || (uint32_t)(idx * 16 + 16) > ub->size) continue;
            const uint32_t *u = (const uint32_t *)(uintptr_t)(ub->addr + (uint64_t)idx * 16);
            float f[4];
            memcpy(f, u, 16);
            LOG("[draw]   cb %c%d[%d] = %.9g %.9g %.9g %.9g (%08x %08x %08x %08x)\n", sg, c, idx, f[0], f[1], f[2], f[3], u[0], u[1],
                u[2], u[3]);
        }
    }
    if (getenv("LARECOMP_DEBUG_DATA")) {  // vertex streams 0-1 and fragment cbufs: the first floats
        for (int i = 0; i < 32; i++) {
            if (!(pr->vi->attr_mask & (1u << i))) continue;
            uint32_t a = i < st.nattrib ? st.attrib[i] : 0, s = (a >> 8) & 0x1F;
            LOG("[draw]   attr%d fmt %#x stream %u off %u stride %u -> vk %d\n", i, a & 0xFF, s, a >> 13,
                s < (uint32_t)st.nstream ? st.stream[s][0] : 0, (int)vertex_format(a & 0xFF));
        }
        for (int i = 0; i < st.rt.ncolor; i++)
            if (st.rt.color[i])
                LOG("[draw]   rt%d %p addr %llx tgt %u depth %u | view %d set %#x layer %u+%u level %u\n", i, (void *)st.rt.color[i],
                    (unsigned long long)st.rt.color[i]->addr, st.rt.color[i]->target, st.rt.color[i]->depth, st.rt.has_cview[i],
                    st.rt.has_cview[i] ? st.rt.cview[i].setmask : 0, st.rt.has_cview[i] ? st.rt.cview[i].min_layer : 0,
                    st.rt.has_cview[i] ? st.rt.cview[i].num_layers : 0, st.rt.has_cview[i] ? st.rt.cview[i].base_level : 0);
        if (st.vbuf[0].addr) {
            const uint8_t *v = (const uint8_t *)(uintptr_t)st.vbuf[0].addr;
            LOG("[draw]   vbuf0 hex:");
            for (int k = 0; k < 48; k++) LOG("%s%02x", k % 4 ? "" : " ", v[k]);
            LOG("\n");
        }
        for (int s = 0; s < 2; s++) {
            if (!st.vbuf[s].addr) continue;
            const float *v = (const float *)(uintptr_t)st.vbuf[s].addr;
            LOG("[draw]   vbuf%d %llx size %llu:", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size);
            for (int k = 0; k < 16; k++) LOG(" %.4g", v[k]);
            LOG("\n");
        }
        for (int c = 3; c < 18; c++) {  // shader cbuf c = st.ubuf[c - 3] (cbuf_info: 0 driver, 1 compiler constants)
            const Range *ub = &st.ubuf[1][c - 3];
            if (!((pr->fi->cbuf_mask >> c) & 1) || !ub->addr) continue;
            const float *u = (const float *)(uintptr_t)ub->addr;
            LOG("[draw]   fcb%d %llx size %u:", c, (unsigned long long)ub->addr, (unsigned)ub->size);
            for (int k = 0; k < 96 && k * 4 < (int)ub->size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
        for (int c = 3; c < 18; c++) {  // vertex-stage cbufs (same numbering)
            const Range *ub = &st.ubuf[0][c - 3];
            if (!((pr->vi->cbuf_mask >> c) & 1) || !ub->addr) continue;
            const float *u = (const float *)(uintptr_t)ub->addr;
            LOG("[draw]   vcb%d %llx size %u:", c, (unsigned long long)ub->addr, (unsigned)ub->size);
            for (int k = 0; k < 96 && k * 4 < (int)ub->size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
    }
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {  // fragment textures: guest size / format, allocation, scale
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) {   // (why: handle slot, raw handle, texture pool size; F9 capture 72819: UI blur source had 3 of these)
            int hs = ((int)pr->fi->tex[i].handle_offset - 0x20) / 8;
            uint64_t hv = hs >= 0 && hs < 32 ? st.tex[1][hs] : 0;
            LOG("[draw]   tex%d: none (handle offset %#x slot %d handle %llx tic %u pool %d%s)\n", i, pr->fi->tex[i].handle_offset, hs,
                (unsigned long long)hv, (unsigned)(hv & 0xFFFFF), st.tpool ? st.tpool->count : -1, pr->fi->tex[i].bindless ? " bindless" : "");
            uint32_t tic = (uint32_t)(hv & 0xFFFFF);
            if (st.tpool && tic < (uint32_t)st.tpool->count) {
                const NvnTexEntry *e = &((const NvnTexEntry *)st.tpool->entries)[tic];
                LOG("[draw]     slot %u: registered %u times, last at frame %lld: %ux%u f%#x addr %llx '%s'; finalized at frame %lld (now %llu)\n",
                    tic, e->dbg_regs, (long long)e->dbg_reg_frame - 1, e->dbg_w, e->dbg_h, e->dbg_fmt, (unsigned long long)e->dbg_addr,
                    e->dbg_label, (long long)e->dbg_fin_frame - 1, (unsigned long long)renderer_present_count());
            }
            for (unsigned k = 0; k < 1024 && k < g_tex_fin_n; k++) {  // (was that slot emptied by nvnTextureFinalize?)
                const NvnFinRec *r = &g_tex_fin[(g_tex_fin_n - 1 - k) & 1023];
                if (r->tic != (hv & 0xFFFFF)) continue;
                LOG("[draw]     tic %u was %ux%u f%#x addr %llx '%s', finalized at frame %llu (now %llu, %u finalizes ago)\n", r->tic, r->w,
                    r->h, r->fmt, (unsigned long long)r->addr, r->label, (unsigned long long)r->frame,
                    (unsigned long long)renderer_present_count(), k);
                break;
            }
            continue;
        }
        int memnz = -1;   // (non-zero bytes in the first 256 of its storage: CPU data present? same as upload_texture reads)
        if (b && !b->gpu_owned && t->addr) { memnz = 0; for (int k = 0; k < 256; k++) memnz += ((const uint8_t *)(uintptr_t)t->addr)[k] != 0; }
        LOG("[draw]   tex%d: %p %ux%u f%#x lv %u -> %ux%u scale %.4f %.4f%s addr %llx tgt %u depth %u layers %u | owned %d up %d dyn %d mem nz %d/256\n", i,
            (void *)t, t->width, t->height, t->format, b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0, b ? b->sx : 0, b ? b->sy : 0,
            b && b->scalable ? " scalable" : "", (unsigned long long)t->addr, t->target, t->depth, b ? b->layers : 0, b ? b->gpu_owned : -1,
            b ? b->uploaded : -1, b ? b->dynamic : -1, memnz);
        // LARECOMP_DEBUG_TEX=<fs hash hex>: contents of each texture that fragment shader samples ([passstat] line); ends the open pass
        const char *dt = getenv("LARECOMP_DEBUG_TEX");
        if (b && dt && strtoull(dt, NULL, 16) == g_cur_fs_hash) {
            static int tex_dump_no;
            char path[96];
            snprintf(path, sizeof path, "build/pass_tex_%03d_%d.png", tex_dump_no++, i);
            renderer_save_texture_tex((TexBackend *)b, path);
        }
    }
    for (int i = 0; i < pr->vi->ntex && i < 64; i++) {  // vertex textures (displacement / height maps: the 2026-10-15 streaks)
        NvnTexture *t = shader_texture_object(0, &pr->vi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) { LOG("[draw]   vtex%d: none (handle offset %#x)\n", i, pr->vi->tex[i].handle_offset); continue; }
        int memnz = -1;
        if (b && !b->gpu_owned && t->addr) { memnz = 0; for (int k = 0; k < 256; k++) memnz += ((const uint8_t *)(uintptr_t)t->addr)[k] != 0; }
        LOG("[draw]   vtex%d: %p %ux%u f%#x lv %u -> %ux%u%s addr %llx tgt %u depth %u layers %u | owned %d up %d dyn %d mem nz %d/256 | smp %llx\n", i,
            (void *)t, t->width, t->height, t->format, b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0,
            b && b->scalable ? " scalable" : "", (unsigned long long)t->addr, t->target, t->depth, b ? b->layers : 0, b ? b->gpu_owned : -1,
            b ? b->uploaded : -1, b ? b->dynamic : -1, memnz, (unsigned long long)pr->vi->tex[i].handle_offset);
        const char *dt = getenv("LARECOMP_DEBUG_VTEX");   // =<fs hash hex>: dump that program's vertex textures
        if (b && dt && strtoull(dt, NULL, 16) == g_cur_fs_hash) {
            static int vtex_dump_no;
            char path[96];
            snprintf(path, sizeof path, "build/pass_vtex_%03d_%d.png", vtex_dump_no++, i);
            renderer_save_texture_tex((TexBackend *)b, path);
        }
    }
}

// ================================================================== texture copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRefRec;

static void copy_region_dims(NvnTexture *t, const TexRefRec *ref, const FormatInfo **fi, int32_t rg[6], int *level,
                             uint64_t *row, uint64_t *img) {
    *fi = nvn_format_info(t->format);
    memcpy(rg, ref->region, sizeof ref->region);
    if (!rg[3]) rg[3] = (int32_t)t->width;
    if (!rg[4]) rg[4] = (int32_t)t->height;
    if (!rg[5]) rg[5] = 1;
    *level = ref->has_view && (ref->view.setmask & 1) ? ref->view.base_level : 0;
    uint32_t wb = ((uint32_t)rg[3] + (*fi)->bw - 1) / (*fi)->bw, hb = ((uint32_t)rg[4] + (*fi)->bh - 1) / (*fi)->bh;
    *row = st.copy_row_stride ? st.copy_row_stride : (uint64_t)wb * (*fi)->bpb;
    *img = st.copy_image_stride ? st.copy_image_stride : *row * hb;
}

static void copy_buffer_to_texture(uint64_t src, const TexRefRec *dst) {
    snap_reason = "buf2tex"; snapshots_invalidate(NULL);
    NvnTexture *t = (NvnTexture *)(uintptr_t)dst->tex;
    if (!t || !src) return;
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, dst, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb) return;
    extern bool tex_watched(uint64_t addr);
    if (tex_watched(t->addr))
        LOG("[texw] buffer %llx -> texture addr %llx level %d region %d,%d,%d %dx%dx%d row %llu img %llu\n", (unsigned long long)src,
            (unsigned long long)t->addr, level, rg[0], rg[1], rg[2], rg[3], rg[4], rg[5], (unsigned long long)row, (unsigned long long)img);
    TexBackend *b = vk_tex(t, false);  // prior CPU contents uploaded first
    extern void rt_first_note(NvnTexture *t, const char *how);
    rt_first_note(t, "buf2tex");
    renderer_mark_gpu_owned(t);
    if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    if (b->astc_bw) {  // (blocks would need the CPU decode: not seen in the game yet)
        static int nlog;
        if (nlog++ < 5) LOG("[vk] buffer -> ASTC texture copy skipped (%ux%u)\n", t->width, t->height);
        return;
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t hb = ((uint32_t)rg[4] + fi->bh - 1) / fi->bh, wb = ((uint32_t)rg[3] + fi->bw - 1) / fi->bw;
    uint32_t out_bpb = fi->bpb == 3 ? 4 : fi->bpb;
    uint32_t nz = (uint32_t)rg[5];
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t out_row = (uint64_t)wb * out_bpb;
    vk_up_kind = 4; VkSpan s = vk_transient(out_row * hb * nz, 16); vk_up_kind = 0;
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < hb; y++) {
            const uint8_t *sr = (const uint8_t *)(uintptr_t)(src + (uint64_t)z * img + (uint64_t)y * row);
            uint8_t *dr = (uint8_t *)s.ptr + ((uint64_t)z * hb + y) * out_row;
            if (out_bpb == fi->bpb) memcpy(dr, sr, out_row);
            else for (uint32_t x = 0; x < wb; x++) { memcpy(dr + x * 4, sr + x * 3, 3); dr[x * 4 + 3] = 255; }
        }
    VkBufferImageCopy c = {.bufferOffset = s.off, .bufferRowLength = wb * fi->bw, .bufferImageHeight = hb * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_to_buffer(const TexRefRec *src, uint64_t dst) {
    NvnTexture *t = (NvnTexture *)(uintptr_t)src->tex;
    if (!t || !dst || dst < HEAP_BASE) return;
    if (g_debug_frame) LOG("[readback] tex %ux%u f%#x -> %llx\n", t->width, t->height, t->format, (unsigned long long)dst);
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, src, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb || fi->bpb == 3 || row % fi->bpb) return;
    TexBackend *b = vk_tex(t, false);
    float tsx, tsy;
    renderer_texture_sample_scale(t, &tsx, &tsy);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if ((tsx != 1 || tsy != 1) && level == 0 && !renderer_format_is_depth(t->format)) {
        // the CPU expects guest-sized texels: filter the scaled allocation down first
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t nz = (uint32_t)rg[5];
    uint64_t total = img * nz;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                           .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();  // the CPU may read the result right after the fence
    memcpy((void *)(uintptr_t)dst, out.ptr, total);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
}

static void copy_texture(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    snap_reason = "tex2tex"; snapshots_invalidate(NULL);
    NvnTexture *st_ = (NvnTexture *)(uintptr_t)src->tex, *dt = (NvnTexture *)(uintptr_t)dst->tex;
    if (!st_ || !dt) return;
    extern bool tex_watched(uint64_t addr);
    if (tex_watched(dt->addr) || tex_watched(st_->addr))
        LOG("[texw] copy texture %llx %ux%ux%u -> %llx %ux%ux%u (src lv %d, dst lv %d, region z %d+%d)\n", (unsigned long long)st_->addr,
            st_->width, st_->height, st_->depth, (unsigned long long)dt->addr, dt->width, dt->height, dt->depth,
            src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0, dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0,
            dst->region[2], dst->region[5]);
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each copy its own submission, waited on)
    gp_mark(vk_cmd(), "copy texture %ux%u -> %ux%u", st_->width, st_->height, dt->width, dt->height);
    dl_note("copy texture %ux%u f%#x -> %ux%u f%#x (%p -> %p)", st_->width, st_->height, st_->format, dt->width, dt->height,
            dt->format, (void *)st_, (void *)dt);
    hang_op(vk_cmd(), HOP_COPY, st_->addr, dt->addr, 0, dt->width, dt->height);
    TexBackend *s = vk_tex(st_, false), *d = vk_tex(dt, true);
    renderer_mark_gpu_owned(dt);
    if (renderer_texture_presented(dt) && st_->width == dt->width && st_->height == dt->height) renderer_texture_mark_presented(st_);
    renderer_texture_first_write(dt);
    renderer_texture_level_fresh(dt);
    const int32_t *sr = src->region, *dr = dst->region;
    float ssx, ssy, dsx, dsy;
    renderer_texture_sample_scale(st_, &ssx, &ssy);
    renderer_texture_scale(dt, &dsx, &dsy);
    int sx0 = (int)lroundf(sr[0] * ssx), sy0 = (int)lroundf(sr[1] * ssy);
    int dx0 = (int)lroundf(dr[0] * dsx), dy0 = (int)lroundf(dr[1] * dsy);
    int sw = sr[3] ? (int)lroundf(sr[3] * ssx) : (int)s->width, sh = sr[4] ? (int)lroundf(sr[4] * ssy) : (int)s->height;
    int dw = dr[3] ? (int)lroundf(dr[3] * dsx) : (int)d->width, dh = dr[4] ? (int)lroundf(dr[4] * dsy) : (int)d->height;
    uint32_t slv = src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0;
    uint32_t dlv = dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0;
    uint32_t sly = src->has_view && (src->view.setmask & 2) ? src->view.min_layer : 0;
    uint32_t dly = dst->has_view && (dst->view.setmask & 2) ? dst->view.min_layer : 0;
    if (slv >= s->levels || dlv >= d->levels || sly >= s->layers || dly >= d->layers) return;
    {   // whole-image regions are the level's extent, and every region stays inside its level: scaled allocations
        // (internal resolution below 1x) round guest regions past the edge, and out-of-bounds copies / blits are
        // undefined in Vulkan (garbage that changed every frame on NVIDIA: flicker in the depth pyramid's users)
        int slw = (int)level_extent(s->width, slv), slh = (int)level_extent(s->height, slv);
        int dlw = (int)level_extent(d->width, dlv), dlh = (int)level_extent(d->height, dlv);
        if (!sr[3]) sw = slw - sx0;
        if (!sr[4]) sh = slh - sy0;
        if (!dr[3]) dw = dlw - dx0;
        if (!dr[4]) dh = dlh - dy0;
        if (sx0 + sw > slw) sw = slw - sx0;
        if (sy0 + sh > slh) sh = slh - sy0;
        if (dx0 + dw > dlw) dw = dlw - dx0;
        if (dy0 + dh > dlh) dh = dlh - dy0;
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    }
    if (g_debug_frame)
        LOG("[copytex] %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] -> %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] phys (%d,%d %dx%d) -> (%d,%d %dx%d) flags %llx\n",
            (void *)st_, st_->width, st_->height, st_->format, slv, sr[0], sr[1], sr[3], sr[4], s->width, s->height, (void *)dt, dt->width,
            dt->height, dt->format, dlv, dr[0], dr[1], dr[3], dr[4], d->width, d->height, sx0, sy0, sw, sh, dx0, dy0, dw, dh,
            (unsigned long long)flags);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    if (s->samples > 1 && d->samples == 1 && sw == dw && sh == dh && s->format == d->format && !(sa & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkImageResolve rv = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                             .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdResolveImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
        return;
    }
    if (s->samples != d->samples) return;
    if (sw == dw && sh == dh && s->format == d->format) {
        VkImageCopy c = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                         .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return;
    }
    // scaled / converting copy: the game builds its depth pyramid this way (nearest), colour copies filter if asked
    bool depth = (sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth && s->format != d->format) return;
    VkImageBlit bl = {.srcSubresource = {depth ? (sa & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : sa) : sa, slv, sly, 1},
                      .srcOffsets = {{sx0, sy0, 0}, {sx0 + sw, sy0 + sh, 1}},
                      .dstSubresource = {depth ? (da & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : da) : da, dlv, dly, 1},
                      .dstOffsets = {{dx0, dy0, 0}, {dx0 + dw, dy0 + dh, 1}}};
    bool linear = (flags & 1) && !depth && !rt_kind(st_->format);
    dl_note("  blit %p lv%u/%u %ux%u (%d,%d %dx%d) -> %p lv%u/%u %ux%u (%d,%d %dx%d) %s", (void *)s->image, slv, s->levels, s->width,
            s->height, sx0, sy0, sw, sh, (void *)d->image, dlv, d->levels, d->width, d->height, dx0, dy0, dw, dh, linear ? "linear" : "nearest");
    vkCmdBlitImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// ================================================================== present helpers (device.c)
static VkPipelineLayout present_layout;
static VkDescriptorSetLayout present_set_layout;

static VkPipeline present_pipeline(VkFormat fmt) {
    static struct { VkFormat f; VkPipeline p; } cache[4];
    for (int i = 0; i < 4; i++) if (cache[i].p && cache[i].f == fmt) return cache[i].p;
    if (!present_layout) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &present_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &present_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &present_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("present.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader("present.frag"), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = present_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 4; i++) if (!cache[i].p) { cache[i].f = fmt; cache[i].p = p; break; }
    return p;
}

// Scanout shows the bytes the game wrote: sRGB images are read through a UNORM view (no decode), like metal.m.
static VkImageView raw_view(TexBackend *b) {
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM
               : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    static const VkComponentMapping id = {0};
    uint32_t ly = dump_layer < b->layers ? dump_layer : 0;
    return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, f, 0, 1, ly, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ---- post passes (Video tab "Anti-aliasing" / "Upscaler"): SMAA 1x at the game image's size, then AMD FSR 1 (EASU
// upscale to the image's size in the window, RCAS sharpen) when it is smaller than its place in the window, as
// metal.m does with MetalFX. Each pass: a fullscreen triangle into an RGBA8 image (gamma-encoded, like the input);
// up to three inputs (bindings 0-2), sampled linear / clamp.
static VkPipelineLayout post_layout;
static VkDescriptorSetLayout post_set_layout;

static VkPipeline post_pipeline(const char *fs) {
    static struct { const char *fs; VkPipeline p; } cache[8];
    for (int i = 0; i < 8; i++) if (cache[i].p && cache[i].fs == fs) return cache[i].p;
    if (!post_layout) {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &post_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &post_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &post_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = post_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 8; i++) if (!cache[i].p) { cache[i].fs = fs; cache[i].p = p; break; }
    return p;
}

// Post image `i` at w x h (RGBA8, kept until the size changes). 0, 1: SMAA edges / weights, 2: SMAA output,
// 3, 4: FSR EASU / RCAS.
static TexBackend *post_image(int i, uint32_t w, uint32_t h) {
    static TexBackend img[5];
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = VK_FORMAT_R8G8B8A8_UNORM; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

static VkImageView post_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// SMAA's lookup textures (runtime/nvn/smaa_data.c): 0 = area (160 x 560 R8G8), 1 = search (64 x 16 R8), uploaded once.
static VkImageView smaa_lut(int i) {
    static TexBackend t[2];
    TexBackend *d = &t[i];
    if (!d->image) {
        extern const unsigned char *smaa_area_tex(void);
        extern const unsigned char *smaa_search_tex(void);
        uint32_t w = i ? 64 : 160, h = i ? 16 : 560, bpp = i ? 1 : 2;
        VkFormat f = i ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        VkSpan s = vk_transient_copy(i ? smaa_search_tex() : smaa_area_tex(), (uint64_t)w * h * bpp, 16);
        VkBufferImageCopy c = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vkCmdCopyBufferToImage(cb, s.buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    return post_view(d);
}

// One post pass (outside the window image's rendering): srcs (bindings 0..n-1) -> dst, fragment shader `fs`.
// clear: start from zero (SMAA's edge pass only writes edge pixels; its weights / output read them).
static VkRect2D g_post_area;   // post_pass draws only here when set (the presented crop of the image)
static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear) {
    VkPipeline pipe = post_pipeline(fs);  // (before the rendering begins: created inside it, the NVIDIA driver crashed)
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(dst),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {dst->width, dst->height}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    if (g_post_area.extent.width) ri.renderArea = g_post_area;
    vkCmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(post_set_layout);
    VkDescriptorImageInfo ii[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){vk_default_sampler(true), srcs[i], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    dyn_invalidate();  // (viewport / scissor only dynamic: the rest of the dynamic state is undefined after this)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dst->width, (float)dst->height, 0, 1};
    VkRect2D sc = g_post_area.extent.width ? g_post_area : (VkRect2D){{0, 0}, {dst->width, dst->height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

// The view the present samples: the game image, or its anti-aliased / upscaled version. ow x oh: the whole image's
// size in the window (its crop is drawn into the viewport).
static VkImageView post_process(VkCommandBuffer cb, TexBackend *b, double ow_d, double oh_d, const float crop[4]) {
    VkImageView v = raw_view(b);
    bool smaa = g_settings.antialias == AA_SMAA;
    uint32_t ow = (uint32_t)lround(ow_d), oh = (uint32_t)lround(oh_d);
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    bool fsr = g_settings.upscaler == UPSCALE_FSR1 && (ow > b->width + 1 || oh > b->height + 1);  // (an upscale only)
    // TOTK_SHARPEN=<stops> (Graphics menu "Sharpening"): FSR 1 RCAS on the presented image, also without an upscale (the
    // Deck shows the 1280x720 canvas 1:1, where the FSR path above never runs). Includes the HUD: the game composites it
    // into the canvas before we see it. Unset: off, and the FSR path keeps AMD's default 0.2.
    static float sharpen = -2;
    if (sharpen == -2) { const char *e = getenv("TOTK_SHARPEN"); sharpen = e && *e ? (float)atof(e) : -1; }
    bool rcas = sharpen >= 0 && !fsr;
    if (!smaa && !fsr && !rcas) return v;
    extern void vk_frame_suspend(void);
    extern void vk_frame_resume(void);
    vk_frame_suspend();
    if (smaa) {  // SMAA 1x: luma edges -> blending weights -> neighbourhood blending
        float w = (float)b->width, h = (float)b->height;
        float m[8] = {1.0f / w, 1.0f / h, w, h};
        TexBackend *edges = post_image(0, b->width, b->height), *weights = post_image(1, b->width, b->height);
        TexBackend *out = post_image(2, b->width, b->height);
        VkImageView in[3] = {v};
        post_pass(cb, "smaa_edge.frag", in, 1, edges, m, true);
        VkImageView wi[3] = {post_view(edges), smaa_lut(0), smaa_lut(1)};
        post_pass(cb, "smaa_weight.frag", wi, 3, weights, m, true);
        VkImageView bi[3] = {v, post_view(weights)};
        post_pass(cb, "smaa_blend.frag", bi, 2, out, m, false);
        v = post_view(out);
    }
    if (fsr) {
        float iw = (float)b->width, ih = (float)b->height;
        TexBackend *e = post_image(3, ow, oh), *r = post_image(4, ow, oh);
        float pe[8] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f, iw, ih};
        VkImageView ei[3] = {v};
        post_pass(cb, "easu.frag", ei, 1, e, pe, false);
        float pr[8] = {exp2f(-(sharpen >= 0 ? sharpen : 0.2f)), 0, 0, 0, (float)ow, (float)oh};  // (0.2 stops: AMD's default)
        VkImageView ri_[3] = {post_view(e)};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        v = post_view(r);
    }
    if (rcas) {   // at the window's size when that is smaller than the image (sharpen what is seen, not what is shrunk)
        uint32_t rw = ow && ow < b->width ? ow : b->width, rh = oh && oh < b->height ? oh : b->height;
        TexBackend *r = post_image(4, rw, rh);
        float pr[8] = {exp2f(-sharpen), rw < b->width || rh < b->height ? 1.0f : 0.0f, 0, 0, (float)b->width, (float)b->height,
                       (float)rw, (float)rh};
        VkImageView ri_[3] = {v};
        // only the presented crop (Deck handheld: the 1280x720 corner of a 1920x1080 image), +1 texel for the bilinear edge
        int32_t x0 = (int32_t)(crop[0] * rw), y0 = (int32_t)(crop[1] * rh);
        uint32_t cw = (uint32_t)ceilf(crop[2] * rw) + 1, ch = (uint32_t)ceilf(crop[3] * rh) + 1;
        g_post_area = (VkRect2D){{x0, y0}, {cw < rw - x0 ? cw : rw - x0, ch < rh - y0 ? ch : rh - y0}};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        g_post_area = (VkRect2D){0};
        v = post_view(r);
    }
    vk_full_barrier(cb);
    vk_frame_resume();
    {
        static uint32_t piw, pih, pow_, poh;
        static int psm = -1, pfs = -1, prc = -1;
        if (b->width != piw || b->height != pih || ow != pow_ || oh != poh || smaa != psm || fsr != pfs || rcas != prc)
            LOG("[post] %ux%u%s%s%s -> %ux%u\n", b->width, b->height, smaa ? " smaa" : "", fsr ? " fsr1" : "", rcas ? " rcas" : "", ow, oh);
        piw = b->width; pih = b->height; pow_ = ow; poh = oh; psm = smaa; pfs = fsr; prc = rcas;
    }
    return v;
}

// Inside the window image's rendering (device.c vk_frame_begin): the game image into (vx, vy, vw, vh).
void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    VkImageView src = post_process(cb, b, pv[6] > 0 ? vw / pv[6] : vw, pv[7] > 0 ? vh / pv[7] : vh, pv + 4);
    VkPipeline p = present_pipeline(g_vk.sc_format);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    dyn_invalidate();
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {vx, vy, vw, vh, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &full);
}

// Screenshot of a game texture (presented crop applied to build/frame_* shots), via the present shader into RGBA8.
void renderer_save_texture_tex(TexBackend *b, const char *path) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    extern const float *vk_shot_crop(void);
    float crop[4] = {0, 0, 1, 1};
    if (strstr(path, "frame_")) memcpy(crop, vk_shot_crop(), sizeof crop);
    uint32_t w = (uint32_t)(b->width * crop[2]), h = (uint32_t)(b->height * crop[3]);
    if (!w || !h) return;
    vk_end_pass();
    // pass dumps go through RGBA32F: the PNG is clamped, the [passstat] line has the real values (HDR, NaN)
    bool fl = strstr(path, "pass_") != NULL;
    VkFormat ifmt = fl ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = fl ? 16 : 4;
    VkImage img;
    VmaAllocation mem;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = ifmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL));
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vk_image_init_layout(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = ifmt, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView tv;
    VK_CHECK(vkCreateImageView(g_vk.dev, &vci, NULL, &tv));
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkPipeline p = present_pipeline(ifmt);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), raw_view(b), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    dyn_invalidate();
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    float pv[8] = {renderer_origin_lower_left() && !strstr(path, "window") ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * bpp);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    if (fl) {
        const float *px = (const float *)out.ptr;
        uint8_t *rgba = malloc((size_t)w * h * 4);
        double sum[4] = {0};
        float mx[4] = {0};
        uint64_t bad = 0, nz = 0;
        char where[160] = "";  // first NaN/Inf pixels (painted magenta in the PNG)
        int nwhere = 0;
        for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
            bool any = false, badpx = false;
            for (int c = 0; c < 4; c++) {
                float v = px[i * 4 + c];
                if (v != v || v > 3.0e38f || v < -3.0e38f) { bad++; v = 0; badpx = true; }
                sum[c] += v;
                if (v > mx[c]) mx[c] = v;
                if (c < 3 && v != 0) any = true;
                rgba[i * 4 + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255.0f + 0.5f);
            }
            if (badpx) {
                rgba[i * 4] = 255; rgba[i * 4 + 1] = 0; rgba[i * 4 + 2] = 255; rgba[i * 4 + 3] = 255;
                if (!nwhere && probe_auto_x < 0 && strstr(path, "pass_")) {
                    probe_auto_x = (int)(i % w); probe_auto_y = (int)(i / w); probe_auto_frame = renderer_present_count();
                }
                if (nwhere < 6) { size_t l = strlen(where); snprintf(where + l, sizeof where - l, " %llu,%llu", (unsigned long long)(i % w), (unsigned long long)(i / w)); nwhere++; }
            }
            nz += any;
        }
        double n = (double)w * h;
        LOG("[passstat] %s %ux%u mean %.4g %.4g %.4g %.4g max %.4g %.4g %.4g %.4g nonzero %.1f%% naninf %llu%s%s\n", path, w, h,
            sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, mx[0], mx[1], mx[2], mx[3], 100.0 * nz / n, (unsigned long long)bad,
            nwhere ? " at" : "", where);
        vk_save_png(path, rgba, w, h);
        free(rgba);
    } else {
        vk_save_png(path, out.ptr, w, h);
    }
    vkDestroyImageView(g_vk.dev, tv, NULL);
    vmaDestroyImage(g_vk.vma, img, mem);
}

// ================================================================== executor (renderer.m exec_list)
static void exec_list(uint64_t handle, int depth) {
    uint64_t p = handle;
    const CmdChunk *k = cmd_chunk_for(handle);   // records live in the host shadow of their guest chunk (nvn.h)
    if (!k) { static int n; if (n++ < 8) LOG("[nvn] exec: unknown command handle %llx\n", (unsigned long long)handle); return; }
    // The chunk's base / shadow in locals (k-> was re-read every record), and the records a few slots ahead prefetched:
    // run363's profile had ~21 % of exec on the load of each record's op (the game thread wrote them on another core).
    const uint8_t *khost = k->host;
    uint64_t kbase = k->base;
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(khost + (p - kbase) * CMD_SHADOW_K);
        __builtin_prefetch((const char *)r + 512, 0, 3);
        __builtin_prefetch((const char *)r + 576, 0, 3);
        void *x = r->extra;
        if (r->op < CMD_COUNT) op_counts[r->op]++;
        static int skip_all = -1;   // LARECOMP_SKIP_ALL_DRAWS=1: no draws or clears at all (upper bound: fps with exec ~free)
        if (skip_all < 0) skip_all = getenv("LARECOMP_SKIP_ALL_DRAWS") != NULL;
        if ((g_render_off || skip_all) && (r->op == CMD_DRAW_ARRAYS || r->op == CMD_DRAW_ELEMENTS || r->op == CMD_CLEAR_COLOR ||
                             r->op == CMD_CLEAR_DEPTH_STENCIL || r->op == CMD_CLEAR_TEXTURE || r->op == CMD_DRAW_TEXTURE)) {
            g_cond_skipped++;
            if (g_debug_frame) LOG("[cond] op %u skipped (render disabled)\n", r->op);
            p += cmd_gsize(r->size);
            continue;
        }
        if (prof && op_t0) { op_ns[op_last] += prof_now() - op_t0; op_t0 = 0; }   // (LARECOMP_VK_PROF) time per command kind
        if (prof && r->op != CMD_CALL && r->op != CMD_JUMP && r->op != CMD_END && r->op < CMD_COUNT) { op_last = r->op; op_t0 = prof_now(); }
        // every command that writes state stage_set reads (bindings, pools, program, render targets, inline UBO data)
        // (bindings / pools / program are compared by content in stage_set: the game re-issues identical binds per draw, run580)
        if (r->op == CMD_SET_RENDER_TARGETS || r->op == CMD_CLEAR_TEXTURE || r->op == CMD_UPDATE_UNIFORM_BUFFER) g_set_gen++;
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP:
            p = r->x[0];
            if (!(k = cmd_chunk_for(p))) { LOG("[nvn] exec: JUMP to unknown chunk %llx\n", (unsigned long long)p); return; }
            khost = k->host;
            kbase = k->base;
            continue;
        case CMD_CALL: {
            const uint64_t *h = x;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) exec_list(h[i], depth + 1);
            break;
        }
        case CMD_BIND_PROGRAM: st.program = (NvnProgram *)(uintptr_t)r->x[0]; break;
        case CMD_BIND_BLEND: { uint8_t *b = x; memcpy(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: memcpy(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: memcpy(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: memcpy(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: memcpy(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB:
            st.nattrib = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.attrib, x, st.nattrib * 4);
            break;
        case CMD_BIND_VERTEX_STREAM:
            st.nstream = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.stream, x, st.nstream * 8);
            break;
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_BIND_TEXTURES_DEFERRED:   // handles read now, from guest memory (identity-mapped)
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6 && r->x[3]; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (const uint8_t *)(uintptr_t)r->x[3] + i * 8, 8);
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_RENDER_TARGETS: {
            const struct { uint64_t tex[8]; NvnTextureView views[8]; uint8_t has_view[8]; uint64_t depth;
                           NvnTextureView dview; uint8_t has_dview; } *rt = x;
            RtState prev = st.rt;  // the same targets again: keep the pass (a restart = full barrier + reload)
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = (int)r->x[0] > 8 ? 8 : (int)r->x[0];
            for (int i = 0; i < st.rt.ncolor; i++) {
                st.rt.color[i] = (NvnTexture *)(uintptr_t)rt->tex[i];
                st.rt.cview[i] = rt->views[i];
                st.rt.has_cview[i] = rt->has_view[i];
            }
            st.rt.depth = (NvnTexture *)(uintptr_t)rt->depth;
            st.rt.dview = rt->dview;
            st.rt.has_dview = rt->has_dview;
            static int always = -1;
            if (always < 0) always = getenv("LARECOMP_VK_PASS_RESTART") != NULL;
            if (always || memcmp(&prev, &st.rt, sizeof prev)) st.rt_dirty = true;
            break;
        }
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_DEPTH_CLAMP: { static int off = -1; if (off < 0) off = getenv("LARECOMP_NO_DEPTH_CLAMP") != NULL; st.depth_clamp = r->x[0] != 0 && !off; break; }
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_CLEAR_COLOR: {
            snap_reason = "clear-color"; snapshots_invalidate(NULL);
            int index = (int)r->x[0];
            // LARECOMP_SKIP_CCLEAR=<addr hex>:<n>[,<n>...][;<addr>:...]: skip the n-th clears (1-based, per frame) of those
            // textures (diagnostic A/B)
            static uint64_t skip_addr[8];
            static uint32_t skip_mask[8], skip_count[8];
            static int skip_init, nskip;
            static uint64_t skip_frame;
            if (!skip_init) {
                skip_init = 1;
                const char *e = getenv("LARECOMP_SKIP_CCLEAR");
                char *p = (char *)e;
                while (p && *p && nskip < 8) {
                    skip_addr[nskip] = strtoull(p, &p, 16);
                    while (*p == ':' || *p == ',') { int n = (int)strtol(p + 1, &p, 10); if (n >= 1 && n <= 32) skip_mask[nskip] |= 1u << (n - 1); }
                    nskip++;
                    if (*p == ';') p++; else break;
                }
            }
            bool skip = false;
            if (nskip && index < st.rt.ncolor && st.rt.color[index]) {
                if (renderer_present_count() != skip_frame) { skip_frame = renderer_present_count(); memset(skip_count, 0, sizeof skip_count); }
                for (int k = 0; k < nskip; k++)
                    if (st.rt.color[index]->addr == skip_addr[k]) {
                        skip_count[k]++;
                        skip = skip_count[k] <= 32 && (skip_mask[k] >> (skip_count[k] - 1) & 1);
                    }
            }
            static int log_clears = -1;
            if (log_clears < 0) log_clears = getenv("LARECOMP_LOG_CLEARS") != NULL;
            if ((g_debug_frame || log_clears) && index < st.rt.ncolor && st.rt.color[index]) {
                const uint32_t *cv = x;
                LOG("[cclear] frame %llu idx %d %p addr %llx %ux%u f%#x mask %#x kind %d color %08x %08x %08x %08x scissor %d %d %d %d"
                    " | view %d set %#x layer %u+%u level %u%s\n",
                    (unsigned long long)renderer_present_count(), index, (void *)st.rt.color[index],
                    (unsigned long long)st.rt.color[index]->addr, st.rt.color[index]->width, st.rt.color[index]->height,
                    st.rt.color[index]->format, (uint32_t)r->x[2], (int)r->x[5], cv[0], cv[1], cv[2], cv[3], st.scissor[0][0],
                    st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], st.rt.has_cview[index],
                    st.rt.has_cview[index] ? st.rt.cview[index].setmask : 0, st.rt.has_cview[index] ? st.rt.cview[index].min_layer : 0,
                    st.rt.has_cview[index] ? st.rt.cview[index].num_layers : 0, st.rt.has_cview[index] ? st.rt.cview[index].base_level : 0,
                    skip ? " SKIPPED" : "");
            }
            if (!skip && index < st.rt.ncolor && st.rt.color[index]) {
                pk_take(st.rt.color[index]);   // (pause capture: keep the canvas before its per-frame clear)
                clear(index, (int)r->x[5], x, (uint32_t)r->x[2], false, 0, false, 0, 0);
            }
            break;
        }
        case CMD_CLEAR_TEXTURE: {
            // ClearTexture(cmd, texture, view, region, color, mask): a render-target clear of the texture alone (renderer.m)
            const struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } *ct = x;
            NvnTexture *t = (NvnTexture *)(uintptr_t)ct->tex;
            if (!t) break;
            snap_reason = "clear-texture"; snapshots_invalidate(NULL);
            RtState saved = st.rt;
            int32_t sc[4];
            memcpy(sc, st.scissor[0], sizeof sc);
            vk_end_pass();
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = 1;
            st.rt.color[0] = t;
            if (ct->has_view) { st.rt.cview[0] = ct->view; st.rt.has_cview[0] = 1; }
            st.rt_dirty = true;
            const int32_t *rg = ct->region;
            if (rg[3] > 0 && rg[4] > 0) { st.scissor[0][0] = rg[0]; st.scissor[0][1] = rg[1]; st.scissor[0][2] = rg[3]; st.scissor[0][3] = rg[4]; }
            else { st.scissor[0][0] = st.scissor[0][1] = 0; st.scissor[0][2] = st.scissor[0][3] = 16384; }
            clear(0, (int)r->x[6], ct->color, (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 0xF, false, 0, false, 0, 0);
            vk_end_pass();
            st.rt = saved;
            memcpy(st.scissor[0], sc, sizeof sc);
            st.rt_dirty = true;
            break;
        }
        case CMD_CLEAR_DEPTH_STENCIL:
            snap_reason = "clear-depth"; snapshots_invalidate(NULL);
            if (st.rt.depth) {
                bool has_stencil = renderer_format_has_stencil(st.rt.depth->format);
                if (g_debug_frame)
                    LOG("[dclear] depth %p f%#x %ux%u: depth %d = %.9f, stencil %d = %#x mask %#x\n", (void *)st.rt.depth,
                        st.rt.depth->format, st.rt.depth->width, st.rt.depth->height, r->x[0] != 0, r->f[0], r->x[2] != 0,
                        (uint32_t)r->x[1], (uint32_t)r->x[2]);
                clear(-1, 0, NULL, 0, r->x[0] != 0, r->f[0], has_stencil && r->x[2] != 0, (uint32_t)r->x[1], (uint32_t)r->x[2]);
            }
            break;
        case CMD_SET_RENDER_ENABLE: {
            // x[0] 0/1 = SetRenderEnable; 2/3 = SetRenderEnableConditional mode 0 (render if equal) / 1 (if not equal) on the
            // two counters at x[1] and x[1] + 16 (cmdbuf.c). Evaluated from guest memory at execution time (values written by
            // GPU work of the same submission may be stale). Applied by default (TotK clears its half-res lighting targets
            // once per frame this way; ignoring it wiped them = black cave, run252); LARECOMP_COND_RENDER=0 = ignore (old).
            static int apply = -1;
            if (apply < 0) apply = getenv("LARECOMP_COND_RENDER") ? atoi(getenv("LARECOMP_COND_RENDER")) : 1;
            if (r->x[0] < 2) { g_render_off = false; break; }
            const uint64_t *ctr = (const uint64_t *)(uintptr_t)r->x[1];
            bool equal = ctr && ctr[0] == ctr[2];
            bool render = r->x[0] == 2 ? equal : !equal;
            g_cond_evals++;
            if (!render) g_cond_off++;
            {   // per-address result pattern of each frame (bit i = eval i rendered); "[condflip]" when it differs from the
                // address's previous frame (fog popping in and out with camera moves, user 2026-10-15). Max 300 lines.
                static struct { uint64_t addr, frame; uint32_t pat, n, ppat, pn; bool prev; } ct[128];
                static int nct, nlog;
                uint64_t fr = renderer_present_count();
                int i = 0;
                while (i < nct && ct[i].addr != r->x[1]) i++;
                if (i == nct && nct < 128) { ct[nct] = (__typeof__(ct[0])){r->x[1], fr, 0, 0, 0, 0, false}; nct++; }
                if (i < nct) {
                    if (ct[i].frame != fr) {
                        uint32_t k = ct[i].n < ct[i].pn ? ct[i].n : ct[i].pn, m = k >= 32 ? ~0u : (1u << k) - 1;
                        if (ct[i].prev && ((ct[i].pat ^ ct[i].ppat) & m)) {
                            g_cond_flips++;
                            uint64_t bk = r->x[1] >= HEAP_BASE && ((r->x[1] - HEAP_BASE) >> GDEV_SH) < GDEV_BLKS ? (r->x[1] - HEAP_BASE) >> GDEV_SH : 0;
                            if (nlog++ < 300)
                                LOG("[condflip] addr %llx frame %llu: %u evals pattern %#x (prev frame %u evals %#x) | now %llx vs %llx mode %llu | shader-written blk seq %llu (submit %llu)\n",
                                    (unsigned long long)r->x[1], (unsigned long long)ct[i].frame, ct[i].n, ct[i].pat, ct[i].pn, ct[i].ppat,
                                    ctr ? (unsigned long long)ctr[0] : 0ull, ctr ? (unsigned long long)ctr[2] : 0ull,
                                    (unsigned long long)(r->x[0] - 2), (unsigned long long)gdev_blk[bk], (unsigned long long)submit_seq);
                        }
                        ct[i].ppat = ct[i].pat, ct[i].pn = ct[i].n, ct[i].prev = true;
                        ct[i].pat = 0, ct[i].n = 0, ct[i].frame = fr;
                    }
                    if (ct[i].n < 32) ct[i].pat |= (uint32_t)render << ct[i].n;
                    ct[i].n++;
                }
            }
            static int log_cond = -1;
            if (log_cond < 0) log_cond = getenv("LARECOMP_LOG_COND") != NULL;
            if (g_debug_frame || (log_cond && g_cond_evals < 400))
                LOG("[cond] frame %llu mode %llu addr %llx: %llx (ts %llx) vs %llx (ts %llx) -> %s%s\n",
                    (unsigned long long)renderer_present_count(), (unsigned long long)(r->x[0] - 2), (unsigned long long)r->x[1],
                    ctr ? (unsigned long long)ctr[0] : 0ull, ctr ? (unsigned long long)ctr[1] : 0ull,
                    ctr ? (unsigned long long)ctr[2] : 0ull, ctr ? (unsigned long long)ctr[3] : 0ull, render ? "render" : "SKIP",
                    apply ? "" : " (not applied)");
            g_render_off = apply && !render;
            break;
        }
        case CMD_DISPATCH: dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]); break;
        case CMD_DRAW_ARRAYS: draw(r, false); break;
        case CMD_DRAW_ELEMENTS: draw(r, true); break;
        case CMD_UPDATE_UNIFORM_BUFFER:
            memcpy((void *)(uintptr_t)(r->x[0] + r->x[2]), x, r->x[3]);
            guest_copy_flush();  // (cbuf copies shared within the submission are stale now)
            if (ubo_zc > 0) {    // (that buffer goes back to snapshots; earlier draws of this list may read the new data once)
                uint64_t lo = r->x[0], hi = r->x[0] + (r->x[1] > r->x[2] + r->x[3] ? r->x[1] : r->x[2] + r->x[3]);
                bool have = false;
                for (int i = 0; i < zc_nexcl; i++) if (lo >= zc_excl[i].lo && hi <= zc_excl[i].hi) { have = true; break; }
                if (!have && zc_nexcl < 64) {
                    zc_excl[zc_nexcl].lo = lo; zc_excl[zc_nexcl].hi = hi; zc_nexcl++;
                    LOG("[vk] inline UpdateUniformBuffer (%llx +%llu): buffer %llx-%llx stays snapshot (%d such)\n",
                        (unsigned long long)(r->x[0] + r->x[2]), (unsigned long long)r->x[3], (unsigned long long)lo,
                        (unsigned long long)hi, zc_nexcl);
                } else if (!have) {
                    ubo_zc = 0;
                    LOG("[vk] inline UpdateUniformBuffer: more than 64 buffers, uniform buffer zero-copy off\n");
                }
            }
            break;
        case CMD_FENCE_SYNC: {
            static int fence_flush = -1;
            if (fence_flush < 0) fence_flush = getenv("LARECOMP_FENCE_FLUSH") ? 1 : 0;
            if (fence_flush) renderer_flush();
            extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq);
            if (chunk_limit()) { vk_cmd(); g_fence_seq_mark = vk_pending_seq(); }  // (the copy-back happens in this submission)
            if (r->x[1]) nvn_sync_set_seq(r->x[1], renderer_fence_seq());
            break;
        }
        case CMD_BARRIER: {   // nvnCommandBufferBarrier: the next pass begins with a full barrier (see begin_pass)
            stat_game_barriers++;
            stat_bar_bits[r->x[0] & 0x1ff]++;
            if (in_pass) stat_bar_inpass++;
            // d27 OBSERVED: 316121 of 331k game barriers per 2 s carry no bits at all (no ordering, no invalidation: a no-op);
            // the rest are 0x12 (ORDER_FRAGMENTS | INVALIDATE_TEXTURE) ~120 per frame, 0x11 and 0x101 ~7 per frame.
            if (!(r->x[0] & 0x1ff)) break;
            game_bar_since_fb = true;
            if (barrier_mode() != BAR_GAME) break;
            pass_barrier_needed = true;
            pbn_cause = 1;
            if (in_pass && pass_unbarriered) {   // (this pass skipped its barrier: what it reads next must see the earlier passes)
                st.rt_dirty = true;
                stat_game_barrier_breaks++;
            }
            break;
        }
        case CMD_REPORT_COUNTER: {
            // The game times its GPU work with these (two per frame) and drops to 30 fps when that looks slow;
            // the value is our translation time, not GPU time. LARECOMP_VK_COUNTER=frozen: the same timestamp for
            // every counter within a presented frame (experiment).
            uint64_t *out = (uint64_t *)(uintptr_t)r->x[1];
            uint32_t type = (uint32_t)r->x[0];
            static uint32_t seen;
            if (type < 32 && !(seen & (1u << type))) { seen |= 1u << type; LOG("[nvn] ReportCounter type %u\n", type); }
            // LARECOMP_VK_COUNTER=scale:<k> (probe, 2026-10-18): the reported clock runs k times faster, so every interval
            // the game measures looks k times longer -> its dynamic resolution sees a 30-fps budget overrun at 90 fps (k 3).
            // TOTK_DYNRES=1 (Graphics menu "Dynamic resolution"): k = TOTK_FPS / 30, so the game's resolution scaler aims at
            // the chosen frame rate. d76_k3 (OBSERVED, Deck 1400 MHz, TOTK_FPS=90): 3D 1024x576 in the heavy standing scene,
            // 90.0 fps in all 55 windows (native: 84.5), HUD unscaled, game speed 1.0x (Cold Resistance timer).
            static int frozen = -1;
            static double cscale = 1;
            if (frozen < 0) {
                const char *e = getenv("LARECOMP_VK_COUNTER"), *dr = getenv("TOTK_DYNRES"), *fp = getenv("TOTK_FPS");
                frozen = e && !strcmp(e, "frozen");
                if (e && !strncmp(e, "scale:", 6) && atof(e + 6) > 0) cscale = atof(e + 6);
                else if (dr && atoi(dr) > 0 && fp && atoi(fp) > 30) cscale = atoi(fp) / 30.0;
                if (cscale != 1) LOG("[vk] counter clock x%.2f (dynamic resolution aims at %.0f fps)\n", cscale, 30 * cscale);
            }
            static uint64_t cbase;
            if (!cbase) cbase = SDL_GetTicksNS();
            static uint64_t frame_seen = ~0ull, frame_ts;
            extern uint64_t vk_frame_counter(void);
            if (frozen && vk_frame_counter() != frame_seen) { frame_seen = vk_frame_counter(); frame_ts = SDL_GetTicksNS(); }
            // type 1 = SAMPLES_PASSED: TotK resets it (nvnCommandBufferResetCounter(1)) and reads it back as an occlusion
            // query. Reporting 0 made every queried light / flame "occluded": Zelda's torch lit the hall only while the
            // camera was close to it (user 2026-10-14 22:40). Until real Vulkan occlusion queries exist, report "visible":
            // a count that grows by 4096 per report (so both "since reset" and "end - begin" reads are > 0).
            // LARECOMP_OCCLUSION=zero = the old behaviour (A/B).
            static int occl_zero = -1;
            if (occl_zero < 0) { const char *e = getenv("LARECOMP_OCCLUSION"); occl_zero = e && !strcmp(e, "zero"); }
            static uint64_t samples;
            uint64_t v = 0;
            if (type == 1 && !occl_zero) v = samples += 4096;
            // LARECOMP_COUNTER_TYPES=<t>[,<t>...] (probe): scale only those counter types (TotK reports 0, 1 and 16). d82
            // (OBSERVED): with the x3 clock, 64 frames >= 14 ms in the light stretch (spaced in multiples of 60 frames,
            // "other" = game threads +4 ms, 0 images, no resolution change) vs 18 unscaled: which reads cause them?
            // d83 (OBSERVED, walk, light-stretch frames >= 14 ms after f5600): all types 64, type 0 only 31 (heavy stretch
            // 90.0, same [drs] steps = type 0 is the resolution scaler's clock), type 16 only 34 (heavy 84.0: no DRS),
            // none 18. So TOTK_DYNRES scales type 0 only; the explicit scale:<k> probe keeps scaling all of them.
            static uint32_t tmask;
            if (!tmask) {
                const char *e = getenv("LARECOMP_COUNTER_TYPES"), *ve = getenv("LARECOMP_VK_COUNTER");
                tmask = e ? 0x80000000u : ve && !strncmp(ve, "scale:", 6) ? ~0u : 0x80000001u;
                for (const char *p = e; p && *p;) { long t = strtol(p, (char **)&p, 10); if (t >= 0 && t < 31) tmask |= 1u << t; if (*p == ',') p++; else break; }
            }
            double k = type < 31 && (tmask & (1u << type)) ? cscale : 1;
            if (out) { out[0] = v; out[1] = frozen ? frame_ts : cbase + (uint64_t)((double)(SDL_GetTicksNS() - cbase) * k); }
            if (g_debug_frame) LOG("[counter] type %u -> %llx value %llu\n", type, (unsigned long long)r->x[1], (unsigned long long)v);
            break;
        }
        case CMD_COPY_TEXTURE_TO_TEXTURE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], r->x[6]); break; }
        case CMD_DOWNSAMPLE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], 1); break; }
        case CMD_COPY_BUFFER_TO_TEXTURE: copy_buffer_to_texture(r->x[0], x); break;
        case CMD_COPY_TEXTURE_TO_BUFFER: copy_texture_to_buffer(x, r->x[3]); break;
        case CMD_COPY_BUFFER_TO_BUFFER:
            if (g_debug_frame) LOG("[bufcopy] %llx -> %llx size %llu\n", (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2]);
            {   // LARECOMP_TEX_WATCH_FMT: memory copies into / out of watched textures' storage
                extern bool tex_watched_range(uint64_t a, uint64_t n);
                extern uint64_t vk_frame_counter(void);
                static int nl;
                if (nl < 600 && (tex_watched_range(r->x[1], r->x[2]) || tex_watched_range(r->x[0], r->x[2]))) {
                    nl++;
                    LOG("[texw] bufcopy %llx -> %llx size %llu frame %llu\n", (unsigned long long)r->x[0], (unsigned long long)r->x[1],
                        (unsigned long long)r->x[2], (unsigned long long)vk_frame_counter());
                }
            }
            renderer_flush();
            memmove((void *)(uintptr_t)r->x[1], (void *)(uintptr_t)r->x[0], r->x[2]);
            break;
        default: {   // (not executed: logged once per op, and every time in a debug frame)
            static uint64_t seen[CMD_COUNT];
            if (r->op < CMD_COUNT && (!seen[r->op]++ || g_debug_frame))
                LOG("[cmd] op %u not executed (%llu so far): x %llx %llx %llx %llx %llx\n", r->op, (unsigned long long)seen[r->op],
                    (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2], (unsigned long long)r->x[3],
                    (unsigned long long)r->x[4]);
            break;
        }
        }
        p += cmd_gsize(r->size);
    }
}

static pthread_mutex_t exec_mu = PTHREAD_MUTEX_INITIALIZER;
void backend_exec_lock(void) { pthread_mutex_lock(&exec_mu); }
void backend_exec_unlock(void) { pthread_mutex_unlock(&exec_mu); }

static void backend_execute_now(uint64_t handle) {
    PROF_T0();
    uint64_t hx0 = SDL_GetTicksNS();   // ([hitch] log, always on)
#ifndef _WIN32
    // on-CPU time vs wall time of exec: a slow exec that ran the whole time = slower CPU (SMT sibling, clock); one that
    // did not = preempted (d86: every op 1.7-3x slower on the periodic hitch frames)
    struct timespec c0; struct rusage r0;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0); getrusage(RUSAGE_THREAD, &r0);
#else
    ULONG64 cy0 = 0;   // (Windows, VK_PROF only: thread cycles in TSC ticks, scaled by prof_now's calibration)
    if (prof) QueryThreadCycleTime(GetCurrentThread(), &cy0);
#endif
    backend_exec_lock();
    alias_epoch++;
    exec_start_seq = vk_pending_seq();
    if (chunk_ab > 0) {
        extern uint64_t renderer_present_count(void);
        chunk_ab_on = (renderer_present_count() / (uint64_t)chunk_ab) % 2 == 0;
    }
    in_execute = true;
    exec_list(handle, 0);
    vk_writeback_flush();
    in_execute = false;
    backend_exec_unlock();
    if (prof && op_t0) { op_ns[op_last] += prof_now() - op_t0; op_t0 = 0; }
    PROF_ADD(prof_exec);
    cum_exec += SDL_GetTicksNS() - hx0;
#ifndef _WIN32
    struct timespec c1; struct rusage r1;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1); getrusage(RUSAGE_THREAD, &r1);
    cum_exec_cpu += (uint64_t)((c1.tv_sec - c0.tv_sec) * 1000000000ll + (c1.tv_nsec - c0.tv_nsec));
    cum_exec_ivcsw += (uint64_t)(r1.ru_nivcsw - r0.ru_nivcsw);
#else
    ULONG64 cy1 = 0;
    if (prof) { QueryThreadCycleTime(GetCurrentThread(), &cy1); cum_exec_cpu += (uint64_t)((double)(cy1 - cy0) * prof_ns_per_tick); }
#endif
    static int sync_submit = -1;
    if (sync_submit < 0) sync_submit = getenv("LARECOMP_SYNC_SUBMIT") ? 1 : 0;
    if (sync_submit) renderer_flush();
    else renderer_submit();
}

// ---- async execute. The game's Presentation Thread called nvnQueueSubmitCommands and we translated + submitted the
// lists inside that call (run378 at 100 s: that thread 71 % busy, 68 % of it in submit, while the main thread waited
// on it 46 % of the time). LARECOMP_ASYNC_EXEC=1: submits, fence syncs, presents and the frees of objects that queued
// lists name by host pointer (cmdbuf.c: render targets, copy textures, programs) go into one ordered queue drained by
// a render thread; the game's thread returns at once, as with a real GPU. nvnSyncWait / nvnQueueFinish wait for the
// queue to pass their fence first. Presents run at most LARECOMP_ASYNC_AHEAD (default 1) frames behind the game.
// The queue grows instead of blocking (finalizers push while holding the exec lock the render thread needs).
enum { AOP_EXEC = 1, AOP_FENCE, AOP_PRESENT, AOP_CALL };
typedef struct { int kind; uint64_t a; void *p; void (*fn)(void *); uint64_t t; } AsyncOp;   // (t: push time, ns)
uint64_t g_op_push_ns;   // (render thread: push time of the op it is running)
static AsyncOp *aq;
static uint64_t aq_cap, aq_head, aq_tail, aq_done, aq_pres_pushed, aq_pres_done;   // head/tail: monotonic op indices
static pthread_mutex_t aq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t aq_cv = PTHREAD_COND_INITIALIZER, aq_done_cv = PTHREAD_COND_INITIALIZER;
static bool aq_started;
static uint64_t aq_stat_ops, aq_stat_wait_ns, aq_stat_waits, aq_stat_pres_wait_ns, aq_stat_maxdepth;
bool backend_async_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_ASYNC_EXEC"); on = e && *e == '1'; if (on) LOG("[async] render thread on\n"); }
    return on;
}
static void *async_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    extern void rt_prof_register_host(const char *name);
    rt_prof_register_host("render");   // (TOTK_PROF samples this thread; hostprof.py --hle render)
#ifndef _WIN32
    extern int pthread_setname_np(pthread_t, const char *);
    pthread_setname_np(pthread_self(), "render");   // (Linux "[cpu]" per-thread line)
    extern void host_affinity(int kind, unsigned long long mask);
    host_affinity(1, 0);   // (TOTK_AFFINITY: render side on its own CPUs)
#endif
    for (;;) {
        pthread_mutex_lock(&aq_mu);
        while (aq_head == aq_tail) pthread_cond_wait(&aq_cv, &aq_mu);
        AsyncOp op = aq[aq_head & (aq_cap - 1)];
        pthread_mutex_unlock(&aq_mu);
        g_op_push_ns = op.t;
        switch (op.kind) {
        case AOP_EXEC: backend_execute_now(op.a); break;
        case AOP_FENCE: {
            NvnSync *s = (NvnSync *)op.p;
            renderer_fence_submit();
            uint64_t q = renderer_fence_seq();
            s->prev_seq = s->rt_seq;
            s->rt_seq = q;
            s->seq = q;
            s->signaled = 1;
            break;
        }
        case AOP_PRESENT: backend_present((NvnTexture *)op.p); break;
        case AOP_CALL: backend_exec_lock(); op.fn(op.p); backend_exec_unlock(); break;
        }
        pthread_mutex_lock(&aq_mu);
        aq_head++;
        aq_done = aq_head;
        aq_stat_ops++;
        if (op.kind == AOP_PRESENT) aq_pres_done++;
        pthread_cond_broadcast(&aq_done_cv);
        pthread_mutex_unlock(&aq_mu);
    }
    return NULL;
}
static uint64_t aq_push(AsyncOp op) {
    op.t = SDL_GetTicksNS();
    pthread_mutex_lock(&aq_mu);
    if (!aq_started) {
        aq_started = true;
        aq_cap = 1024;
        aq = calloc(aq_cap, sizeof *aq);
        pthread_t th;
        pthread_create(&th, NULL, async_main, NULL);
        pthread_detach(th);
    }
    if (aq_tail - aq_head == aq_cap) {   // grow: same logical indices in a twice-as-large ring
        AsyncOp *n = calloc(aq_cap * 2, sizeof *n);
        for (uint64_t i = aq_head; i < aq_tail; i++) n[i & (aq_cap * 2 - 1)] = aq[i & (aq_cap - 1)];
        free(aq);
        aq = n;
        aq_cap *= 2;
    }
    aq[aq_tail & (aq_cap - 1)] = op;
    uint64_t ticket = ++aq_tail;
    if (aq_tail - aq_head > aq_stat_maxdepth) aq_stat_maxdepth = aq_tail - aq_head;
    if (op.kind == AOP_PRESENT) aq_pres_pushed++;
    pthread_cond_signal(&aq_cv);
    pthread_mutex_unlock(&aq_mu);
    return ticket;
}
void backend_async_wait(uint64_t ticket) {
    if (!backend_async_on()) return;
    uint64_t t0 = SDL_GetTicksNS();
    pthread_mutex_lock(&aq_mu);
    if (!ticket) ticket = aq_tail;
    bool waited = aq_done < ticket;
    while (aq_done < ticket) pthread_cond_wait(&aq_done_cv, &aq_mu);
    if (waited) { aq_stat_waits++; aq_stat_wait_ns += SDL_GetTicksNS() - t0; }
    pthread_mutex_unlock(&aq_mu);
}
void backend_execute(uint64_t handle) {
    if (backend_async_on()) aq_push((AsyncOp){AOP_EXEC, handle, NULL, NULL});
    else backend_execute_now(handle);
}
void backend_async_call(void (*fn)(void *), void *arg) {
    if (backend_async_on()) aq_push((AsyncOp){AOP_CALL, 0, arg, fn});
    else { backend_exec_lock(); fn(arg); backend_exec_unlock(); }
}
void backend_async_fence(NvnSync *s) {
    s->signaled = 1;
    s->seq = 0;
    s->ticket = aq_push((AsyncOp){AOP_FENCE, 0, s, NULL});
}
void backend_async_present(NvnTexture *t) {
    static int ahead = -1;
    if (ahead < 0) { const char *e = getenv("LARECOMP_ASYNC_AHEAD"); ahead = e ? atoi(e) : 1; }
    aq_push((AsyncOp){AOP_PRESENT, 0, t, NULL});
    uint64_t t0 = SDL_GetTicksNS();
    pthread_mutex_lock(&aq_mu);
    while (aq_pres_pushed - aq_pres_done > (uint64_t)ahead) pthread_cond_wait(&aq_done_cv, &aq_mu);
    aq_stat_pres_wait_ns += SDL_GetTicksNS() - t0;
    static uint64_t last_log;
    if (t0 - last_log > 5000000000ull) {   // every 5 s: game-side waits on the render thread
        if (last_log)
            LOG("[async] 5 s: %llu ops (max queue %llu), present throttle %.0f ms, sync/finish waits %llu = %.0f ms\n",
                (unsigned long long)aq_stat_ops, (unsigned long long)aq_stat_maxdepth, aq_stat_pres_wait_ns / 1e6,
                (unsigned long long)aq_stat_waits, aq_stat_wait_ns / 1e6);
        last_log = t0;
        aq_stat_ops = aq_stat_maxdepth = aq_stat_pres_wait_ns = aq_stat_waits = aq_stat_wait_ns = 0;
    }
    pthread_mutex_unlock(&aq_mu);
}

#include "vkq_impl.h"   // (deferred command recording, LARECOMP_VKQ=1)

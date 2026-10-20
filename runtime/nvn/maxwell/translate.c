// Maxwell (SM 5.x) shader binary -> Metal Shading Language, or Vulkan GLSL (key->glsl; contract in shader.h).
//
// The binary is an NVN header (0x30), the shader program header (SPH, 0x50; graphics stages), then
// instructions in groups of four 64-bit words (one scheduling word + three instructions).
// Registers are emitted as `uint` variables; float ops use as_type<> conversions. Shaders with
// branches become a `switch (pc)` state machine with explicit SSY/PBK/PCNT stacks.
//
// GLSL: the instruction body is generated in the MSL spelling and converted by glsl_rewrite() (a token-aware
// pass: as_type<T>() -> as_u/as_f overloads, float4 -> vec4, ulong -> uint64_t, char()/short() casts -> sx8/sx16,
// in.x / out.x stage I/O -> GLSL globals and built-ins), helpers with MSL names (saturate, fmin, clz, mulhi, ...)
// are defined in the GLSL prelude, and the structurally different constructs (texture lookups, global memory,
// half-float packing, subgroup shuffles, barriers, discard, declarations and entry point) are emitted per dialect.
// The MSL text is byte-identical to the MSL-only translator (tools/test_translate.c checks it).
#if defined(_WIN32)
#include <malloc.h>  // alloca
#endif
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shader.h"

MAXWELL_TLS uint64_t maxwell_gmem_base, maxwell_gmem_size, maxwell_gdev_size;
// The program's compiler constants (c1) while translating it, when the caller has them (exec.c translate_program: the live
// program; the archive precompile has only the code): BRX jump tables are read from them (brx_targets). NULL: every
// instruction may be a BRX target.
MAXWELL_TLS const uint8_t *maxwell_const_data;
MAXWELL_TLS size_t maxwell_const_size;
// Changes whenever this file is recompiled (= the translator may emit different GLSL): the archive warm-up record
// (exec.c pa_warm_*) is only trusted for the translator build that wrote it.
const char maxwell_build_stamp[] = __DATE__ " " __TIME__;
const uint8_t *maxwell_dbg_code;
uint32_t maxwell_dbg_pc;
// LARECOMP_GUARD_STAGE=v|f|c (with LARECOMP_GUARD_LIMIT): the limit applies to that stage only (A/B: which stage loops)
static unsigned long guard_limit_stage(int stage) {
    static unsigned long g;
    static int only = -2;
    if (!g) { const char *e = getenv("LARECOMP_GUARD_LIMIT"); g = e && atol(e) > 0 ? (unsigned long)atol(e) : 100000ul; }
    if (only == -2) {
        const char *e = getenv("LARECOMP_GUARD_STAGE");
        only = !e ? -1 : *e == 'v' ? 0 : *e == 'f' ? 1 : *e == 'c' ? 5 : -1;
    }
    return only >= 0 && stage != only ? 100000ul : g;
}
#define guard_limit() guard_limit_stage(stage)
int maxwell_dbg_reg;
const uint8_t *maxwell_dbg_vs;

#define RZ 255
#define LMEM_WORDS 256  // local memory per invocation (1 KB)
#define PT 7

typedef struct {
    char *buf;
    size_t len, cap;
} Str;

static void sp(Str *s, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (s->len + n + 1 > s->cap) {
        s->cap = (s->len + n + 1) * 2;
        s->buf = realloc(s->buf, s->cap);
    }
    vsnprintf(s->buf + s->len, n + 1, fmt, ap2);
    va_end(ap2);
    s->len += n;
}

typedef struct {
    const uint8_t *code;   // first instruction group
    size_t ninsts;         // number of 8-byte words to scan
    int stage;
    const ShaderKey *key;
    ShaderInfo *info;
    Str body;
    uint8_t reg_used[256];
    uint8_t pred_used[8];
    uint32_t in_attr_mask, out_attr_mask;
    uint32_t vs_in_comp[32];
    bool uses_branches, uses_cc, uses_local, uses_vid, uses_iid, uses_frag_coord, uses_front_facing, uses_lane;
    bool uses_rscale;  // resolution-scale table (fragment/compute only)
    uint32_t flat_mask;           // fragment inputs using constant (flat) interpolation
    uint32_t noperspective_mask;  // fragment inputs using pass (linear) interpolation
    uint32_t omap_target, omap_depth;
    int tex_slot[8192];           // texture index -> Metal slot + 1
    bool tex_weak[64];            // slot type only guessed (from a size query): the next sample decides
    bool fail;
    bool branchy;  // program uses the pc state machine
    int ipa_mode;  // interpolation mode of the IPA being translated (-1 for plain loads)
    int ipa_mult;  // its multiplier register (IPA.MUL)
    // GLSL (fields kept last: the MSL path's layout, including its out-of-range reg_used writes, is unchanged)
    bool glsl;               // emit Vulkan GLSL: dialect-specific constructs here, the rest via glsl_rewrite()
    bool uses_gmem_store;    // STG present: the gmem buffer is not readonly
    bool uses_shared;        // TotK: STS/LDS present (compute): shared uint smem[] (Maxwell max 48 KiB)
    bool uses_gdev;          // a gmem access also routes to the device window (gdev_)
    uint32_t cbuf_need[18];  // highest constant vec4 index + 1 per cbuf (array size of the GLSL block)
    bool trace;       // LARECOMP_DEBUG_SHADER: printf every register write at one pixel
    bool solid;       // LARECOMP_DEBUG_SOLID=<fs hash hex>: that fragment shader writes magenta to c0 and never discards
    bool ssfetch;     // maxwell_ssfetch_next: 2D texel fetches at mip 0 average their guest texel's scaled footprint
    uint32_t ssf_slots;   // texture slots those fetches use (each gets an ssf<slot>() helper)
    bool nodiscard;   // LARECOMP_NO_DISCARD (=1: every shader, =<fs hash hex>: that fragment shader): KIL emits nothing
    bool kil_demote;  // a KIL became GLSL demote (needs GL_EXT_demote_to_helper_invocation in this shader's header)
    bool minpc;       // GLSL fragment pc loop runs only the lanes at the subgroup's lowest pc (GL_KHR_shader_subgroup_arithmetic)
    bool use_imap;    // fragment: SPH input map read (LARECOMP_NO_IMAP=1: every input perspective, the old behaviour)
    uint8_t imap[32]; // SPH ps.imap_generic_vector: 2 bits per component, 1 constant, 2 perspective, 3 screen-linear
    uint32_t cur_pc;
    bool flat;            // forward-only branches: blocks in address order as flag-guarded ifs, no pc loop (flat_plan)
    uint32_t *flat_sync;  // flat: the static target (pc) of the SYNC in each slot (0: unreachable)
    uint32_t dag_guard;   // pc loop without backward branches: its iteration cap (2 x labels + 16); 0 = guard_limit()
    uint32_t dag_why;     // (no DAG guard) the pc of the first backward branch / loop / call op + 1 (GLSL comment)
} Ctx;

// GLSL global access lines (using ga, ok and gmem_.w) as emitted into a scratch Str: append them to the body, and
// with the device window on, a second copy on gdev_ for addresses past the heap (ga rebased, ok = true there).
static void gdev_split(Ctx *c, Str *lines) {
    const char *l = lines->buf ? lines->buf : "";
    if (!maxwell_gdev_size) { sp(&c->body, "%s", l); free(lines->buf); return; }
    c->uses_gdev = true;
    sp(&c->body, "      if (!(ga >= 0x%llxul && ga - 0x%llxul < 0x%llxul)) {\n%s      } else { ga -= 0x%llxul; ok = true;\n",
       (unsigned long long)maxwell_gmem_size, (unsigned long long)maxwell_gmem_size, (unsigned long long)maxwell_gdev_size, l,
       (unsigned long long)maxwell_gmem_size);
    for (const char *p = l; *p;) {
        const char *q = strstr(p, "gmem_.");
        if (!q) { sp(&c->body, "%s", p); break; }
        sp(&c->body, "%.*sgdev_.", (int)(q - p), p);
        p = q + 6;
    }
    sp(&c->body, "      }\n");
    free(lines->buf);
}

static uint64_t bits(uint64_t w, int lo, int n) { return (w >> lo) & ((n == 64) ? ~0ull : ((1ull << n) - 1)); }
static int64_t sbits(uint64_t w, int lo, int n) {
    int64_t v = (int64_t)bits(w, lo, n);
    return (v & (1ll << (n - 1))) ? v - (1ll << n) : v;
}

static void fail(Ctx *c, const char *fmt, ...) {
    if (c->fail) return;
    c->fail = true;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->info->error, sizeof c->info->error, fmt, ap);
    va_end(ap);
}

// Branch targets may point at a scheduling word; execution continues at the next instruction.
static uint32_t norm_target(int64_t t) {
    if (t >= 0 && ((t / 8) % 4) == 0) t += 8;
    return (uint32_t)t;
}

// ------------------------------------------------------------------ operands
static const char *reg(Ctx *c, int r, char *buf) {
    if (r == RZ) return "0u";
    c->reg_used[r] = 1;
    sprintf(buf, "r%d", r);
    return buf;
}

#define R(r) reg(c, (r), alloca(16))
#define F(expr) fmt_f(alloca(512), (expr))
static char *fmt_f(char *b, const char *e) { sprintf(b, "as_type<float>(%s)", e); return b; }

static const char *pred(Ctx *c, int p, bool neg, char *buf) {
    if (p == PT) return neg ? "false" : "true";
    c->pred_used[p] = 1;
    sprintf(buf, "%sp%d", neg ? "!" : "", p);
    return buf;
}
#define P(p, n) pred(c, (p), (n), alloca(16))

static const char *cbuf_ref(Ctx *c, int idx, int64_t off, const char *dyn, char *buf) {
    c->info->cbuf_mask |= 1u << idx;
    if (off + 16 > (int64_t)c->info->cbuf_size[idx]) c->info->cbuf_size[idx] = (uint32_t)((off + 16 + 255) & ~255);
    if (dyn) c->info->cbuf_dynamic |= 1u << idx;
    if (!dyn && idx < 18 && off >= 0 && (uint32_t)(off >> 4) + 1 > c->cbuf_need[idx]) c->cbuf_need[idx] = (uint32_t)(off >> 4) + 1;
    if (dyn && off < 0) sprintf(buf, "cbr(cb%d, (%s) - %lldu)", idx, dyn, -(long long)off);
    else if (dyn) sprintf(buf, "cbr(cb%d, (%s) + %lldu)", idx, dyn, (long long)off);
    else sprintf(buf, "cb%d[%lld][%lld]", idx, (long long)(off >> 4), (long long)((off >> 2) & 3));
    return buf;
}

// Operand B in its register / constant buffer / immediate forms, selected by the opcode class.
typedef enum { B_REG, B_CBUF, B_IMM } BForm;

static BForm b_form(uint64_t w) {
    uint64_t top = w >> 60;
    if (top == 0x5) return B_REG;
    if (top == 0x4) return B_CBUF;
    return B_IMM;
}

static const char *operand_b(Ctx *c, uint64_t w, bool float_imm, char *buf) {
    switch (b_form(w)) {
    case B_REG: return reg(c, (int)bits(w, 20, 8), buf);
    case B_CBUF: return cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, buf);
    default: {
        uint32_t v = (uint32_t)bits(w, 20, 19) | ((uint32_t)bits(w, 56, 1) << 19);
        if (float_imm) v <<= 12;
        else if (v & 0x80000u) v |= 0xFFF00000u;  // sign-extend 20 bits
        sprintf(buf, "0x%08xu", v);
        return buf;
    }
    }
}
#define B(fi) operand_b(c, w, (fi), alloca(128))

// The "C" operand of three-source ops: REG_39, or the cbuf/reg swap form (0x51/0x53/... classes).
static const char *operand_c_for_ffma(Ctx *c, uint64_t w, char *buf, const char **b_out, char *bbuf) {
    uint64_t op8 = w >> 56;
    if ((op8 & 0xF8) == 0x50 && (op8 & 7) != 0) {  // e.g. 0x51: B = REG_39, C = cbuf
        *b_out = reg(c, (int)bits(w, 39, 8), bbuf);
        return cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, buf);
    }
    *b_out = operand_b(c, w, true, bbuf);
    return reg(c, (int)bits(w, 39, 8), buf);
}

static void set_reg(Ctx *c, int r, const char *expr) {
    if (r == RZ) return;
    c->reg_used[r] = 1;
    sp(&c->body, "    r%d = %s;\n", r, expr);
    if (c->trace) sp(&c->body, "    if (trace_on) debugPrintfEXT(\"[trace] %%x r%%u=%%08x\", %uu, %du, r%d);\n", c->cur_pc, r, r);
}

static const char *fneg_abs(const char *e, bool neg, bool abs_, char *buf) {
    if (abs_ && neg) sprintf(buf, "(-abs(%s))", e);
    else if (abs_) sprintf(buf, "abs(%s)", e);
    else if (neg) sprintf(buf, "(-%s)", e);
    else return e;
    return buf;
}
#define NA(e, n, a) fneg_abs((e), (n), (a), alloca(600))

static const char *fresult(bool glsl, const char *e, bool sat, char *buf) {
    // GLSL: saturate as a select (no function call; see FMNMX), e is pure so it may appear twice
    if (sat && glsl && 2 * strlen(e) + 80 < 700) sprintf(buf, "as_type<uint>(float(mix(clamp(%s, 0.0f, 1.0f), 0.0f, isnan(%s))))", e, e);
    else if (sat) sprintf(buf, "as_type<uint>(saturate(%s))", e);
    else sprintf(buf, "as_type<uint>(%s)", e);
    return buf;
}
#define FRES(e, sat) fresult(c->glsl, (e), (sat), alloca(700))

// FADD / FMUL / FFMA results through a `precise` temporary (Eden: these ops carry no_contraction and its GLSL backend
// emits them precise). The guest code is already compiled: the host compiler must not reassociate or fuse across its
// rounding steps. TotK's deferred lighting computes 1 / ((1 - x) + 1e-10); NVIDIA's driver reassociated it to
// (1 + 1e-10) - x = 0 for x = 1 -> 1 / 0 -> 0 * inf = NaN (white sparkles on metal; run120 shader trace, pc f88/f98/1028).
// One `precise float pf_` per shader (declared in main) instead of a new block + variable per op: same NoContraction
// decorations, OpVariables per stage 472 -> 223 (Deck, 120 programs, 2026-10-20: RADV 32.0 -> 33.8 programs/s, 1 thread).
static void set_reg_precise(Ctx *c, int d, const char *e, bool sat) {
    if (d == RZ) return;
    if (!c->glsl) { set_reg(c, d, FRES(e, sat)); return; }
    sp(&c->body, "    pf_ = %s;\n", e);
    set_reg(c, d, sat ? "floatBitsToUint(mix(clamp(pf_, 0.0f, 1.0f), 0.0f, isnan(pf_)))" : "floatBitsToUint(pf_)");
}

// F64 (DADD / DMUL / DFMA / F2F.F64): a double lives in a register pair (low word in the even register). 32 'material'
// vertex shaders compute positions in double precision (cbuf floats -> F2F.F64, DFMA/DMUL/DADD, F2F.F32); they failed
// to translate and their draws were dropped (grass missing, user 2026-10-15). GLSL only (Vulkan shaderFloat64).
// Operand forms as Eden (GetDoubleReg20 / GetDoubleCbuf / GetDoubleImm20).
static const char *dreg(Ctx *c, int r, char *buf) {
    if (r == RZ) return "packDouble2x32(uint2(0u, 0u))";
    c->reg_used[r] = c->reg_used[r + 1] = 1;
    sprintf(buf, "packDouble2x32(uint2(r%d, r%d))", r, r + 1);
    return buf;
}
static const char *dcbuf(Ctx *c, uint64_t w, char *buf) {
    int idx = (int)bits(w, 34, 5);
    int64_t off = bits(w, 20, 14) * 4;
    char lo[128], hi[128];
    if (off & 4) {  // unaligned: high word from the given offset, low word 0
        strcpy(lo, "0u");
        cbuf_ref(c, idx, off, NULL, hi);
    } else {
        cbuf_ref(c, idx, off, NULL, lo);
        cbuf_ref(c, idx, off + 4, NULL, hi);
    }
    sprintf(buf, "packDouble2x32(uint2(%s, %s))", lo, hi);
    return buf;
}
static const char *dsrc_b(Ctx *c, uint64_t w, char *buf) {
    switch (b_form(w)) {
    case B_REG: return dreg(c, (int)bits(w, 20, 8), buf);
    case B_CBUF: return dcbuf(c, w, buf);
    default: {
        uint32_t hi = ((uint32_t)bits(w, 20, 19) << 12) | ((uint32_t)bits(w, 56, 1) << 31);
        sprintf(buf, "packDouble2x32(uint2(0u, 0x%08xu))", hi);
        return buf;
    }
    }
}
#define DR(r) dreg(c, (r), alloca(64))
#define DB() dsrc_b(c, w, alloca(320))
static void set_dreg(Ctx *c, int d, const char *e) {
    if (d == RZ) return;
    c->reg_used[d] = c->reg_used[d + 1] = 1;
    sp(&c->body, "    { precise double pd_ = %s; uint2 dv_ = unpackDouble2x32(pd_); r%d = dv_.x; r%d = dv_.y; }\n", e, d, d + 1);
}

static const char *cmp_op(int op) {
    static const char *const ops[] = {"false", "<", "==", "<=", ">", "!=", ">=", "num", "nan",
                                      "<", "==", "<=", ">", "!=", ">=", "true"};
    return ops[op & 15];
}

// float comparison including unordered variants
static void fcmp_expr(char *out, int op, const char *a, const char *b) {
    switch (op) {
    case 0: strcpy(out, "false"); break;
    case 15: strcpy(out, "true"); break;
    case 7: sprintf(out, "(!isnan(%s) && !isnan(%s))", a, b); break;
    case 8: sprintf(out, "(isnan(%s) || isnan(%s))", a, b); break;
    default:
        if (op < 7) sprintf(out, "(%s %s %s)", a, cmp_op(op), b);
        else sprintf(out, "((%s %s %s) || isnan(%s) || isnan(%s))", a, cmp_op(op), b, a, b);
    }
}

static const char *bool_op(int op) { return op == 0 ? "&&" : op == 1 ? "||" : "!="; }

static void icmp_expr(char *out, int op, const char *a, const char *b, bool is_signed) {
    static const char *const ops[] = {"false", "<", "==", "<=", ">", "!=", ">=", "true"};
    if (op == 0) strcpy(out, "false");
    else if (op == 7) strcpy(out, "true");
    else if (is_signed) sprintf(out, "(int(%s) %s int(%s))", a, ops[op], b);
    else sprintf(out, "(%s %s %s)", a, ops[op], b);
}

// ------------------------------------------------------------------ attributes
// Maxwell attribute space: 0x70 position, 0x80 + 16*i generic attribute i, 0x2F8 instance id,
// 0x2FC vertex id, 0x3FC front facing, 0x6C point size.
static void attr_load(Ctx *c, int dst, uint32_t addr, const char *idx) {
    char e[256];
    if (idx) { fail(c, "indexed attribute load"); return; }
    if (c->stage == STAGE_VERTEX) {
        if (addr >= 0x80 && addr < 0x280) {
            int a = (addr - 0x80) >> 4, comp = (addr >> 2) & 3;
            c->in_attr_mask |= 1u << a;
            c->vs_in_comp[a] |= 1u << comp;
            int kind = c->key ? c->key->attr_kind[a] : 0;
            // integer formats (kind 1 int4, 2 uint4): uint() keeps the bits; GLSL floatBitsToUint(ivec4 comp) would convert
            // the value (v1 builtin casts: skinned meshes stretched into spikes, Deck 2026-10-20)
            if (kind == 0) sprintf(e, "as_type<uint>(in.a%d[%d])", a, comp);
            else sprintf(e, "uint(in.a%d[%d])", a, comp);
        } else if (addr == 0x2F8) { c->uses_iid = true; strcpy(e, "iid"); }
        else if (addr == 0x2FC) { c->uses_vid = true; strcpy(e, "vid"); }
        else { sprintf(e, "0u /* attr %#x */", addr); }
    } else if (c->stage == STAGE_FRAGMENT) {
        if (addr >= 0x70 && addr < 0x80) {
            c->uses_frag_coord = true;
            int comp = (addr >> 2) & 3;
            if (comp < 2) {  // guest pixel units: the render target may be allocated larger
                c->uses_rscale = true;
                sprintf(e, "as_type<uint>((in.pos[%d] - rscale[0][%d]) / rscale[0][%d])", comp, comp + 2, comp);
            } else {
                sprintf(e, "as_type<uint>(in.pos[%d])", comp);
            }
        } else if (addr >= 0x80 && addr < 0x280) {
            int a = (addr - 0x80) >> 4, comp = (addr >> 2) & 3;
            // (MSL: a mask of 0 means unknown. GLSL: the renderer always passes the vertex shader's outputs, and
            // Vulkan requires every fragment input to be a vertex output, so 0 means none.)
            if (c->key && (c->key->vs_out_mask || c->key->glsl) && !(c->key->vs_out_mask & (1u << a))) {
                strcpy(e, comp == 3 ? "0x3f800000u" : "0u");  // not provided by the vertex shader
            } else {
                c->in_attr_mask |= 1u << a;
                // IPA "pass" yields attr/w (the shader multiplies by w itself); Metal gives attr.
                // IPA "multiply" yields attr/w * Rmult (Eden). Shaders usually pass Rmult = 1/IPA(position.w) = w,
                // giving attr: that exact case keeps the plain attribute (bit-identical); any other multiplier gets the
                // full product. TotK's soft particles read screen UVs as v.xy * (w / v.w) this way (torch flame faded
                // to 0 at distance: its depth lookup used undivided clip coordinates). LARECOMP_IPA_MUL_FOLD=1: old.
                // The SPH input map says how each component is interpolated (Eden ipa.cpp): only perspective ones get
                // the * w; screen-linear ones are noperspective inputs read as is, constant ones flat. (TotK's point-light
                // volumes pass their view ray screen-linear: with the * w the reconstructed position was wrong and the
                // torch light on the floor faded out with distance.)
                int im = c->use_imap ? (c->imap[a] >> (comp * 2)) & 3 : 2;
                if (im == 3) c->noperspective_mask |= 1u << a;
                if (im == 1) c->flat_mask |= 1u << a;
                static int fold = -1;
                if (fold < 0) fold = getenv("LARECOMP_IPA_MUL_FOLD") != NULL;
                if (im == 1 || im == 3) {
                    if (c->ipa_mode == 1) {
                        char mb[16];
                        sprintf(e, "as_type<uint>(in.v%d[%d] * as_type<float>(%s))", a, comp, reg(c, c->ipa_mult, mb));
                    } else sprintf(e, "as_type<uint>(in.v%d[%d])", a, comp);
                } else if (c->ipa_mode == 0) sprintf(e, "as_type<uint>(in.v%d[%d] * in.pos.w)", a, comp);
                else if (c->ipa_mode == 1 && !fold) {
                    char mb[16];
                    const char *m = reg(c, c->ipa_mult, mb);
                    if (c->glsl)   // a select: glslang makes ?: with expression operands an if/else + temporary (~21 per FS)
                        sprintf(e, "mix(as_type<uint>(in.v%d[%d] * in.pos.w * as_type<float>(%s)), as_type<uint>(in.v%d[%d]), "
                                   "%s == as_type<uint>(1.0f / in.pos.w))", a, comp, m, a, comp, m);
                    else sprintf(e, "(%s == as_type<uint>(1.0f / in.pos.w) ? as_type<uint>(in.v%d[%d]) : "
                                    "as_type<uint>(in.v%d[%d] * in.pos.w * as_type<float>(%s)))", m, a, comp, a, comp, m);
                } else sprintf(e, "as_type<uint>(in.v%d[%d])", a, comp);
            }
        } else if (addr == 0x3FC) { c->uses_front_facing = true; strcpy(e, "(front_facing ? 0xFFFFFFFFu : 0u)"); }
        else if (addr == 0x2E0 || addr == 0x2E4) { sprintf(e, "as_type<uint>(point_coord[%d])", (addr >> 2) & 1); }
        else { sprintf(e, "0u /* attr %#x */", addr); }
    } else {
        sprintf(e, "0u /* attr %#x */", addr);
    }
    set_reg(c, dst, e);
}

static void attr_store(Ctx *c, int src, uint32_t addr) {
    char b[16];
    const char *v = reg(c, src, b);
    if (c->glsl && c->stage != STAGE_VERTEX) {  // no vertex outputs in this stage (MSL: undeclared `out`)
        if (addr >= 0x80 && addr < 0x280) c->out_attr_mask |= 1u << ((addr - 0x80) >> 4);
        if (addr == 0x6C) c->info->writes_point_size = true;
        sp(&c->body, "    /* store attr %#x */\n", addr);
        return;
    }
    if (addr >= 0x70 && addr < 0x80) {
        sp(&c->body, "    out.pos[%d] = as_type<float>(%s);\n", (addr >> 2) & 3, v);
    } else if (addr >= 0x80 && addr < 0x280) {
        int a = (addr - 0x80) >> 4;
        c->out_attr_mask |= 1u << a;
        sp(&c->body, "    out.v%d[%d] = as_type<float>(%s);\n", a, (addr >> 2) & 3, v);
    } else if (addr == 0x6C) {
        c->info->writes_point_size = true;
        sp(&c->body, "    out.psize = as_type<float>(%s);\n", v);
    } else {
        sp(&c->body, "    /* store attr %#x */\n", addr);
    }
}


// ------------------------------------------------------------------ half precision (SM53)
// Swizzle: 0 H1_H0 (both halves), 1 F32 (one f32 used for both lanes), 2 H0_H0, 3 H1_H1.
// Merge: 0 H1_H0 (pack both), 1 F32 (lane 0 as f32), 2 MRG_H0 (replace low half), 3 MRG_H1 (replace high half).
static const char *hsrc(Ctx *c, const char *v, int swz, bool neg, bool abs_, char *buf) {
    (void)c;
    char t[300];
    sprintf(t, "hx(%s, %du)", v, swz);
    if (abs_ && neg) sprintf(buf, "(-abs(%s))", t);
    else if (abs_) sprintf(buf, "abs(%s)", t);
    else if (neg) sprintf(buf, "(-%s)", t);
    else strcpy(buf, t);
    return buf;
}
#define HS(v, sw, n, ab) hsrc(c, (v), (sw), (n), (ab), alloca(400))

static void hstore(Ctx *c, int d, const char *expr, int merge, bool sat) {
    char t[2048], e[1700];
    if (sat) sprintf(e, "saturate(%s)", expr); else strcpy(e, expr);
    if (d == RZ) return;
    c->reg_used[d] = 1;
    sprintf(t, "hpack(%s, %du, r%d)", e, merge, d);
    set_reg(c, d, t);
}

// immediate halves: 9 bits of each half's top (the low 6 mantissa bits are zero) plus separate signs
static uint32_t himm(uint64_t w) {
    uint32_t lo = ((uint32_t)bits(w, 20, 9) << 6) | ((uint32_t)bits(w, 29, 1) << 15);
    uint32_t hi = ((uint32_t)bits(w, 30, 9) << 6) | ((uint32_t)bits(w, 56, 1) << 15);
    return lo | (hi << 16);
}

static bool emit_half(Ctx *c, uint64_t w) {
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8);
    char t[1800], ib[32], cb[256];
    uint16_t op = (uint16_t)(w >> 48);
    // ---- HADD2 / HMUL2 (reg 5d10/5d08, cbuf 7a8x/788x, imm 7a0x/780x, 32I 2c/2a)
    int kind = -1;  // 0 add, 1 mul
    int form = -1;  // 0 reg, 1 cbuf, 2 imm, 3 32i
    if ((op & 0xfff8) == 0x5d10) { kind = 0; form = 0; }
    else if ((op & 0xfff8) == 0x5d08) { kind = 1; form = 0; }
    else if ((op & 0xfe80) == 0x7a80) { kind = 0; form = 1; }
    else if ((op & 0xfe80) == 0x7a00) { kind = 0; form = 2; }
    else if ((op & 0xfe80) == 0x7880) { kind = 1; form = 1; }
    else if ((op & 0xfe80) == 0x7800) { kind = 1; form = 2; }
    else if ((op & 0xfe00) == 0x2c00) { kind = 0; form = 3; }
    else if ((op & 0xfe00) == 0x2a00) { kind = 1; form = 3; }
    if (kind >= 0) {
        int merge, swa, swb = 0;
        bool sat, nega, absa = false, negb = false, absb = false;
        const char *bv;
        if (form == 3) {
            merge = 0; sat = bits(w, 52, 1); swa = (int)bits(w, 53, 2); nega = kind == 0 ? bits(w, 56, 1) : false;
            sprintf(ib, "0x%08xu", (uint32_t)bits(w, 20, 32)); bv = ib;
        } else {
            // field layout per Ryujinx's InstDecoders (InstHadd2*/InstHmul2*)
            merge = (int)bits(w, 49, 2); swa = (int)bits(w, 47, 2); nega = bits(w, 43, 1); absa = bits(w, 44, 1);
            if (form == 0) {
                sat = bits(w, 32, 1); absb = bits(w, 30, 1); swb = (int)bits(w, 28, 2);
                if (kind == 0) negb = bits(w, 31, 1);
                else { nega = bits(w, 31, 1); negb = false; }  // HMUL2 reg: the negate bit is A's at 31
                bv = R((int)bits(w, 20, 8));
            } else if (form == 1) {
                sat = bits(w, 52, 1); negb = kind == 0 ? bits(w, 56, 1) : false; absb = bits(w, 54, 1); swb = 1;
                bv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb);
            } else {
                sat = bits(w, 52, 1); swb = 0;
                sprintf(ib, "0x%08xu", himm(w)); bv = ib;
            }
        }
        bool hfmz = kind == 1 && bits(w, form == 3 ? 55 : 39, 2) == 2;  // HMUL2 .FMZ per half (Eden half_floating_point_multiply.cpp)
        sprintf(t, kind == 0 ? "(%s + %s)" : (hfmz ? "fmz_mul2(%s, %s)" : "(%s * %s)"), HS(R(a), swa, nega, absa), HS(bv, swb, negb, absb));
        hstore(c, d, t, merge, sat);
        return true;
    }
    // ---- HFMA2 (reg 5d00, cr 70xx|0x80, rc 60xx|0x80, imm 70xx, 32I 28)
    form = -1;
    if ((op & 0xfff8) == 0x5d00) form = 0;
    else if ((op & 0xf880) == 0x7080) form = 1;  // b = cbuf, c = reg
    else if ((op & 0xf880) == 0x6080) form = 2;  // b = reg, c = cbuf
    else if ((op & 0xf880) == 0x7000) form = 3;  // b = imm, c = reg
    else if ((op & 0xfe00) == 0x2800) form = 4;  // 32I: c = d
    if (form >= 0) {
        int merge, swa = (int)bits(w, 47, 2), swb = 0, swc = 0;
        bool sat, negb, negc;
        const char *bv, *cv;
        char cb2[256];
        switch (form) {  // field layout per Ryujinx's InstDecoders (InstHfma2*); NegA is applied to B (same product)
        case 0:
            merge = (int)bits(w, 49, 2); sat = bits(w, 32, 1); negb = bits(w, 31, 1); negc = bits(w, 30, 1);
            swb = (int)bits(w, 28, 2); swc = (int)bits(w, 35, 2);
            bv = R((int)bits(w, 20, 8)); cv = R((int)bits(w, 39, 8));
            break;
        case 1:  // b = cbuf, c = reg
            merge = (int)bits(w, 49, 2); sat = bits(w, 52, 1); negb = bits(w, 56, 1); negc = bits(w, 51, 1);
            swb = 1; swc = (int)bits(w, 53, 2);
            bv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb); cv = R((int)bits(w, 39, 8));
            break;
        case 2:  // b = reg (field at 39), c = cbuf
            merge = (int)bits(w, 49, 2); sat = bits(w, 52, 1); negb = bits(w, 56, 1); negc = bits(w, 51, 1);
            swb = (int)bits(w, 53, 2); swc = 1;
            bv = R((int)bits(w, 39, 8)); cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb2);
            break;
        case 3:  // b = imm, c = reg
            merge = (int)bits(w, 49, 2); sat = bits(w, 52, 1); negb = false; negc = bits(w, 51, 1);
            swb = 0; swc = (int)bits(w, 53, 2);
            sprintf(ib, "0x%08xu", himm(w)); bv = ib; cv = R((int)bits(w, 39, 8));
            break;
        default:  // 32I: c = d
            merge = 0; sat = false; negb = false; negc = bits(w, 51, 1);
            swa = (int)bits(w, 47, 2); swb = 0; swc = 0;
            sprintf(ib, "0x%08xu", (uint32_t)bits(w, 20, 32)); bv = ib; cv = R(d);
            break;
        }
        bool hfmz = bits(w, form == 0 ? 37 : form == 4 ? 55 : 57, 2) == 2;  // HFMA2 .FMZ per half (Eden half_floating_point_fused_multiply_add.cpp)
        sprintf(t, hfmz ? "fmz_fma2(%s, %s, %s)" : "fma(%s, %s, %s)", HS(R(a), swa, false, false), HS(bv, swb, negb, false), HS(cv, swc, negc, false));
        hstore(c, d, t, merge, sat);
        return true;
    }
    // ---- HSETP2 / HSET2 (reg forms)
    if ((op & 0xfff8) == 0x5d20 || (op & 0xfff8) == 0x5d18) {
        bool setp = (op & 0xfff8) == 0x5d20;
        const char *x = HS(R(a), (int)bits(w, 47, 2), bits(w, 43, 1), bits(w, 44, 1));
        const char *y = HS(R((int)bits(w, 20, 8)), (int)bits(w, 28, 2), bits(w, 31, 1), bits(w, 30, 1));
        int cop = (int)bits(w, 35, 4), bop = (int)bits(w, 45, 2);
        const char *pp = P((int)bits(w, 39, 3), bits(w, 42, 1));
        char lx[512], ly[512];
        sp(&c->body, "    { float2 hx_ = %s, hy_ = %s; bool pp = %s;\n", x, y, pp);
        fcmp_expr(lx, cop, "hx_.x", "hy_.x");
        fcmp_expr(ly, cop, "hx_.y", "hy_.y");
        sp(&c->body, "      bool l = %s, h = %s;\n", lx, ly);
        if (setp) {
            int pa = (int)bits(w, 3, 3), pb = (int)bits(w, 0, 3);
            bool hand = bits(w, 49, 1);
            if (hand) sp(&c->body, "      l = l && h; h = !l;\n");
            if (pa != PT) { c->pred_used[pa] = 1; sp(&c->body, "      p%d = l %s pp;\n", pa, bool_op(bop)); }
            if (pb != PT) { c->pred_used[pb] = 1; sp(&c->body, "      p%d = h %s pp;\n", pb, bool_op(bop)); }
        } else {
            bool bf = bits(w, 49, 1);
            if (d != RZ) {
                c->reg_used[d] = 1;
                sp(&c->body, "      l = l %s pp; h = h %s pp;\n", bool_op(bop), bool_op(bop));
                sp(&c->body, bf ? "      r%d = (l ? 0x3c00u : 0u) | (h ? 0x3c000000u : 0u);\n"
                                : "      r%d = (l ? 0xFFFFu : 0u) | (h ? 0xFFFF0000u : 0u);\n", d);
            }
        }
        sp(&c->body, "    }\n");
        return true;
    }
    return false;
}

// ------------------------------------------------------------------ textures
static int tex_slot(Ctx *c, uint32_t index, int type, bool shadow, bool array) {
    if (index >= 8192) return 0;
    if (c->tex_slot[index]) {
        int s = c->tex_slot[index] - 1;
        if (type >= 0 && c->tex_weak[s]) {
            ShaderTexture *t = &c->info->tex[s];
            t->type = (uint8_t)type;
            t->shadow = shadow;
            t->is_array = array;
            c->tex_weak[s] = false;
        } else if (c->glsl && type >= 0 && (c->info->tex[s].type != type || c->info->tex[s].shadow != shadow)) {
            // one GLSL sampler type per binding: the same texture handle sampled as another type (2D + shadow, 2D +
            // array, ...) gets its own binding of that type. (Failing here skipped every draw of the program: run314,
            // 250-550 skipped draws per 2 s in the forest from fs "texture 8 used as two types".)
            for (int k = 0; k < c->info->ntex; k++) {
                const ShaderTexture *o = &c->info->tex[k];
                if (o->handle_offset == index * 4 && !o->bindless && o->type == type && o->shadow == shadow && o->is_array == array) return k;
            }
            int n = c->info->ntex++;
            if (n >= 31) { fail(c, "too many textures"); return 0; }
            c->tex_weak[n] = false;
            ShaderTexture *t = &c->info->tex[n];
            t->handle_offset = index * 4;
            t->type = (uint8_t)type;
            t->shadow = shadow;
            t->is_array = array;
            static int nlog;
            if (nlog++ < 8) printf("[shader] texture %u sampled as two types: extra binding t%d (type %d shadow %d)\n", index, n, type, shadow);
            return n;
        }
        return s;
    }
    bool weak = type < 0;
    if (weak) type = TEXTYPE_2D;
    int s = c->info->ntex++;
    if (s >= 31) { fail(c, "too many textures"); return 0; }
    c->tex_slot[index] = s + 1;
    c->tex_weak[s] = weak;
    ShaderTexture *t = &c->info->tex[s];
    t->handle_offset = index * 4;
    t->type = (uint8_t)type;
    t->shadow = shadow;
    t->is_array = array;
    return s;
}

static const char *tex_type_name(int t) {
    switch (t) {
    case TEXTYPE_1D: return "1d";
    case TEXTYPE_3D: return "3d";
    case TEXTYPE_CUBE: return "cube";
    case TEXTYPE_2D_ARRAY: return "2d_array";
    case TEXTYPE_CUBE_ARRAY: return "cube_array";
    case TEXTYPE_1D_ARRAY: return "1d_array";
    default: return "2d";
    }
}

// Write texture results: component mask selects which of r,g,b,a go to consecutive registers.
static void write_tex_result(Ctx *c, int slot, const char *call, uint32_t mask, int d0, int d1, bool texs, bool f16) {
    sp(&c->body, "    { float4 t = %s;\n", call);
    (void)slot;
    if (f16) {
        // TEXS/TLDS .F16: components (in mask order) packed as half pairs into d0 (lo, hi) and d1 (lo, hi)
        int comps[4], n = 0;
        for (int comp = 0; comp < 4; comp++) if (mask & (1u << comp)) comps[n++] = comp;
        char v[4][16];
        for (int i = 0; i < 4; i++) { if (i < n) sprintf(v[i], "t[%d]", comps[i]); else strcpy(v[i], "0.0f"); }
        const char *pk = c->glsl ? "      r%d = packHalf2x16(vec2(%s, %s));\n" : "      r%d = as_type<uint>(half2(half(%s), half(%s)));\n";
        if (d0 != RZ) { c->reg_used[d0] = 1; sp(&c->body, pk, d0, v[0], v[1]); }
        if (d1 != RZ) { c->reg_used[d1] = 1; sp(&c->body, pk, d1, v[2], v[3]); }
        sp(&c->body, "    }\n");
        return;
    }
    int regs[4], n = 0;
    if (texs) {
        // TEXS: up to two components in d0/d0+1, two more in d1/d1+1
        int out[4] = {d0, d0 == RZ ? RZ : d0 + 1, d1, d1 == RZ ? RZ : d1 + 1};
        for (int i = 0; i < 4; i++) regs[i] = out[i];
    } else {
        for (int i = 0; i < 4; i++) regs[i] = d0 == RZ ? RZ : d0 + i;
    }
    for (int comp = 0; comp < 4; comp++) {
        if (!(mask & (1u << comp))) continue;
        int r = regs[n++];
        if (r != RZ) {
            c->reg_used[r] = 1;
            sp(&c->body, "      r%d = as_type<uint>(t[%d]);\n", r, comp);
            if (c->trace) sp(&c->body, "      if (trace_on) debugPrintfEXT(\"[trace] %%x r%%u=%%08x\", %uu, %du, r%d);\n", c->cur_pc, r, r);
        }
    }
    sp(&c->body, "    }\n");
}

// Integer texel coordinates in guest units -> the (possibly larger) allocation of texture slot `slot`.
static const char *RSC2(Ctx *c, int slot, const char *fmt, ...) {
    static __thread char bufs[4][1024];   // (thread-local: parallel translations)
    static __thread int bi;
    char *o = bufs[bi++ & 3];
    char inner[700];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(inner, sizeof inner, fmt, ap);
    va_end(ap);
    c->uses_rscale = true;   // (vertex stage too: TotK vertex shaders size dynamic-resolution quads from TXQ / fetches)
    snprintf(o, 1024, "rsc(%s, rscale[%d])", inner, slot + 1);
    return o;
}

// maxwell_ssfetch_next (TotK's upscale pass): a 2D texel fetch at mip 0 of guest texel `coord` (a uint2 expression)
// becomes ssf<slot>(), the average of that texel's whole footprint in the scaled allocation, not one texel of it.
static bool SSF2(Ctx *c, char *call, int slot, const char *coord) {
    if (!c->ssfetch || slot >= 32) return false;
    c->uses_rscale = true;
    c->ssf_slots |= 1u << slot;
    sprintf(call, "ssf%d(ivec2(%s), rscale[%d])", slot, coord, slot + 1);
    return true;
}

// TEXS / TLDS: texture_info selects target and coordinate layout
static const char *alayer(Ctx *c, const char *r, int slot, char *buf) {
    // the array index is the low 16 bits of the register (Eden: BitFieldExtract(Ra, 0, 16); W13 M3)
    if (c->glsl) sprintf(buf, "min((%s) & 0xFFFFu, uint(tsize%d(0).z) - 1u)", r, slot);
    else sprintf(buf, "min((%s) & 0xFFFFu, t%d.get_array_size() - 1u)", r, slot);
    return buf;
}
#define ALAYER(r, slot) alayer(c, (r), (slot), alloca(96))

// GLSL texture lookup on slot `slot` (its type is final here: a conflicting use fails in tex_slot).
// co: coordinate expressions (1 for 1D, 2 for 2D, 3 for 3D / cube), layer: uint expression or NULL, dref: depth
// reference or NULL, lod_kind 0 implicit / 1 level / 2 bias (lod), off: ivec2 texel offset or NULL.
// TotK: lod_kind 3 = gather (TLD4/TLD4S; lod = component "0".."3", ignored for depth compare), 4 = LOD query
// (TMML; vec4(textureQueryLod, 0, 0), fragment only, the layer and dref are not part of the query coordinate).
// GLSL offsets must be constants, so register offsets move the coordinates by offset / size of the level
// (level 0 for implicit-LOD lookups). Non-fragment stages have no implicit LOD: level 0 (+ bias).
static void glsl_sample(Ctx *c, char *out, int slot, const char *const *co, const char *layer, const char *dref,
                        int lod_kind, const char *lod, const char *off) {
    if (c->fail) { strcpy(out, "vec4(0.0f)"); return; }  // e.g. a type conflict: tex[slot] is not this lookup's type
    const ShaderTexture *t = &c->info->tex[slot];
    int type = t->type;
    bool cube = type == TEXTYPE_CUBE || type == TEXTYPE_CUBE_ARRAY;
    int n = (type == TEXTYPE_1D || type == TEXTYPE_1D_ARRAY) ? 1 : (type == TEXTYPE_2D || type == TEXTYPE_2D_ARRAY) ? 2 : 3;
    if (lod_kind == 3 && (n == 1 || type == TEXTYPE_3D)) { fail(c, "gather on a 1d/3d texture"); strcpy(out, "vec4(0.0f)"); return; }
    if (lod_kind == 4) {   // textureQueryLod: only fragment shaders have derivatives
        if (c->stage != STAGE_FRAGMENT) { strcpy(out, "vec4(0.0f)"); return; }
        char q[400];
        if (n == 1) snprintf(q, sizeof q, "%s", co[0]);
        else if (n == 2) snprintf(q, sizeof q, "vec2(%s, %s)", co[0], co[1]);
        else snprintf(q, sizeof q, "vec3(%s, %s, %s)", co[0], co[1], co[2]);
        sprintf(out, "vec4(textureQueryLod(t%d, %s), 0.0f, 0.0f)", slot, q);
        return;
    }
    if (c->stage != STAGE_FRAGMENT && lod_kind != 1 && lod_kind != 3 && lod_kind != 5) {   // (5: explicit gradients, any stage)
        if (lod_kind == 0) lod = "0.0f";
        lod_kind = 1;
    }
    if (t->shadow && type == TEXTYPE_3D) { fail(c, "3d shadow texture"); strcpy(out, "vec4(0.0f)"); return; }
    if (layer && type != TEXTYPE_1D_ARRAY && type != TEXTYPE_2D_ARRAY && type != TEXTYPE_CUBE_ARRAY) {
        fail(c, "array lookup on a 3d texture");  // TEX .3D with the array bit
        strcpy(out, "vec4(0.0f)");
        return;
    }
    char L[600], xs[1600], P[2000];
    if (lod_kind == 1) snprintf(L, sizeof L, "int(%s)", lod); else strcpy(L, "0");
    if (n == 1) {
        if (off) snprintf(xs, sizeof xs, "(%s + float((%s).x) / float(tsize%d(%s).x))", co[0], off, slot, L);
        else snprintf(xs, sizeof xs, "%s", co[0]);
    } else if (n == 2) {
        if (off) snprintf(xs, sizeof xs, "(vec2(%s, %s) + vec2(%s) / vec2(tsize%d(%s).xy))", co[0], co[1], off, slot, L);
        else snprintf(xs, sizeof xs, "vec2(%s, %s)", co[0], co[1]);
    } else {
        if (off && !cube) snprintf(xs, sizeof xs, "vec3(vec2(%s, %s) + vec2(%s) / vec2(tsize%d(%s).xy), %s)", co[0], co[1], off, slot, L, co[2]);
        else snprintf(xs, sizeof xs, "vec3(%s, %s, %s)", co[0], co[1], co[2]);  // cube lookups take no offset
    }
    const char *extra = NULL;  // separate compare argument (cube array shadow)
    if (lod_kind == 3) {       // textureGather(sampler, P, comp) / textureGather(samplerShadow, P, refZ)
        if (layer) snprintf(P, sizeof P, "vec%d(%s, float(%s))", n + 1, xs, layer);
        else snprintf(P, sizeof P, "%s", xs);
        if (t->shadow) sprintf(out, "textureGather(t%d, %s, %s)", slot, P, dref ? dref : "0.0f");
        else sprintf(out, "textureGather(t%d, %s, %s)", slot, P, lod);
        return;
    }
    if (!t->shadow) {
        if (layer) snprintf(P, sizeof P, "vec%d(%s, float(%s))", n + 1, xs, layer);
        else snprintf(P, sizeof P, "%s", xs);
    } else if (type == TEXTYPE_CUBE_ARRAY) {
        snprintf(P, sizeof P, "vec4(%s, float(%s))", xs, layer ? layer : "0u");
        extra = dref;
    } else if (type == TEXTYPE_1D) {
        snprintf(P, sizeof P, "vec3(%s, 0.0f, %s)", xs, dref);
    } else if (layer) {
        snprintf(P, sizeof P, "vec%d(%s, float(%s), %s)", n + 2, xs, layer, dref);
    } else {
        snprintf(P, sizeof P, "vec%d(%s, %s)", n + 1, xs, dref);
    }
    char e[2000], ex[300] = "";
    if (extra) snprintf(ex, sizeof ex, ", %s", extra);
    if (lod_kind == 0) snprintf(e, sizeof e, "texture(t%d, %s%s)", slot, P, ex);
    else if (lod_kind == 1) snprintf(e, sizeof e, "textureLod(t%d, %s%s, %s)", slot, P, ex, lod);
    else if (lod_kind == 5) snprintf(e, sizeof e, "textureGrad(t%d, %s%s, %s)", slot, P, ex, lod);   // lod = "dPdx, dPdy" (TXD)
    else snprintf(e, sizeof e, "texture(t%d, %s%s, %s)", slot, P, ex, lod);
    if (t->shadow) sprintf(out, "vec4(vec3(%s), 1.0f)", e);   // alpha 1.0 (Eden Extract / TEX; W13 M4)
    else strcpy(out, e);
}

// TEXS / TLDS in GLSL (same slot allocation order and result layout as the MSL switch in emit_texs)
static bool glsl_texs(Ctx *c, uint64_t w, bool fetch, int *slot_out, char *call, const char *A, const char *A1,
                      const char *Bv, const char *B1) {
    uint32_t index = (uint32_t)bits(w, 36, 13);
    int info = (int)bits(w, 53, 4);
    int slot;
    const char *co[3];
    char fa[64], fa1[64], fb[64], fb1[64];
    sprintf(fa, "as_type<float>(%s)", A); sprintf(fa1, "as_type<float>(%s)", A1);
    sprintf(fb, "as_type<float>(%s)", Bv); sprintf(fb1, "as_type<float>(%s)", B1);
    if (!fetch) {
        switch (info) {
        case 0: slot = tex_slot(c, index, TEXTYPE_1D, false, false); co[0] = fa;   // 1D.LZ (Eden; W13 M1)
            glsl_sample(c, call, slot, co, NULL, NULL, 1, "0.0f", NULL); break;
        case 1: slot = tex_slot(c, index, TEXTYPE_2D, false, false); co[0] = fa; co[1] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 0, NULL, NULL); break;
        case 2: slot = tex_slot(c, index, TEXTYPE_2D, false, false); co[0] = fa; co[1] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 1, "0.0f", NULL); break;
        case 3: slot = tex_slot(c, index, TEXTYPE_2D, false, false); co[0] = fa; co[1] = fa1;
            glsl_sample(c, call, slot, co, NULL, NULL, 1, fb, NULL); break;
        case 4: slot = tex_slot(c, index, TEXTYPE_2D, true, false); co[0] = fa; co[1] = fa1;
            glsl_sample(c, call, slot, co, NULL, fb, 0, NULL, NULL); break;
        case 5: slot = tex_slot(c, index, TEXTYPE_2D, true, false); co[0] = fa; co[1] = fa1;
            glsl_sample(c, call, slot, co, NULL, fb1, 1, fb, NULL); break;
        case 6: slot = tex_slot(c, index, TEXTYPE_2D, true, false); co[0] = fa; co[1] = fa1;
            glsl_sample(c, call, slot, co, NULL, fb, 1, "0.0f", NULL); break;
        case 7: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true); co[0] = fa1; co[1] = fb;
            glsl_sample(c, call, slot, co, ALAYER(A, slot), NULL, 0, NULL, NULL); break;
        case 8: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true); co[0] = fa1; co[1] = fb;
            glsl_sample(c, call, slot, co, ALAYER(A, slot), NULL, 1, "0.0f", NULL); break;
        case 9: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, true, true); co[0] = fa1; co[1] = fb;
            glsl_sample(c, call, slot, co, ALAYER(A, slot), fb1, 1, "0.0f", NULL); break;
        case 10: slot = tex_slot(c, index, TEXTYPE_3D, false, false); co[0] = fa; co[1] = fa1; co[2] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 0, NULL, NULL); break;
        case 11: slot = tex_slot(c, index, TEXTYPE_3D, false, false); co[0] = fa; co[1] = fa1; co[2] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 1, "0.0f", NULL); break;
        case 12: slot = tex_slot(c, index, TEXTYPE_CUBE, false, false); co[0] = fa; co[1] = fa1; co[2] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 0, NULL, NULL); break;
        case 13: slot = tex_slot(c, index, TEXTYPE_CUBE, false, false); co[0] = fa; co[1] = fa1; co[2] = fb;
            glsl_sample(c, call, slot, co, NULL, NULL, 1, fb1, NULL); break;
        default: fail(c, "texs info %d", info); return false;
        }
    } else {
        switch (info) {
        case 0: slot = tex_slot(c, index, TEXTYPE_1D, false, false);
            sprintf(call, "texelFetch(t%d, int(%s), 0)", slot, A); break;
        case 1: slot = tex_slot(c, index, TEXTYPE_1D, false, false);
            sprintf(call, "texelFetch(t%d, int(%s), int(%s))", slot, A, Bv); break;
        case 2: { slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            char co2[700];
            snprintf(co2, sizeof co2, "uint2(%s, %s)", A, Bv);
            if (!SSF2(c, call, slot, co2)) sprintf(call, "texelFetch(t%d, ivec2(%s), 0)", slot, RSC2(c, slot, "%s", co2));
            break; }
        case 4: { slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            char co2[700];
            snprintf(co2, sizeof co2, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A, A1, Bv, Bv);
            if (!SSF2(c, call, slot, co2)) sprintf(call, "texelFetch(t%d, ivec2(%s), 0)", slot, RSC2(c, slot, "%s", co2));
            break; }
        case 5: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "texelFetch(t%d, ivec2(%s), int(%s))", slot, RSC2(c, slot, "uint2(%s, %s)", A, A1), Bv); break;
        case 7: slot = tex_slot(c, index, TEXTYPE_3D, false, false);
            sprintf(call, "texelFetch(t%d, ivec3(%s, %s, %s), 0)", slot, A, A1, Bv); break;
        case 8: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true);
            sprintf(call, "texelFetch(t%d, ivec3(ivec2(%s), int((%s) & 0xFFFFu)), 0)", slot, RSC2(c, slot, "uint2(%s, %s)", Bv, B1), A); break;   // Eden: (Rb, Rb+1), Ra & 0xFFFF (W13 M2)
        case 12: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "texelFetch(t%d, ivec2(%s), int(%s))", slot, RSC2(c, slot, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A, A1, B1, B1), Bv); break;
        default: fail(c, "tlds info %d", info); return false;
        }
    }
    *slot_out = slot;
    return true;
}

static void emit_texs(Ctx *c, uint64_t w, bool fetch) {
    int d0 = (int)bits(w, 0, 8), d1 = (int)bits(w, 28, 8), a = (int)bits(w, 8, 8), b = (int)bits(w, 20, 8);
    uint32_t index = (uint32_t)bits(w, 36, 13);
    int info = (int)bits(w, 53, 4);
    int msel = (int)bits(w, 50, 3);
    uint32_t mask;
    if (d1 == RZ) {
        static const uint32_t m2[] = {1, 2, 4, 8, 3, 9, 10, 12};
        mask = m2[msel];
    } else {
        static const uint32_t m4[] = {7, 11, 13, 14, 15, 15, 15, 15};
        mask = m4[msel];
    }
    char ra[16], ra1[16], rb[16], rb1[16];
    const char *A = reg(c, a, ra), *A1 = reg(c, a == RZ ? RZ : a + 1, ra1), *Bv = reg(c, b, rb),
               *B1 = reg(c, b == RZ ? RZ : b + 1, rb1);
    char call[1024];
    int slot;
    if (c->glsl) {
        char gcall[4096];
        if (!glsl_texs(c, w, fetch, &slot, gcall, A, A1, Bv, B1)) return;
        write_tex_result(c, slot, gcall, mask, d0, d1, true, !bits(w, 59, 1));
        return;
    }
    if (!fetch) {
        switch (info) {
        case 0: slot = tex_slot(c, index, TEXTYPE_1D, false, false);
            sprintf(call, "t%d.sample(s%d, %s)", slot, slot, F(A)); break;
        case 1: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.sample(s%d, float2(%s, %s))", slot, slot, F(A), F(Bv)); break;
        case 2: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.sample(s%d, float2(%s, %s), level(0))", slot, slot, F(A), F(Bv)); break;
        case 3: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.sample(s%d, float2(%s, %s), level(%s))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 4: slot = tex_slot(c, index, TEXTYPE_2D, true, false);
            sprintf(call, "float4(t%d.sample_compare(s%d, float2(%s, %s), %s))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 5: slot = tex_slot(c, index, TEXTYPE_2D, true, false);
            sprintf(call, "float4(t%d.sample_compare(s%d, float2(%s, %s), %s, level(%s)))", slot, slot, F(A), F(A1), F(B1), F(Bv)); break;
        case 6: slot = tex_slot(c, index, TEXTYPE_2D, true, false);
            sprintf(call, "float4(t%d.sample_compare(s%d, float2(%s, %s), %s, level(0)))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 7: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true);
            sprintf(call, "t%d.sample(s%d, float2(%s, %s), %s)", slot, slot, F(A1), F(Bv), ALAYER(A, slot)); break;
        case 8: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true);
            sprintf(call, "t%d.sample(s%d, float2(%s, %s), %s, level(0))", slot, slot, F(A1), F(Bv), ALAYER(A, slot)); break;
        case 9: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, true, true);
            sprintf(call, "float4(t%d.sample_compare(s%d, float2(%s, %s), %s, %s, level(0)))", slot, slot, F(A1), F(Bv), A, F(B1)); break;
        case 10: slot = tex_slot(c, index, TEXTYPE_3D, false, false);
            sprintf(call, "t%d.sample(s%d, float3(%s, %s, %s))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 11: slot = tex_slot(c, index, TEXTYPE_3D, false, false);
            sprintf(call, "t%d.sample(s%d, float3(%s, %s, %s), level(0))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 12: slot = tex_slot(c, index, TEXTYPE_CUBE, false, false);
            sprintf(call, "t%d.sample(s%d, float3(%s, %s, %s))", slot, slot, F(A), F(A1), F(Bv)); break;
        case 13: slot = tex_slot(c, index, TEXTYPE_CUBE, false, false);
            sprintf(call, "t%d.sample(s%d, float3(%s, %s, %s), level(%s))", slot, slot, F(A), F(A1), F(Bv), F(B1)); break;
        default: fail(c, "texs info %d", info); return;
        }
    } else {
        switch (info) {
        case 0: slot = tex_slot(c, index, TEXTYPE_1D, false, false);
            sprintf(call, "t%d.read(uint(%s))", slot, A); break;
        case 1: slot = tex_slot(c, index, TEXTYPE_1D, false, false);
            sprintf(call, "t%d.read(uint(%s), uint(%s))", slot, A, Bv); break;
        case 2: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.read(%s)", slot, RSC2(c, slot, "uint2(%s, %s)", A, Bv)); break;
        case 4: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.read(%s)", slot, RSC2(c, slot, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A, A1, Bv, Bv)); break;
        case 5: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.read(%s, uint(%s))", slot, RSC2(c, slot, "uint2(%s, %s)", A, A1), Bv); break;
        case 7: slot = tex_slot(c, index, TEXTYPE_3D, false, false);
            sprintf(call, "t%d.read(uint3(%s, %s, %s))", slot, A, A1, Bv); break;
        case 8: slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true);
            sprintf(call, "t%d.read(%s, uint(%s) & 0xFFFFu)", slot, RSC2(c, slot, "uint2(%s, %s)", Bv, B1), A); break;
        case 12: slot = tex_slot(c, index, TEXTYPE_2D, false, false);
            sprintf(call, "t%d.read(%s, uint(%s))", slot, RSC2(c, slot, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A, A1, B1, B1), Bv); break;
        default: fail(c, "tlds info %d", info); return;
        }
    }
    write_tex_result(c, slot, call, mask, d0, d1, true, !bits(w, 59, 1));  // 0xd0/0xd2: .F16 results
}

// TLD (full form; Eden texture_fetch.cpp): integer coordinates from gpr8 (array layer first, low 16 bits), then from
// gpr20 the LOD (bit 55) and the packed 4-bit signed offsets (AOFFI, bit 35). TotK: cave_quad_mesh_generic / _lod0
// fragment shaders (TLD.LZ.AOFFI 2D) failed to translate (log 2026-10-15 "unknown instruction dc3a032aa2171a16").
static void emit_tld(Ctx *c, uint64_t w) {
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8), mr = (int)bits(w, 20, 8);
    int type = (int)bits(w, 28, 3);
    uint32_t mask = (uint32_t)bits(w, 31, 4), index = (uint32_t)bits(w, 36, 13);
    bool aoffi = bits(w, 35, 1), lod = bits(w, 55, 1);
    if (!c->glsl) { fail(c, "TLD needs GLSL"); return; }
    if (bits(w, 50, 1) || bits(w, 54, 1) || bits(w, 51, 3) != PT) { fail(c, "TLD ms/clamp/sparse %016llx", (unsigned long long)w); return; }
#define NEXT(r) ((r) == RZ ? RZ : (r) + 1)
    char lodv[64] = "0", o[16] = "0u";
    if (lod) { sprintf(lodv, "int(%s)", R(mr)); mr = NEXT(mr); }
    if (aoffi) strcpy(o, R(mr));
    const char *A = R(a), *A1 = R(NEXT(a)), *A2 = R(NEXT(NEXT(a)));
    char call[2048];
    int slot;
    switch (type) {
    case 0:  // 1D
        slot = tex_slot(c, index, TEXTYPE_1D, false, false);
        sprintf(call, "texelFetch(t%d, int(%s) + (int(%s << 28) >> 28), %s)", slot, A, o, lodv);
        break;
    case 2: {  // 2D
        slot = tex_slot(c, index, TEXTYPE_2D, false, false);
        char co2[700];
        snprintf(co2, sizeof co2, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A, A1, o, o);
        if (lod || !SSF2(c, call, slot, co2)) sprintf(call, "texelFetch(t%d, ivec2(%s), %s)", slot, RSC2(c, slot, "%s", co2), lodv);
        break; }
    case 3:  // 2D array: layer in the first register
        slot = tex_slot(c, index, TEXTYPE_2D_ARRAY, false, true);
        sprintf(call, "texelFetch(t%d, ivec3(ivec2(%s), int((%s) & 0xFFFFu)), %s)", slot,
                RSC2(c, slot, "uint2(int2(%s, %s) + int2(int(%s << 28) >> 28, int(%s << 24) >> 28))", A1, A2, o, o), A, lodv);
        break;
    case 4:  // 3D
        slot = tex_slot(c, index, TEXTYPE_3D, false, false);
        sprintf(call, "texelFetch(t%d, ivec3(int(%s), int(%s), int(%s)) + ivec3(int(%s << 28) >> 28, int(%s << 24) >> 28, int(%s << 20) >> 28), %s)",
                slot, A, A1, A2, o, o, o, lodv);
        break;
    default: fail(c, "TLD type %d", type); return;
    }
#undef NEXT
    write_tex_result(c, slot, call, mask, d, RZ, false, false);
}

// TEX (full form): coordinates from consecutive registers at gpr8, extra parameters at gpr20.
static void emit_tex(Ctx *c, uint64_t w, bool bindless) {
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8), b = (int)bits(w, 20, 8);
    uint32_t index = bindless ? 0 : (uint32_t)bits(w, 36, 13);
    int target = (int)bits(w, 29, 2);
    bool array = bits(w, 28, 1), dc = bits(w, 50, 1), aoffi = bindless ? bits(w, 36, 1) : bits(w, 54, 1);
    uint32_t mask = (uint32_t)bits(w, 31, 4);
    int lod_mode = bindless ? (int)bits(w, 37, 3) : (int)bits(w, 55, 3);  // 0 none, 1 lz, 2 lb, 3 ll, 6 lba, 7 lla
    if (bindless) { fail(c, "bindless tex"); return; }
    int type = target == 0 ? (array ? TEXTYPE_1D_ARRAY : TEXTYPE_1D)
             : target == 1 ? (array ? TEXTYPE_2D_ARRAY : TEXTYPE_2D)
             : target == 2 ? TEXTYPE_3D
                           : (array ? TEXTYPE_CUBE_ARRAY : TEXTYPE_CUBE);
    int ncoord = target == 0 ? 1 : target == 1 ? 2 : 3;
    int slot = tex_slot(c, index, type, dc, array);
    if (c->glsl) {  // same register layout as below
        char cs[3][40], lay[96], lodv[48], offv[96], drv[48], gcall[4096];
        const char *co[3], *layp = NULL, *lodp = NULL, *offp = NULL, *drp = NULL;
        int ar2 = a, p2 = b, lk = 0;
        if (array) { sprintf(lay, "min(uint(r%d), uint(tsize%d(0).z) - 1u)", a, slot); c->reg_used[a] = 1; ar2 = a + 1; layp = lay; }
        for (int i = 0; i < ncoord; i++) { sprintf(cs[i], "as_type<float>(r%d)", ar2 + i); c->reg_used[ar2 + i] = 1; co[i] = cs[i]; }
        if (lod_mode == 2 || lod_mode == 3 || lod_mode == 6 || lod_mode == 7) {
            c->reg_used[p2] = 1;
            sprintf(lodv, "as_type<float>(r%d)", p2);
            lodp = lodv;
            lk = lod_mode == 2 || lod_mode == 6 ? 2 : 1;
            p2++;
        } else if (lod_mode == 1) {
            lk = 1;
            lodp = "0.0f";
        }
        if (aoffi) { c->reg_used[p2] = 1; sprintf(offv, "int2(int(r%d << 28) >> 28, int(r%d << 24) >> 28)", p2, p2); offp = offv; p2++; }
        if (dc) { c->reg_used[p2] = 1; sprintf(drv, "as_type<float>(r%d)", p2); drp = drv; p2++; }
        glsl_sample(c, gcall, slot, co, layp, drp, lk, lodp, offp);
        write_tex_result(c, slot, gcall, mask, d, RZ, false, false);
        return;
    }
    char coords[256] = "", layer[64] = "", tmp[64];
    int ar = a;
    // the hardware clamps the array index to the layer count
    if (array) { sprintf(layer, "min(uint(r%d), t%d.get_array_size() - 1u)", a, slot); c->reg_used[a] = 1; ar = a + 1; }
    for (int i = 0; i < ncoord; i++) {
        sprintf(tmp, "%sas_type<float>(r%d)", i ? ", " : "", ar + i);
        c->reg_used[ar + i] = 1;
        strcat(coords, tmp);
    }
    int pr = b;
    char lod[96] = "", dref[64] = "", off[128] = "";
    if (lod_mode == 2 || lod_mode == 3 || lod_mode == 6 || lod_mode == 7) {
        c->reg_used[pr] = 1;
        sprintf(lod, lod_mode == 2 || lod_mode == 6 ? ", bias(as_type<float>(r%d))" : ", level(as_type<float>(r%d))", pr);
        pr++;
    } else if (lod_mode == 1) {
        strcpy(lod, ", level(0)");
    }
    if (aoffi) {
        c->reg_used[pr] = 1;
        sprintf(off, ", int2(int(r%d << 28) >> 28, int(r%d << 24) >> 28)", pr, pr);
        pr++;
    }
    if (dc) { c->reg_used[pr] = 1; sprintf(dref, "as_type<float>(r%d)", pr); pr++; }
    const char *vec = ncoord == 1 ? "" : ncoord == 2 ? "float2" : "float3";
    char call[1024];
    if (dc)
        sprintf(call, "float4(t%d.sample_compare(s%d, %s(%s)%s%s, %s%s%s))", slot, slot, vec, coords, array ? ", " : "",
                layer, dref, lod, off);
    else
        sprintf(call, "t%d.sample(s%d, %s(%s)%s%s%s%s)", slot, slot, vec, coords, array ? ", " : "", layer, lod, off);
    write_tex_result(c, slot, call, mask, d, RZ, false, false);
}

// TXD: sample with explicit derivatives (TotK: 113 of the 8712 archive programs failed without it, run343). GLSL only.
// Layout from Eden texture_gradient.cpp + emit_spirv_image.cpp (research/sweep/w34_txd_encoding.md, quoted there):
// type bits 28-30 (0 1D, 1 1D array, 2 2D, 3 2D array; 3D / cube not implemented in Eden either), mask 31-34, AOFFI 35,
// texture 36-48, LC 50. Coordinates from Ra; the register after them holds the array index (low 16 bits) and the AOFFI
// offsets (signed 4-bit at bits 16-19 / 20-23); derivatives from Rb (bits 20-27), interleaved dPdx.x, dPdy.x, dPdx.y,
// dPdy.y. TXD.B (handle in Ra, coordinates shifted by one) is not handled.
static void emit_txd(Ctx *c, uint64_t w, bool bindless) {
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8), b = (int)bits(w, 20, 8);
    int ttype = (int)bits(w, 28, 3);
    uint32_t mask = (uint32_t)bits(w, 31, 4), index = (uint32_t)bits(w, 36, 13);
    bool aoffi = bits(w, 35, 1), lc = bits(w, 50, 1);
    if (bindless) { fail(c, "bindless txd"); return; }
    if (lc) { fail(c, "txd lod clamp"); return; }
    if (!c->glsl) { fail(c, "txd (msl)"); return; }
    if (ttype > 3) { fail(c, "txd 3d / cube"); return; }
    bool array = ttype & 1;
    int n = ttype < 2 ? 1 : 2;
    int type = n == 1 ? (array ? TEXTYPE_1D_ARRAY : TEXTYPE_1D) : (array ? TEXTYPE_2D_ARRAY : TEXTYPE_2D);
    int slot = tex_slot(c, index, type, false, array);
    char cs[2][40], lay[96], offv[96], grad[320], gcall[4096];
    const char *co[3] = {NULL, NULL, NULL}, *layp = NULL, *offp = NULL;
    for (int i = 0; i < n; i++) { sprintf(cs[i], "as_type<float>(r%d)", a + i); c->reg_used[a + i] = 1; co[i] = cs[i]; }
    int last = a + n;
    if (array) { sprintf(lay, "min(uint(r%d) & 0xffffu, uint(tsize%d(0).z) - 1u)", last, slot); c->reg_used[last] = 1; layp = lay; }
    if (aoffi) { sprintf(offv, "int2(int(r%d << 12) >> 28, int(r%d << 8) >> 28)", last, last); c->reg_used[last] = 1; offp = offv; }
    for (int i = 0; i < 2 * n; i++) c->reg_used[b + i] = 1;
    if (n == 1) sprintf(grad, "as_type<float>(r%d), as_type<float>(r%d)", b, b + 1);
    else sprintf(grad, "vec2(as_type<float>(r%d), as_type<float>(r%d)), vec2(as_type<float>(r%d), as_type<float>(r%d))", b, b + 2, b + 1, b + 3);
    glsl_sample(c, gcall, slot, co, layp, NULL, 5, grad, offp);
    write_tex_result(c, slot, gcall, mask, d, RZ, false, false);
}

// TotK: TLD4 / TLD4S / TMML (GLSL only). Encodings: research/sweep/maxwell_tmml_tld4.md (yuzu/Citron
// texture_gather*.cpp, texture_mipmap_level.cpp; sample words hand-decoded). AOFFI offsets are signed 6-bit per axis
// (bits 0-5, 8-13) as in yuzu; GLSL textureGather returns the footprint in the same order as the hardware (inferred:
// yuzu lowers TLD4 straight to OpImageGather).
static int tex_type_of(int target, bool array) {
    return target == 0 ? (array ? TEXTYPE_1D_ARRAY : TEXTYPE_1D) : target == 1 ? (array ? TEXTYPE_2D_ARRAY : TEXTYPE_2D)
         : target == 2 ? TEXTYPE_3D : (array ? TEXTYPE_CUBE_ARRAY : TEXTYPE_CUBE);
}
static void emit_tld4(Ctx *c, uint64_t w) {
    if (!c->glsl) { fail(c, "tld4 without glsl"); return; }
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8), b = (int)bits(w, 20, 8);
    uint32_t index = (uint32_t)bits(w, 36, 13), mask = (uint32_t)bits(w, 31, 4);
    int target = (int)bits(w, 29, 2), toff = (int)bits(w, 54, 2), comp = (int)bits(w, 56, 2);
    bool array = bits(w, 28, 1), dc = bits(w, 50, 1);
    if (bits(w, 51, 3) != 7) { fail(c, "tld4 with a sparse predicate"); return; }
    if (toff >= 2) { fail(c, "tld4 ptp offsets"); return; }
    int type = tex_type_of(target, array), ncoord = target == 0 ? 1 : target == 1 ? 2 : 3;
    int slot = tex_slot(c, index, type, dc, array);
    char cs[3][40], lay[96], offv[96], drv[48], cmp[4], gcall[4096];
    const char *co[3], *layp = NULL, *offp = NULL, *drp = NULL;
    int ar = a, p = b;
    if (array) { sprintf(lay, "min(uint(r%d) & 0xffffu, uint(tsize%d(0).z) - 1u)", a, slot); c->reg_used[a] = 1; ar = a + 1; layp = lay; }
    for (int i = 0; i < ncoord; i++) { sprintf(cs[i], "as_type<float>(r%d)", ar + i); c->reg_used[ar + i] = 1; co[i] = cs[i]; }
    if (toff == 1) { c->reg_used[p] = 1; sprintf(offv, "int2(int(r%d << 26) >> 26, int(r%d << 18) >> 26)", p, p); offp = offv; p++; }
    if (dc) { c->reg_used[p] = 1; sprintf(drv, "as_type<float>(r%d)", p); drp = drv; p++; }
    sprintf(cmp, "%d", comp);
    glsl_sample(c, gcall, slot, co, layp, drp, 3, cmp, offp);
    write_tex_result(c, slot, gcall, mask, d, RZ, false, false);
}
static void emit_tld4s(Ctx *c, uint64_t w) {
    if (!c->glsl) { fail(c, "tld4s without glsl"); return; }
    int da = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8), b = (int)bits(w, 20, 8), db = (int)bits(w, 28, 8);
    uint32_t index = (uint32_t)bits(w, 36, 13);
    bool dc = bits(w, 50, 1), aoffi = bits(w, 51, 1), f16 = bits(w, 55, 1);
    int comp = (int)bits(w, 52, 2);
    int slot = tex_slot(c, index, TEXTYPE_2D, dc, false);
    char cs[2][40], offv[96], drv[48], cmp[4], gcall[4096];
    const char *co[2], *offp = NULL, *drp = NULL;
    int y = (aoffi || dc) ? a + 1 : b;   // (Ra, Rb) without extras, else (Ra, Ra+1) with offset / dref in Rb, Rb+1
    sprintf(cs[0], "as_type<float>(r%d)", a); sprintf(cs[1], "as_type<float>(r%d)", y);
    c->reg_used[a] = 1; c->reg_used[y] = 1; co[0] = cs[0]; co[1] = cs[1];
    int p = b;
    if (aoffi) { c->reg_used[p] = 1; sprintf(offv, "int2(int(r%d << 26) >> 26, int(r%d << 18) >> 26)", p, p); offp = offv; p++; }
    if (dc) { c->reg_used[p] = 1; sprintf(drv, "as_type<float>(r%d)", p); drp = drv; }
    sprintf(cmp, "%d", comp);
    glsl_sample(c, gcall, slot, co, NULL, drp, 3, cmp, offp);
    write_tex_result(c, slot, gcall, 0xF, da, db, true, f16);
}
// TMML: elements 0/1 (LOD, LOD relative to the base level) as 8.8 fixed point, 2/3 as float; compacted writes.
static void emit_tmml(Ctx *c, uint64_t w) {
    if (!c->glsl) { fail(c, "tmml without glsl"); return; }
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8);
    uint32_t index = (uint32_t)bits(w, 36, 13), mask = (uint32_t)bits(w, 31, 4);
    int target = (int)bits(w, 29, 2);
    bool array = bits(w, 28, 1);
    int ncoord = target == 0 ? 1 : target == 1 ? 2 : 3;
    // keep an existing slot's type (an LOD query does not care about depth compare); else declare it from the bits
    int slot = c->tex_slot[index < 8192 ? index : 0] ? c->tex_slot[index] - 1 : tex_slot(c, index, tex_type_of(target, array), false, array);
    char cs[3][40], gcall[1024];
    const char *co[3];
    int ar = array ? a + 1 : a;   // the layer (Ra) is not part of the query
    for (int i = 0; i < ncoord; i++) { sprintf(cs[i], "as_type<float>(r%d)", ar + i); c->reg_used[ar + i] = 1; co[i] = cs[i]; }
    glsl_sample(c, gcall, slot, co, NULL, NULL, 4, NULL, NULL);
    sp(&c->body, "    { float4 t = %s;\n", gcall);
    int r = d;
    for (int e = 0; e < 4 && r != RZ; e++) {
        if (!(mask & (1u << e))) continue;
        c->reg_used[r] = 1;
        // unsigned 8.8: a negative LOD (magnified) is 0, as yuzu's ConvertFToU gives. int() first wrapped it to
        // ~0xFFFFFF80, and the game's unsigned I2F made that LOD ~1.7e7 -> textureLod read the 1x1 mip: TotK's cloud
        // boxes (fs 454c58a8) came out as 2x2 quads of one flat value, white ones among them (run280-282 cloud layer).
        if (e < 2) sp(&c->body, "      r%d = uint(max(t[%d], 0.0f) * 256.0f);\n", r, e);
        else sp(&c->body, "      r%d = as_type<uint>(t[%d]);\n", r, e);
        r++;
    }
    sp(&c->body, "    }\n");
}

// ------------------------------------------------------------------ instruction translation
static void emit_insn(Ctx *c, uint64_t w, uint32_t pc) {
    char t[2048], u[2048];
    c->cur_pc = pc;
    int d = (int)bits(w, 0, 8), a = (int)bits(w, 8, 8);
    int pr = (int)bits(w, 16, 3);
    bool prn = bits(w, 19, 1);
    uint16_t op13 = (uint16_t)(w >> 51);  // top 13 bits
    // predicate guard. SSY/PBK/PCNT/PRET take no predicate: those bits are zero in the encoding,
    // which would otherwise read as @P0.
    uint64_t top12 = w & 0xfff0000000000000ull;
    if (top12 == 0xe290000000000000ull || top12 == 0xe2a0000000000000ull || top12 == 0xe2b0000000000000ull ||
        top12 == 0xe270000000000000ull) {
        pr = PT;
        prn = false;
    }
    bool guarded = !(pr == PT && !prn);
    if (pr == PT && prn) return;  // @!pt: never executes
    if (guarded) sp(&c->body, "  if (%s) {\n", P(pr, prn));

#define OPM(val, mask) ((w & (uint64_t)(mask)) == (uint64_t)(val))
    if (OPM(0xe300000000000000ull, 0xfff0000000000000ull)) {  // EXIT
        sp(&c->body, c->flat ? "    tk_ = true;\n" : c->branchy ? "    exit_now = true; break;\n" : "    break;\n");
    } else if (OPM(0xe330000000000000ull, 0xfff0000000000000ull)) {  // KIL
        c->info->uses_discard = true;
        if (c->nodiscard) {
        } else if (!c->glsl) sp(&c->body, "    discard_fragment();\n");
        else if (c->stage == STAGE_FRAGMENT) {
            // KIL = demote to helper (as yuzu / Eden translate it): the killed pixel stops writing but keeps feeding its
            // 2x2 quad's derivatives, as on Maxwell. A plain discard ends it, so later TMML / implicit-LOD samples in the
            // rest of the quad get undefined derivatives: TotK's cloud boxes (fs 454c58a8) KIL zero-density pixels and
            // then textureQueryLod -> random mip per pixel -> speckled cloud layer -> blocky white squares after the
            // depth-aware upsample (run278/279). LARECOMP_KIL_DISCARD=1: discard (the old behaviour).
            static int kd = -1;
            if (kd < 0) kd = getenv("LARECOMP_KIL_DISCARD") != NULL;
            if (kd) sp(&c->body, "    discard;\n");
            else { sp(&c->body, "    demote;\n"); c->kil_demote = true; }
        }
        else sp(&c->body, "    /* kil */\n");  // GLSL has discard in fragment shaders only
    } else if (OPM(0xe240000000000000ull, 0xfff0000000000020ull)) {  // BRA
        uint32_t target = norm_target((int64_t)pc + 8 + sbits(w, 20, 24));
        c->uses_branches = true;
        if (c->flat) sp(&c->body, "    tk_ = true; f_%u = true;\n", target);
        else sp(&c->body, "    pc = %uu; continue;\n", target);
    } else if (OPM(0xe290000000000000ull, 0xfff0000000000020ull)) {  // SSY
        c->uses_branches = true;
        if (!c->flat) sp(&c->body, "    ssy[ssy_n++] = %uu;\n", norm_target((int64_t)pc + 8 + sbits(w, 20, 24)));
    } else if (OPM(0xe2a0000000000000ull, 0xfff0000000000020ull)) {  // PBK
        c->uses_branches = true;
        sp(&c->body, "    pbk[pbk_n++] = %uu;\n", norm_target((int64_t)pc + 8 + sbits(w, 20, 24)));
    } else if (OPM(0xe2b0000000000000ull, 0xfff0000000000020ull)) {  // PCNT
        c->uses_branches = true;
        sp(&c->body, "    pcnt[pcnt_n++] = %uu;\n", norm_target((int64_t)pc + 8 + sbits(w, 20, 24)));
    } else if (OPM(0xf0f8000000000000ull, 0xfff8000000000000ull)) {  // SYNC
        if (!c->flat) sp(&c->body, "    pc = ssy[--ssy_n]; continue;\n");
        else if (c->flat_sync[pc / 8]) sp(&c->body, "    tk_ = true; f_%u = true;\n", c->flat_sync[pc / 8]);
        else sp(&c->body, "    tk_ = true;\n");   // (unreachable block)
    } else if (OPM(0xe340000000000000ull, 0xfff0000000000000ull)) {  // BRK
        sp(&c->body, "    pc = pbk[--pbk_n]; continue;\n");
    } else if (OPM(0xe350000000000000ull, 0xfff0000000000000ull)) {  // CONT
        sp(&c->body, "    pc = pcnt[pcnt_n - 1]; continue;\n");
    } else if (OPM(0xf0f0000000000000ull, 0xfff8000000000000ull) || OPM(0x50b0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0xef98000000000000ull, 0xfff8000000000000ull)) {
        // DEPBAR / NOP / MEMBAR
    } else if (OPM(0xe00000000000ff00ull, 0xff0000400000ff00ull) || OPM(0xe000004000000000ull, 0xff00004000000000ull)) {
        // IPA: interpolate fragment input
        bool idx = bits(w, 38, 1);
        uint32_t addr = (uint32_t)bits(w, 28, 10);
        int mode = (int)bits(w, 54, 2);  // 0 pass, 1 multiply, 2 constant, 3 sc
        if (idx) { fail(c, "ipa idx"); }
        if (addr >= 0x80 && addr < 0x280) {
            int at = (addr - 0x80) >> 4;
            if (mode == 2) c->flat_mask |= 1u << at;
        }
        c->ipa_mode = mode;
        c->ipa_mult = (int)bits(w, 20, 8);
        attr_load(c, d, addr, NULL);
        c->ipa_mode = -1;
        if (bits(w, 51, 1) && d != RZ) sp(&c->body, "    r%d = as_type<uint>(saturate(as_type<float>(r%d)));\n", d, d);
    } else if (OPM(0xefd8000000000000ull, 0xfff8000000000000ull)) {  // LD attribute
        int n = 1 + (int)bits(w, 47, 2);
        uint32_t addr = (uint32_t)bits(w, 20, 10);
        int ia = (int)bits(w, 8, 8);
        if (ia != RZ) { fail(c, "indexed ld a[]"); }
        for (int i = 0; i < n; i++) attr_load(c, d + i, addr + 4 * i, NULL);
    } else if (OPM(0xeff0000000000000ull, 0xfff8000000000000ull)) {  // ST attribute
        int n = 1 + (int)bits(w, 47, 2);
        uint32_t addr = (uint32_t)bits(w, 20, 10);
        for (int i = 0; i < n; i++) attr_store(c, d + i, addr + 4 * i);
    } else if (OPM(0xef90000000000000ull, 0xfff8000000000000ull)) {  // LD c[idx][reg + imm]
        int idx = (int)bits(w, 36, 5);
        // Address = Ra + sign-extended imm16 (Eden load_constant.cpp: BitField<20, 16, s64> offset, IAdd(reg, s32 imm)).
        // TotK skinning shaders index bone palettes with negative offsets (cbr(cb5, r76 + 65504u) read 64 KB past the
        // palette: spiky, flat-textured skinned meshes, run116). With RZ the offset stays unsigned (same below 0x8000).
        int64_t off = a == RZ ? (int64_t)bits(w, 20, 16) : sbits(w, 20, 16);
        int sz = (int)bits(w, 48, 3);
        int n = sz == 5 ? 2 : 1;
        char rb[16];
        const char *dyn = a == RZ ? NULL : reg(c, a, rb);
        // LDC.64 R12, c[R12 + 8]: the address register is also the first destination. Latch it before writing, or the
        // second word is read from the just-loaded value (TotK fs afe3834a: one 50-px cell looped 100000 times, 23 ms
        // of a 31 ms GPU frame on Sky Island, run261)
        bool latch = dyn && n > 1 && a >= d && a < d + n;
        if (latch) { sp(&c->body, "    { uint ldc_a = %s;\n", dyn); dyn = "ldc_a"; }
        if (sz > 5 || bits(w, 44, 2)) {  // IL/IS/ISL index modes: not seen yet
            static int nlog;
            if (nlog++ < 8) printf("[shader] LDC size %d mode %d not handled (read as 32-bit)\n", sz, (int)bits(w, 44, 2));
        }
        for (int i = 0; i < n; i++) {
            char cb[256];
            cbuf_ref(c, idx, off + 4 * i, dyn, cb);
            if (dyn) c->info->cbuf_size[idx] = 0x10000;
            if (sz < 4) {  // U8 / S8 / U16 / S16 (Eden: size 0-3): the byte / halfword at the address inside its word
                char sh[300], t2[800];
                if (dyn) sprintf(sh, "int((((%s) + %uu) & 3u) * 8u)", dyn, (uint32_t)off);
                else sprintf(sh, "%d", (int)((off & 3) * 8));
                if (sz & 1) sprintf(t2, "uint(bitfieldExtract(int(%s), %s, %d))", cb, sh, sz < 2 ? 8 : 16);
                else sprintf(t2, "bitfieldExtract(%s, %s, %d)", cb, sh, sz < 2 ? 8 : 16);
                set_reg(c, d, t2);
                break;
            }
            set_reg(c, d + i, cb);
        }
        if (latch) sp(&c->body, "    }\n");
    } else if (OPM(0xd000000000000000ull, 0xf600000000000000ull)) {  // TEXS
        emit_texs(c, w, false);
    } else if (OPM(0xd200000000000000ull, 0xf600000000000000ull)) {  // TLDS
        emit_texs(c, w, true);
    } else if (OPM(0xc038000000000000ull, 0xfc38000000000000ull)) {  // TEX
        emit_tex(c, w, false);
    } else if (OPM(0xdc38000000000000ull, 0xff38000000000000ull)) {  // TLD (Eden: 1101 1100 --11 1---)
        emit_tld(c, w);
    } else if (OPM(0xdeb8000000000000ull, 0xfff8000000000000ull)) {  // TEX.B (bindless)
        emit_tex(c, w, true);
    } else if (OPM(0xde00000000000000ull, 0xffc0000000000000ull)) {  // TXD (Eden: 1101 1110 00)
        emit_txd(c, w, false);
    } else if (OPM(0xde40000000000000ull, 0xffc0000000000000ull)) {  // TXD.B
        emit_txd(c, w, true);
    } else if (OPM(0xc800000000000000ull, 0xfc00000000000000ull)) {  // TLD4 (TotK)
        emit_tld4(c, w);
    } else if (OPM(0xdf00000000000000ull, 0xff40000000000000ull)) {  // TLD4S (TotK)
        emit_tld4s(c, w);
    } else if (OPM(0xdf58000000000000ull, 0xfff8000000000000ull)) {  // TMML (TotK)
        emit_tmml(c, w);
    } else if (OPM(0xdf48000000000000ull, 0xfff8000000000000ull)) {  // TXQ
        uint32_t index = (uint32_t)bits(w, 36, 13);
        int qt = (int)bits(w, 22, 6);
        uint32_t mask = (uint32_t)bits(w, 31, 4);
        int slot = tex_slot(c, index, -1, false, false);  // the type comes from a sample, if any
        if (qt == 1 && c->glsl) {  // tsize<slot>() is emitted with the slot's final type
            c->uses_rscale = true;
            sp(&c->body, "    { uint lv = %s; uvec4 q = uvec4(uint(floor(float(tsize%d(int(lv)).x) / rscale[%d].x + 0.5f)), "
               "uint(floor(float(tsize%d(int(lv)).y) / rscale[%d].y + 0.5f)), 1u, uint(textureQueryLevels(t%d)));\n",
               R(a), slot, slot + 1, slot, slot + 1, slot);
            int n = 0;
            for (int i = 0; i < 4; i++)
                if (mask & (1u << i)) { c->reg_used[d + n] = 1; sp(&c->body, "      r%d = q[%d];\n", d + n, i); n++; }
            sp(&c->body, "    }\n");
        } else if (qt == 1) {  // dimension
            if (c->stage != STAGE_VERTEX) {  // report the guest size of scaled allocations
                c->uses_rscale = true;
                sp(&c->body, "    { uint lv = %s; uint4 q = uint4(uint(round(float(t%d.get_width(lv)) / rscale[%d].x)), "
                   "uint(round(float(t%d.get_height(lv)) / rscale[%d].y)), 1, t%d.get_num_mip_levels());\n",
                   R(a), slot, slot + 1, slot, slot + 1, slot);
            } else {
                sp(&c->body, "    { uint lv = %s; uint4 q = uint4(t%d.get_width(lv), t%d.get_height(lv), 1, t%d.get_num_mip_levels());\n",
                   R(a), slot, slot, slot);
            }
            int n = 0;
            for (int i = 0; i < 4; i++)
                if (mask & (1u << i)) { c->reg_used[d + n] = 1; sp(&c->body, "      r%d = q[%d];\n", d + n, i); n++; }
            sp(&c->body, "    }\n");
        } else {
            fail(c, "txq type %d", qt);
        }
    } else if (OPM(0x5080000000000000ull, 0xfff8000000000000ull)) {  // MUFU
        int fn = (int)bits(w, 20, 4);
        const char *x = NA(F(R(a)), bits(w, 48, 1), bits(w, 46, 1));
        switch (fn) {
        case 0: sprintf(t, "cos(%s)", x); break;
        case 1: sprintf(t, "sin(%s)", x); break;
        case 2: sprintf(t, "exp2(%s)", x); break;
        case 3: sprintf(t, "log2(%s)", x); break;
        case 4: sprintf(t, "(1.0f / %s)", x); break;
        case 5: sprintf(t, "rsqrt(%s)", x); break;
        case 8: sprintf(t, "sqrt(%s)", x); break;
        default: fail(c, "mufu %d", fn); return;
        }
        set_reg(c, d, FRES(t, bits(w, 50, 1)));
    } else if (OPM(0x5c90000000000000ull, 0xfff8000000000000ull) || OPM(0x4c90000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3890000000000000ull, 0xfef8000000000000ull)) {  // RRO: range reduction (no-op for us)
        const char *x = NA(F(B(true)), bits(w, 45, 1), bits(w, 49, 1));
        set_reg(c, d, FRES(x, false));
    } else if (OPM(0x5c70000000000000ull, 0xfff8000000000000ull) || OPM(0x4c70000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3870000000000000ull, 0xfef8000000000000ull)) {  // DADD (Eden double_add.cpp)
        if (!c->glsl) { fail(c, "DADD needs GLSL"); return; }
        if (bits(w, 47, 1)) { fail(c, "DADD.CC"); return; }
        sprintf(t, "(%s + %s)", NA(DR(a), bits(w, 48, 1), bits(w, 46, 1)), NA(DB(), bits(w, 45, 1), bits(w, 49, 1)));
        set_dreg(c, d, t);
    } else if (OPM(0x5c80000000000000ull, 0xfff8000000000000ull) || OPM(0x4c80000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3880000000000000ull, 0xfef8000000000000ull)) {  // DMUL (Eden double_multiply.cpp)
        if (!c->glsl) { fail(c, "DMUL needs GLSL"); return; }
        if (bits(w, 47, 1)) { fail(c, "DMUL.CC"); return; }
        sprintf(t, "(%s * %s)", DR(a), NA(DB(), bits(w, 48, 1), false));
        set_dreg(c, d, t);
    } else if (OPM(0x5b70000000000000ull, 0xfff0000000000000ull) || OPM(0x4b70000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x5370000000000000ull, 0xfff0000000000000ull) || OPM(0x3670000000000000ull, 0xfff0000000000000ull)) {  // DFMA
        if (!c->glsl) { fail(c, "DFMA needs GLSL"); return; }
        if (bits(w, 47, 1)) { fail(c, "DFMA.CC"); return; }
        const char *bv, *cv;
        if (OPM(0x5370000000000000ull, 0xfff0000000000000ull)) { bv = DR((int)bits(w, 39, 8)); cv = dcbuf(c, w, alloca(320)); }  // rc
        else { bv = DB(); cv = DR((int)bits(w, 39, 8)); }  // reg, cr, imm
        sprintf(t, "fma(%s, %s, %s)", DR(a), NA(bv, bits(w, 48, 1), false), NA(cv, bits(w, 49, 1), false));
        set_dreg(c, d, t);
    } else if (OPM(0x5c58000000000000ull, 0xfff8000000000000ull) || OPM(0x4c58000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3858000000000000ull, 0xfef8000000000000ull)) {  // FADD
        sprintf(t, "(%s + %s)", NA(F(R(a)), bits(w, 48, 1), bits(w, 46, 1)), NA(F(B(true)), bits(w, 45, 1), bits(w, 49, 1)));
        set_reg_precise(c, d, t, bits(w, 50, 1));
    } else if (OPM(0x0800000000000000ull, 0xfc00000000000000ull)) {  // FADD32I
        sprintf(u, "as_type<float>(0x%08xu)", (uint32_t)bits(w, 20, 32));
        sprintf(t, "(%s + %s)", NA(F(R(a)), bits(w, 56, 1), bits(w, 54, 1)), NA(u, bits(w, 53, 1), bits(w, 57, 1)));
        set_reg_precise(c, d, t, false);
    } else if (OPM(0x5c68000000000000ull, 0xfff8000000000000ull) || OPM(0x4c68000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3868000000000000ull, 0xfef8000000000000ull)) {  // FMUL
        static const char *const scale[] = {"", " * 0.5f", " * 0.25f", " * 0.125f", " * 8.0f", " * 4.0f", " * 2.0f", ""};
        // .FMZ (fmz bits 44-45 == 2, Eden fmul.cpp): 0 * anything (inf, NaN) = 0
        sprintf(t, bits(w, 44, 2) == 2 ? "(fmz_mul(%s, %s)%s)" : "(%s * %s%s)", F(R(a)), NA(F(B(true)), bits(w, 48, 1), false), scale[bits(w, 41, 3)]);
        set_reg_precise(c, d, t, bits(w, 50, 1));
    } else if (OPM(0x1e00000000000000ull, 0xff00000000000000ull)) {  // FMUL32I
        sprintf(t, bits(w, 53, 2) == 2 ? "fmz_mul(%s, as_type<float>(0x%08xu))" : "(%s * as_type<float>(0x%08xu))", F(R(a)), (uint32_t)bits(w, 20, 32));
        set_reg_precise(c, d, t, bits(w, 55, 1));
    } else if (OPM(0x5980000000000000ull, 0xff80000000000000ull) || OPM(0x4980000000000000ull, 0xff80000000000000ull) ||
               OPM(0x5180000000000000ull, 0xff80000000000000ull) || OPM(0x3280000000000000ull, 0xfe80000000000000ull)) {  // FFMA
        const char *bv;
        char cbuf_c[256], bbuf[256];
        const char *cv = operand_c_for_ffma(c, w, cbuf_c, &bv, bbuf);
        sprintf(t, bits(w, 53, 2) == 2 ? "fmz_fma(%s, %s, %s)" : "fma(%s, %s, %s)", F(R(a)), NA(F(bv), bits(w, 48, 1), false), NA(F(cv), bits(w, 49, 1), false));
        set_reg_precise(c, d, t, bits(w, 50, 1));
    } else if (OPM(0x0c00000000000000ull, 0xfc00000000000000ull)) {  // FFMA32I
        sprintf(u, "as_type<float>(0x%08xu)", (uint32_t)bits(w, 20, 32));
        sprintf(t, bits(w, 53, 2) == 2 ? "fmz_fma(%s, %s, %s)" : "fma(%s, %s, %s)", NA(F(R(a)), bits(w, 56, 1), false), u, NA(F(R(d)), bits(w, 57, 1), false));
        set_reg_precise(c, d, t, bits(w, 55, 1));
    } else if (OPM(0x5c60000000000000ull, 0xfff8000000000000ull) || OPM(0x4c60000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3860000000000000ull, 0xfef8000000000000ull)) {  // FMNMX
        const char *x = NA(F(R(a)), bits(w, 48, 1), bits(w, 46, 1)), *y = NA(F(B(true)), bits(w, 45, 1), bits(w, 49, 1));
        const char *pr = P((int)bits(w, 39, 3), bits(w, 42, 1));
        if (c->glsl && 8 * (strlen(x) + strlen(y)) + 160 < sizeof t) {
            // GLSL: fmin / fmax (the non-NaN operand wins) as selects, not the prelude functions: a stage with no
            // function call skips RADV's whole-shader copy_prop_vars (radv_shader.c: only after inlining made progress),
            // and mix(.., bool) is an OpSelect where ?: with expression operands is an if/else (glslang). Operands are
            // pure, so evaluating both is safe. Null Van Gogh, 120 programs (2026-10-20): RADV -10 % with saturate below.
            char mn[1024], mx[1024];
            snprintf(mn, sizeof mn, "mix(mix(min(%s, %s), %s, isnan(%s)), %s, isnan(%s))", x, y, x, y, y, x);
            snprintf(mx, sizeof mx, "mix(mix(max(%s, %s), %s, isnan(%s)), %s, isnan(%s))", x, y, x, y, y, x);
            if (!strcmp(pr, "true")) sprintf(t, "float(%s)", mn);
            else if (!strcmp(pr, "false")) sprintf(t, "float(%s)", mx);
            else sprintf(t, "float(mix(%s, %s, %s))", mx, mn, pr);
        } else sprintf(t, "(%s ? fmin(%s, %s) : fmax(%s, %s))", pr, x, y, x, y);
        set_reg(c, d, FRES(t, false));
    } else if (OPM(0x5bb0000000000000ull, 0xfff0000000000000ull) || OPM(0x4bb0000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x36b0000000000000ull, 0xfef0000000000000ull)) {  // FSETP
        const char *x = NA(F(R(a)), bits(w, 43, 1), bits(w, 7, 1)), *y = NA(F(B(true)), bits(w, 6, 1), bits(w, 44, 1));
        char cmp[512];
        fcmp_expr(cmp, (int)bits(w, 48, 4), x, y);
        int p = (int)bits(w, 3, 3), q = (int)bits(w, 0, 3), bo = (int)bits(w, 45, 2);
        const char *pc_ = P((int)bits(w, 39, 3), bits(w, 42, 1));
        if (p != PT) { c->pred_used[p] = 1; sp(&c->body, "    { bool cmp = %s; bool pp = %s;\n      p%d = cmp %s pp;\n", cmp, pc_, p, bool_op(bo)); }
        else sp(&c->body, "    { bool cmp = %s; bool pp = %s;\n", cmp, pc_);
        if (q != PT) { c->pred_used[q] = 1; sp(&c->body, "      p%d = !cmp %s pp;\n", q, bool_op(bo)); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5800000000000000ull, 0xff00000000000000ull) || OPM(0x4800000000000000ull, 0xfe00000000000000ull) ||
               OPM(0x3000000000000000ull, 0xfe00000000000000ull)) {  // FSET
        const char *x = NA(F(R(a)), bits(w, 43, 1), bits(w, 54, 1)), *y = NA(F(B(true)), bits(w, 53, 1), bits(w, 44, 1));
        char cmp[512];
        fcmp_expr(cmp, (int)bits(w, 48, 4), x, y);
        sprintf(t, "((%s) %s %s)", cmp, bool_op((int)bits(w, 45, 2)), P((int)bits(w, 39, 3), bits(w, 42, 1)));
        sprintf(u, bits(w, 52, 1) ? "(%s ? 0x3f800000u : 0u)" : "(%s ? 0xFFFFFFFFu : 0u)", t);
        set_reg(c, d, u);
    } else if (OPM(0x5ba0000000000000ull, 0xfff0000000000000ull) || OPM(0x4ba0000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x36a0000000000000ull, 0xfef0000000000000ull) || OPM(0x53a0000000000000ull, 0xfff0000000000000ull)) {  // FCMP
        char cmp[512], cb[256], bb[256];
        const char *bv, *cv;
        if (OPM(0x53a0000000000000ull, 0xfff0000000000000ull)) {
            bv = R((int)bits(w, 39, 8));
            cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb);
        } else {
            bv = operand_b(c, w, true, bb);
            cv = R((int)bits(w, 39, 8));
        }
        fcmp_expr(cmp, (int)bits(w, 48, 4), F(cv), "0.0f");
        sprintf(t, "(%s ? %s : %s)", cmp, R(a), bv);
        set_reg(c, d, t);
    } else if (OPM(0x5c98000000000000ull, 0xfff8000000000000ull) || OPM(0x4c98000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3898000000000000ull, 0xfef8000000000000ull)) {  // MOV
        set_reg(c, d, B(false));
    } else if (OPM(0x0100000000000000ull, 0xfff0000000000000ull)) {  // MOV32I
        sprintf(t, "0x%08xu", (uint32_t)bits(w, 20, 32));
        set_reg(c, d, t);
    } else if (OPM(0x5ca0000000000000ull, 0xfff8000000000000ull) || OPM(0x4ca0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x38a0000000000000ull, 0xfef8000000000000ull)) {  // SEL
        sprintf(t, "(%s ? %s : %s)", P((int)bits(w, 39, 3), bits(w, 42, 1)), R(a), B(false));
        set_reg(c, d, t);
    } else if (OPM(0x5cb0000000000000ull, 0xfff8000000000000ull) || OPM(0x4cb0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x38b0000000000000ull, 0xfef8000000000000ull)) {  // F2I (src fmt bits 10-11: 1 F16; dst size bits 8-9)
        char srcv[512];
        if (bits(w, 10, 2) == 1 && c->glsl) sprintf(srcv, "h2f((%s >> %du) & 0xFFFFu)", B(false), 16 * (int)bits(w, 41, 1));
        else if (bits(w, 10, 2) == 1) sprintf(srcv, "float(as_type<half2>(%s)[%d])", B(false), (int)bits(w, 41, 1));
        else strcpy(srcv, F(B(true)));
        const char *x = NA(srcv, bits(w, 45, 1), bits(w, 49, 1));
        int rnd = (int)bits(w, 39, 2);
        bool sgn = bits(w, 12, 1), i16 = bits(w, 8, 2) == 1;
        static const char *const rf[] = {"rint", "floor", "ceil", "trunc"};
        if (i16)
            sprintf(t, sgn ? "uint(int(clamp(%s(%s), -32768.0f, 32767.0f)))" : "uint(clamp(%s(%s), 0.0f, 65535.0f))", rf[rnd], x);
        else
            sprintf(t, sgn ? "uint(int(clamp(%s(%s), -2147483648.0f, 2147483520.0f)))" : "uint(clamp(%s(%s), 0.0f, 4294967040.0f))", rf[rnd], x);
        set_reg(c, d, t);
    } else if (OPM(0x5cb8000000000000ull, 0xfff8000000000000ull) || OPM(0x4cb8000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x38b8000000000000ull, 0xfef8000000000000ull)) {  // I2F
        bool sgn = bits(w, 13, 1);
        int sz = (int)bits(w, 10, 2), sel = (int)bits(w, 41, 2);
        char src[256];
        const char *bv = B(false);
        if (sz == 0) sprintf(src, sgn ? "int(char((%s >> %d) & 0xFF))" : "((%s >> %d) & 0xFFu)", bv, sel * 8);
        else if (sz == 1) sprintf(src, sgn ? "int(short((%s >> %d) & 0xFFFF))" : "((%s >> %d) & 0xFFFFu)", bv, sel * 8);
        else sprintf(src, sgn ? "int(%s)" : "%s", bv);
        sprintf(u, "float(%s)", src);
        if (bits(w, 8, 2) == 1) {  // F16 result in the low half
            char h[700];
            snprintf(h, sizeof h, c->glsl ? "(packHalf2x16(vec2(%s, 0.0f)))" : "(uint(as_type<ushort>(half(%s))))",
                     NA(u, bits(w, 45, 1), bits(w, 49, 1)));
            set_reg(c, d, h);
        } else {
            set_reg(c, d, FRES(NA(u, bits(w, 45, 1), bits(w, 49, 1)), false));
        }
    } else if (OPM(0x5ca8000000000000ull, 0xfff8000000000000ull) || OPM(0x4ca8000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x38a8000000000000ull, 0xfef8000000000000ull)) {  // F2F (formats: 1 F16, 2 F32; bit 41 = high half)
        int dfmt = (int)bits(w, 8, 2), sfmt = (int)bits(w, 10, 2);
        char srcv[512];
        if (dfmt == 3 || sfmt == 3) {  // F64 source and/or result (register pairs; Eden F2F with FPConvert)
            if (!c->glsl || dfmt == 1) { fail(c, "F2F F64 dst %d src %d", dfmt, sfmt); return; }
            if (sfmt == 3) strcpy(srcv, DB());
            else if (sfmt == 2) sprintf(srcv, "double(%s)", F(B(true)));
            else sprintf(srcv, "double(h2f((%s >> %du) & 0xFFFFu))", B(false), 16 * (int)bits(w, 41, 1));
            const char *x = NA(srcv, bits(w, 45, 1), bits(w, 49, 1));
            static const char *const rf[] = {"rint", "floor", "ceil", "trunc"};
            int mode = (int)((bits(w, 39, 2)) | (bits(w, 42, 1) << 2));
            if (mode >= 4) sprintf(t, "%s(%s)", rf[mode - 4], x);
            else sprintf(t, "%s", x);
            if (dfmt == 3) {
                if (bits(w, 50, 1)) { sprintf(u, "clamp(%s, double(0.0f), double(1.0f))", t); strcpy(t, u); }
                set_dreg(c, d, t);
            } else {
                sprintf(u, "float(%s)", t);
                set_reg(c, d, FRES(u, bits(w, 50, 1)));
            }
        } else {
        if (sfmt == 1 && c->glsl) sprintf(srcv, "h2f((%s >> %du) & 0xFFFFu)", B(false), 16 * (int)bits(w, 41, 1));
        else if (sfmt == 1) sprintf(srcv, "float(as_type<half2>(%s)[%d])", B(false), (int)bits(w, 41, 1));
        else strcpy(srcv, F(B(true)));
        const char *x = NA(srcv, bits(w, 45, 1), bits(w, 49, 1));
        int mode = (int)((bits(w, 39, 2)) | (bits(w, 42, 1) << 2));
        switch (mode) {
        case 4: sprintf(t, "rint(%s)", x); break;
        case 5: sprintf(t, "floor(%s)", x); break;
        case 6: sprintf(t, "ceil(%s)", x); break;
        case 7: sprintf(t, "trunc(%s)", x); break;
        default: sprintf(t, "%s", x);
        }
        if (dfmt == 1) {  // F16 result in the low half, high half zero
            char h[1100];
            snprintf(h, sizeof h, c->glsl ? "(packHalf2x16(vec2(%s, 0.0f)))" : "(uint(as_type<ushort>(half(%s))))",
                     bits(w, 50, 1) ? (sprintf(u, "saturate(%s)", t), u) : t);
            set_reg(c, d, h);
        } else {
            set_reg(c, d, FRES(t, bits(w, 50, 1)));
        }
        }
    } else if (OPM(0x5c10000000000000ull, 0xfff8000000000000ull) || OPM(0x4c10000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3810000000000000ull, 0xfef8000000000000ull)) {  // IADD
        if (bits(w, 43, 1) || bits(w, 47, 1)) c->uses_cc = true;
        sprintf(t, "(%s%s + %s%s%s)", bits(w, 49, 1) ? "-" : "", R(a), bits(w, 48, 1) ? "-" : "", B(false),
                bits(w, 43, 1) ? " + cc" : "");
        if (bits(w, 47, 1)) {
            sp(&c->body, "    { uint s = %s%s; uint t2 = %s%s; uint res = s + t2%s; cc = (res < s) ? 1u : 0u;\n", bits(w, 49, 1) ? "-" : "",
               R(a), bits(w, 48, 1) ? "-" : "", B(false), bits(w, 43, 1) ? " + cc" : "");
            if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = res;\n", d); }
            sp(&c->body, "    }\n");
        } else {
            set_reg(c, d, t);
        }
    } else if (OPM(0x1c00000000000000ull, 0xfe00000000000000ull)) {  // IADD32I (CC bit 52, X bit 53)
        bool wcc = bits(w, 52, 1), x = bits(w, 53, 1);
        if (wcc || x) c->uses_cc = true;
        sp(&c->body, "    { ulong sum = ulong(%s%s) + ulong(0x%08xu)%s;%s\n", bits(w, 56, 1) ? "-" : "", R(a), (uint32_t)bits(w, 20, 32),
           x ? " + ulong(cc)" : "", wcc ? " cc = uint(sum >> 32) & 1u;" : "");
        if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = uint(sum);\n", d); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5cc0000000000000ull, 0xfff0000000000000ull) || OPM(0x4cc0000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x38c0000000000000ull, 0xfef0000000000000ull)) {  // IADD3
        bool wcc3 = bits(w, 47, 1), x3 = bits(w, 48, 1);
        if (wcc3 || x3) {  // carry in/out (64-bit address math)
            c->uses_cc = true;
            sp(&c->body, "    { ulong sum = ulong(%s%s) + ulong(%s%s) + ulong(%s%s)%s;\n", bits(w, 51, 1) ? "-" : "", R(a),
               bits(w, 50, 1) ? "-" : "", B(false), bits(w, 49, 1) ? "-" : "", R((int)bits(w, 39, 8)), x3 ? " + ulong(cc)" : "");
            if (wcc3) sp(&c->body, "      cc = uint(sum >> 32) & 1u;\n");
            if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = uint(sum);\n", d); }
            sp(&c->body, "    }\n");
        } else {
            sprintf(t, "(%s%s + %s%s + %s%s)", bits(w, 51, 1) ? "-" : "", R(a), bits(w, 50, 1) ? "-" : "", B(false),
                    bits(w, 49, 1) ? "-" : "", R((int)bits(w, 39, 8)));
            set_reg(c, d, t);
        }
    } else if (OPM(0x5c18000000000000ull, 0xfff8000000000000ull) || OPM(0x4c18000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3818000000000000ull, 0xfef8000000000000ull)) {  // ISCADD
        sprintf(t, "((%s%s << %du) + %s%s)", bits(w, 49, 1) ? "-" : "", R(a), (int)bits(w, 39, 5), bits(w, 48, 1) ? "-" : "",
                B(false));
        if (bits(w, 47, 1)) {  // .CC: carry out for a following IADD.X (64-bit address math)
            c->uses_cc = true;
            sp(&c->body, "    { uint s = %s%s << %du; uint t2 = %s%s; uint res = s + t2; cc = (res < s) ? 1u : 0u;\n",
               bits(w, 49, 1) ? "-" : "", R(a), (int)bits(w, 39, 5), bits(w, 48, 1) ? "-" : "", B(false));
            if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = res;\n", d); }
            sp(&c->body, "    }\n");
        } else {
            set_reg(c, d, t);
        }
    } else if (OPM(0x1400000000000000ull, 0xfc00000000000000ull)) {  // ISCADD32I
        sprintf(t, "((%s << %du) + 0x%08xu)", R(a), (int)bits(w, 53, 5), (uint32_t)bits(w, 20, 32));
        set_reg(c, d, t);
    } else if (OPM(0x5b60000000000000ull, 0xfff0000000000000ull) || OPM(0x4b60000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x3660000000000000ull, 0xfef0000000000000ull)) {  // ISETP
        char cmp[512];
        icmp_expr(cmp, (int)bits(w, 49, 3), R(a), B(false), bits(w, 48, 1));
        int p = (int)bits(w, 3, 3), q = (int)bits(w, 0, 3), bo = (int)bits(w, 45, 2);
        const char *pp = P((int)bits(w, 39, 3), bits(w, 42, 1));
        sp(&c->body, "    { bool cmp = %s; bool pp = %s;\n", cmp, pp);
        if (p != PT) { c->pred_used[p] = 1; sp(&c->body, "      p%d = cmp %s pp;\n", p, bool_op(bo)); }
        if (q != PT) { c->pred_used[q] = 1; sp(&c->body, "      p%d = !cmp %s pp;\n", q, bool_op(bo)); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5b50000000000000ull, 0xfff0000000000000ull) || OPM(0x4b50000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x3650000000000000ull, 0xfef0000000000000ull)) {  // ISET
        char cmp[512];
        icmp_expr(cmp, (int)bits(w, 49, 3), R(a), B(false), bits(w, 48, 1));
        sprintf(t, "((%s) %s %s)", cmp, bool_op((int)bits(w, 45, 2)), P((int)bits(w, 39, 3), bits(w, 42, 1)));
        sprintf(u, bits(w, 44, 1) ? "(%s ? 0x3f800000u : 0u)" : "(%s ? 0xFFFFFFFFu : 0u)", t);
        set_reg(c, d, u);
    } else if (OPM(0x5c20000000000000ull, 0xfff8000000000000ull) || OPM(0x4c20000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3820000000000000ull, 0xfef8000000000000ull)) {  // IMNMX
        bool sgn = bits(w, 48, 1);
        const char *x = R(a), *y = B(false), *p = P((int)bits(w, 39, 3), bits(w, 42, 1));
        if (sgn) sprintf(t, "uint(%s ? min(int(%s), int(%s)) : max(int(%s), int(%s)))", p, x, y, x, y);
        else sprintf(t, "(%s ? min(%s, %s) : max(%s, %s))", p, x, y, x, y);
        set_reg(c, d, t);
    } else if (OPM(0x5c48000000000000ull, 0xfff8000000000000ull) || OPM(0x4c48000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3848000000000000ull, 0xfef8000000000000ull)) {  // SHL
        const char *bv = B(false);
        if (bits(w, 39, 1)) sprintf(t, "(%s << (%s & 31u))", R(a), bv);
        else sprintf(t, "((%s) >= 32u ? 0u : (%s << (%s)))", bv, R(a), bv);
        set_reg(c, d, t);
    } else if (OPM(0x5c28000000000000ull, 0xfff8000000000000ull) || OPM(0x4c28000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3828000000000000ull, 0xfef8000000000000ull)) {  // SHR
        const char *bv = B(false);
        bool sgn = bits(w, 48, 1);
        if (sgn) sprintf(t, "uint(int(%s) >> min(%s, 31u))", R(a), bv);
        else sprintf(t, "((%s) >= 32u ? 0u : (%s >> (%s)))", bv, R(a), bv);
        set_reg(c, d, t);
    } else if (OPM(0x5c40000000000000ull, 0xfff8000000000000ull) || OPM(0x4c40000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3840000000000000ull, 0xfef8000000000000ull)) {  // LOP
        char x[64], y[256];
        sprintf(x, bits(w, 39, 1) ? "(~%s)" : "%s", R(a));
        sprintf(y, bits(w, 40, 1) ? "(~%s)" : "%s", B(false));
        static const char *const ops[] = {"&", "|", "^", ""};
        int op = (int)bits(w, 41, 2);
        if (op == 3) sprintf(t, "%s", y);
        else sprintf(t, "(%s %s %s)", x, ops[op], y);
        int pd = (int)bits(w, 48, 3);
        if (pd != PT) {
            c->pred_used[pd] = 1;
            int pop = (int)bits(w, 44, 2);
            sp(&c->body, "    { uint res = %s;\n", t);
            if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = res;\n", d); }
            sp(&c->body, "      p%d = %s; }\n", pd, pop == 2 ? "(res == 0u)" : pop == 3 ? "(res != 0u)" : pop == 1 ? "true" : "false");
        } else {
            set_reg(c, d, t);
        }
    } else if (OPM(0x0400000000000000ull, 0xfc00000000000000ull)) {  // LOP32I
        char x[64], y[64];
        sprintf(x, bits(w, 55, 1) ? "(~%s)" : "%s", R(a));
        sprintf(y, bits(w, 56, 1) ? "(~0x%08xu)" : "0x%08xu", (uint32_t)bits(w, 20, 32));
        static const char *const ops[] = {"&", "|", "^", ""};
        int op = (int)bits(w, 53, 2);
        if (op == 3) sprintf(t, "%s", y);
        else sprintf(t, "(%s %s %s)", x, ops[op], y);
        set_reg(c, d, t);
    } else if (OPM(0x5be0000000000000ull, 0xfff8000000000000ull) || OPM(0x3c00000000000000ull, 0xfc00000000000000ull) ||
               OPM(0x0200000000000000ull, 0xfe00000000000000ull)) {  // LOP3
        uint32_t lut;
        const char *bv, *cv;
        char bb[256];
        if ((w >> 56) == 0x5b) { lut = (uint32_t)bits(w, 28, 8); bv = R((int)bits(w, 20, 8)); cv = R((int)bits(w, 39, 8)); }
        else if ((w >> 58) == 0xf) { lut = (uint32_t)bits(w, 48, 8); sprintf(bb, "0x%08xu", (uint32_t)((uint32_t)bits(w, 20, 19) | ((uint32_t)bits(w, 56, 1) << 19) | (bits(w, 56, 1) ? 0xFFF00000u : 0))); bv = bb; cv = R((int)bits(w, 39, 8)); }
        else { lut = (uint32_t)bits(w, 48, 8); bv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, bb); cv = R((int)bits(w, 39, 8)); }
        char e[1024] = "0u";
        // sum of minterms
        int first = 1;
        for (int i = 0; i < 8; i++) {
            if (!(lut & (1u << i))) continue;
            char term[256];
            sprintf(term, "(%s%s & %s%s & %s%s)", (i & 4) ? "" : "~", R(a), (i & 2) ? "" : "~", bv, (i & 1) ? "" : "~", cv);
            if (first) { strcpy(e, term); first = 0; }
            else { strcat(e, " | "); strcat(e, term); }
        }
        sprintf(t, "(%s)", e);
        set_reg(c, d, t);
    } else if (OPM(0x5c00000000000000ull, 0xfff8000000000000ull) || OPM(0x4c00000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3800000000000000ull, 0xfef8000000000000ull)) {  // BFE
        const char *bv = B(false);
        bool sgn = bits(w, 48, 1);
        if (bits(w, 40, 1)) fail(c, "bfe brev");
        sprintf(t, sgn ? "bfe_s(%s, %s)" : "bfe_u(%s, %s)", R(a), bv);
        set_reg(c, d, t);
    } else if (OPM(0x5bf0000000000000ull, 0xfff8000000000000ull) || OPM(0x4bf0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x36f0000000000000ull, 0xfef8000000000000ull) || OPM(0x53f0000000000000ull, 0xfff8000000000000ull)) {  // BFI
        const char *bv, *cv;
        char cb[256];
        if (OPM(0x53f0000000000000ull, 0xfff8000000000000ull)) {
            bv = R((int)bits(w, 39, 8)); cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb);
            sprintf(t, "bfi(%s, %s, %s)", R(a), cv, bv);  // insert a into base (cbuf) with b = position/length
            // bfi with cbuf form: B = REG_39 as position/length, C = cbuf as base
            sprintf(t, "bfi(%s, %s, %s)", R(a), bv, cv);
        } else {
            bv = B(false); cv = R((int)bits(w, 39, 8));
            sprintf(t, "bfi(%s, %s, %s)", R(a), bv, cv);
        }
        set_reg(c, d, t);
    } else if (OPM(0x5b00000000000000ull, 0xffc0000000000000ull) || OPM(0x4e00000000000000ull, 0xfe00000000000000ull) ||
               OPM(0x3600000000000000ull, 0xfec0000000000000ull) || OPM(0x5100000000000000ull, 0xff80000000000000ull)) {  // XMAD
        // per Ryujinx's InstXmad* decoders and EmitXmad
        const char *bv, *cv;
        char cb[256], bb[256];
        bool hb, psl, mrg, ext, ha = bits(w, 53, 1);
        int cmode;
        uint64_t top = w >> 56;
        if (top == 0x5b) {  // reg, reg
            bv = R((int)bits(w, 20, 8)); cv = R((int)bits(w, 39, 8));
            hb = bits(w, 35, 1); psl = bits(w, 36, 1); mrg = bits(w, 37, 1); ext = bits(w, 38, 1); cmode = (int)bits(w, 50, 3);
        } else if ((top & 0xfe) == 0x4e) {  // cbuf B
            bv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb); cv = R((int)bits(w, 39, 8));
            hb = bits(w, 52, 1); psl = bits(w, 55, 1); mrg = bits(w, 56, 1); ext = bits(w, 54, 1); cmode = (int)bits(w, 50, 2);
        } else if ((top & 0xfe) == 0x36) {  // imm16 B
            sprintf(bb, "0x%xu", (uint32_t)bits(w, 20, 16)); bv = bb; cv = R((int)bits(w, 39, 8));
            hb = false; psl = bits(w, 36, 1); mrg = bits(w, 37, 1); ext = bits(w, 38, 1); cmode = (int)bits(w, 50, 3);
        } else {  // 0x51: B = reg39, C = cbuf
            bv = R((int)bits(w, 39, 8)); cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb);
            hb = bits(w, 52, 1); psl = false; mrg = false; ext = bits(w, 54, 1); cmode = (int)bits(w, 50, 2);
        }
        bool sa = bits(w, 48, 1), sb = bits(w, 49, 1), wcc = bits(w, 47, 1);
        if (ext || wcc) c->uses_cc = true;
        sp(&c->body, "    { uint xa = (%s >> %d) & 0xFFFFu; uint xb = (%s >> %d) & 0xFFFFu; uint xbu = %s;\n", R(a), ha ? 16 : 0, bv, hb ? 16 : 0, bv);
        if (sa) sp(&c->body, "      xa = uint(int(short(xa)));\n");
        if (sb) sp(&c->body, "      xb = uint(int(short(xb)));\n");
        sp(&c->body, "      uint prod = xa * xb;%s\n", psl ? " prod <<= 16;" : "");
        switch (cmode) {
        case 1: sp(&c->body, "      uint cc_ = %s & 0xFFFFu;\n", cv); break;                                  // clo
        case 2: sp(&c->body, "      uint cc_ = %s >> 16;\n", cv); break;                                      // chi
        case 3: sp(&c->body, "      uint cc_ = %s - (((xa >> 31) << 16) + ((xb >> 31) << 16));\n", cv); break;  // csfu
        case 4: sp(&c->body, "      uint cc_ = %s + (xbu << 16);\n", cv); break;                              // cbcc
        default: sp(&c->body, "      uint cc_ = %s;\n", cv);                                                  // cfull
        }
        if (ext) sp(&c->body, "      uint res = prod + (cc & 1u);\n");
        else sp(&c->body, "      uint res = prod + cc_;\n");
        if (wcc) sp(&c->body, "      cc = (res < prod) ? 1u : 0u;\n");
        if (mrg) sp(&c->body, "      res = (res & 0xFFFFu) | (xbu << 16);\n");
        if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = res;\n", d); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5090000000000000ull, 0xfff8000000000000ull)) {  // PSETP
        const char *pa = P((int)bits(w, 12, 3), bits(w, 15, 1)), *pb = P((int)bits(w, 29, 3), bits(w, 32, 1));
        const char *pcc = P((int)bits(w, 39, 3), bits(w, 42, 1));
        int bo1 = (int)bits(w, 24, 2), bo2 = (int)bits(w, 45, 2);
        int p = (int)bits(w, 3, 3), q = (int)bits(w, 0, 3);
        sp(&c->body, "    { bool v = (%s %s %s); bool pp = %s;\n", pa, bool_op(bo1), pb, pcc);
        if (p != PT) { c->pred_used[p] = 1; sp(&c->body, "      p%d = v %s pp;\n", p, bool_op(bo2)); }
        if (q != PT) { c->pred_used[q] = 1; sp(&c->body, "      p%d = !v %s pp;\n", q, bool_op(bo2)); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5088000000000000ull, 0xfff8000000000000ull)) {  // PSET
        const char *pa = P((int)bits(w, 12, 3), bits(w, 15, 1)), *pb = P((int)bits(w, 29, 3), bits(w, 32, 1));
        sprintf(t, "(((%s %s %s)) %s %s)", pa, bool_op((int)bits(w, 24, 2)), pb, bool_op((int)bits(w, 45, 2)),
                P((int)bits(w, 39, 3), bits(w, 42, 1)));
        sprintf(u, bits(w, 44, 1) ? "(%s ? 0x3f800000u : 0u)" : "(%s ? 0xFFFFFFFFu : 0u)", t);
        set_reg(c, d, u);
    } else if (OPM(0xf0c8000000000000ull, 0xfff8000000000000ull)) {  // S2R
        int sr = (int)bits(w, 20, 8);
        switch (sr) {
        case 0x12: set_reg(c, d, "0xbf800000u"); break;  // y_direction: -1 (lower-left)
        case 0x21: set_reg(c, d, "tid.x"); break;
        case 0x22: set_reg(c, d, "tid.y"); break;
        case 0x23: set_reg(c, d, "tid.z"); break;
        case 0x25: set_reg(c, d, "ctaid.x"); break;
        case 0x26: set_reg(c, d, "ctaid.y"); break;
        case 0x27: set_reg(c, d, "ctaid.z"); break;
        // SR numbering as yuzu's SpecialRegister: LANEID 0, THREAD_KILL 0x13 (helper invocation), EQ/LT/LE/GT/GEMASK
        // 0x38-0x3c. LANEID was 0u: the game's warp-aggregated atomics (leader = 31 - clz(ballot) == laneid) then
        // almost never fired.
        case 0x00: set_reg(c, d, c->glsl ? "gl_SubgroupInvocationID" : "0u"); break;
        case 0x13: set_reg(c, d, c->glsl && c->stage == STAGE_FRAGMENT ? "(gl_HelperInvocation ? 1u : 0u)" : "0u"); break;
        case 0x38: set_reg(c, d, c->glsl ? "gl_SubgroupEqMask.x" : "0u"); break;
        case 0x39: set_reg(c, d, c->glsl ? "gl_SubgroupLtMask.x" : "0u"); break;
        case 0x3a: set_reg(c, d, c->glsl ? "gl_SubgroupLeMask.x" : "0u"); break;
        case 0x3b: set_reg(c, d, c->glsl ? "gl_SubgroupGtMask.x" : "0u"); break;
        case 0x3c: set_reg(c, d, c->glsl ? "gl_SubgroupGeMask.x" : "0u"); break;
        default: sprintf(t, "0u /* sr %#x */", sr); set_reg(c, d, t);
        }
    } else if (OPM(0x5c30000000000000ull, 0xfff8000000000000ull) || OPM(0x4c30000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3830000000000000ull, 0xfef8000000000000ull)) {  // FLO
        const char *bv = B(false);
        sprintf(t, bits(w, 48, 1) ? "(int(%s) < 0 ? (31u - clz(~%s)) : (31u - clz(%s)))" : "(31u - clz(%s))", bv, bv, bv);
        set_reg(c, d, t);
    } else if (OPM(0x5c08000000000000ull, 0xfff8000000000000ull) || OPM(0x4c08000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3808000000000000ull, 0xfef8000000000000ull)) {  // POPC
        sprintf(t, bits(w, 40, 1) ? "popcount(~%s)" : "popcount(%s)", B(false));
        set_reg(c, d, t);
    } else if (OPM(0x5c38000000000000ull, 0xfff8000000000000ull) || OPM(0x4c38000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x3838000000000000ull, 0xfef8000000000000ull)) {  // IMUL
        if (bits(w, 39, 1)) sprintf(t, bits(w, 40, 1) ? "uint(mulhi(int(%s), int(%s)))" : "mulhi(%s, %s)", R(a), B(false));
        else sprintf(t, "(%s * %s)", R(a), B(false));
        set_reg(c, d, t);
    } else if (OPM(0x1f00000000000000ull, 0xff00000000000000ull)) {  // IMUL32I
        sprintf(t, "(%s * 0x%08xu)", R(a), (uint32_t)bits(w, 20, 32));
        set_reg(c, d, t);
    } else if (OPM(0x5ce0000000000000ull, 0xfff8000000000000ull) || OPM(0x4ce0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x38e0000000000000ull, 0xfef8000000000000ull)) {  // I2I
        int dt = (int)((bits(w, 8, 2)) | (bits(w, 12, 1) << 2)), st = (int)((bits(w, 10, 2)) | (bits(w, 13, 1) << 2));
        int sel = (int)bits(w, 41, 2);
        char src[256];
        const char *bv = B(false);
        int ssz = st & 3;
        bool ssg = st & 4;
        if (ssz == 0) sprintf(src, ssg ? "int(char((%s >> %d) & 0xFFu))" : "int((%s >> %d) & 0xFFu)", bv, sel * 8);
        else if (ssz == 1) sprintf(src, ssg ? "int(short((%s >> %d) & 0xFFFFu))" : "int((%s >> %d) & 0xFFFFu)", bv, sel * 8);
        else sprintf(src, "int(%s)", bv);
        if (bits(w, 49, 1)) { sprintf(u, "abs(%s)", src); strcpy(src, u); }
        if (bits(w, 45, 1)) { sprintf(u, "(-%s)", src); strcpy(src, u); }
        int dsz = dt & 3;
        bool dsg = dt & 4, sat = bits(w, 50, 1);
        if (dsz == 0) sprintf(t, sat ? (dsg ? "uint(clamp(%s, -128, 127))" : "uint(clamp(%s, 0, 255))") : "(uint(%s) & 0xFFu)", src);
        else if (dsz == 1) sprintf(t, sat ? (dsg ? "uint(clamp(%s, -32768, 32767))" : "uint(clamp(%s, 0, 65535))") : "(uint(%s) & 0xFFFFu)", src);
        else sprintf(t, "uint(%s)", src);
        set_reg(c, d, t);
        if (bits(w, 47, 1)) {  // .CC: zero/sign flags of the result
            c->uses_cc = true;
            sp(&c->body, "    { uint ccv = %s; ccz = ccv == 0u; ccs = int(ccv) < 0; }\n", t);
        }
    } else if (OPM(0x5bc0000000000000ull, 0xfff0000000000000ull) || OPM(0x4bc0000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x36c0000000000000ull, 0xfef0000000000000ull) || OPM(0x53c0000000000000ull, 0xfff0000000000000ull)) {  // PRMT
        const char *bv, *cv;
        char cb[256];
        if (OPM(0x53c0000000000000ull, 0xfff0000000000000ull)) { bv = R((int)bits(w, 39, 8)); cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb); }
        else { bv = B(false); cv = R((int)bits(w, 39, 8)); }
        if (bits(w, 48, 4)) fail(c, "prmt mode");
        sprintf(t, "prmt(%s, %s, %s)", R(a), cv, bv);
        set_reg(c, d, t);
    } else if (OPM(0x5cf8000000000000ull, 0xfff8000000000000ull) || OPM(0x38f8000000000000ull, 0xfef8000000000000ull) ||
               OPM(0x5bf8000000000000ull, 0xfff8000000000000ull) || OPM(0x36f8000000000000ull, 0xfef8000000000000ull)) {  // SHF
        bool right = (w >> 56) == 0x5c || (w >> 57) == (0x38 >> 1);
        const char *sh = ((w >> 60) == 3) ? B(false) : R((int)bits(w, 20, 8));
        const char *hi = R((int)bits(w, 39, 8));
        if (right) sprintf(t, "uint((((ulong(%s) << 32) | ulong(%s)) >> (%s & 31u)))", hi, R(a), sh);
        else sprintf(t, "uint((((ulong(%s) << 32) | ulong(%s)) << (%s & 31u)) >> 32)", hi, R(a), sh);
        set_reg(c, d, t);
    } else if (OPM(0xeed0000000000000ull, 0xfff8000000000000ull) || OPM(0xeed8000000000000ull, 0xfff8000000000000ull)) {  // LDG / STG
        bool store = bits(w, 51, 1);
        bool wide = bits(w, 45, 1);
        int size = (int)bits(w, 48, 3);
        int64_t off = sbits(w, 20, 24);
        c->info->uses_gmem = true;
        char addr[256];
        if (a == RZ) sprintf(addr, "ulong(%lld)", (long long)off);
        else if (wide) sprintf(addr, "((ulong(%s) << 32 | ulong(%s)) + ulong(%lld))", R(a + 1), R(a), (long long)off);
        else sprintf(addr, "(ulong(%s) + ulong(%lld))", R(a), (long long)off);
        sp(&c->body, "    { ulong ga = %s - 0x%llxul; bool ok = ga < 0x%llxul;\n", addr,
           (unsigned long long)maxwell_gmem_base, (unsigned long long)maxwell_gmem_size);
        static const char *const ld_small[] = {"uint(gmem[ga])", "uint(int(char(gmem[ga])))",
                                               "uint(*(device ushort *)(gmem + ga))", "uint(int(*(device short *)(gmem + ga)))"};
        int nw = size == 5 ? 2 : size >= 6 ? 4 : 1;
        if (c->glsl) {  // gmem_.w[]: 32-bit words; sub-word accesses within the containing word
            static const char *const gl_small[] = {
                "(gmem_.w[uint(ga >> 2)] >> (uint(ga & 3ul) * 8u)) & 0xFFu",
                "uint(sx8(gmem_.w[uint(ga >> 2)] >> (uint(ga & 3ul) * 8u)))",
                "(gmem_.w[uint(ga >> 2)] >> (uint(ga & 2ul) * 8u)) & 0xFFFFu",
                "uint(sx16(gmem_.w[uint(ga >> 2)] >> (uint(ga & 2ul) * 8u)))"};
            Str saved = c->body;
            c->body = (Str){0};
            if (!store) {
                if (size < 4) {
                    if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      if (ok) r%d = %s; else r%d = 0u;\n", d, gl_small[size], d); }
                } else {
                    for (int k = 0; k < nw; k++)
                        if (d != RZ && d + k < RZ) {
                            c->reg_used[d + k] = 1;
                            sp(&c->body, "      if (ok) r%d = gmem_.w[uint(ga >> 2) + %du]; else r%d = 0u;\n", d + k, k, d + k);
                        }
                }
            } else if (c->stage == STAGE_FRAGMENT && getenv("LARECOMP_NO_FS_STG")) {   // (2026-10-19 NVIDIA hang A/B)
                sp(&c->body, "      // STG dropped (LARECOMP_NO_FS_STG A/B)\n");
            } else {
                c->uses_gmem_store = true;
                if (size < 4)
                    sp(&c->body, "      if (ok) { uint wi = uint(ga >> 2); uint sh = uint(ga & %s) * 8u; atomicAnd(gmem_.w[wi], ~(0x%xu << sh)); "
                       "atomicOr(gmem_.w[wi], (%s & 0x%xu) << sh); }\n", size < 2 ? "3ul" : "2ul", size < 2 ? 0xFFu : 0xFFFFu, R(d),
                       size < 2 ? 0xFFu : 0xFFFFu);
                else
                    for (int k = 0; k < nw; k++)
                        sp(&c->body, "      if (ok) gmem_.w[uint(ga >> 2) + %du] = %s;\n", k, R(d == RZ ? RZ : d + k));
            }
            Str lines = c->body;
            c->body = saved;
            gdev_split(c, &lines);
        } else if (!store) {
            if (size < 4) {
                if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = ok ? %s : 0u;\n", d, ld_small[size]); }
            } else {
                for (int k = 0; k < nw; k++)
                    if (d != RZ && d + k < RZ) { c->reg_used[d + k] = 1; sp(&c->body, "      r%d = ok ? *(device uint *)(gmem + ga + %du) : 0u;\n", d + k, k * 4); }
            }
        } else {
            if (size == 0 || size == 1) sp(&c->body, "      if (ok) gmem[ga] = uchar(%s);\n", R(d));
            else if (size == 2 || size == 3) sp(&c->body, "      if (ok) *(device ushort *)(gmem + ga) = ushort(%s);\n", R(d));
            else
                for (int k = 0; k < nw; k++)
                    sp(&c->body, "      if (ok) *(device uint *)(gmem + ga + %du) = %s;\n", k * 4, R(d == RZ ? RZ : d + k));
        }
        sp(&c->body, "    }\n");
    } else if (emit_half(c, w)) {
    } else if (OPM(0x50f8000000000000ull, 0xfff8000000000000ull)) {  // FSWZADD
        // per quad lane (laneid & 3), 2 bits of the swizzle pick: a+b, b-a, a-b, b
        uint32_t swz = (uint32_t)bits(w, 28, 8);
        if (c->stage != STAGE_FRAGMENT) {
            sprintf(t, "as_type<uint>(%s + %s)", F(R(a)), F(R((int)bits(w, 20, 8))));
        } else {
            c->uses_lane = true;
            sprintf(t, "as_type<uint>(fswzadd(%s, %s, 0x%02xu, lane))", F(R(a)), F(R((int)bits(w, 20, 8))), swz);
        }
        set_reg(c, d, t);
    } else if (OPM(0x5b40000000000000ull, 0xfff0000000000000ull) || OPM(0x4b40000000000000ull, 0xfff0000000000000ull) ||
               OPM(0x3640000000000000ull, 0xfef0000000000000ull) || OPM(0x5340000000000000ull, 0xfff0000000000000ull)) {  // ICMP
        char cmp[512], cb[256], bb[256];
        const char *bv, *cv;
        if (OPM(0x5340000000000000ull, 0xfff0000000000000ull)) {
            bv = R((int)bits(w, 39, 8));
            cv = cbuf_ref(c, (int)bits(w, 34, 5), bits(w, 20, 14) * 4, NULL, cb);
        } else {
            bv = operand_b(c, w, false, bb);
            cv = R((int)bits(w, 39, 8));
        }
        icmp_expr(cmp, (int)bits(w, 49, 3), cv, "0u", bits(w, 48, 1));
        sprintf(t, "(%s ? %s : %s)", cmp, R(a), bv);
        set_reg(c, d, t);
    } else if (OPM(0xef40000000000000ull, 0xfff8000000000000ull) || OPM(0xef50000000000000ull, 0xfff8000000000000ull)) {  // LDL / STL
        // thread-private local memory, modelled as a word array (byte/short accesses within a word)
        bool store = bits(w, 52, 1);
        int size = (int)bits(w, 48, 3);
        int32_t off = (int32_t)sbits(w, 20, 24);
        c->uses_local = true;
        char addr[128];
        if (a == RZ) sprintf(addr, "%du", off);
        else sprintf(addr, "(%s + %du)", R(a), (uint32_t)off);
        int nw = size == 5 ? 2 : size >= 6 ? 4 : 1;
        sp(&c->body, "    { uint la = %s;\n", addr);
        if (!store) {
            if (size < 4) {
                static const char *const ext[] = {"(v >> sh) & 0xFFu", "uint(int(char((v >> sh) & 0xFFu)))", "(v >> sh) & 0xFFFFu",
                                                  "uint(int(short((v >> sh) & 0xFFFFu)))"};
                if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      uint v = lmem[(la >> 2) & %du]; uint sh = (la & 3u) * 8u; r%d = %s;\n", LMEM_WORDS - 1, d, ext[size]); }
            } else {
                for (int k = 0; k < nw; k++)
                    if (d != RZ && d + k < RZ) { c->reg_used[d + k] = 1; sp(&c->body, "      r%d = lmem[((la >> 2) + %du) & %du];\n", d + k, k, LMEM_WORDS - 1); }
            }
        } else {
            if (size < 4) {
                uint32_t m = size < 2 ? 0xFFu : 0xFFFFu;
                sp(&c->body, "      uint sh = (la & 3u) * 8u; uint i = (la >> 2) & %du; lmem[i] = (lmem[i] & ~(0x%xu << sh)) | ((%s & 0x%xu) << sh);\n",
                   LMEM_WORDS - 1, m, R(d), m);
            } else {
                for (int k = 0; k < nw; k++)
                    sp(&c->body, "      lmem[((la >> 2) + %du) & %du] = %s;\n", k, LMEM_WORDS - 1, R(d == RZ ? RZ : d + k));
            }
        }
        sp(&c->body, "    }\n");
    } else if (OPM(0xef10000000000000ull, 0xfff8000000000000ull)) {  // SHFL
        int mode = (int)bits(w, 30, 2);
        char bb[32];
        const char *bv = bits(w, 28, 1) ? (sprintf(bb, "%uu", (unsigned)bits(w, 20, 5)), bb) : R((int)bits(w, 20, 8));
        int pd = (int)bits(w, 48, 3);
        if (c->stage == STAGE_VERTEX) {
            set_reg(c, d, R(a));  // no SIMD-group functions in vertex functions: lane-local value
        } else if (c->glsl) {  // a quad xor (lane mask < 4) is the subgroup xor within the quad
            static const char *const fn[] = {"subgroupShuffle", "subgroupShuffleUp", "subgroupShuffleDown", "subgroupShuffleXor"};
            sprintf(t, "%s(%s, uint(%s))", fn[mode], R(a), bv);
            set_reg(c, d, t);
        } else if (mode == 3) {
            bool quad = bits(w, 28, 1) && bits(w, 20, 5) < 4;
            sprintf(t, quad ? "quad_shuffle_xor(%s, ushort(%s))" : "simd_shuffle_xor(%s, ushort(%s))", R(a), bv);
            set_reg(c, d, t);
        } else {
            static const char *const fn[] = {"simd_shuffle", "simd_shuffle_up", "simd_shuffle_down"};
            sprintf(t, "%s(%s, ushort(%s))", fn[mode], R(a), bv);
            set_reg(c, d, t);
        }
        if (pd != PT) { c->pred_used[pd] = 1; sp(&c->body, "    p%d = true;\n", pd); }
    } else if (OPM(0x38f0000000000000ull, 0xfef8000000000000ull) || OPM(0x5cf0000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x4cf0000000000000ull, 0xfff8000000000000ull)) {  // R2P
        if (bits(w, 40, 1)) { fail(c, "r2p cc"); }
        int shift = (int)bits(w, 41, 2) * 8;
        const char *mask = B(false);
        for (int i = 0; i < 7; i++) {
            c->pred_used[i] = 1;
            sp(&c->body, c->glsl ? "    if (((%s >> %du) & 1u) != 0u) p%d = ((%s >> %du) & 1u) != 0u;\n"
                                 : "    if ((%s >> %du) & 1u) p%d = ((%s >> %du) & 1u) != 0u;\n", mask, i, i, R(a), shift + i);
        }
    } else if (OPM(0x38e8000000000000ull, 0xfef8000000000000ull) || OPM(0x5ce8000000000000ull, 0xfff8000000000000ull) ||
               OPM(0x4ce8000000000000ull, 0xfff8000000000000ull)) {  // P2R
        int shift = (int)bits(w, 41, 2) * 8;
        const char *mask = B(false);
        if (bits(w, 40, 1)) {  // .CC: the flags Z, S, C, O as bits 0-3 (overflow isn't tracked: 0)
            c->uses_cc = true;
            sp(&c->body, "    { uint pr = (ccz ? 1u : 0u) | (ccs ? 2u : 0u) | ((cc & 1u) << 2);\n");
            sp(&c->body, "      uint m = (%s & 0xFu) << %du;\n", mask, shift);
            if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = (%s & ~m) | ((pr << %du) & m);\n", d, R(a), shift); }
            sp(&c->body, "    }\n");
            goto p2r_done;
        }
        sp(&c->body, "    { uint pr = 0u;\n");
        for (int i = 0; i < 7; i++) { c->pred_used[i] = 1; sp(&c->body, "      pr |= (p%d ? 1u : 0u) << %du;\n", i, i); }
        sp(&c->body, "      uint m = (%s & 0x7Fu) << %du;\n", mask, shift);
        if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = (%s & ~m) | ((pr << %du) & m);\n", d, R(a), shift); }
        sp(&c->body, "    }\n");
    p2r_done:;
    } else if (OPM(0xe250000000000000ull, 0xfff0000000000020ull)) {  // BRX reg + imm (relative)
        // computed targets can name a scheduling word; execution continues at the next slot (as for BRA)
        sp(&c->body, "    { uint t = %uu + %s + 0x%08xu; if (((t >> 3) & 3u) == 0u) t += 8u; pc = t; continue; }\n", pc + 8, R(a),
           (uint32_t)sbits(w, 20, 24));
    } else if (OPM(0x50a0000000000000ull, 0xfff8000000000000ull)) {  // CSETP (zero/sign flags only)
        int test = (int)bits(w, 8, 5);
        const char *cond;
        switch (test) {
        case 0: cond = "false"; break;
        case 15: cond = "true"; break;
        case 2: case 10: cond = "ccz"; break;
        case 5: case 13: cond = "!ccz"; break;
        case 1: case 9: cond = "ccs"; break;
        case 6: case 14: cond = "!ccs"; break;
        case 4: case 12: cond = "(!ccz && !ccs)"; break;
        case 3: case 11: cond = "(ccz || ccs)"; break;
        default: cond = "false"; fail(c, "csetp test %d", test);
        }
        c->uses_cc = true;
        int p = (int)bits(w, 3, 3), q = (int)bits(w, 0, 3), bo = (int)bits(w, 45, 2);
        const char *pc_ = P((int)bits(w, 39, 3), bits(w, 42, 1));
        sp(&c->body, "    { bool cmp = %s; bool pp = %s;\n", cond, pc_);
        if (p != PT) { c->pred_used[p] = 1; sp(&c->body, "      p%d = cmp %s pp;\n", p, bool_op(bo)); }
        if (q != PT) { c->pred_used[q] = 1; sp(&c->body, "      p%d = !cmp %s pp;\n", q, bool_op(bo)); }
        sp(&c->body, "    }\n");
    } else if (OPM(0x5098000000000000ull, 0xfff8000000000000ull)) {  // CSET
        fail(c, "cset");
    } else if (c->glsl && (OPM(0xef58000000000000ull, 0xfff8000000000000ull) || OPM(0xef48000000000000ull, 0xfff8000000000000ull))) {
        // TotK: STS / LDS (research/sweep/maxwell_more_opcodes.md): addr = Ra == RZ ? uimm24 : Ra + simm24 (bits 20-43),
        // size bits 48-50 (U8 S8 U16 S16 B32 B64 B128). Compute only; out-of-range accesses are dropped / read 0.
        bool store = bits(w, 52, 1) == 1;   // ef58 STS, ef48 LDS
        int size = (int)bits(w, 48, 3);
        if (c->stage != STAGE_COMPUTE) { fail(c, "shared memory outside compute"); return; }
        c->uses_shared = true;
        char addr[96];
        if (a == RZ) sprintf(addr, "%lluu", (unsigned long long)bits(w, 20, 24));
        else sprintf(addr, "(%s + %uu)", R(a), (uint32_t)(int32_t)sbits(w, 20, 24));
        sp(&c->body, "    { uint sa = %s; bool ok = sa < 49152u; uint si = sa >> 2;\n", addr);
        int nw = size == 5 ? 2 : size >= 6 ? 4 : 1;
        if (size < 4) {
            const char *mask = size < 2 ? "0xFFu" : "0xFFFFu", *lane = size < 2 ? "(sa & 3u)" : "(sa & 2u)";
            if (store) sp(&c->body, "      if (ok) { uint sh = %s * 8u; atomicAnd(smem[si], ~(%s << sh)); atomicOr(smem[si], (%s & %s) << sh); }\n",
                          lane, mask, R(d), mask);
            else if (d != RZ) {
                c->reg_used[d] = 1;
                static const char *const ext[] = {"(smem[si] >> ((sa & 3u) * 8u)) & 0xFFu", "uint(sx8(smem[si] >> ((sa & 3u) * 8u)))",
                                                  "(smem[si] >> ((sa & 2u) * 8u)) & 0xFFFFu", "uint(sx16(smem[si] >> ((sa & 2u) * 8u)))"};
                sp(&c->body, "      r%d = ok ? %s : 0u;\n", d, ext[size]);
            }
        } else {
            for (int k = 0; k < nw; k++) {
                if (store) sp(&c->body, "      if (ok) smem[si + %du] = %s;\n", k, R(d == RZ ? RZ : d + k));
                else if (d != RZ && d + k < RZ) { c->reg_used[d + k] = 1; sp(&c->body, "      r%d = ok ? smem[si + %du] : 0u;\n", d + k, k); }
            }
        }
        sp(&c->body, "    }\n");
    } else if (c->glsl && OPM(0xebf8000000000000ull, 0xfff8000000000000ull)) {
        // TotK: RED (global reduction, no result): addr = (E bit 48 ? R[a+1]:R[a] : zext R[a]) + off (bits 28-47; signed
        // unless Ra == RZ); size bits 20-22 (U32 S32 U64 F32 ...), op bits 23-25 (ADD MIN MAX INC DEC AND OR XOR).
        int size = (int)bits(w, 20, 3), op = (int)bits(w, 23, 3);
        bool e = bits(w, 48, 1);
        int64_t off = a == RZ ? (int64_t)bits(w, 28, 20) : sbits(w, 28, 20);
        if (size > 1 || op == 3 || op == 4) { fail(c, "red size %d op %d", size, op); return; }
        if (getenv("LARECOMP_NO_RED")) { sp(&c->body, "    // RED dropped (LARECOMP_NO_RED A/B)\n"); goto red_done; }
        c->info->uses_gmem = true;
        c->uses_gmem_store = true;
        char addr[256];
        if (a == RZ) sprintf(addr, "ulong(%lld)", (long long)off);
        else if (e) sprintf(addr, "((ulong(%s) << 32 | ulong(%s)) + ulong(%lld))", R(a + 1), R(a), (long long)off);
        else sprintf(addr, "(ulong(%s) + ulong(%lld))", R(a), (long long)off);
        sp(&c->body, "    { ulong ga = %s - 0x%llxul; bool ok = ga < 0x%llxul;\n", addr,
           (unsigned long long)maxwell_gmem_base, (unsigned long long)maxwell_gmem_size);
        static const char *const fn[] = {"atomicAdd", "atomicMin", "atomicMax", "", "", "atomicAnd", "atomicOr", "atomicXor"};
        Str saved = c->body;
        c->body = (Str){0};
        if (size == 1 && (op == 1 || op == 2))   // signed min/max on the uint buffer: compare-and-swap loop
            sp(&c->body, "      if (ok) { uint wi = uint(ga >> 2); uint o = gmem_.w[wi]; for (int it = 0; it < 64; it++) { uint n = uint(%s(int(o), int(%s))); "
                         "if (n == o) break; uint p = atomicCompSwap(gmem_.w[wi], o, n); if (p == o) break; o = p; } }\n",
               op == 1 ? "min" : "max", R(d));
        else sp(&c->body, "      if (ok) { uint wi = uint(ga >> 2); %s(gmem_.w[wi], %s); }\n", fn[op], R(d));
        Str lines = c->body;
        c->body = saved;
        gdev_split(c, &lines);
        sp(&c->body, "    }\n");
    red_done:;
    } else if (c->glsl && OPM(0xed00000000000000ull, 0xff00000000000000ull)) {
        // TotK: ATOM (global atomic with result; Eden atomic_operations_global_memory.cpp): Rd = old value, operand R[20-27],
        // address as RED (Ra, E bit 48, off bits 28-47), size bits 49-51 (U32 S32 ...), op bits 52-55 (ADD MIN MAX INC DEC AND OR XOR EXCH)
        int size = (int)bits(w, 49, 3), op = (int)bits(w, 52, 4), rb = (int)bits(w, 20, 8);
        bool e = bits(w, 48, 1);
        int64_t off = a == RZ ? (int64_t)bits(w, 28, 20) : sbits(w, 28, 20);
        if (size > 1 || op == 3 || op == 4 || op > 8) { fail(c, "atom size %d op %d", size, op); return; }
        c->info->uses_gmem = true;
        c->uses_gmem_store = true;
        char addr[256];
        if (a == RZ) sprintf(addr, "ulong(%lld)", (long long)off);
        else if (e) sprintf(addr, "((ulong(%s) << 32 | ulong(%s)) + ulong(%lld))", R(a + 1), R(a), (long long)off);
        else sprintf(addr, "(ulong(%s) + ulong(%lld))", R(a), (long long)off);
        sp(&c->body, "    { ulong ga = %s - 0x%llxul; bool ok = ga < 0x%llxul; uint av = 0u; uint ao = %s;\n", addr,
           (unsigned long long)maxwell_gmem_base, (unsigned long long)maxwell_gmem_size, R(rb));
        static const char *const fn[] = {"atomicAdd", "atomicMin", "atomicMax", "", "", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange"};
        Str saved = c->body;
        c->body = (Str){0};
        if (size == 1 && (op == 1 || op == 2))   // signed min/max on the uint buffer: compare-and-swap loop, av = old value
            sp(&c->body, "      if (ok) { uint wi = uint(ga >> 2); uint o = gmem_.w[wi]; for (int it = 0; it < 64; it++) { uint n = uint(%s(int(o), int(ao))); "
                         "if (n == o) break; uint p = atomicCompSwap(gmem_.w[wi], o, n); if (p == o) break; o = p; } av = o; }\n",
               op == 1 ? "min" : "max");
        else sp(&c->body, "      if (ok) { uint wi = uint(ga >> 2); av = %s(gmem_.w[wi], ao); }\n", fn[op]);
        Str lines = c->body;
        c->body = saved;
        gdev_split(c, &lines);
        if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = av;\n", d); }
        sp(&c->body, "    }\n");
    } else if (c->glsl && OPM(0x3a00000000000000ull, 0xfe00000000000000ull)) {
        // TotK: VMNMX (video min/max), Eden video_minimum_maximum.cpp layout (research/sweep/maxwell_more_opcodes.md; the
        // selector/width fields overlap there and are copied as is): A = Ra (sel 36-37, width 37-38, sign 48); B = bit 50 ?
        // Rb (20-27, sel 28-29, width 29-30, sign 49) : uimm16 (20-35, Short, sel 0); C = R39.
        // Rd = (op 51-53: MIN 5 / MAX 6, sign 54)((bit 56 ? max : min)(A', B', sign of B), C). CC (47) and SAT (55) unsupported.
        int op = (int)bits(w, 51, 3);
        bool breg = bits(w, 50, 1), sa = bits(w, 48, 1), sb = bits(w, 49, 1), sd = bits(w, 54, 1), mx = bits(w, 56, 1);
        if (bits(w, 47, 1) || bits(w, 55, 1) || (op != 5 && op != 6)) { fail(c, "vmnmx cc/sat/op %d", op); return; }
        char ea[160], eb[160], t2[700];
        for (int k = 0; k < 2; k++) {   // ExtractVideoOperandValue: Byte/Unknown -> 8 bits at sel*8, Short -> 16 at sel*16, Word -> all
            char *o = k ? eb : ea;
            int width = k ? (breg ? (int)bits(w, 29, 2) : 2) : (int)bits(w, 37, 2);
            int sel = k ? (breg ? (int)bits(w, 28, 2) : 0) : (int)bits(w, 36, 2);
            bool sg = k ? sb : sa;
            char src[64];
            if (k && !breg) sprintf(src, "%uu", (uint32_t)bits(w, 20, 16));
            else sprintf(src, "%s", R(k ? (int)bits(w, 20, 8) : a));
            if (width == 3) sprintf(o, "%s", src);
            else {
                int n = width == 2 ? 16 : 8;
                if (sg) sprintf(o, "uint(bitfieldExtract(int(%s), %d, %d))", src, sel * n, n);
                else sprintf(o, "bitfieldExtract(%s, %d, %d)", src, sel * n, n);
            }
        }
        char lhs[400];
        if (sb) sprintf(lhs, "uint(%s(int(%s), int(%s)))", mx ? "max" : "min", ea, eb);
        else sprintf(lhs, "%s(%s, %s)", mx ? "max" : "min", ea, eb);
        if (sd) sprintf(t2, "uint(%s(int(%s), int(%s)))", op == 6 ? "max" : "min", lhs, R((int)bits(w, 39, 8)));
        else sprintf(t2, "%s(%s, %s)", op == 6 ? "max" : "min", lhs, R((int)bits(w, 39, 8)));
        set_reg(c, d, t2);
    } else if (c->glsl && OPM(0x50d8000000000000ull, 0xfff8000000000000ull)) {
        // TotK: VOTE: Rd = ballot of the source predicate (bits 39-41, neg 42); pred_b (45-47) = ALL/ANY/EQ (op 48-49)
        int pa = (int)bits(w, 39, 3), pb = (int)bits(w, 45, 3), vop = (int)bits(w, 48, 2);
        bool na = bits(w, 42, 1);
        sp(&c->body, "    { bool vp = %s;\n", P(pa, na));
        if (d != RZ) { c->reg_used[d] = 1; sp(&c->body, "      r%d = subgroupBallot(vp).x;\n", d); }
        if (pb != PT) {
            c->pred_used[pb] = 1;
            sp(&c->body, "      p%d = %s;\n", pb, vop == 0 ? "subgroupAll(vp)" : vop == 1 ? "subgroupAny(vp)" : "subgroupAllEqual(vp)");
        }
        sp(&c->body, "    }\n");
    } else if (c->glsl && OPM(0x36d0000000000000ull, 0xfef8000000000000ull)) {
        // TotK: LEA (lo, imm): Rd = imm19 (bits 20-38, minus 2^19 if bit 56) + ((neg 45 ? -Ra : Ra) << scale 39-43)
        int64_t imm = (int64_t)bits(w, 20, 19) - (bits(w, 56, 1) ? (1ll << 19) : 0);
        int scale = (int)bits(w, 39, 5);
        if (bits(w, 46, 2) || bits(w, 48, 3) != 7) { fail(c, "lea lo with x/cc/pred"); return; }
        char t2[160];
        sprintf(t2, "(%uu + ((%s%s) << %du))", (uint32_t)imm, bits(w, 45, 1) ? "0u - " : "", R(a), scale);
        set_reg(c, d, t2);
    } else if (c->glsl && OPM(0x1800000000000000ull, 0xfc00000000000000ull)) {
        // TotK: LEA (hi, cbuf): Rd = c[idx][off] + low32(pack64(Ra, R39) (negated if bit 56) >> (32 - scale 51-55))
        int scale = (int)bits(w, 51, 5), hi = (int)bits(w, 39, 8);
        if (bits(w, 57, 1) || bits(w, 47, 1) || bits(w, 48, 3) != 7) { fail(c, "lea hi with x/cc/pred"); return; }
        char cb[96], t2[400];
        cbuf_ref(c, (int)bits(w, 34, 5), (int64_t)bits(w, 20, 14) * 4, NULL, cb);
        sprintf(t2, "(as_u(%s) + uint((%s(ulong(%s) << 32 | ulong(%s))) >> %du))", cb, bits(w, 56, 1) ? "0ul - " : "", R(hi), R(a),
                32 - scale);
        set_reg(c, d, t2);
    } else if (OPM(0xf0a8000000000000ull, 0xfff8000000000000ull)) {  // BAR
        if (!c->glsl) sp(&c->body, "    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        else if (c->stage == STAGE_COMPUTE) sp(&c->body, "    barrier();\n");
        else sp(&c->body, "    /* bar */\n");  // GLSL barrier() is compute / tessellation only
    } else {
        fail(c, "unknown instruction %016llx at %#x", (unsigned long long)w, pc);
    }
    (void)op13;
    if (guarded) sp(&c->body, "  }\n");
}

// ------------------------------------------------------------------ program assembly
static bool is_branch_target_op(uint64_t w) {
    return ((w & 0xfff0000000000020ull) == 0xe240000000000000ull);
}

// ------------------------------------------------------------------ GLSL output
static bool id_start(char ch) { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'; }
static bool id_char(char ch) { return id_start(ch) || (ch >= '0' && ch <= '9'); }
// as_type<float>(x) / as_type<uint>(x) whose operand has a fixed GLSL type: the builtin bit cast instead of the as_f / as_u
// overloads. glslang (no optimizer) turns every overload call into a function call plus a parameter variable (a median
// stage: ~1500), which the driver inlines and promotes on every compile. Deck, 120 programs (2026-10-20, gplbench):
// RADV 78 -> 31 ms per program, 34.9 -> 90.1 programs/s on 7 threads; shaderc 23.3 -> 18.1 ms per stage.
// uint operands (uintBitsToFloat): r<n>, cb<n>[...][<n>], 0x..u / <n>u literals, then ')'. A float operand (floatBitsToUint)
// is the expression up to the cast's ')' when its first primary (after '-' / '(') is float-typed and no '?' or ',' stands
// outside call parentheses: GLSL arithmetic converts the other operands to float, and anything else (a comparison: bool,
// a vector) does not compile. Only a ?: or a comma could make it a uint, which floatBitsToUint would convert by value.
// Float primaries: pf_, t[N] (float4 texture result), in.a<n>[N] of float attributes, in.v<n>[N] (vec4 varyings), in.pos
// (gl_FragCoord), point_coord[N], float literals and float-returning calls.
static bool fixed_cast_operand(const Ctx *c, const char *e, bool uint_operand) {
    if (uint_operand) {
        if (e[0] == 'r' && e[1] >= '0' && e[1] <= '9') { e++; while (*e >= '0' && *e <= '9') e++; }
        else if (e[0] == 'c' && e[1] == 'b' && e[2] >= '0' && e[2] <= '9') {
            e += 2; while (*e >= '0' && *e <= '9') e++;
            if (*e++ != '[') return false;
            while (*e && *e != ']' && *e != '[' && *e != '(' && *e != ')') e++;
            if (*e++ != ']' || *e++ != '[' || !(*e >= '0' && *e <= '9')) return false;
            while (*e >= '0' && *e <= '9') e++;
            if (*e++ != ']') return false;
        } else if (e[0] == '0' && e[1] == 'x') {
            e += 2;
            while ((*e >= '0' && *e <= '9') || (*e >= 'a' && *e <= 'f') || (*e >= 'A' && *e <= 'F')) e++;
            if (*e++ != 'u') return false;
        } else if (*e >= '0' && *e <= '9') { while (*e >= '0' && *e <= '9') e++; if (*e++ != 'u') return false; }
        else return false;
        return *e == ')';
    }
    static const char *const ffn[] = {"saturate(", "float(", "as_type<float>(", "cos(", "sin(", "exp2(", "log2(", "rsqrt(",
                                      "sqrt(", "rint(", "floor(", "ceil(", "trunc(", "abs(float(", "abs(as_type<float>(", "fswzadd(",
                                      "in.pos[", "in.pos.", "point_coord[", "t["};
    const char *q = e;
    while (*q == '-' || *q == '(') q++;
    bool f = false;
    for (size_t k = 0; k < sizeof ffn / sizeof ffn[0] && !f; k++) f = !strncmp(q, ffn[k], strlen(ffn[k]));
    if (!f && !strncmp(q, "pf_", 3)) f = !id_char(q[3]);
    if (!f && (!strncmp(q, "in.a", 4) || !strncmp(q, "in.v", 4)) && q[4] >= '0' && q[4] <= '9') {
        int n = atoi(q + 4);
        const char *z = q + 4;
        while (*z >= '0' && *z <= '9') z++;
        f = *z == '[' && (q[3] == 'v' || !c->key || (n < 32 && c->key->attr_kind[n] == 0));
    }
    if (!f && *q >= '0' && *q <= '9') {   // float literal: digits . digits [e[+-]digits] f
        while (*q >= '0' && *q <= '9') q++;
        if (*q == '.' && q[1] >= '0' && q[1] <= '9') {
            for (q++; *q >= '0' && *q <= '9'; q++) {}
            if (*q == 'e') { q++; if (*q == '-' || *q == '+') q++; while (*q >= '0' && *q <= '9') q++; }
            f = *q == 'f';
        }
    }
    if (!f) return false;
    char call[64];   // per open parenthesis: a call's (its type is the function's) or a grouping one
    int d = 0;
    for (const char *z = e; *z; z++) {
        if (*z == '(') {
            if (d == (int)sizeof call) return false;
            call[d++] = z > e && (id_char(z[-1]) || z[-1] == '>');
        } else if (*z == ')') {
            if (!d) return true;   // the cast's own ')'
            d--;
        } else if ((*z == '?' || *z == ',') && (!d || !call[d - 1])) return false;
    }
    return false;
}

// MSL-spelled instruction body -> GLSL (token-aware; comments and numbers are copied as they are).
static void glsl_rewrite(Ctx *c, Str *o, const char *src) {
    static const char *const map[][2] = {
        {"float2", "vec2"}, {"float3", "vec3"}, {"float4", "vec4"}, {"int2", "ivec2"}, {"int3", "ivec3"},
        {"int4", "ivec4"}, {"uint2", "uvec2"}, {"uint3", "uvec3"}, {"uint4", "uvec4"}, {"bool2", "bvec2"},
        {"ulong", "uint64_t"}, {"rsqrt", "inversesqrt"}, {"rint", "roundEven"}, {"front_facing", "gl_FrontFacing"},
        {"point_coord", "gl_PointCoord"}, {"lane", "gl_SubgroupInvocationID"}, {"vid", "uint(gl_VertexIndex)"},
        {"iid", "uint(gl_InstanceIndex)"},
    };
    const char *p = src;
    while (*p) {
        if (p[0] == '/' && p[1] == '*') {
            const char *e = strstr(p + 2, "*/");
            e = e ? e + 2 : p + strlen(p);
            sp(o, "%.*s", (int)(e - p), p);
            p = e;
            continue;
        }
        if (p[0] == '/' && p[1] == '/') {
            const char *e = strchr(p, '\n');
            if (!e) e = p + strlen(p);
            sp(o, "%.*s", (int)(e - p), p);
            p = e;
            continue;
        }
        if (*p >= '0' && *p <= '9') {  // number (0x..u, 1.0f, ...)
            const char *e = p;
            while (id_char(*e) || *e == '.') e++;
            sp(o, "%.*s", (int)(e - p), p);
            p = e;
            continue;
        }
        if (!id_start(*p)) { sp(o, "%c", *p); p++; continue; }
        const char *e = p;
        while (id_char(*e)) e++;
        int n = (int)(e - p);
        char id[64];
        if (n >= (int)sizeof id || (p > src && p[-1] == '.')) {  // member / swizzle
            sp(o, "%.*s", n, p);
            p = e;
            continue;
        }
        memcpy(id, p, n);
        id[n] = 0;
        p = e;
        if (!strcmp(id, "as_type") && *p == '<') {
            const char *gt = strchr(p, '>');
            if (!gt) { fail(c, "glsl: bad as_type"); return; }
            int tn = (int)(gt - p - 1);
            if (tn == 5 && !strncmp(p + 1, "float", 5)) sp(o, gt[1] == '(' && fixed_cast_operand(c, gt + 2, true) ? "uintBitsToFloat" : "as_f");
            else if (tn == 4 && !strncmp(p + 1, "uint", 4)) sp(o, gt[1] == '(' && fixed_cast_operand(c, gt + 2, false) ? "floatBitsToUint" : "as_u");
            else if (tn == 3 && !strncmp(p + 1, "int", 3)) sp(o, "as_i");
            else if (tn == 5 && !strncmp(p + 1, "half2", 5)) sp(o, "unpack_h2");
            else { fail(c, "glsl: as_type<%.*s>", tn, p + 1); return; }
            p = gt + 1;
            continue;
        }
        if ((!strcmp(id, "in") || !strcmp(id, "out")) && *p == '.') {  // stage I/O struct members
            const char *m = p + 1, *me = m;
            while (id_char(*me)) me++;
            char mem[32];
            snprintf(mem, sizeof mem, "%.*s", (int)(me - m), m);
            if (!strcmp(mem, "pos")) sp(o, c->stage == STAGE_FRAGMENT ? "gl_FragCoord" : "gl_Position");
            else if (!strcmp(mem, "psize")) sp(o, "gl_PointSize");
            else sp(o, "%s", mem);  // a<i> (vertex input), v<i> (varying)
            p = me;
            continue;
        }
        if ((!strcmp(id, "char") || !strcmp(id, "short")) && *p == '(') { sp(o, id[0] == 'c' ? "sx8" : "sx16"); continue; }
        if (!strcmp(id, "tid") || !strcmp(id, "ctaid")) {
            sp(o, c->stage != STAGE_COMPUTE ? "uvec3(0u)" : id[0] == 't' ? "gl_LocalInvocationID" : "gl_WorkGroupID");
            continue;
        }
        if (id[0] == 'r' && n > 1 && n < 6) {  // registers beyond r255 (RZ) come from out-of-range encodings
            bool num = true;
            for (int i = 1; i < n; i++) num = num && id[i] >= '0' && id[i] <= '9';
            if (num && atoi(id + 1) > 255) { fail(c, "register %s out of range", id); return; }
        }
        const char *rep = id;
        for (size_t i = 0; i < sizeof map / sizeof map[0]; i++)
            if (!strcmp(id, map[i][0])) { rep = map[i][1]; break; }
        sp(o, "%s", rep);
    }
}

MAXWELL_TLS bool maxwell_trace_next, maxwell_nodiscard_next, maxwell_solid_next, maxwell_ssfetch_next;
MAXWELL_TLS bool maxwell_vtrace_next;     // LARECOMP_DEBUG_VS: trace the next vertex shader translated (exec.c sets it)
MAXWELL_TLS int maxwell_vtrace_vid = -1;   // -1: print only off-screen vertices (inputs + clip position); else also every register write of that gl_VertexIndex
MAXWELL_TLS int maxwell_trace_x, maxwell_trace_y, maxwell_trace_x2 = -1, maxwell_trace_y2 = -1;
static const char glsl_prelude[] =
    "#version 460\n"
    "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
    "#extension GL_KHR_shader_subgroup_basic : require\n"
    "#extension GL_KHR_shader_subgroup_shuffle : require\n"
    "#extension GL_KHR_shader_subgroup_shuffle_relative : require\n"
    "#extension GL_KHR_shader_subgroup_ballot : require\n"
    "#extension GL_KHR_shader_subgroup_vote : require\n"
    "#extension GL_EXT_texture_shadow_lod : require\n"
    "#define cbr(cb, off) cb[min((off) >> 4, uint(cb.length() - 1))][((off) >> 2) & 3u]\n"
    "uint as_u(float x) { return floatBitsToUint(x); }\n"
    "uint as_u(uint x) { return x; }\n"
    "uint as_u(int x) { return uint(x); }\n"
    "float as_f(uint x) { return uintBitsToFloat(x); }\n"
    "float as_f(int x) { return intBitsToFloat(x); }\n"
    "float as_f(float x) { return x; }\n"
    "int as_i(uint x) { return int(x); }\n"
    "int as_i(float x) { return floatBitsToInt(x); }\n"
    "int sx8(uint v) { return bitfieldExtract(int(v), 0, 8); }\n"
    "int sx8(int v) { return bitfieldExtract(v, 0, 8); }\n"
    "int sx16(uint v) { return bitfieldExtract(int(v), 0, 16); }\n"
    "int sx16(int v) { return bitfieldExtract(v, 0, 16); }\n"
    // MSL semantics: saturate(NaN) = 0, fmin/fmax return the non-NaN operand
    "float saturate(float x) { return isnan(x) ? 0.0f : clamp(x, 0.0f, 1.0f); }\n"
    "vec2 saturate(vec2 x) { return vec2(saturate(x.x), saturate(x.y)); }\n"
    "float fmin(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : min(a, b)); }\n"
    "float fmax(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : max(a, b)); }\n"
    // Maxwell .FMZ multiply (Eden fmul.cpp / ffma.cpp: Select(any_zero, 0 or c, value)): a zero factor gives 0 even
    // against inf/NaN; plain GLSL gives NaN (black dots)
    "float fmz_mul(float a, float b) { return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }\n"
    "float fmz_fma(float a, float b, float c) { return (a == 0.0f || b == 0.0f) ? c : fma(a, b, c); }\n"
    "vec2 fmz_mul2(vec2 a, vec2 b) { return vec2(fmz_mul(a.x, b.x), fmz_mul(a.y, b.y)); }\n"
    "vec2 fmz_fma2(vec2 a, vec2 b, vec2 c) { return vec2(fmz_fma(a.x, b.x, c.x), fmz_fma(a.y, b.y, c.y)); }\n"
    "uint clz(uint v) { return v == 0u ? 32u : uint(31 - findMSB(v)); }\n"
    "uint popcount(uint v) { return uint(bitCount(v)); }\n"
    "uint mulhi(uint a, uint b) { uint hi, lo; umulExtended(a, b, hi, lo); return hi; }\n"
    "int mulhi(int a, int b) { int hi, lo; imulExtended(a, b, hi, lo); return hi; }\n"
    "uint bfe_u(uint v, uint ctl) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u) return 0u; if (pos >= 32u) return 0u; return bitfieldExtract(v, int(pos), int(min(len, 32u - pos))); }\n"
    "uint bfe_s(uint v, uint ctl) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u) return 0u; if (pos >= 32u) return (int(v) < 0) ? 0xFFFFFFFFu : 0u; return uint(bitfieldExtract(int(v), int(pos), int(min(len, 32u - pos)))); }\n"
    "uint bfi(uint ins, uint ctl, uint base) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u || pos >= 32u) return base; return bitfieldInsert(base, ins, int(pos), int(min(len, 32u - pos))); }\n"
    "float fswzadd(float a, float b, uint swz, uint lane) { uint m = (swz >> ((lane & 3u) * 2u)) & 3u; float ma = m == 1u ? -1.0f : (m == 3u ? 0.0f : 1.0f); float mb = m == 2u ? -1.0f : 1.0f; return a * ma + b * mb; }\n"
    "uvec2 rsc(uvec2 c, vec4 s) { return (s.x == 1.0f && s.y == 1.0f) ? c : uvec2(floor((vec2(c) + 0.5f) * s.xy)); }\n"
    // half -> float by bits, not unpackHalf2x16: NVIDIA's driver recognises unpacked halves and runs some of the
    // fp32 math after them in native fp16, differently per shader, so a depth pre-pass and its colour pass (EQUAL
    // test) placed wind-animated foliage apart: black slashes / flicker (tools: notes/PLAN.md 2026-10-08)
    "float h2f(uint h) { uint s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;"
    " if (e == 0u) { float d = float(m) * 5.9604644775390625e-8; return s != 0u ? -d : d; }"
    " if (e == 31u) return uintBitsToFloat(s | 0x7F800000u | (m << 13));"
    " return uintBitsToFloat(s | ((e + 112u) << 23) | (m << 13)); }\n"
    "vec2 unpack_h2(uint v) { return vec2(h2f(v & 0xFFFFu), h2f(v >> 16)); }\n"
    "vec2 hx(uint v, uint swz) { if (swz == 1u) return vec2(uintBitsToFloat(v)); vec2 h = unpack_h2(v); return swz == 2u ? h.xx : (swz == 3u ? h.yy : h); }\n"
    "uint hpack(vec2 r, uint merge, uint old) { if (merge == 1u) return floatBitsToUint(r.x); uint lo = packHalf2x16(vec2(r.x, 0.0f)), hi = packHalf2x16(vec2(r.y, 0.0f)); if (merge == 2u) return (old & 0xFFFF0000u) | lo; if (merge == 3u) return (old & 0xFFFFu) | (hi << 16); return lo | (hi << 16); }\n"
    "uint prmt(uint a, uint b, uint sel) { uint r = 0u; for (uint i = 0u; i < 4u; i++) { uint s = (sel >> (i * 4u)) & 7u; uint bt = s < 4u ? (a >> (s * 8u)) & 0xFFu : (b >> ((s - 4u) * 8u)) & 0xFFu; r |= bt << (i * 8u); } return r; }\n";
// LARECOMP_NATIVE_MINMAX=1 (experiment, 2026-10-17): saturate / fmin / fmax as plain min / max. The NaN-safe forms
// above came from the Metal backend; AMD (v_min/v_max_f32) and NVIDIA already return the non-NaN operand like Maxwell
// FMNMX, and max(NaN, 0) = 0 makes min(max(x, 0), 1) a NaN -> 0 saturate. The cloud FS (fs 195655ef) has 56 fmin/fmax
// and 27 saturate calls, each 3-4 extra ops in the NaN-safe form.
static char *str_replace1(char *src, const char *old, const char *rep_) {
    char *at = strstr(src, old);
    if (!at) return src;
    size_t n = strlen(src) - strlen(old) + strlen(rep_) + 1;
    char *d = malloc(n);
    memcpy(d, src, (size_t)(at - src));
    strcpy(d + (at - src), rep_);
    strcat(d, at + strlen(old));
    free(src);
    return d;
}
static const char *prelude_text(void) {
    static __thread char *p;
    static __thread int init;
    if (!init) {
        init = 1;
        const char *e = getenv("LARECOMP_NATIVE_MINMAX");
        if (e && *e == '1') {
            p = strdup(glsl_prelude);
            p = str_replace1(p, "float saturate(float x) { return isnan(x) ? 0.0f : clamp(x, 0.0f, 1.0f); }",
                             "float saturate(float x) { return min(max(x, 0.0f), 1.0f); }");
            p = str_replace1(p, "float fmin(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : min(a, b)); }",
                             "float fmin(float a, float b) { return min(a, b); }");
            p = str_replace1(p, "float fmax(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : max(a, b)); }",
                             "float fmax(float a, float b) { return max(a, b); }");
        }
    }
    return p ? p : glsl_prelude;
}

static const char *glsl_sampler_type(const ShaderTexture *t) {
    static const char *const names[][2] = {
        {"sampler1D", "sampler1DShadow"}, {"sampler2D", "sampler2DShadow"}, {"sampler3D", "sampler3D"},
        {"samplerCube", "samplerCubeShadow"}, {"sampler2DArray", "sampler2DArrayShadow"},
        {"samplerCubeArray", "samplerCubeArrayShadow"}, {"sampler1DArray", "sampler1DArrayShadow"}};
    int ty = t->type <= TEXTYPE_1D_ARRAY ? t->type : TEXTYPE_2D;
    return names[ty][t->shadow ? 1 : 0];
}

// Declarations + entry point around the (rewritten) body; mirrors the MSL assembly in maxwell_translate.
static ShaderInfo *glsl_finish(Ctx *c, bool any_branch, bool dbg) {
    ShaderInfo *info = c->info;
    const ShaderKey *key = c->key;
    int stage = c->stage;
    Str body = {0};
    glsl_rewrite(c, &body, c->body.buf ? c->body.buf : "");
    // tsize<i>(l) of a 3D / 2D-array / cube-array texture is textureSize itself: call that directly, so the stage keeps
    // no function call (RADV skips its whole-shader copy_prop_vars then; see FMNMX)
    for (int i = 0; i < info->ntex; i++) {
        int ty = info->tex[i].type;
        if (ty != TEXTYPE_3D && ty != TEXTYPE_2D_ARRAY && ty != TEXTYPE_CUBE_ARRAY) continue;
        char from[24], to[32];
        snprintf(from, sizeof from, "tsize%d(", i);
        snprintf(to, sizeof to, "textureSize(t%d, ", i);
        Str r = {0};
        const char *p = body.buf ? body.buf : "", *q;
        while ((q = strstr(p, from))) { sp(&r, "%.*s%s", (int)(q - p), p, to); p = q + strlen(from); }
        sp(&r, "%s", p);
        free(body.buf);
        body = r;
    }
    if (info->uses_gmem && (info->cbuf_mask & (1u << GMEM_BUFFER_INDEX))) fail(c, "cbuf %d clashes with gmem", GMEM_BUFFER_INDEX);
    if (info->cbuf_mask >> 18) fail(c, "cbuf index out of range");
    for (int i = 0; i < info->ntex; i++)
        if (info->tex[i].shadow && info->tex[i].type == TEXTYPE_3D) fail(c, "3d shadow texture");
    if (c->fail) { free(body.buf); free(c->body.buf); free(c); return info; }
    int set = stage == STAGE_FRAGMENT ? 1 : 0;
    Str s = {0};
    const char *kil_ext = c->kil_demote && c->minpc ? "#extension GL_EXT_demote_to_helper_invocation : require\n#extension GL_KHR_shader_subgroup_arithmetic : require\n"
                          : c->kil_demote ? "#extension GL_EXT_demote_to_helper_invocation : require\n"
                          : c->minpc ? "#extension GL_KHR_shader_subgroup_arithmetic : require\n" : "";
    const char *pre = prelude_text();
    if (c->trace) sp(&s, "#version 460\n#extension GL_EXT_debug_printf : enable\n%s%s", kil_ext, pre + strlen("#version 460\n"));
    else if (*kil_ext) sp(&s, "#version 460\n%s%s", kil_ext, pre + strlen("#version 460\n"));
    else sp(&s, "%s", pre);
    for (int i = 0; i < 18; i++) {
        if (!(info->cbuf_mask & (1u << i))) continue;
        uint32_t n = info->cbuf_size[i] / 16;
        if (c->cbuf_need[i] > n) n = c->cbuf_need[i];
        if ((info->cbuf_dynamic & (1u << i)) && n < 4096) n = 4096;
        if (!n) n = 1;
        sp(&s, "layout(std140, set = %d, binding = %d) uniform CB%d { uvec4 cb%d[%u]; };\n", set, i, i, i, n);
    }
    if (info->uses_gmem)
        sp(&s, "layout(std430, set = %d, binding = %d) %sbuffer GMem { uint w[]; } gmem_;\n", set, GMEM_BUFFER_INDEX,
           c->uses_gmem_store ? "" : "readonly ");
    if (info->uses_gmem && c->uses_gdev)
        sp(&s, "layout(std430, set = %d, binding = %d) %sbuffer GDev { uint w[]; } gdev_;\n", set, GDEV_BUFFER_INDEX,
           c->uses_gmem_store ? "" : "readonly ");
    info->writes_gmem = c->uses_gmem_store;
    if (c->uses_rscale) {
        sp(&s, "layout(std140, set = %d, binding = 18) uniform RScale { vec4 rscale[65]; };\n", set);
        info->uses_rscale = true;
    }
    bool tint = stage == STAGE_FRAGMENT && !dbg && (c->omap_target & 0x7) == 0x7 && (!key || key->rt_kind[0] == 0);
    if (tint) {
        sp(&s, "layout(std140, set = %d, binding = 19) uniform Tint { vec4 ltint[2]; };\n", set);
        info->uses_tint = true;
    }
    for (int i = 0; i < info->ntex; i++) {
        ShaderTexture *t = &info->tex[i];
        sp(&s, "layout(set = %d, binding = %d) uniform %s t%d;\n", set, 32 + i, glsl_sampler_type(t), i);
        // size of level l as (width, height, depth or layers)
        switch (t->type) {
        case TEXTYPE_1D: sp(&s, "ivec3 tsize%d(int l) { return ivec3(textureSize(t%d, l), 1, 1); }\n", i, i); break;
        case TEXTYPE_1D_ARRAY: sp(&s, "ivec3 tsize%d(int l) { ivec2 s = textureSize(t%d, l); return ivec3(s.x, 1, s.y); }\n", i, i); break;
        case TEXTYPE_3D: case TEXTYPE_2D_ARRAY: case TEXTYPE_CUBE_ARRAY:
            sp(&s, "ivec3 tsize%d(int l) { return textureSize(t%d, l); }\n", i, i); break;
        default: sp(&s, "ivec3 tsize%d(int l) { return ivec3(textureSize(t%d, l), 1); }\n", i, i);
        }
        if (i < 32 && (c->ssf_slots & (1u << i)))   // area average, each texel weighted by its coverage of guest texel c;
            // averaged through x / (1 + max(x)) and back, so HDR highlights do not swamp their edges
            sp(&s, "vec4 ssf%d(ivec2 c, vec4 s) {\n"
                   "  if (s.x == 1.0f && s.y == 1.0f) return texelFetch(t%d, c, 0);\n"
                   "  ivec2 mx = textureSize(t%d, 0) - 1;\n"
                   "  vec2 lo = vec2(c) * s.xy, hi = lo + s.xy;\n"
                   "  ivec2 a = ivec2(floor(lo)), b = min(ivec2(ceil(hi)) - 1, a + 5);\n"
                   "  vec4 acc = vec4(0.0f); float ws = 0.0f;\n"
                   "  for (int y = a.y; y <= b.y; y++) {\n"
                   "    float wy = min(hi.y, float(y + 1)) - max(lo.y, float(y));\n"
                   "    for (int x = a.x; x <= b.x; x++) {\n"
                   "      float w = wy * (min(hi.x, float(x + 1)) - max(lo.x, float(x)));\n"
                   "      vec4 v = texelFetch(t%d, clamp(ivec2(x, y), ivec2(0), mx), 0);\n"
                   "      v.rgb /= 1.0f + max(max(v.r, v.g), max(v.b, 0.0f));\n"
                   "      acc += w * v; ws += w;\n"
                   "    }\n"
                   "  }\n"
                   "  acc /= max(ws, 1e-6f);\n"
                   "  acc.rgb /= max(1.0f - max(max(acc.r, acc.g), acc.b), 1e-4f);\n"
                   "  return acc;\n"
                   "}\n", i, i, i, i);
    }
    if (stage == STAGE_VERTEX) {
        for (int i = 0; i < 32; i++) {
            if (!(c->in_attr_mask & (1u << i))) continue;
            int kind = key ? key->attr_kind[i] : 0;
            sp(&s, "layout(location = %d) in %s a%d;\n", i, kind == 1 ? "ivec4" : kind == 2 ? "uvec4" : "vec4", i);
        }
        // depth pre-pass + EQUAL test needs identical positions from different shaders: invariant, and precise so no
        // compiler fuses or reorders the math feeding it (MSL gets this from preserveInvariance; without it foliage
        // drawn twice flickered black)
        sp(&s, "invariant gl_Position;\nprecise gl_Position;\n");
        for (int i = 0; i < 32; i++)
            if (c->out_attr_mask & (1u << i)) sp(&s, "layout(location = %d) out vec4 v%d;\n", i, i);
    } else if (stage == STAGE_FRAGMENT) {
        for (int i = 0; i < 32; i++) {
            if (!(c->in_attr_mask & (1u << i))) continue;
            const char *interp = (c->flat_mask & (1u << i)) ? "flat " : (c->noperspective_mask & (1u << i)) ? "noperspective " : "";
            sp(&s, "layout(location = %d) %sin vec4 v%d;\n", i, interp, i);
            if (c->use_imap && (c->flat_mask | c->noperspective_mask) & (1u << i)) {  // one qualifier per vec4 in GLSL
                bool persp = false;
                for (int k = 0; k < 4; k++) persp |= ((c->imap[i] >> (k * 2)) & 3) == 2;
                static int mixed;
                if (persp && mixed++ < 8) printf("[shader] fs input v%d mixes perspective and %s components (imap %02x)\n", i, interp, c->imap[i]);
            }
        }
        for (int rt = 0; rt < 8; rt++) {
            if (!((c->omap_target >> (rt * 4)) & 0xF)) continue;
            int kind = key ? key->rt_kind[rt] : 0;
            sp(&s, "layout(location = %d) out %s c%d;\n", rt, kind == 1 ? "ivec4" : kind == 2 ? "uvec4" : "vec4", rt);
        }
        if (c->omap_depth) info->writes_depth = true;
    } else {
        sp(&s, "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n");
        if (c->uses_shared) sp(&s, "shared uint smem[12288];\n");   // 48 KiB, the Maxwell maximum (size not read from the program)
    }
    sp(&s, "void main() {\n");
    if (c->trace && stage == STAGE_VERTEX) sp(&s, "  bool trace_on = int(gl_VertexIndex) == %d;\n", maxwell_vtrace_vid);
    else if (c->trace) {
        // a second pixel (LARECOMP_DEBUG_SHADER=<hash>,x,y,x2,y2): its lines start after a "[trace] at x2,y2" marker line
        if (maxwell_trace_x2 >= 0)
            sp(&s, "  bool trace_on = all(equal(uvec2(gl_FragCoord.xy), uvec2(%du, %du))) || all(equal(uvec2(gl_FragCoord.xy), uvec2(%du, %du)));\n",
               maxwell_trace_x, maxwell_trace_y, maxwell_trace_x2, maxwell_trace_y2);
        else sp(&s, "  bool trace_on = all(equal(uvec2(gl_FragCoord.xy), uvec2(%du, %du)));\n", maxwell_trace_x, maxwell_trace_y);
        // the incoming fragment depth too (pc 0, "r999"): which depth an EQUAL-tested quad really carries
        if (stage == STAGE_FRAGMENT) sp(&s, "  if (trace_on) debugPrintfEXT(\"[trace] at %%u,%%u r999=%%08x\", uint(gl_FragCoord.x), uint(gl_FragCoord.y), floatBitsToUint(gl_FragCoord.z));\n");
    }
    if (stage == STAGE_VERTEX) {
        sp(&s, "  gl_Position = vec4(0, 0, 0, 1);\n");
        for (int i = 0; i < 32; i++)
            if (c->out_attr_mask & (1u << i)) sp(&s, "  v%d = vec4(0, 0, 0, 1);\n", i);
        if (info->writes_point_size) sp(&s, "  gl_PointSize = 0.0f;\n");
    }
    for (int i = 0; i < 256; i++)
        if (c->reg_used[i]) sp(&s, "  uint r%d = 0u;\n", i);
    for (int i = 0; i < 7; i++)
        if (c->pred_used[i]) sp(&s, "  bool p%d = false;\n", i);
    sp(&s, "  precise float pf_;\n");   // (set_reg_precise)
    if (c->uses_cc) sp(&s, "  uint cc = 0u; bool ccz = false, ccs = false;\n");
    if (c->uses_local) sp(&s, "  uint lmem[%d]; for (int li_ = 0; li_ < %d; li_++) lmem[li_] = 0u;\n", LMEM_WORDS, LMEM_WORDS);
    if (dbg) sp(&s, maxwell_dbg_pc == 0xffffffffu ? "  float dbgv = 1.0f;\n" : "  float dbgv = -1.0f;\n");
    if (any_branch) {
        sp(&s, "  uint pc = 8u; uint ssy[16]; uint ssy_n = 0u; uint pbk[16]; uint pbk_n = 0u; uint pcnt[16]; uint pcnt_n = 0u;\n");
        sp(&s, "  bool exit_now = false;\n");
        // LARECOMP_GUARD_LIMIT=<n>: the dispatch loop's iteration cap (default 100000); LARECOMP_GUARD_VIS=1 paints
        // fragments that hit it with their stuck pc (R = pc>>3 & 0xff, G = pc>>11 & 0xff, B = 1) in the first float RT
        if (c->dag_why) sp(&s, "  // no DAG guard: backward branch / loop / call at pc 0x%x\n", c->dag_why - 1);
        sp(&s, "  uint guard = 0u;\n  for (; guard < %luu; guard++) {\n", c->dag_guard ? (unsigned long)c->dag_guard : guard_limit());
        if (c->minpc) sp(&s, "  if (pc != subgroupMin(pc)) continue;   // min-pc scheduling (reconvergence)\n");
        sp(&s, "  switch (pc) {\n");
        sp(&s, "%s", body.buf ? body.buf : "");
        sp(&s, "    exit_now = true; break;\n  default: exit_now = true; break;\n  }\n  if (exit_now) break;\n  }\n");
    } else {
        sp(&s, "  do {\n%s  } while (false);\n", body.buf ? body.buf : "");
    }
    free(body.buf);
    if (stage == STAGE_VERTEX) {
        if (c->trace) {   // vertex trace: every vertex far outside the view (or w <= 0): its inputs + clip position
            sp(&s, "  if (abs(gl_Position.y) > 1.5f * abs(gl_Position.w) || abs(gl_Position.x) > 1.5f * abs(gl_Position.w) || gl_Position.w <= 0.0f || trace_on)\n");
            sp(&s, "    debugPrintfEXT(\"[vtrace] vid %%d inst %%d pos %%f %%f %%f %%f");
            for (int i = 0; i < 32; i++) if (c->in_attr_mask & (1u << i)) sp(&s, " a%d %%08x %%08x %%08x %%08x", i);
            sp(&s, "\", int(gl_VertexIndex), int(gl_InstanceIndex), gl_Position.x, gl_Position.y, gl_Position.z, gl_Position.w");
            for (int i = 0; i < 32; i++)
                if (c->in_attr_mask & (1u << i)) {
                    int kind = key ? key->attr_kind[i] : 0;
                    for (int k = 0; k < 4; k++) sp(&s, kind ? ", uint(a%d[%d])" : ", floatBitsToUint(a%d[%d])", i, k);
                }
            sp(&s, ");\n");
        }
        if (key && key->flip_y) sp(&s, "  gl_Position.y = -gl_Position.y;\n");
        if (key && key->depth_minus_one) sp(&s, "  gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5f;\n");
    } else if (stage == STAGE_FRAGMENT) {
        int r = 0;
        for (int rt = 0; rt < 8; rt++) {
            uint32_t m = (c->omap_target >> (rt * 4)) & 0xF;
            if (!m) continue;
            int kind = key ? key->rt_kind[rt] : 0;
            const char *cv = kind == 1 ? "int" : kind == 2 ? "uint" : "";
            for (int comp = 0; comp < 4; comp++) {
                if (!(m & (1u << comp))) continue;
                if (dbg) {
                    if (comp == 3) sp(&s, "  c%d[%d] = %s(1.0f);\n", rt, comp, cv);
                    else sp(&s, "  c%d[%d] = %s(dbgv);\n", rt, comp, cv);
                } else if (r < 256 && c->reg_used[r]) {
                    if (kind == 0) sp(&s, "  c%d[%d] = uintBitsToFloat(r%d);\n", rt, comp, r);
                    else sp(&s, "  c%d[%d] = %s(r%d);\n", rt, comp, cv, r);
                } else {
                    sp(&s, "  c%d[%d] = 0;\n", rt, comp);
                }
                r++;
            }
        }
        if (c->solid && (c->omap_target & 0xF)) sp(&s, "  c0 = vec4(1.0f, 0.0f, 1.0f, 1.0f);\n");
        if (any_branch && getenv("LARECOMP_GUARD_VIS"))
            for (int rt = 0; rt < 8; rt++)
                if (((c->omap_target >> (rt * 4)) & 0xF) && !(key && key->rt_kind[rt])) {
                    sp(&s, "  if (guard >= %luu) c%d = vec4(float((pc >> 3u) & 0xFFu) / 255.0f, float((pc >> 11u) & 0xFFu) / 255.0f, 1.0f, 1.0f);\n",
                       c->dag_guard ? (unsigned long)c->dag_guard : guard_limit(), rt);
                    break;
                }
        if (tint)
            sp(&s, "  if (ltint[0].w > 0.0f) { vec3 g = c0.rgb; float l = dot(g, vec3(0.299f, 0.587f, 0.114f));\n"
                   "    c0.rgb = mix(g, ltint[0].rgb * pow(l, ltint[1].w) + ltint[1].rgb * l, ltint[0].w); }\n");
        if (c->omap_depth) {
            if (c->reg_used[r + 1]) sp(&s, "  gl_FragDepth = uintBitsToFloat(r%d);\n", r + 1);
            else sp(&s, "  gl_FragDepth = gl_FragCoord.z;\n");
        }
    }
    sp(&s, "}\n");
    info->msl = s.buf;
    info->attr_mask = c->in_attr_mask;
    info->out_mask = stage == STAGE_FRAGMENT ? c->omap_target : c->out_attr_mask;
    free(c->body.buf);
    free(c);
    return info;
}

// BRX Ra + imm at slot i: the targets it can reach, marked 2 in mark[]. NVIDIA's switch lowering (as Eden's
// TrackIndirectBranchTable reads it): IMNMX.U32 Rx, Ry, bound; SHL Rx, Rx, 2; LDC Ra, c[0x1][Rx + base]; BRX Ra - k. The
// LDC that last wrote Ra must read c1 with a register index; every word of c1 from base on is taken as a possible entry
// (a superset of the table: extra labels only split blocks). false: not that pattern (the caller marks every slot).
static bool brx_targets(const uint64_t *code, size_t i, size_t end, uint8_t *mark) {
    if (!maxwell_const_data || maxwell_const_size < 4) return false;
    uint64_t w = code[i];
    int areg = (int)bits(w, 8, 8);
    if (areg == RZ) return false;
    for (size_t k = i; k-- > 0 && k + 24 > i;) {
        if (k % 4 == 0) continue;
        uint64_t v = code[k];
        if ((v & 0xfff0000000000000ull) == 0xe240000000000000ull || (v & 0xfff0000000000000ull) == 0xe250000000000000ull ||
            (v & 0xfff8000000000000ull) == 0xf0f8000000000000ull || (v & 0xfff0000000000000ull) == 0xe340000000000000ull ||
            (v & 0xfff0000000000000ull) == 0xe350000000000000ull)
            return false;   // (control flow in between: not a straight-line table lookup)
        if ((int)bits(v, 0, 8) != areg) continue;
        // the last instruction with Ra as its destination field: must be an unpredicated 32-bit LDC from c1, register-indexed
        if ((v & 0xfff8000000000000ull) != 0xef90000000000000ull) return false;
        if (bits(v, 16, 4) != PT || bits(v, 36, 5) != 1 || bits(v, 44, 2) || bits(v, 48, 3) != 4 || (int)bits(v, 8, 8) == RZ) return false;
        int64_t base = sbits(v, 20, 16);
        uint32_t brx_pc = (uint32_t)(i * 8);
        int64_t imm = sbits(w, 20, 24);
        int n = 0;
        for (int64_t o = base < 0 ? 0 : base; o + 4 <= (int64_t)maxwell_const_size; o += 4) {
            uint32_t e;
            memcpy(&e, maxwell_const_data + o, 4);
            uint32_t t = brx_pc + 8 + e + (uint32_t)imm;
            if (((t >> 3) & 3u) == 0u) t += 8u;
            if ((t & 7u) || t / 8 >= end) continue;
            if (!mark[t / 8]) mark[t / 8] = 2;   // (never downgrades a BRA / SSY target)
            n++;
        }
        return n > 0;
    }
    return false;
}

// Flat plan (2026-10-16, Deck d10: the cloud box fs 195655ef = 5.5 of ~15 ms GPU, a pc-loop shader with only forward
// branches): when every branch goes forward (BRA ahead, SSY / SYNC, EXIT, KIL; no BRX, PBK / BRK, PCNT / CONT, calls) the
// blocks run in address order, each guarded by a flag its predecessors set: "if (f_<pc>) { bool tk_ = false; ... }"; a
// taken branch sets tk_ and the target's flag, a block that ends without a taken branch sets the next block's flag. Address
// order is what min-pc scheduling gives (lanes meet at a join before it runs), so reconvergence and quad derivatives behave
// the same, without the loop, subgroupMin or the switch. Each SYNC's target comes from the SSY stack simulated along the
// edges; a join reached with two different stacks, a SYNC on an empty stack or any other flow op -> false (pc loop).
// bstart[i] = 1 marks block starts. LARECOMP_NO_FLAT=1: off.
// NVIDIA (2026-10-19, driver 617.14, RTX 5070; CURRENT.md): its compiler turns some programs into a jump to an invalid
// address (GPU hang, device fault "ip invalid") in one form and not the other, chaotically (one statement more or less
// flips it): the flat form of d4201cbc's VS faults on every draw and its pc-loop form draws (run477/478); with the flat
// plan off for all, the pc-loop forms of 927c / 5549's programs faulted instead (3 hangs in ~65 min at 2x, runs
// 479/480/486) while their flat forms ran the user's play hours. So the flat plan stays the default there and the VS
// below (code hash = FNV-1a of the 0x200 bytes after the 0x30 header, exec.c prog_code_hash) take the pc loop.
bool maxwell_nvidia;
static const uint64_t nv_pcloop_vs[] = {0x49c2e0dcab7071d4ull};   // (fs d4201cbc: grass layer, Hyrule surface)
static bool nv_pcloop(const uint8_t *blob, size_t max_size, int stage) {
    if (!maxwell_nvidia || stage != STAGE_VERTEX || max_size <= 0x30) return false;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0, n = max_size >= 0x230 ? 0x200 : max_size - 0x30; i < n; i++) { h ^= blob[0x30 + i]; h *= 0x100000001B3ull; }
    for (size_t i = 0; i < sizeof nv_pcloop_vs / sizeof nv_pcloop_vs[0]; i++) if (h == nv_pcloop_vs[i]) return true;
    return false;
}
// Why the last flat_plan said no (tools/shadertool.c statistics): reason + the pc where it was found.
static MAXWELL_TLS const char *maxwell_flat_why;
static MAXWELL_TLS uint32_t maxwell_flat_why_pc;
#define FLAT_NO(WHY, PC) do { if (ok) { maxwell_flat_why = (WHY); maxwell_flat_why_pc = (uint32_t)(PC); } ok = false; } while (0)
static bool flat_plan(const uint64_t *code, size_t end, const uint8_t *label, uint8_t *bstart, uint32_t *sync_t) {
    enum { MAXD = 8 };
    maxwell_flat_why = NULL;
    typedef struct { int8_t n; uint32_t t[MAXD]; } Stk;   // n < 0: no edge into this block yet
    Stk *in = malloc(sizeof(Stk) * (end + 1));
    if (!in) return false;
    bool ok = true;
    for (size_t i = 0; i <= end; i++) in[i].n = -1;
    for (size_t i = 0; i < end && ok; i++) {
        if (i % 4 == 0) continue;
        uint64_t w = code[i];
        uint32_t pc = (uint32_t)(i * 8);
        uint32_t top = (uint32_t)(w >> 52);
        bool is_bra = (w & 0xfff0000000000020ull) == 0xe240000000000000ull;
        bool is_sync = (w & 0xfff8000000000000ull) == 0xf0f8000000000000ull;
        bool is_ssy = (w & 0xfff0000000000020ull) == 0xe290000000000000ull;
        if (label[i]) bstart[i] = 1;
        if (is_bra) {
            int64_t t = norm_target((int64_t)pc + 8 + sbits(w, 20, 24));
            if (t <= pc || (size_t)(t / 8) >= end) FLAT_NO(t <= pc ? "backward BRA" : "BRA out of range", pc);   // (backward: a loop)
        } else if (is_ssy) {
            int64_t t = norm_target((int64_t)pc + 8 + sbits(w, 20, 24));
            if (t <= pc || (size_t)(t / 8) >= end) FLAT_NO("backward SSY", pc);
        } else if (top >= 0xe20 && top <= 0xe3f && top != 0xe30 && top != 0xe33) {
            static const char *const flow_names[32] = {"JMX", "JMP", "JCAL", "PEXIT", "BRA.CC", "BRX", "CAL", "PRET", "", "SSY.C", "PBK",
                "PCNT", "", "", "", "", "EXIT", "LONGJMP", "RET", "KIL", "BRK", "CONT", "RTT", "SAM", "RAM", "IDE", "BPT"};
            const char *nm = flow_names[top - 0xe20];
            FLAT_NO(nm && *nm ? nm : "other flow op", pc);   // (any other flow op: BRX, JMP/JMX, CAL/JCAL, PRET/RET, PBK/BRK, PCNT/CONT, PEXIT, ... or BRA.CC)
        }
        if (is_bra || is_sync || top == 0xe30) {   // (EXIT) ends its block: the next slot starts one
            size_t n = i + 1;
            if (n % 4 == 0) n++;
            if (n < end) bstart[n] = 1;
        }
    }
    bstart[1] = 1;
    // SSY stack along the edges, blocks in address order (every edge goes forward)
    Stk cur = {0};
    bool live = false;
    #define FLAT_EDGE(T, ST) do { size_t ti_ = (T) / 8; if (ti_ >= end) { ok = false; break; } \
        if (in[ti_].n < 0) in[ti_] = (ST); \
        else if (in[ti_].n != (ST).n || memcmp(in[ti_].t, (ST).t, sizeof(uint32_t) * (size_t)(ST).n)) FLAT_NO("join with two SSY stacks", ti_ * 8); } while (0)
    in[1].n = 0;
    for (size_t i = 1; i < end && ok; i++) {
        if (i % 4 == 0) continue;
        uint64_t w = code[i];
        uint32_t pc = (uint32_t)(i * 8);
        if (bstart[i]) {
            if (live && i > 1) FLAT_EDGE(pc, cur);   // fall-through from the previous block (when it can)
            live = in[i].n >= 0;
            if (live) cur = in[i];
        }
        if (!live) continue;
        bool pt = bits(w, 16, 4) == PT;   // unconditional
        if ((w & 0xfff0000000000020ull) == 0xe290000000000000ull) {   // SSY
            if (cur.n >= MAXD) { FLAT_NO("SSY stack too deep", pc); break; }
            cur.t[cur.n++] = norm_target((int64_t)pc + 8 + sbits(w, 20, 24));
        } else if ((w & 0xfff0000000000020ull) == 0xe240000000000000ull) {   // BRA
            if (bits(w, 16, 4) == (PT | 8)) continue;   // (@!PT: never)
            FLAT_EDGE(norm_target((int64_t)pc + 8 + sbits(w, 20, 24)), cur);
            if (pt) live = false;
        } else if ((w & 0xfff8000000000000ull) == 0xf0f8000000000000ull) {   // SYNC
            if (bits(w, 16, 4) == (PT | 8)) continue;
            if (cur.n <= 0) { FLAT_NO("SYNC on empty stack", pc); break; }
            Stk popped = cur;
            popped.n--;
            uint32_t t = cur.t[cur.n - 1];
            if (t <= pc) { FLAT_NO("backward SYNC", pc); break; }
            sync_t[i] = t;
            FLAT_EDGE(t, popped);
            if (pt) live = false;
        } else if ((uint32_t)(w >> 52) == 0xe30) {   // EXIT
            if (pt) live = false;
        }
    }
    #undef FLAT_EDGE
    free(in);
    if (!ok && !maxwell_flat_why) maxwell_flat_why = "?";
    return ok;
}

ShaderInfo *maxwell_translate(const uint8_t *blob, size_t max_size, int stage, const ShaderKey *key) {
    ShaderInfo *info = calloc(1, sizeof *info);
    info->stage = stage;
    Ctx *c = calloc(1, sizeof *c);
    c->info = info;
    c->key = key;
    c->stage = stage;
    c->ipa_mode = -1;
    c->glsl = key && key->glsl;
    c->trace = ((maxwell_trace_next && stage == STAGE_FRAGMENT) || (maxwell_vtrace_next && stage == STAGE_VERTEX)) && c->glsl;
    if (c->trace && stage == STAGE_VERTEX) { maxwell_vtrace_next = false; printf("[trace] translating vertex shader %p: vertex trace (vid %d)\n", (void *)blob, maxwell_vtrace_vid); }
    {
        const char *nd = getenv("LARECOMP_NO_DISCARD");
        c->nodiscard = (nd && !strcmp(nd, "1")) || (maxwell_nodiscard_next && stage == STAGE_FRAGMENT);
        if (stage == STAGE_FRAGMENT) maxwell_nodiscard_next = false;
        c->solid = maxwell_solid_next && stage == STAGE_FRAGMENT;
        if (c->solid) c->nodiscard = true;
        if (stage == STAGE_FRAGMENT) maxwell_solid_next = false;
        c->ssfetch = maxwell_ssfetch_next && stage == STAGE_FRAGMENT && c->glsl;
        if (stage == STAGE_FRAGMENT) maxwell_ssfetch_next = false;
    }
    if (c->trace) { maxwell_trace_next = false; printf("[trace] translating fragment shader %p with a pixel trace at %d,%d\n", (void *)blob, maxwell_trace_x, maxwell_trace_y); }
    const uint8_t *sph = blob + 0x30;
    const uint8_t *code = blob + 0x80;
    // compute programs have no SPH. LASRecomp's had no NVN header either (code at the blob); TotK's carry a 0x100-byte
    // header with magic 0x12345679 (graphics: 0x12345678), zeros up to +0x100 (run81: first nonzero qword at +0x100,
    // and 0x100 + code size (control+0x6f8 = 0x1e00) = const offset (control+0x700 = 0x1f00))
    if (stage == STAGE_COMPUTE) code = *(const uint32_t *)blob == 0x12345679 ? blob + 0x100 : blob;
    if (stage == STAGE_FRAGMENT) {
        c->omap_target = *(const uint32_t *)(sph + 0x48);
        c->omap_depth = (*(const uint32_t *)(sph + 0x4C) >> 1) & 1;
        c->use_imap = !getenv("LARECOMP_NO_IMAP");
        memcpy(c->imap, sph + 0x18, 32);
    }
    // find the end: the last EXIT followed by self-branch padding, or max_size
    size_t n = (max_size - (code - blob)) / 8;
    if (n > 65536) n = 65536;
    size_t end = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t w = ((const uint64_t *)code)[i];
        if (i % 4 == 0) continue;
        // bra self: the padding after the last EXIT. Its offset is -8, or -16 right after a scheduling word (the target is
        // that word, which norm_target moves to the next slot): until 2026-10-16 only -8 was seen, so 178 of 593 shaders in
        // the Deck d5 dump kept the padding as a reachable self-loop label -> the whole shader wrapped in the pc loop
        // (+ subgroupMin per turn in fragment shaders) although it has no other branch.
        if ((w & 0xfff0000000000000ull) == 0xe240000000000000ull && end &&
            norm_target((int64_t)(i * 8) + 8 + sbits(w, 20, 24)) == (uint32_t)(i * 8)) break;
        if (w == 0 && i > 4) break;
        end = i + 1;
    }
    // collect branch targets (labels)
    // label[i]: 1 = target of BRA / SSY / PBK / PCNT (or the entry), 2 = only a possible BRX target. In min-pc mode only
    // kind 1 gets the explicit jump in front of its case (reconvergence point); kind 2 is entered by fall-through as well.
    // Before 2026-10-16 every BRX made every instruction a kind-1 label: one pc-loop turn (subgroupMin + a linear walk of
    // the switch) per instruction. Deck d5: fs e5210cf2e1279deb (1-pixel, 391 cases) 4.4 ms of a ~28 ms GPU frame.
    uint8_t *label = calloc(end + 4, 1);
    label[1] = 1;
    bool any_branch = false, brx_any = false, backward = false;
    for (size_t i = 0; i < end; i++) {
        if (i % 4 == 0) continue;
        uint64_t w = ((const uint64_t *)code)[i];
        uint32_t pc = (uint32_t)(i * 8);
        bool is_flow = (w & 0xfff0000000000020ull) == 0xe240000000000000ull ||
                       (w & 0xfff0000000000020ull) == 0xe290000000000000ull ||
                       (w & 0xfff0000000000020ull) == 0xe2a0000000000000ull ||
                       (w & 0xfff0000000000020ull) == 0xe2b0000000000000ull;
        if (is_flow) {
            int64_t t = norm_target((int64_t)pc + 8 + sbits(w, 20, 24));
            if (t >= 0 && (size_t)(t / 8) < end) label[t / 8] = 1;
            if (t <= (int64_t)pc || (w & 0xfff0000000000000ull) == 0xe2b0000000000000ull) {   // (PCNT: a loop)
                if (!backward) c->dag_why = pc + 1;
                backward = true;
            }
            any_branch = true;
        }
        // CONT, CAL, JCAL, RET, JMX, JMP: loops / calls (no static iteration bound)
        if ((w & 0xfff0000000000000ull) == 0xe350000000000000ull || (w & 0xfff0000000000000ull) == 0xe260000000000000ull ||
            (w & 0xfff0000000000000ull) == 0xe220000000000000ull || (w & 0xfff0000000000000ull) == 0xe320000000000000ull ||
            (w & 0xfff0000000000000ull) == 0xe200000000000000ull || (w & 0xfff0000000000000ull) == 0xe210000000000000ull) {
            if (!backward) c->dag_why = pc + 1;
            backward = true;
        }
        if ((w & 0xfff0000000000000ull) == 0xe250000000000000ull) {  // BRX: its jump table's targets, or any instruction
            any_branch = true;
            if (!brx_targets((const uint64_t *)code, i, end, label)) brx_any = true;
        }
        if ((w & 0xfff8000000000000ull) == 0xf0f8000000000000ull || (w & 0xfff0000000000000ull) == 0xe340000000000000ull ||
            (w & 0xfff0000000000000ull) == 0xe350000000000000ull)
            any_branch = true;
        (void)is_branch_target_op;
    }
    if (brx_any)
        for (size_t i = 0; i < end; i++) if (!label[i] && i % 4) label[i] = 2;
    if (brx_any && !maxwell_const_data) c->info->brx_blind = true;
    // DAG guard (2026-10-17, Deck d35: the cloud VS 7f..3f1e00, forward branches + one BRX, took 5.4 ms for 108 vertices
    // with the 100000 cap and 0.09 ms with LARECOMP_GUARD_LIMIT=1000 -> some lanes cycle until the cap although no
    // static path loops): without backward branches, loops or calls every block runs at most once per invocation, so
    // 2 x labels + 16 turns is a safe cap (BRX / pop targets ahead of their jump). LARECOMP_NO_DAG_GUARD=1: off.
    c->dag_guard = 0;
    {
        static int no_dag = -1;
        if (no_dag < 0) no_dag = getenv("LARECOMP_NO_DAG_GUARD") != NULL;
        if (any_branch && !backward && !no_dag) {
            uint32_t nl = 0;
            for (size_t i = 0; i < end; i++) if (label[i] && i % 4) nl++;
            c->dag_guard = 2 * nl + 16;
        }
    }
    // translate
    bool dbg = maxwell_dbg_code && blob == maxwell_dbg_code && stage == STAGE_FRAGMENT;
    static int annotate = -1;
    if (annotate < 0) annotate = getenv("LARECOMP_SAVE_MSL") != NULL;
    c->branchy = any_branch;
    // Min-pc scheduling (GLSL fragment shaders with branches): each turn of the pc loop runs only the lanes at the
    // subgroup's lowest pc, so lanes that took different paths meet again at the join block and run it together, as the
    // Maxwell SSY/SYNC reconvergence does. Without it a quad whose lanes took different paths reaches a TMML / implicit-LOD
    // texture block in different loop turns, and its derivatives (so its mip level) are garbage: TotK's cloud boxes
    // (fs 454c58a8, paths 8->1240 / 8->904->1240 / 8->904->984->1240) came out as 2x2 blocks of one value, bright white
    // ones among them (run280 cloud layer) -> white squares in the clouds. Case labels get an explicit jump in front so no
    // lane falls through into the next block on its own. LARECOMP_NO_MINPC=1: off (the old behaviour).
    static int no_minpc = -1;
    if (no_minpc < 0) no_minpc = getenv("LARECOMP_NO_MINPC") != NULL;
    c->minpc = any_branch && c->glsl && stage == STAGE_FRAGMENT && !no_minpc;
    static int no_flat = -1;
    if (no_flat < 0) no_flat = getenv("LARECOMP_NO_FLAT") != NULL;
    uint8_t *bstart = NULL;
    if (any_branch && c->glsl && !brx_any && !no_flat && !dbg && !c->trace && !nv_pcloop(blob, max_size, stage)) {
        bstart = calloc(end + 4, 1);
        c->flat_sync = calloc(end + 4, sizeof(uint32_t));
        if (bstart && c->flat_sync && flat_plan((const uint64_t *)code, end, label, bstart, c->flat_sync)) {
            c->flat = true;
            c->minpc = false;
            c->branchy = false;
            for (size_t i = 2; i < end; i++)   // (the block flags, declared ahead of the first block)
                if (bstart[i] && i % 4) sp(&c->body, "  bool f_%u = false;\n", (uint32_t)(i * 8));
        }
    }
    bool any_label = false;
    for (size_t i = 0; i < end && !c->fail; i++) {
        if (i % 4 == 0) continue;
        uint32_t pc = (uint32_t)(i * 8);
        if (c->flat && bstart[i]) {
            if (any_label) sp(&c->body, "    if (!tk_) f_%u = true;\n  }\n", pc);   // (falls through unless a branch was taken)
            if (i == 1) sp(&c->body, "  { bool tk_ = false;\n");
            else sp(&c->body, "  if (f_%u) { bool tk_ = false;\n", pc);
            any_label = true;
        } else if (!c->flat && any_branch && label[i]) {
            if (c->minpc && any_label && label[i] == 1) sp(&c->body, "    pc = %uu; continue;\n", pc);   // (no fall-through; see above)
            sp(&c->body, "  case %uu:\n", pc);
            any_label = true;
        }
        if (annotate) sp(&c->body, "    // pc %x: %016llx\n", pc, (unsigned long long)((const uint64_t *)code)[i]);
        emit_insn(c, ((const uint64_t *)code)[i], pc);
        if (dbg && pc == maxwell_dbg_pc) {
            int rr = maxwell_dbg_reg % 1000, half = maxwell_dbg_reg / 1000;  // +1000: low f16, +2000: high f16
            c->reg_used[rr] = 1;
            if (half) sp(&c->body, c->glsl ? "    dbgv = unpackHalf2x16(r%d)[%d];\n" : "    dbgv = float(as_type<half2>(r%d)[%d]);\n", rr, half - 1);
            else sp(&c->body, "    dbgv = as_type<float>(r%d);\n", rr);
        }
        info->num_insts++;
    }
    if (c->flat && any_label) sp(&c->body, "  }\n");
    free(label);
    free(bstart);
    free(c->flat_sync);
    c->flat_sync = NULL;
    if (c->fail) { free(c->body.buf); free(c); return info; }
    if (c->glsl) return glsl_finish(c, any_branch && !c->flat, dbg);

    // ---------------------------------------------------------- MSL source
    Str s = {0};
    sp(&s, "#include <metal_stdlib>\nusing namespace metal;\n");
    sp(&s, "static inline uint cbr(constant uint4 *cb, uint off) { return cb[off >> 4][(off >> 2) & 3]; }\n");
    sp(&s, "static inline uint bfe_u(uint v, uint ctl) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u) return 0u; if (pos >= 32u) return 0u; return extract_bits(v, pos, min(len, 32u - pos)); }\n");
    sp(&s, "static inline uint bfe_s(uint v, uint ctl) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u) return 0u; if (pos >= 32u) return (int(v) < 0) ? 0xFFFFFFFFu : 0u; return uint(extract_bits(int(v), pos, min(len, 32u - pos))); }\n");
    sp(&s, "static inline uint bfi(uint ins, uint ctl, uint base) { uint pos = ctl & 0xFFu, len = (ctl >> 8) & 0xFFu; if (len == 0u || pos >= 32u) return base; return insert_bits(base, ins, pos, min(len, 32u - pos)); }\n");
    sp(&s, "static inline float fswzadd(float a, float b, uint swz, uint lane) { uint m = (swz >> ((lane & 3u) * 2u)) & 3u; float ma = m == 1u ? -1.0f : (m == 3u ? 0.0f : 1.0f); float mb = m == 2u ? -1.0f : 1.0f; return a * ma + b * mb; }\n");
    sp(&s, "static inline uint2 rsc(uint2 c, float4 s) { return (s.x == 1.0f && s.y == 1.0f) ? c : uint2(floor((float2(c) + 0.5f) * s.xy)); }\n");
    sp(&s, "static inline float2 hx(uint v, uint swz) { if (swz == 1u) return float2(as_type<float>(v)); float2 h = float2(as_type<half2>(v)); return swz == 2u ? h.xx : swz == 3u ? h.yy : h; }\n");
    sp(&s, "static inline uint hpack(float2 r, uint merge, uint old) { if (merge == 1u) return as_type<uint>(r.x); uint lo = uint(as_type<ushort>(half(r.x))), hi = uint(as_type<ushort>(half(r.y))); if (merge == 2u) return (old & 0xFFFF0000u) | lo; if (merge == 3u) return (old & 0xFFFFu) | (hi << 16); return lo | (hi << 16); }\n");
    sp(&s, "static inline uint prmt(uint a, uint b, uint sel) { uint r = 0; for (uint i = 0; i < 4; i++) { uint s = (sel >> (i * 4)) & 7u; uint byte = s < 4 ? (a >> (s * 8)) & 0xFFu : (b >> ((s - 4) * 8)) & 0xFFu; r |= byte << (i * 8); } return r; }\n");

    // inputs/outputs
    if (stage == STAGE_VERTEX) {
        sp(&s, "struct VIn {\n");
        for (int i = 0; i < 32; i++) {
            if (!(c->in_attr_mask & (1u << i))) continue;
            int kind = key ? key->attr_kind[i] : 0;
            sp(&s, "  %s4 a%d [[attribute(%d)]];\n", kind == 1 ? "int" : kind == 2 ? "uint" : "float", i, i);
        }
        sp(&s, "};\n");
        sp(&s, "struct VOut {\n  float4 pos [[position, invariant]];\n");  // depth pre-pass + EQUAL test needs identical positions
        for (int i = 0; i < 32; i++)
            if (c->out_attr_mask & (1u << i)) sp(&s, "  float4 v%d [[user(locn%d)]];\n", i, i);
        if (info->writes_point_size) sp(&s, "  float psize [[point_size]];\n");
        sp(&s, "};\n");
    } else if (stage == STAGE_FRAGMENT) {
        sp(&s, "struct FIn {\n  float4 pos [[position]];\n");
        for (int i = 0; i < 32; i++) {
            if (!(c->in_attr_mask & (1u << i))) continue;
            const char *interp = (c->flat_mask & (1u << i)) ? " [[flat]]" : (c->noperspective_mask & (1u << i)) ? " [[center_no_perspective]]" : "";
            sp(&s, "  float4 v%d [[user(locn%d)]]%s;\n", i, i, interp);
        }
        sp(&s, "};\n");
        sp(&s, "struct FOut {\n");
        for (int rt = 0; rt < 8; rt++) {
            if (!((c->omap_target >> (rt * 4)) & 0xF)) continue;
            int kind = key ? key->rt_kind[rt] : 0;
            sp(&s, "  %s4 c%d [[color(%d)]];\n", kind == 1 ? "int" : kind == 2 ? "uint" : "float", rt, rt);
        }
        if (c->omap_depth) { sp(&s, "  float depth [[depth(any)]];\n"); info->writes_depth = true; }
        sp(&s, "};\n");
    }

    // entry point
    Str args = {0};
    for (int i = 0; i < 18; i++)
        if (info->cbuf_mask & (1u << i)) sp(&args, ", constant uint4 *cb%d [[buffer(%d)]]", i, i);
    if (info->uses_gmem) {
        if (info->cbuf_mask & (1u << GMEM_BUFFER_INDEX)) {
            free(args.buf);
            args = (Str){0};  // (was a use after free + double free of args.buf)
            fail(c, "cbuf %d clashes with gmem", GMEM_BUFFER_INDEX);
        }
        sp(&args, ", device uchar *gmem [[buffer(%d)]]", GMEM_BUFFER_INDEX);
    }
    if (c->uses_rscale && stage != STAGE_VERTEX) {
        sp(&args, ", constant float4 *rscale [[buffer(18)]]");
        info->uses_rscale = true;
    }
    // Runtime highlight (e.g. golden unchecked pots): fragment shaders writing a float color target 0 get a
    // per-draw tint, float4 [0] = colour.rgb + strength, [1] = added glow.rgb (strength 0 = untouched).
    bool tint = stage == STAGE_FRAGMENT && !dbg && (c->omap_target & 0x7) == 0x7 && (!key || key->rt_kind[0] == 0);
    if (tint) {
        sp(&args, ", constant float4 *ltint [[buffer(19)]]");
        info->uses_tint = true;
    }
    for (int i = 0; i < info->ntex; i++) {
        ShaderTexture *t = &info->tex[i];
        const char *tn = tex_type_name(t->type);
        if (t->shadow) sp(&args, ", depth%s<float> t%d [[texture(%d)]], sampler s%d [[sampler(%d)]]", tn, i, i, i, i);
        else sp(&args, ", texture%s<float> t%d [[texture(%d)]], sampler s%d [[sampler(%d)]]", tn, i, i, i, i);
    }
    if (stage == STAGE_VERTEX) {
        sp(&s, "vertex VOut main0(VIn in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]]%s%s) {\n",
           (maxwell_dbg_vs && blob == maxwell_dbg_vs) ? ", device float4 *dbgout [[buffer(29)]]" : "",
           args.buf ? args.buf : "");
        sp(&s, "  VOut out = {};\n  out.pos = float4(0, 0, 0, 1);\n");
        for (int i = 0; i < 32; i++)
            if (c->out_attr_mask & (1u << i)) sp(&s, "  out.v%d = float4(0, 0, 0, 1);\n", i);
    } else if (stage == STAGE_FRAGMENT) {
        sp(&s, "fragment FOut main0(FIn in [[stage_in]], bool front_facing [[front_facing]]%s%s) {\n",
           c->uses_lane ? ", uint lane [[thread_index_in_simdgroup]]" : "", args.buf ? args.buf : "");
    } else {
        sp(&s, "kernel void main0(uint3 tid [[thread_position_in_threadgroup]], uint3 ctaid [[threadgroup_position_in_grid]]%s) {\n",
           args.buf ? args.buf : "");
    }
    free(args.buf);
    for (int i = 0; i < 256; i++)
        if (c->reg_used[i]) sp(&s, "  uint r%d = 0;\n", i);
    for (int i = 0; i < 7; i++)
        if (c->pred_used[i]) sp(&s, "  bool p%d = false;\n", i);
    if (c->uses_cc) sp(&s, "  uint cc = 0; bool ccz = false, ccs = false;\n");
    if (c->uses_local) sp(&s, "  uint lmem[%d] = {};\n", LMEM_WORDS);
    if (dbg) sp(&s, maxwell_dbg_pc == 0xffffffffu ? "  float dbgv = 1.0f;\n" : "  float dbgv = -1.0f;\n");
    if (any_branch) {
        sp(&s, "  uint pc = 8u; uint ssy[16]; uint ssy_n = 0; uint pbk[16]; uint pbk_n = 0; uint pcnt[16]; uint pcnt_n = 0;\n");
        sp(&s, "  bool exit_now = false;\n");
        sp(&s, "  for (uint guard = 0; guard < 100000u; guard++) {\n  switch (pc) {\n");
        sp(&s, "%s", c->body.buf ? c->body.buf : "");
        sp(&s, "    exit_now = true; break;\n  default: exit_now = true; break;\n  }\n  if (exit_now) break;\n  }\n");
    } else {
        sp(&s, "  do {\n%s  } while (false);\n", c->body.buf ? c->body.buf : "");
    }
    if (stage == STAGE_VERTEX) {
        if (key && key->flip_y) sp(&s, "  out.pos.y = -out.pos.y;\n");
        if (key && key->depth_minus_one) sp(&s, "  out.pos.z = (out.pos.z + out.pos.w) * 0.5f;\n");
        if (maxwell_dbg_vs && blob == maxwell_dbg_vs) sp(&s, "  if (iid < 64u && vid < 8u) { dbgout[iid * 8u + vid] = out.pos; dbgout[512u + iid * 8u + vid] = %s; }\n", (c->out_attr_mask & 2u) ? "out.v1" : "float4(0)");
        sp(&s, "  return out;\n}\n");
    } else if (stage == STAGE_FRAGMENT) {
        sp(&s, "  FOut o;\n");
        int r = 0;
        for (int rt = 0; rt < 8; rt++) {
            uint32_t m = (c->omap_target >> (rt * 4)) & 0xF;
            if (!m) continue;
            int kind = key ? key->rt_kind[rt] : 0;
            for (int comp = 0; comp < 4; comp++) {
                if (!(m & (1u << comp))) continue;
                if (dbg) {
                    sp(&s, comp == 3 ? "  o.c%d[%d] = 1.0f;\n" : "  o.c%d[%d] = dbgv;\n", rt, comp);
                } else if (r < 256 && c->reg_used[r]) {
                    if (kind == 0) sp(&s, "  o.c%d[%d] = as_type<float>(r%d);\n", rt, comp, r);
                    else sp(&s, "  o.c%d[%d] = r%d;\n", rt, comp, r);
                } else {
                    sp(&s, "  o.c%d[%d] = 0;\n", rt, comp);
                }
                r++;
            }
        }
        if (tint)
            sp(&s, "  if (ltint[0].w > 0.0f) { float3 g = o.c0.rgb; float l = dot(g, float3(0.299f, 0.587f, 0.114f));\n"
                   "    o.c0.rgb = mix(g, ltint[0].rgb * pow(l, ltint[1].w) + ltint[1].rgb * l, ltint[0].w); }\n");
        if (c->omap_depth) {
            if (c->reg_used[r + 1]) sp(&s, "  o.depth = as_type<float>(r%d);\n", r + 1);
            else sp(&s, "  o.depth = in.pos.z;\n");
        }
        sp(&s, "  return o;\n}\n");
    } else {
        sp(&s, "  return;\n}\n");
    }
    info->msl = s.buf;
    info->attr_mask = c->in_attr_mask;
    info->out_mask = stage == STAGE_FRAGMENT ? c->omap_target : c->out_attr_mask;
    free(c->body.buf);
    free(c);
    return info;
}

void maxwell_free(ShaderInfo *s) {
    if (!s) return;
    free(s->msl);
    free(s);
}

void maxwell_disassemble(const uint8_t *code, size_t max_size, int stage, void *file) {
    (void)code; (void)max_size; (void)stage; (void)file;
}

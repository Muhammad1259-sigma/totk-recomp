// spvgen: GLSL (the subset the TotK Maxwell translator emits) -> SPIR-V 1.6, mirroring glslang 15.1.0 + shaderc 2023.8
// (Vulkan 1.3, optimization zero) instruction for instruction: same function body (spvcmp.py canonical form), so RADV
// gets the same driver input as from shaderc. Anything outside the subset returns NULL and the caller uses shaderc.
// The glslang behaviour mirrored here is noted at each place (GlslangToSpv.cpp / SpvBuilder.cpp / Intermediate.cpp /
// Constant.cpp / propagateNoContraction.cpp / InReadableOrder.cpp of glslang 15.1.0).
#include "spvgen.h"
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- utilities
typedef struct { uint32_t *w; uint32_t n, cap; } Words;
static void wp(Words *v, uint32_t x) {
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 64; v->w = realloc(v->w, v->cap * 4); }
    v->w[v->n++] = x;
}
static void wcat(Words *v, const uint32_t *x, uint32_t n) { for (uint32_t i = 0; i < n; i++) wp(v, x[i]); }

typedef struct Chunk { struct Chunk *next; size_t used, cap; char data[]; } Chunk;
typedef struct G G;
static __thread G *g;
static void *amal(size_t n);

// ---------------------------------------------------------------- types
enum { B_VOID, B_BOOL, B_INT, B_UINT, B_FLOAT, B_DOUBLE, B_INT64, B_UINT64, B_STRUCT, B_SAMPLER };
static int is_f(int b) { return b == B_FLOAT || b == B_DOUBLE; }
static int is_i(int b) { return b == B_INT || b == B_UINT || b == B_INT64 || b == B_UINT64; }
static int is_u(int b) { return b == B_UINT || b == B_UINT64; }
static int is_s(int b) { return b == B_INT || b == B_INT64; }
static int bits(int b) { return (b == B_DOUBLE || b == B_INT64 || b == B_UINT64) ? 64 : 32; }

typedef struct Member { struct Intern *name; struct Ty *t; int builtin; } Member;
typedef struct StructDef {
    int nm; Member m[4]; int block, std430, readonly, is_pervertex;
    uint32_t spv;   // OpTypeStruct id
} StructDef;
typedef struct SampDef { int dim, arrayed, shadow; } SampDef;   // dim: 1 = 2D, 3 = Cube (SPIR-V Dim)
typedef struct Ty {
    uint8_t b, n;       // basic type, vector size (1 = scalar)
    int32_t arr;        // 0 not an array, > 0 sized, -1 runtime
    StructDef *sd; SampDef *samp;
} Ty;

// ---------------------------------------------------------------- interned identifiers, variables
enum { KW_NONE, KW_TYPE, KW_IF, KW_ELSE, KW_DO, KW_WHILE, KW_FOR, KW_SWITCH, KW_CASE, KW_DEFAULT, KW_BREAK, KW_CONTINUE,
       KW_RETURN, KW_DISCARD, KW_DEMOTE, KW_TRUE, KW_FALSE, KW_LAYOUT, KW_IN, KW_OUT, KW_UNIFORM, KW_BUFFER, KW_FLAT,
       KW_INVARIANT, KW_PRECISE, KW_READONLY, KW_CONST, KW_STRUCT, KW_FUNC };
typedef struct Var Var;
typedef struct Intern { struct Intern *next; const char *s; int len; uint32_t h; int kw; Ty kwty; int fn; Var *cur; } Intern;

enum { ST_LOCAL, ST_IN, ST_OUT, ST_UNIFORM, ST_BUFFER, ST_SAMPLER, ST_BUILTIN_IN };
struct Var {
    Intern *name; Ty t; int st; int precise; int uid;
    uint32_t id;                  // SPIR-V variable id (0 = not created yet)
    Var *blk; int member;         // anonymous block member: the block variable and the member index
    int location, set, binding, flat, builtin, invariant;
    Var *shadow;                  // previous binding of the same name (scopes)
    Var *gnext;                   // global list
};

// ---------------------------------------------------------------- AST
enum {   // node kinds / ops (glslang TOperator subset)
    N_CONST = 1, N_SYM,
    // binary
    O_ASSIGN, O_ADD_A, O_SUB_A, O_MUL_A, O_DIV_A, O_MOD_A, O_SHL_A, O_SHR_A, O_AND_A, O_OR_A, O_XOR_A,
    O_ADD, O_SUB, O_MUL, O_DIV, O_MOD, O_VTS, O_SHL, O_SHR, O_AND, O_OR, O_XOR, O_LAND, O_LOR, O_LXOR,
    O_LT, O_GT, O_LE, O_GE, O_EQ, O_NE, O_IDX_D, O_IDX_I, O_IDX_S, O_SWZ,
    // unary
    O_NEG, O_LNOT, O_BNOT, O_PREINC, O_PREDEC, O_POSTINC, O_POSTDEC, O_CONV,
    O_SIN, O_COS, O_EXP2, O_LOG2, O_SQRT, O_ISQRT, O_FLOOR, O_TRUNC, O_RNDEVEN, O_CEIL, O_FRACT, O_ABS, O_ISNAN,
    O_ISINF, O_F2I, O_F2U, O_I2F, O_U2F, O_PACKD, O_UNPACKD, O_BITCNT, O_FMSB, O_ANY, O_ALL, O_SGMIN, O_SGALL, O_ARRLEN,
    // aggregate
    O_MIN, O_MAX, O_CLAMP, O_MIX, O_FMA, O_POW, O_DOT, O_BFE, O_BFI, O_CTOR, O_TEX, O_ATOMIC, O_SGSHUF,
    // other
    O_SEL,
    // statements
    S_SEQ, S_IF, S_LOOP, S_SWITCH, S_CASE, S_DEFAULT, S_BREAK, S_CONTINUE, S_RETURN, S_DEMOTE, S_KILL,
};
typedef union { int64_t i; uint64_t u; double d; } CV;
typedef struct Node Node;
struct Node {
    uint16_t op; uint8_t nc, prec, cnst;   // nc: NoContraction mark; prec: precise qualifier; cnst: EvqConst
    Ty t;
    Node *l, *r, *c;                       // binary l r; unary l; ?: c ? l : r; loop: body l, test c, terminal r
    Node **a; int na;                      // aggregate args / sequence statements
    Var *v;
    CV cv[4];                              // constant components
    uint8_t sw[4], nsw;
    uint8_t test_first;                    // loops
};

// ---------------------------------------------------------------- tokens
enum { T_EOF, T_ID, T_INT, T_FLT, T_P };
typedef struct Tok { uint8_t k; uint8_t bt; uint16_t p; Intern *id; uint64_t iv; double fv; } Tok;
#define P2(a, b) ((a) << 8 | (b))
#define P3(a, b, c) ((a) << 16 | (b) << 8 | (c))

// ---------------------------------------------------------------- builder state
typedef struct Block {
    uint32_t label; Words w;
    int *succ; int nsucc, csucc, hdr;
    int merge_blk, cont_blk;         // from the block's merge instruction (if last-but-one instruction is a merge)
    uint16_t last_op, prev_op;
    uint8_t reason, visited, delayed, reach, rewritten;
} Block;
typedef struct IdInfo { uint32_t type; uint8_t op; uint8_t isconst; int32_t blk; uint32_t a, b, c; } IdInfo;
// for types: op = OpType*, a/b/c = operands (vector: comp,count; pointer: sc,pointee; array: elem,len-const-id;
// int: width,signed; float: width); for constants: isconst, a = value (scalar 32-bit) / b = high word

typedef struct AC {   // SpvBuilder::AccessChain
    uint32_t base; uint32_t idx[16]; int nidx; uint32_t instr; uint8_t swz[4]; int nswz; uint32_t component;
    uint32_t preswz; int rvalue;
} AC;

typedef struct HE { uint32_t h, id, off, n; } HE;   // dedup hash entries (types / constants)

struct G {
    jmp_buf jb; const char *why;
    Chunk *arena;
    // lexer
    Intern **itab; int icap, icnt;
    Tok *tk; int ntk, ctk, pos;
    // symbols
    Var *globals, **gtail; int nuid;
    Var **undo; int nundo, cundo;
    int fs;
    Var *perv;           // gl_PerVertex
    int invariant_pos, precise_pos;
    Node *main;
    // NoContraction
    // builder
    uint32_t next_id;
    IdInfo *ii; uint32_t cii;
    Words types, decos, caps_w, ext_w, modes, entry_iface;
    uint32_t caps[16]; int ncaps; const char *exts[4]; int nexts;
    HE *ht; uint32_t hcap, hcnt; Words hwords;
    Block *bl; int nbl, cbl; int cur;
    Words locals;        // OpVariable (Function) instructions, in creation order
    uint32_t std450, t_void, t_bool, t_int, t_uint, t_float;
    AC ac;
    int *brk; int nbrk;  // breakForLoop stack (1 = loop, 0 = switch)
    int *loopm, *loopc, nloop;   // loop merge / continue block stacks
    int *swm; int nsw;           // switch merge block stack
    uint32_t *ncs; int nnc, cnc; // NoContraction decoration targets
    int depth_replacing;
};

static __attribute__((noreturn)) void fail(const char *why) { g->why = why; longjmp(g->jb, 1); }

static void *amal(size_t n) {
    n = (n + 15) & ~(size_t)15;
    Chunk *c = g->arena;
    if (!c || c->used + n > c->cap) {
        size_t cap = n > (1 << 20) ? n : (1 << 20);
        c = malloc(sizeof(Chunk) + cap); c->next = g->arena; c->used = 0; c->cap = cap; g->arena = c;
    }
    void *p = c->data + c->used; c->used += n; memset(p, 0, n); return p;
}

// ---------------------------------------------------------------- interning
static uint32_t hstr(const char *s, int n) { uint32_t h = 2166136261u; for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u; return h; }
static Intern *intern(const char *s, int n) {
    uint32_t h = hstr(s, n);
    for (Intern *e = g->itab[h & (g->icap - 1)]; e; e = e->next)
        if (e->h == h && e->len == n && !memcmp(e->s, s, n)) return e;
    Intern *e = amal(sizeof *e); e->s = s; e->len = n; e->h = h;
    e->next = g->itab[h & (g->icap - 1)]; g->itab[h & (g->icap - 1)] = e; g->icnt++;
    return e;
}
static Intern *intern_c(const char *s) { return intern(s, (int)strlen(s)); }

static void kw(const char *s, int k) { intern_c(s)->kw = k; }
static void kwt(const char *s, int b, int n) { Intern *e = intern_c(s); e->kw = KW_TYPE; e->kwty.b = b; e->kwty.n = n; }
static void kwf(const char *s, int op) { intern_c(s)->fn = op; }
enum { F_NONE, F_FBU, F_UBF, F_FBI, F_IBF, F_FMA, F_MIX, F_ISNAN, F_ISINF, F_CLAMP, F_ABS, F_TRUNC, F_EXP2, F_LOG2,
       F_ISQRT, F_MIN, F_MAX, F_FLOOR, F_SQRT, F_SIN, F_COS, F_DOT, F_POW, F_RNDEVEN, F_CEIL, F_FRACT, F_PACKD,
       F_UNPACKD, F_BFE, F_BFI, F_BITCNT, F_FMSB, F_ANY, F_ALL,
       F_TEXTURE, F_TEXLOD, F_TEXSIZE, F_TEXQLOD, F_TEXQLEVELS, F_TEXFETCH, F_TEXGATHER, F_TEXGRAD, F_ATOMMAX,
       F_SGMIN, F_SGSHUFXOR, F_SGALL, F_UNSUPPORTED };
static void init_keywords(void) {
    kw("if", KW_IF); kw("else", KW_ELSE); kw("do", KW_DO); kw("while", KW_WHILE); kw("for", KW_FOR);
    kw("switch", KW_SWITCH); kw("case", KW_CASE); kw("default", KW_DEFAULT); kw("break", KW_BREAK);
    kw("continue", KW_CONTINUE); kw("return", KW_RETURN); kw("discard", KW_DISCARD); kw("demote", KW_DEMOTE);
    kw("true", KW_TRUE); kw("false", KW_FALSE); kw("layout", KW_LAYOUT); kw("in", KW_IN); kw("out", KW_OUT);
    kw("uniform", KW_UNIFORM); kw("buffer", KW_BUFFER); kw("flat", KW_FLAT); kw("invariant", KW_INVARIANT);
    kw("precise", KW_PRECISE); kw("readonly", KW_READONLY); kw("const", KW_CONST); kw("struct", KW_STRUCT);
    kwt("void", B_VOID, 1); kwt("bool", B_BOOL, 1); kwt("int", B_INT, 1); kwt("uint", B_UINT, 1);
    kwt("float", B_FLOAT, 1); kwt("double", B_DOUBLE, 1); kwt("int64_t", B_INT64, 1); kwt("uint64_t", B_UINT64, 1);
    static const char *vn[] = {"vec", "ivec", "uvec", "bvec", "dvec", "i64vec", "u64vec"};
    static const int vb[] = {B_FLOAT, B_INT, B_UINT, B_BOOL, B_DOUBLE, B_INT64, B_UINT64};
    char buf[16];
    for (int i = 0; i < 7; i++)
        for (int n = 2; n <= 4; n++) { snprintf(buf, sizeof buf, "%s%d", vn[i], n); Intern *e = intern(strcpy(amal(16), buf), (int)strlen(buf)); e->kw = KW_TYPE; e->kwty.b = vb[i]; e->kwty.n = n; }
    kwf("floatBitsToUint", F_FBU); kwf("uintBitsToFloat", F_UBF); kwf("floatBitsToInt", F_FBI); kwf("intBitsToFloat", F_IBF);
    kwf("fma", F_FMA); kwf("mix", F_MIX); kwf("isnan", F_ISNAN); kwf("isinf", F_ISINF); kwf("clamp", F_CLAMP);
    kwf("abs", F_ABS); kwf("trunc", F_TRUNC); kwf("exp2", F_EXP2); kwf("log2", F_LOG2); kwf("inversesqrt", F_ISQRT);
    kwf("min", F_MIN); kwf("max", F_MAX); kwf("floor", F_FLOOR); kwf("sqrt", F_SQRT); kwf("sin", F_SIN);
    kwf("cos", F_COS); kwf("dot", F_DOT); kwf("pow", F_POW); kwf("roundEven", F_RNDEVEN); kwf("ceil", F_CEIL);
    kwf("fract", F_FRACT); kwf("packDouble2x32", F_PACKD); kwf("unpackDouble2x32", F_UNPACKD);
    kwf("bitfieldExtract", F_BFE); kwf("bitfieldInsert", F_BFI); kwf("bitCount", F_BITCNT); kwf("findMSB", F_FMSB);
    kwf("any", F_ANY); kwf("all", F_ALL);
    kwf("texture", F_TEXTURE); kwf("textureLod", F_TEXLOD); kwf("textureSize", F_TEXSIZE);
    kwf("textureQueryLod", F_TEXQLOD); kwf("textureQueryLevels", F_TEXQLEVELS); kwf("texelFetch", F_TEXFETCH);
    kwf("textureGather", F_TEXGATHER); kwf("textureGrad", F_TEXGRAD); kwf("atomicMax", F_ATOMMAX);
    kwf("subgroupMin", F_SGMIN); kwf("subgroupShuffleXor", F_SGSHUFXOR); kwf("subgroupAll", F_SGALL);
}

// ---------------------------------------------------------------- lexer
static void tpush(Tok t) {
    if (g->ntk == g->ctk) { g->ctk = g->ctk ? g->ctk * 2 : 4096; g->tk = realloc(g->tk, g->ctk * sizeof(Tok)); }
    g->tk[g->ntk++] = t;
}
static const char CBR_BODY[] = "cb[min((off) >> 4, uint(cb.length() - 1))][((off) >> 2) & 3u]";
static void lex(const char *s, size_t len) {
    const char *e = s + len;
    int bol = 1;
    while (s < e) {
        char ch = *s;
        if (ch == '\n') { bol = 1; s++; continue; }
        if (ch == ' ' || ch == '\t' || ch == '\r') { s++; continue; }
        if (ch == '/' && s + 1 < e && s[1] == '/') { while (s < e && *s != '\n') s++; continue; }
        if (ch == '/' && s + 1 < e && s[1] == '*') { s += 2; while (s + 1 < e && !(s[0] == '*' && s[1] == '/')) s++; s += 2; continue; }
        if (ch == '#' && bol) {
            const char *l = s; while (s < e && *s != '\n') s++;
            int n = (int)(s - l);
            if (n >= 8 && !memcmp(l, "#version", 8)) continue;
            if (n >= 10 && !memcmp(l, "#extension", 10)) continue;
            if (n > 19 && !memcmp(l, "#define cbr(cb, off) ", 21)) {
                const char *b = l + 21; int bn = (int)(s - b); while (bn && (b[bn - 1] == '\r' || b[bn - 1] == ' ')) bn--;
                if (bn == (int)sizeof CBR_BODY - 1 && !memcmp(b, CBR_BODY, bn)) continue;
            }
            fail("preprocessor");
        }
        bol = 0;
        Tok t = {0};
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_') {
            const char *b = s; while (s < e && ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || (*s >= '0' && *s <= '9') || *s == '_')) s++;
            t.k = T_ID; t.id = intern(b, (int)(s - b)); tpush(t); continue;
        }
        if ((ch >= '0' && ch <= '9') || (ch == '.' && s + 1 < e && s[1] >= '0' && s[1] <= '9')) {
            const char *b = s; int flt = 0;
            if (ch == '0' && s + 1 < e && (s[1] == 'x' || s[1] == 'X')) {
                s += 2; uint64_t v = 0;
                while (s < e && ((*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'f') || (*s >= 'A' && *s <= 'F'))) { v = v * 16 + (*s <= '9' ? *s - '0' : (*s | 32) - 'a' + 10); s++; }
                t.iv = v;
            } else {
                while (s < e && ((*s >= '0' && *s <= '9') || *s == '.' || *s == 'e' || *s == 'E' || ((*s == '+' || *s == '-') && (s[-1] == 'e' || s[-1] == 'E')))) { if (*s == '.' || *s == 'e' || *s == 'E') flt = 1; s++; }
                if (flt) { char buf[64]; int n = (int)(s - b); if (n > 63) fail("literal"); memcpy(buf, b, n); buf[n] = 0; t.fv = strtod(buf, NULL); }
                else { uint64_t v = 0; for (const char *q = b; q < s; q++) v = v * 10 + (*q - '0'); if (b[0] == '0' && s - b > 1) fail("octal"); t.iv = v; }
            }
            if (flt) {
                t.k = T_FLT; t.bt = B_FLOAT;
                if (s < e && (*s == 'f' || *s == 'F')) s++;
                else if (s + 1 < e && (*s == 'l' || *s == 'L') && (s[1] == 'f' || s[1] == 'F')) { s += 2; t.bt = B_DOUBLE; }
            } else {
                t.k = T_INT; t.bt = B_INT;
                if (s + 1 < e && (*s == 'u' || *s == 'U') && (s[1] == 'l' || s[1] == 'L')) { s += 2; t.bt = B_UINT64; }
                else if (s + 1 < e && (*s == 'l' || *s == 'L') && (s[1] == 'u' || s[1] == 'U')) { s += 2; t.bt = B_UINT64; }
                else if (s < e && (*s == 'u' || *s == 'U')) { s++; t.bt = B_UINT; }
                else if (s < e && (*s == 'l' || *s == 'L')) { s++; t.bt = B_INT64; }
                if ((t.bt == B_INT || t.bt == B_UINT) && t.iv > 0xFFFFFFFFull) fail("literal range");
                if (t.bt == B_INT && t.iv > 0x7FFFFFFFull && !(b[0] == '0' && (b[1] == 'x' || b[1] == 'X'))) fail("int literal range");
            }
            if (s < e && ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || *s == '_')) fail("literal suffix");
            tpush(t); continue;
        }
        t.k = T_P;
        static const char *p3[] = {"<<=", ">>=", NULL};
        static const char *p2[] = {"<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "^^", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", NULL};
        int done = 0;
        for (int i = 0; p3[i] && !done; i++) if (s + 2 < e && !memcmp(s, p3[i], 3)) { t.p = 0; t.iv = P3(s[0], s[1], s[2]); s += 3; done = 1; }
        for (int i = 0; p2[i] && !done; i++) if (s + 1 < e && !memcmp(s, p2[i], 2)) { t.iv = P2(s[0], s[1]); s += 2; done = 1; }
        if (!done) { if (!strchr("(){}[];,.+-*/%<>=!~&|^?:", ch)) fail("char"); t.iv = (uint8_t)ch; s++; }
        tpush(t);
    }
    Tok t = {0}; t.k = T_EOF; tpush(t);
}

// expand cbr(cb, off) -> cb[min((off) >> 4, uint(cb.length() - 1))][((off) >> 2) & 3u] (token level, like cpp)
static __thread Intern *I_cbr, *I_length, *I_min, *I_uint, *I_main, *I_glpos;
static void expand_macros(void) {
    int any = 0;
    for (int i = 0; i < g->ntk; i++) if (g->tk[i].k == T_ID && g->tk[i].id == I_cbr) { any = 1; break; }
    if (!any) return;
    for (int round = 0; round < 8; round++) {
        Tok *src = g->tk; int n = g->ntk; int found = 0;
        g->tk = NULL; g->ntk = g->ctk = 0;
        for (int i = 0; i < n; i++) {
            if (!(src[i].k == T_ID && src[i].id == I_cbr && src[i + 1].k == T_P && src[i + 1].iv == '(')) { tpush(src[i]); continue; }
            found = 1;
            int j = i + 2, depth = 0, a0 = j, comma = -1;
            for (; j < n; j++) {
                if (src[j].k == T_EOF) fail("cbr");
                if (src[j].k != T_P) continue;
                if (src[j].iv == '(' || src[j].iv == '[') depth++;
                else if (src[j].iv == ')' || src[j].iv == ']') { if (depth == 0) break; depth--; }
                else if (src[j].iv == ',' && depth == 0) { if (comma >= 0) fail("cbr args"); comma = j; }
            }
            if (comma < 0) fail("cbr args");
            int a1 = comma + 1, end = j;
            #define PT(c) do { Tok t_ = {0}; t_.k = T_P; t_.iv = (c); tpush(t_); } while (0)
            #define CB() for (int q = a0; q < comma; q++) tpush(src[q])
            #define OFF() for (int q = a1; q < end; q++) tpush(src[q])
            Tok t;
            CB(); PT('['); memset(&t, 0, sizeof t); t.k = T_ID; t.id = I_min; tpush(t); PT('('); PT('('); OFF(); PT(')');
            PT(P2('>', '>')); memset(&t, 0, sizeof t); t.k = T_INT; t.bt = B_INT; t.iv = 4; tpush(t); PT(',');
            memset(&t, 0, sizeof t); t.k = T_ID; t.id = I_uint; tpush(t); PT('('); CB(); PT('.');
            memset(&t, 0, sizeof t); t.k = T_ID; t.id = I_length; tpush(t); PT('('); PT(')'); PT('-');
            memset(&t, 0, sizeof t); t.k = T_INT; t.bt = B_INT; t.iv = 1; tpush(t); PT(')'); PT(')'); PT(']');
            PT('['); PT('('); PT('('); OFF(); PT(')'); PT(P2('>', '>')); memset(&t, 0, sizeof t); t.k = T_INT; t.bt = B_INT; t.iv = 2; tpush(t);
            PT(')'); PT('&'); memset(&t, 0, sizeof t); t.k = T_INT; t.bt = B_UINT; t.iv = 3; tpush(t); PT(']');
            i = end;
        }
        free(src);
        if (!found) return;
    }
    fail("cbr nesting");
}

// ---------------------------------------------------------------- AST construction helpers
static Ty ty(int b, int n) { Ty t = {0}; t.b = b; t.n = n; return t; }
static int ty_eq(Ty a, Ty b) { return a.b == b.b && a.n == b.n && a.arr == b.arr && a.sd == b.sd && a.samp == b.samp; }
static Node *nn(int op, Ty t) { Node *n = amal(sizeof *n); n->op = op; n->t = t; return n; }
static Node *mk_const(Ty t) { Node *n = nn(N_CONST, t); n->cnst = 1; return n; }
static Node *c_uint(uint32_t v) { Node *n = mk_const(ty(B_UINT, 1)); n->cv[0].u = v; return n; }
static Node *c_int(int32_t v) { Node *n = mk_const(ty(B_INT, 1)); n->cv[0].i = v; return n; }
static int isc(Node *n) { return n->op == N_CONST; }

// normalize a constant component to its basic type's storage (TConstUnion keeps 32-bit ints as int / unsigned int)
static CV cnorm(int b, CV v) {
    if (b == B_INT) v.i = (int32_t)v.i;
    else if (b == B_UINT) v.u = (uint32_t)v.u;
    else if (b == B_BOOL) v.i = v.i != 0;
    return v;
}
// Constant.cpp numeric conversion fold
static CV cconv(int from, int to, CV v) {
    double vf = 0; int64_t vi = 0; uint64_t vu = 0; int vb = 0; int src;
    if (is_f(from)) { vf = v.d; src = 0; } else if (is_s(from)) { vi = v.i; src = 1; } else if (is_u(from)) { vu = v.u; src = 2; } else { vb = (int)v.i; src = 3; }
    CV r; r.u = 0;
    if (to == B_BOOL) { r.i = src == 0 ? vf != 0.0 : src == 1 ? vi != 0 : src == 2 ? vu != 0 : vb; }
    else if (is_f(to)) { r.d = src == 3 ? (double)vb : src == 1 ? (double)vi : src == 2 ? (double)vu : vf; }
    else if (is_s(to)) { vi = src == 3 ? vb : src == 0 ? (int64_t)vf : src == 2 ? (int64_t)vu : vi; r.i = to == B_INT ? (int32_t)vi : vi; }
    else { vu = src == 3 ? (uint64_t)vb : src == 0 ? (uint64_t)vf : src == 1 ? (uint64_t)vi : vu; r.u = to == B_UINT ? (uint32_t)vu : vu; }
    return r;
}

// Intermediate::addConversion (basic type change; same shape); folds constants
static Node *conv(Node *x, int to) {
    if (x->t.b == to) return x;
    if (x->t.arr || x->t.b == B_STRUCT || x->t.b == B_SAMPLER || to == B_VOID) fail("conv");
    Ty t = ty(to, x->t.n);
    if (isc(x)) {
        Node *c = mk_const(t);
        for (int i = 0; i < x->t.n; i++) c->cv[i] = cconv(x->t.b, to, x->cv[i]);
        return c;
    }
    Node *n = nn(O_CONV, t); n->l = x; return n;
}

// GLSL implicit promotion destination (Intermediate::getConversionDestinationType, GLSL 460)
static int rank(int b) { return b == B_INT || b == B_UINT ? 1 : 2; }
static int promote_to(int a, int b) {
    if (a == b) return a;
    if (a == B_DOUBLE || b == B_DOUBLE) { int o = a == B_DOUBLE ? b : a; if (is_i(o) || o == B_FLOAT) return B_DOUBLE; return -1; }
    if (a == B_FLOAT || b == B_FLOAT) { int o = a == B_FLOAT ? b : a; if (is_i(o)) return B_FLOAT; return -1; }
    if (is_i(a) && is_i(b)) {
        if (is_s(a) == is_s(b)) return rank(a) < rank(b) ? b : a;
        if (is_u(a) && rank(a) > rank(b)) return a;
        if (is_u(b) && rank(b) > rank(a)) return b;
        int s = is_s(a) ? a : b, u = is_s(a) ? b : a;
        if (rank(s) > rank(u)) return s;           // canSignedIntTypeRepresentAllUnsignedValues
        return s == B_INT ? B_UINT : B_UINT64;
    }
    return -1;
}

// ---- constant folding (Constant.cpp)
static CV fold2(int op, int b, CV x, CV y, int *ok) {
    CV r; r.u = 0; *ok = 1;
    int f = is_f(b), u = is_u(b), s = is_s(b);
    switch (op) {
    case O_ADD: if (f) r.d = x.d + y.d; else if (u) r.u = x.u + y.u; else r.i = (int64_t)((uint64_t)x.i + (uint64_t)y.i); break;
    case O_SUB: if (f) r.d = x.d - y.d; else if (u) r.u = x.u - y.u; else r.i = (int64_t)((uint64_t)x.i - (uint64_t)y.i); break;
    case O_MUL: case O_VTS: if (f) r.d = x.d * y.d; else if (u) r.u = x.u * y.u; else r.i = (int64_t)((uint64_t)x.i * (uint64_t)y.i); break;
    case O_AND: r.u = x.u & y.u; break;
    case O_OR: r.u = x.u | y.u; break;
    case O_XOR: r.u = x.u ^ y.u; break;
    case O_LAND: r.i = x.i && y.i; break;
    case O_LOR: r.i = x.i || y.i; break;
    case O_LXOR: r.i = (x.i != 0) != (y.i != 0); break;
    default: *ok = 0;
    }
    (void)s;
    return cnorm(b, r);
}
static int cmp_fold(int op, int b, CV x, CV y) {
    int lt, eq;
    if (is_f(b)) { lt = x.d < y.d; eq = x.d == y.d; if (op == O_GT) return x.d > y.d; if (op == O_GE) return x.d >= y.d; if (op == O_LE) return x.d <= y.d; }
    else if (is_u(b)) { lt = x.u < y.u; eq = x.u == y.u; }
    else { lt = x.i < y.i; eq = x.i == y.i; }
    switch (op) { case O_LT: return lt; case O_GT: return !lt && !eq; case O_LE: return lt || eq; case O_GE: return !lt; case O_EQ: return eq; default: return !eq; }
}

// ---------------------------------------------------------------- parser
static Tok *T(void) { return &g->tk[g->pos]; }
static int isp(int c) { return T()->k == T_P && T()->iv == (uint64_t)c; }
static int isp_at(int i, int c) { return g->tk[g->pos + i].k == T_P && g->tk[g->pos + i].iv == (uint64_t)c; }
static void expect(int c) { if (!isp(c)) fail("syntax"); g->pos++; }
static int accept(int c) { if (isp(c)) { g->pos++; return 1; } return 0; }
static int iskw(int k) { return T()->k == T_ID && T()->id->kw == k; }

static void declare(Var *v) {
    if (g->nundo == g->cundo) { g->cundo = g->cundo ? g->cundo * 2 : 256; g->undo = realloc(g->undo, g->cundo * sizeof(Var *)); }
    v->shadow = v->name->cur; v->name->cur = v; g->undo[g->nundo++] = v;
}
static void scope_pop(int mark) { while (g->nundo > mark) { Var *v = g->undo[--g->nundo]; v->name->cur = v->shadow; } }

static Node *expr(void);
static Node *assign_expr(void);

static Node *sym(Var *v) {
    Node *n;
    if (v->blk) {   // anonymous block member: blockvar.member (EOpIndexDirectStruct)
        Node *b = nn(N_SYM, v->blk->t); b->v = v->blk; b->prec = v->blk->precise;
        n = nn(O_IDX_S, v->t); n->l = b; n->r = c_int(v->member);
        n->prec = v->precise;
        return n;
    }
    n = nn(N_SYM, v->t); n->v = v; n->prec = v->precise;
    return n;
}

// Intermediate::addBinaryMath + promote (scalars/vectors only)
static Node *binop(int op, Node *l, Node *r) {
    if (l->t.arr || r->t.arr || l->t.b >= B_STRUCT || r->t.b >= B_STRUCT) fail("binop type");
    int logical = op == O_LAND || op == O_LOR || op == O_LXOR;
    int shift = op == O_SHL || op == O_SHR;
    if (logical) { if (l->t.b != B_BOOL || r->t.b != B_BOOL || l->t.n != 1 || r->t.n != 1) fail("logical"); }
    else if (shift) { if (!is_i(l->t.b) || !is_i(r->t.b)) fail("shift"); }
    else if (l->t.b != r->t.b) {
        int d = promote_to(l->t.b, r->t.b);
        if (d < 0) fail("promote");
        l = conv(l, d); r = conv(r, d);
    }
    Ty t;
    int cmp = op >= O_LT && op <= O_NE;
    if (shift) {
        if (l->t.n != r->t.n && r->t.n != 1) fail("shift shape");
        t = l->t;
    } else if (cmp) {
        if (l->t.n != r->t.n) fail("compare shape");
        if (op != O_EQ && op != O_NE && l->t.n != 1) fail("vector compare");
        t = ty(B_BOOL, 1);
    } else {
        if (l->t.n != r->t.n) {
            if (l->t.n != 1 && r->t.n != 1) fail("shape");
            if (op == O_MUL) op = O_VTS;
            t = l->t.n > 1 ? l->t : r->t;
        } else t = l->t;
        if ((op == O_AND || op == O_OR || op == O_XOR || op == O_MOD) && !is_i(t.b)) fail("int op");
    }
    if (isc(l) && isc(r)) {
        Node *c = mk_const(t); int ok = 1;
        for (int i = 0; i < t.n && ok; i++) {
            CV x = l->cv[l->t.n == 1 ? 0 : i], y = r->cv[r->t.n == 1 ? 0 : i];
            if (cmp) { if (t.n != 1) ok = 0; else { int eq = 1; if (op == O_EQ || op == O_NE) { for (int k = 0; k < l->t.n; k++) eq &= cmp_fold(O_EQ, l->t.b, l->cv[k], r->cv[k]); c->cv[0].i = op == O_EQ ? eq : !eq; } else c->cv[0].i = cmp_fold(op, l->t.b, x, y); } }
            else if (shift) {
                uint64_t sh = is_u(r->t.b) ? y.u : (uint64_t)y.i;
                if (op == O_SHL) c->cv[i] = cnorm(t.b, (CV){.u = x.u << sh});
                else if (is_u(t.b)) c->cv[i] = cnorm(t.b, (CV){.u = x.u >> sh});
                else c->cv[i] = cnorm(t.b, (CV){.i = x.i >> sh});
            } else c->cv[i] = fold2(op, t.b, x, y, &ok);
        }
        if (ok) return c;
    }
    Node *n = nn(op, t); n->l = l; n->r = r;
    return n;
}

static int is_lvalue_ok(Node *n) {
    while (n->op == O_IDX_D || n->op == O_IDX_I || n->op == O_IDX_S || n->op == O_SWZ) n = n->l;
    return n->op == N_SYM && (n->v->st == ST_LOCAL || n->v->st == ST_OUT || n->v->st == ST_BUFFER);
}

// Intermediate::addAssign
static Node *assignop(int op, Node *l, Node *r) {
    if (!is_lvalue_ok(l)) fail("lvalue");
    if (l->t.arr || l->t.b >= B_STRUCT) fail("assign type");
    if (op == O_SHL_A || op == O_SHR_A) { if (!is_i(l->t.b) || !is_i(r->t.b)) fail("shift assign"); }
    else r = conv(r, l->t.b);
    if (r->t.n != l->t.n) {
        if (!(r->t.n == 1 && op != O_ASSIGN)) fail("assign shape");
    }
    if (op == O_MUL_A && r->t.n != l->t.n) fail("vts assign");
    Node *n = nn(op, l->t); n->l = l; n->r = r; n->prec = l->prec;
    return n;
}

static int swz_char(char c) {
    switch (c) { case 'x': case 'r': case 's': return 0; case 'y': case 'g': case 't': return 1; case 'z': case 'b': case 'p': return 2; case 'w': case 'a': case 'q': return 3; }
    return -1;
}

static Node *unary_builtin(int op, Node *x, Ty rt) {
    // Intermediate::addBuiltInFunctionCall (unary): fold constants where Constant.cpp folds
    if (isc(x)) {
        int foldable = op == O_SIN || op == O_COS || op == O_EXP2 || op == O_LOG2 || op == O_SQRT || op == O_ISQRT ||
                       op == O_FLOOR || op == O_TRUNC || op == O_RNDEVEN || op == O_CEIL || op == O_FRACT || op == O_ABS ||
                       op == O_ISNAN || op == O_ISINF;
        if (foldable) {
            if (op == O_ABS && !(is_f(x->t.b) || x->t.b == B_INT)) foldable = 1;
            Node *c = mk_const(rt);
            for (int i = 0; i < x->t.n; i++) {
                double d = x->cv[i].d;
                switch (op) {
                case O_SIN: c->cv[i].d = sin(d); break; case O_COS: c->cv[i].d = cos(d); break;
                case O_EXP2: c->cv[i].d = exp2(d); break; case O_LOG2: c->cv[i].d = log2(d); break;
                case O_SQRT: c->cv[i].d = sqrt(d); break; case O_ISQRT: c->cv[i].d = 1.0 / sqrt(d); break;
                case O_FLOOR: c->cv[i].d = floor(d); break; case O_CEIL: c->cv[i].d = ceil(d); break;
                case O_TRUNC: c->cv[i].d = d > 0 ? floor(d) : ceil(d); break;
                case O_FRACT: c->cv[i].d = d - floor(d); break;
                case O_RNDEVEN: { double fl = floor(d); int even = fl / 2.0 == floor(fl / 2.0); c->cv[i].d = even ? ceil(d - 0.5) : floor(d + 0.5); break; }
                case O_ABS: if (is_f(x->t.b)) c->cv[i].d = fabs(d); else if (x->t.b == B_INT) c->cv[i].i = x->cv[i].i < 0 ? -x->cv[i].i : x->cv[i].i; else c->cv[i] = x->cv[i]; break;
                case O_ISNAN: c->cv[i].i = isnan(d); break; case O_ISINF: c->cv[i].i = isinf(d); break;
                }
                c->cv[i] = cnorm(rt.b, c->cv[i]);
            }
            return c;
        }
    }
    Node *n = nn(op, rt); n->l = x; return n;
}

static Node *agg(int op, Ty rt, Node **a, int na) {
    Node *n = nn(op, rt); n->a = amal(sizeof(Node *) * na); memcpy(n->a, a, sizeof(Node *) * na); n->na = na; return n;
}

// constructor: TParseContext::handleConstructor / constructBuiltIn + foldConstructor
static Node *construct(Ty t, Node **a, int na) {
    if (t.b == B_VOID || t.b >= B_STRUCT || t.arr) fail("ctor type");
    int total = 0;
    for (int i = 0; i < na; i++) { if (a[i]->t.arr || a[i]->t.b >= B_STRUCT) fail("ctor arg"); total += a[i]->t.n; }
    if (t.n == 1) {
        if (na != 1) fail("scalar ctor");
        Node *x = a[0];
        if (x->t.n != 1) fail("scalar ctor from vector");   // (would be a component selection)
        return conv(x, t.b);
    }
    if (na == 1 && a[0]->t.n == t.n) return conv(a[0], t.b);   // same-size vector conversion
    if (!(na == 1 && a[0]->t.n == 1) && total < t.n) fail("ctor size");
    Node **c = amal(sizeof(Node *) * na);
    int allc = 1;
    for (int i = 0; i < na; i++) { c[i] = conv(a[i], t.b); allc &= isc(c[i]); }
    if (allc) {   // foldConstructor
        Node *k = mk_const(t); int j = 0;
        if (na == 1) { for (int i = 0; i < t.n; i++) k->cv[i] = c[0]->cv[0]; }
        else for (int i = 0; i < na && j < t.n; i++) for (int q = 0; q < c[i]->t.n && j < t.n; q++) k->cv[j++] = c[i]->cv[q];
        return k;
    }
    return agg(O_CTOR, t, c, na);
}

// texture builtins (glslang's prototypes for the sampler types the translator declares); cv[0] = kind
enum { TX_SAMPLE, TX_LOD, TX_GRAD, TX_GATHER, TX_FETCH, TX_SIZE, TX_QLOD, TX_QLEVELS };
static Node *farg(Node *x, int n) {   // float scalar/vector parameter (implicit int -> float conversion)
    if (x->t.arr || x->t.b >= B_STRUCT || x->t.n != n) fail("tex arg shape");
    if (x->t.b != B_FLOAT) { if (x->t.b != B_INT && x->t.b != B_UINT) fail("tex arg type"); x = conv(x, B_FLOAT); }
    return x;
}
static Node *iarg(Node *x, int n) { if (x->t.arr || x->t.b != B_INT || x->t.n != n) fail("tex int arg"); return x; }
static Node *tex_call(int f, Node **a, int na) {
    if (na < 1 || a[0]->op != N_SYM || a[0]->t.b != B_SAMPLER) fail("tex sampler");
    SampDef *s = a[0]->t.samp;
    if (s->dim == 3 && s->arrayed) fail("cube array");
    int base = s->dim == 1 ? 2 : 3, pc = base + s->arrayed + s->shadow, kind;
    Ty rt = s->shadow ? ty(B_FLOAT, 1) : ty(B_FLOAT, 4);
    #define NA(k) do { if (na != (k)) fail("tex arg count"); } while (0)
    switch (f) {
    case F_TEXTURE: kind = TX_SAMPLE; if (na != 2 && na != 3) fail("tex arg count"); if (na == 3 && s->shadow && s->arrayed) fail("tex bias");
        a[1] = farg(a[1], pc); if (na == 3) a[2] = farg(a[2], 1); break;
    case F_TEXLOD: kind = TX_LOD; NA(3); a[1] = farg(a[1], pc); a[2] = farg(a[2], 1); break;
    case F_TEXGRAD: kind = TX_GRAD; NA(4); a[1] = farg(a[1], pc); { int gn = s->dim == 1 ? 2 : 3; a[2] = farg(a[2], gn); a[3] = farg(a[3], gn); } break;
    case F_TEXGATHER: kind = TX_GATHER; rt = ty(B_FLOAT, 4); a[1] = farg(a[1], base + s->arrayed);
        if (s->shadow) { NA(3); a[2] = farg(a[2], 1); }
        else { if (na != 2 && na != 3) fail("tex arg count"); if (na == 3 && !(isc(a[2]) && a[2]->t.b == B_INT && a[2]->t.n == 1)) fail("gather comp"); }
        break;
    case F_TEXFETCH: kind = TX_FETCH; NA(3); if (s->shadow || s->dim == 3) fail("fetch sampler"); rt = ty(B_FLOAT, 4);
        a[1] = iarg(a[1], base + s->arrayed); a[2] = iarg(a[2], 1); break;
    case F_TEXSIZE: kind = TX_SIZE; NA(2); a[1] = iarg(a[1], 1); rt = ty(B_INT, (s->dim == 2 ? 3 : 2) + s->arrayed); break;
    case F_TEXQLOD: kind = TX_QLOD; NA(2); a[1] = farg(a[1], s->dim == 1 ? 2 : 3); rt = ty(B_FLOAT, 2); break;
    case F_TEXQLEVELS: kind = TX_QLEVELS; NA(1); rt = ty(B_INT, 1); break;
    default: fail("tex op");
    }
    Node *n = agg(O_TEX, rt, a, na); n->cv[0].i = kind;
    return n;
}
// atomicMax(buffer element, value): uint/int overloads (int data converts to uint; uint data never to int)
static Node *atomic_call(Node **a, int na) {
    if (na != 2) fail("atomic args");
    Node *p = a[0], *q = p;
    while (q->op == O_IDX_D || q->op == O_IDX_I || q->op == O_IDX_S) q = q->l;
    if (q->op != N_SYM || q->v->st != ST_BUFFER || q->v->t.sd->readonly) fail("atomic target");
    if (p->t.n != 1 || p->t.arr || !(p->t.b == B_UINT || p->t.b == B_INT)) fail("atomic type");
    Node *v = a[1];
    if (v->t.n != 1 || v->t.arr) fail("atomic value");
    if (v->t.b != p->t.b) { if (p->t.b == B_UINT && v->t.b == B_INT) v = conv(v, B_UINT); else fail("atomic value type"); }
    a[1] = v;
    return agg(O_ATOMIC, p->t, a, 2);
}

static Node *call(Intern *name, Node **a, int na) {
    int f = name->fn;
    if (!f) fail(name->cur ? "call non-function" : "user function call");
    #define NARGS(k) do { if (na != (k)) fail("arg count"); } while (0)
    Node *x = na ? a[0] : NULL;
    switch (f) {
    case F_FBU: NARGS(1); if (x->t.b != B_FLOAT) fail("fbu arg"); return unary_builtin(O_F2U, x, ty(B_UINT, x->t.n));
    case F_FBI: NARGS(1); if (x->t.b != B_FLOAT) fail("fbi arg"); return unary_builtin(O_F2I, x, ty(B_INT, x->t.n));
    case F_UBF: NARGS(1); if (x->t.b == B_INT) fail("ubf int"); x = conv(x, B_UINT); return unary_builtin(O_U2F, x, ty(B_FLOAT, x->t.n));
    case F_IBF: NARGS(1); if (x->t.b == B_UINT) fail("ibf uint"); x = conv(x, B_INT); return unary_builtin(O_I2F, x, ty(B_FLOAT, x->t.n));
    case F_ISNAN: case F_ISINF: NARGS(1); x = conv(x, B_FLOAT); return unary_builtin(f == F_ISNAN ? O_ISNAN : O_ISINF, x, ty(B_BOOL, x->t.n));
    case F_ABS: NARGS(1); if (x->t.b == B_UINT) x = conv(x, B_FLOAT); return unary_builtin(O_ABS, x, x->t);
    case F_TRUNC: case F_EXP2: case F_LOG2: case F_ISQRT: case F_FLOOR: case F_SQRT: case F_SIN: case F_COS:
    case F_RNDEVEN: case F_CEIL: case F_FRACT: {
        NARGS(1); if (!is_f(x->t.b)) x = conv(x, B_FLOAT);
        static const int m[F_UNSUPPORTED] = {[F_TRUNC] = O_TRUNC, [F_EXP2] = O_EXP2, [F_LOG2] = O_LOG2, [F_ISQRT] = O_ISQRT,
            [F_FLOOR] = O_FLOOR, [F_SQRT] = O_SQRT, [F_SIN] = O_SIN, [F_COS] = O_COS, [F_RNDEVEN] = O_RNDEVEN,
            [F_CEIL] = O_CEIL, [F_FRACT] = O_FRACT};
        return unary_builtin(m[f], x, x->t);
    }
    case F_FMA: case F_POW: case F_MIN: case F_MAX: case F_CLAMP: case F_MIX: case F_DOT: {
        int want = f == F_FMA || f == F_CLAMP || f == F_MIX ? 3 : 2; NARGS(want);
        int b = a[0]->t.b; int n = a[0]->t.n;
        if (f == F_MIX && a[2]->t.b == B_BOOL) {
            int d = promote_to(a[0]->t.b, a[1]->t.b); if (d < 0) fail("mix types");
            for (int i = 0; i < 2; i++) a[i] = conv(a[i], d);
            b = d;
        } else {
            for (int i = 1; i < na; i++) { int d = promote_to(b, a[i]->t.b); if (d < 0) fail("builtin promote"); b = d; if (a[i]->t.n > n) n = a[i]->t.n; }
            if (f == F_FMA || f == F_POW || f == F_MIX || f == F_DOT) { if (is_i(b)) b = B_FLOAT; }
            for (int i = 0; i < na; i++) a[i] = conv(a[i], b);
        }
        if (f == F_FMA && !is_f(b)) fail("fma int");
        for (int i = 0; i < na; i++) if (a[i]->t.n != n && !(a[i]->t.n == 1 && (f == F_MIN || f == F_MAX || f == F_CLAMP || f == F_MIX))) fail("builtin shape");
        if (a[0]->t.n != n) fail("builtin shape0");
        int allc = 1; for (int i = 0; i < na; i++) allc &= isc(a[i]);
        int op = f == F_FMA ? O_FMA : f == F_POW ? O_POW : f == F_MIN ? O_MIN : f == F_MAX ? O_MAX : f == F_CLAMP ? O_CLAMP : f == F_MIX ? O_MIX : O_DOT;
        Ty rt = f == F_DOT ? ty(b, 1) : ty(b, n);
        if (allc && (op == O_MIN || op == O_MAX || op == O_CLAMP)) {   // TIntermediate::fold(aggregate)
            Node *c = mk_const(rt);
            for (int i = 0; i < n; i++) {
                CV v = a[0]->cv[i];
                for (int k = 1; k < na; k++) {
                    CV w = a[k]->cv[a[k]->t.n == 1 ? 0 : i];
                    int lt = is_f(b) ? w.d < v.d : is_u(b) ? w.u < v.u : w.i < v.i;
                    int gt = is_f(b) ? w.d > v.d : is_u(b) ? w.u > v.u : w.i > v.i;
                    if (op == O_MIN || (op == O_CLAMP && k == 2)) { if (lt) v = w; }
                    else if (gt) v = w;
                }
                c->cv[i] = v;
            }
            return c;
        }
        if (allc) fail("const builtin");
        return agg(op, rt, a, na);
    }
    case F_BFE: NARGS(3); if (!is_i(a[0]->t.b) || a[0]->t.n != 1) fail("bfe"); a[1] = conv(a[1], B_INT); a[2] = conv(a[2], B_INT); if (isc(a[0]) && isc(a[1]) && isc(a[2])) fail("const bfe"); return agg(O_BFE, a[0]->t, a, 3);
    case F_BFI: NARGS(4); fail("bfi");
    // (none of these fold in Constant.cpp's unary list as used here: constant operands fall back)
    case F_PACKD: NARGS(1); if (x->t.b != B_UINT || x->t.n != 2 || x->t.arr || isc(x)) fail("packDouble arg"); return unary_builtin(O_PACKD, x, ty(B_DOUBLE, 1));
    case F_UNPACKD: NARGS(1); if (x->t.b != B_DOUBLE || x->t.n != 1 || x->t.arr || isc(x)) fail("unpackDouble arg"); return unary_builtin(O_UNPACKD, x, ty(B_UINT, 2));
    case F_BITCNT: case F_FMSB: NARGS(1); if (!(x->t.b == B_UINT || x->t.b == B_INT) || x->t.arr || isc(x)) fail("bit builtin arg");
        return unary_builtin(f == F_BITCNT ? O_BITCNT : O_FMSB, x, ty(B_INT, x->t.n));
    case F_ANY: case F_ALL: NARGS(1); if (x->t.b != B_BOOL || x->t.n < 2 || x->t.arr || isc(x)) fail("any/all arg");
        return unary_builtin(f == F_ANY ? O_ANY : O_ALL, x, ty(B_BOOL, 1));
    case F_SGMIN: NARGS(1); if (!(is_f(x->t.b) || is_i(x->t.b)) || x->t.arr) fail("subgroupMin arg"); return unary_builtin(O_SGMIN, x, x->t);
    case F_SGALL: NARGS(1); if (x->t.b != B_BOOL || x->t.n != 1) fail("subgroupAll arg"); return unary_builtin(O_SGALL, x, x->t);
    case F_SGSHUFXOR: NARGS(2); if (x->t.arr || x->t.b >= B_STRUCT) fail("shuffle arg");
        if (a[1]->t.n != 1 || a[1]->t.arr) fail("shuffle mask"); if (a[1]->t.b == B_INT) a[1] = conv(a[1], B_UINT); else if (a[1]->t.b != B_UINT) fail("shuffle mask");
        return agg(O_SGSHUF, x->t, a, 2);
    case F_TEXTURE: case F_TEXLOD: case F_TEXGRAD: case F_TEXGATHER: case F_TEXFETCH: case F_TEXSIZE: case F_TEXQLOD:
    case F_TEXQLEVELS: return tex_call(f, a, na);
    case F_ATOMMAX: return atomic_call(a, na);
    default: fail("builtin");
    }
}

static Ty parse_type_tok(void) {
    if (!(T()->k == T_ID && T()->id->kw == KW_TYPE)) fail("type");
    Ty t = T()->id->kwty; g->pos++; return t;
}

static Node *primary(void) {
    Tok *t = T();
    if (t->k == T_INT) {
        g->pos++; Node *c = mk_const(ty(t->bt, 1));
        if (t->bt == B_INT) c->cv[0].i = (int32_t)(uint32_t)t->iv; else c->cv[0].u = t->iv;
        if (t->bt == B_INT64) c->cv[0].i = (int64_t)t->iv;
        return c;
    }
    if (t->k == T_FLT) { g->pos++; Node *c = mk_const(ty(t->bt, 1)); c->cv[0].d = t->fv; return c; }
    if (accept('(')) { Node *e = expr(); expect(')'); return e; }
    if (t->k != T_ID) fail("primary");
    Intern *id = t->id;
    if (id->kw == KW_TRUE || id->kw == KW_FALSE) { g->pos++; Node *c = mk_const(ty(B_BOOL, 1)); c->cv[0].i = id->kw == KW_TRUE; return c; }
    if (id->kw == KW_TYPE) {
        Ty ct = parse_type_tok();
        if (isp('[')) fail("array ctor");
        expect('(');
        Node *args[8]; int na = 0;
        if (!isp(')')) do { if (na == 8) fail("args"); args[na++] = assign_expr(); } while (accept(','));
        expect(')');
        return construct(ct, args, na);
    }
    g->pos++;
    if (isp('(')) {
        g->pos++;
        Node *args[8]; int na = 0;
        if (!isp(')')) do { if (na == 8) fail("args"); args[na++] = assign_expr(); } while (accept(','));
        expect(')');
        return call(id, args, na);
    }
    if (!id->cur) fail("undeclared");
    return sym(id->cur);
}

static Node *index_node(Node *b, Node *ix) {
    if (!is_i(ix->t.b) || ix->t.n != 1) fail("index type");
    Ty et;
    if (b->t.arr) { et = b->t; et.arr = 0; }
    else if (b->t.n > 1) et = ty(b->t.b, 1);
    else fail("index scalar");
    Node *n;
    if (isc(ix)) {
        // TParseContext::handleBracketDereference: constant index into an rvalue constant would fold; not reached here
        if (isc(b)) fail("const index const");
        n = nn(O_IDX_D, et); n->l = b; n->r = ix;
    } else { n = nn(O_IDX_I, et); n->l = b; n->r = ix; }
    n->prec = b->prec;
    return n;
}

static Node *postfix(void) {
    Node *n = primary();
    for (;;) {
        if (accept('[')) { Node *ix = expr(); expect(']'); n = index_node(n, ix); continue; }
        if (accept('.')) {
            if (T()->k != T_ID) fail("field");
            Intern *f = T()->id; g->pos++;
            if (f == I_length) {
                expect('('); expect(')');
                if (n->t.arr > 0) { n = c_int(n->t.arr); continue; }
                fail("runtime length");
            }
            if (n->t.b == B_STRUCT) {
                StructDef *sd = n->t.sd; int k = -1;
                for (int i = 0; i < sd->nm; i++) if (sd->m[i].name == f) k = i;
                if (k < 0) fail("member");
                Node *s = nn(O_IDX_S, *sd->m[k].t); s->l = n; s->r = c_int(k); s->prec = n->prec; n = s; continue;
            }
            if (n->t.arr || n->t.b >= B_STRUCT) fail("swizzle base");
            if (f->len > 4) fail("swizzle len");
            Node *s = nn(O_SWZ, ty(n->t.b, f->len));
            for (int i = 0; i < f->len; i++) { int c = swz_char(f->s[i]); if (c < 0 || c >= n->t.n) fail("swizzle"); s->sw[i] = c; }
            s->nsw = f->len;
            if (isc(n)) {   // constant swizzle folds
                Node *c = mk_const(s->t); for (int i = 0; i < f->len; i++) c->cv[i] = n->cv[s->sw[i]]; n = c; continue;
            }
            if (f->len == 1) {   // TParseContext::handleDotDereference: single component -> EOpIndexDirect
                Node *d = nn(O_IDX_D, ty(n->t.b, 1)); d->l = n; d->r = c_int(s->sw[0]); d->prec = n->prec; n = d; continue;
            }
            s->l = n; s->prec = n->prec; n = s; continue;
        }
        if (isp(P2('+', '+')) || isp(P2('-', '-'))) {
            int inc = isp(P2('+', '+')); g->pos++;
            if (!is_lvalue_ok(n) || n->t.arr || n->t.b >= B_STRUCT || n->t.b == B_BOOL) fail("postinc");
            Node *u = nn(inc ? O_POSTINC : O_POSTDEC, n->t); u->l = n; u->prec = n->prec; n = u; continue;
        }
        return n;
    }
}

static Node *unary(void) {
    if (accept('-')) {
        Node *x = unary();
        if (x->t.b == B_BOOL || x->t.arr || x->t.b >= B_STRUCT) fail("neg type");
        if (isc(x)) {
            Node *c = mk_const(x->t);
            for (int i = 0; i < x->t.n; i++) {
                if (is_f(x->t.b)) c->cv[i].d = -x->cv[i].d;
                else if (x->t.b == B_INT) c->cv[i].i = (uint32_t)x->cv[i].i == 0x80000000u ? INT32_MIN : -(int32_t)x->cv[i].i;
                else if (x->t.b == B_UINT) c->cv[i].u = (uint32_t)(-(int32_t)x->cv[i].u);
                else if (x->t.b == B_INT64) c->cv[i].i = x->cv[i].i == INT64_MIN ? INT64_MIN : -x->cv[i].i;
                else c->cv[i].u = (uint64_t)(-(int64_t)x->cv[i].u);
            }
            return c;
        }
        Node *n = nn(O_NEG, x->t); n->l = x; return n;
    }
    if (accept('+')) return unary();
    if (accept('!')) {
        Node *x = unary(); if (x->t.b != B_BOOL || x->t.n != 1) fail("not type");
        if (isc(x)) { Node *c = mk_const(x->t); c->cv[0].i = !x->cv[0].i; return c; }
        Node *n = nn(O_LNOT, x->t); n->l = x; return n;
    }
    if (accept('~')) {
        Node *x = unary(); if (!is_i(x->t.b)) fail("bnot type");
        if (isc(x)) { Node *c = mk_const(x->t); for (int i = 0; i < x->t.n; i++) c->cv[i] = cnorm(x->t.b, (CV){.u = ~x->cv[i].u}); return c; }
        Node *n = nn(O_BNOT, x->t); n->l = x; return n;
    }
    if (isp(P2('+', '+')) || isp(P2('-', '-'))) {
        int inc = isp(P2('+', '+')); g->pos++;
        Node *x = unary();
        if (!is_lvalue_ok(x) || x->t.arr || x->t.b >= B_STRUCT || x->t.b == B_BOOL) fail("preinc");
        Node *u = nn(inc ? O_PREINC : O_PREDEC, x->t); u->l = x; u->prec = x->prec; return u;
    }
    return postfix();
}

static int binprec(int *op) {
    Tok *t = T(); if (t->k != T_P) return -1;
    switch (t->iv) {
    case P2('|', '|'): *op = O_LOR; return 1;
    case P2('^', '^'): *op = O_LXOR; return 2;
    case P2('&', '&'): *op = O_LAND; return 3;
    case '|': *op = O_OR; return 4;
    case '^': *op = O_XOR; return 5;
    case '&': *op = O_AND; return 6;
    case P2('=', '='): *op = O_EQ; return 7; case P2('!', '='): *op = O_NE; return 7;
    case '<': *op = O_LT; return 8; case '>': *op = O_GT; return 8; case P2('<', '='): *op = O_LE; return 8; case P2('>', '='): *op = O_GE; return 8;
    case P2('<', '<'): *op = O_SHL; return 9; case P2('>', '>'): *op = O_SHR; return 9;
    case '+': *op = O_ADD; return 10; case '-': *op = O_SUB; return 10;
    case '*': *op = O_MUL; return 11; case '/': *op = O_DIV; return 11; case '%': *op = O_MOD; return 11;
    }
    return -1;
}
static Node *binary(int minp) {
    Node *l = unary();
    for (;;) {
        int op, p = binprec(&op);
        if (p < 0 || p < minp) return l;
        g->pos++;
        Node *r = binary(p + 1);
        if (op == O_DIV && is_i(l->t.b) && is_i(r->t.b) && isc(r) && isc(l)) fail("const int div");
        if ((op == O_DIV || op == O_MOD) && isc(l) && isc(r)) fail("const div");
        l = binop(op, l, r);
    }
}
static Node *cond_expr(void) {
    Node *c = binary(1);
    if (!accept('?')) return c;
    Node *a = expr(); expect(':'); Node *b = assign_expr();
    if (c->t.b != B_BOOL || c->t.n != 1) fail("?: cond");
    if (a->t.arr || b->t.arr || a->t.b >= B_STRUCT || b->t.b >= B_STRUCT) fail("?: type");
    if (a->t.b != b->t.b) { int d = promote_to(a->t.b, b->t.b); if (d < 0) fail("?: promote"); a = conv(a, d); b = conv(b, d); }
    if (a->t.n != b->t.n) fail("?: shape");
    if (isc(c) && isc(a) && isc(b)) return c->cv[0].i ? a : b;   // Intermediate::addSelection folds all-constant
    Node *n = nn(O_SEL, a->t); n->c = c; n->l = a; n->r = b;
    return n;
}
static Node *assign_expr(void) {
    int save = g->pos;
    Node *l = cond_expr();
    Tok *t = T();
    if (t->k != T_P) return l;
    int op = 0;
    switch (t->iv) {
    case '=': op = O_ASSIGN; break; case P2('+', '='): op = O_ADD_A; break; case P2('-', '='): op = O_SUB_A; break;
    case P2('*', '='): op = O_MUL_A; break; case P2('/', '='): op = O_DIV_A; break; case P2('%', '='): op = O_MOD_A; break;
    case P3('<', '<', '='): op = O_SHL_A; break; case P3('>', '>', '='): op = O_SHR_A; break;
    case P2('&', '='): op = O_AND_A; break; case P2('|', '='): op = O_OR_A; break; case P2('^', '='): op = O_XOR_A; break;
    default: return l;
    }
    (void)save;
    g->pos++;
    Node *r = assign_expr();
    return assignop(op, l, r);
}
static Node *expr(void) {
    Node *e = assign_expr();
    if (isp(',')) fail("comma");
    return e;
}

// ---- statements
static Node *seq_new(void) { Node *s = nn(S_SEQ, ty(B_VOID, 1)); return s; }
static void seq_add(Node *s, Node *x) {
    if (!x) return;
    if ((s->na & (s->na - 1)) == 0 && s->na >= 4) { Node **a = amal(sizeof(Node *) * s->na * 2); memcpy(a, s->a, sizeof(Node *) * s->na); s->a = a; }
    else if (s->na == 0) s->a = amal(sizeof(Node *) * 4);
    s->a[s->na++] = x;
}
static Node *statement(void);
static Var *new_local(Intern *name, Ty t, int precise) {
    Var *v = amal(sizeof *v); v->name = name; v->t = t; v->st = ST_LOCAL; v->precise = precise; v->uid = ++g->nuid;
    declare(v); return v;
}
// declaration statement: [precise] type name [= init] {, name [= init]} ;   -> sequence of assignments (or NULL)
static Node *declaration(void) {
    int precise = 0;
    if (iskw(KW_PRECISE)) { precise = 1; g->pos++; }
    if (iskw(KW_CONST)) fail("const local");
    Ty bt = parse_type_tok();
    if (bt.b == B_VOID) fail("void decl");
    Node *out = NULL; int count = 0; Node *seq = NULL;
    do {
        if (T()->k != T_ID || T()->id->kw) fail("decl name");
        Intern *name = T()->id; g->pos++;
        Ty t = bt;
        if (accept('[')) { if (T()->k != T_INT) fail("array size"); t.arr = (int)T()->iv; g->pos++; expect(']'); if (t.arr <= 0) fail("array size"); }
        Node *init = NULL;
        if (accept('=')) {
            if (t.arr) fail("array init");
            init = assign_expr();
        }
        Var *v = new_local(name, t, precise);   // the initializer is parsed before the name is visible (GLSL scoping)
        if (init) {
            Node *s = sym(v);
            Node *a = assignop(O_ASSIGN, s, init);
            if (count == 0) out = a;
            else { if (!seq) { seq = seq_new(); seq_add(seq, out); } seq_add(seq, a); out = seq; }
            count++;
        }
    } while (accept(','));
    expect(';');
    return out;
}
static int is_decl_start(void) {
    if (iskw(KW_PRECISE) || iskw(KW_CONST)) return 1;
    return T()->k == T_ID && T()->id->kw == KW_TYPE && g->tk[g->pos + 1].k == T_ID;
}
static Node *compound(void) {   // after '{'
    int mark = g->nundo;
    Node *s = seq_new();
    while (!accept('}')) seq_add(s, statement());
    scope_pop(mark);
    return s->na ? s : NULL;
}
static Node *sub_statement(void) {   // statement with its own scope (if/loop bodies): `statement_scoped`
    int mark = g->nundo;
    Node *s;
    if (accept('{')) { s = compound(); }
    else s = statement();
    scope_pop(mark);
    return s;
}
static Node *mk_stmt(int op) { return nn(op, ty(B_VOID, 1)); }
static Node *statement(void) {
    if (accept(';')) return NULL;
    if (accept('{')) return compound();
    if (T()->k == T_ID) {
        switch (T()->id->kw) {
        case KW_IF: {
            g->pos++; expect('('); Node *c = expr(); expect(')');
            if (c->t.b != B_BOOL || c->t.n != 1) fail("if cond");
            Node *n = mk_stmt(S_IF); n->c = c; n->l = sub_statement();
            if (iskw(KW_ELSE)) { g->pos++; n->r = sub_statement(); }
            return n;
        }
        case KW_DO: {
            g->pos++;
            Node *n = mk_stmt(S_LOOP); n->test_first = 0;
            g->nbrk++; n->l = sub_statement(); g->nbrk--;
            if (!iskw(KW_WHILE)) fail("do while"); g->pos++;
            expect('('); n->c = expr(); expect(')'); expect(';');
            if (n->c->t.b != B_BOOL) fail("loop test");
            return n;
        }
        case KW_WHILE: {
            g->pos++; expect('(');
            int mark = g->nundo;
            Node *n = mk_stmt(S_LOOP); n->test_first = 1; n->c = expr(); expect(')');
            if (n->c->t.b != B_BOOL) fail("loop test");
            n->l = sub_statement(); scope_pop(mark);
            return n;
        }
        case KW_FOR: {
            g->pos++; expect('(');
            int mark = g->nundo;
            Node *init = NULL;
            if (!accept(';')) { if (is_decl_start()) init = declaration(); else { init = expr(); expect(';'); } }
            Node *n = mk_stmt(S_LOOP); n->test_first = 1;
            if (!isp(';')) { n->c = expr(); if (n->c->t.b != B_BOOL) fail("loop test"); }
            expect(';');
            if (!isp(')')) n->r = expr();
            expect(')');
            n->l = sub_statement();
            scope_pop(mark);
            // TParseContext: for-loop -> sequence [init, loop]
            Node *s = seq_new(); seq_add(s, init); seq_add(s, n);
            return s;
        }
        case KW_SWITCH: {
            g->pos++; expect('('); Node *c = expr(); expect(')');
            if (!is_i(c->t.b) || c->t.n != 1 || bits(c->t.b) != 32) fail("switch selector");
            expect('{');
            int mark = g->nundo;
            Node *n = mk_stmt(S_SWITCH); n->c = c; n->l = seq_new();
            // body: case/default labels and runs of statements (each run one sequence: wrapupSwitchSubsequence)
            Node *run = NULL;
            while (!accept('}')) {
                if (iskw(KW_CASE) || iskw(KW_DEFAULT)) {
                    if (run) { seq_add(n->l, run); run = NULL; }
                    Node *lab;
                    if (iskw(KW_CASE)) {
                        g->pos++; Node *v = expr(); if (!isc(v) || !is_i(v->t.b)) fail("case value");
                        lab = mk_stmt(S_CASE); lab->cv[0] = v->cv[0];
                    } else { g->pos++; lab = mk_stmt(S_DEFAULT); }
                    expect(':');
                    seq_add(n->l, lab);
                    continue;
                }
                Node *st = statement();
                if (st) { if (!run) run = seq_new(); seq_add(run, st); }
                else if (!run) { /* an empty statement before anything: nothing */ }
            }
            if (run) seq_add(n->l, run);
            scope_pop(mark);
            return n;
        }
        case KW_BREAK: g->pos++; expect(';'); return mk_stmt(S_BREAK);
        case KW_CONTINUE: g->pos++; expect(';'); return mk_stmt(S_CONTINUE);
        case KW_RETURN: g->pos++; if (!isp(';')) fail("return value"); expect(';'); return mk_stmt(S_RETURN);
        case KW_DISCARD: g->pos++; expect(';'); if (!g->fs) fail("discard vs"); return mk_stmt(S_KILL);
        case KW_DEMOTE: g->pos++; expect(';'); if (!g->fs) fail("demote vs"); return mk_stmt(S_DEMOTE);
        default: break;
        }
        if (is_decl_start()) return declaration();
    }
    Node *e = expr(); expect(';');
    return e;
}

// ---- global declarations
static Var *new_global(Intern *name, Ty t, int st) {
    Var *v = amal(sizeof *v); v->name = name; v->t = t; v->st = st; v->uid = ++g->nuid; v->location = -1;
    *g->gtail = v; g->gtail = &v->gnext;
    if (name) declare(v);
    return v;
}
static Ty *ty_dup(Ty t) { Ty *p = amal(sizeof *p); *p = t; return p; }
static void builtin_var(const char *name, Ty t, int st, int builtin) {
    Var *v = amal(sizeof *v); v->name = intern_c(name); v->t = t; v->st = st; v->builtin = builtin; v->uid = ++g->nuid;
    v->location = -1; declare(v);
}
static void setup_builtins(void) {
    if (g->fs) {
        builtin_var("gl_FragCoord", ty(B_FLOAT, 4), ST_BUILTIN_IN, 15);
        builtin_var("gl_FrontFacing", ty(B_BOOL, 1), ST_BUILTIN_IN, 17);
        builtin_var("gl_FragDepth", ty(B_FLOAT, 1), ST_OUT, 22);
    } else {
        builtin_var("gl_VertexIndex", ty(B_INT, 1), ST_BUILTIN_IN, 42);
        builtin_var("gl_InstanceIndex", ty(B_INT, 1), ST_BUILTIN_IN, 43);
        // gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[1]; float gl_CullDistance[1]; }
        StructDef *sd = amal(sizeof *sd); sd->nm = 4; sd->block = 1; sd->is_pervertex = 1;
        Ty fl = ty(B_FLOAT, 1), fa = fl; fa.arr = 1;
        static const char *mn[] = {"gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_CullDistance"};
        static const int mb[] = {0, 1, 3, 4};
        Ty mt[4] = {ty(B_FLOAT, 4), fl, fa, fa};
        for (int i = 0; i < 4; i++) { sd->m[i].name = intern_c(mn[i]); sd->m[i].t = ty_dup(mt[i]); sd->m[i].builtin = mb[i]; }
        Ty bt = {0}; bt.b = B_STRUCT; bt.n = 1; bt.sd = sd;
        Var *pv = amal(sizeof *pv); pv->t = bt; pv->st = ST_OUT; pv->uid = ++g->nuid; pv->location = -1;
        g->perv = pv;
        for (int i = 0; i < 4; i++) {
            Var *m = amal(sizeof *m); m->name = sd->m[i].name; m->t = mt[i]; m->st = ST_OUT; m->blk = pv; m->member = i; m->uid = pv->uid; declare(m);
        }
    }
}

static void layout_q(int *set, int *binding, int *location, int *std430) {
    expect('(');
    for (;;) {
        if (T()->k != T_ID) fail("layout");
        Intern *k = T()->id; g->pos++;
        int has = accept('='); int val = 0;
        if (has) { if (T()->k != T_INT) fail("layout value"); val = (int)T()->iv; g->pos++; }
        if (k->len == 3 && !memcmp(k->s, "set", 3)) *set = val;
        else if (k->len == 7 && !memcmp(k->s, "binding", 7)) *binding = val;
        else if (k->len == 8 && !memcmp(k->s, "location", 8)) *location = val;
        else if (k->len == 6 && !memcmp(k->s, "std140", 6)) *std430 = 0;
        else if (k->len == 6 && !memcmp(k->s, "std430", 6)) *std430 = 1;
        else fail("layout qualifier");
        if (accept(')')) break;
        expect(',');
    }
}
static Ty sampler_type(Intern *id) {
    static const struct { const char *n; int dim, arr, sh; } st[] = {
        {"sampler2D", 1, 0, 0}, {"sampler2DArray", 1, 1, 0}, {"samplerCube", 3, 0, 0}, {"sampler2DArrayShadow", 1, 1, 1},
        {"sampler2DShadow", 1, 0, 1}, {"sampler3D", 2, 0, 0}, {"samplerCubeShadow", 3, 0, 1}};
    for (unsigned i = 0; i < sizeof st / sizeof *st; i++)
        if ((int)strlen(st[i].n) == id->len && !memcmp(st[i].n, id->s, id->len)) {
            SampDef *sd = amal(sizeof *sd); sd->dim = st[i].dim; sd->arrayed = st[i].arr; sd->shadow = st[i].sh;
            Ty t = {0}; t.b = B_SAMPLER; t.n = 1; t.samp = sd; return t;
        }
    fail("sampler type");
}
static void global_decl(void) {
    int set = -1, binding = -1, location = -1, std430 = 0, flat = 0, readonly = 0, st = -1;
    if (iskw(KW_INVARIANT) || iskw(KW_PRECISE)) {
        int k = T()->id->kw;
        if (g->tk[g->pos + 1].k == T_ID && g->tk[g->pos + 1].id == I_glpos && isp_at(2, ';')) {
            if (g->fs) fail("gl_Position in fs");
            if (k == KW_INVARIANT) g->invariant_pos = 1;
            else { g->precise_pos = 1; if (I_glpos->cur) I_glpos->cur->precise = 1; }   // the member's type gets noContraction
            g->pos += 3; return;
        }
    }
    if (iskw(KW_LAYOUT)) { g->pos++; layout_q(&set, &binding, &location, &std430); }
    for (;;) {
        if (iskw(KW_FLAT)) { flat = 1; g->pos++; continue; }
        if (iskw(KW_READONLY)) { readonly = 1; g->pos++; continue; }
        break;
    }
    if (iskw(KW_IN)) st = ST_IN; else if (iskw(KW_OUT)) st = ST_OUT; else if (iskw(KW_UNIFORM)) st = ST_UNIFORM; else if (iskw(KW_BUFFER)) st = ST_BUFFER;
    else fail("global decl");
    g->pos++;
    if (T()->k != T_ID) fail("global decl type");
    if ((st == ST_UNIFORM || st == ST_BUFFER) && !T()->id->kw && isp_at(1, '{')) {   // block
        g->pos += 2;
        StructDef *sd = amal(sizeof *sd); sd->block = 1; sd->std430 = std430; sd->readonly = readonly;
        while (!accept('}')) {
            if (sd->nm == 4) fail("block members");
            Ty mt = parse_type_tok();
            if (T()->k != T_ID) fail("member name");
            Intern *mn = T()->id; g->pos++;
            if (accept('[')) { if (accept(']')) mt.arr = -1; else { if (T()->k != T_INT) fail("member array"); mt.arr = (int)T()->iv; g->pos++; expect(']'); } }
            expect(';');
            sd->m[sd->nm].name = mn; sd->m[sd->nm].t = ty_dup(mt); sd->nm++;
        }
        Ty bt = {0}; bt.b = B_STRUCT; bt.n = 1; bt.sd = sd;
        if (set < 0 || binding < 0) fail("block binding");
        if (T()->k == T_ID) {   // instance name
            Intern *in = T()->id; g->pos++; expect(';');
            Var *v = new_global(in, bt, st); v->set = set; v->binding = binding;
        } else {
            expect(';');
            Var *v = new_global(NULL, bt, st); v->set = set; v->binding = binding;
            for (int i = 0; i < sd->nm; i++) {
                Var *m = amal(sizeof *m); m->name = sd->m[i].name; m->t = *sd->m[i].t; m->st = st; m->blk = v; m->member = i; m->uid = v->uid;
                declare(m);
            }
        }
        return;
    }
    Ty t;
    if (T()->id->kw == KW_TYPE) t = parse_type_tok();
    else { t = sampler_type(T()->id); g->pos++; if (st != ST_UNIFORM) fail("sampler storage"); st = ST_SAMPLER; }
    if (T()->k != T_ID) fail("global name");
    Intern *name = T()->id; g->pos++;
    if (isp('[')) fail("global array");
    expect(';');
    if (st == ST_UNIFORM || st == ST_BUFFER) fail("loose uniform");
    if ((st == ST_IN || st == ST_OUT) && location < 0) fail("no location");
    if (st == ST_SAMPLER && (set < 0 || binding < 0)) fail("sampler binding");
    Var *v = new_global(name, t, st); v->set = set; v->binding = binding; v->location = location; v->flat = flat;
    if (readonly) fail("readonly");
}
static void skip_braces(void) {
    int d = 0;
    do { if (T()->k == T_EOF) fail("eof"); if (isp('{')) d++; else if (isp('}')) d--; g->pos++; } while (d);
}
static void parse_program(void) {
    while (T()->k != T_EOF) {
        if (T()->k == T_ID && T()->id->kw == KW_TYPE && g->tk[g->pos + 1].k == T_ID && isp_at(2, '(')) {
            Intern *fname = g->tk[g->pos + 1].id;
            if (fname == I_main) {
                g->pos += 3; expect(')'); expect('{');
                int mark = g->nundo;
                g->main = compound();
                scope_pop(mark);
                if (!g->main) g->main = seq_new();
                continue;
            }
            // helper function definition: skipped; a call to it makes the shader fall back (F_NONE)
            g->pos += 3; while (!isp(')')) { if (T()->k == T_EOF) fail("eof"); g->pos++; }
            g->pos++;
            if (!isp('{')) fail("prototype");
            skip_braces();
            continue;
        }
        global_decl();
    }
    if (!g->main) fail("no main");
}

// ---------------------------------------------------------------- NoContraction (propagateNoContraction.cpp)
// object chains: (symbol uid, struct member or -1); a chain is a prefix of another when uid matches and member is -1
typedef struct Def { Node *node; int uid; } Def;
static __thread Def *defs; static __thread int ndefs, cdefs;
static __thread int64_t *work; static __thread int nwork, cwork;
// TSymbolDefinitionCollectingTraverser's current_object_ after traversing n, starting from prev: binary / unary / aggregate
// visits clear it first; constants leave it alone; ?: (default traversal) runs on through its three children. A deref's
// access-chain mapping is this value, so e.g. (a + b).x maps to b and textureSize(t, 0).z to nothing (NC_EMPTY).
#define NC_EMPTY (-2)
static int64_t cur_obj(Node *n, int64_t prev) {
    switch (n->op) {
    case N_CONST: return prev;
    case N_SYM: return (int64_t)n->v->uid << 8 | 0xFF;
    case O_SEL: return cur_obj(n->r, cur_obj(n->l, cur_obj(n->c, prev)));
    case O_IDX_S: {
        int64_t c = cur_obj(n->l, NC_EMPTY);
        if (c == NC_EMPTY) return NC_EMPTY;   // "/k": its front element names no symbol
        if ((c & 0xFF) != 0xFF) fail("nc chain depth");
        return (c & ~0xFFll) | (n->r->cv[0].i & 0x7F);
    }
    case O_IDX_D: case O_IDX_I: case O_SWZ: return cur_obj(n->l, NC_EMPTY);
    }
    if (n->op >= O_ASSIGN && n->op <= O_NE) return cur_obj(n->r, NC_EMPTY);
    if (n->op >= O_NEG && n->op <= O_ARRLEN) return NC_EMPTY;
    if (n->op >= O_MIN && n->op <= O_SGSHUF) { int64_t x = NC_EMPTY; for (int i = 0; i < n->na; i++) x = cur_obj(n->a[i], NC_EMPTY); return x; }
    fail("nc chain");
}
static int64_t chain_of(Node *n) { int64_t c = cur_obj(n, NC_EMPTY); if (c == NC_EMPTY) fail("nc lvalue"); return c; }
static void add_precise(int64_t c) {
    if (nwork == cwork) { cwork = cwork ? cwork * 2 : 64; work = realloc(work, cwork * sizeof *work); }
    work[nwork++] = c;
}
static int is_assign_op(int op) { return (op >= O_ASSIGN && op <= O_XOR_A) || (op >= O_PREINC && op <= O_POSTDEC); }
static int is_arith(int op) {
    return op == O_ADD_A || op == O_SUB_A || op == O_MUL_A || op == O_DIV_A || op == O_MOD_A || op == O_NEG || op == O_ADD ||
           op == O_SUB || op == O_MUL || op == O_DIV || op == O_MOD || op == O_VTS || (op >= O_PREINC && op <= O_POSTDEC);
}
static int is_deref(int op) { return op == O_IDX_D || op == O_IDX_I || op == O_IDX_S || op == O_SWZ; }
static void collect(Node *n) {
    if (!n) return;
    if (is_assign_op(n->op)) {
        Node *lhs = n->l;
        int64_t c = chain_of(lhs);
        if (c < 0) fail("nc chain");
        if (lhs->prec) add_precise(c);
        if (ndefs == cdefs) { cdefs = cdefs ? cdefs * 2 : 256; defs = realloc(defs, cdefs * sizeof *defs); }
        defs[ndefs].node = n; defs[ndefs].uid = (int)(c >> 8); ndefs++;
        // (the collector's lhs traversal records nothing: deref right children are skipped)
        if (n->op < O_PREINC) collect(n->r);
        return;
    }
    if (is_deref(n->op)) { collect(n->l); return; }
    switch (n->op) {
    case N_CONST: case N_SYM: return;
    case S_SEQ: case O_CTOR: case O_MIN: case O_MAX: case O_CLAMP: case O_MIX: case O_FMA: case O_POW: case O_DOT: case O_BFE: case O_BFI: case O_TEX: case O_ATOMIC: case O_SGSHUF:
        for (int i = 0; i < n->na; i++) collect(n->a[i]); return;
    default:
        collect(n->c); collect(n->l); collect(n->r); return;
    }
}
static void propagate(Node *n, int remained) {   // TNoContractionPropagator (remained: -1 none, else member index)
    if (!n) return;
    if (is_deref(n->op) || n->op == N_SYM) {   // no descent into a deref's children
        int64_t c = cur_obj(n, NC_EMPTY);
        if (remained < 0) n->nc = 1;
        if (c == NC_EMPTY) return;              // a chain without a symbol: no definitions to reach
        if (remained >= 0) { if ((c & 0xFF) != 0xFF) fail("nc remained"); c = (c & ~0xFFll) | remained; }
        add_precise(c);   // (re-queued chains are skipped by the done flags in no_contraction)
        return;
    }
    if (n->op >= O_ADD && n->op <= O_SWZ) {   // binary
        if (is_arith(n->op) && n->t.b != B_INT) n->nc = 1;
        propagate(n->l, remained); propagate(n->r, remained); return;
    }
    if (n->op >= O_NEG && n->op <= O_ARRLEN) { if (is_arith(n->op)) n->nc = 1; propagate(n->l, remained); return; }
    if (n->op >= O_ASSIGN && n->op <= O_XOR_A) { if (is_arith(n->op) && n->t.b != B_INT) n->nc = 1; propagate(n->l, remained); propagate(n->r, remained); return; }
    switch (n->op) {
    case O_CTOR: case O_MIN: case O_MAX: case O_CLAMP: case O_MIX: case O_FMA: case O_POW: case O_DOT: case O_BFE: case O_BFI: case O_TEX: case O_ATOMIC: case O_SGSHUF:
        for (int i = 0; i < n->na; i++) propagate(n->a[i], remained); return;
    case O_SEL: propagate(n->c, remained); propagate(n->l, remained); propagate(n->r, remained); return;
    default: return;
    }
}
static void no_contraction(void) {
    ndefs = nwork = 0;
    collect(g->main);
    if (!nwork) return;
    // processed chains (a chain is processed at most twice in glslang; marking is idempotent, so once is enough)
    int cap = g->nuid + 2;
    uint8_t *done_whole = calloc(cap, 1), *done_mem = calloc((size_t)cap * 8, 1);
    // index defs by uid
    int *head = malloc(sizeof(int) * cap), *nxt = malloc(sizeof(int) * (ndefs + 1));
    for (int i = 0; i < cap; i++) head[i] = -1;
    for (int i = ndefs - 1; i >= 0; i--) { nxt[i] = head[defs[i].uid]; head[defs[i].uid] = i; }
    while (nwork) {
        int64_t c = work[--nwork];
        int uid = (int)(c >> 8), mem = (int)(c & 0xFF);
        if (uid >= cap) fail("nc uid");
        if (mem == 0xFF) { if (done_whole[uid]) continue; done_whole[uid] = 1; }
        else { if (mem >= 8) fail("nc member"); if (done_mem[uid * 8 + mem]) continue; done_mem[uid * 8 + mem] = 1; }
        for (int d = head[uid]; d >= 0; d = nxt[d]) {
            Node *dn = defs[d].node;
            // TNoContractionAssigneeCheckingTraverser
            int am = (int)(chain_of(dn->l) & 0xFF);
            int contain = 0, remained = -1;
            if (dn->l->prec || mem == 0xFF || am == mem) contain = 1;     // precise lhs, or assignee starts with precise
            else if (am == 0xFF) { contain = 1; remained = mem; }          // precise starts with assignee
            if (!contain) continue;
            if (dn->op < O_PREINC) {
                propagate(dn->r, remained);
                if (is_arith(dn->op)) dn->nc = 1;
            } else {
                propagate(dn->l, remained);
                dn->nc = 1;
            }
        }
    }
    free(done_whole); free(done_mem); free(head); free(nxt);
}

// ---------------------------------------------------------------- builder
enum { OpExtension = 10, OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
       OpCapability = 17, OpTypeVoid = 19, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeMatrix, OpTypeImage,
       OpTypeSampler, OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypePointer = 32,
       OpTypeFunction = 33, OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
       OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
       OpDecorate = 71, OpMemberDecorate = 72, OpVectorExtractDynamic = 77, OpVectorShuffle = 79,
       OpCompositeConstruct = 80, OpCompositeExtract = 81, OpCompositeInsert = 82, OpConvertFToU = 109,
       OpConvertFToS = 110, OpConvertSToF = 111, OpConvertUToF = 112, OpUConvert = 113, OpSConvert = 114,
       OpFConvert = 115, OpBitcast = 124, OpSNegate = 126, OpFNegate = 127, OpIAdd = 128, OpFAdd, OpISub, OpFSub,
       OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpSMod, OpFRem, OpFMod, OpVectorTimesScalar = 142,
       OpDot = 148, OpAny = 154, OpAll = 155, OpIsNan = 156, OpIsInf = 157, OpLogicalEqual = 164, OpLogicalNotEqual,
       OpLogicalOr, OpLogicalAnd, OpLogicalNot, OpSelect = 169, OpIEqual = 170, OpINotEqual, OpUGreaterThan,
       OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan, OpULessThanEqual,
       OpSLessThanEqual, OpFOrdEqual = 180, OpFUnordEqual, OpFOrdNotEqual, OpFUnordNotEqual, OpFOrdLessThan,
       OpFUnordLessThan, OpFOrdGreaterThan, OpFUnordGreaterThan, OpFOrdLessThanEqual, OpFUnordLessThanEqual,
       OpFOrdGreaterThanEqual, OpFUnordGreaterThanEqual, OpShiftRightLogical = 194, OpShiftRightArithmetic,
       OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd, OpNot = 200, OpBitFieldInsert = 201,
       OpBitFieldSExtract = 202, OpBitFieldUExtract = 203, OpBitCount = 205, OpPhi = 245, OpLoopMerge = 246,
       OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpSwitch = 251,
       OpReturn = 253, OpUnreachable = 255, OpDemoteToHelperInvocation = 5380,
       OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod, OpImageSampleDrefImplicitLod, OpImageSampleDrefExplicitLod,
       OpImageFetch = 95, OpImageGather, OpImageDrefGather, OpImage = 100, OpImageQuerySizeLod = 103,
       OpImageQueryLod = 105, OpImageQueryLevels = 106, OpAtomicSMax = 238, OpAtomicUMax = 239 };
enum { SC_UniformConstant = 0, SC_Input = 1, SC_Uniform = 2, SC_Output = 3, SC_Function = 7, SC_StorageBuffer = 12 };

static uint32_t newid(void) {
    uint32_t id = g->next_id++;
    if (id >= g->cii) { uint32_t n = g->cii ? g->cii * 2 : 4096; while (n <= id) n *= 2; g->ii = realloc(g->ii, n * sizeof(IdInfo)); memset(g->ii + g->cii, 0, (n - g->cii) * sizeof(IdInfo)); g->cii = n; }
    g->ii[id].blk = -1;
    return id;
}
static void add_cap(uint32_t c) { for (int i = 0; i < g->ncaps; i++) if (g->caps[i] == c) return; if (g->ncaps == 16) fail("caps"); g->caps[g->ncaps++] = c; }
static void add_ext(const char *e) { for (int i = 0; i < g->nexts; i++) if (!strcmp(g->exts[i], e)) return; if (g->nexts < 4) g->exts[g->nexts++] = e; }

// dedup table for types and constants: key = instruction words without the result id
static uint32_t dedup(const uint32_t *k, int n, uint32_t extra) {
    uint32_t h = 2166136261u ^ extra;
    for (int i = 0; i < n; i++) h = (h ^ k[i]) * 16777619u;
    if (g->hcnt * 2 >= g->hcap) {
        uint32_t nc = g->hcap ? g->hcap * 2 : 1024; HE *nt = calloc(nc, sizeof(HE));
        for (uint32_t i = 0; i < g->hcap; i++) if (g->ht[i].id) { uint32_t j = g->ht[i].h & (nc - 1); while (nt[j].id) j = (j + 1) & (nc - 1); nt[j] = g->ht[i]; }
        free(g->ht); g->ht = nt; g->hcap = nc;
    }
    uint32_t j = h & (g->hcap - 1);
    while (g->ht[j].id) {
        HE *e = &g->ht[j];
        if (e->h == h && e->n == (uint32_t)n + 1 && g->hwords.w[e->off] == extra && !memcmp(g->hwords.w + e->off + 1, k, n * 4)) return e->id;
        j = (j + 1) & (g->hcap - 1);
    }
    HE *e = &g->ht[j]; e->h = h; e->n = n + 1; e->off = g->hwords.n; e->id = 0;
    wp(&g->hwords, extra); wcat(&g->hwords, k, n);
    g->hcnt++;
    return j | 0x80000000u;   // not found: slot index to fill
}
static uint32_t dd_fill(uint32_t slot, uint32_t id) { g->ht[slot & 0x7FFFFFFF].id = id; return id; }
static void inst(Words *w, int op, const uint32_t *ops, int n) { wp(w, (uint32_t)(n + 1) << 16 | op); wcat(w, ops, n); }

static uint32_t mk_type(int op, const uint32_t *ops, int n, uint32_t extra) {
    uint32_t k[8]; k[0] = op; memcpy(k + 1, ops, n * 4);
    uint32_t r = dedup(k, n + 1, extra);
    if (!(r & 0x80000000u)) return r;
    uint32_t id = newid();
    uint32_t o[8]; o[0] = id; memcpy(o + 1, ops, n * 4);
    inst(&g->types, op, o, n + 1);
    IdInfo *ii = &g->ii[id]; ii->op = op; ii->a = n > 0 ? ops[0] : 0; ii->b = n > 1 ? ops[1] : 0; ii->c = n > 2 ? ops[2] : 0;
    if (op == OpTypeInt && ops[0] == 64) add_cap(11);
    if (op == OpTypeFloat && ops[0] == 64) add_cap(10);
    return dd_fill(r, id);
}
static uint32_t t_void(void) { return mk_type(OpTypeVoid, NULL, 0, 0); }
static uint32_t t_bool(void) { return mk_type(OpTypeBool, NULL, 0, 0); }
static uint32_t t_int(int w, int s) { uint32_t o[2] = {w, s}; return mk_type(OpTypeInt, o, 2, 0); }
static uint32_t t_float(int w) { uint32_t o[1] = {w}; return mk_type(OpTypeFloat, o, 1, 0); }
static uint32_t t_vec(uint32_t c, int n) { if (n == 1) return c; uint32_t o[2] = {c, n}; return mk_type(OpTypeVector, o, 2, 0); }
static uint32_t t_ptr(int sc, uint32_t t) { uint32_t o[2] = {sc, t}; return mk_type(OpTypePointer, o, 2, 0); }
static uint32_t scalar_t(int b) {
    switch (b) { case B_BOOL: return t_bool(); case B_INT: return t_int(32, 1); case B_UINT: return t_int(32, 0);
    case B_FLOAT: return t_float(32); case B_DOUBLE: return t_float(64); case B_INT64: return t_int(64, 1); case B_UINT64: return t_int(64, 0); }
    fail("scalar type");
}
static uint32_t c_scalar(uint32_t type, uint32_t lo, uint32_t hi, int wide);
static uint32_t cu(uint32_t v) { return c_scalar(t_int(32, 0), v, 0, 0); }
static uint32_t ci(int32_t v) { return c_scalar(t_int(32, 1), (uint32_t)v, 0, 0); }
static uint32_t c_scalar(uint32_t type, uint32_t lo, uint32_t hi, int wide) {
    uint32_t k[4] = {OpConstant, type, lo, hi};
    if (g->ii[type].op == OpTypeBool) { k[0] = lo ? OpConstantTrue : OpConstantFalse; k[2] = 0; }
    int n = g->ii[type].op == OpTypeBool ? 2 : wide ? 4 : 3;
    uint32_t r = dedup(k, n, 1);
    if (!(r & 0x80000000u)) return r;
    uint32_t id = newid();
    uint32_t o[4] = {type, id, lo, hi};
    inst(&g->types, k[0], o, n == 2 ? 2 : n);
    IdInfo *ii = &g->ii[id]; ii->type = type; ii->isconst = 1; ii->a = lo; ii->b = hi;
    return dd_fill(r, id);
}
static uint32_t c_composite(uint32_t type, const uint32_t *m, int n) {
    uint32_t k[8]; k[0] = OpConstantComposite; k[1] = type; memcpy(k + 2, m, n * 4);
    uint32_t r = dedup(k, n + 2, 2);
    if (!(r & 0x80000000u)) return r;
    uint32_t id = newid();
    uint32_t o[8]; o[0] = type; o[1] = id; memcpy(o + 2, m, n * 4);
    inst(&g->types, OpConstantComposite, o, n + 2);
    g->ii[id].type = type; g->ii[id].isconst = 2;
    return dd_fill(r, id);
}
static uint32_t ty_id(Ty t);
static uint32_t ty_id(Ty t) {
    if (t.arr) fail("array type id");
    if (t.b == B_STRUCT || t.b == B_SAMPLER) fail("ty_id struct");
    return t_vec(scalar_t(t.b), t.n);
}
static uint32_t const_node(Node *n) {   // createSpvConstantFromConstUnionArray
    uint32_t st = scalar_t(n->t.b), comps[4];
    for (int i = 0; i < n->t.n; i++) {
        CV v = n->cv[i];
        switch (n->t.b) {
        case B_BOOL: comps[i] = c_scalar(st, v.i != 0, 0, 0); break;
        case B_INT: case B_UINT: comps[i] = c_scalar(st, (uint32_t)v.u, 0, 0); break;
        case B_FLOAT: { float f = (float)v.d; uint32_t u; memcpy(&u, &f, 4); comps[i] = c_scalar(st, u, 0, 0); break; }
        case B_DOUBLE: { uint64_t u; memcpy(&u, &v.d, 8); comps[i] = c_scalar(st, (uint32_t)u, (uint32_t)(u >> 32), 1); break; }
        default: comps[i] = c_scalar(st, (uint32_t)v.u, (uint32_t)(v.u >> 32), 1); break;
        }
    }
    if (n->t.n == 1) return comps[0];
    return c_composite(ty_id(n->t), comps, n->t.n);
}
static uint32_t typeof_id(uint32_t id) { return g->ii[id].type; }
static int ncomp(uint32_t type) { return g->ii[type].op == OpTypeVector ? (int)g->ii[type].b : 1; }
static uint32_t scalar_of(uint32_t type) { return g->ii[type].op == OpTypeVector ? g->ii[type].a : type; }

// ---- blocks and instructions
static int new_block(void) {
    if (g->nbl == g->cbl) { g->cbl = g->cbl ? g->cbl * 2 : 256; g->bl = realloc(g->bl, g->cbl * sizeof(Block)); }
    Block *b = &g->bl[g->nbl]; memset(b, 0, sizeof *b);
    b->label = newid(); b->merge_blk = b->cont_blk = -1;
    return g->nbl++;
}
static Block *cb(void) { return &g->bl[g->cur]; }
static void emit(int op, const uint32_t *ops, int n) {
    Block *b = cb();
    inst(&b->w, op, ops, n);
    b->prev_op = b->last_op; b->last_op = op;
}
static uint32_t emit_r(int op, uint32_t type, const uint32_t *ops, int n) {
    uint32_t id = newid(), o[16];
    if (n > 14) fail("operands");
    o[0] = type; o[1] = id; memcpy(o + 2, ops, n * 4);
    emit(op, o, n + 2);
    g->ii[id].type = type; g->ii[id].blk = g->cur;
    return id;
}
static uint32_t op1(int op, uint32_t type, uint32_t a) { return emit_r(op, type, &a, 1); }
static uint32_t op2(int op, uint32_t type, uint32_t a, uint32_t b) { uint32_t o[2] = {a, b}; return emit_r(op, type, o, 2); }
static uint32_t op3(int op, uint32_t type, uint32_t a, uint32_t b, uint32_t c) { uint32_t o[3] = {a, b, c}; return emit_r(op, type, o, 3); }
static uint32_t ext1(int f, uint32_t type, const uint32_t *args, int n) {
    uint32_t o[8]; o[0] = g->std450; o[1] = f; memcpy(o + 2, args, n * 4);
    return emit_r(OpExtInst, type, o, n + 2);
}
static void add_nc(uint32_t id) {
    if (g->nnc == g->cnc) { g->cnc = g->cnc ? g->cnc * 2 : 256; g->ncs = realloc(g->ncs, g->cnc * 4); }
    g->ncs[g->nnc++] = id;
}
static int terminated(void) {
    int op = cb()->last_op;
    return op == OpBranch || op == OpBranchConditional || op == OpSwitch || op == OpReturn || op == OpUnreachable || op == 252 || op == 4416;
}
static void add_succ(int from, int to) {
    Block *b = &g->bl[from];
    if (b->nsucc == b->csucc) { b->csucc = b->csucc ? b->csucc * 2 : 4; b->succ = realloc(b->succ, b->csucc * sizeof(int)); }
    b->succ[b->nsucc++] = to;
}
static void branch(int to) { uint32_t l = g->bl[to].label; emit(OpBranch, &l, 1); add_succ(g->cur, to); }
static void cond_branch(uint32_t c, int t, int f) {
    uint32_t o[3] = {c, g->bl[t].label, g->bl[f].label}; emit(OpBranchConditional, o, 3);
    add_succ(g->cur, t); add_succ(g->cur, f);
}
static void sel_merge(int m) { uint32_t o[2] = {g->bl[m].label, 0}; emit(OpSelectionMerge, o, 2); cb()->merge_blk = m; cb()->cont_blk = -1; }
static void dead_block(void) { g->cur = new_block(); }   // createAndSetNoPredecessorBlock

// function-scope OpVariable (placed at the start of the entry block)
static uint32_t local_var(uint32_t type) {
    uint32_t pt = t_ptr(SC_Function, type), id = newid();
    uint32_t o[3] = {pt, id, SC_Function};
    inst(&g->locals, OpVariable, o, 3);
    g->ii[id].type = pt; g->ii[id].blk = 0;
    return id;
}
static uint32_t pointee(uint32_t ptrid) { return g->ii[typeof_id(ptrid)].b; }
static int storage_of(uint32_t ptrid) { return (int)g->ii[typeof_id(ptrid)].a; }

// ---- types of variables
static uint32_t array_t(uint32_t elem, int len, int stride) {
    uint32_t o[2] = {elem, len > 0 ? cu(len) : 0};
    uint32_t id = len > 0 ? mk_type(OpTypeArray, o, 2, (uint32_t)stride) : mk_type(OpTypeRuntimeArray, o, 1, (uint32_t)stride);
    if (stride && !g->ii[id].c) { g->ii[id].c = 1; uint32_t d[3] = {id, 6, (uint32_t)stride}; inst(&g->decos, OpDecorate, d, 3); }
    if (len > 0) { g->ii[id].b = (uint32_t)len; }
    return id;
}
static uint32_t var_type(Ty t, int explicit_layout) {
    if (t.b == B_STRUCT) {
        StructDef *sd = t.sd;
        if (sd->spv) return sd->spv;
        uint32_t mem[4];
        for (int i = 0; i < sd->nm; i++) {
            Ty mt = *sd->m[i].t;
            uint32_t et = ty_id((Ty){.b = mt.b, .n = mt.n});
            if (mt.arr) {
                int stride = 0;
                if (!sd->is_pervertex) { int sz = (bits(mt.b) / 8) * (mt.n == 3 ? 4 : mt.n); stride = sd->std430 ? sz : (sz + 15) & ~15; }
                mem[i] = array_t(et, mt.arr, stride);
            } else mem[i] = et;
        }
        uint32_t id = newid();
        uint32_t o[5]; o[0] = id; memcpy(o + 1, mem, sd->nm * 4);
        inst(&g->types, OpTypeStruct, o, sd->nm + 1);
        g->ii[id].op = OpTypeStruct; g->ii[id].a = sd->nm;
        sd->spv = id;
        uint32_t d[4] = {id, 2}; inst(&g->decos, OpDecorate, d, 2);
        for (int i = 0; i < sd->nm; i++) {
            if (sd->is_pervertex) {
                uint32_t md[4] = {id, (uint32_t)i, 11, (uint32_t)sd->m[i].builtin}; inst(&g->decos, OpMemberDecorate, md, 4);
                if (i == 0 && g->invariant_pos) { uint32_t mi[3] = {id, 0, 18}; inst(&g->decos, OpMemberDecorate, mi, 3); }
            } else {
                if (sd->readonly) { uint32_t mr[3] = {id, (uint32_t)i, 24}; inst(&g->decos, OpMemberDecorate, mr, 3); }
                uint32_t md[4] = {id, (uint32_t)i, 35, 0}; inst(&g->decos, OpMemberDecorate, md, 4);
                if (i) fail("multi-member block");
            }
        }
        (void)explicit_layout;
        return id;
    }
    if (t.b == B_SAMPLER) {
        SampDef *s = t.samp;
        uint32_t io[7] = {t_float(32), (uint32_t)s->dim, (uint32_t)s->shadow, (uint32_t)s->arrayed, 0, 1, 0};
        uint32_t img = mk_type(OpTypeImage, io, 7, 0);
        return mk_type(OpTypeSampledImage, &img, 1, 0);
    }
    if (t.arr) return array_t(ty_id((Ty){.b = t.b, .n = t.n}), t.arr, 0);
    return ty_id(t);
}
static void deco(uint32_t id, uint32_t d, int nv, uint32_t v) { uint32_t o[3] = {id, d, v}; inst(&g->decos, OpDecorate, o, 2 + nv); }
static void iface_add(uint32_t id) { wp(&g->entry_iface, id); }
static uint32_t global_var_id(Var *v) {
    if (v->id) return v->id;
    int sc;
    switch (v->st) {
    case ST_IN: case ST_BUILTIN_IN: sc = SC_Input; break;
    case ST_OUT: sc = SC_Output; break;
    case ST_UNIFORM: sc = SC_Uniform; break;
    case ST_BUFFER: sc = SC_StorageBuffer; break;
    case ST_SAMPLER: sc = SC_UniformConstant; break;
    default: fail("global sc");
    }
    uint32_t t = var_type(v->t, 1);
    uint32_t pt = t_ptr(sc, t), id = newid();
    uint32_t o[3] = {pt, id, (uint32_t)sc};
    inst(&g->types, OpVariable, o, 3);
    g->ii[id].type = pt;
    v->id = id;
    if (v->builtin) deco(id, 11, 1, (uint32_t)v->builtin);
    if (v->st == ST_IN && v->flat) deco(id, 14, 0, 0);
    if (v->builtin == 15 || v->builtin == 17) { /* fs builtins */ }
    if (v->builtin == 42 || v->builtin == 43) { }
    if (v->location >= 0) deco(id, 30, 1, (uint32_t)v->location);
    if (v->st == ST_UNIFORM || v->st == ST_BUFFER || v->st == ST_SAMPLER) { deco(id, 34, 1, (uint32_t)v->set); deco(id, 33, 1, (uint32_t)v->binding); }
    if (v->st == ST_BUFFER && v->t.sd->readonly) deco(id, 24, 0, 0);   // readonly block: NonWritable on the variable too
    if (v->builtin == 22) g->depth_replacing = 1;
    iface_add(id);
    return id;
}

// ---------------------------------------------------------------- emission (GlslangToSpv traverser)
static void ac_clear(void) { memset(&g->ac, 0, sizeof g->ac); }
static void ac_rvalue(uint32_t id) { ac_clear(); g->ac.base = id; g->ac.rvalue = 1; }
static void ac_lvalue(uint32_t id) { ac_clear(); g->ac.base = id; }
static void ac_push(uint32_t idx) { if (g->ac.nidx == 16) fail("chain"); g->ac.idx[g->ac.nidx++] = idx; }

static uint32_t contained(uint32_t type, uint32_t member) {   // Builder::getContainedTypeId
    IdInfo *t = &g->ii[type];
    switch (t->op) {
    case OpTypeVector: case OpTypeArray: case OpTypeRuntimeArray: return t->a;
    case OpTypePointer: return t->b;
    case OpTypeStruct: {
        // struct member types: re-read from the instruction stream is costly; we keep them in a side table
        extern uint32_t struct_member(uint32_t, uint32_t); return struct_member(type, member);
    }
    }
    fail("contained");
}
// struct member type table
static __thread uint32_t sm_t[64][5]; static __thread int nsm;
uint32_t struct_member(uint32_t st, uint32_t m) {
    for (int i = 0; i < nsm; i++) if (sm_t[i][0] == st) return sm_t[i][1 + m];
    // find in the types stream
    Words *w = &g->types;
    for (uint32_t i = 0; i < w->n;) {
        uint32_t wc = w->w[i] >> 16, op = w->w[i] & 0xFFFF;
        if (op == OpTypeStruct && w->w[i + 1] == st) {
            if (nsm == 64) fail("structs");
            sm_t[nsm][0] = st; for (uint32_t k = 0; k + 2 < wc && k < 4; k++) sm_t[nsm][1 + k] = w->w[i + 2 + k];
            return sm_t[nsm++][1 + m];
        }
        i += wc;
    }
    fail("struct member");
}
static int const_scalar(uint32_t id, uint32_t *v) { if (g->ii[id].isconst != 1) return 0; *v = g->ii[id].a; return 1; }

// Builder::accessChainGetInferredType
static uint32_t ac_inferred(void) {
    AC *a = &g->ac;
    uint32_t type = typeof_id(a->base);
    if (!a->rvalue) type = contained(type, 0);
    for (int i = 0; i < a->nidx; i++) {
        if (g->ii[type].op == OpTypeStruct) { uint32_t m; if (!const_scalar(a->idx[i], &m)) fail("struct index"); type = contained(type, m); }
        else type = contained(type, 0);
    }
    if (a->nswz == 1) type = contained(type, 0);
    else if (a->nswz > 1) type = t_vec(contained(type, 0), a->nswz);
    if (a->component) type = contained(type, 0);
    return type;
}
static void ac_simplify_swz(void) {
    AC *a = &g->ac;
    if (ncomp(a->preswz) > a->nswz) return;
    for (int i = 0; i < a->nswz; i++) if (a->swz[i] != i) return;
    a->nswz = 0;
    if (!a->component) a->preswz = 0;
}
static void ac_push_swz(const uint8_t *s, int n, uint32_t preswz) {
    AC *a = &g->ac;
    if (!a->preswz) a->preswz = preswz;
    if (a->nswz) { uint8_t old[4]; memcpy(old, a->swz, 4); for (int i = 0; i < n; i++) a->swz[i] = old[s[i]]; a->nswz = n; }
    else { memcpy(a->swz, s, n); a->nswz = n; }
    ac_simplify_swz();
}
static void ac_transfer_swz(int dynamic) {
    AC *a = &g->ac;
    if (!a->nswz && !a->component) return;
    if (a->nswz > 1) return;
    if (a->nswz == 1) { ac_push(cu(a->swz[0])); a->nswz = 0; a->preswz = 0; }
    else if (dynamic && a->component) { ac_push(a->component); a->preswz = 0; a->component = 0; }
}
static uint32_t ac_collapse(void) {
    AC *a = &g->ac;
    if (a->instr) return a->instr;
    if (a->component && a->nswz > 1) {   // remapDynamicSwizzle
        uint32_t m[4]; for (int i = 0; i < a->nswz; i++) m[i] = cu(a->swz[i]);
        uint32_t map = c_composite(t_vec(t_int(32, 0), a->nswz), m, a->nswz);
        a->component = op2(OpVectorExtractDynamic, t_int(32, 0), map, a->component);
        a->nswz = 0;
    }
    if (a->component) { ac_push(a->component); a->component = 0; }
    if (!a->nidx) return a->base;
    // getResultingAccessChainType
    uint32_t type = pointee(a->base);
    for (int i = 0; i < a->nidx; i++) {
        if (g->ii[type].op == OpTypeStruct) { uint32_t m; if (!const_scalar(a->idx[i], &m)) fail("struct idx"); type = contained(type, m); }
        else type = contained(type, 0);
    }
    uint32_t o[17]; o[0] = a->base; memcpy(o + 1, a->idx, a->nidx * 4);
    a->instr = emit_r(OpAccessChain, t_ptr(storage_of(a->base), type), o, a->nidx + 1);
    return a->instr;
}
static uint32_t rvalue_swizzle(uint32_t type, uint32_t src, const uint8_t *s, int n) {
    if (n == 1) { uint32_t o[2] = {src, s[0]}; return emit_r(OpCompositeExtract, type, o, 2); }
    uint32_t o[6] = {src, src}; for (int i = 0; i < n; i++) o[2 + i] = s[i];
    return emit_r(OpVectorShuffle, type, o, n + 2);
}
// Builder::accessChainLoad (precision/nonuniform/memory access: none in this subset)
static uint32_t ac_load_t(uint32_t restype) {
    AC *a = &g->ac; uint32_t id;
    if (a->rvalue) {
        ac_transfer_swz(0);
        if (a->nidx) {
            uint32_t swzbase = a->preswz ? a->preswz : restype;
            int allc = 1; uint32_t ix[16];
            for (int i = 0; i < a->nidx; i++) if (!const_scalar(a->idx[i], &ix[i])) { allc = 0; break; }
            if (allc) { uint32_t o[17]; o[0] = a->base; memcpy(o + 1, ix, a->nidx * 4); id = emit_r(OpCompositeExtract, swzbase, o, a->nidx + 1); }
            else fail("indexable");   // glslang copies the r-value into an \"indexable\" Function variable
        } else id = a->base;
    } else {
        ac_transfer_swz(1);
        uint32_t p = ac_collapse();
        id = op1(OpLoad, pointee(p), p);
    }
    if (!a->nswz && !a->component) return id;
    if (a->nswz) {
        uint32_t st = scalar_of(typeof_id(id));
        if (a->nswz > 1) st = t_vec(st, a->nswz);
        id = rvalue_swizzle(st, id, a->swz, a->nswz);
    }
    if (a->component) id = op2(OpVectorExtractDynamic, restype, id, a->component);
    return id;
}
static uint32_t ac_load(void) { return ac_load_t(ac_inferred()); }
static uint32_t lvalue_swizzle(uint32_t type, uint32_t target, uint32_t source, const uint8_t *s, int n) {
    if (n == 1 && ncomp(typeof_id(source)) == 1) { uint32_t o[3] = {source, target, s[0]}; return emit_r(OpCompositeInsert, type, o, 3); }
    int nt = ncomp(typeof_id(target));
    uint32_t o[6] = {target, source}; uint32_t comp[4];
    for (int i = 0; i < nt; i++) comp[i] = i;
    for (int i = 0; i < n; i++) comp[s[i]] = nt + i;
    for (int i = 0; i < nt; i++) o[2 + i] = comp[i];
    return emit_r(OpVectorShuffle, type, o, nt + 2);
}
static uint32_t ac_result_type(void) {   // Builder::getResultingAccessChainType
    AC *a = &g->ac;
    uint32_t type = pointee(a->base);
    for (int i = 0; i < a->nidx; i++) {
        if (g->ii[type].op == OpTypeStruct) { uint32_t m; if (!const_scalar(a->idx[i], &m)) fail("struct idx"); type = contained(type, m); }
        else type = contained(type, 0);
    }
    return type;
}
// Builder::accessChainStore
static void ac_store(uint32_t rv) {
    AC *a = &g->ac;
    ac_transfer_swz(1);
    if (a->nswz && ncomp(ac_result_type()) != a->nswz && !a->component) {
        for (int i = 0; i < a->nswz; i++) {
            ac_push(cu(a->swz[i])); a->instr = 0;
            uint32_t base = ac_collapse();
            a->nidx--; a->instr = 0;
            uint32_t o[2] = {rv, (uint32_t)i};
            uint32_t src = emit_r(OpCompositeExtract, scalar_of(typeof_id(rv)), o, 2);
            uint32_t so[2] = {base, src}; emit(OpStore, so, 2);
        }
        return;
    }
    uint32_t base = ac_collapse(), src = rv;
    if (a->nswz) {
        uint32_t t = op1(OpLoad, pointee(base), base);
        src = lvalue_swizzle(typeof_id(t), t, src, a->swz, a->nswz);
    }
    uint32_t so[2] = {base, src}; emit(OpStore, so, 2);
}


// ---------------------------------------------------------------- part 2: traversal (TGlslangToSpvTraverser), CFG post-processing and module assembly.

static void ev(Node *n);
static void stmt(Node *n);
static uint32_t ld_expr(Node *n) { ac_clear(); ev(n); return ac_load(); }

// createBinaryOperation (non-matrix): typeProxy = basic type deciding float / unsigned
static uint32_t bin_emit(int op, int nc, uint32_t type, uint32_t l, uint32_t r, int proxy) {
    int f = is_f(proxy), u = is_u(proxy), b = proxy == B_BOOL;
    int sop = 0, cmp = 0, match = 1;
    switch (op) {
    case O_ADD: case O_ADD_A: sop = f ? OpFAdd : OpIAdd; break;
    case O_SUB: case O_SUB_A: sop = f ? OpFSub : OpISub; break;
    case O_MUL: case O_MUL_A: sop = f ? OpFMul : OpIMul; break;
    case O_VTS:
        if (f && (ncomp(typeof_id(l)) > 1 || ncomp(typeof_id(r)) > 1)) {
            if (ncomp(typeof_id(r)) > 1) { uint32_t t = l; l = r; r = t; }
            match = 0; sop = OpVectorTimesScalar;
        } else sop = f ? OpFMul : OpIMul;
        break;
    case O_DIV: case O_DIV_A: sop = f ? OpFDiv : u ? OpUDiv : OpSDiv; break;
    case O_MOD: case O_MOD_A: sop = f ? OpFMod : u ? OpUMod : OpSMod; break;
    case O_SHR: case O_SHR_A: sop = u ? OpShiftRightLogical : OpShiftRightArithmetic; break;
    case O_SHL: case O_SHL_A: sop = OpShiftLeftLogical; break;
    case O_AND: case O_AND_A: sop = OpBitwiseAnd; break;
    case O_OR: case O_OR_A: sop = OpBitwiseOr; break;
    case O_XOR: case O_XOR_A: sop = OpBitwiseXor; break;
    case O_LAND: match = 0; sop = OpLogicalAnd; break;
    case O_LOR: match = 0; sop = OpLogicalOr; break;
    case O_LXOR: match = 0; sop = OpLogicalNotEqual; break;
    default: cmp = 1;
    }
    if (sop) {
        if (match) {   // Builder::promoteScalar
            int dl = ncomp(typeof_id(l)) > 1, dr = ncomp(typeof_id(r)) > 1;
            int dir = dr - dl;   // !isScalar(right) - !isScalar(left)
            if (dir > 0) { uint32_t vt = t_vec(typeof_id(l), ncomp(typeof_id(r))); uint32_t o[4] = {l, l, l, l}; l = emit_r(OpCompositeConstruct, vt, o, ncomp(vt)); }
            else if (dir < 0) { uint32_t vt = t_vec(typeof_id(r), ncomp(typeof_id(l))); uint32_t o[4] = {r, r, r, r}; r = emit_r(OpCompositeConstruct, vt, o, ncomp(vt)); }
        }
        uint32_t id = op2(sop, type, l, r);
        if (nc) add_nc(id);
        return id;
    }
    if (!cmp) fail("binop");
    if ((op == O_EQ || op == O_NE) && ncomp(typeof_id(l)) > 1) {   // createCompositeCompare (vectors)
        uint32_t bt = t_vec(t_bool(), ncomp(typeof_id(l)));
        uint32_t c = op2(f ? (op == O_EQ ? OpFOrdEqual : OpFUnordNotEqual) : b ? (op == O_EQ ? OpLogicalEqual : OpLogicalNotEqual) : (op == O_EQ ? OpIEqual : OpINotEqual), bt, l, r);
        return op1(op == O_EQ ? OpAll : OpAny, t_bool(), c);
    }
    switch (op) {
    case O_LT: sop = f ? OpFOrdLessThan : u ? OpULessThan : OpSLessThan; break;
    case O_GT: sop = f ? OpFOrdGreaterThan : u ? OpUGreaterThan : OpSGreaterThan; break;
    case O_LE: sop = f ? OpFOrdLessThanEqual : u ? OpULessThanEqual : OpSLessThanEqual; break;
    case O_GE: sop = f ? OpFOrdGreaterThanEqual : u ? OpUGreaterThanEqual : OpSGreaterThanEqual; break;
    case O_EQ: sop = f ? OpFOrdEqual : b ? OpLogicalEqual : OpIEqual; break;
    case O_NE: sop = f ? OpFUnordNotEqual : b ? OpLogicalNotEqual : OpINotEqual; break;
    }
    uint32_t id = op2(sop, type, l, r);
    if (nc) add_nc(id);
    return id;
}

static int smear_n(uint32_t type) { return ncomp(type); }
static uint32_t smear_const(uint32_t c, int n) {   // makeSmearedConstant
    if (n <= 1) return c;
    uint32_t m[4] = {c, c, c, c};
    return c_composite(t_vec(typeof_id(c), n), m, n);
}

// createConversion
static uint32_t conv_emit(Node *n, uint32_t operand) {
    int to = n->t.b, from = n->l->t.b, vn = n->t.n > 1 ? n->t.n : 0;
    uint32_t dt = ty_id(n->t);
    int op = 0;
    if (is_s(from) && is_f(to)) op = OpConvertSToF;
    if (is_u(from) && is_f(to)) op = OpConvertUToF;
    if (is_f(from) && is_s(to)) op = OpConvertFToS;
    if (is_f(from) && is_u(to)) op = OpConvertFToU;
    if (is_s(from) && is_s(to)) op = OpSConvert;
    if (is_u(from) && is_u(to)) op = OpUConvert;
    if (is_f(from) && is_f(to)) op = OpFConvert;
    if (is_i(from) && is_i(to) && is_u(from) != is_u(to)) {
        if (bits(from) != bits(to)) fail("int width sign conv");   // createIntWidthConversion
        op = OpBitcast;
    }
    if (to == B_BOOL) {
        if (is_i(from)) { uint32_t z = smear_const(c_scalar(t_int(bits(from), 0), 0, 0, bits(from) == 64), vn); return op2(OpINotEqual, dt, operand, z); }
        uint32_t z = from == B_DOUBLE ? c_scalar(t_float(64), 0, 0, 1) : c_scalar(t_float(32), 0, 0, 0);
        return op2(OpFUnordNotEqual, dt, operand, smear_const(z, vn));
    }
    if (from == B_BOOL) {
        uint32_t zero, one;
        if (is_i(to)) { uint32_t zt = t_int(bits(to), is_s(to)); int w = bits(to) == 64; zero = c_scalar(zt, 0, 0, w); one = c_scalar(zt, 1, 0, w); }
        else if (to == B_DOUBLE) { uint64_t o = 0x3FF0000000000000ull; zero = c_scalar(t_float(64), 0, 0, 1); one = c_scalar(t_float(64), (uint32_t)o, (uint32_t)(o >> 32), 1); }
        else { zero = c_scalar(t_float(32), 0, 0, 0); one = c_scalar(t_float(32), 0x3F800000u, 0, 0); }
        zero = smear_const(zero, vn); one = smear_const(one, vn);
        return op3(OpSelect, dt, operand, one, zero);
    }
    if (!op) fail("conversion");
    return op1(op, dt, operand);
}

static int std450_of(int op, int fl, int un) {
    switch (op) {
    case O_SIN: return 13; case O_COS: return 14; case O_EXP2: return 29; case O_LOG2: return 30; case O_SQRT: return 31;
    case O_ISQRT: return 32; case O_FLOOR: return 8; case O_TRUNC: return 3; case O_RNDEVEN: return 2; case O_CEIL: return 9;
    case O_FRACT: return 10; case O_ABS: return fl ? 4 : 5; case O_FMSB: return un ? 75 : 74;
    case O_PACKD: return 59; case O_UNPACKD: return 65;
    }
    return -1;
}

// is the node a trivial leaf for short-circuiting (isTrivialLeaf)
static int trivial_leaf(Node *n) {
    if (isc(n)) return 1;
    if (n->op != N_SYM) return 0;
    return n->v->st == ST_LOCAL || n->v->st == ST_UNIFORM;   // EvqTemporary / EvqUniform (pipeline in/out, builtins: no)
}
static int is_conv_to_bool(Node *n) { return n->op == O_CONV && n->t.b == B_BOOL; }
static int trivial(Node *n) {
    if (n->t.n != 1 || n->t.arr) return 1;
    if (trivial_leaf(n)) return 1;
    int binary = n->op >= O_ADD && n->op <= O_SWZ;
    binary |= n->op >= O_ASSIGN && n->op <= O_XOR_A;
    int unary = n->op >= O_NEG && n->op <= O_ARRLEN;
    if (!binary && !unary) return 0;
    if (binary && (!trivial_leaf(n->l) || !trivial_leaf(n->r))) return 0;
    if (unary && !trivial_leaf(n->l)) return 0;
    if (is_conv_to_bool(n)) return 1;
    switch (n->op) {
    case O_LNOT: case O_EQ: case O_NE: case O_LT: case O_GT: case O_LE: case O_GE: case O_IDX_D: case O_IDX_S: case O_LXOR: case O_ANY: case O_ALL: return 1;
    }
    return 0;
}

// Builder::If
typedef struct IfB { uint32_t cond; int header, then_b, else_b, merge; } IfB;
static void if_begin(IfB *f, uint32_t cond) {
    f->cond = cond; f->then_b = new_block(); f->merge = new_block(); f->else_b = -1;
    f->header = g->cur;
    sel_merge(f->merge);
    g->cur = f->then_b;
}
static void if_else(IfB *f) { branch(f->merge); f->else_b = new_block(); g->cur = f->else_b; }
static void if_end(IfB *f) {
    branch(f->merge);
    g->cur = f->header;
    cond_branch(f->cond, f->then_b, f->else_b >= 0 ? f->else_b : f->merge);
    g->cur = f->merge;
}
// Note on block order: glslang adds blocks to the function list as they are made current; the dump order is the
// readable order (InReadableOrder.cpp), which only depends on successors, so our block indices need no list order.

static uint32_t short_circuit(Node *n) {
    uint32_t bt = t_bool();
    uint32_t l = ld_expr(n->l);
    uint32_t phi[4]; phi[0] = l; phi[1] = g->bl[g->cur].label;
    if (n->op == O_LOR) l = op1(OpLogicalNot, bt, l);
    IfB f; if_begin(&f, l);
    uint32_t r = ld_expr(n->r);
    phi[2] = r; phi[3] = g->bl[g->cur].label;
    if_end(&f);
    return emit_r(OpPhi, bt, phi, 4);
}

static uint32_t sym_id(Var *v) {
    if (v->id) return v->id;
    if (v->st == ST_LOCAL) { v->id = local_var(var_type(v->t, 0)); return v->id; }
    return global_var_id(v);
}

static uint32_t misc_emit(Node *n, uint32_t *ops) {
    int b = n->t.b, f = is_f(b), u = is_u(b);
    uint32_t type = ty_id(n->t);
    int lib = -1, op = 0, k = n->na;
    switch (n->op) {
    case O_MIN: case O_MAX: {
        lib = n->op == O_MIN ? (f ? 37 : u ? 38 : 39) : (f ? 40 : u ? 41 : 42);
        goto promote_fb;
    }
    case O_CLAMP: lib = f ? 43 : u ? 44 : 45;
        for (int pass = 1; pass <= 2; pass++) {
            int dl = ncomp(typeof_id(ops[0])) > 1, dr = ncomp(typeof_id(ops[pass])) > 1;
            if (dr - dl < 0) { uint32_t vt = t_vec(typeof_id(ops[pass]), ncomp(typeof_id(ops[0]))); uint32_t o[4] = {ops[pass], ops[pass], ops[pass], ops[pass]}; ops[pass] = emit_r(OpCompositeConstruct, vt, o, ncomp(vt)); }
            else if (dr - dl > 0) fail("clamp promote");
        }
        break;
    case O_MIX:
        if (scalar_of(typeof_id(ops[2])) != t_bool()) lib = 46;
        else { op = OpSelect; uint32_t t = ops[0]; ops[0] = ops[2]; ops[2] = t; }
        goto promote_fb;
    case O_FMA: lib = 50; break;
    case O_POW: lib = 26; break;
    case O_DOT: if (ncomp(typeof_id(ops[0])) == 1) return bin_emit(O_MUL, 0, type, ops[0], ops[1], b); op = OpDot; break;
    case O_BFE: op = u ? OpBitFieldUExtract : OpBitFieldSExtract; break;
    case O_BFI: op = OpBitFieldInsert; break;
    default: fail("misc");
    }
    goto done;
promote_fb: {
        int dl = ncomp(typeof_id(ops[0])) > 1, dr = ncomp(typeof_id(ops[k - 1])) > 1;
        if (dr - dl > 0) { uint32_t vt = t_vec(typeof_id(ops[0]), ncomp(typeof_id(ops[k - 1]))); uint32_t o[4] = {ops[0], ops[0], ops[0], ops[0]}; ops[0] = emit_r(OpCompositeConstruct, vt, o, ncomp(vt)); }
        else if (dr - dl < 0) { uint32_t vt = t_vec(typeof_id(ops[k - 1]), ncomp(typeof_id(ops[0]))); uint32_t o[4] = {ops[k - 1], ops[k - 1], ops[k - 1], ops[k - 1]}; ops[k - 1] = emit_r(OpCompositeConstruct, vt, o, ncomp(vt)); }
    }
done:
    if (lib >= 0) return ext1(lib, type, ops, k);
    if (op == OpSelect) return op3(OpSelect, type, ops[0], ops[1], ops[2]);
    return emit_r(op, type, ops, k);
}

// Builder::createConstructor (vectors from scalars / vectors)
static uint32_t ctor_emit(Node *n, uint32_t *src) {
    uint32_t rt = ty_id(n->t), st = scalar_of(rt);
    int nt = n->t.n;
    if (n->na == 1 && ncomp(typeof_id(src[0])) == 1 && nt > 1) {   // smear
        uint32_t o[4] = {src[0], src[0], src[0], src[0]}; return emit_r(OpCompositeConstruct, rt, o, nt);
    }
    if (n->na == 1 && ncomp(typeof_id(src[0])) == nt) return src[0];
    uint32_t c[4]; int k = 0;
    for (int i = 0; i < n->na && k < nt; i++) {
        int sn = ncomp(typeof_id(src[i]));
        if (sn == 1) c[k++] = src[i];
        else for (int s = 0; s < sn && k < nt; s++) { uint32_t o[2] = {src[i], (uint32_t)s}; c[k++] = emit_r(OpCompositeExtract, st, o, 2); }
    }
    return emit_r(OpCompositeConstruct, rt, c, k);
}

static void ev(Node *n) {
    switch (n->op) {
    case N_CONST: ac_rvalue(const_node(n)); return;
    case N_SYM: ac_lvalue(sym_id(n->v)); return;
    case O_IDX_S: ev(n->l); ac_push(ci((int32_t)n->r->cv[0].i)); return;
    case O_IDX_D:
        ev(n->l);
        if (!n->l->t.arr && n->l->t.n > 1) { uint8_t s = (uint8_t)n->r->cv[0].i; ac_push_swz(&s, 1, ty_id(n->l->t)); }
        else ac_push(ci((int32_t)n->r->cv[0].i));
        return;
    case O_IDX_I: {
        ev(n->l);
        AC partial = g->ac;
        uint32_t ix = ld_expr(n->r);
        g->ac = partial;
        if (!n->l->t.arr && n->l->t.n > 1) { g->ac.component = ix; if (!g->ac.preswz) g->ac.preswz = ty_id(n->l->t); }
        else ac_push(ix);
        return;
    }
    case O_SWZ: ev(n->l); ac_push_swz(n->sw, n->nsw, ty_id(n->l->t)); return;
    case O_ASSIGN: case O_ADD_A: case O_SUB_A: case O_MUL_A: case O_DIV_A: case O_MOD_A: case O_SHL_A: case O_SHR_A:
    case O_AND_A: case O_OR_A: case O_XOR_A: {
        ac_clear(); ev(n->l); AC lv = g->ac;
        uint32_t rv = ld_expr(n->r);
        if (n->op != O_ASSIGN) {
            g->ac = lv; uint32_t left = ac_load();
            rv = bin_emit(n->op, n->nc, ty_id(n->t), left, rv, n->t.b);
        }
        g->ac = lv; ac_store(rv);
        ac_rvalue(rv);
        return;
    }
    case O_LAND: case O_LOR:
        if (!trivial(n->r)) { uint32_t r = short_circuit(n); ac_rvalue(r); return; }
        /* fallthrough */
    case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_MOD: case O_VTS: case O_SHL: case O_SHR: case O_AND:
    case O_OR: case O_XOR: case O_LXOR: case O_LT: case O_GT: case O_LE: case O_GE: case O_EQ: case O_NE: {
        uint32_t l = ld_expr(n->l);
        uint32_t r = ld_expr(n->r);
        uint32_t id = bin_emit(n->op, n->nc, ty_id(n->t), l, r, n->l->t.b);
        ac_rvalue(id);
        return;
    }
    case O_SEL: {
        uint32_t cond = ld_expr(n->c);
        int both = (n->l->op == N_SYM || isc(n->l)) && (n->r->op == N_SYM || isc(n->r));
        uint32_t rt = ty_id(n->t);
        if (both) {
            uint32_t a = ld_expr(n->l), b = ld_expr(n->r);
            uint32_t id = op3(OpSelect, rt, cond, a, b);
            ac_rvalue(id);
            return;
        }
        uint32_t var = local_var(rt);
        IfB f; if_begin(&f, cond);
        uint32_t a = ld_expr(n->l);
        ac_lvalue(var); ac_store(a);
        if_else(&f);
        uint32_t b = ld_expr(n->r);
        ac_lvalue(var); ac_store(b);
        if_end(&f);
        ac_lvalue(var);
        return;
    }
    case O_PREINC: case O_PREDEC: case O_POSTINC: case O_POSTDEC: {
        ac_clear(); ev(n->l);
        uint32_t operand = ac_load();
        uint32_t one;
        if (n->t.b == B_FLOAT) one = c_scalar(t_float(32), 0x3F800000u, 0, 0);
        else if (n->t.b == B_DOUBLE) one = c_scalar(t_float(64), 0, 0x3FF00000u, 1);
        else if (bits(n->t.b) == 64) one = c_scalar(t_int(64, 1), 1, 0, 1);
        else one = ci(1);
        int inc = n->op == O_PREINC || n->op == O_POSTINC;
        uint32_t r = bin_emit(inc ? O_ADD : O_SUB, n->nc, ty_id(n->t), operand, one, n->t.b);
        ac_store(r);
        ac_rvalue(n->op == O_PREINC || n->op == O_PREDEC ? r : operand);
        return;
    }
    case O_CONV: { uint32_t x = ld_expr(n->l); ac_rvalue(conv_emit(n, x)); return; }
    case O_NEG: case O_LNOT: case O_BNOT: case O_SIN: case O_COS: case O_EXP2: case O_LOG2: case O_SQRT: case O_ISQRT:
    case O_FLOOR: case O_TRUNC: case O_RNDEVEN: case O_CEIL: case O_FRACT: case O_ABS: case O_ISNAN: case O_ISINF:
    case O_F2I: case O_F2U: case O_I2F: case O_U2F: case O_PACKD: case O_UNPACKD: case O_BITCNT: case O_FMSB: case O_ANY: case O_ALL: {
        uint32_t x = ld_expr(n->l);
        uint32_t t = ty_id(n->t), id;
        int pf = is_f(n->l->t.b), pu = is_u(n->l->t.b);
        int lib = std450_of(n->op, pf, pu);
        if (lib >= 0) id = ext1(lib, t, &x, 1);
        else {
            int op;
            switch (n->op) {
            case O_NEG: op = pf ? OpFNegate : OpSNegate; break;
            case O_LNOT: op = OpLogicalNot; break;
            case O_BNOT: op = OpNot; break;
            case O_ISNAN: op = OpIsNan; break;
            case O_ISINF: op = OpIsInf; break;
            case O_F2I: case O_F2U: case O_I2F: case O_U2F: op = OpBitcast; break;
            case O_BITCNT: op = OpBitCount; break;
            case O_ANY: op = OpAny; break;
            case O_ALL: op = OpAll; break;
            default: fail("unary");
            }
            id = op1(op, t, x);
        }
        if (n->nc) add_nc(id);
        ac_rvalue(id);
        return;
    }
    case O_MIN: case O_MAX: case O_CLAMP: case O_MIX: case O_FMA: case O_POW: case O_DOT: case O_BFE: case O_BFI: {
        uint32_t ops[4];
        for (int i = 0; i < n->na; i++) ops[i] = ld_expr(n->a[i]);
        ac_rvalue(misc_emit(n, ops));
        return;
    }
    case O_TEX: {   // createImageTextureFunctionCall + Builder::createTextureCall / createTextureQueryCall
        uint32_t args[4];
        for (int i = 0; i < n->na; i++) args[i] = ld_expr(n->a[i]);   // translateArguments
        SampDef *s = n->a[0]->t.samp;
        int kind = (int)n->cv[0].i;
        uint32_t smp = args[0], rt = ty_id(n->t), id;
        if (kind == TX_SIZE || kind == TX_QLEVELS || kind == TX_FETCH) smp = op1(OpImage, g->ii[typeof_id(smp)].a, smp);
        if (kind == TX_SIZE) { id = op2(OpImageQuerySizeLod, rt, smp, args[1]); add_cap(50); }
        else if (kind == TX_QLEVELS) { id = op1(OpImageQueryLevels, rt, smp); add_cap(50); }
        else if (kind == TX_QLOD) { id = op2(OpImageQueryLod, t_vec(scalar_of(typeof_id(args[1])), 2), smp, args[1]); add_cap(50); }
        else {
            int fetch = kind == TX_FETCH, gather = kind == TX_GATHER, extra = 0, bias_p = 0, noimp = 0;
            uint32_t coords = args[1], dref = 0, lod = 0, gx = 0, gy = 0, comp = 0, bias = 0;
            if (kind != TX_LOD && kind != TX_GRAD && !fetch && n->na > 2 + gather) bias_p = 1;
            if (s->shadow && gather) { dref = args[2]; extra++; }
            else if (s->shadow) { uint32_t o[2] = {coords, (uint32_t)ncomp(typeof_id(coords)) - 1}; dref = emit_r(OpCompositeExtract, scalar_of(typeof_id(coords)), o, 2); }
            if (kind == TX_LOD || fetch) lod = args[2 + extra++];
            else if (!g->fs) noimp = 1;
            if (kind == TX_GRAD) { gx = args[2 + extra]; gy = args[3 + extra]; extra += 2; }
            if (gather && !s->shadow) comp = 2 + extra < n->na ? args[2 + extra++] : ci(0);
            if (bias_p) bias = args[2 + extra++];
            uint32_t o[10]; int k = 0, expl = 0; uint32_t mask = 0;
            o[k++] = smp; o[k++] = coords; if (dref) o[k++] = dref; if (comp) o[k++] = comp;
            int mpos = k++;
            if (bias) { mask |= 1; o[k++] = bias; }
            if (lod) { mask |= 2; o[k++] = lod; expl = 1; }
            else if (gx) { mask |= 4; o[k++] = gx; o[k++] = gy; expl = 1; }
            else if (noimp && !fetch && !gather) { mask |= 2; o[k++] = c_scalar(t_float(32), 0, 0, 0); expl = 1; }
            if (mask) o[mpos] = mask; else k = mpos;
            int op = fetch ? OpImageFetch : gather ? (dref ? OpImageDrefGather : OpImageGather)
                   : expl ? (dref ? OpImageSampleDrefExplicitLod : OpImageSampleExplicitLod)
                          : (dref ? OpImageSampleDrefImplicitLod : OpImageSampleImplicitLod);
            id = emit_r(op, rt, o, k);
        }
        ac_rvalue(id);
        return;
    }
    case O_SGMIN: case O_SGALL: case O_SGSHUF: {   // createSubgroupOperation: Subgroup scope, [Reduce], operands
        uint32_t o[4]; int k = 0;
        o[k++] = cu(3);
        add_cap(61);
        int op, b = n->op == O_SGSHUF ? n->t.b : n->l->t.b;   // typeProxy: node type (aggregate) / operand type (unary)
        if (n->op == O_SGMIN) { add_cap(63); op = is_f(b) ? 355 : is_u(b) ? 354 : 353; o[k++] = 0; o[k++] = ld_expr(n->l); }
        else if (n->op == O_SGALL) { add_cap(62); op = 334; o[k++] = ld_expr(n->l); }
        else { add_cap(65); op = 346; uint32_t x = ld_expr(n->a[0]); uint32_t y = ld_expr(n->a[1]); o[k++] = x; o[k++] = y; }
        ac_rvalue(emit_r(op, ty_id(n->t), o, k));
        return;
    }
    case O_ATOMIC: {   // atomic RMW: pointer (accessChainGetLValue), Device scope, relaxed semantics, value
        ac_clear(); ev(n->a[0]); ac_transfer_swz(1);
        uint32_t ptr = ac_collapse();
        uint32_t val = ld_expr(n->a[1]);
        uint32_t o[4] = {ptr, cu(1), cu(0), val};
        ac_rvalue(emit_r(is_u(n->t.b) ? OpAtomicUMax : OpAtomicSMax, ty_id(n->t), o, 4));
        return;
    }
    case O_CTOR: {
        uint32_t src[4];
        if (n->na > 4) fail("ctor args");
        for (int i = 0; i < n->na; i++) src[i] = ld_expr(n->a[i]);   // translateArguments
        ac_rvalue(ctor_emit(n, src));
        return;
    }
    }
    fail("expression kind");
}

// ---- statements
static void loop_exit(void) { branch(g->loopm[g->nloop - 1]); dead_block(); }
static void stmt(Node *n) {
    if (!n) return;
    switch (n->op) {
    case S_SEQ: for (int i = 0; i < n->na; i++) stmt(n->a[i]); return;
    case S_IF: {
        uint32_t c = ld_expr(n->c);
        IfB f; if_begin(&f, c);
        stmt(n->l);
        if (n->r) { if_else(&f); stmt(n->r); }
        if_end(&f);
        return;
    }
    case S_LOOP: {
        int head = new_block(), body = new_block(), merge = new_block(), cont = new_block();   // makeNewLoop
        g->loopm[g->nloop] = merge; g->loopc[g->nloop] = cont; g->nloop++;
        if (g->nloop >= 64) fail("loop depth");
        branch(head);
        g->cur = head;
        { uint32_t o[3] = {g->bl[merge].label, g->bl[cont].label, 0}; emit(OpLoopMerge, o, 3); cb()->merge_blk = merge; cb()->cont_blk = cont; }
        if (n->test_first && n->c) {
            int test = new_block();
            branch(test);
            g->cur = test;
            uint32_t c = ld_expr(n->c);
            cond_branch(c, body, merge);
            g->cur = body;
            g->brk[g->nbrk++] = 1;
            stmt(n->l);
            branch(cont);
            g->nbrk--;
            g->cur = cont;
            if (n->r) { ac_clear(); ev(n->r); }
            branch(head);
        } else {
            branch(body);
            g->brk[g->nbrk++] = 1;
            g->cur = body;
            stmt(n->l);
            branch(cont);
            g->nbrk--;
            g->cur = cont;
            if (n->r) { ac_clear(); ev(n->r); }
            if (n->c) { uint32_t c = ld_expr(n->c); cond_branch(c, head, merge); }
            else branch(head);
        }
        g->cur = merge;
        g->nloop--;
        return;
    }
    case S_SWITCH: {
        uint32_t sel = ld_expr(n->c);
        Node *body = n->l;
        int dflt = -1, nseg = 0, ncase = 0;
        uint32_t cv[256]; int c2s[256]; Node *segs[256];
        for (int i = 0; i < body->na; i++) {
            Node *x = body->a[i];
            if (x->op == S_DEFAULT) dflt = nseg;
            else if (x->op == S_CASE) { if (ncase == 256) fail("cases"); c2s[ncase] = nseg; cv[ncase++] = (uint32_t)x->cv[0].u; }
            else { if (nseg == 255) fail("segments"); segs[nseg++] = x; }
        }
        if ((ncase && nseg == c2s[ncase - 1]) || nseg == dflt) segs[nseg++] = NULL;
        int sb[256];
        for (int s = 0; s < nseg; s++) sb[s] = new_block();
        int merge = new_block();
        sel_merge(merge);
        {
            uint32_t o[2 + 512]; int k = 0;
            int dm = dflt >= 0 ? sb[dflt] : merge;
            o[k++] = sel; o[k++] = g->bl[dm].label;
            add_succ(g->cur, dm);
            for (int i = 0; i < ncase; i++) { o[k++] = cv[i]; o[k++] = g->bl[sb[c2s[i]]].label; add_succ(g->cur, sb[c2s[i]]); }
            emit(OpSwitch, o, k);
        }
        if (g->nsw >= 16) fail("switch depth");
        g->swm[g->nsw++] = merge;
        g->brk[g->nbrk++] = 0;
        for (int s = 0; s < nseg; s++) {
            if (s > 0 && !terminated()) branch(sb[s]);
            g->cur = sb[s];
            if (segs[s]) stmt(segs[s]);
            else { branch(merge); dead_block(); }
        }
        g->nbrk--;
        if (!terminated()) { branch(merge); dead_block(); }
        g->cur = merge;
        g->nsw--;
        return;
    }
    case S_BREAK:
        if (g->nbrk && g->brk[g->nbrk - 1]) loop_exit();
        else { if (!g->nsw) fail("break"); branch(g->swm[g->nsw - 1]); dead_block(); }
        return;
    case S_CONTINUE: if (!g->nloop) fail("continue"); branch(g->loopc[g->nloop - 1]); dead_block(); return;
    case S_RETURN: emit(OpReturn, NULL, 0); dead_block(); return;
    case S_DEMOTE: emit(OpDemoteToHelperInvocation, NULL, 0); add_ext("SPV_EXT_demote_to_helper_invocation"); add_cap(5379); return;
    case S_KILL: emit(4416, NULL, 0); dead_block(); return;   // SPIR-V 1.6 GLSL: OpTerminateInvocation (makeStatementTerminator)
    case S_CASE: case S_DEFAULT: fail("case outside switch");
    }
    ac_clear(); ev(n);   // expression statement
}

// ---------------------------------------------------------------- CFG post-processing (Builder::postProcessCFG + dump)
enum { R_CF = 1, R_DEADCONT, R_DEADMERGE };
static __thread int *order; static __thread int norder;
static void rvisit(int b, int why, int header, int pass) {
    Block *k = &g->bl[b];
    if (why == R_CF) k->reach = 1;
    if (k->visited || k->delayed) return;
    if (pass == 0) { k->reason = why; k->hdr = header; }   // header: rewrite target of a dead continue
    order[norder++] = b;
    k->visited = 1;
    int m = -1, c = -1;
    if (!k->rewritten && (k->prev_op == OpSelectionMerge || k->prev_op == OpLoopMerge)) {
        m = k->merge_blk; g->bl[m].delayed = 1;
        if (k->prev_op == OpLoopMerge) { c = k->cont_blk; g->bl[c].delayed = 1; }
    }
    if (why == R_CF) for (int i = 0; i < k->nsucc; i++) rvisit(g->bl[b].succ[i], R_CF, -1, pass);
    if (c >= 0) { int w = g->bl[c].reach ? R_CF : R_DEADCONT; g->bl[c].delayed = 0; rvisit(c, w, b, pass); }
    if (m >= 0) { int w = g->bl[m].reach ? R_CF : R_DEADMERGE; g->bl[m].delayed = 0; rvisit(m, w, b, pass); }
}

// ---------------------------------------------------------------- driver
uint32_t *spvgen(const char *src, size_t len, int fs, size_t *nwords, const char **why) {
    G gs; memset(&gs, 0, sizeof gs); g = &gs;
    uint32_t *out = NULL;
    defs = NULL; work = NULL; ndefs = cdefs = nwork = cwork = 0; nsm = 0; order = NULL; norder = 0;
    if (setjmp(g->jb)) goto done;
    g->fs = fs;
    g->icap = 2048; g->itab = calloc(g->icap, sizeof(Intern *));
    init_keywords();
    I_cbr = intern_c("cbr"); I_length = intern_c("length"); I_min = intern_c("min"); I_uint = intern_c("uint");
    I_main = intern_c("main"); I_glpos = intern_c("gl_Position");
    g->gtail = &g->globals;
    lex(src, len);
    expand_macros();
    setup_builtins();
    parse_program();
    no_contraction();

    // ---- module setup (GlslangToSpv constructor)
    g->next_id = 1;
    newid();   // id 0 unused
    g->next_id = 1;
    g->std450 = newid();
    uint32_t tvoid = t_void();
    uint32_t tfn = mk_type(OpTypeFunction, &tvoid, 1, 0);
    uint32_t fn = newid();
    g->brk = malloc(sizeof(int) * 256); g->loopm = malloc(sizeof(int) * 64); g->loopc = malloc(sizeof(int) * 64); g->swm = malloc(sizeof(int) * 64);
    // linker objects: every declared global gets its variable (and an interface entry)
    for (Var *v = g->globals; v; v = v->gnext) global_var_id(v);
    g->cur = new_block();   // entry block
    stmt(g->main);
    if (!terminated()) emit(OpReturn, NULL, 0);   // leaveFunction

    // ---- readable order, dead merge / continue rewrite, second traversal for the dump
    order = malloc(sizeof(int) * (g->nbl + 1));
    norder = 0;
    rvisit(0, R_CF, -1, 0);
    for (int i = 0; i < norder; i++) {
        Block *k = &g->bl[order[i]];
        if (k->reason == R_DEADMERGE) { k->w.n = 0; k->nsucc = 0; inst(&k->w, OpUnreachable, NULL, 0); k->prev_op = 0; k->last_op = OpUnreachable; k->rewritten = 1; }
        else if (k->reason == R_DEADCONT) {   // (its successor is the header; never descended: not reached via control flow)
            k->w.n = 0; k->nsucc = 0; uint32_t l = g->bl[k->hdr].label; inst(&k->w, OpBranch, &l, 1);
            k->prev_op = 0; k->last_op = OpBranch; k->rewritten = 1;
        }
    }
    for (int i = 0; i < g->nbl; i++) { g->bl[i].visited = g->bl[i].delayed = g->bl[i].reach = 0; }
    int *live = calloc(g->nbl, sizeof(int));
    norder = 0;
    rvisit(0, R_CF, -1, 1);
    for (int i = 0; i < norder; i++) live[order[i]] = g->bl[order[i]].rewritten ? 2 : 1;

    // ---- assemble
    Words m = {0};
    wp(&m, 0x07230203); wp(&m, 0x00010600); wp(&m, 0x0008000B); wp(&m, 0); wp(&m, 0);
    add_cap(1);
    for (int i = 0; i < g->ncaps; i++) { uint32_t c = g->caps[i]; inst(&m, OpCapability, &c, 1); }
    for (int i = 0; i < g->nexts; i++) {
        uint32_t s[16] = {0}; size_t n = strlen(g->exts[i]); memcpy(s, g->exts[i], n);
        inst(&m, OpExtension, s, (int)(n / 4 + 1));
    }
    { uint32_t oo[5]; oo[0] = g->std450; memcpy(oo + 1, "GLSL.std.450\0\0\0\0", 16); inst(&m, OpExtInstImport, oo, 5); }
    { uint32_t o[2] = {0, 1}; inst(&m, OpMemoryModel, o, 2); }
    {
        Words e = {0}; wp(&e, fs ? 4 : 0); wp(&e, fn); wp(&e, 0x6E69616D); wp(&e, 0);   // "main"
        wcat(&e, g->entry_iface.w, g->entry_iface.n);
        inst(&m, OpEntryPoint, e.w, (int)e.n); free(e.w);
    }
    if (fs) { uint32_t o[2] = {fn, 7}; inst(&m, OpExecutionMode, o, 2); }
    if (g->depth_replacing) { uint32_t o[2] = {fn, 12}; inst(&m, OpExecutionMode, o, 2); }
    wcat(&m, g->decos.w, g->decos.n);
    for (int i = 0; i < g->nnc; i++) {
        uint32_t id = g->ncs[i]; int b = g->ii[id].blk;
        if (b >= 0 && live[b] == 1) { uint32_t o[2] = {id, 42}; inst(&m, OpDecorate, o, 2); }
    }
    wcat(&m, g->types.w, g->types.n);
    { uint32_t o[4] = {tvoid, fn, 0, tfn}; inst(&m, OpFunction, o, 4); }
    for (int i = 0; i < norder; i++) {
        Block *k = &g->bl[order[i]];
        inst(&m, OpLabel, &k->label, 1);
        if (order[i] == 0) wcat(&m, g->locals.w, g->locals.n);
        wcat(&m, k->w.w, k->w.n);
    }
    inst(&m, 56, NULL, 0);
    m.w[3] = g->next_id;
    free(live);
    out = m.w; *nwords = m.n;
done:
    if (why) *why = out ? NULL : g->why;
    // free
    for (int i = 0; i < g->nbl; i++) { free(g->bl[i].w.w); free(g->bl[i].succ); }
    free(g->bl); free(g->ii); free(g->types.w); free(g->decos.w); free(g->locals.w); free(g->entry_iface.w);
    free(g->ht); free(g->hwords.w); free(g->tk); free(g->undo); free(g->itab); free(g->ncs);
    free(g->brk); free(g->loopm); free(g->loopc); free(g->swm);
    free(defs); free(work); free(order);
    while (g->arena) { Chunk *c = g->arena; g->arena = c->next; free(c); }
    g = NULL;
    return out;
}

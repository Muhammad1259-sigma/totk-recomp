// HLE: printf family with guest (AArch64 AAPCS64) varargs. Conversions are rendered one at a time by the host
// snprintf after fetching the argument from the guest va_list; guest 'long' is 64-bit, so 'l' becomes 'll'.
#include "hle.h"
#include <cstdio>
#include <cstdarg>
#include <string>

// AAPCS64 va_list: { void* stack; void* gr_top; void* vr_top; int gr_offs; int vr_offs; }
struct GVaList { u64 stack, gr_top, vr_top; s32 gr_offs, vr_offs; };
static u64 va_int(GVaList* v) {
    if (v->gr_offs < 0) { u64 a = v->gr_top + v->gr_offs; v->gr_offs += 8; return *(u64*)a; }
    u64 a = v->stack; v->stack += 8; return *(u64*)a;
}
static double va_dbl(GVaList* v) {
    if (v->vr_offs < 0) { u64 a = v->vr_top + v->vr_offs; v->vr_offs += 16; return *(double*)a; }
    u64 a = v->stack; v->stack += 8; return *(double*)a;
}
// Build a va_list for a variadic call made directly into us: named args took `ngr` GPRs and `nvr` FP regs.
struct VaFrame { u64 gr[8]; V128 vr[8]; GVaList va; };
static void va_from_regs(Ctx* c, int ngr, int nvr, VaFrame* f) {
    for (int i = 0; i < 8; i++) { f->gr[i] = c->x[i]; f->vr[i] = c->v[i]; }
    f->va.stack = c->sp; f->va.gr_top = (u64)&f->gr[8]; f->va.vr_top = (u64)&f->vr[8];
    f->va.gr_offs = -(8 - ngr) * 8; f->va.vr_offs = -(8 - nvr) * 16;
}

static std::string gformat(const char* fmt, GVaList* v) {
    std::string out; char spec[32], buf[512];
    for (const char* p = fmt; *p; ) {
        if (*p != '%') { out += *p++; continue; }
        const char* s = p++;
        if (*p == '%') { out += '%'; p++; continue; }
        std::string sp = "%";
        while (*p && strchr("-+ #0'", *p)) sp += *p++;
        if (*p == '*') { sp += std::to_string((int)va_int(v)); p++; } else while (*p >= '0' && *p <= '9') sp += *p++;
        if (*p == '.') { sp += *p++; if (*p == '*') { sp += std::to_string((int)va_int(v)); p++; } else while (*p >= '0' && *p <= '9') sp += *p++; }
        int len = 0;                                  // 0 int, 1 char, 2 short, 3 long(64), 4 size/ptrdiff/intmax (64), 5 long double
        if (*p == 'h') { len = 2; p++; if (*p == 'h') { len = 1; p++; } }
        else if (*p == 'l') { len = 3; p++; if (*p == 'l') p++; }
        else if (*p == 'z' || *p == 'j' || *p == 't') { len = 4; p++; }
        else if (*p == 'L') { len = 5; p++; }
        char conv = *p ? *p++ : 0;
        switch (conv) {
        case 'd': case 'i': case 'u': case 'x': case 'X': case 'o': {
            u64 a = va_int(v);
            bool sgn = conv == 'd' || conv == 'i';
            if (len == 0) a = sgn ? (u64)(s64)(s32)a : (u32)a;
            else if (len == 1) a = sgn ? (u64)(s64)(s8)a : (u8)a;
            else if (len == 2) a = sgn ? (u64)(s64)(s16)a : (u16)a;
            snprintf(spec, sizeof spec, "%sll%c", sp.c_str(), conv);
            snprintf(buf, sizeof buf, spec, a); out += buf; break;
        }
        case 'c': { snprintf(spec, sizeof spec, "%sc", sp.c_str()); snprintf(buf, sizeof buf, spec, (int)va_int(v)); out += buf; break; }
        case 's': {
            const char* str = (const char*)va_int(v); if (!str) str = "(null)";
            if (len == 3) { out += "(wide)"; break; }
            snprintf(spec, sizeof spec, "%ss", sp.c_str());
            int n = snprintf(nullptr, 0, spec, str); std::string t(n, 0); snprintf(t.data(), n + 1, spec, str); out += t; break;
        }
        case 'p': { snprintf(buf, sizeof buf, "0x%llx", va_int(v)); out += buf; break; }
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': {
            double d = va_dbl(v);                   // %Lf (128-bit long double) not handled: never seen yet
            snprintf(spec, sizeof spec, "%s%c", sp.c_str(), conv); snprintf(buf, sizeof buf, spec, d); out += buf; break;
        }
        case 'n': { u64 a = va_int(v); if (a) *(s32*)a = (s32)out.size(); break; }
        default: out.append(s, p - s); break;      // unknown: copy through
        }
    }
    return out;
}
static u64 put(char* dst, u64 cap, const std::string& s) {
    if (dst && cap) { u64 n = s.size() < cap - 1 ? s.size() : cap - 1; memcpy(dst, s.data(), n); dst[n] = 0; }
    return s.size();
}

HLE_AS(util_VSNPrintf, "_ZN2nn4util9VSNPrintfEPcmPKcSt9__va_list") {
    GVaList v = *(GVaList*)c->x[3];
    c->x[0] = put((char*)c->x[0], c->x[1], gformat((const char*)c->x[2], &v));
}
HLE(vsnprintf) {
    GVaList v = *(GVaList*)c->x[3];
    c->x[0] = put((char*)c->x[0], c->x[1], gformat((const char*)c->x[2], &v));
}
HLE_AS(util_SNPrintf, "_ZN2nn4util8SNPrintfEPcmPKcz") {
    VaFrame f; va_from_regs(c, 3, 0, &f);
    c->x[0] = put((char*)c->x[0], c->x[1], gformat((const char*)c->x[2], &f.va));
}
HLE(snprintf) {
    VaFrame f; va_from_regs(c, 3, 0, &f);
    c->x[0] = put((char*)c->x[0], c->x[1], gformat((const char*)c->x[2], &f.va));
}
HLE(printf) {
    VaFrame f; va_from_regs(c, 1, 0, &f);
    std::string s = gformat((const char*)c->x[0], &f.va);
    printf("[guest] %s", s.c_str()); if (s.empty() || s.back() != '\n') printf("\n");
    c->x[0] = s.size();
}
// sscanf(str, fmt, ...): every argument is a pointer; translate 'l' to 'll' for integer conversions.
HLE(sscanf) {
    VaFrame f; va_from_regs(c, 2, 0, &f);
    const char* fmt = (const char*)c->x[1]; std::string hf; int n = 0;
    for (const char* p = fmt; *p; p++) {
        hf += *p;
        if (*p != '%') continue;
        if (p[1] == '%') { hf += *++p; continue; }
        bool suppress = p[1] == '*';
        while (p[1] && !strchr("diouxXcsfFeEgGaAp[n", p[1])) {
            p++;
            if (*p == 'l') {                       // l / ll: integer conversions need ll on the host, %lf stays l
                if (p[1] == 'l') p++;
                hf += (p[1] && strchr("diouxXn", p[1])) ? "ll" : "l";
            } else hf += *p;
        }
        if (p[1] == '[') { while (p[1] && p[1] != ']') hf += *++p; }
        if (p[1]) hf += *++p;
        if (!suppress) n++;
    }
    void* a[16] = {};
    for (int i = 0; i < n && i < 16; i++) a[i] = (void*)va_int(&f.va);
    c->x[0] = (u64)(s64)sscanf((const char*)c->x[0], hf.c_str(), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                               a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
}

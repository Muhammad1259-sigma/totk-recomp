"""AArch64 (TotK main NSO) -> C++ static recompiler.

usage: python gen.py <img_prefix> <funcs.json> <outdir> [--only 0x1bc,0x2c0] [--tu-insns N]

One C++ function per FDE function: `void f_<off>(Ctx* __restrict c)`.
Guest memory is identity-mapped; the image is loaded at IMGBASE.
Writes: tu_NNN.cpp, ftab.cpp (address -> function table), imports.cpp (import thunks),
report.json (coverage: unhandled instructions by mnemonic/shape).
"""
import collections, io, json, os, re, struct, sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

IMGBASE = 0x7100000000
# symbol prefixes: main keeps f_/imp_/g_; other modules (--prefix s) get s_/s_imp_/g_s_
FN, IMP, TAB = 'f_', 'imp_', 'g_'

# ------------------------------------------------------------------ operand helpers
class Unhandled(Exception):
    pass

def split_ops(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '[{':
            depth += 1
        elif ch in ']}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out

def imm(s):
    s = s.strip().lstrip('#')
    if re.match(r'^-?0x[0-9a-f]+$', s) or re.match(r'^-?\d+$', s):
        return int(s, 0)
    raise Unhandled('imm ' + s)

def fimm(s):
    return float(s.strip().lstrip('#'))

GPR = re.compile(r'^([xw])(\d+)$')
NOWRITE_D = {'cmp', 'cmn', 'tst', 'ccmp', 'ccmn', 'fcmp', 'fccmp', 'cbz', 'cbnz', 'tbz', 'tbnz', 'br', 'blr', 'ret', 'prfm'}   # first operand is read, not written
def is_gpr(o): return bool(GPR.match(o)) or o in ('sp', 'wsp', 'xzr', 'wzr')
def gsize(o):
    if o in ('sp', 'xzr') or o.startswith('x'): return 64
    if o in ('wsp', 'wzr') or o.startswith('w'): return 32
    raise Unhandled('gsize ' + o)
def T(n): return 'u64' if n == 64 else 'u32'
def ST(n): return 's64' if n == 64 else 's32'

def R(o):
    """read expression of a GPR operand"""
    if o == 'xzr': return '0ull'
    if o == 'wzr': return '0u'
    if o == 'sp': return 'c->sp'
    if o == 'wsp': return '(u32)c->sp'
    m = GPR.match(o)
    if not m: raise Unhandled('reg ' + o)
    return f'c->x[{m.group(2)}]' if m.group(1) == 'x' else f'(u32)c->x[{m.group(2)}]'

def W(o, e):
    """write statement for a GPR operand"""
    if o in ('xzr', 'wzr'): return f'(void)({e});'
    if o == 'sp': return f'c->sp = (u64)({e});'
    if o == 'wsp': return f'c->sp = (u32)({e});'
    m = GPR.match(o)
    if not m: raise Unhandled('wreg ' + o)
    return f'c->x[{m.group(2)}] = (u64)({e});' if m.group(1) == 'x' else f'c->x[{m.group(2)}] = (u32)({e});'

SHIFTS = ('lsl', 'lsr', 'asr', 'ror')
EXTS = ('uxtb', 'uxth', 'uxtw', 'uxtx', 'sxtb', 'sxth', 'sxtw', 'sxtx')

def shift_expr(val, kind, amt, n):
    t = T(n)
    if kind == 'lsl': return f'(({t})({val}) << {amt})' if amt else f'({t})({val})'
    if kind == 'lsr': return f'(({t})({val}) >> {amt})' if amt else f'({t})({val})'
    if kind == 'asr': return f'({t})(({ST(n)})({val}) >> {amt})'
    if kind == 'ror': return f'ror{n}(({t})({val}), {amt})'
    raise Unhandled('shift ' + kind)

def ext_expr(val, kind, amt, n):
    t = T(n)
    cast = {'uxtb': '(u8)', 'uxth': '(u16)', 'uxtw': '(u32)', 'uxtx': '(u64)',
            'sxtb': '(s8)', 'sxth': '(s16)', 'sxtw': '(s32)', 'sxtx': '(s64)'}[kind]
    e = f'({t})({ST(n)}){cast}({val})' if kind.startswith('s') else f'({t}){cast}({val})'
    return f'({e} << {amt})' if amt else e

def parse_mod(s):
    parts = s.split()
    kind = parts[0]
    amt = imm(parts[1]) if len(parts) > 1 else 0
    return kind, amt

def op2(ops, n):
    """second operand (reg/imm with optional shift/extend) as an expression of width n"""
    a = ops[0]
    if a.startswith('#'):
        v = imm(a)
        if len(ops) > 1:
            k, amt = parse_mod(ops[1])
            if k != 'lsl': raise Unhandled('imm mod')
            v <<= amt
        v &= (1 << n) - 1
        return f'{v:#x}ull' if n == 64 else f'{v:#x}u'
    val = R(a)
    if len(ops) > 1:
        k, amt = parse_mod(ops[1])
        if k in SHIFTS: return shift_expr(val, k, amt, n)
        if k in EXTS: return ext_expr(val, k, amt, n)
        raise Unhandled('mod ' + k)
    return f'({T(n)})({val})' if n == 32 and a.startswith('x') else val

COND = {'eq': 'C_EQ', 'ne': 'C_NE', 'hs': 'C_HS', 'cs': 'C_HS', 'lo': 'C_LO', 'cc': 'C_LO', 'mi': 'C_MI', 'pl': 'C_PL',
        'vs': 'C_VS', 'vc': 'C_VC', 'hi': 'C_HI', 'ls': 'C_LS', 'ge': 'C_GE', 'lt': 'C_LT', 'gt': 'C_GT', 'le': 'C_LE', 'al': 'C_AL'}

# ------------------------------------------------------------------ SIMD/FP register helpers
VREG = re.compile(r'^v(\d+)\.(\d*)([bhsdq])$')
VELEM = re.compile(r'^v(\d+)\.([bhsd])\[(\d+|0x[0-9a-f]+)\]$')
FREG = re.compile(r'^([bhsdq])(\d+)$')
ESZ = {'b': 8, 'h': 16, 's': 32, 'd': 64, 'q': 128}

def varr(o):
    m = VREG.match(o)
    if not m: raise Unhandled('varr ' + o)
    reg, cnt, e = int(m.group(1)), int(m.group(2) or 1), m.group(3)
    return reg, cnt, ESZ[e]

def velem(o):
    m = VELEM.match(o)
    if not m: raise Unhandled('velem ' + o)
    return int(m.group(1)), ESZ[m.group(2)], int(m.group(3), 0)

def freg(o):
    m = FREG.match(o)
    if not m: raise Unhandled('freg ' + o)
    return m.group(1), int(m.group(2))

UF = {8: 'b', 16: 'h', 32: 's', 64: 'd'}       # unsigned lane field
SF = {8: 'sb', 16: 'sh', 32: 'ss', 64: 'sd'}   # signed lane field
FF = {32: 'f', 64: 'fd'}                       # float lane field
UT = {8: 'u8', 16: 'u16', 32: 'u32', 64: 'u64'}
STT = {8: 's8', 16: 's16', 32: 's32', 64: 's64'}
FT = {32: 'float', 64: 'double'}

def fscal(o):
    """scalar FP register read expr + width (s/d)"""
    k, r = freg(o)
    if k == 's': return f'c->v[{r}].f[0]', 32
    if k == 'd': return f'c->v[{r}].fd[0]', 64
    if k == 'h': return f'c->v[{r}].h[0]', 16
    raise Unhandled('fscal ' + o)

def fwrite(o, e):
    k, r = freg(o)
    if k == 's': return f'c->v[{r}] = vS({e});'
    if k == 'd': return f'c->v[{r}] = vD({e});'
    if k == 'h': return f'c->v[{r}] = vU16({e});'
    raise Unhandled('fwrite ' + o)

def lanes_loop(dreg, cnt, esz, body, field):
    """emit V128 loop: body uses 'i' and returns lane value"""
    return f'{{ V128 r_ = vzero(); LANES(0, {cnt}, r_.{field}[i] = {body}); c->v[{dreg}] = r_; }}'

def lanes_fp(dreg, cnt, field, exact, native):
    """FP vector op (add/sub/mul/div/fma): native lanes + one NaN test for the whole vector; the exact ARM NaN path (the
    branch-free A forms, per lane) runs only when a lane came out NaN (a NaN operand always gives a NaN result for these
    ops). The A forms per lane were ~76 x86 insns per 4-lane op (rt.h fast forms, 2026-10-17). Sources are read before
    c->v[dreg] is written, so the exact path still sees the original operands."""
    return (f'{{ V128 r_ = vzero(); bool nn_ = false; '
            f'LANES(0, {cnt}, {{ auto x_ = {native}; nn_ |= x_ != x_; r_.{field}[i] = x_; }}); '
            f'if (__builtin_expect(nn_, 0)) {{ LANES(0, {cnt}, r_.{field}[i] = {exact}); }} c->v[{dreg}] = r_; }}')

# ------------------------------------------------------------------ memory operands
MEM = re.compile(r'^\[(\w+)(?:,\s*(.+?))?\](!?)$')

def mem_ea(ops_after):
    """ops_after: operand list starting at the memory operand. Returns (setup, ea_expr, post_stmt)"""
    m = MEM.match(ops_after[0])
    if not m: raise Unhandled('mem ' + ops_after[0])
    base, rest, pre = m.group(1), m.group(2), m.group(3) == '!'
    b = R(base)
    post = ''
    if rest is None:
        ea = b
        if len(ops_after) > 1:  # post-index
            p = ops_after[1]
            inc = str(imm(p)) + 'll' if p.startswith('#') else R(p)
            post = W(base, f'{b} + (u64)({inc})')
    else:
        rp = split_ops(rest)
        if rp[0].startswith('#'):
            ea = f'{b} + (u64)({imm(rp[0])}ll)'
        else:
            idx = rp[0]
            n = 64
            if len(rp) > 1:
                k, amt = parse_mod(rp[1])
                if k == 'lsl': off = f'(({R(idx)}) << {amt})'
                elif k in EXTS: off = ext_expr(R(idx), k, amt, 64)
                else: raise Unhandled('memmod ' + k)
            else:
                off = R(idx)
            ea = f'{b} + (u64)({off})'
        if pre:
            post = W(base, 'ea')
    return ea, post

# ------------------------------------------------------------------ the emitter
class Emitter:
    def __init__(self, img, info, fde, plt, imports):
        self.adr_sites = []           # br sites resolved only by the in-function adr fallback (report.json)
        self.trace = set()            # --trace: functions that log entry (x0-x7, lr) and each ret (x0) via rt_trace
        self.hooks = set()            # --hook: functions emitted as <name>_orig; <name> calls the runtime's hook_<name>
        self.img = img
        self.fde = fde                # start -> size
        self.fstarts = set(fde)
        self.plt = plt                # plt addr -> import index
        self.imports = imports        # index -> name
        self.unhandled = collections.Counter()
        self.unhandled_ex = {}
        self.stats = collections.Counter()
        self.md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        self.md.skipdata = True

    # -------------------------------------------------- per function
    def function(self, start):
        size = self.fde[start]
        end = start + size
        insns = list(self.md.disasm_lite(self.img[start:end], start))
        self.cur_refs = set()
        self.cur_imps = set()
        labels = set()
        jt = {}
        # `adr xN, #L` with L inside this function, later `br xN`: a local continuation (not a tail call). Such br
        # sites switch over every in-function adr target and fall back to call_ind (measured: f_8631a0, U5).
        adr_t = sorted({t for (a, sz, m, o) in insns if m == 'adr'
                        for t in [imm(split_ops(o)[-1])] if start <= t < end})
        for i, (a, sz, m, o) in enumerate(insns):
            if m == 'b' or m.startswith('b.') or m in ('cbz', 'cbnz', 'tbz', 'tbnz'):
                t = imm(split_ops(o)[-1])
                if start <= t < end:
                    labels.add(t)
            elif m == 'br':
                tab = self.jump_table(insns, i, start, end)
                if tab:
                    jt[a] = tab
                    labels.update(tab)
                    self.stats['br_jumptable'] += 1
                elif adr_t:
                    jt[a] = adr_t
                    labels.update(adr_t)
                    self.stats['br_adr'] += 1; self.adr_sites.append(a)
                else:
                    self.stats['br_indirect'] += 1
        self.labels = labels
        self.fstart, self.fend = start, end
        items = []                    # (address, code, falls through); labels go in front when the body is assembled
        for i, (a, sz, m, o) in enumerate(insns):
            try:
                code = self.insn(a, m, o, jt.get(a))
            except (Unhandled, ValueError, KeyError, IndexError) as e:
                shape = re.sub(r'\b[xw]\d+\b', 'R', o)
                shape = re.sub(r'#-?[0-9a-fx.e+]+', '#', shape)
                key = f'{m} {shape}'
                self.unhandled[key] += 1
                self.unhandled_ex.setdefault(key, f'{a:#x}: {m} {o} ({e})')
                code = f'{{ LSO; rt_trap(cx, IMGBASE + {a:#x}ull, "unhandled: {m} {o}"); }}'
                m = '.byte'   # (traps: no successor)
            self.stats['insns'] += 1
            items.append((a, code, m not in ('ret', 'b', 'br', '.byte')))
        # fall-through off the end
        last = insns[-1][2] if insns else ''
        if last not in ('ret', 'b', 'br'):
            items.append((end, self.tail(end, end), False))
        codes = self.lctx_sync([it[0] for it in items], [it[1] for it in items], [it[2] for it in items])
        body = [f'rt_trace(cx, IMGBASE + {start:#x}ull, 0);'] if start in self.trace else []
        for (a, _, _), code in zip(items, codes):
            if a in labels:
                body.append(f'L_{a:x}:;')
            body.append(f'/*{a:x}*/ {code}')
        text = 'LCtx lc_(cx); LCtx* const c = &lc_;\n' + '\n'.join(body)
        if start in self.hooks:  # runtime wrapper (e.g. TotK's frame-delta setter): it calls <name>_orig itself
            return (f'void hook_{FN}{start:x}(Ctx* __restrict c);\n'
                    f'void {FN}{start:x}(Ctx* __restrict c) {{ hook_{FN}{start:x}(c); }}\n'
                    f'void {FN}{start:x}_orig(Ctx* __restrict cx) {{\n' + text + '\n}\n')
        return f'void {FN}{start:x}(Ctx* __restrict cx) {{\n' + text + '\n}\n'

    # Registers live in an LCtx local (runtime/rt.h): `c` is that local inside generated functions, `cx` the real Ctx.
    # LSO (before a call / return / trap) writes back the registers that may have changed since the last sync on any
    # path here (x0-x30 bits 0-30, sp 31, NZCV 32); LSI (after a call returns) reloads all of them, so they are clean
    # again. "May have changed" = forward dataflow over the function's instructions: union over predecessors
    # (fall-through unless ret / b / br / trap, plus every `goto L_x` in the code: may over-approximate, never under;
    # fall-through after returns took f_704d7c's syncs from 18.8 registers on average). A first version stored
    # every register the function writes anywhere at every sync: f_704d7c grew from 5,920 to 9,351 x86 instructions.
    # Every register write in the emitted text is a plain `c->x[N] = ` / `c->sp = ` assignment (checked over gen/main +
    # gen/sdk 2026-10-16: no compound forms) and NZCV changes only through the flag helpers below.
    LCTX_FLAGS = re.compile(r'\b(?:adds|subs)(?:32|64)\(c,|\bnz(?:32|64)\(c,|\bsetnzcv\(c,|\bfcmpA\(c,|c->(?:n|z|c|vf)\s*=(?!=)')
    LCTX_X = re.compile(r'c->x\[(\d+)\]\s*=(?!=)')
    LCTX_SP = re.compile(r'c->sp\s*=(?!=)')
    LCTX_GOTO = re.compile(r'goto L_([0-9a-f]+);')
    def lctx_wmask(self, s):
        m = 0
        for mm in self.LCTX_X.finditer(s):
            m |= 1 << int(mm.group(1))
        if self.LCTX_SP.search(s): m |= 1 << 31
        if self.LCTX_FLAGS.search(s): m |= 1 << 32
        return m
    def lctx_sync(self, addrs, codes, falls):
        n = len(codes)
        idx = {a: i for i, a in enumerate(addrs)}
        pre, out_fixed, allm, succ = [], [], [], []
        for i, cd in enumerate(codes):
            parts = cd.split('LSO;')
            cum, p = 0, []
            for part in parts[:-1]:
                cum |= self.lctx_wmask(part)
                p.append(cum)
            pre.append(p)
            allm.append(cum | self.lctx_wmask(parts[-1]))
            k = cd.rfind('LSI;')   # an unconditional call: clean after it (conditional code keeps the union)
            out_fixed.append(self.lctx_wmask(cd[k + 4:]) if k >= 0 and not cd.lstrip().startswith('if') else None)
            t = [idx[g] for g in (int(h, 16) for h in self.LCTX_GOTO.findall(cd)) if g in idx]
            if i + 1 < n and falls[i]: t.append(i + 1)
            succ.append(t)
        din = [0] * n                 # entry: the locals were just loaded from the real Ctx
        changed = True
        while changed:
            changed = False
            for i in range(n):
                out = out_fixed[i] if out_fixed[i] is not None else din[i] | allm[i]
                for j in succ[i]:
                    v = din[j] | out
                    if v != din[j]:
                        din[j] = v; changed = True
        res = []
        for i, cd in enumerate(codes):
            if 'LSO;' in cd:
                parts = cd.split('LSO;')
                o = parts[0]
                for k, part in enumerate(parts[1:]):
                    m = din[i] | pre[i][k]
                    self.stats['lctx_syncs'] += 1
                    self.stats['lctx_sync_regs'] += bin(m).count('1')
                    o += f'lso<{m:#x}ull>(cx, lc_);' + part
                cd = o
            res.append(cd.replace('LSI;', 'lsi(cx, lc_);'))
        self.stats['lctx_funcs'] += 1
        return res

    def fname(self, t):
        self.cur_refs.add(t)
        return f'{FN}{t:x}'

    def tail(self, t, pc):
        """tail transfer to absolute offset t"""
        if self.fstart <= t < self.fend and t in self.labels:
            return f'goto L_{t:x};'
        if t in self.fstarts:
            return f'{{ LSO; [[clang::musttail]] return {self.fname(t)}(cx); }}'
        if t in self.plt:
            self.cur_imps.add(self.plt[t])
            return f'{{ LSO; [[clang::musttail]] return {IMP}{self.plt[t]}(cx); }}'
        return f'{{ LSO; call_ind(cx, IMGBASE + {t:#x}ull); return; }}'

    def call(self, t, ret):
        pre = f'c->x[30] = IMGBASE + {ret:#x}ull; '
        if t in self.fstarts:
            return pre + f'LSO; {self.fname(t)}(cx); LSI;'
        if t in self.plt:
            self.cur_imps.add(self.plt[t])
            return pre + f'LSO; {IMP}{self.plt[t]}(cx); LSI;'
        self.stats['bl_unknown'] += 1
        return pre + f'LSO; call_ind(cx, IMGBASE + {t:#x}ull); LSI;'

    # -------------------------------------------------- jump tables
    def jump_table(self, insns, i, start, end):
        reg = insns[i][3].strip()
        # look back for: add <reg>, xA, wI, sxt?/uxt? #2 ; adr xA, #anchor ; ldr* wI, [xT, xK...] ; adrp/add xT
        rn = lambda r: r[1:] if GPR.match(r) else r      # register number, width-agnostic
        # 48 back: the compiler schedules unrelated (NEON) code between the table add and the br (f_23b8..: 18 insns)
        W = 48
        window = insns[max(0, i - W):i]
        addm = None; ai = len(window)
        for wi in range(len(window) - 1, -1, -1):
            a, sz, m, o = window[wi]
            ops = split_ops(o)
            # 3-operand form too: `ldrsw x10,[x11,x8,lsl #2]; add x9,x9,x10` (index already extended; f_b44..: br 0xb4448c
            # matched nothing, fell back to its adr constant, and case 0xb444d8 trapped in run50)
            if m == 'add' and rn(ops[0]) == rn(reg) and len(ops) in (3, 4) and GPR.match(ops[2]):
                addm = ops; ai = wi; break
            if ops and rn(ops[0]) == rn(reg) and m not in NOWRITE_D and not m.startswith('st'):
                break                    # the br register comes from something else
        if not addm: return None
        anchor_reg, idx_reg = addm[1], addm[2]
        k, amt = parse_mod(addm[3]) if len(addm) == 4 else ('', 0)
        anchor = None; load = None; li = i
        for wi in range(len(window) - 1, -1, -1):
            a, sz, m, o = window[wi]
            ops = split_ops(o)
            if m == 'adr' and rn(ops[0]) == rn(anchor_reg) and anchor is None:
                anchor = imm(ops[1])
            # the table load comes before the add and is register-indexed (f_cc9..: `ldrh w12,[x13,x9,lsl #1]`, then
            # x12 reused by `ldr x12,[sp,#0x88]` between the add and the br)
            if (m in ('ldrb', 'ldrh', 'ldrsw', 'ldrsb', 'ldrsh', 'ldr') and rn(ops[0]) == rn(idx_reg) and load is None
                    and wi < ai and len(ops) > 1 and (mo := MEM.match(ops[1])) and mo.group(2) and GPR.match(split_ops(mo.group(2))[0])):
                load = (m, ops); li = max(0, i - W) + wi
        if anchor is None or load is None: return None
        m, ops = load
        mm = MEM.match(ops[1])
        if not mm or not mm.group(2): return None
        tbase_reg = mm.group(1)
        idx_of_table = split_ops(mm.group(2))[0]
        # table base: adrp + add
        # The base may be set long before (a callee-saved reg hoisted out of a loop: f_106f138's x26, set 120 insns
        # earlier), so scan back through the whole function, stopping at any other write to the base register.
        # Scan from the table load, not the br: the base register may be reused in between (f_2cb5b8: `ldrb w14,[x10,x12]`
        # then `mov w10,#1` before `br x13`).
        page = off = None
        for a, sz, mn, o in reversed(insns[:li]):
            p = split_ops(o)
            if not p or rn(p[0]) != rn(tbase_reg): continue
            if mn in ('add', 'sub') and len(p) == 3 and p[2].startswith('#') and rn(p[1]) == rn(tbase_reg):
                off = (off or 0) + (imm(p[2]) if mn == 'add' else -imm(p[2])); continue   # adrp + add + add (f_4d9..)
            if mn == 'adrp':
                page = imm(p[1]); break
            if mn in NOWRITE_D or mn.startswith('st'): continue
            if mn.startswith('ld') and '[sp' in o: continue     # epilogue restore on another path
            break
        # (no adrp as the nearest definition, e.g. f_7a5078's `add x21,x19,#0xd0` on another path: use the alternates below)
        tabs = [page + (off or 0)] if page is not None else []
        # The base register can hold different tables on different paths into a shared dispatch (f_7a5078's x21: three
        # loops each load their own table at 0x7a5140/0x7a5320/0x7a53e0, then branch to common `ldrh w10,[x21,x8,lsl #1]`
        # + br sites; the nearest definition by address gave the wrong table and case 0x7a59b0 trapped). Read every
        # adrp(+add) definition of it in the function: a case only matches when the computed target really is that address.
        for qi, (a, sz, mn, o) in enumerate(insns):
            p = split_ops(o)
            if mn != 'adrp' or not p or rn(p[0]) != rn(tbase_reg): continue
            t2 = imm(p[1])
            for a2, sz2, mn2, o2 in insns[qi + 1:qi + 6]:
                p2 = split_ops(o2)
                if not p2 or rn(p2[0]) != rn(tbase_reg): continue
                if mn2 == 'add' and len(p2) == 3 and rn(p2[1]) == rn(tbase_reg) and p2[2].startswith('#'): t2 += imm(p2[2])
                break
            if t2 not in tabs: tabs.append(t2)
        esz = {'ldrb': 1, 'ldrsb': 1, 'ldrh': 2, 'ldrsh': 2, 'ldrsw': 4, 'ldr': 4}[m]
        signed = m in ('ldrsb', 'ldrsh', 'ldrsw') or k.startswith('s')
        # bound: 'cmp <table index reg>, #N' anywhere earlier in the function (nearest first)
        # follow `mov idx, wN` copies (f_8df0bc: `cmp w24,#7 ... mov w8,w24 ... ldrb w10,[x28,x8]`; an unrelated
        # earlier `cmp w8,#3` gave a 4-entry table and the 5th case trapped)
        n = None; idx_regs = {rn(idx_of_table)}
        for a, sz, mn, o in reversed(insns[:li]):
            p = split_ops(o)
            if mn == 'cmp' and len(p) == 2 and p[1].startswith('#') and rn(p[0]) in idx_regs:
                n = imm(p[1]) + 1; break
            if mn == 'mov' and len(p) == 2 and rn(p[0]) in idx_regs and GPR.match(p[1]):
                idx_regs.add(rn(p[1]))
        bounded = n is not None and n <= 4096
        if not bounded: n = 1024
        fmt = {1: 'b', 2: 'h', 4: 'i'}[esz]
        if not signed: fmt = fmt.upper()

        # `udf #0` words after the function end are the compiler's unreachable-default case: tables point there for
        # impossible indices (f_95550's byte table, entry 4 -> 0x95964). Skip such entries instead of ending the table
        # there: the cases after them were lost and trapped (run222, Dm_OP_0004 cutscene: case 0x9584c).
        def udf_at(t):
            return 0 <= t <= len(self.img) - 4 and self.img[t:t + 4] == bytes(4)

        def read_table(tab):
            targets = []
            for j in range(n):
                if tab + (j + 1) * esz > len(self.img): break
                v = struct.unpack_from('<' + fmt, self.img, tab + j * esz)[0]
                if k.startswith('s') and not signed:  # sxtb on unsigned load
                    v = struct.unpack('<' + fmt.lower(), struct.pack('<' + fmt, v))[0]
                t = anchor + (v << amt)
                # out of function / misaligned: end of table. (Bounded tables used to be dropped whole here, but the
                # bound can come from an unrelated, larger cmp: br 0xb4448c's 4-entry table then lost every case.)
                if not (start <= t < end) and not (t & 3) and udf_at(t):
                    self.stats['jt_udf_entries'] += 1; continue
                if not (start <= t < end) or t & 3: break
                targets.append(t)
            if not targets: return None
            if bounded and len(targets) == n:
                # A wrongly small bound silently drops cases (they trap); keep reading while entries stay in-function.
                # Extra cases are harmless: a case only matches when the computed target really is that address.
                j = n
                while j < 1024 and tab + (j + 1) * esz <= len(self.img):
                    v = struct.unpack_from('<' + fmt, self.img, tab + j * esz)[0]
                    if k.startswith('s') and not signed: v = struct.unpack('<' + fmt.lower(), struct.pack('<' + fmt, v))[0]
                    t = anchor + (v << amt)
                    if not (start <= t < end) and not (t & 3) and udf_at(t):
                        self.stats['jt_udf_entries'] += 1; j += 1; continue
                    if not (start <= t < end) or t & 3: break
                    targets.append(t); j += 1
                if j > n: self.stats['jt_extended'] += 1
            return targets

        targets = set()
        for ti, tab in enumerate(tabs):
            got = read_table(tab)
            if got: targets.update(got)
            if got and ti: self.stats['jt_alt_base'] += 1
        if not targets: return None
        self.stats['jt_bounded' if bounded else 'jt_unbounded'] += 1
        return sorted(targets)

    # -------------------------------------------------- instructions
    def insn(self, a, m, o, jt):
        ops = split_ops(o)
        nxt = a + 4
        G = lambda off: f'(IMGBASE + {off:#x}ull)'
        h = getattr(self, 'i_' + m.replace('.', '_'), None)
        if m == '.byte':   # undefined words (clang trap 0xe7ffdefe, data in code): trap like the hardware would
            self.stats['undefined_words'] += 1
            return f'{{ LSO; rt_trap(cx, IMGBASE + {a:#x}ull, "undefined instruction"); }}'
        if m.startswith('b.'):
            return f'if ({COND[m[2:]]}) {self.tail(imm(ops[0]), a)}'
        if m == 'b':
            return self.tail(imm(ops[0]), a)
        if m == 'bl':
            return self.call(imm(ops[0]), nxt)
        if m in ('cbz', 'cbnz'):
            t = imm(ops[1])
            return f'if ({R(ops[0])} {"==" if m == "cbz" else "!="} 0) {self.tail(t, a)}'
        if m in ('tbz', 'tbnz'):
            bit = imm(ops[1]); t = imm(ops[2])
            v = R(ops[0])
            return f'if ((({v} >> {bit}) & 1) {"==" if m == "tbz" else "!="} 0) {self.tail(t, a)}'
        if m == 'ret':
            if self.fstart in self.trace: return f'{{ LSO; rt_trace(cx, IMGBASE + {self.fstart:#x}ull, 1); return; }}'
            return '{ LSO; return; }'
        if m == 'blr':
            return f'{{ u64 t_ = {R(ops[0])}; c->x[30] = IMGBASE + {nxt:#x}ull; LSO; call_ind(cx, t_); LSI; }}'
        if m == 'br':
            if jt:
                cases = ' '.join(f'case IMGBASE + {t:#x}ull: goto L_{t:x};' for t in jt)
                return f'{{ u64 t_ = {R(ops[0])}; switch (t_) {{ {cases} default: LSO; call_ind(cx, t_); return; }} }}'
            return f'{{ LSO; call_ind(cx, {R(ops[0])}); return; }}'
        if m == 'adrp' or m == 'adr':
            return W(ops[0], G(imm(ops[1])))
        if h is None:
            raise Unhandled('no handler')
        return h(a, ops)

    # ---------------- integer ALU
    def _arith(self, ops, op, flags):
        d, n_ = ops[0], ops[1]
        if d.startswith('v'):
            return self.vint(ops, {'add': '+', 'sub': '-'}[op])
        n = gsize(d) if not flags or d not in ('xzr', 'wzr') else gsize(n_)
        b = op2(ops[2:], n)
        a = R(n_) if gsize(n_) == n else f'({T(n)})({R(n_)})'
        if flags:
            f = f'adds{n}(c, {a}, {b})' if op == 'add' else f'subs{n}(c, {a}, {b})'
            return W(d, f)
        return W(d, f'{a} {"+" if op == "add" else "-"} {b}')
    def i_add(self, a, ops): return self._arith(ops, 'add', False)
    def i_sub(self, a, ops): return self._arith(ops, 'sub', False)
    def i_adds(self, a, ops): return self._arith(ops, 'add', True)
    def i_subs(self, a, ops): return self._arith(ops, 'sub', True)
    def i_cmp(self, a, ops): n = gsize(ops[0]); return f'subs{n}(c, {R(ops[0])}, {op2(ops[1:], n)});'
    def i_cmn(self, a, ops): n = gsize(ops[0]); return f'adds{n}(c, {R(ops[0])}, {op2(ops[1:], n)});'
    def i_neg(self, a, ops):
        if ops[0].startswith('v'): return self.vun(ops, lambda e, t: f'({t})(0 - {e})', signed=True)
        n = gsize(ops[0]); return W(ops[0], f'({T(n)})0 - {op2(ops[1:], n)}')
    def i_negs(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'subs{n}(c, 0, {op2(ops[1:], n)})')
    def i_adcs(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'adds{n}(c, {R(ops[1])}, {R(ops[2])}, c->c)')
    def i_sbcs(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'adds{n}(c, {R(ops[1])}, ~({T(n)})({R(ops[2])}), c->c)')
    def i_ngcs(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'adds{n}(c, 0, ~({T(n)})({R(ops[1])}), c->c)')

    def _logic(self, ops, op, inv=False, flags=False):
        d = ops[0]
        if d.startswith('v'):
            return self.vlogic(ops, op, inv)
        n = gsize(d) if not flags or d not in ('xzr', 'wzr') else gsize(ops[1])
        b = op2(ops[2:], n)
        if inv: b = f'~({T(n)})({b})'
        e = f'({T(n)})({R(ops[1])} {op} {b})'
        if flags:
            return f'{{ {T(n)} r_ = {e}; nz{n}(c, r_); c->c = 0; c->vf = 0; {W(d, "r_")} }}'
        return W(d, e)
    def i_and(self, a, ops): return self._logic(ops, '&')
    def i_orr(self, a, ops): return self._logic(ops, '|')
    def i_eor(self, a, ops): return self._logic(ops, '^')
    def i_bic(self, a, ops): return self._logic(ops, '&', inv=True)
    def i_orn(self, a, ops): return self._logic(ops, '|', inv=True)
    def i_eon(self, a, ops): return self._logic(ops, '^', inv=True)
    def i_ands(self, a, ops): return self._logic(ops, '&', flags=True)
    def i_bics(self, a, ops): return self._logic(ops, '&', inv=True, flags=True)
    def i_tst(self, a, ops):
        n = gsize(ops[0])
        return f'{{ {T(n)} r_ = ({T(n)})({R(ops[0])} & {op2(ops[1:], n)}); nz{n}(c, r_); c->c = 0; c->vf = 0; }}'
    def i_mvn(self, a, ops):
        if ops[0].startswith('v'): return self.vun(ops, lambda e, t: f'({t})~{e}')
        n = gsize(ops[0]); return W(ops[0], f'~({T(n)})({op2(ops[1:], n)})')

    def i_mov(self, a, ops):
        d, s = ops
        if is_gpr(d) and (is_gpr(s) or s.startswith('#')):
            if s.startswith('#'):
                n = gsize(d); v = imm(s) & ((1 << n) - 1)
                return W(d, f'{v:#x}ull')
            return W(d, R(s))
        if VREG.match(d) and VREG.match(s):     # orr alias
            dr, cnt, e = varr(d); sr, _, _ = varr(s)
            return f'c->v[{dr}] = {"vlo" if cnt * e == 64 else ""}(c->v[{sr}]);'
        if VELEM.match(d):                       # ins
            dr, e, i = velem(d)
            if is_gpr(s):
                return f'c->v[{dr}] = vins(c->v[{dr}], {e // 8}, {i}, {R(s)});'
            sr, e2, j = velem(s)
            return f'c->v[{dr}] = vins(c->v[{dr}], {e // 8}, {i}, vext_lane(c->v[{sr}], {e // 8}, {j}));'
        if is_gpr(d) and VELEM.match(s):         # umov
            sr, e, j = velem(s)
            return W(d, f'vext_lane(c->v[{sr}], {e // 8}, {j})')
        if FREG.match(d) and VELEM.match(s):     # dup element to scalar
            k, dr = freg(d); sr, e, j = velem(s)
            return f'c->v[{dr}] = v2(vext_lane(c->v[{sr}], {e // 8}, {j}), 0);'
        raise Unhandled('mov form')
    def i_umov(self, a, ops): return self.i_mov(a, ops)
    def i_smov(self, a, ops):
        sr, e, j = velem(ops[1])
        return W(ops[0], f'(s64)({STT[e]})vext_lane(c->v[{sr}], {e // 8}, {j})')
    def i_movz(self, a, ops):
        n = gsize(ops[0]); v = imm(ops[1])
        if len(ops) > 2: v <<= parse_mod(ops[2])[1]
        return W(ops[0], f'{v & ((1 << n) - 1):#x}ull')
    def i_movk(self, a, ops):
        n = gsize(ops[0]); v = imm(ops[1]); s = parse_mod(ops[2])[1] if len(ops) > 2 else 0
        return W(ops[0], f'(({R(ops[0])} & ~(0xffffull << {s})) | ({v:#x}ull << {s}))')

    def _shift(self, ops, kind):
        n = gsize(ops[0])
        if ops[2].startswith('#'):
            return W(ops[0], shift_expr(R(ops[1]), kind, imm(ops[2]), n))
        fn = {'lsl': 'lsl', 'lsr': 'lsr', 'asr': 'asr', 'ror': 'ror'}[kind]
        return W(ops[0], f'{fn}{n}({R(ops[1])}, (unsigned){R(ops[2])})')
    def i_lsl(self, a, ops): return self._shift(ops, 'lsl')
    def i_lsr(self, a, ops): return self._shift(ops, 'lsr')
    def i_asr(self, a, ops): return self._shift(ops, 'asr')
    def i_ror(self, a, ops): return self._shift(ops, 'ror')

    def i_bfi(self, a, ops):
        n = gsize(ops[0]); return W(ops[0], f'bfi{n}({R(ops[0])}, {R(ops[1])}, {imm(ops[2])}, {imm(ops[3])})')
    def i_bfxil(self, a, ops):
        n = gsize(ops[0]); w = imm(ops[3])
        return W(ops[0], f'bfi{n}({R(ops[0])}, ubfx{n}({R(ops[1])}, {imm(ops[2])}, {w}), 0, {w})')
    def i_ubfx(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'ubfx{n}({R(ops[1])}, {imm(ops[2])}, {imm(ops[3])})')
    def i_sbfx(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'sbfx{n}({R(ops[1])}, {imm(ops[2])}, {imm(ops[3])})')
    def i_ubfiz(self, a, ops):
        n = gsize(ops[0]); return W(ops[0], f'(ubfx{n}({R(ops[1])}, 0, {imm(ops[3])}) << {imm(ops[2])})')
    def i_sbfiz(self, a, ops):
        n = gsize(ops[0]); return W(ops[0], f'(sbfx{n}({R(ops[1])}, 0, {imm(ops[3])}) << {imm(ops[2])})')
    def i_sxtb(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})({ST(n)})(s8)({R(ops[1])})')
    def i_sxth(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})({ST(n)})(s16)({R(ops[1])})')
    def i_sxtw(self, a, ops): return W(ops[0], f'(u64)(s64)(s32)({R(ops[1])})')
    def i_uxtb(self, a, ops): return W(ops[0], f'(u8)({R(ops[1])})')
    def i_uxth(self, a, ops): return W(ops[0], f'(u16)({R(ops[1])})')
    def i_extr(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'extr{n}({R(ops[1])}, {R(ops[2])}, {imm(ops[3])})')

    def i_madd(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})({R(ops[3])} + ({T(n)})({R(ops[1])}) * ({T(n)})({R(ops[2])}))')
    def i_msub(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})({R(ops[3])} - ({T(n)})({R(ops[1])}) * ({T(n)})({R(ops[2])}))')
    def i_mul(self, a, ops):
        if ops[0].startswith('v'): return self.vint(ops, '*')
        n = gsize(ops[0]); return W(ops[0], f'({T(n)})(({T(n)})({R(ops[1])}) * ({T(n)})({R(ops[2])}))')
    def i_mneg(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})(0 - ({T(n)})({R(ops[1])}) * ({T(n)})({R(ops[2])}))')
    def i_smaddl(self, a, ops): return W(ops[0], f'{R(ops[3])} + (u64)((s64)(s32){R(ops[1])} * (s64)(s32){R(ops[2])})')
    def i_smsubl(self, a, ops): return W(ops[0], f'{R(ops[3])} - (u64)((s64)(s32){R(ops[1])} * (s64)(s32){R(ops[2])})')
    def i_umaddl(self, a, ops): return W(ops[0], f'{R(ops[3])} + (u64)(u32){R(ops[1])} * (u64)(u32){R(ops[2])}')
    def i_smull(self, a, ops): return W(ops[0], f'(u64)((s64)(s32){R(ops[1])} * (s64)(s32){R(ops[2])})')
    def i_umull(self, a, ops): return W(ops[0], f'(u64)(u32){R(ops[1])} * (u64)(u32){R(ops[2])}')
    def i_smulh(self, a, ops): return W(ops[0], f'smulh((s64){R(ops[1])}, (s64){R(ops[2])})')
    def i_umulh(self, a, ops): return W(ops[0], f'umulh({R(ops[1])}, {R(ops[2])})')
    def i_sdiv(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})sdiv{n}(({ST(n)}){R(ops[1])}, ({ST(n)}){R(ops[2])})')
    def i_udiv(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'udiv{n}({R(ops[1])}, {R(ops[2])})')

    def _csel(self, ops, f):
        n = gsize(ops[0]); cnd = COND[ops[3]]
        return W(ops[0], f'({cnd} ? ({T(n)})({R(ops[1])}) : ({T(n)})({f(f"({T(n)})({R(ops[2])})", T(n))}))')
    def i_csel(self, a, ops): return self._csel(ops, lambda e, t: e)
    def i_csinc(self, a, ops): return self._csel(ops, lambda e, t: f'{e} + 1')
    def i_csinv(self, a, ops): return self._csel(ops, lambda e, t: f'~{e}')
    def i_csneg(self, a, ops): return self._csel(ops, lambda e, t: f'({t})0 - {e}')
    def i_cset(self, a, ops): return W(ops[0], f'({COND[ops[1]]} ? 1 : 0)')
    def i_csetm(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({COND[ops[1]]} ? ~({T(n)})0 : 0)')
    def i_cinc(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({T(n)})({R(ops[1])} + ({COND[ops[2]]} ? 1 : 0))')
    def i_cinv(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({COND[ops[2]]} ? ~({T(n)})({R(ops[1])}) : ({T(n)})({R(ops[1])}))')
    def i_cneg(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'({COND[ops[2]]} ? ({T(n)})0 - ({T(n)})({R(ops[1])}) : ({T(n)})({R(ops[1])}))')
    def _ccmp(self, ops, fn):
        n = gsize(ops[0]); b = op2([ops[1]], n)
        return f'if ({COND[ops[3]]}) {fn}{n}(c, {R(ops[0])}, {b}); else setnzcv(c, {imm(ops[2])});'
    def i_ccmp(self, a, ops): return self._ccmp(ops, 'subs')
    def i_ccmn(self, a, ops): return self._ccmp(ops, 'adds')

    def i_clz(self, a, ops):
        if ops[0].startswith('v'): return self.vun(ops, lambda e, t: f'({t})clz32({e})')
        n = gsize(ops[0]); return W(ops[0], f'clz{n}({R(ops[1])})')
    def i_rbit(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'rbit{n}({R(ops[1])})')
    def i_rev(self, a, ops):
        n = gsize(ops[0]); return W(ops[0], f'__builtin_bswap{n}({R(ops[1])})')
    def i_rev16(self, a, ops): return W(ops[0], f'rev16_32({R(ops[1])})')
    def _crc(self, name):
        return lambda a, ops: W(ops[0], f'{name}({R(ops[1])}, {R(ops[2])})')
    def __getattr__(self, k):
        if k.startswith('i_crc32'):
            return self._crc(k[2:])
        raise AttributeError(k)
    def i_nop(self, a, ops): return ';'
    def i_yield(self, a, ops): return ';'
    def i_prfm(self, a, ops): return ';'
    def i_xpaclri(self, a, ops): return ';'     # strip PAC from lr: no pointer auth here
    def i_prfum(self, a, ops): return ';'
    def i_dmb(self, a, ops): return 'dmb_full();' if ops[0] in ('ish', 'sy') else 'dmb_light();'
    def i_clrex(self, a, ops): return 'clrex(c);'
    def i_mrs(self, a, ops):
        if ops[1] == 'tpidr_el0': return W(ops[0], 'c->tpidr')
        if ops[1] == 'tpidrro_el0': return W(ops[0], 'rt_tpidrro(cx)')     # Horizon thread-local region (sdk only)
        if ops[1] == 'cntpct_el0': return W(ops[0], 'rt_cntpct()')         # 19.2 MHz system counter (sdk only)
        if ops[1] == 'fpsr': return W(ops[0], 'c->fpsr')
        if ops[1] == 'fpcr': return W(ops[0], 'rt_read_fpcr(cx)')
        raise Unhandled('mrs ' + ops[1])
    def i_msr(self, a, ops):
        if ops[0] == 'fpcr': return f'rt_write_fpcr(cx, {R(ops[1])});'
        if ops[0] == 'tpidr_el0': return f'c->tpidr = {R(ops[1])};'
        if ops[0] == 'fpsr': return f'c->fpsr = (u32){R(ops[1])};'
        raise Unhandled('msr ' + ops[0])

    # ---------------- loads / stores
    def _ld_scalar(self, rt, mn):
        """returns (load_fn(ea)->expr, assign(expr)->stmt) for register rt and mnemonic"""
        base = mn.replace('ldur', 'ldr')
        if FREG.match(rt):
            k, r = freg(rt)
            ld = {'b': lambda e: f'vU8(ld8({e}))', 'h': lambda e: f'vU16(ld16({e}))', 's': lambda e: f'vU32(ld32({e}))',
                  'd': lambda e: f'vU64(ld64({e}))', 'q': lambda e: f'ld128({e})'}[k]
            return ld, (lambda e: f'c->v[{r}] = {e};'), ESZ[k] // 8, 'V128'
        n = gsize(rt)
        if base == 'ldr': fn, sz = ('ld64' if n == 64 else 'ld32'), n // 8; cast = ''
        elif base == 'ldrb': fn, sz, cast = 'ld8', 1, ''
        elif base == 'ldrh': fn, sz, cast = 'ld16', 2, ''
        elif base == 'ldrsb': fn, sz, cast = 'ld8', 1, '(s8)'
        elif base == 'ldrsh': fn, sz, cast = 'ld16', 2, '(s16)'
        elif base == 'ldrsw': fn, sz, cast = 'ld32', 4, '(s32)'
        else: raise Unhandled('ld ' + mn)
        if cast:
            ld = lambda e: f'({T(n)})({ST(n)}){cast}{fn}({e})'
        else:
            ld = lambda e: f'{fn}({e})'
        return ld, (lambda e: W(rt, e)), sz, T(n)

    def _st_scalar(self, rt, mn):
        base = mn.replace('stur', 'str')
        if FREG.match(rt):
            k, r = freg(rt)
            st = {'b': lambda e: f'st8({e}, c->v[{r}].b[0]);', 'h': lambda e: f'st16({e}, c->v[{r}].h[0]);',
                  's': lambda e: f'st32({e}, c->v[{r}].s[0]);', 'd': lambda e: f'st64({e}, c->v[{r}].d[0]);',
                  'q': lambda e: f'st128({e}, c->v[{r}]);'}[k]
            return st, ESZ[k] // 8
        n = gsize(rt)
        if base == 'str': return (lambda e: f'st{n}({e}, {R(rt)});'), n // 8
        if base == 'strb': return (lambda e: f'st8({e}, (u8){R(rt)});'), 1
        if base == 'strh': return (lambda e: f'st16({e}, (u16){R(rt)});'), 2
        raise Unhandled('st ' + mn)

    def _load(self, a, ops, mn):
        if ops[1].startswith('#'):      # literal
            ld, asg, sz, t = self._ld_scalar(ops[0], mn)
            return asg(ld(f'(IMGBASE + {imm(ops[1]):#x}ull)'))
        ea, post = mem_ea(ops[1:])
        ld, asg, sz, t = self._ld_scalar(ops[0], mn)
        return f'{{ u64 ea = {ea}; auto v_ = {ld("ea")}; {post} {asg("v_")} }}'
    def _store(self, a, ops, mn):
        ea, post = mem_ea(ops[1:])
        st, sz = self._st_scalar(ops[0], mn)
        return f'{{ u64 ea = {ea}; {st("ea")} {post} }}'
    for _m in ('ldr', 'ldrb', 'ldrh', 'ldrsb', 'ldrsh', 'ldrsw', 'ldur', 'ldurb', 'ldurh', 'ldursb', 'ldursh', 'ldursw'):
        exec(f'def i_{_m}(self, a, ops): return self._load(a, ops, "{_m}")')
    for _m in ('str', 'strb', 'strh', 'stur', 'sturb', 'sturh'):
        exec(f'def i_{_m}(self, a, ops): return self._store(a, ops, "{_m}")')

    def i_ldp(self, a, ops, mn='ldp'):
        ea, post = mem_ea(ops[2:])
        if mn == 'ldpsw':
            l1 = l2 = lambda e: f'(u64)(s64)(s32)ld32({e})'; a1 = lambda e: W(ops[0], e); a2 = lambda e: W(ops[1], e); sz = 4
        else:
            l1, a1, sz, _ = self._ld_scalar(ops[0], 'ldr')
            l2, a2, _, _ = self._ld_scalar(ops[1], 'ldr')
        return f'{{ u64 ea = {ea}; auto v1_ = {l1("ea")}; auto v2_ = {l2(f"ea + {sz}")}; {post} {a1("v1_")} {a2("v2_")} }}'
    def i_ldpsw(self, a, ops): return self.i_ldp(a, ops, 'ldpsw')
    def i_stp(self, a, ops):
        ea, post = mem_ea(ops[2:])
        s1, sz = self._st_scalar(ops[0], 'str'); s2, _ = self._st_scalar(ops[1], 'str')
        return f'{{ u64 ea = {ea}; {s1("ea")} {s2(f"ea + {sz}")} {post} }}'

    def _ldx(self, ops, fn8, fn32, fn64, rt_idx=0):
        rt = ops[rt_idx]; ea = R(MEM.match(ops[rt_idx + 1]).group(1))
        return rt, ea
    def i_ldxr(self, a, ops):
        rt = ops[0]; ea = R(MEM.match(ops[1]).group(1)); n = gsize(rt)
        return W(rt, f'ldx{n}(c, {ea})')
    i_ldaxr = i_ldxr
    def i_ldxrb(self, a, ops): return W(ops[0], f'ldx8(c, {R(MEM.match(ops[1]).group(1))})')
    i_ldaxrb = i_ldxrb
    def i_stxr(self, a, ops):
        ws, rt = ops[0], ops[1]; ea = R(MEM.match(ops[2]).group(1)); n = gsize(rt)
        return W(ws, f'stx{n}(c, {ea}, {R(rt)})')
    i_stlxr = i_stxr
    def i_stxrb(self, a, ops): return W(ops[0], f'stx8(c, {R(MEM.match(ops[2]).group(1))}, (u8){R(ops[1])})')
    i_stlxrb = i_stxrb
    def i_ldar(self, a, ops): n = gsize(ops[0]); return W(ops[0], f'lda{n}({R(MEM.match(ops[1]).group(1))})')
    def i_ldarb(self, a, ops): return W(ops[0], f'lda8({R(MEM.match(ops[1]).group(1))})')
    def i_stlr(self, a, ops): n = gsize(ops[0]); return f'stl{n}({R(MEM.match(ops[1]).group(1))}, {R(ops[0])});'
    def i_stlrb(self, a, ops): return f'stl8({R(MEM.match(ops[1]).group(1))}, (u8){R(ops[0])});'

    # ---------------- SIMD structure loads/stores
    def _vlist(self, s):
        """'{v0.4s, v1.4s}' or '{v1.s}[2]' -> (regs, cnt, esz, lane)"""
        m = re.match(r'^\{(.+?)\}(?:\[(\w+)\])?$', s)
        if not m: raise Unhandled('vlist ' + s)
        items = [x.strip() for x in m.group(1).split(',')]
        lane = int(m.group(2), 0) if m.group(2) else None
        regs = []
        cnt = esz = None
        for it in items:
            mm = re.match(r'^v(\d+)\.(\d*)([bhsdq])$', it)
            if not mm: raise Unhandled('vlist item ' + it)
            regs.append(int(mm.group(1))); cnt = int(mm.group(2) or 1); esz = ESZ[mm.group(3)]
        return regs, cnt, esz, lane

    def _vmem(self, ops, total_bytes):
        m = MEM.match(ops[0])
        base = R(m.group(1))
        post = ''
        if len(ops) > 1:
            p = ops[1]
            inc = f'{imm(p)}ll' if p.startswith('#') else R(p)
            post = W(m.group(1), f'{base} + (u64)({inc})')
        return base, post

    def i_ld1(self, a, ops):
        regs, cnt, esz, lane = self._vlist(ops[0])
        base, post = self._vmem(ops[1:], 0)
        if lane is not None:
            r = regs[0]; b = esz // 8
            return f'{{ u64 ea = {base}; c->v[{r}] = vins(c->v[{r}], {b}, {lane}, ld{esz}(ea)); {post} }}'
        nb = cnt * esz // 8
        parts = []
        for k, r in enumerate(regs):
            if nb == 16: parts.append(f'c->v[{r}] = ld128(ea + {16 * k});')
            elif nb == 8: parts.append(f'c->v[{r}] = vU64(ld64(ea + {8 * k}));')
            else: raise Unhandled('ld1 size')
        return f'{{ u64 ea = {base}; {" ".join(parts)} {post} }}'
    def i_st1(self, a, ops):
        regs, cnt, esz, lane = self._vlist(ops[0])
        base, post = self._vmem(ops[1:], 0)
        if lane is not None:
            r = regs[0]; b = esz // 8
            return f'{{ u64 ea = {base}; st{esz}(ea, ({UT[esz]})vext_lane(c->v[{r}], {b}, {lane})); {post} }}'
        nb = cnt * esz // 8
        parts = []
        for k, r in enumerate(regs):
            if nb == 16: parts.append(f'st128(ea + {16 * k}, c->v[{r}]);')
            elif nb == 8: parts.append(f'st64(ea + {8 * k}, c->v[{r}].d[0]);')
            else: raise Unhandled('st1 size')
        return f'{{ u64 ea = {base}; {" ".join(parts)} {post} }}'
    def i_ld1r(self, a, ops):
        regs, cnt, esz, lane = self._vlist(ops[0])
        base, post = self._vmem(ops[1:], 0)
        return f'{{ u64 ea = {base}; c->v[{regs[0]}] = vdup({esz // 8}, {cnt * esz // 8}, ld{esz}(ea)); {post} }}'
    def _ldn(self, ops, store):
        regs, cnt, esz, lane = self._vlist(ops[0])
        base, post = self._vmem(ops[1:], 0)
        n = len(regs); b = esz // 8
        if lane is not None:
            if store:
                return f'{{ u64 ea = {base}; ' + ' '.join(
                    f'st{esz}(ea + {k * b}, ({UT[esz]})vext_lane(c->v[{r}], {b}, {lane}));' for k, r in enumerate(regs)) + f' {post} }}'
            return f'{{ u64 ea = {base}; ' + ' '.join(f'{UT[esz]} t{k}_ = ld{esz}(ea + {k * b});' for k in range(n)) + ' ' + ' '.join(
                f'c->v[{r}] = vins(c->v[{r}], {b}, {lane}, t{k}_);' for k, r in enumerate(regs)) + f' {post} }}'
        f = UF[esz]
        if store:
            body = f'LANES(0, {cnt}, ' + ' '.join(
                f'st{esz}(ea + (i * {n} + {k}) * {b}, c->v[{r}].{f}[i]);' for k, r in enumerate(regs)) + ');'
            return f'{{ u64 ea = {base}; {body} {post} }}'
        decl = ' '.join(f'V128 t{k}_ = vzero();' for k in range(n))
        body = f'LANES(0, {cnt}, ' + ' '.join(
            f't{k}_.{f}[i] = ld{esz}(ea + (i * {n} + {k}) * {b});' for k in range(n)) + ');'
        asg = ' '.join(f'c->v[{r}] = t{k}_;' for k, r in enumerate(regs))
        return f'{{ u64 ea = {base}; {decl} {body} {asg} {post} }}'
    def i_ld2(self, a, ops): return self._ldn(ops, False)
    def i_ld3(self, a, ops): return self._ldn(ops, False)
    def i_ld4(self, a, ops): return self._ldn(ops, False)
    def i_st2(self, a, ops): return self._ldn(ops, True)
    def i_st3(self, a, ops): return self._ldn(ops, True)
    def i_st4(self, a, ops): return self._ldn(ops, True)
    def _ldnr(self, ops):
        regs, cnt, esz, lane = self._vlist(ops[0])
        base, post = self._vmem(ops[1:], 0)
        b = esz // 8
        return f'{{ u64 ea = {base}; ' + ' '.join(f'{UT[esz]} t{k}_ = ld{esz}(ea + {k * b});' for k in range(len(regs))) + ' ' + ' '.join(
            f'c->v[{r}] = vdup({b}, {cnt * b}, t{k}_);' for k, r in enumerate(regs)) + f' {post} }}'
    def i_ld2r(self, a, ops): return self._ldnr(ops)
    def i_ld3r(self, a, ops): return self._ldnr(ops)
    def i_ld4r(self, a, ops): return self._ldnr(ops)

    # ---------------- SIMD integer
    def vbin(self, ops, fn, signed=False):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        f = SF[esz] if signed else UF[esz]; t = STT[esz] if signed else UT[esz]
        if ops[2].startswith('#'):
            bexp = str(imm(ops[2]))
        elif VELEM.match(ops[2]):
            m_, e2, j = velem(ops[2]); bexp = f'c->v[{m_}].{f}[{j}]'
        else:
            m_, _, _ = varr(ops[2]); bexp = f'c->v[{m_}].{f}[i]'
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{f}[i]', bexp, t), f)
    def vun(self, ops, fn, signed=False):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        f = SF[esz] if signed else UF[esz]; t = STT[esz] if signed else UT[esz]
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{f}[i]', t), f)
    def vint(self, ops, op):
        return self.vbin(ops, lambda x, y, t: f'({t})({x} {op} {y})')
    def vlogic(self, ops, op, inv):
        d, cnt, esz = varr(ops[0])
        if ops[1].startswith('#'):   # bic/orr immediate (vector, imm, lsl)
            v = imm(ops[1]); s = parse_mod(ops[2])[1] if len(ops) > 2 else 0
            v <<= s
            f = UF[esz]; t = UT[esz]
            e = f'({t})~({t}){v:#x}' if inv else f'({t}){v:#x}'
            return lanes_loop(d, cnt, esz, f'({t})(c->v[{d}].{f}[i] {op} {e})', f)
        nb = cnt * esz // 8
        n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2])
        b = f'~c->v[{m_}].d[i]' if inv else f'c->v[{m_}].d[i]'
        return lanes_loop(d, nb // 8, 64, f'c->v[{n}].d[i] {op} {b}', 'd')
    def i_bsl(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2]); nb = cnt * esz // 64
        return lanes_loop(d, nb, 64, f'(c->v[{d}].d[i] & c->v[{n}].d[i]) | (~c->v[{d}].d[i] & c->v[{m_}].d[i])', 'd')
    def i_bit(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2]); nb = cnt * esz // 64
        return lanes_loop(d, nb, 64, f'(c->v[{d}].d[i] & ~c->v[{m_}].d[i]) | (c->v[{n}].d[i] & c->v[{m_}].d[i])', 'd')
    def i_bif(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2]); nb = cnt * esz // 64
        return lanes_loop(d, nb, 64, f'(c->v[{d}].d[i] & c->v[{m_}].d[i]) | (c->v[{n}].d[i] & ~c->v[{m_}].d[i])', 'd')
    def _vcmp(self, ops, op, signed):
        return self.vbin(ops, lambda x, y, t: f'({x} {op} ({t}){y}) ? ({t})~0ull : 0', signed=signed)
    def i_cmeq(self, a, ops): return self._vcmp(ops, '==', False)
    def i_cmgt(self, a, ops): return self._vcmp(ops, '>', True)
    def i_cmhi(self, a, ops): return self._vcmp(ops, '>', False)
    def i_cmtst(self, a, ops): return self.vbin(ops, lambda x, y, t: f'(({x} & {y}) != 0) ? ({t})~0ull : 0')
    def i_smax(self, a, ops): return self.vbin(ops, lambda x, y, t: f'({x} > {y} ? {x} : {y})', signed=True)
    def i_smin(self, a, ops): return self.vbin(ops, lambda x, y, t: f'({x} < {y} ? {x} : {y})', signed=True)
    def i_mla(self, a, ops):
        d, cnt, esz = varr(ops[0])
        return self.vbin(ops, lambda x, y, t: f'({t})(c->v[{d}].{UF[esz]}[i] + {x} * {y})')
    def i_mls(self, a, ops):
        d, cnt, esz = varr(ops[0])
        return self.vbin(ops, lambda x, y, t: f'({t})(c->v[{d}].{UF[esz]}[i] - {x} * {y})')
    def i_sqadd(self, a, ops): return self.vbin(ops, lambda x, y, t: f'satT<{t}>((s64){x} + (s64){y})', signed=True)
    def _pairwise(self, ops, fn, field_kind):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2])
        f = {'u': UF, 's': SF, 'f': FF}[field_kind][esz]
        h = cnt // 2
        body = (f'(i < {h} ? {fn(f"c->v[{n}].{f}[2*i]", f"c->v[{n}].{f}[2*i+1]")} : '
                f'{fn(f"c->v[{m_}].{f}[2*(i-{h})]", f"c->v[{m_}].{f}[2*(i-{h})+1]")})')
        return lanes_loop(d, cnt, esz, body, f)
    def i_addp(self, a, ops): d, cnt, esz = varr(ops[0]); return self._pairwise(ops, lambda x, y: f'({UT[esz]})({x} + {y})', 'u')
    def i_sminp(self, a, ops): return self._pairwise(ops, lambda x, y: f'({x} < {y} ? {x} : {y})', 's')
    def i_faddp(self, a, ops): return self._pairwise(ops, lambda x, y: f'faddA({x}, {y})', 'f')
    def i_fmaxp(self, a, ops): return self._pairwise(ops, lambda x, y: f'fmaxA({x}, {y})', 'f')
    def i_fminp(self, a, ops): return self._pairwise(ops, lambda x, y: f'fminA({x}, {y})', 'f')
    def _reduce(self, ops, fn, field_kind, outw):
        k, d = freg(ops[0]); n, cnt, esz = varr(ops[1])
        f = {'u': UF, 's': SF, 'f': FF}[field_kind][esz]
        t = {'u': UT, 's': STT, 'f': FT}[field_kind][esz]
        return (f'{{ {t} acc_ = c->v[{n}].{f}[0]; LANES(1, {cnt}, acc_ = {fn("acc_", f"c->v[{n}].{f}[i]", t)}); '
                f'V128 r_ = vzero(); r_.{f}[0] = acc_; c->v[{d}] = r_; }}')
    def i_addv(self, a, ops): return self._reduce(ops, lambda x, y, t: f'({t})({x} + {y})', 'u', 0)
    def i_sminv(self, a, ops): return self._reduce(ops, lambda x, y, t: f'({x} < {y} ? {x} : {y})', 's', 0)
    def i_fmaxnmv(self, a, ops):  # ARM reduces pairwise tree for 4s: max(max(a0,a1),max(a2,a3))
        k, d = freg(ops[0]); n, cnt, esz = varr(ops[1])
        return f'c->v[{d}] = vS(fmaxnmA(fmaxnmA(c->v[{n}].f[0], c->v[{n}].f[1]), fmaxnmA(c->v[{n}].f[2], c->v[{n}].f[3])));'
    def i_fminnmv(self, a, ops):
        k, d = freg(ops[0]); n, cnt, esz = varr(ops[1])
        return f'c->v[{d}] = vS(fminnmA(fminnmA(c->v[{n}].f[0], c->v[{n}].f[1]), fminnmA(c->v[{n}].f[2], c->v[{n}].f[3])));'
    def i_cnt(self, a, ops): return self.vun(ops, lambda e, t: f'({t})__builtin_popcount({e})')
    def _vshift_imm(self, ops, fn, signed):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); s = imm(ops[2])
        f = SF[esz] if signed else UF[esz]; t = STT[esz] if signed else UT[esz]
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{f}[i]', s, t, f, d), f)
    def i_shl(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})({x} << {s})', False)
    def i_sshr(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})({x} >> {s})', True)
    def i_ushr(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})({x} >> {s})', False)
    def i_ssra(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})(c->v[{d}].{f}[i] + ({x} >> {s}))', True)
    def i_srshr(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})(((s64){x} + (1ll << ({s}-1))) >> {s})', True)
    def i_srsra(self, a, ops): return self._vshift_imm(ops, lambda x, s, t, f, d: f'({t})(c->v[{d}].{f}[i] + (((s64){x} + (1ll << ({s}-1))) >> {s}))', True)
    def _vshift_reg(self, ops, signed):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2])
        f = SF[esz] if signed else UF[esz]; t = STT[esz] if signed else UT[esz]
        sh = f'(s8)c->v[{m_}].{UF[esz]}[i]'
        x = f'c->v[{n}].{f}[i]'
        # positive: left shift; negative: right shift (arith for signed); out of range -> 0 / sign
        if signed:
            body = (f'({sh} >= 0 ? ({sh} >= {esz} ? ({t})0 : ({t})((u64){x} << {sh})) : '
                    f'(-{sh} >= {esz} ? ({t})({x} < 0 ? -1 : 0) : ({t})({x} >> -{sh})))')
        else:
            body = (f'({sh} >= 0 ? ({sh} >= {esz} ? ({t})0 : ({t})({x} << {sh})) : '
                    f'(-{sh} >= {esz} ? ({t})0 : ({t})({x} >> -{sh})))')
        return lanes_loop(d, cnt, esz, body, f)
    def i_sshl(self, a, ops): return self._vshift_reg(ops, True)
    def i_ushl(self, a, ops): return self._vshift_reg(ops, False)
    def _widen(self, ops, signed, part2, shift):
        d, cnt, esz = varr(ops[0]); n, ncnt, nesz = varr(ops[1])
        s = imm(ops[2]) if shift else 0
        fs = SF[nesz] if signed else UF[nesz]; fd = SF[esz] if signed else UF[esz]
        off = cnt if part2 else 0
        return lanes_loop(d, cnt, esz, f'({STT[esz] if signed else UT[esz]})(({STT[esz] if signed else UT[esz]})c->v[{n}].{fs}[i + {off}] << {s})', fd)
    def i_sshll(self, a, ops): return self._widen(ops, True, False, True)
    def i_sshll2(self, a, ops): return self._widen(ops, True, True, True)
    def i_ushll(self, a, ops): return self._widen(ops, False, False, True)
    def i_ushll2(self, a, ops): return self._widen(ops, False, True, True)
    def i_shll(self, a, ops): return self._widen(ops, False, False, True)
    def i_uaddw(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, mesz = varr(ops[2])
        return lanes_loop(d, cnt, esz, f'({UT[esz]})(c->v[{n}].{UF[esz]}[i] + c->v[{m_}].{UF[mesz]}[i])', UF[esz])
    def _narrow(self, ops, fn, part2, signed_src=False):
        d, cnt, esz = varr(ops[0]); n, ncnt, nesz = varr(ops[1])
        fs = SF[nesz] if signed_src else UF[nesz]
        if part2:
            # write upper half, keep lower
            return (f'{{ V128 r_ = c->v[{d}]; LANES(0, {ncnt}, r_.{UF[esz]}[i + {ncnt}] = {fn(f"c->v[{n}].{fs}[i]")}); c->v[{d}] = r_; }}')
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{fs}[i]'), UF[esz])
    def i_xtn(self, a, ops): d, cnt, esz = varr(ops[0]); return self._narrow(ops, lambda x: f'({UT[esz]}){x}', False)
    def i_sqxtn(self, a, ops): d, cnt, esz = varr(ops[0]); return self._narrow(ops, lambda x: f'({UT[esz]})satT<{STT[esz]}>({x})', False, True)
    def i_sqxtn2(self, a, ops): d, cnt, esz = varr(ops[0]); return self._narrow(ops, lambda x: f'({UT[esz]})satT<{STT[esz]}>({x})', True, True)
    def i_sqxtun(self, a, ops): d, cnt, esz = varr(ops[0]); return self._narrow(ops, lambda x: f'satT<{UT[esz]}>({x})', False, True)
    def i_sqxtun2(self, a, ops): d, cnt, esz = varr(ops[0]); return self._narrow(ops, lambda x: f'satT<{UT[esz]}>({x})', True, True)
    def i_shrn(self, a, ops):
        d, cnt, esz = varr(ops[0]); s = imm(ops[2])
        return self._narrow(ops[:2], lambda x: f'({UT[esz]})({x} >> {s})', False)
    def i_rev64(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        return f'c->v[{d}] = vrev64<{UT[esz]}>(c->v[{n}], {cnt * esz // 8});'
    def i_ext(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2])
        return f'c->v[{d}] = vext(c->v[{n}], c->v[{m_}], {imm(ops[3])}, {cnt * esz // 8});'
    def i_tbl(self, a, ops):
        d, cnt, esz = varr(ops[0]); regs, _, _, _ = self._vlist(ops[1]); m_, _, _ = varr(ops[2])
        tabs = ', '.join(f'c->v[{r}]' for r in regs)
        return f'{{ V128 t_[{len(regs)}] = {{ {tabs} }}; c->v[{d}] = vtbl(t_, {len(regs)}, c->v[{m_}], {cnt * esz // 8}); }}'
    def _perm(self, ops, fn, part):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); m_, _, _ = varr(ops[2])
        return f'c->v[{d}] = {fn}<{UT[esz]}>(c->v[{n}], c->v[{m_}], {part}, {cnt * esz // 8});'
    def i_zip1(self, a, ops): return self._perm(ops, 'vzip', 0)
    def i_zip2(self, a, ops): return self._perm(ops, 'vzip', 1)
    def i_uzp1(self, a, ops): return self._perm(ops, 'vuzp', 0)
    def i_uzp2(self, a, ops): return self._perm(ops, 'vuzp', 1)
    def i_trn1(self, a, ops): return self._perm(ops, 'vtrn', 0)
    def i_trn2(self, a, ops): return self._perm(ops, 'vtrn', 1)
    def i_dup(self, a, ops):
        d, cnt, esz = varr(ops[0]); nb = cnt * esz // 8
        if VELEM.match(ops[1]):
            n, e2, j = velem(ops[1])
            return f'c->v[{d}] = vdup({esz // 8}, {nb}, vext_lane(c->v[{n}], {esz // 8}, {j}));'
        return f'c->v[{d}] = vdup({esz // 8}, {nb}, {R(ops[1])});'
    def _movi_val(self, ops, esz):
        v = imm(ops[1])
        if len(ops) > 2:
            k, s = parse_mod(ops[2])
            v = (v << s) | ((1 << s) - 1) if k == 'msl' else v << s
        return v
    def i_movi(self, a, ops):
        if FREG.match(ops[0]):  # movi Dd, #imm64
            k, d = freg(ops[0]); return f'c->v[{d}] = vU64({imm(ops[1]):#x}ull);'
        d, cnt, esz = varr(ops[0])
        v = self._movi_val(ops, esz) & ((1 << esz) - 1)
        return f'c->v[{d}] = vdup({esz // 8}, {cnt * esz // 8}, {v:#x}ull);'
    def i_mvni(self, a, ops):
        d, cnt, esz = varr(ops[0])
        v = (~self._movi_val(ops, esz)) & ((1 << esz) - 1)
        return f'c->v[{d}] = vdup({esz // 8}, {cnt * esz // 8}, {v:#x}ull);'

    # ---------------- floating point (scalar + vector)
    def _fbin(self, ops, fn, fast=None, nat=None, vop=None):
        # fast: scalar form (rt.h fast forms); nat: native lane expression for lanes_fp (fn stays the exact path);
        # vop: rt.h vfop<OP, CNT, F> (one SIMD op + one NaN test, exact lanes out of line) for the vector forms
        if FREG.match(ops[0]):
            x, w = fscal(ops[1])
            if VELEM.match(ops[2]):
                m_, e, j = velem(ops[2]); y = f'c->v[{m_}].{FF[e]}[{j}]'
            else:
                y, _ = fscal(ops[2])
            return fwrite(ops[0], (fast or fn)(x, y))
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); f = FF[esz]
        if vop is not None and cnt * esz in (64, 128):
            if VELEM.match(ops[2]):
                m_, e, j = velem(ops[2]); b = f'vbcast<{FT[esz]}>(c->v[{m_}].{f}[{j}])'
            elif ops[2].startswith('#'):
                b = 'vzero()'
            else:
                m_, _, _ = varr(ops[2]); b = f'c->v[{m_}]'
            return f'c->v[{d}] = vfop<{vop}, {cnt}, {FT[esz]}>(c->v[{n}], {b}, c->v[{d}]);'
        if VELEM.match(ops[2]):
            m_, e, j = velem(ops[2]); y = f'c->v[{m_}].{f}[{j}]'
        elif ops[2].startswith('#'):
            y = f'({FT[esz]})0'
        else:
            m_, _, _ = varr(ops[2]); y = f'c->v[{m_}].{f}[i]'
        if nat:
            return lanes_fp(d, cnt, f, fn(f'c->v[{n}].{f}[i]', y), nat(f'c->v[{n}].{f}[i]', y))
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{f}[i]', y), f)
    def i_fadd(self, a, ops): return self._fbin(ops, lambda x, y: f'faddA({x}, {y})', lambda x, y: f'faddS({x}, {y})', lambda x, y: f'({x} + {y})', 0)
    def i_fsub(self, a, ops): return self._fbin(ops, lambda x, y: f'fsubA({x}, {y})', lambda x, y: f'fsubS({x}, {y})', lambda x, y: f'({x} - {y})', 1)
    def i_fmul(self, a, ops): return self._fbin(ops, lambda x, y: f'fmulA({x}, {y})', lambda x, y: f'fmulS({x}, {y})', lambda x, y: f'({x} * {y})', 2)
    def i_fdiv(self, a, ops): return self._fbin(ops, lambda x, y: f'fdivA({x}, {y})', lambda x, y: f'fdivS({x}, {y})', lambda x, y: f'({x} / {y})', 3)
    def i_fnmul(self, a, ops): return self._fbin(ops, lambda x, y: f'fnmul({x}, {y})', lambda x, y: f'fnmulS({x}, {y})')
    def i_fmax(self, a, ops): return self._fbin(ops, lambda x, y: f'fmaxA({x}, {y})')
    def i_fmin(self, a, ops): return self._fbin(ops, lambda x, y: f'fminA({x}, {y})')
    def i_fmaxnm(self, a, ops): return self._fbin(ops, lambda x, y: f'fmaxnmA({x}, {y})')
    def i_fminnm(self, a, ops): return self._fbin(ops, lambda x, y: f'fminnmA({x}, {y})')
    def i_fabd(self, a, ops): return self._fbin(ops, lambda x, y: f'fabdA({x}, {y})')
    def i_frecps(self, a, ops): return self._fbin(ops, lambda x, y: f'frecps_f32({x}, {y})')
    def i_frsqrts(self, a, ops): return self._fbin(ops, lambda x, y: f'frsqrts_f32({x}, {y})')
    def _fcmpv(self, ops, op, absval=False):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); f = FF[esz]; t = UT[esz]
        if ops[2].startswith('#'): y = '0'
        else: m_, _, _ = varr(ops[2]); y = f'c->v[{m_}].{f}[i]'
        x = f'c->v[{n}].{f}[i]'
        if absval: x, y = f'std::fabs({x})', f'std::fabs({y})'
        return lanes_loop(d, cnt, esz, f'({x} {op} {y}) ? ({t})~0ull : ({t})0', UF[esz])
    def i_fcmeq(self, a, ops): return self._fcmpv(ops, '==')
    def i_fcmge(self, a, ops): return self._fcmpv(ops, '>=')
    def i_fcmgt(self, a, ops): return self._fcmpv(ops, '>')
    def i_fcmle(self, a, ops): return self._fcmpv(ops, '<=')
    def i_fcmlt(self, a, ops): return self._fcmpv(ops, '<')
    def i_facge(self, a, ops): return self._fcmpv(ops, '>=', True)
    def _fun(self, ops, fn):
        if FREG.match(ops[0]):
            x, w = fscal(ops[1]); return fwrite(ops[0], fn(x))
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1]); f = FF[esz]
        return lanes_loop(d, cnt, esz, fn(f'c->v[{n}].{f}[i]'), f)
    def i_fabs(self, a, ops): return self._fun(ops, lambda x: f'std::fabs({x})')
    def i_fneg(self, a, ops): return self._fun(ops, lambda x: f'-({x})')
    def i_fsqrt(self, a, ops): return self._fun(ops, lambda x: f'fsqrtA({x})')
    def i_frecpe(self, a, ops): return self._fun(ops, lambda x: f'frecpe_f32({x})')
    def i_frsqrte(self, a, ops): return self._fun(ops, lambda x: f'frsqrte_f32({x})')
    def i_frinta(self, a, ops): return self._fun(ops, lambda x: f'frintA({x}, 4)')
    def i_frinti(self, a, ops): return self._fun(ops, lambda x: f'frintA({x}, 0)')
    def i_frintm(self, a, ops): return self._fun(ops, lambda x: f'frintA({x}, 2)')
    def i_frintp(self, a, ops): return self._fun(ops, lambda x: f'frintA({x}, 1)')
    def i_fmla(self, a, ops):
        d, cnt, esz = varr(ops[0]); f = FF[esz]
        return self._fbin(ops, lambda x, y: f'fmlaA(c->v[{d}].{f}[i], {x}, {y})',
                          nat=lambda x, y: f'std::fma({x}, {y}, c->v[{d}].{f}[i])', vop=4)
    def i_fmls(self, a, ops):
        d, cnt, esz = varr(ops[0]); f = FF[esz]
        return self._fbin(ops, lambda x, y: f'fmlaA(c->v[{d}].{f}[i], -({x}), {y})',
                          nat=lambda x, y: f'std::fma(-({x}), {y}, c->v[{d}].{f}[i])', vop=5)
    def _fma4(self, ops, neg_prod, neg_acc):
        x, w = fscal(ops[1]); y, _ = fscal(ops[2]); acc, _ = fscal(ops[3])
        if neg_prod: x = f'-({x})'
        if neg_acc: acc = f'-({acc})'
        return fwrite(ops[0], f'fmlaS({acc}, {x}, {y})')
    def i_fmadd(self, a, ops): return self._fma4(ops, False, False)
    def i_fmsub(self, a, ops): return self._fma4(ops, True, False)
    def i_fnmadd(self, a, ops): return self._fma4(ops, True, True)
    def i_fnmsub(self, a, ops): return self._fma4(ops, False, True)
    def i_fcmp(self, a, ops):
        x, w = fscal(ops[0])
        y = f'({FT[w]})0' if ops[1].startswith('#') else fscal(ops[1])[0]
        return f'fcmpA(c, {x}, {y});'
    def i_fccmp(self, a, ops):
        x, w = fscal(ops[0]); y, _ = fscal(ops[1])
        return f'if ({COND[ops[3]]}) fcmpA(c, {x}, {y}); else setnzcv(c, {imm(ops[2])});'
    def i_fcsel(self, a, ops):
        k, d = freg(ops[0]); _, n = freg(ops[1]); _, m_ = freg(ops[2])
        f = 'vU32(c->v[{}].s[0])' if k == 's' else 'vU64(c->v[{}].d[0])'
        return f'c->v[{d}] = {COND[ops[3]]} ? {f.format(n)} : {f.format(m_)};'
    def i_fcvt(self, a, ops):
        kd, d = freg(ops[0]); ks, s = freg(ops[1])
        src = {'s': f'c->v[{s}].f[0]', 'd': f'c->v[{s}].fd[0]', 'h': f'f16tof32(c->v[{s}].h[0])'}[ks]
        if kd == 'd': return f'c->v[{d}] = vD(fixnan1((double){src}, (double){src}));'
        if kd == 's': return f'c->v[{d}] = vS(fixnan1((float){src}, (float){src}));'
        if kd == 'h': return f'c->v[{d}] = vU16(f32tof16({src}));'
        raise Unhandled('fcvt')
    def i_fcvtl(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        return lanes_loop(d, cnt, esz, f'f16tof32(c->v[{n}].h[i])', 'f')
    def i_fcvtn(self, a, ops):
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        return lanes_loop(d, cnt, esz, f'f32tof16(c->v[{n}].f[i])', 'h')
    def _fcvt_int(self, ops, signed, mode):
        if is_gpr(ops[0]):
            n = gsize(ops[0]); x, w = fscal(ops[1])
            if len(ops) > 2: x = f'std::ldexp({x}, {imm(ops[2])})'
            it = ST(n) if signed else T(n)
            fn = f'fcvtzT<{it}>({x})' if mode == 3 else f'fcvt_rm<{it}>({x}, {mode})'
            return W(ops[0], f'({T(n)}){fn}')
        if FREG.match(ops[0]):
            k, d = freg(ops[0]); x, w = fscal(ops[1])
            it = ('s32' if signed else 'u32') if w == 32 else ('s64' if signed else 'u64')
            fn = f'fcvtzT<{it}>({x})' if mode == 3 else f'fcvt_rm<{it}>({x}, {mode})'
            return f'c->v[{d}] = vU{w}(({UT[w]}){fn});'
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        it = STT[esz] if signed else UT[esz]
        return lanes_loop(d, cnt, esz, f'({UT[esz]})fcvtzT<{it}>(c->v[{n}].{FF[esz]}[i])', UF[esz])
    def i_fcvtzs(self, a, ops): return self._fcvt_int(ops, True, 3)
    def i_fcvtzu(self, a, ops): return self._fcvt_int(ops, False, 3)
    def i_fcvtms(self, a, ops): return self._fcvt_int(ops, True, 2)
    def i_fcvtps(self, a, ops): return self._fcvt_int(ops, True, 1)
    def i_fcvtpu(self, a, ops): return self._fcvt_int(ops, False, 1)
    def i_fcvtns(self, a, ops): return self._fcvt_int(ops, True, 0)
    def i_fcvtnu(self, a, ops): return self._fcvt_int(ops, False, 0)
    def i_fcvtas(self, a, ops): return self._fcvt_int(ops, True, 4)
    def i_fcvtau(self, a, ops): return self._fcvt_int(ops, False, 4)
    def i_fcvtmu(self, a, ops): return self._fcvt_int(ops, False, 2)
    def _icvt(self, ops, signed):
        if FREG.match(ops[0]) and is_gpr(ops[1]):
            k, d = freg(ops[0]); n = gsize(ops[1])
            src = f'({ST(n) if signed else T(n)}){R(ops[1])}'
            return f'c->v[{d}] = {"vS((float)" if k == "s" else "vD((double)"}{src});'
        if FREG.match(ops[0]):
            k, d = freg(ops[0]); _, s = freg(ops[1])
            if k == 's': return f'c->v[{d}] = vS((float)({"(s32)" if signed else "(u32)"}c->v[{s}].s[0]));'
            return f'c->v[{d}] = vD((double)({"(s64)" if signed else "(u64)"}c->v[{s}].d[0]));'
        d, cnt, esz = varr(ops[0]); n, _, _ = varr(ops[1])
        src = f'c->v[{n}].{SF[esz] if signed else UF[esz]}[i]'
        return lanes_loop(d, cnt, esz, f'({FT[esz]}){src}', FF[esz])
    def i_scvtf(self, a, ops): return self._icvt(ops, True)
    def i_ucvtf(self, a, ops): return self._icvt(ops, False)
    def i_fmov(self, a, ops):
        d, s = ops
        if s.startswith('#'):
            v = fimm(s)
            if FREG.match(d):
                k, r = freg(d)
                return f'c->v[{r}] = {"vS(" + repr(v) + "f)" if k == "s" else "vD(" + repr(v) + ")"};'
            dr, cnt, esz = varr(d)
            bits = struct.unpack('<I', struct.pack('<f', v))[0] if esz == 32 else struct.unpack('<Q', struct.pack('<d', v))[0]
            return f'c->v[{dr}] = vdup({esz // 8}, {cnt * esz // 8}, {bits:#x}ull);'
        if FREG.match(d) and is_gpr(s):
            k, r = freg(d)
            return f'c->v[{r}] = {"vU32((u32)" if k == "s" else "vU64("}{R(s)});'
        if is_gpr(d) and FREG.match(s):
            k, r = freg(s)
            return W(d, f'c->v[{r}].{"s" if k == "s" else "d"}[0]')
        if FREG.match(d) and FREG.match(s):
            k, r = freg(d); _, r2 = freg(s)
            return f'c->v[{r}] = {"vU32(c->v[%d].s[0])" % r2 if k == "s" else "vU64(c->v[%d].d[0])" % r2};'
        raise Unhandled('fmov form')

# ------------------------------------------------------------------ driver
_LANE_I = re.compile(r'\bi\b')
def expand_lanes(text):
    """LANES(B, E, stmt) -> stmt with i = B..E-1 written out. Lane loops must not reach the compiler as loops
    (thousands per function made LLVM loop passes quadratic), and lambda-based unrolling cost 46k template instances."""
    out, pos = [], 0
    while True:
        k = text.find('LANES(', pos)
        if k < 0: out.append(text[pos:]); return ''.join(out)
        out.append(text[pos:k])
        j = k + 6; depth = 1
        while depth:
            ch = text[j]
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            j += 1
        b, e, body = text[k + 6:j - 1].split(',', 2)
        body = body.strip()
        out.append(' '.join(_LANE_I.sub(str(v), body) + ';' for v in range(int(b), int(e))))
        pos = j
        if text.startswith(';', pos): pos += 1


class Out(io.StringIO):
    """write-if-changed file: unchanged outputs keep their timestamp, so ninja rebuilds only what changed"""
    def __init__(self, path): super().__init__(); self.path = path
    def __exit__(self, *a):
        new = self.getvalue()
        try: old = open(self.path).read()
        except OSError: old = None
        if old != new: open(self.path, 'w').write(new)
        return super().__exit__(*a)


def plt_map(img, info, imports_by_got, text_end, nslots=0):
    """map PLT stub address -> import name by decoding adrp/ldr x17 pairs"""
    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    out = {}
    plt0 = None
    # PLT lies at the end of .text; scan back far enough for 'adrp x16; ldr x17, [x16, #off]; add; br x17' x nslots
    # (sdk: 8,932 slots = 143 KB, ending ~6 KB before .text end)
    lo = max(0, text_end - max(0x10000, 16 * nslots + 0x10000))
    ins = list(md.disasm_lite(img[lo:text_end], lo))
    for i in range(len(ins) - 3):
        a, _, m1, o1 = ins[i]
        if m1 == 'adrp' and o1.startswith('x16,') and ins[i + 1][2] == 'ldr' and ins[i + 1][3].startswith('x17, [x16'):
            page = imm(split_ops(o1)[1])
            mm = MEM.match(split_ops(ins[i + 1][3])[1])
            off = imm(mm.group(2)) if mm and mm.group(2) else 0
            got = page + off
            if got in imports_by_got and ins[i + 3][2] == 'br':
                out[a] = imports_by_got[got]
    return out

def main():
    pre, fj, outdir = sys.argv[1:4]
    global IMGBASE, FN, IMP, TAB
    only = None; tu_insns = 60000; plt_all = False; trace = set(); hooks = set()
    args = sys.argv[4:]
    for i, x in enumerate(args):
        if x == '--only': only = [int(v, 16) for v in args[i + 1].split(',')]
        if x == '--tu-insns': tu_insns = int(args[i + 1])
        if x == '--base': IMGBASE = int(args[i + 1], 16)
        if x == '--prefix': FN, IMP, TAB = f'{args[i + 1]}_', f'{args[i + 1]}_imp_', f'g_{args[i + 1]}_'
        # every JUMP_SLOT is an import, also ones the module defines itself: the sdk calls its own exports through
        # its PLT (interposable), and the runtime binds each slot (HLE > main export > own lifted function)
        if x == '--plt-all': plt_all = True
        # debugging: log entry args / return value of these functions (hex offsets); only their TUs change
        if x == '--trace': trace = {int(v, 16) for v in args[i + 1].split(',')}
        # runtime hooks: these functions become <prefix><off>_orig, and <prefix><off> calls hook_<prefix><off>(c)
        if x == '--hook': hooks = {int(v, 16) for v in args[i + 1].split(',')}
    info = json.load(open(pre + '.json'))
    img = open(pre + '.bin', 'rb').read()
    fde = {int(k, 16): v for k, v in json.load(open(fj))['fde'].items()}
    text = next(s for s in info['segments'] if s['name'] == 'text')
    # imports: JMPREL (JUMP_SLOT) and GLOB_DAT in RELA
    dyn = info['dynamic']
    strtab, symtab = dyn['STRTAB'][0], dyn['SYMTAB'][0]
    def symname(i):
        no = struct.unpack_from('<I', img, symtab + i * 24)[0]
        return img[strtab + no:img.index(b'\0', strtab + no)].decode()
    imports = []           # index -> name
    imp_index = {}
    got_imports = {}       # got slot -> import index
    relocs = []
    def add_imp(name):
        if name not in imp_index:
            imp_index[name] = len(imports); imports.append(name)
        return imp_index[name]
    for tag_rel, tag_sz in (('JMPREL', 'PLTRELSZ'), ('RELA', 'RELASZ')):
        if tag_rel not in dyn: continue
        r, sz = dyn[tag_rel][0], dyn[tag_sz][0]
        for o in range(r, r + sz, 24):
            off, rinfo, add = struct.unpack_from('<QQq', img, o)
            typ, sym = rinfo & 0xFFFFFFFF, rinfo >> 32
            if typ in (1025, 1026, 257):   # GLOB_DAT, JUMP_SLOT, ABS64
                nm = symname(sym)
                shndx = struct.unpack_from('<H', img, symtab + sym * 24 + 6)[0]
                if shndx == 0 or (plt_all and typ == 1026):
                    idx = add_imp(nm)
                    if typ == 1026: got_imports[off] = idx
    plt = plt_map(img, info, got_imports, text['mem'] + text['size'], len(got_imports))
    em = Emitter(img, info, fde, plt, imports)
    em.trace = trace
    em.hooks = hooks
    os.makedirs(outdir, exist_ok=True)
    funcs = sorted(fde) if only is None else only
    tu, tu_n, tu_idx, tu_refs, tu_imps = [], 0, 0, set(), set()
    defined_all = set(funcs)

    def flush():
        nonlocal tu, tu_n, tu_idx, tu_refs, tu_imps
        if not tu: return
        decl = ''.join(f'void {FN}{r:x}(Ctx* __restrict);\n' for r in sorted(tu_refs))
        if any('rt_tpidrro(' in x for x in tu): tu.insert(0, 'u64 rt_tpidrro(Ctx*);')
        if any('rt_cntpct(' in x for x in tu): tu.insert(0, 'u64 rt_cntpct();')
        if any('rt_trace(' in x for x in tu): tu.insert(0, 'void rt_trace(Ctx*, u64, int);')
        idecl = ''.join(f'void {IMP}{i}(Ctx*);\n' for i in sorted(tu_imps))
        with Out(os.path.join(outdir, f'tu_{tu_idx:03d}.cpp')) as f:
            f.write(f'// generated by tools/recomp/gen.py - do not edit\n#include "rt.h"\n#define IMGBASE {IMGBASE:#x}ull\n'
                    f'#pragma clang diagnostic ignored "-Wunused-label"\n{decl}{idecl}\n')
            f.write(expand_lanes('\n'.join(tu)))
        tu, tu_n, tu_refs, tu_imps = [], 0, set(), set()
        tu_idx += 1

    for s in funcs:
        code = em.function(s)
        tu.append(code); tu_n += fde[s] // 4
        tu_refs |= em.cur_refs; tu_imps |= em.cur_imps
        if tu_n >= tu_insns: flush()
    flush()
    # function table + import thunks
    with Out(os.path.join(outdir, 'ftab.cpp')) as f:
        f.write('// generated\n#include "rt.h"\n')
        for s in funcs: f.write(f'void {FN}{s:x}(Ctx* __restrict);\n')
        f.write(f'struct FtabEnt {{ u64 off; GuestFn fn; }};\nextern const FtabEnt {TAB}ftab_init[] = {{\n')
        for s in funcs: f.write(f'{{{s:#x}, {FN}{s:x}}},\n')
        f.write(f'}};\nextern const u64 {TAB}ftab_count = sizeof({TAB}ftab_init) / sizeof({TAB}ftab_init[0]);\n')
    with Out(os.path.join(outdir, 'imports.cpp')) as f:
        f.write(f'// generated\n#include "rt.h"\nextern GuestFn {TAB}imp[];\n')
        f.write(f'extern const u64 {TAB}imp_count = {len(imports)};\nextern const char* const {TAB}imp_names[] = {{\n')
        for n in imports: f.write(f'"{n}",\n')
        f.write('};\n')
        for i in range(len(imports)): f.write(f'void {IMP}{i}(Ctx* c) {{ {TAB}imp[{i}](c); }}\n')
    rep = dict(functions=len(funcs), insns=em.stats['insns'], tus=tu_idx, imports=len(imports), plt_stubs=len(plt),
               unhandled_total=sum(em.unhandled.values()), stats=dict(em.stats),
               unhandled=em.unhandled.most_common(), unhandled_examples=em.unhandled_ex,
               plt_map={hex(k): imports[v] for k, v in plt.items()}, adr_sites=[hex(x) for x in em.adr_sites])
    json.dump(rep, open(os.path.join(outdir, 'report.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('unhandled_examples', 'plt_map')}, indent=1)[:3000])

if __name__ == '__main__':
    main()

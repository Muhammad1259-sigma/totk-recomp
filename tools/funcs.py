"""Function-start discovery for an NSO image (from nso.py).

usage: python funcs.py <img_prefix> <out.json>
Sources: .eh_frame_hdr FDE table (start + length), BL targets, RELATIVE relocations pointing into
.text, init/fini arrays, exported symbols. Reports overlap between the sources.
"""
import json, struct, sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

pre, out = sys.argv[1], sys.argv[2]
info = json.load(open(pre + '.json'))
img = open(pre + '.bin', 'rb').read()
text = next(s for s in info['segments'] if s['name'] == 'text')
t0, t1 = text['mem'], text['mem'] + text['size']
in_text = lambda a: t0 <= a < t1 and a % 4 == 0

def read_enc(off, enc, pc):
    """DWARF pointer encoding reader (subset used by eh_frame_hdr)."""
    fmt = enc & 0x0F
    app = enc & 0x70
    if fmt == 0x0B:   # sdata4
        v = struct.unpack_from('<i', img, off)[0]; n = 4
    elif fmt == 0x03:  # udata4
        v = struct.unpack_from('<I', img, off)[0]; n = 4
    elif fmt in (0x04, 0x0C):
        v = struct.unpack_from('<q', img, off)[0]; n = 8
    else:
        raise ValueError(hex(enc))
    if app == 0x10:
        v += pc
    elif app == 0x30:
        v += hdr
    return v, n

# --- eh_frame_hdr
hdr = info['mod0']['eh_frame_hdr_start']
ver, eh_ptr_enc, fde_count_enc, table_enc = img[hdr:hdr + 4]
off = hdr + 4
eh_frame, n = read_enc(off, eh_ptr_enc, off); off += n
fde_count, n = read_enc(off, fde_count_enc, off); off += n
fde = {}
for i in range(fde_count):
    loc, n = read_enc(off, table_enc, off); off += n
    fde_addr, n = read_enc(off, table_enc, off); off += n
    # FDE: length(4) CIE_ptr(4) pc_begin pc_range (assume sdata4 pcrel, the usual clang encoding)
    pc_begin = struct.unpack_from('<i', img, fde_addr + 8)[0] + fde_addr + 8
    pc_range = struct.unpack_from('<I', img, fde_addr + 12)[0]
    fde[loc] = pc_range
    assert pc_begin == loc, (hex(loc), hex(pc_begin))

# --- relocations
dyn = info['dynamic']
rel_text = set()
nrel = 0
if 'RELA' in dyn:
    r, sz = dyn['RELA'][0], dyn['RELASZ'][0]
    for o in range(r, r + sz, 24):
        roff, rinfo, addend = struct.unpack_from('<QQq', img, o)
        nrel += 1
        if rinfo & 0xFFFFFFFF == 1027 and in_text(addend):  # R_AARCH64_RELATIVE
            rel_text.add(addend)

# --- BL targets
md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
md.skipdata = True
bl = set()
for a, s, m, ops in md.disasm_lite(img[t0:t1], t0):
    if m == 'bl':
        try:
            bl.add(int(ops.lstrip('#'), 16))
        except ValueError:
            pass

fs = set(fde)
covered = sum(fde.values())
res = dict(text_bytes=t1 - t0, fde_count=len(fde), fde_bytes_covered=covered,
           fde_coverage=round(covered / (t1 - t0), 4),
           bl_targets=len(bl), bl_not_fde_start=len(bl - fs), rel_text_ptrs=len(rel_text),
           rel_not_fde_start=len(rel_text - fs), relocs=nrel,
           fde_start_not_bl_or_rel=len(fs - bl - rel_text))
print(json.dumps(res, indent=1))
res['bl_not_fde_sample'] = [hex(a) for a in sorted(bl - fs)[:20]]
res['rel_not_fde_sample'] = [hex(a) for a in sorted(rel_text - fs)[:20]]
json.dump(dict(summary=res, fde={hex(k): v for k, v in sorted(fde.items())},
               rel_text=sorted(rel_text), bl=sorted(bl)), open(out, 'w'))

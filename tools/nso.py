"""NSO -> flat memory image + JSON layout.

usage: python nso.py <nso> <out_prefix>
writes <out_prefix>.bin (text|ro|data laid out at their memory offsets, bss not included)
and <out_prefix>.json (build id, segments, MOD0, dynamic info).
"""
import hashlib, json, struct, sys
try:
    import lz4.block
    def lz4_block(src, usize):
        return lz4.block.decompress(src, uncompressed_size=usize)
except ImportError:      # no `pip install lz4` (e.g. a stock SteamOS python): a plain-python LZ4 block decoder (slower, same output)
    def lz4_block(src, usize):
        dst = bytearray()
        i, n = 0, len(src)
        while i < n:
            tok = src[i]; i += 1
            lit = tok >> 4
            if lit == 15:
                while True:
                    b = src[i]; i += 1; lit += b
                    if b != 255: break
            dst += src[i:i + lit]; i += lit
            if i >= n: break
            off = src[i] | src[i + 1] << 8; i += 2
            ml = tok & 15
            if ml == 15:
                while True:
                    b = src[i]; i += 1; ml += b
                    if b != 255: break
            ml += 4
            start = len(dst) - off
            if off >= ml: dst += dst[start:start + ml]
            else:                                   # overlapping copy: the last `off` bytes repeat
                pat = bytes(dst[start:])
                dst += pat * (ml // off) + pat[:ml % off]
        assert len(dst) == usize, (len(dst), usize)
        return bytes(dst)

def load(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'NSO0', 'not an NSO'
    flags = struct.unpack_from('<I', b, 0xC)[0]
    segs = []
    for i, name in enumerate(('text', 'ro', 'data')):
        foff, moff, size = struct.unpack_from('<III', b, 0x10 + i * 0x10)
        csize = struct.unpack_from('<I', b, 0x60 + i * 4)[0]
        raw = b[foff:foff + csize]
        if flags & (1 << i):
            raw = lz4_block(raw, size)
        assert len(raw) == size, (name, len(raw), size)
        if flags & (1 << (i + 3)):
            assert hashlib.sha256(raw).digest() == b[0xA0 + i * 0x20:0xC0 + i * 0x20], name + ' hash'
        segs.append(dict(name=name, mem=moff, size=size, data=raw))
    bss = struct.unpack_from('<I', b, 0x3C)[0]
    build_id = b[0x40:0x60].hex()
    return segs, bss, build_id

def main():
    nso, out = sys.argv[1], sys.argv[2]
    segs, bss, build_id = load(nso)
    end = max(s['mem'] + s['size'] for s in segs)
    img = bytearray(end)
    for s in segs:
        img[s['mem']:s['mem'] + s['size']] = s['data']
    open(out + '.bin', 'wb').write(img)
    info = dict(build_id=build_id, bss=bss, image_size=end,
                segments=[{k: v for k, v in s.items() if k != 'data'} for s in segs])
    # MOD0 header: text+4 holds offset of MOD0 relative to image base
    mod0 = struct.unpack_from('<I', img, 4)[0]
    if img[mod0:mod0 + 4] == b'MOD0':
        dyn, bss_s, bss_e, eh_s, eh_e, modobj = struct.unpack_from('<iiiiii', img, mod0 + 4)
        info['mod0'] = dict(at=mod0, dynamic=mod0 + dyn, bss_start=mod0 + bss_s, bss_end=mod0 + bss_e,
                            eh_frame_hdr_start=mod0 + eh_s, eh_frame_hdr_end=mod0 + eh_e, module_object=mod0 + modobj)
        # parse dynamic section
        d = info['mod0']['dynamic']
        tags = {}
        while True:
            tag, val = struct.unpack_from('<qQ', img, d)
            if tag == 0:
                break
            tags.setdefault(tag, []).append(val)
            d += 16
        names = {1: 'NEEDED', 4: 'HASH', 5: 'STRTAB', 6: 'SYMTAB', 7: 'RELA', 8: 'RELASZ', 10: 'STRSZ', 11: 'SYMENT',
                 23: 'JMPREL', 2: 'PLTRELSZ', 3: 'PLTGOT', 12: 'INIT', 13: 'FINI', 25: 'INIT_ARRAY', 27: 'INIT_ARRAYSZ',
                 26: 'FINI_ARRAY', 28: 'FINI_ARRAYSZ', 0x6ffffef5: 'GNU_HASH', 0x6ffffff9: 'RELACOUNT'}
        info['dynamic'] = {names.get(k, hex(k)): v for k, v in tags.items()}
    json.dump(info, open(out + '.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'dynamic'}, indent=1))

main()

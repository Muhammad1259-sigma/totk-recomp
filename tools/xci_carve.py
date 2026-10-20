"""Carve files (or the first N bytes of a file) out of an XCI's secure HFS0 partition.

usage: python -I xci_carve.py <xci> <outdir> [name=maxbytes ...]
Without name args it lists the secure partition. Reads only what it copies.
"""
import os, struct, sys

def hfs0(f, off):
    f.seek(off)
    magic, n, strsz, _ = struct.unpack('<4sIII', f.read(16))
    assert magic == b'HFS0', (hex(off), magic)
    ents = [struct.unpack('<QQII', f.read(0x40)[:24]) for _ in range(n)]
    strs = f.read(strsz)
    data0 = off + 16 + n * 0x40 + strsz
    out = {}
    for doff, size, noff, _ in ents:
        name = strs[noff:strs.index(b'\0', noff)].decode()
        out[name] = (data0 + doff, size)
    return out

def main():
    xci, outdir = sys.argv[1], sys.argv[2]
    want = dict(a.split('=') for a in sys.argv[3:])
    with open(xci, 'rb') as f:
        f.seek(0x130)
        root_off = struct.unpack('<Q', f.read(8))[0]
        root = hfs0(f, root_off)
        sec_off, _ = root['secure']
        sec = hfs0(f, sec_off)
        for name, (o, s) in sec.items():
            print(f'{name} off=0x{o:x} size={s}')
        os.makedirs(outdir, exist_ok=True)
        for name, lim in want.items():
            o, s = sec[name]
            n = s if lim == 'all' else min(s, int(lim, 0))
            f.seek(o)
            with open(os.path.join(outdir, name), 'wb') as w:
                left = n
                while left:
                    b = f.read(min(left, 8 << 20))
                    w.write(b)
                    left -= len(b)
            print(f'wrote {name} {n} bytes')

if __name__ == "__main__":
    main()

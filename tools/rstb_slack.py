"""Write an overlay copy of TotK's resource size table (RESTBL) with slack added to selected entries.

The game sizes each resource's heap from this table (heap = entry + 0x1008, measured for
Model/SystemModel.DeferredMain.bfres). Our recompiled build allocates slightly more than hardware on some
resource heaps (U9 in CURRENT.md), so the overlay gives those entries room while the root cause is open.
hle_fs.cpp serves overlay/romfs/<path> in place of extract/romfs/<path>; the game files are not touched.

usage: rstb_slack.py [--ext .bfres,...] [--pct 12.5] [--min 0x1000]
"""
import argparse, ctypes, ctypes.util, os, struct, sys, zlib

# Windows defaults as before; elsewhere (Linux build, tools/build_from_xci.sh): TOTK_ROMFS = the extracted romfs, TOTK_OVERLAY_OUT = the
# overlay romfs directory to write (the runtime reads <exe dir>/data/overlay/romfs, or TOTK_OVERLAY), libzstd from the system.
ROMFS = os.environ.get("TOTK_ROMFS", "C:/totk/extract/romfs")
REL = "System/Resource/ResourceSizeTable.Product.100.rsizetable.zs"
OUT = os.environ.get("TOTK_OVERLAY_OUT", "C:/totk/overlay/romfs") + "/" + REL
if sys.platform == "win32":
    ZSTD = ctypes.CDLL(r'C:\Program Files\Git\mingw64\bin\libzstd.dll')
else:
    ZSTD = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
ZSTD.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
ZSTD.ZSTD_decompress.restype = ctypes.c_size_t
ZSTD.ZSTD_compressBound.restype = ctypes.c_size_t
ZSTD.ZSTD_compress.restype = ctypes.c_size_t
ZSTD.ZSTD_isError.restype = ctypes.c_uint


def unzstd(b):
    n = ZSTD.ZSTD_getFrameContentSize(b, ctypes.c_size_t(len(b)))
    out = ctypes.create_string_buffer(n)
    r = ZSTD.ZSTD_decompress(out, ctypes.c_size_t(n), b, ctypes.c_size_t(len(b)))
    assert not ZSTD.ZSTD_isError(ctypes.c_size_t(r)) and r == n, r
    return out.raw[:r]


def zstd(b, level=19):
    cap = ZSTD.ZSTD_compressBound(ctypes.c_size_t(len(b)))
    out = ctypes.create_string_buffer(cap)
    r = ZSTD.ZSTD_compress(out, ctypes.c_size_t(cap), b, ctypes.c_size_t(len(b)), level)
    assert not ZSTD.ZSTD_isError(ctypes.c_size_t(r)), r
    return out.raw[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ext', default='.bfres', help='comma-separated suffixes of entries to grow (after stripping .zs)')
    ap.add_argument('--pct', type=float, default=12.5)
    ap.add_argument('--min', type=lambda s: int(s, 0), default=0x1000, help='minimum bytes added per grown entry')
    a = ap.parse_args()
    exts = tuple(e.strip() for e in a.ext.split(',') if e.strip())

    d = bytearray(unzstd(open(os.path.join(ROMFS, REL), 'rb').read()))
    assert d[:6] == b'RESTBL', d[:6]
    ver, slen, ncrc, nname = struct.unpack_from('<4I', d, 6)
    hdr = 22
    names = {}                                            # crc32(path) -> path for the selected suffixes
    for root, _, files in os.walk(ROMFS):
        for f in files:
            p = os.path.relpath(os.path.join(root, f), ROMFS).replace('\\', '/')
            for sfx in ('.zs', '.mc'):                     # RSTB keys name the decoded file (x.bfres for x.bfres.mc)
                if p.endswith(sfx): p = p[:-len(sfx)]
            if p.endswith(exts): names[zlib.crc32(p.encode())] = p
    grown = 0; added = 0
    for i in range(ncrc):
        o = hdr + 8 * i
        crc, size = struct.unpack_from('<2I', d, o)
        if crc in names:
            extra = max(a.min, int(size * a.pct / 100))
            struct.pack_into('<I', d, o + 4, size + extra); grown += 1; added += extra
    o = hdr + 8 * ncrc
    for i in range(nname):
        nm = bytes(d[o:o + slen]).split(b'\0')[0].decode()
        if nm.endswith(exts):
            size, = struct.unpack_from('<I', d, o + slen)
            extra = max(a.min, int(size * a.pct / 100))
            struct.pack_into('<I', d, o + slen, size + extra); grown += 1; added += extra
        o += slen + 4
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'wb').write(zstd(bytes(d)))
    print(f'RESTBL v{ver}: {ncrc} crc + {nname} name entries; {len(names)} romfs paths match {exts}; '
          f'grew {grown} entries by {a.pct}% (min {a.min:#x}), +{added / 1048576:.1f} MiB total -> {OUT}')


if __name__ == '__main__':
    main()

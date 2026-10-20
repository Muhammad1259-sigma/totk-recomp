# Build the quick-menu background fix overlay: a copy of UI/LayoutArchive/Common.Product.100.Nin_NX_NVN.blarc whose
# anim/ShortCutBG_00_In.bflan (weapon / shield / material / ability quick menus) and anim/AutoBuilder_00_In.bflan show
# their capture panes (C_CaptureBG_00, C_CaptureClearBG_00, P_CaptureClearBG_00) at animation frame 0.0 instead of 1.0.
# Why: the menu captures the game picture when those panes become visible. UI animations advance by the game's frame
# ratio, so above 30 fps (hle_vi.cpp speed fix: ratio 0.5 at 60 fps, ~0.33 at 90) frame 1.0 comes 2-3 frames after the
# world stopped drawing and the background is black. Found and fixed this way by MacNativeGames
# (github.com/ly05010419/MacNativeGames, MIT, games/totk/scripts/make_romfs_patches.py). Deck 2026-10-18: d97/d98/d100
# 0 black captures with it, d96 5 without. hle_hid.cpp dpad_gate stays on: without it a giant "Sort" label showed at
# 90 fps with and without this edit (d99/d100), not with the gate (d102) or at 30 fps (d101).
# The edit is 4 bytes per key in place, so the decompressed size (and the RSTB entry) stays the same.
# Paths as tools/make_gameover_fix.py: TOTK_ROMFS = the extracted romfs, TOTK_OVERLAY_OUT = the overlay romfs to write
# (Windows defaults C:/totk/extract/romfs and C:/totk/overlay/romfs). Needs only libzstd.
import ctypes, ctypes.util, os, struct, sys

if sys.platform == "win32":
    Z = ctypes.CDLL(r'C:\Program Files\Git\mingw64\bin\libzstd.dll')
else:
    Z = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
for f in ("ZSTD_getFrameContentSize",): getattr(Z, f).restype = ctypes.c_ulonglong
for f in ("ZSTD_decompress", "ZSTD_compressBound", "ZSTD_decompress_usingDict", "ZSTD_compress_usingDict"): getattr(Z, f).restype = ctypes.c_size_t
Z.ZSTD_isError.restype = ctypes.c_uint
Z.ZSTD_createDCtx.restype = Z.ZSTD_createCCtx.restype = ctypes.c_void_p
S = ctypes.c_size_t

def unzstd(b, d=b""):
    n = Z.ZSTD_getFrameContentSize(b, S(len(b)))
    out = ctypes.create_string_buffer(n)
    r = Z.ZSTD_decompress_usingDict(ctypes.c_void_p(Z.ZSTD_createDCtx()), out, S(n), b, S(len(b)), d, S(len(d)))
    assert not Z.ZSTD_isError(S(r)) and r == n, r
    return out.raw[:r]

def zstd(b, d, level=19):
    cap = Z.ZSTD_compressBound(S(len(b)))
    out = ctypes.create_string_buffer(cap)
    r = Z.ZSTD_compress_usingDict(ctypes.c_void_p(Z.ZSTD_createCCtx()), out, S(cap), b, S(len(b)), d, S(len(d)), level)
    assert not Z.ZSTD_isError(S(r)), r
    return out.raw[:r]

def sarc_member(data, want):
    assert data[:4] == b"SARC"
    header, start = struct.unpack_from("<H", data, 4)[0], struct.unpack_from("<I", data, 12)[0]
    count = struct.unpack_from("<H", data, header + 6)[0]; names = header + 12 + count * 16 + 8
    for i in range(count):
        _, attr, lo, hi = struct.unpack_from("<4I", data, header + 12 + i * 16)
        o = names + (attr & 0xffffff) * 4
        if data[o:data.index(0, o)].decode() == want: return data[start + lo:start + hi]
    raise KeyError(want)


ROMFS = os.environ.get('TOTK_ROMFS', 'C:/totk/extract/romfs')
REL = 'UI/LayoutArchive/Common.Product.100.Nin_NX_NVN.blarc.zs'
OUT = os.environ.get('TOTK_OVERLAY_OUT', 'C:/totk/overlay/romfs') + '/' + REL
MEMBERS = {'anim/ShortCutBG_00_In.bflan': ('C_CaptureBG_00', 'C_CaptureClearBG_00', 'P_CaptureClearBG_00'),
           'anim/AutoBuilder_00_In.bflan': ('C_CaptureBG_00',)}

def sarc_members(data):
    """name -> (absolute start, end) of every SARC member."""
    assert data[:4] == b'SARC' and data[6:8] == b'\xff\xfe'
    header, size, start = struct.unpack_from('<HxxII', data, 4)
    assert size == len(data) and data[header:header + 4] == b'SFAT'
    count = struct.unpack_from('<H', data, header + 6)[0]; names = header + 12 + count * 16
    assert data[names:names + 4] == b'SFNT'; names += 8
    out = {}
    for i in range(count):
        _, attribute, lo, hi = struct.unpack_from('<4I', data, header + 12 + i * 16)
        offset = names + (attribute & 0xffffff) * 4
        out[data[offset:data.index(0, offset, start)].decode()] = (start + lo, start + hi)
    return out

def flvi_keys(data, base):
    """(pane name, absolute key offset, keys) of every FLVI (visibility) step track in the BFLAN at data[base:]."""
    assert data[base:base + 4] == b'FLAN' and data[base + 4:base + 6] == b'\xff\xfe'
    pos, out = base + 0x14, []
    for _ in range(struct.unpack_from('<H', data, base + 0x10)[0]):
        magic, size = data[pos:pos + 4], struct.unpack_from('<I', data, pos + 4)[0]
        if magic == b'pai1':
            count, table = struct.unpack_from('<H', data, pos + 14)[0], pos + struct.unpack_from('<I', data, pos + 16)[0]
            for e in range(count):
                eo = pos + struct.unpack_from('<I', data, table + 4 * e)[0]
                name = data[eo:eo + 28].split(b'\0')[0].decode()
                for t in range(data[eo + 28]):
                    to = eo + struct.unpack_from('<I', data, eo + 32 + 4 * t)[0]
                    if data[to:to + 4] != b'FLVI': continue
                    for g in range(data[to + 4]):
                        go = to + struct.unpack_from('<I', data, to + 8 + 4 * g)[0]
                        assert data[go + 2] == 1, 'visibility tracks are step curves'
                        ko = go + struct.unpack_from('<I', data, go + 8)[0]
                        keys = [struct.unpack_from('<fH', data, ko + 8 * k) for k in range(struct.unpack_from('<H', data, go + 4)[0])]
                        out.append((name, ko, keys))
        pos += size
    return out

dic = sarc_member(unzstd(open(f'{ROMFS}/Pack/ZsDic.pack.zs', 'rb').read()), 'zs.zsdic')
original = unzstd(open(f'{ROMFS}/{REL}', 'rb').read(), dic)
blarc = bytearray(original)
members = sarc_members(blarc)
for member, panes in MEMBERS.items():
    lo, hi = members[member]
    changed = []
    for name, ko, keys in flvi_keys(blarc, lo):
        if name not in panes: continue
        assert keys == [(0.0, 0), (1.0, 1)], (member, name, keys)
        struct.pack_into('<f', blarc, ko + 8, 0.0)   # the second key (visible) from frame 1.0 to 0.0
        changed.append(name)
    assert sorted(changed) == sorted(panes), (member, changed)
    print(f'{member}: {", ".join(changed)} visible at frame 0.0 (was 1.0)')
diff = [i for i in range(len(original)) if original[i] != blarc[i]]
assert diff and all(any(lo <= i < hi for lo, hi in (members[m] for m in MEMBERS)) for i in diff), diff
comp = zstd(bytes(blarc), dic)
assert unzstd(comp, dic) == bytes(blarc)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'wb').write(comp)
print(f'{len(diff)} bytes changed, size {len(blarc)} unchanged | wrote {OUT} ({len(comp)} bytes)')

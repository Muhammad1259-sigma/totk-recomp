# Build the game-over Retry fix overlay: a copy of Event/EventFlow/DmF_SY_GameOver.bfevfl whose Event14
# (EventTriggerPauseMgrRequestPause GameOverPause) ends its branch instead of going on to Event25 (EventWait Frames 1).
# Why: above 30 fps the speed fix (hle_vi.cpp hook_f_7a83e4) gives the game frame steps under one frame (k 0.5 at 60 fps,
# ~0.33 at 90), so the 1-frame wait needs several frames, the pause it just requested stops game time after the first,
# and the Continue / Quit menu never closes. Reproduced on the Deck at 60 fps (d94: Game Over stayed through ~80 A presses;
# d95 with this overlay: the same death reloaded the save, 2026-10-18). Found and fixed this way by MacNativeGames
# (github.com/ly05010419/MacNativeGames, MIT, games/totk/scripts/make_romfs_patches.py); the same v1.0.0 file, pinned
# by its decompressed SHA-256. The edit is Event14's 2-byte "next event" index (25 -> none).
# Paths as tools/rstb_slack.py: TOTK_ROMFS = the extracted romfs, TOTK_OVERLAY_OUT = the overlay romfs to write
# (Windows defaults C:/totk/extract/romfs and C:/totk/overlay/romfs). Needs only libzstd (like rstb_slack.py).
import ctypes, ctypes.util, hashlib, os, struct, sys

ROMFS = os.environ.get("TOTK_ROMFS", "C:/totk/extract/romfs")
REL = "Event/EventFlow/DmF_SY_GameOver.bfevfl.zs"
OUT = os.environ.get("TOTK_OVERLAY_OUT", "C:/totk/overlay/romfs") + "/" + REL
SHA256 = "c59975494fb30063b7a339fd66bbde8b456b11c73a5b4332bc070d63ece66b7d"
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

dic = sarc_member(unzstd(open(f"{ROMFS}/Pack/ZsDic.pack.zs", "rb").read()), "zs.zsdic")
raw = open(f"{ROMFS}/{REL}", "rb").read()
flow = bytearray(unzstd(raw, dic))
assert hashlib.sha256(flow).hexdigest() == SHA256, "DmF_SY_GameOver.bfevfl is not the analysed v1.0.0 file"
def event(i):   # BFEVFL events: 0x28 bytes from 0x1f0, name ref at +0, type at +8, next event index at +0xa
    e = 0x1f0 + 0x28 * i; ref = struct.unpack_from("<Q", flow, e)[0]; n = struct.unpack_from("<H", flow, ref)[0]
    return e, flow[ref + 2:ref + 2 + n].decode(), flow[e + 8], struct.unpack_from("<H", flow, e + 0xa)[0]
e14, name14, type14, next14 = event(7); _, name25, type25, next25 = event(25)
assert (name14, type14, next14) == ("Event14", 0, 25) and (name25, type25, next25) == ("Event25", 0, 0xffff)
struct.pack_into("<H", flow, e14 + 0xa, 0xffff)
data = zstd(bytes(flow), dic)
assert unzstd(data, dic) == bytes(flow)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "wb").write(data)
print(f"game-over fix: Event14 next Event25 (EventWait 1 frame) -> end | wrote {OUT} ({len(data)} bytes)")

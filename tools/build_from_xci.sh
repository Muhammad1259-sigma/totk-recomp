#!/usr/bin/env bash
# Build the TotK Linux port from YOUR OWN dump of the game. Nothing derived from the game ships in this package; this script makes it
# on your machine: extract (hactool) -> decompress the NSOs -> recompile AArch64 to C++ (tools/recomp/gen.py) -> build the binary.
#
# usage: build_from_xci.sh <TearsOfTheKingdom.xci> <prod.keys> [--out DIR] [--jobs N] [--gen-jobs N] [--gen-opt -O1] [--null] [--keep-nca] [--skip-extract] [--prepare-only]
#   <game.xci>  your cartridge dump, version 1.0.0 (title 0100F2C0115B6000, main build ID 082CE09B06E33A12). Read only.
#   <prod.keys> your Switch keys (needs header_key, key_area_key_application_xx and master_key_xx for the game's master key revision
#               0x0E). A filtered temporary copy is made inside DIR/work for hactool and deleted at the end; keys never go anywhere else.
#   --out DIR   everything is created here (default ./totk-build): data/{exefs,romfs,img,overlay}, gen/, deps/, build/, dist/totk-recomp
#   --jobs N    parallel jobs for small compiles and the link (default: all cores)
#   --gen-jobs  parallel jobs for the ~215 generated C++ files (each needs 1-3 GB of RAM; default: min(jobs, RAM_GB / 4), at least 1)
#   --gen-opt   optimisation level of the generated C++ (default -O2, the level the runtime was tuned at). -O1 compiles about 3x faster
#               (measured: clang 19, 3 jobs, the whole generated code in ~2 h instead of ~6-10 h) but the code runs slower.
#   --null      also build the headless null-GPU variant (dist/totk-recomp-null) for smoke tests without a GPU
#   --keep-nca  keep work/program.nca (17 GB) after extraction
#   --skip-extract  reuse an earlier DIR/data (exefs, romfs) and only regenerate + build (no XCI / keys needed)
#   --prepare-only  stop after the recompiler and the overlay: no third-party builds, no compile
# Disk: ~40 GB peak (program NCA copy 17 GB + romfs 17 GB), ~20 GB afterwards (romfs + images + generated sources + objects).
# Time: extraction ~10-30 min, recompiler ~2 min, compile of the generated C++ several hours (see --gen-opt; ~1.3 GB RAM per job, up to 2 GB).
# Needs: python3 + capstone EXACTLY 5.0.9 (gen.py parses capstone's disassembly text; 4.0.2, Ubuntu's apt python3-capstone, leaves thousands more instructions unhandled; the script makes a venv with the pinned version from PyPI when the system module is not 5.0.9), clang >= 17 + lld, ninja, cmake, make,
# gcc, curl, libzstd (python ctypes), Vulkan headers + libvulkan, static libshaderc_combined.a + libvpx.a (+ SDL3 build deps, see
# tools/linux/build_deps.sh). Debian / Ubuntu 24.04:
#   apt install clang lld ninja-build cmake make gcc curl python3 python3-venv libvulkan-dev libshaderc-dev glslang-dev spirv-tools \
#       libvpx-dev libzstd-dev \
#       libx11-dev libxext-dev libxcursor-dev libxi-dev libxrandr-dev libxss-dev libxfixes-dev libxtst-dev libwayland-dev wayland-protocols \
#       libxkbcommon-dev libdecor-0-dev libudev-dev libdbus-1-dev libasound2-dev libpulse-dev libpipewire-0.3-dev libdrm-dev libgbm-dev libegl-dev
# (SteamOS: the root file system is read-only; build in a container (distrobox with an Ubuntu 24.04 image) or on a PC and copy dist/.)
set -euo pipefail

XCI=""; KEYS=""; OUT="$PWD/totk-build"; JOBS="$(nproc)"; GENJOBS=""; GENOPT=""; NULL=0; KEEPNCA=0; SKIPEXTRACT=0; PREPARE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift ;; --jobs) JOBS="$2"; shift ;; --gen-jobs) GENJOBS="$2"; shift ;; --gen-opt) GENOPT="$2"; shift ;;
    --null) NULL=1 ;; --keep-nca) KEEPNCA=1 ;; --skip-extract) SKIPEXTRACT=1 ;; --prepare-only) PREPARE=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) if [ -z "$XCI" ]; then XCI="$1"; elif [ -z "$KEYS" ]; then KEYS="$1"; else echo "unexpected argument $1" >&2; exit 2; fi ;;
  esac
  shift
done
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
if [ -d "$HERE/runtime" ]; then ROOT="$HERE"; else ROOT="$(cd "$HERE/.." && pwd)"; fi   # package root (runtime/, tools/, third_party/)
[ "$SKIPEXTRACT" = 1 ] || { [ -f "$XCI" ] && [ -f "$KEYS" ] || { echo "usage: $0 <game.xci> <prod.keys> [options]  (--help)" >&2; exit 2; }; }
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
DATA="$OUT/data"; WORK="$OUT/work"; GEN="$OUT/gen"; DEPS="$OUT/deps"; BUILD="$OUT/build"; OBJ="$OUT/obj"; DIST="$OUT/dist"
mkdir -p "$DATA" "$WORK" "$DEPS"
say() { printf '\n== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
[ -n "$GENJOBS" ] || { GENJOBS=$((RAM_GB / 4)); [ "$GENJOBS" -gt "$JOBS" ] && GENJOBS="$JOBS"; [ "$GENJOBS" -lt 1 ] && GENJOBS=1; }
echo "root $ROOT, out $OUT, $JOBS jobs ($GENJOBS for generated code), ${RAM_GB} GB RAM"

# ---- prerequisites
miss=""
for t in python3 ninja cmake make gcc curl; do command -v "$t" >/dev/null || miss="$miss $t"; done
CXXB="${CXX:-}"; CCB="${CC:-}"
if [ -z "$CXXB" ]; then for c in clang++-19 clang++-18 clang++-17 clang++; do command -v "$c" >/dev/null && { CXXB="$c"; break; }; done; fi
if [ -z "$CCB" ]; then for c in clang-19 clang-18 clang-17 clang; do command -v "$c" >/dev/null && { CCB="$c"; break; }; done; fi
[ -n "$CXXB" ] && [ -n "$CCB" ] || miss="$miss clang(>=17)"
command -v ld.lld >/dev/null || command -v ld.lld-19 >/dev/null || ls /usr/lib/llvm-*/bin/ld.lld >/dev/null 2>&1 || miss="$miss lld"
[ -f /usr/include/vulkan/vulkan.h ] || miss="$miss vulkan-headers"
[ -z "$miss" ] || { echo "missing tools:$miss (see the header of this script)" >&2; exit 1; }
export CC="$CCB" CXX="$CXXB"
SYSLIB=""; for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do [ -f "$d/libvpx.a" ] && SYSLIB="$d" && break; done
[ -n "$SYSLIB" ] && [ -f "$SYSLIB/libshaderc_combined.a" ] || { echo "need static libvpx.a and libshaderc_combined.a (libvpx-dev, libshaderc-dev)" >&2; exit 1; }
# (Debian / Ubuntu's libshaderc_combined.a is a thin archive: glslang and SPIRV-Tools must be installed as static libs too; a shaderc built
# from google/shaderc already contains them)
[ -f "$SYSLIB/libglslang.a" ] && [ -f "$SYSLIB/libSPIRV-Tools-opt.a" ] || [ "$(stat -c %s "$SYSLIB/libshaderc_combined.a")" -gt 5000000 ] ||
  { echo "libshaderc_combined.a is the thin Debian archive: also install glslang-dev and spirv-tools (static libglslang.a, libSPIRV-Tools-opt.a)" >&2; exit 1; }
PY=python3
if ! $PY -c 'import capstone,sys; sys.exit(0 if capstone.__version__ == "5.0.9" else 1)' 2>/dev/null; then
  say "python venv with capstone==5.0.9"
  { $PY -m venv "$OUT/venv" && "$OUT/venv/bin/pip" install --quiet capstone==5.0.9; } || { echo "cannot install capstone==5.0.9 (needs python3-venv and network access to PyPI)" >&2; exit 1; }
  PY="$OUT/venv/bin/python"
fi

# ---- 1. hactool + SDL3 + Opus
[ "$SKIPEXTRACT" = 1 ] || say "third-party: hactool 1.4.0"
[ "$SKIPEXTRACT" = 1 ] || bash "$ROOT/tools/linux/build_deps.sh" "$DEPS" --hactool --no-libs --jobs "$JOBS"
HACTOOL="$DEPS/hactool/hactool"

# ---- 2. extract exefs + romfs from the XCI
if [ "$SKIPEXTRACT" = 0 ]; then
  KF="$WORK/keys.filtered"
  trap 'rm -f "$KF"' EXIT
  # hactool aborts on lines that are not plain "name = hex" keys; prod.keys from other tools has extras
  tr -d '\r' < "$KEYS" | grep -E '^[[:space:]]*[A-Za-z0-9_]+[[:space:]]*=[[:space:]]*([0-9A-Fa-f]{32}|[0-9A-Fa-f]{64})[[:space:]]*$' > "$KF"
  chmod 600 "$KF"
  say "XCI secure partition"
  $PY "$ROOT/tools/xci_carve.py" "$XCI" "$WORK/xci" | tee "$WORK/xci_list.txt"
  NCA="$(grep -E '\.nca off=' "$WORK/xci_list.txt" | grep -v '\.cnmt\.nca' | sed -E 's/^(\S+) off=\S+ size=([0-9]+)$/\2 \1/' | sort -rn | head -1 | cut -d' ' -f2)"
  [ -n "$NCA" ] || { echo "no program NCA found in the XCI" >&2; exit 1; }
  AVAIL_GB=$(df -P "$OUT" | awk 'NR==2 {printf "%d", $4/1048576}')
  [ "$AVAIL_GB" -ge 40 ] || echo "WARNING: only ${AVAIL_GB} GB free in $OUT, the extraction needs ~40 GB at its peak"
  say "copy the program NCA ($NCA)"
  $PY "$ROOT/tools/xci_carve.py" "$XCI" "$WORK/xci" "$NCA=all" > /dev/null
  say "extract exefs + romfs (17 GB, slow)"
  mkdir -p "$DATA/exefs" "$DATA/romfs"
  "$HACTOOL" -k "$KF" -t nca --exefsdir="$DATA/exefs" --romfsdir="$DATA/romfs" "$WORK/xci/$NCA" > "$WORK/hactool.log" 2>&1 || { tail -20 "$WORK/hactool.log"; exit 1; }
  rm -f "$KF"
  [ "$KEEPNCA" = 1 ] || rm -rf "$WORK/xci"
  [ -f "$DATA/exefs/main" ] && [ -f "$DATA/romfs/System/Resource/ResourceSizeTable.Product.100.rsizetable.zs" ] ||
    { echo "extraction incomplete (see $WORK/hactool.log)" >&2; exit 1; }
fi

# ---- 3. images, function tables, recompiler
say "NSO images"
mkdir -p "$DATA/img"
for m in main sdk rtld subsdk0; do $PY "$ROOT/tools/nso.py" "$DATA/exefs/$m" "$DATA/img/$m" > /dev/null; done
GOT="$($PY -c "import json;print(json.load(open('$DATA/img/main.json'))['build_id'][:16].upper())")"
echo "main build id prefix $GOT (expected 082CE09B06E33A12)"
[ "$GOT" = "082CE09B06E33A12" ] || echo "WARNING: this is not game version 1.0.0; the recompiler's fixes and hooks are for 1.0.0 and the result will not work" >&2
say "function tables"
$PY "$ROOT/tools/funcs.py" "$DATA/img/main" "$WORK/funcs_main.json" > /dev/null
$PY "$ROOT/tools/funcs.py" "$DATA/img/sdk" "$WORK/funcs_sdk.json" > /dev/null
say "recompile AArch64 -> C++ (main, then the lifted sdk)"
$PY "$ROOT/tools/recomp/gen.py" "$DATA/img/main" "$WORK/funcs_main.json" "$GEN/main" --hook 7a83e4 > "$WORK/gen_main.log"
$PY "$ROOT/tools/recomp/gen.py" "$DATA/img/sdk" "$WORK/funcs_sdk.json" "$GEN/sdk" --base 0x7180000000 --prefix s --plt-all > "$WORK/gen_sdk.log"
say "resource-table overlay"
TOTK_ROMFS="$DATA/romfs" TOTK_OVERLAY_OUT="$DATA/overlay/romfs" $PY "$ROOT/tools/rstb_slack.py"
say "game-over Retry fix overlay (the Continue / Quit menu froze above 30 fps)"
TOTK_ROMFS="$DATA/romfs" TOTK_OVERLAY_OUT="$DATA/overlay/romfs" $PY "$ROOT/tools/make_gameover_fix.py"
say "quick-menu background fix overlay (the menus captured a black background above 30 fps)"
TOTK_ROMFS="$DATA/romfs" TOTK_OVERLAY_OUT="$DATA/overlay/romfs" $PY "$ROOT/tools/make_quickmenu_fix.py"

# ---- 4. build
[ "$PREPARE" = 0 ] || { say "prepare-only: stopping before the build (generated sources are in $GEN)"; exit 0; }
say "third-party: SDL3 3.2.30, Opus 1.5.2"
bash "$ROOT/tools/linux/build_deps.sh" "$DEPS" --jobs "$JOBS"
say "build.ninja (Vulkan)"
GENARGS="--gen-pool $GENJOBS"; [ -z "$GENOPT" ] || GENARGS="$GENARGS --gen-opt $GENOPT"
$PY "$ROOT/tools/recomp/mkninja_linux.py" --root "$ROOT" --gen "$GEN/main" --out "$BUILD" --obj "$OBJ" --deps "$DEPS" --syslib "$SYSLIB" $GENARGS
say "compile the generated C++ ($GENJOBS at a time; this is the long step - it can be interrupted and resumed by re-running with --skip-extract)"
nice -n 10 ninja -C "$BUILD" -j"$GENJOBS" objs-gen
say "compile the runtime and link"
nice -n 10 ninja -C "$BUILD" -j"$JOBS"
if [ "$NULL" = 1 ]; then
  say "null-GPU variant"
  $PY "$ROOT/tools/recomp/mkninja_linux.py" --null --prebuilt-gen --root "$ROOT" --gen "$GEN/main" --out "$OUT/build-null" --obj "$OBJ" --deps "$DEPS" --syslib "$SYSLIB"
  nice -n 10 ninja -C "$OUT/build-null" -j"$JOBS"
fi

# ---- 5. assemble dist/totk-recomp
say "assemble $DIST/totk-recomp"
D="$DIST/totk-recomp"
rm -rf "$D"; mkdir -p "$D/lib"
cp "$BUILD/totk" "$D/totk"
cp -a "$DEPS/sdl3-install/lib/libSDL3.so.0" "$DEPS/sdl3-install/lib/libSDL3.so.0.2.30" "$D/lib/"
cp "$ROOT/tools/linux/run.sh" "$D/run.sh"; chmod +x "$D/run.sh" "$D/totk"
ln -s "$DATA" "$D/data"          # data/{img,romfs,overlay} stay where they were extracted (copy with cp -rL to move the folder)
echo
echo "Done. Run:  $D/run.sh      (add it to Steam as a non-Steam game for the Deck's Gaming Mode)"

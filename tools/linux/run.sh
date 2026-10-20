#!/bin/bash
# TotK recomp launcher (Steam Deck / Linux). Add THIS FILE to Steam as a non-Steam game (Games > Add a Non-Steam Game > Browse,
# pick run.sh; in Desktop mode, or "Add a Non-Steam Game" from the library) and launch it from Gaming Mode.
# Everything it needs is found relative to this file: totk (the binary), lib/libSDL3.so.0, data/img, data/romfs, data/overlay.
# Saves and caches live in ${XDG_DATA_HOME:-~/.local/share}/totk-recomp; the log is run/totk.log there.
# Environment you can set before launching (all optional):
#   TOTK_ROMFS=/run/media/mmcblk0p1/totk/romfs   the extracted romfs on an SD card / USB disk (default: data/romfs next to this file)
#   TOTK_FPS=90 (default: the OLED panel's 90 Hz) | 60 | 30 | 0 (the game's own 30 fps pacing) ; TOTK_UNLOCK_FPS=1 = no limiter
#   TOTK_DYNRES=1 (default here) the game's own dynamic resolution aims at TOTK_FPS (3D only, the HUD stays sharp; d76/d77:
#                 90.0 in every bench window, 1024x576 in heavy scenes, back to 1280x720 in light ones). 0 = fixed 1280x720
#   In game: L3 + R3 (or F10) = Graphics menu (resolution, dynamic resolution, upscaler, sharpening, frame rate); its
#   choices are saved and override these defaults on the next start.
#   TOTK_HANDHELD=1 (default here) the game runs in its handheld mode: 1280x720 rendering, handheld detail, the right fit for the
#                   Deck's 1280x800 screen and GPU. TOTK_HANDHELD=0 = docked mode (1600x900, more detail, much heavier)
#   LARECOMP_ASYNC_EXEC=1 (default here) GPU command translation on its own thread (=0: on the game's thread)
#   TOTK_FULLSCREEN=0   start in a window (F11 / Alt+Enter toggles either way) ; F3 = fps readout
#   TOTK_STATE=<dir>    where saves / caches / the log go
#   TOTK_USERDATA=<dir> saves only (a COPY of a Windows userdata folder works: it is the same layout)
#   TOTK_GYRO=0 (no motion)  TOTK_RUMBLE=0..4 (gain)  TOTK_VOLUME=0..2  TOTK_AUDIO=null (no sound)  TOTK_PAD_LABELS=1
#   LARECOMP_VK_PRESENT=fifo|mailbox|immediate   TOTK_RES_SCALE=1.0 (render scale; "auto" = window height / 1080)
set -u
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
state="${TOTK_STATE:-${XDG_DATA_HOME:-$HOME/.local/share}/totk-recomp}"
mkdir -p "$state/userdata" "$state/run"

export TOTK_IMG="${TOTK_IMG:-$here/data/img}"
export TOTK_ROMFS="${TOTK_ROMFS:-$here/data/romfs}"
export TOTK_USERDATA="${TOTK_USERDATA:-$state/userdata}"
[ -d "$here/data/overlay/romfs" ] && export TOTK_OVERLAY="${TOTK_OVERLAY:-$here/data/overlay/romfs}"
[ -d "$here/data/overlay/skip_opening/romfs" ] && export TOTK_SKIP_DIR="${TOTK_SKIP_DIR:-$here/data/overlay/skip_opening/romfs}"
[ -f "$here/data/fonts/DejaVuSans-Bold.ttf" ] && export TOTK_FONT="${TOTK_FONT:-$here/data/fonts/DejaVuSans-Bold.ttf}"
export TOTK_FPS="${TOTK_FPS:-90}"
export TOTK_DYNRES="${TOTK_DYNRES:-1}"
export TOTK_HANDHELD="${TOTK_HANDHELD:-1}"
export LARECOMP_ASYNC_EXEC="${LARECOMP_ASYNC_EXEC:-1}"
export LARECOMP_VKQ="${LARECOMP_VKQ:-1}"                              # vkCmd recording on its own thread (d43: +1.8 % at 3770 draws/frame, render thread 92 -> 71 %)
# Thread pinning (TOTK_AFFINITY=1) is OFF by default since 2026-10-18: render + vk-record pinned together to core 3 were
# preempted 134-294x per stutter frame. Light-stretch hitches >= 14 ms at 90 fps: pinned 18-64 per run (6 runs), unpinned
# 1 / 7 / 11 (d87_noaff, d88_noaff, d88_noaff_fix); heavy fps the same (84.7 vs 84.5 uncapped); guest-only pinning
# ("0-1,2-3,4-5:") was worst (d88_guest heavy 75.9, 475 hitches). It was +4 % in d49 (before VKQ / DRS). Set by hand to test.
export TOTK_FULLSCREEN="${TOTK_FULLSCREEN:-1}"
export LARECOMP_MEMREPORT_AT="${LARECOMP_MEMREPORT_AT:-300,1500}"   # test builds: "[mem]" memory breakdown in the log at these frames
export LARECOMP_VK_PROF="${LARECOMP_VK_PROF:-0}"                    # =1: GPU ms + render-thread CPU split per 2 s in the stats line (costs ~5 % fps on the Deck, d36_noprof)
export LARECOMP_GPUPROF_AT="${LARECOMP_GPUPROF_AT:-1500,2500}"       # test builds: "[gpuprof]" GPU time per pass of these frames
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -f "$TOTK_ROMFS/System/Resource/ResourceSizeTable.Product.100.rsizetable.zs" ]; then
  msg="TotK recomp: the romfs was not found at $TOTK_ROMFS. Copy your extracted romfs there (see README.txt) or set TOTK_ROMFS."
  echo "$msg" >&2
  command -v zenity >/dev/null 2>&1 && zenity --error --no-markup --text="$msg" 2>/dev/null
  exit 1
fi
if [ ! -f "$TOTK_IMG/main.bin" ]; then echo "TotK recomp: $TOTK_IMG/main.bin is missing" >&2; exit 1; fi

# shader precompile seed (read on every start, next to the cache's own recording): the shader programs, pipeline states and
# compute programs recorded on the PC and the Deck, plus the game's shader archives (archive_shaders.bin: every material
# shader, warmed in the background). Builds before 2026-10-17 copied it into a fresh cache once instead.
if [ -f "$here/data/cache-seed/precompile.bin" ]; then
  export TOTK_PRECOMPILE_SEED="${TOTK_PRECOMPILE_SEED:-$here/data/cache-seed}"
fi
cd "$state/run"
# Remote test runs (the PC over SSH, 2026-10-16): a one-shot test.env next to the log is read by the next start and renamed to
# test.env.used, so a normal start afterwards is unaffected. TOTK_TEST_SECONDS=<s> ends the game after that long.
if [ -f test.env ]; then set -a; . ./test.env; set +a; mv -f test.env test.env.used; fi
# Cloud saves (2026-10-17, same design as the GoW3R one): when $state/cloudsave/cloudsave.json exists (it names the save hub on
# the PC), the save is synced before the game starts (asks when the other device's save is newer) and uploaded after the game
# closes. Log: $state/cloudsave/cloudsave.log. Skipped for test runs, a save folder other than the default (TOTK_USERDATA
# set by hand) and TOTK_CLOUDSAVE=0. The shader warm-up syncs too: "play now" on its screen starts real play (2026-10-20),
# and an unchanged save uploads nothing ("save unchanged, nothing to upload").
cs_py=$(command -v python3 || true)
cs_sync="$here/cloudsave/totk_cloudsave.py"; cs_conf="$state/cloudsave/cloudsave.json"
if [ "${TOTK_CLOUDSAVE:-1}" != 0 ] && [ -z "${TOTK_TEST_SECONDS:-}" ] && \
   [ "$TOTK_USERDATA" = "$state/userdata" ] && [ -n "$cs_py" ] && [ -f "$cs_sync" ] && [ -f "$cs_conf" ]; then
  # Steam's runtime libraries and overlay can break system tools, so the sync runs without them.
  env -u LD_PRELOAD -u LD_LIBRARY_PATH "$cs_py" "$cs_sync" pre --config "$cs_conf"
  # The upload waits for this PID (the game after exec) outside Steam's process tree, so "Exit game" can't kill it. If it
  # can't start, the next start uploads the save instead (pre: "only this device changed since the last sync").
  env -u LD_PRELOAD -u LD_LIBRARY_PATH systemd-run --user --collect --quiet "$cs_py" "$cs_sync" post --config "$cs_conf" \
    --wait-pid $$ > /dev/null 2>&1 || true
fi
[ -f totk.log ] && mv -f totk.log totk.prev.log
# TOTK_PERF=<file> (remote tests): the game runs under perf record (cpu-clock samples of every thread, TOTK_PERF_HZ per second,
# default 499). Yama lets perf sample only its own children, so attaching to a running game counts nothing (2026-10-16).
perf=""
[ -n "${TOTK_PERF:-}" ] && command -v perf >/dev/null 2>&1 && perf="perf record -q -e cpu-clock -F ${TOTK_PERF_HZ:-499} -o $TOTK_PERF --"
# Memory ceiling: the game runs in its own systemd scope, so if it ever grows past the limit the kernel closes the game instead
# of pushing the whole Deck into swap (the first Deck run, 2026-10-16, froze the system until a forced power-off).
# Default: total RAM minus 2.5 GB (at least 6 GB); TOTK_MEM_LIMIT_MB=<MB> sets it, TOTK_MEM_LIMIT_MB=0 turns it off.
lim="${TOTK_MEM_LIMIT_MB:-}"
if [ -z "$lim" ]; then
  tot=$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)
  lim=$(( tot - 2560 )); [ "$lim" -lt 6144 ] && lim=6144
fi
if [ "$lim" != 0 ] && command -v systemd-run >/dev/null 2>&1 &&
   systemd-run --user --scope --quiet -p MemoryMax="${lim}M" -p MemorySwapMax=512M true >/dev/null 2>&1; then
  echo "[run.sh] memory ceiling ${lim} MB, swap 512 MB (systemd scope; TOTK_MEM_LIMIT_MB=0 turns it off)" > totk.log
  exec ${TOTK_TEST_SECONDS:+timeout -k 10 "$TOTK_TEST_SECONDS"} systemd-run --user --scope --quiet -p MemoryMax="${lim}M" -p MemorySwapMax=512M $perf "$here/totk" "$@" >> totk.log 2>&1
fi
echo "[run.sh] no memory ceiling (systemd-run --user --scope not available or TOTK_MEM_LIMIT_MB=0)" > totk.log
exec ${TOTK_TEST_SECONDS:+timeout -k 10 "$TOTK_TEST_SECONDS"} $perf "$here/totk" "$@" >> totk.log 2>&1

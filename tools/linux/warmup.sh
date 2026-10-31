#!/bin/bash
# One-time shader warm-up (2026-10-17). Compiles every shader in the game's shader archives (data/cache-seed/
# archive_shaders.bin, ~18k programs) on all CPU cores but one, saves them to the shader caches, then exits by itself.
# Run it once after installing or updating (from Desktop mode, or add it to Steam like run.sh); on the Steam Deck it
# takes ~4.5 min on a fresh install (2026-10-20, measured: 263 s from launch to exit, 18,325 archive programs in
# 225 s; spvgen builds SPIR-V without glslang; before it 396 s, and 747 s with the 10-19 translator). It can be
# stopped and restarted (finished shaders are kept). Without it the warm-up runs slowly in the
# background while you play (TOTK_PRECOMPILE_ARCHIVE_DUTY, default 20 % of one core on 4-core machines).
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TOTK_WARMUP=1 exec "$here/run.sh" "$@"

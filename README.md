# totk-recomp

A static recompilation of *The Legend of Zelda: Tears of the Kingdom* (Switch, version 1.0.0) to native Linux x86-64, aimed at the Steam Deck.

**This repository contains no game code, no game data and no keys.** It holds the recompiler, the runtime (Switch OS services and an NVN-on-Vulkan graphics layer), a setup launcher and the build scripts. You build the game on your own machine from **your own cartridge dump and your own console keys**. Nothing built from the game may be shared.

Not affiliated with or endorsed by Nintendo.

## How it works

1. `tools/xci_carve.py` + hactool copy the program NCA out of your XCI and extract exefs and romfs.
2. `tools/nso.py` decompresses the NSO modules; `tools/funcs.py` finds the functions.
3. `tools/recomp/gen.py` translates the AArch64 code of `main` (and parts of the SDK) to C++.
4. The generated C++ is compiled with clang and linked against the runtime (`runtime/`): HLE of the Switch services the game uses, and a Maxwell-shader-to-SPIR-V translator with an NVN layer on Vulkan.
5. Small romfs overlays are written from your own files (`tools/rstb_slack.py`, `tools/make_gameover_fix.py`, `tools/make_quickmenu_fix.py`) to fix problems above 30 fps.

## Status

- Runs on a Steam Deck in the author's tests: about 60 fps outdoors (62-64 fps measured on one walk).
- The first start compiles shaders. `tools/linux/warmup.sh` runs a shader warm-up (about 4.5 min on a fresh install on the Deck). Press A on its screen to start playing at once; the rest keeps compiling in the background.
- Only version 1.0.0 is supported (main build ID `082CE09B06E33A12`). Other versions will not work.
- **Build time is the big open issue.** The generated C++ is large. Measured on the author's PC with clang -O2: about 17 min wall time (about 2.1 h of CPU time over all cores). A build on the Deck itself has not been measured yet; expect 1-1.5 h. A much faster code generator is in progress and not part of this repository yet.
- The launcher (`launcher/`) does the setup steps (dump and keys pickers, version check, romfs extraction, settings). Building the game from inside the launcher is not wired up yet; until it is, run `build_from_xci.sh`.

## Requirements

- Your own *Tears of the Kingdom* 1.0.0 cartridge dump (`.xci`, title `0100F2C0115B6000`) and your own `prod.keys` (master key revision 0x0E).
- Linux x86-64 with AVX2 (Zen 2 / Haswell or newer), about 45 GB free disk at the peak (20 GB afterwards), 16 GB RAM.
- clang >= 17 with lld, ninja, cmake, make, gcc, curl, python3 with venv (capstone **exactly 5.0.9** is installed into a venv from PyPI if needed; other versions leave thousands of instructions unhandled), libzstd, Vulkan headers and libvulkan, static `libshaderc_combined.a` (plus glslang-dev and spirv-tools on Debian/Ubuntu) and `libvpx.a`, and the SDL3 build dependencies. The full Ubuntu 24.04 package list is in the header of [`tools/build_from_xci.sh`](tools/build_from_xci.sh).
- SteamOS has a read-only root file system: build in a distrobox with an Ubuntu 24.04 image, or on a Linux PC, and copy `dist/` over.

## Build

```bash
./tools/build_from_xci.sh /path/to/TearsOfTheKingdom.xci /path/to/prod.keys --out ~/totk-build
```

It builds hactool 1.4.0, SDL3 3.2.30 and Opus 1.5.2 from their official release sources (SHA-256 pinned in `tools/linux/build_deps.sh`), extracts the game, recompiles it, compiles everything and assembles `~/totk-build/dist/totk-recomp`. A filtered temporary copy of your keys exists in the work folder only while hactool runs.

Useful options: `--jobs N`, `--gen-jobs N` (generated files compiled at once; RAM is the limit, about 1.3-2 GB each), `--gen-opt -O1` (faster compile, slower code), `--skip-extract` (reuse an earlier extraction), `--prepare-only` (stop before compiling). The build can be interrupted and re-run; finished steps are reused. `--help` lists everything.

## Run

```bash
~/totk-build/dist/totk-recomp/run.sh
```

Add `run.sh` to Steam as a non-Steam game for Gaming Mode. Saves and caches go to `~/.local/share/totk-recomp`. Settings are environment variables read by `run.sh` (see its header), for example `TOTK_FPS` (90 default, 60, 30, or 0 for the game's own 30 fps pacing) and `TOTK_UNLOCK_FPS=1` (no frame limiter).

## Launcher

`tools/linux/launcher_deps.sh` fetches Dear ImGui 1.91.8, stb_image and the Inter font from their official sources; `tools/linux/launcher_build.sh` builds `launcher/launcher.cpp` against the SDL3 from the game build. Both default to `/root/totk-linux/deps` for the third-party folder: pass it as the argument of `launcher_deps.sh` and as `DEPS=...` (plus `OBJ=...` for the object folder) to `launcher_build.sh`, whose argument is the output folder.

## Layout

| Path | What |
|---|---|
| `tools/recomp/gen.py` | AArch64 to C++ recompiler |
| `tools/recomp/mkninja_linux.py` | writes the ninja build |
| `tools/build_from_xci.sh` | the whole build from a dump |
| `runtime/` | runtime: CPU state, memory, Switch service HLE, audio, input, OSD |
| `runtime/nvn/` | NVN graphics layer on Vulkan, Maxwell shader translator |
| `launcher/` | setup launcher (Dear ImGui + SDL3) |
| `tools/linux/` | run, warm-up, dependency and launcher build scripts |
| `third_party/` | Opus headers, SMAA |

## Credits and licenses

Our code is under the MIT license (see [LICENSE](LICENSE)). Included third-party code keeps its own license:

- NVN-on-Vulkan layer ported from **LAS Recomp** (MIT, Peiikai): `runtime/nvn/LICENSE.LASRecomp`
- **SMAA** (MIT, Jorge Jimenez et al.): `third_party/smaa/LICENSE.txt`, `runtime/nvn/smaa/LICENSE.txt`
- **Opus** headers (BSD-3-Clause, Xiph.Org): `third_party/opus/COPYING`
- **stb_truetype** (MIT or public domain, Sean Barrett): `runtime/linux/stb_truetype.h`

Downloaded and built at build time, under their own licenses: SDL3 (zlib), Opus (BSD-3-Clause), hactool (ISC), capstone (BSD-3-Clause), Dear ImGui (MIT), stb (MIT / public domain), Inter (SIL OFL 1.1).

Hardware and OS behaviour was checked against the documentation and behaviour of the Eden, yuzu and Ryujinx emulators and Atmosphère's cheat format. The game-over menu fix follows the one found by MacNativeGames (MIT).

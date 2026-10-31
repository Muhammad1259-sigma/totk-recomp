#!/bin/bash
# Builds the launcher in WSL (Ubuntu 24.04, same glibc floor as the game build) against the bundled SDL3.
# usage: launcher_build.sh [out dir]   (default /root/totk-linux/launcher-out)
#   out/launcher, out/launcher-data/Inter-*.ttf, out/launcher-data/LICENSES.txt   (libSDL3.so.0 ships next to the game)
set -euo pipefail
DEPS="${DEPS:-/root/totk-linux/deps}"
L=$DEPS/launcher
IM=$L/imgui-1.91.8
SDL=$DEPS/sdl3-install
OUT="${1:-/root/totk-linux/launcher-out}"
SRC="$(cd "$(dirname "$(readlink -f "$0")")/../../launcher" && pwd)"   # this tree's launcher/
OBJ="${OBJ:-/root/totk-linux/launcher-obj}"
mkdir -p "$OUT/launcher-data" "$OBJ"
CXX="${CXX:-g++}"
FLAGS="-O2 -std=c++17 -I$IM -I$IM/backends -I$L -I$SDL/include -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS"
objs=()
for f in $IM/imgui.cpp $IM/imgui_draw.cpp $IM/imgui_tables.cpp $IM/imgui_widgets.cpp \
         $IM/backends/imgui_impl_sdl3.cpp $IM/backends/imgui_impl_sdlrenderer3.cpp; do
  o=$OBJ/$(basename "$f" .cpp).o
  [ "$o" -nt "$f" ] || $CXX $FLAGS -c "$f" -o "$o" &
  objs+=("$o")
done
wait
$CXX $FLAGS -Wall -Wno-format-truncation -c $SRC/launcher.cpp -o $OBJ/launcher.o
$CXX -o "$OUT/launcher" $OBJ/launcher.o "${objs[@]}" -L$SDL/lib -lSDL3 -Wl,-rpath,'$ORIGIN' -Wl,-rpath,'$ORIGIN/lib' -lpthread
strip "$OUT/launcher"
for w in Regular Medium SemiBold Bold ExtraBold; do cp -u $L/inter/extras/ttf/Inter-$w.ttf "$OUT/launcher-data/"; done
{ echo "Inter (rsms/inter 4.1): SIL Open Font License 1.1"; cat $L/inter/LICENSE.txt 2>/dev/null || true; echo
  echo "Dear ImGui 1.91.8: MIT"; cat $IM/LICENSE.txt; echo
  echo "stb_image / stb_image_write (nothings/stb): public domain (Unlicense) or MIT"; } > "$OUT/launcher-data/LICENSES.txt"
cp -u $SDL/lib/libSDL3.so.0 "$OUT/" 2>/dev/null || cp -L $SDL/lib/libSDL3.so.0 "$OUT/"
ls -la "$OUT" "$OUT/launcher-data"

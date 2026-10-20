#!/bin/bash
# Third-party pieces of the launcher (official sources, checksums printed for pinning):
#   Dear ImGui 1.91.8 (ocornut/imgui tag, MIT), stb_image / stb_image_write (nothings/stb, public domain),
#   Inter 4.1 font (rsms/inter release, SIL OFL 1.1)
# usage: launcher_deps.sh [deps dir]   (default /root/totk-linux/deps)
set -euo pipefail
D="${1:-/root/totk-linux/deps}/launcher"
mkdir -p "$D" && cd "$D"
[ -f imgui-1.91.8.tar.gz ] || curl -fsSL -o imgui-1.91.8.tar.gz https://github.com/ocornut/imgui/archive/refs/tags/v1.91.8.tar.gz
[ -d imgui-1.91.8 ] || tar xf imgui-1.91.8.tar.gz
for f in stb_image.h stb_image_write.h; do
  [ -f $f ] || curl -fsSL -o $f https://raw.githubusercontent.com/nothings/stb/master/$f
done
[ -f Inter-4.1.zip ] || curl -fsSL -o Inter-4.1.zip https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
[ -d inter ] || python3 -c "import zipfile; zipfile.ZipFile('Inter-4.1.zip').extractall('inter')"
sha256sum imgui-1.91.8.tar.gz stb_image.h stb_image_write.h Inter-4.1.zip
ls imgui-1.91.8/backends | grep -i sdl3
find inter -iname "*.ttf" | head -20

#!/usr/bin/env bash
# Build the third-party pieces of the Linux build from their official GitHub releases (checksum-pinned):
#   SDL3 3.2.30   libsdl-org/SDL   shared (X11 + Wayland + PipeWire / PulseAudio / ALSA + udev / HIDAPI / D-Bus + Vulkan), bundled as lib/libSDL3.so.0
#   Opus 1.5.2    xiph/opus        static (hle_codec.cpp: the game's hardware Opus decoder)
#   Vulkan-Headers 1.4.357  KhronosGroup/Vulkan-Headers  headers only (VK_KHR_unified_image_layouts etc. are newer than distro headers)
#   hactool 1.4.0 SciresM/hactool  only with --hactool (build_from_xci.sh: NCA / romfs extraction)
# usage: build_deps.sh <deps dir> [--hactool] [--no-libs] [--jobs N]   (--no-libs: skip SDL3 and Opus)
# The system provides (Debian / Ubuntu names): clang lld ninja-build cmake make gcc curl libvulkan-dev libshaderc-dev (static
# libshaderc_combined.a) libvpx-dev (static libvpx.a) libx11-dev libwayland-dev libxkbcommon-dev libudev-dev libasound2-dev
# libpulse-dev libpipewire-0.3-dev libdbus-1-dev libdrm-dev libgbm-dev libegl-dev libxext-dev libxcursor-dev libxi-dev libxrandr-dev ...
set -euo pipefail
DEPS="${1:?usage: build_deps.sh <deps dir> [--hactool] [--jobs N]}"; shift
WANT_HACTOOL=0; WANT_LIBS=1; JOBS="$(nproc)"
while [ $# -gt 0 ]; do
  case "$1" in --hactool) WANT_HACTOOL=1 ;; --no-libs) WANT_LIBS=0 ;; --jobs) JOBS="$2"; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
  shift
done
CC="${CC:-clang}"; CXX="${CXX:-clang++}"
command -v "$CC" >/dev/null || { CC=clang-19; CXX=clang++-19; }
mkdir -p "$DEPS"; DEPS="$(cd "$DEPS" && pwd)"

fetch() {   # url sha256 file
  if [ ! -f "$DEPS/$3" ]; then curl -fL --retry 3 -o "$DEPS/$3.part" "$1" && mv "$DEPS/$3.part" "$DEPS/$3"; fi
  echo "$2  $DEPS/$3" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $3" >&2; exit 1; }
}

if [ "$WANT_LIBS" = 1 ] && [ ! -f "$DEPS/sdl3-install/lib/libSDL3.so.0" ]; then
  echo "== SDL3 3.2.30"
  fetch https://github.com/libsdl-org/SDL/releases/download/release-3.2.30/SDL3-3.2.30.tar.gz \
        4c3b09330d866dc52eb65b66259a6684ad387252ca8c57901b3a2b534eb42e3d SDL3-3.2.30.tar.gz
  rm -rf "$DEPS/SDL3-3.2.30" "$DEPS/sdl3-build"; tar -C "$DEPS" -xf "$DEPS/SDL3-3.2.30.tar.gz"
  CC="$CC" CXX="$CXX" cmake -G Ninja -S "$DEPS/SDL3-3.2.30" -B "$DEPS/sdl3-build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEPS/sdl3-install" -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=znver2" \
    -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF -DSDL_INSTALL_TESTS=OFF \
    -DSDL_X11=ON -DSDL_WAYLAND=ON -DSDL_PIPEWIRE=ON -DSDL_PULSEAUDIO=ON -DSDL_ALSA=ON \
    -DSDL_HIDAPI=ON -DSDL_HIDAPI_LIBUSB=OFF -DSDL_LIBUDEV=ON -DSDL_DBUS=ON -DSDL_VULKAN=ON \
    -DSDL_KMSDRM=ON -DSDL_OPENGL=ON -DSDL_OPENGLES=ON -DSDL_JACK=OFF -DSDL_SNDIO=OFF -DSDL_OSS=OFF -DSDL_UNIX_CONSOLE_BUILD=OFF
  ninja -C "$DEPS/sdl3-build" -j"$JOBS" install
fi

if [ "$WANT_LIBS" = 1 ] && [ ! -f "$DEPS/opus-install/lib/libopus.a" ]; then
  echo "== Opus 1.5.2"
  fetch https://github.com/xiph/opus/releases/download/v1.5.2/opus-1.5.2.tar.gz \
        65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1 opus-1.5.2.tar.gz
  rm -rf "$DEPS/opus-1.5.2"; tar -C "$DEPS" -xf "$DEPS/opus-1.5.2.tar.gz"
  CC="$CC" CFLAGS="-O2 -march=x86-64-v3 -mtune=znver2 -fPIC" cmake -G Ninja -S "$DEPS/opus-1.5.2" -B "$DEPS/opus-1.5.2/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF \
    -DOPUS_INSTALL_PKG_CONFIG_MODULE=OFF -DOPUS_INSTALL_CMAKE_CONFIG_MODULE=OFF -DCMAKE_INSTALL_PREFIX="$DEPS/opus-install"
  ninja -C "$DEPS/opus-1.5.2/build" -j"$JOBS" install
fi

if [ "$WANT_LIBS" = 1 ] && [ ! -f "$DEPS/vulkan-headers/include/vulkan/vulkan_core.h" ]; then
  echo "== Vulkan-Headers 1.4.357"
  fetch https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.357.tar.gz         7dc0dbcf1d49dd3d7da3761c251c6097dfbaac475321a4a8a99269d3d5abecdc Vulkan-Headers-1.4.357.tar.gz
  rm -rf "$DEPS/vulkan-headers"; mkdir -p "$DEPS/vulkan-headers"
  tar -C "$DEPS/vulkan-headers" --strip-components=1 -xf "$DEPS/Vulkan-Headers-1.4.357.tar.gz" Vulkan-Headers-1.4.357/include
fi

if [ "$WANT_HACTOOL" = 1 ] && [ ! -x "$DEPS/hactool/hactool" ]; then
  echo "== hactool 1.4.0"
  fetch https://github.com/SciresM/hactool/archive/refs/tags/1.4.0.tar.gz \
        1373c98ef8c58da42cb1e81b361283955caad0e1b8b5ad389b0ee49b7a21da43 hactool-1.4.0.tar.gz
  rm -rf "$DEPS/hactool-src"; mkdir -p "$DEPS/hactool-src"; tar -C "$DEPS/hactool-src" --strip-components=1 -xf "$DEPS/hactool-1.4.0.tar.gz"
  ( cd "$DEPS/hactool-src" && cp config.mk.template config.mk && make -j"$JOBS" CC="${HACTOOL_CC:-gcc}" )
  mkdir -p "$DEPS/hactool"; cp "$DEPS/hactool-src/hactool" "$DEPS/hactool/hactool"
fi
echo "deps ready in $DEPS"

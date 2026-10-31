#!/bin/bash
# Launcher "Extract from the dump": the game files (romfs) from the player's own XCI / NSP into <data>/romfs.
# usage: extract_romfs.sh <dump.xci|dump.nsp> <filtered keys file> <data dir> <hactool>
# Peak disk use ~ program NCA (16 GB) + romfs (17 GB); the NCA copy is deleted at the end.
set -euo pipefail
DUMP="$1"; KF="$2"; DATA="$3"; HAC="$4"
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
W="$DATA/.extract"
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT
echo "Reading the dump..."
case "${DUMP,,}" in
  *.xci)
    python3 -I "$here/xci_carve.py" "$DUMP" "$W/x" > "$W/list.txt"
    NCA="$(grep -E '\.nca off=' "$W/list.txt" | grep -v '\.cnmt\.nca' | sed -E 's/^(\S+) off=\S+ size=([0-9]+)$/\2 \1/' | sort -rn | head -1 | cut -d' ' -f2)"
    [ -n "$NCA" ] || { echo "no program NCA in the dump"; exit 1; }
    echo "Copying the program data ($NCA, about 16 GB)..."
    python3 -I "$here/xci_carve.py" "$DUMP" "$W/x" "$NCA=all" > /dev/null
    SRC="$W/x/$NCA" ;;
  *.nsp)
    echo "Unpacking the NSP (about 16 GB)..."
    "$HAC" -t pfs0 --pfs0dir="$W/x" "$DUMP" > "$W/pfs0.log" 2>&1 || { tail -5 "$W/pfs0.log"; exit 1; }
    SRC="$(ls -S "$W"/x/*.nca | grep -v '\.cnmt\.nca' | head -1)" ;;
  *) echo "not an .xci or .nsp file"; exit 1 ;;
esac
echo "Extracting the game files (17 GB, this takes a while)..."
rm -rf "$DATA/romfs.partial"; mkdir -p "$DATA/romfs.partial"
"$HAC" -k "$KF" -t nca --romfsdir="$DATA/romfs.partial" "$SRC" > "$W/hactool.log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 5
  echo "Extracting the game files: $(du -sm "$DATA/romfs.partial" | cut -f1) MB of ~17000"
done
wait $pid || { tail -5 "$W/hactool.log"; exit 1; }
[ -f "$DATA/romfs.partial/System/Resource/ResourceSizeTable.Product.100.rsizetable.zs" ] || { echo "extraction incomplete"; tail -5 "$W/hactool.log"; exit 1; }
rm -rf "$DATA/romfs"; mv "$DATA/romfs.partial" "$DATA/romfs"
echo "Done: $(du -sh "$DATA/romfs" | cut -f1) of game files."

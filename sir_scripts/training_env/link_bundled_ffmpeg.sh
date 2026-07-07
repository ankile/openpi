#!/bin/bash
# Make torchcodec (lerobot[dataset] video decode) work on a host with NO system FFmpeg
# (e.g. delta) by exposing PyAV's bundled FFmpeg 7 under the standard sonames torchcodec
# dlopens. Self-contained: reuses the ffmpeg already inside the locked venv — no system
# install, no modules, no extra downloads. iris finds system ffmpeg and does not need this.
#
# Creates $OPENPI_DIR/.ffmpeg-compat/lib{av,sw}*.so.<major> -> venv av.libs/<hashed> and
# prints the LD_LIBRARY_PATH the launch script sets on delta (.ffmpeg-compat + av.libs;
# the second entry resolves the ffmpeg libs' hashed transitive deps like libdrm).
#
# Usage: run from the openpi checkout root (the dir with .venv), or pass it as $1.
set -euo pipefail
OPENPI="${1:-$(pwd)}"
VENV="$OPENPI/.venv"
AVLIBS="$VENV/lib/python3.12/site-packages/av.libs"
COMPAT="$OPENPI/.ffmpeg-compat"
[ -d "$AVLIBS" ] || { echo "FATAL: $AVLIBS not found (is the venv synced with lerobot[dataset]?)"; exit 1; }
rm -rf "$COMPAT"; mkdir -p "$COMPAT"
shopt -s nullglob
n=0
for f in "$AVLIBS"/lib{av,sw}*.so.*; do
  base=$(basename "$f"); name=${base%%-*}; ver=${base#*.so.}; major=${ver%%.*}
  ln -sf "$f" "$COMPAT/${name}.so.${major}"; n=$((n+1))
done
echo "linked $n bundled ffmpeg libs into $COMPAT"
echo "LD_LIBRARY_PATH=$COMPAT:$AVLIBS"
# Verify torchcodec actually loads.
LD_LIBRARY_PATH="$COMPAT:$AVLIBS:${LD_LIBRARY_PATH:-}" "$VENV/bin/python" -c \
  "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec', torchcodec.__version__, 'OK')"

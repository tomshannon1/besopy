#!/usr/bin/env bash
# Michell-style 3D space truss: four pinned base corners and a point load
# at the centre of the roof. Low volume, so members form instead of a solid.
# A couple of minutes on a laptop CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
beso \
  --problem truss \
  --load-placement center \
  --nelx 24 --nely 16 --nelz 16 \
  --length 24 --width 16 --height 16 \
  --volume 0.12 \
  --load -1 \
  --evolution-rate 0.02 \
  --filter-mode node --filter-radius 1.8 \
  --max-iterations 120 \
  --snapshot-every 1 \
  --surface-sigma 0.55 --surface-smoothing-iterations 6 \
  --output output/examples/truss
echo "Playback: python -m http.server --directory output/examples/truss 8000"

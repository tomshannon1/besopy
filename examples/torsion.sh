#!/usr/bin/env bash
# Torsion tower: four pinned base corners and a roof force couple.
# The couple twists the box into a helix. Low volume, a couple of minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
beso \
  --problem torsion \
  --load-placement couple \
  --nelx 12 --nely 12 --nelz 24 \
  --length 12 --width 12 --height 24 \
  --volume 0.20 \
  --load -1 \
  --evolution-rate 0.02 \
  --filter-mode node --filter-radius 1.8 \
  --max-iterations 120 \
  --snapshot-every 1 \
  --surface-sigma 0.55 --surface-smoothing-iterations 6 \
  --output output/examples/torsion
echo "Playback: python -m http.server --directory output/examples/torsion 8000"

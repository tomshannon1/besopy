#!/usr/bin/env bash
# Thesis-scale 3D cantilever: wall at x=0, uniform load on the free-end
# bottom edge. About 20 minutes on a laptop CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
beso \
  --problem cantilever \
  --nelx 40 --nely 12 --nelz 20 \
  --length 40 --width 12 --height 20 \
  --volume 0.15 \
  --young-modulus 200e9 --poisson-ratio 0.4 \
  --load -1000 --load-placement bottom-edge \
  --evolution-rate 0.01 \
  --filter-mode node --filter-radius 1.5 \
  --max-iterations 250 \
  --snapshot-every 1 \
  --surface-sigma 0.7 --surface-smoothing-iterations 6 \
  --output output/examples/cantilever
echo "Playback: python -m http.server --directory output/examples/cantilever 8000"

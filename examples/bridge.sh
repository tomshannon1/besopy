#!/usr/bin/env bash
# Xie-style 3D bridge: abutments at both bottom ends, a live top-layer deck,
# and a three-axle convoy on the centre roadway. Forms an arch / truss
# between the piers. A few minutes on a laptop CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
beso \
  --problem bridge \
  --load-placement convoy \
  --nelx 32 --nely 8 --nelz 10 \
  --length 32 --width 8 --height 10 \
  --volume 0.30 \
  --load -1 \
  --evolution-rate 0.02 \
  --filter-mode node --filter-radius 1.5 \
  --max-iterations 80 \
  --snapshot-every 1 \
  --surface-sigma 0.6 --surface-smoothing-iterations 6 \
  --output output/examples/bridge
echo "Playback: python -m http.server --directory output/examples/bridge 8000"

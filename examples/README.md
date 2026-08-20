# Examples

Runnable BESO cases. They write VTK, a smoothed STL, and an HTML iteration
viewer under `output/examples/`. Serve the output folder over HTTP to play the
surface history.

```bash
source .venv/bin/activate
./examples/cantilever.sh
./examples/bridge.sh
```

## Cantilever

The thesis-scale 3D beam: `40 × 12 × 20`, 15% volume, node-filter radius 1.5.
The wall at `x = 0` is fully clamped. The total load is split equally across
the free-end **bottom-edge** nodes (`x = L`, `z = 0`). Expect about 20 minutes.

A faster smoke version:

```bash
beso --problem cantilever --nelx 12 --nely 4 --nelz 4 --volume 0.30 \
  --max-iterations 40 --snapshot-every 1 --output output/examples/cantilever-fast
```

## Bridge

A Xie-style simply supported span. Bottom end-edges are fixed abutments.
A **live deck** (the top element layer) stays solid so a roadway remains.
The default load is a **convoy**: three axle lines at quarter-span, mid-span,
and three-quarter-span on the centre lane. Connectivity is enforced from the
piers and the deck so the optimizer cannot drop the roadway. The typical
result is an arch or truss under traffic.

Other placements:

- `deck` — traffic spread along the whole centre lane
- `eccentric` — all traffic on one fascia (bending plus torsion)
- `midspan` — the textbook single axle line

```bash
beso --problem bridge --load-placement convoy \
  --nelx 16 --nely 4 --nelz 6 --volume 0.32 \
  --max-iterations 50 --snapshot-every 1 --output output/examples/bridge-fast
```

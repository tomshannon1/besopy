# Examples

Runnable BESO cases. They write VTK, a smoothed STL, and an HTML iteration
viewer under `output/examples/`. Serve the output folder over HTTP to play the
surface history. Looping GIFs of completed runs live in `docs/media/` and are
embedded in the project README.

```bash
source .venv/bin/activate
./examples/cantilever.sh
./examples/bridge.sh
./examples/truss.sh
./examples/torsion.sh
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

## Truss

A 3D Michell space frame. The four bottom corners are pinned; nothing else
on the base is held. The default load is a **centre** roof joint — one hanging
mass at the top of the box. Dead-end members are pruned, so the result is a
frame from the pins to the load rather than a solid block.

Other placements:

- `offset` — the load on one roof edge (bending plus torsion)
- `span` — two joints along the roof ridge

```bash
beso --problem truss --load-placement center \
  --nelx 12 --nely 8 --nelz 8 --volume 0.18 \
  --max-iterations 60 --snapshot-every 1 --output output/examples/truss-fast
```

## Torsion

A tall box on four pinned corners, with a **force couple** on a live roof
plate: two equal-and-opposite horizontal loads at opposite side-midpoints.
The optimizer usually answers with a helix under the cap.

Other placements:

- `diagonal` — the couple on opposite roof corners
- `spin` — four tangential loads around the roof

```bash
beso --problem torsion --load-placement couple \
  --nelx 8 --nely 8 --nelz 16 --volume 0.22 \
  --max-iterations 70 --snapshot-every 1 --output output/examples/torsion-fast
```

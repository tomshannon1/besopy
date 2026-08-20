from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import write_vtk
from .voxel import voxelize_stl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beso-import-stl",
        description="Convert a watertight STL into a structured BESO voxel domain.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    domain = voxelize_stl(args.input, args.pitch)
    args.output.mkdir(parents=True, exist_ok=True)

    write_vtk(args.output / "domain.vtk", domain.mesh, domain.occupied)
    write_vtk(
        args.output / "domain-solid.vtk",
        domain.mesh,
        domain.occupied,
        active_only=True,
    )
    np.save(
        args.output / "design-domain.npy",
        domain.occupied.astype(np.uint8),
        allow_pickle=False,
    )
    np.savez_compressed(
        args.output / "body-masks.npz",
        **{
            f"body_{index}": mask.astype(np.uint8)
            for index, mask in enumerate(domain.body_masks, start=1)
        },
    )
    summary = {
        "source": str(args.input),
        "source_vertices": domain.source_vertices,
        "source_faces": domain.source_faces,
        "source_bodies": domain.source_bodies,
        "bodies": [
            {
                "index": index,
                "bounds": bounds.tolist(),
                "source_volume": volume,
                "occupied_elements": int(mask.sum()),
            }
            for index, (bounds, volume, mask) in enumerate(
                zip(
                    domain.body_bounds,
                    domain.body_volumes,
                    domain.body_masks,
                    strict=True,
                ),
                start=1,
            )
        ],
        "source_watertight": domain.source_watertight,
        "source_volume": domain.source_volume,
        "source_bounds": domain.source_bounds.tolist(),
        "source_to_local_offset": domain.source_to_local_offset.tolist(),
        "pitch": domain.pitch,
        "mesh": {
            "elements": [
                domain.mesh.nelx,
                domain.mesh.nely,
                domain.mesh.nelz,
            ],
            "dimensions": [
                domain.mesh.length,
                domain.mesh.width,
                domain.mesh.height,
            ],
            "element_count": domain.mesh.num_elements,
            "node_count": domain.mesh.num_nodes,
            "dof_count": domain.mesh.num_dofs,
        },
        "occupied_elements": domain.occupied_elements,
        "occupied_fraction": domain.occupied_fraction,
        "voxelized_volume": domain.occupied_elements * domain.pitch**3,
    }
    (args.output / "import-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"mesh={domain.mesh.nelx}x{domain.mesh.nely}x{domain.mesh.nelz} "
        f"elements={domain.mesh.num_elements:,} "
        f"occupied={domain.occupied_elements:,} dofs={domain.mesh.num_dofs:,}"
    )
    print(f"wrote {args.output / 'domain-solid.vtk'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

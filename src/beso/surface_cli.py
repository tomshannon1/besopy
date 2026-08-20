from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mesh import StructuredMesh
from .surface import density_surface, read_vtk_densities, write_binary_stl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beso-surface",
        description="Convert a BESO voxel result into a smoothed printable STL.",
    )
    parser.add_argument("result", type=Path, help="BESO output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--smoothing-iterations", type=int, default=10)
    parser.add_argument(
        "--keep-fragments",
        action="store_true",
        help="Keep detached components instead of retaining only the largest",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    summary = json.loads((args.result / "summary.json").read_text(encoding="utf-8"))
    elements = summary["mesh"]["elements"]
    dimensions = summary["mesh"]["dimensions"]
    mesh = StructuredMesh(
        *elements,
        length=dimensions[0],
        width=dimensions[1],
        height=dimensions[2],
    )
    densities = read_vtk_densities(args.result / "design.vtk", mesh.num_elements)
    surface = density_surface(
        mesh,
        densities,
        sigma=args.sigma,
        level=args.level,
        smoothing_iterations=args.smoothing_iterations,
        keep_largest_component=not args.keep_fragments,
    )
    output = args.output or args.result / "design-smoothed.stl"
    write_binary_stl(output, surface)

    size = np.ptp(surface.bounds, axis=0)
    print(
        f"vertices={len(surface.vertices):,} triangles={len(surface.faces):,} "
        f"watertight={surface.is_watertight}"
    )
    print(f"size={size[0]:.3f}x{size[1]:.3f}x{size[2]:.3f}")
    print(f"wrote {output}")
    return 0 if surface.is_watertight else 2


if __name__ == "__main__":
    raise SystemExit(main())

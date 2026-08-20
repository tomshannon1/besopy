from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .fea import FiniteElementSolver
from .history import SurfaceHistory, build_overlay
from .io import write_vtk
from .mesh import StructuredMesh
from .optimizer import BESOOptimizer, IterationRecord
from .surface import density_surface, write_binary_stl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beso-optimize-domain",
        description="Optimize an imported voxel design domain.",
    )
    parser.add_argument("imported", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume", type=float, default=0.7)
    parser.add_argument("--protect-body", type=int, action="append", default=[])
    parser.add_argument("--protect-bottom-layers", type=int, default=2)
    parser.add_argument("--load-body", type=int, required=True)
    parser.add_argument("--vertical-load", type=float, default=-1.0)
    parser.add_argument("--lateral-load", type=float, default=0.3)
    parser.add_argument("--young-modulus", type=float, default=1.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--evolution-rate", type=float, default=0.02)
    parser.add_argument("--filter-radius", type=float, default=1.5)
    parser.add_argument("--min-thickness", type=int, default=1)
    parser.add_argument(
        "--keep-connected",
        action="store_true",
        help="Delete islands that are not connected to supports or protected solids",
    )
    parser.add_argument(
        "--prune-dead-ends",
        action="store_true",
        help="Remove dangling one-voxel-wide chains that do not carry load",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--surface-sigma", type=float, default=0.7)
    parser.add_argument("--surface-smoothing-iterations", type=int, default=10)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Write a smoothed STL every N iterations (0 disables history export)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _mesh_from_summary(summary: dict) -> StructuredMesh:
    elements = summary["mesh"]["elements"]
    dimensions = summary["mesh"]["dimensions"]
    return StructuredMesh(
        *elements,
        length=dimensions[0],
        width=dimensions[1],
        height=dimensions[2],
    )


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    started_at = time.perf_counter()
    import_summary = json.loads(
        (args.imported / "import-summary.json").read_text(encoding="utf-8")
    )
    mesh = _mesh_from_summary(import_summary)
    domain = np.load(args.imported / "design-domain.npy", allow_pickle=False).astype(
        bool
    )
    body_archive = np.load(args.imported / "body-masks.npz", allow_pickle=False)

    def body_mask(index: int) -> np.ndarray:
        key = f"body_{index}"
        if key not in body_archive:
            raise ValueError(f"Imported geometry has no body {index}")
        return body_archive[key].astype(bool)

    passive = np.zeros(mesh.num_elements, dtype=bool)
    for index in args.protect_body:
        passive |= body_mask(index)
    grid_shape = (mesh.nelx, mesh.nely, mesh.nelz)
    if args.protect_bottom_layers > 0:
        bottom = np.zeros(grid_shape, dtype=bool)
        bottom[:, :, : args.protect_bottom_layers] = domain.reshape(grid_shape)[
            :, :, : args.protect_bottom_layers
        ]
        passive |= bottom.reshape(-1)

    active_nodes = np.unique(mesh.connectivity[domain])
    inactive_nodes = np.setdiff1d(
        np.arange(mesh.num_nodes, dtype=np.int64),
        active_nodes,
        assume_unique=True,
    )
    bottom_elements = np.zeros(grid_shape, dtype=bool)
    bottom_elements[:, :, 0] = domain.reshape(grid_shape)[:, :, 0]
    bottom_nodes = np.unique(mesh.connectivity[bottom_elements.reshape(-1)])
    contact_nodes = bottom_nodes[np.isclose(mesh.coordinates[bottom_nodes, 2], 0.0)]
    fixed_nodes = np.unique(np.concatenate((inactive_nodes, contact_nodes)))
    fixed_dofs = (3 * fixed_nodes[:, None] + np.arange(3, dtype=np.int64)).reshape(-1)

    load_nodes = np.unique(mesh.connectivity[body_mask(args.load_body)])
    load_nodes = np.intersect1d(load_nodes, active_nodes, assume_unique=True)
    forces = np.zeros(mesh.num_dofs, dtype=float)
    forces[3 * load_nodes] = args.lateral_load / len(load_nodes)
    forces[3 * load_nodes + 2] = args.vertical_load / len(load_nodes)

    solver = FiniteElementSolver(
        mesh,
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
    )
    optimizer = BESOOptimizer(
        solver,
        args.volume,
        evolution_rate=args.evolution_rate,
        filter_radius=args.filter_radius,
        max_iterations=args.max_iterations,
        design_domain=domain,
        passive_solid=passive,
        connectivity_seeds=passive | bottom_elements.reshape(-1),
        min_thickness_voxels=args.min_thickness,
        require_connectivity=args.keep_connected,
        prune_dead_ends=args.prune_dead_ends,
    )
    recent_durations: deque[float] = deque(maxlen=5)
    args.output.mkdir(parents=True, exist_ok=True)
    history = SurfaceHistory(
        args.output,
        mesh,
        every=args.snapshot_every,
        sigma=args.surface_sigma,
        smoothing_iterations=args.surface_smoothing_iterations,
        overlay=build_overlay(
            mesh,
            forces=forces,
            sensitivity_kind="element",
            support_points=mesh.coordinates[contact_nodes],
            max_arrows=12,
        ),
    )

    def report(
        record: IterationRecord,
        densities: object,
        sensitivities: object = None,
    ) -> None:
        history.capture(
            record.iteration,
            np.asarray(densities, dtype=float),
            volume_fraction=record.volume_fraction,
            record=record,
            sensitivities=(
                None
                if sensitivities is None
                else np.asarray(sensitivities, dtype=float)
            ),
        )
        if args.quiet:
            return
        recent_durations.append(record.duration_seconds)
        mean_duration = sum(recent_durations) / len(recent_durations)
        eta = mean_duration * (args.max_iterations - record.iteration)
        print(
            f"iter={record.iteration:03d}/{args.max_iterations:03d} "
            f"volume={record.volume_fraction:.4f} "
            f"compliance={record.compliance:.6g} "
            f"changed={record.changed_elements} "
            f"step={record.duration_seconds:.2f}s eta={eta:.1f}s",
            flush=True,
        )

    if not args.quiet:
        print(
            f"domain={int(domain.sum()):,} passive={int(passive.sum()):,} "
            f"active-dofs={3 * len(active_nodes):,} "
            f"contact-nodes={len(contact_nodes):,} load-nodes={len(load_nodes):,}",
            flush=True,
        )
    history.capture(
        0,
        domain.astype(float),
        volume_fraction=1.0,
    )
    result = optimizer.run(
        forces,
        fixed_dofs,
        initial_densities=domain.astype(float),
        callback=report,
    )
    if history.enabled and result.history:
        last = result.history[-1]
        history.capture(
            last.iteration,
            result.densities,
            volume_fraction=last.volume_fraction,
            record=last,
            force=True,
        )

    write_vtk(args.output / "design.vtk", mesh, result.densities)
    write_vtk(
        args.output / "design-solid.vtk",
        mesh,
        result.densities,
        active_only=True,
    )
    surface = density_surface(
        mesh,
        result.densities,
        sigma=args.surface_sigma,
        smoothing_iterations=args.surface_smoothing_iterations,
    )
    write_binary_stl(args.output / "design-smoothed.stl", surface)
    elapsed = time.perf_counter() - started_at
    initial_compliance = result.history[0].compliance
    summary = {
        "scenario": "combined vertical and lateral landing load",
        "converged": result.converged,
        "iterations": len(result.history),
        "elapsed_seconds": elapsed,
        "target_domain_volume_fraction": args.volume,
        "domain_elements": int(domain.sum()),
        "retained_elements": int(result.densities[domain].sum()),
        "retained_domain_volume_fraction": float(result.densities[domain].mean()),
        "material_reduction_percent": 100.0
        * (1.0 - float(result.densities[domain].mean())),
        "passive_elements": int(passive.sum()),
        "min_thickness_voxels": args.min_thickness,
        "require_connectivity": args.keep_connected,
        "prune_dead_ends": args.prune_dead_ends,
        "contact_nodes": len(contact_nodes),
        "load_nodes": len(load_nodes),
        "loads": {
            "vertical": args.vertical_load,
            "lateral_x": args.lateral_load,
        },
        "compliance": result.analysis.compliance,
        "initial_compliance": initial_compliance,
        "compliance_change_percent": 100.0
        * (result.analysis.compliance / initial_compliance - 1.0),
        "watertight_stl": surface.is_watertight,
        "surface": {
            "sigma": args.surface_sigma,
            "smoothing_iterations": args.surface_smoothing_iterations,
        },
        "snapshot_every": args.snapshot_every,
        "history_frames": len(history.frames),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if history.frames and not args.quiet:
        print(
            f"history frames={len(history.frames)} viewer={args.output / 'index.html'}",
            flush=True,
        )
    if not args.quiet:
        print(
            f"finished elapsed={elapsed:.1f}s "
            f"volume={summary['retained_domain_volume_fraction']:.4f} "
            f"compliance={result.analysis.compliance:.6g} "
            f"watertight={surface.is_watertight}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

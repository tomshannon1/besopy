from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

from .fea import FiniteElementSolver
from .io import write_vtk
from .mesh import StructuredMesh
from .multiresolution import coarsen_shape, expand_design, resample_densities
from .optimizer import BESOOptimizer, IterationRecord
from .problems import cantilever_load_case


def _format_duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beso",
        description="Run 3D BESO topology optimization on a cantilever beam.",
    )
    parser.add_argument("--nelx", type=int, default=24)
    parser.add_argument("--nely", type=int, default=6)
    parser.add_argument("--nelz", type=int, default=6)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--volume", type=float, default=0.2)
    parser.add_argument("--evolution-rate", type=float, default=0.02)
    parser.add_argument("--filter-radius", type=float, default=1.5)
    parser.add_argument(
        "--filter-mode",
        choices=("element", "node"),
        default="element",
        help="Element filter uses voxel units; node filter uses physical units",
    )
    parser.add_argument("--young-modulus", type=float, default=1.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--load", type=float, default=-1.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--coarse-factor",
        type=int,
        help="Warm-start from a mesh coarsened by this integer factor",
    )
    parser.add_argument(
        "--warm-start-volume",
        type=float,
        help="Fine-mesh starting volume; defaults to twice the target volume",
    )
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    total_started_at = time.perf_counter()
    args = build_parser().parse_args(arguments)

    def emit(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    mesh = StructuredMesh(
        args.nelx,
        args.nely,
        args.nelz,
        length=args.length,
        width=args.width,
        height=args.height,
    )
    emit(
        f"mesh={mesh.nelx}x{mesh.nely}x{mesh.nelz} "
        f"elements={mesh.num_elements:,} nodes={mesh.num_nodes:,} "
        f"dofs={mesh.num_dofs:,} target-volume={args.volume:.3f} "
        f"max-iterations={args.max_iterations}"
    )

    def make_report(phase: str, max_iterations: int):
        recent_durations: deque[float] = deque(maxlen=5)

        def report(record: IterationRecord, _densities: object) -> None:
            recent_durations.append(record.duration_seconds)
            average_duration = sum(recent_durations) / len(recent_durations)
            remaining_iterations = max_iterations - record.iteration
            eta = average_duration * remaining_iterations
            best = (
                ""
                if record.best_near_target_compliance is None
                else f" best-near={record.best_near_target_compliance:.6g}"
            )
            prefix = f"phase={phase} " if args.coarse_factor is not None else ""
            emit(
                f"{prefix}iter={record.iteration:03d}/{max_iterations:03d} "
                f"volume={record.volume_fraction:.4f} "
                f"compliance={record.compliance:.6g}{best} "
                f"changed={record.changed_elements} "
                f"assembly={_format_duration(record.assembly_seconds)} "
                f"solve={_format_duration(record.linear_solve_seconds)} "
                f"post={_format_duration(record.postprocess_seconds)} "
                f"update={_format_duration(record.update_seconds)} "
                f"step={_format_duration(record.duration_seconds)} "
                f"elapsed={_format_duration(record.elapsed_seconds)} "
                f"eta={_format_duration(eta)}"
            )

        return report

    initial_densities = None
    coarse_mesh = None
    coarse_result = None
    if args.coarse_factor is not None:
        coarse_elements = coarsen_shape(mesh, args.coarse_factor)
        coarse_mesh = StructuredMesh(
            *coarse_elements,
            length=args.length,
            width=args.width,
            height=args.height,
        )
        emit(
            f"coarse-mesh={coarse_mesh.nelx}x{coarse_mesh.nely}x{coarse_mesh.nelz} "
            f"elements={coarse_mesh.num_elements:,} dofs={coarse_mesh.num_dofs:,}"
        )
        emit("starting coarse optimization...")
        coarse_load_case = cantilever_load_case(coarse_mesh, load=args.load)
        coarse_solver = FiniteElementSolver(
            coarse_mesh,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
        )
        coarse_optimizer = BESOOptimizer(
            coarse_solver,
            args.volume,
            evolution_rate=args.evolution_rate,
            filter_radius=args.filter_radius,
            filter_mode=args.filter_mode,
            max_iterations=args.max_iterations,
        )
        coarse_result = coarse_optimizer.run(
            coarse_load_case.forces,
            coarse_load_case.fixed_dofs,
            callback=make_report("coarse", args.max_iterations),
        )
        write_vtk(
            args.output / "coarse-design-solid.vtk",
            coarse_mesh,
            coarse_result.densities,
            active_only=True,
        )
        initial_densities = resample_densities(
            coarse_mesh, mesh, coarse_result.densities
        )
        transferred_volume = float(initial_densities.mean())
        warm_start_volume = (
            min(1.0, 2.0 * args.volume)
            if args.warm_start_volume is None
            else args.warm_start_volume
        )
        if not args.volume <= warm_start_volume <= 1.0:
            raise ValueError("Warm-start volume must be between target volume and one")
        initial_densities = expand_design(mesh, initial_densities, warm_start_volume)
        emit(
            f"transferred-volume={transferred_volume:.4f} "
            f"warm-start-volume={initial_densities.mean():.4f} "
            f"coarse-time={_format_duration(coarse_result.elapsed_seconds)}"
        )

    emit("starting fine optimization...")
    load_case = cantilever_load_case(mesh, load=args.load)
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
        filter_mode=args.filter_mode,
        max_iterations=args.max_iterations,
    )

    result = optimizer.run(
        load_case.forces,
        load_case.fixed_dofs,
        initial_densities=initial_densities,
        callback=make_report("fine", args.max_iterations),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_vtk(args.output / "design.vtk", mesh, result.densities)
    write_vtk(
        args.output / "design-solid.vtk",
        mesh,
        result.densities,
        active_only=True,
    )
    total_elapsed_seconds = time.perf_counter() - total_started_at

    def timing_totals(history: tuple[IterationRecord, ...]) -> dict[str, float]:
        return {
            "assembly_seconds": sum(record.assembly_seconds for record in history),
            "linear_solve_seconds": sum(
                record.linear_solve_seconds for record in history
            ),
            "postprocess_seconds": sum(
                record.postprocess_seconds for record in history
            ),
            "update_seconds": sum(record.update_seconds for record in history),
        }

    summary = {
        "mode": "multiresolution" if coarse_result is not None else "single-resolution",
        "filter": {
            "mode": args.filter_mode,
            "radius": args.filter_radius,
            "radius_units": (
                "physical" if args.filter_mode == "node" else "minimum-element-size"
            ),
        },
        "converged": result.converged,
        "iterations": len(result.history),
        "optimization_elapsed_seconds": result.elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "timings": timing_totals(result.history),
        "volume_fraction": float(result.densities.mean()),
        "compliance": result.analysis.compliance,
        "tip_displacement": float(result.analysis.displacements[load_case.load_dof]),
        "mesh": {
            "elements": [mesh.nelx, mesh.nely, mesh.nelz],
            "dimensions": [mesh.length, mesh.width, mesh.height],
            "element_count": mesh.num_elements,
            "node_count": mesh.num_nodes,
            "dof_count": mesh.num_dofs,
        },
        "history": [
            {
                "iteration": record.iteration,
                "volume_fraction": record.volume_fraction,
                "compliance": record.compliance,
                "changed_elements": record.changed_elements,
                "duration_seconds": record.duration_seconds,
                "elapsed_seconds": record.elapsed_seconds,
                "best_near_target_compliance": (record.best_near_target_compliance),
                "assembly_seconds": record.assembly_seconds,
                "linear_solve_seconds": record.linear_solve_seconds,
                "postprocess_seconds": record.postprocess_seconds,
                "update_seconds": record.update_seconds,
            }
            for record in result.history
        ],
    }
    if coarse_result is not None and coarse_mesh is not None:
        summary["coarse"] = {
            "mesh": [coarse_mesh.nelx, coarse_mesh.nely, coarse_mesh.nelz],
            "element_count": coarse_mesh.num_elements,
            "dof_count": coarse_mesh.num_dofs,
            "converged": coarse_result.converged,
            "iterations": len(coarse_result.history),
            "elapsed_seconds": coarse_result.elapsed_seconds,
            "volume_fraction": float(coarse_result.densities.mean()),
            "compliance": coarse_result.analysis.compliance,
            "timings": timing_totals(coarse_result.history),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    status = "converged" if result.converged else "stopped"
    emit(
        f"{status} iterations={len(result.history)} "
        f"optimization={_format_duration(result.elapsed_seconds)} "
        f"total={_format_duration(total_elapsed_seconds)} "
        f"volume={result.densities.mean():.4f} "
        f"compliance={result.analysis.compliance:.6g}"
    )
    emit(f"wrote {args.output / 'design-solid.vtk'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

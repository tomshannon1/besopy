from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import cKDTree

from .fea import FiniteElementSolver, StaticResult


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    compliance: float
    volume_fraction: float
    changed_elements: int
    duration_seconds: float
    elapsed_seconds: float
    best_near_target_compliance: float | None
    assembly_seconds: float
    linear_solve_seconds: float
    postprocess_seconds: float
    update_seconds: float


@dataclass(frozen=True)
class OptimizationResult:
    densities: NDArray[np.float64]
    analysis: StaticResult
    history: tuple[IterationRecord, ...]
    converged: bool
    elapsed_seconds: float


class BESOOptimizer:
    """Bi-directional evolutionary structural optimization for compliance."""

    def __init__(
        self,
        solver: FiniteElementSolver,
        target_volume_fraction: float,
        *,
        evolution_rate: float = 0.02,
        filter_radius: float = 1.5,
        filter_mode: str = "element",
        convergence_tolerance: float = 1e-3,
        max_iterations: int = 100,
    ) -> None:
        if not 0 < target_volume_fraction <= 1:
            raise ValueError("Target volume fraction must be in (0, 1]")
        if not 0 < evolution_rate < 1:
            raise ValueError("Evolution rate must be in (0, 1)")
        if filter_radius <= 0:
            raise ValueError("Filter radius must be positive")
        if filter_mode not in {"element", "node"}:
            raise ValueError("Filter mode must be 'element' or 'node'")
        if convergence_tolerance <= 0:
            raise ValueError("Convergence tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("At least one iteration is required")

        self.solver = solver
        self.target_volume_fraction = target_volume_fraction
        self.evolution_rate = evolution_rate
        self.filter_radius = filter_radius
        self.filter_mode = filter_mode
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self._filter = self._build_filter()

    def _build_filter(self) -> sparse.csr_matrix:
        if self.filter_mode == "node":
            locations = self.solver.mesh.coordinates
            radius = self.filter_radius
        else:
            locations = self.solver.mesh.element_centers
            characteristic_length = min(self.solver.mesh.element_size)
            radius = self.filter_radius * characteristic_length
        neighborhoods = cKDTree(locations).query_ball_point(locations, radius)

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row, neighbors in enumerate(neighborhoods):
            distances = np.linalg.norm(locations[neighbors] - locations[row], axis=1)
            weights = np.maximum(0.0, radius - distances)
            weights /= weights.sum()
            rows.extend([row] * len(neighbors))
            columns.extend(neighbors)
            values.extend(weights)
        return sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(locations), len(locations)),
        )

    def _filter_sensitivities(
        self, sensitivities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self.filter_mode == "element":
            return np.asarray(self._filter @ sensitivities).reshape(-1)

        nodal_sensitivities = np.zeros(self.solver.mesh.num_nodes, dtype=float)
        np.add.at(
            nodal_sensitivities,
            self.solver.mesh.connectivity.ravel(),
            np.repeat(sensitivities, 8),
        )
        nodal_sensitivities /= 8.0
        filtered_nodes = np.asarray(self._filter @ nodal_sensitivities).reshape(-1)
        return filtered_nodes[self.solver.mesh.connectivity].mean(axis=1)

    def _select_design(
        self, sensitivities: NDArray[np.float64], volume_fraction: float
    ) -> NDArray[np.float64]:
        solid_count = max(
            1,
            min(
                self.solver.mesh.num_elements,
                round(volume_fraction * self.solver.mesh.num_elements),
            ),
        )
        order = np.argsort(sensitivities, kind="stable")
        design = np.zeros(self.solver.mesh.num_elements, dtype=float)
        design[order[-solid_count:]] = 1.0
        return design

    def _has_converged(
        self, compliance_history: list[float], volume_fraction: float
    ) -> bool:
        volume_tolerance = 1.5 / self.solver.mesh.num_elements
        if (
            volume_fraction > self.target_volume_fraction + volume_tolerance
            or len(compliance_history) < 10
        ):
            return False
        previous = sum(compliance_history[-10:-5])
        current = sum(compliance_history[-5:])
        scale = max(abs(previous), np.finfo(float).eps)
        return abs(current - previous) / scale <= self.convergence_tolerance

    def run(
        self,
        forces: NDArray[np.float64],
        fixed_dofs: NDArray[np.int64],
        *,
        initial_densities: NDArray[np.float64] | None = None,
        callback: Callable[[IterationRecord, NDArray[np.float64]], None] | None = None,
    ) -> OptimizationResult:
        if initial_densities is None:
            densities = np.ones(self.solver.mesh.num_elements, dtype=float)
        else:
            densities = np.asarray(initial_densities, dtype=float).reshape(-1).copy()
            if densities.shape != (self.solver.mesh.num_elements,):
                raise ValueError("Initial design requires one density per element")
            if np.any((densities < 0) | (densities > 1)):
                raise ValueError("Initial densities must lie between zero and one")
        previous_sensitivities: NDArray[np.float64] | None = None
        compliance_history: list[float] = []
        records: list[IterationRecord] = []
        converged = False
        best_densities: NDArray[np.float64] | None = None
        best_analysis: StaticResult | None = None
        target_tolerance = 1.5 / self.solver.mesh.num_elements
        started_at = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            iteration_started_at = time.perf_counter()
            analysis = self.solver.solve(densities, forces, fixed_dofs)
            if densities.mean() <= self.target_volume_fraction + target_tolerance and (
                best_analysis is None or analysis.compliance < best_analysis.compliance
            ):
                best_densities = densities.copy()
                best_analysis = analysis
            stiffness_factors = (
                self.solver.minimum_density
                + (1.0 - self.solver.minimum_density)
                * densities**self.solver.penalization
            )
            raw_sensitivities = stiffness_factors * analysis.element_strain_energy
            sensitivities = self._filter_sensitivities(raw_sensitivities)
            if previous_sensitivities is not None:
                sensitivities = 0.5 * (sensitivities + previous_sensitivities)
            previous_sensitivities = sensitivities

            next_volume_fraction = max(
                self.target_volume_fraction,
                densities.mean() * (1.0 - self.evolution_rate),
            )
            next_densities = self._select_design(sensitivities, next_volume_fraction)
            iteration_finished_at = time.perf_counter()
            fea_seconds = (
                analysis.assembly_seconds
                + analysis.linear_solve_seconds
                + analysis.postprocess_seconds
            )
            record = IterationRecord(
                iteration=iteration,
                compliance=analysis.compliance,
                volume_fraction=float(next_densities.mean()),
                changed_elements=int(np.count_nonzero(next_densities != densities)),
                duration_seconds=iteration_finished_at - iteration_started_at,
                elapsed_seconds=iteration_finished_at - started_at,
                best_near_target_compliance=(
                    None if best_analysis is None else best_analysis.compliance
                ),
                assembly_seconds=analysis.assembly_seconds,
                linear_solve_seconds=analysis.linear_solve_seconds,
                postprocess_seconds=analysis.postprocess_seconds,
                update_seconds=max(
                    0.0,
                    iteration_finished_at - iteration_started_at - fea_seconds,
                ),
            )
            records.append(record)
            compliance_history.append(analysis.compliance)
            densities = next_densities
            if callback is not None:
                callback(record, densities.copy())

            if self._has_converged(compliance_history, densities.mean()):
                converged = True
                break

        final_analysis = self.solver.solve(densities, forces, fixed_dofs)
        if densities.mean() <= self.target_volume_fraction + target_tolerance and (
            best_analysis is None
            or final_analysis.compliance < best_analysis.compliance
        ):
            best_densities = densities.copy()
            best_analysis = final_analysis
        if best_densities is not None and best_analysis is not None:
            densities = best_densities
            final_analysis = best_analysis
        return OptimizationResult(
            densities=densities,
            analysis=final_analysis,
            history=tuple(records),
            converged=converged,
            elapsed_seconds=time.perf_counter() - started_at,
        )

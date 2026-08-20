import numpy as np
import pytest

from beso import (
    BESOOptimizer,
    FiniteElementSolver,
    StructuredMesh,
    cantilever_load_case,
)


def test_optimizer_reaches_requested_volume() -> None:
    mesh = StructuredMesh(6, 2, 2, length=3.0, width=1.0, height=1.0)
    load_case = cantilever_load_case(mesh)
    solver = FiniteElementSolver(mesh, young_modulus=1.0)
    optimizer = BESOOptimizer(
        solver,
        target_volume_fraction=0.5,
        evolution_rate=0.25,
        max_iterations=8,
    )

    result = optimizer.run(load_case.forces, load_case.fixed_dofs)

    assert result.densities.mean() == 0.5
    assert set(np.unique(result.densities)) <= {0.0, 1.0}
    assert np.isfinite(result.analysis.compliance)
    assert result.analysis.compliance > 0
    assert len(result.history) <= 8


def test_optimizer_accepts_near_target_initial_design() -> None:
    mesh = StructuredMesh(6, 2, 2, length=3.0, width=1.0, height=1.0)
    load_case = cantilever_load_case(mesh)
    optimizer = BESOOptimizer(
        FiniteElementSolver(mesh, young_modulus=1.0),
        target_volume_fraction=0.5,
        max_iterations=2,
    )
    initial = np.zeros(mesh.num_elements)
    initial[: mesh.num_elements // 2] = 1.0

    result = optimizer.run(
        load_case.forces,
        load_case.fixed_dofs,
        initial_densities=initial,
    )

    assert len(result.history) == 2
    assert result.history[0].volume_fraction == 0.5
    assert result.history[0].assembly_seconds > 0
    assert result.history[0].linear_solve_seconds > 0


def test_node_filter_suppresses_checkerboard_sensitivities() -> None:
    mesh = StructuredMesh(3, 3, 3, length=3.0, width=3.0, height=3.0)
    optimizer = BESOOptimizer(
        FiniteElementSolver(mesh, young_modulus=1.0),
        target_volume_fraction=0.5,
        filter_mode="node",
        filter_radius=1.5,
    )
    checkerboard = (np.arange(mesh.num_elements) % 2).astype(float)

    filtered = optimizer._filter_sensitivities(checkerboard)

    assert filtered.std() < checkerboard.std()


def test_optimizer_rejects_unknown_filter_mode() -> None:
    mesh = StructuredMesh(1, 1, 1)

    with pytest.raises(ValueError, match="Filter mode"):
        BESOOptimizer(
            FiniteElementSolver(mesh, young_modulus=1.0),
            target_volume_fraction=0.5,
            filter_mode="unknown",
        )


def test_optimizer_respects_design_domain_and_passive_solids() -> None:
    mesh = StructuredMesh(4, 2, 2, length=2.0, width=1.0, height=1.0)
    load_case = cantilever_load_case(mesh)
    domain = np.zeros(mesh.num_elements, dtype=bool)
    domain[:12] = True
    passive = np.zeros(mesh.num_elements, dtype=bool)
    passive[[0, 1]] = True
    optimizer = BESOOptimizer(
        FiniteElementSolver(mesh, young_modulus=1.0),
        target_volume_fraction=0.5,
        evolution_rate=0.5,
        max_iterations=2,
        design_domain=domain,
        passive_solid=passive,
    )

    result = optimizer.run(load_case.forces, load_case.fixed_dofs)

    assert result.densities[domain].mean() == 0.5
    assert np.all(result.densities[passive] == 1.0)
    assert np.all(result.densities[~domain] == 0.0)

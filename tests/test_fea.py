import numpy as np

from beso import FiniteElementSolver, StructuredMesh, cantilever_load_case
from beso.fea import hexahedron_stiffness


def test_hexahedron_stiffness_is_symmetric_and_semidefinite() -> None:
    stiffness = hexahedron_stiffness((1.0, 1.0, 1.0), 1.0, 0.3)
    eigenvalues = np.linalg.eigvalsh(stiffness)

    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-12)
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-9) == 6
    assert eigenvalues.min() > -1e-12


def test_cantilever_deflects_in_the_load_direction() -> None:
    mesh = StructuredMesh(4, 2, 2, length=4.0, width=1.0, height=1.0)
    load_case = cantilever_load_case(mesh, load=-1.0)
    solver = FiniteElementSolver(mesh, young_modulus=1000.0)

    result = solver.solve(
        np.ones(mesh.num_elements),
        load_case.forces,
        load_case.fixed_dofs,
    )

    assert result.displacements[load_case.load_dof] < 0
    assert result.compliance > 0
    assert np.all(result.element_strain_energy >= -1e-12)
    np.testing.assert_allclose(
        result.compliance,
        result.displacements @ load_case.forces,
        rtol=1e-10,
    )


def test_cantilever_matches_euler_bernoulli_benchmark() -> None:
    length = 4.0
    young_modulus = 1000.0
    mesh = StructuredMesh(16, 4, 4, length=length, width=1.0, height=1.0)
    load_case = cantilever_load_case(mesh, load=-1.0, placement="face-center")
    result = FiniteElementSolver(mesh, young_modulus).solve(
        np.ones(mesh.num_elements),
        load_case.forces,
        load_case.fixed_dofs,
    )

    second_moment_of_area = 1.0 / 12.0
    beam_theory_displacement = length**3 / (3 * young_modulus * second_moment_of_area)
    np.testing.assert_allclose(
        -result.displacements[load_case.load_dof],
        beam_theory_displacement,
        rtol=0.05,
    )


def test_cantilever_load_is_uniform_on_free_end_bottom_edge() -> None:
    mesh = StructuredMesh(4, 3, 2, length=4.0, width=3.0, height=2.0)
    load_case = cantilever_load_case(mesh, load=-8.0)

    tip = mesh.nodes_on_x(mesh.length)
    bottom = tip[np.isclose(mesh.coordinates[tip, 2], 0.0)]
    nodal_forces = load_case.forces.reshape(-1, 3)
    loaded = np.flatnonzero(np.linalg.norm(nodal_forces, axis=1) > 0)

    assert len(bottom) == 4
    np.testing.assert_array_equal(np.sort(loaded), np.sort(bottom))
    np.testing.assert_allclose(nodal_forces[bottom, 2], -2.0)
    np.testing.assert_allclose(nodal_forces[bottom, :2], 0.0)
    np.testing.assert_array_equal(load_case.load_nodes, np.sort(bottom))

import numpy as np
import pytest

from beso.mesh import StructuredMesh
from beso.problems import bridge_load_case, make_problem, resolve_load_placement


def test_bridge_fixes_both_abutments_and_loads_midspan_deck() -> None:
    mesh = StructuredMesh(8, 2, 4, length=8.0, width=2.0, height=4.0)
    load_case, stories = bridge_load_case(mesh, load=-6.0, placement="midspan")
    coords = mesh.coordinates
    loaded = load_case.load_nodes
    support = np.unique(load_case.fixed_dofs // 3)

    assert np.allclose(coords[loaded, 0], 4.0)
    assert np.allclose(coords[loaded, 2], 4.0)
    assert len(loaded) == 3
    nodal = load_case.forces.reshape(-1, 3)
    np.testing.assert_allclose(nodal[loaded, 2], -2.0)
    assert len(stories) == 1

    left = support[np.isclose(coords[support, 0], 0.0)]
    right = support[np.isclose(coords[support, 0], 8.0)]
    assert np.allclose(coords[left, 2], 0.0)
    assert np.allclose(coords[right, 2], 0.0)
    assert len(left) == 3
    assert len(right) == 3


def test_bridge_convoy_puts_three_axles_on_the_centre_lane() -> None:
    mesh = StructuredMesh(8, 2, 4, length=8.0, width=2.0, height=4.0)
    load_case, stories = bridge_load_case(mesh, load=-6.0, placement="convoy")
    coords = mesh.coordinates
    loaded = load_case.load_nodes
    xs = np.unique(np.round(coords[loaded, 0], 6))

    assert np.allclose(coords[loaded, 2], 4.0)
    assert np.allclose(coords[loaded, 1], 1.0)
    np.testing.assert_allclose(xs, [2.0, 4.0, 6.0])
    assert len(stories) == 3
    np.testing.assert_allclose(load_case.forces.reshape(-1, 3)[loaded, 2].sum(), -6.0)


def test_bridge_deck_spreads_load_along_the_roadway() -> None:
    mesh = StructuredMesh(8, 2, 4, length=8.0, width=2.0, height=4.0)
    load_case, stories = bridge_load_case(mesh, load=-9.0, placement="deck")
    coords = mesh.coordinates
    loaded = load_case.load_nodes

    assert np.allclose(coords[loaded, 2], 4.0)
    assert np.allclose(coords[loaded, 1], 1.0)
    assert len(np.unique(np.round(coords[loaded, 0], 6))) > 3
    assert 3 <= len(stories) <= 5


def test_bridge_eccentric_loads_one_fascia() -> None:
    mesh = StructuredMesh(8, 2, 4, length=8.0, width=2.0, height=4.0)
    load_case, stories = bridge_load_case(mesh, load=-9.0, placement="eccentric")
    coords = mesh.coordinates
    loaded = load_case.load_nodes

    assert np.allclose(coords[loaded, 2], 4.0)
    assert np.allclose(coords[loaded, 1], 0.0)
    assert len(loaded) > 3
    assert 3 <= len(stories) <= 5


def test_make_problem_bridge_keeps_a_live_deck() -> None:
    mesh = StructuredMesh(6, 2, 3, length=6.0, width=2.0, height=3.0)
    problem = make_problem("bridge", mesh, load=-1.0, load_placement=None)
    assert problem.require_connectivity
    assert problem.passive_solid is not None
    assert int(problem.passive_solid.sum()) == mesh.nelx * mesh.nely
    centers = mesh.element_centers[problem.passive_solid]
    assert np.allclose(centers[:, 2], mesh.height - mesh.element_size[2] / 2.0)
    assert problem.connectivity_seeds is not None
    assert np.all(problem.passive_solid <= problem.connectivity_seeds)
    assert problem.overlay_kwargs["support_piers"]
    assert problem.overlay_kwargs["deck"]
    assert len(problem.overlay_kwargs["story_loads"]) == 3
    assert problem.overlay_kwargs["story"]["load_label"].startswith("TRAFFIC")


def test_resolve_load_placement_defaults_and_rejects_cross_problem_names() -> None:
    assert resolve_load_placement("cantilever", None) == "bottom-edge"
    assert resolve_load_placement("bridge", None) == "convoy"
    with pytest.raises(ValueError, match="Bridge load placement"):
        resolve_load_placement("bridge", "bottom-edge")
    with pytest.raises(ValueError, match="Cantilever load placement"):
        resolve_load_placement("cantilever", "convoy")

import numpy as np
import pytest

from beso.mesh import StructuredMesh
from beso.problems import (
    bridge_load_case,
    make_problem,
    resolve_load_placement,
    truss_load_case,
)


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
    assert resolve_load_placement("truss", None) == "center"
    with pytest.raises(ValueError, match="Bridge load placement"):
        resolve_load_placement("bridge", "bottom-edge")
    with pytest.raises(ValueError, match="Cantilever load placement"):
        resolve_load_placement("cantilever", "convoy")
    with pytest.raises(ValueError, match="Truss load placement"):
        resolve_load_placement("truss", "convoy")


def test_truss_pins_four_corners_and_loads_the_roof_center() -> None:
    mesh = StructuredMesh(8, 6, 6, length=8.0, width=6.0, height=6.0)
    load_case, stories = truss_load_case(mesh, load=-4.0, placement="center")
    coords = mesh.coordinates
    loaded = load_case.load_nodes
    support = np.unique(load_case.fixed_dofs // 3)
    xs = coords[support, 0]
    ys = coords[support, 1]

    assert np.allclose(coords[loaded, 2], 6.0)
    assert np.mean(coords[loaded, 0]) == pytest.approx(4.0, abs=1.1)
    assert np.mean(coords[loaded, 1]) == pytest.approx(3.0, abs=1.1)
    assert len(stories) == 1
    np.testing.assert_allclose(load_case.forces.reshape(-1, 3)[loaded, 2].sum(), -4.0)

    assert np.allclose(coords[support, 2], 0.0)
    assert xs.min() == pytest.approx(0.0)
    assert xs.max() == pytest.approx(8.0)
    assert ys.min() == pytest.approx(0.0)
    assert ys.max() == pytest.approx(6.0)
    assert ((xs < 2) & (ys < 2)).any()
    assert ((xs < 2) & (ys > 4)).any()
    assert ((xs > 6) & (ys < 2)).any()
    assert ((xs > 6) & (ys > 4)).any()
    middle = support[(np.abs(xs - 4) < 1.5) & (np.abs(ys - 3) < 1.5)]
    assert len(middle) == 0


def test_truss_offset_and_span_move_the_roof_joints() -> None:
    mesh = StructuredMesh(8, 6, 6, length=8.0, width=6.0, height=6.0)
    offset, offset_stories = truss_load_case(mesh, load=-2.0, placement="offset")
    span, span_stories = truss_load_case(mesh, load=-2.0, placement="span")

    assert np.allclose(mesh.coordinates[offset.load_nodes, 2], 6.0)
    assert mesh.coordinates[offset.load_nodes, 1].mean() == pytest.approx(0.0, abs=1.1)
    assert len(offset_stories) == 1
    assert len(span_stories) == 2
    xs = np.unique(np.round(mesh.coordinates[span.load_nodes, 0], 6))
    assert xs.min() < 4.0
    assert xs.max() > 4.0


def test_make_problem_truss_keeps_pin_and_load_seeds() -> None:
    mesh = StructuredMesh(6, 4, 4, length=6.0, width=4.0, height=4.0)
    problem = make_problem("truss", mesh, load=-1.0, load_placement=None)
    assert problem.require_connectivity
    assert problem.prune_dead_ends
    assert problem.passive_solid is None
    assert problem.connectivity_seeds is not None
    assert int(problem.connectivity_seeds.sum()) >= 4
    assert len(problem.overlay_kwargs["support_piers"]) == 4
    assert len(problem.overlay_kwargs["story_loads"]) == 1
    assert "Michell" in problem.overlay_kwargs["story"]["load_label"]

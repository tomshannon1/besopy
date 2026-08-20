import numpy as np

from beso.topology_constraints import (
    apply_minimum_thickness,
    keep_connected_to_seeds,
    prune_dead_ends,
)


def test_keep_connected_removes_islands() -> None:
    solid = np.zeros((3, 3, 3), dtype=bool)
    solid[0, 0, 0] = True
    solid[2, 2, 2] = True
    seeds = np.zeros_like(solid)
    seeds[0, 0, 0] = True
    domain = np.ones_like(solid)
    passive = np.zeros_like(solid)

    connected = keep_connected_to_seeds(solid, seeds, domain=domain, passive=passive)

    assert connected[0, 0, 0]
    assert not connected[2, 2, 2]


def test_minimum_thickness_removes_single_voxel_bridges() -> None:
    solid = np.zeros((5, 3, 3), dtype=bool)
    solid[0] = True
    solid[4] = True
    solid[1:4, 1, 1] = True
    domain = np.ones_like(solid)
    passive = np.zeros_like(solid)
    passive[0] = True
    passive[4] = True

    thickened = apply_minimum_thickness(solid, domain=domain, passive=passive, voxels=2)

    assert np.all(thickened[0])
    assert np.all(thickened[4])
    assert not thickened[2, 1, 1]


def test_prune_dead_ends_removes_dangling_chain() -> None:
    solid = np.zeros((4, 2, 2), dtype=bool)
    solid[0] = True
    solid[1, 0, 0] = True
    solid[2, 0, 0] = True
    domain = np.ones_like(solid)
    passive = np.zeros_like(solid)
    passive[0] = True

    pruned = prune_dead_ends(solid, domain=domain, passive=passive)

    assert np.all(pruned[0])
    assert not pruned[2, 0, 0]

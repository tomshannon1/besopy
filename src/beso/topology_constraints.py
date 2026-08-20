from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    binary_propagation,
    convolve,
    generate_binary_structure,
)

FACE_STRUCTURE = generate_binary_structure(3, 1)


def _as_grid(mask: NDArray[np.bool_], shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    return np.asarray(mask, dtype=bool).reshape(shape)


def apply_minimum_thickness(
    solid: NDArray[np.bool_],
    *,
    domain: NDArray[np.bool_],
    passive: NDArray[np.bool_],
    voxels: int,
) -> NDArray[np.bool_]:
    if voxels < 1:
        raise ValueError("Minimum thickness must be at least one voxel")
    if voxels == 1:
        return solid & domain | passive

    opened = binary_opening(
        solid,
        structure=FACE_STRUCTURE,
        iterations=voxels - 1,
    )
    return (opened & domain) | passive


def keep_connected_to_seeds(
    solid: NDArray[np.bool_],
    seeds: NDArray[np.bool_],
    *,
    domain: NDArray[np.bool_],
    passive: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    if not np.any(seeds & solid):
        return (solid & False) | passive
    reachable = binary_propagation(seeds & solid, mask=solid, structure=FACE_STRUCTURE)
    return (reachable & domain) | passive


def prune_dead_ends(
    solid: NDArray[np.bool_],
    *,
    domain: NDArray[np.bool_],
    passive: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Remove non-protected chains that terminate without connecting elsewhere."""

    current = (solid & domain) | passive
    kernel = FACE_STRUCTURE.astype(int)
    kernel[1, 1, 1] = 0

    changed = True
    while changed:
        counts = convolve(current.astype(int), kernel, mode="constant", cval=0)
        leaves = current & ~passive & (counts <= 1)
        changed = bool(np.any(leaves))
        current = current & ~leaves
    return current


def restore_volume(
    solid: NDArray[np.bool_],
    sensitivities: NDArray[np.float64],
    *,
    domain: NDArray[np.bool_],
    target_count: int,
) -> NDArray[np.bool_]:
    current = solid.copy()
    flat_sensitivities = np.asarray(sensitivities, dtype=float).reshape(-1)
    while int(current.sum()) < target_count:
        frontier = (
            binary_dilation(current, structure=FACE_STRUCTURE) & domain & ~current
        )
        candidates = np.flatnonzero(frontier.reshape(-1))
        if candidates.size == 0:
            break
        needed = min(target_count - int(current.sum()), candidates.size)
        chosen = candidates[
            np.argpartition(flat_sensitivities[candidates], -needed)[-needed:]
        ]
        current.reshape(-1)[chosen] = True
    return current


def enforce_printable_topology(
    densities: NDArray[np.float64],
    sensitivities: NDArray[np.float64],
    *,
    shape: tuple[int, int, int],
    domain: NDArray[np.bool_],
    passive: NDArray[np.bool_],
    seeds: NDArray[np.bool_],
    min_thickness_voxels: int,
    target_count: int,
    require_connectivity: bool,
    prune: bool,
) -> NDArray[np.float64]:
    solid = _as_grid(np.asarray(densities) > 0.5, shape)
    domain_grid = _as_grid(domain, shape)
    passive_grid = _as_grid(passive, shape)
    seed_grid = _as_grid(seeds, shape)

    solid = apply_minimum_thickness(
        solid,
        domain=domain_grid,
        passive=passive_grid,
        voxels=min_thickness_voxels,
    )
    if prune:
        solid = prune_dead_ends(solid, domain=domain_grid, passive=passive_grid)
    if require_connectivity:
        solid = keep_connected_to_seeds(
            solid,
            seed_grid,
            domain=domain_grid,
            passive=passive_grid,
        )
    solid = restore_volume(
        solid,
        sensitivities,
        domain=domain_grid,
        target_count=target_count,
    )
    if require_connectivity:
        solid = keep_connected_to_seeds(
            solid,
            seed_grid,
            domain=domain_grid,
            passive=passive_grid,
        )
    return solid.astype(float).reshape(-1)

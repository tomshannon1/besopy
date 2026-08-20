from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from .mesh import StructuredMesh


def coarsen_shape(mesh: StructuredMesh, factor: int) -> tuple[int, int, int]:
    if factor < 2:
        raise ValueError("Coarsening factor must be at least two")
    return tuple(
        max(1, int(np.ceil(elements / factor)))
        for elements in (mesh.nelx, mesh.nely, mesh.nelz)
    )


def resample_densities(
    source_mesh: StructuredMesh,
    target_mesh: StructuredMesh,
    densities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer cell densities using nearest normalized element centers."""

    source = np.asarray(densities, dtype=float).reshape(-1)
    if source.shape != (source_mesh.num_elements,):
        raise ValueError("Source design requires one density per element")

    source_grid = source.reshape(source_mesh.nelx, source_mesh.nely, source_mesh.nelz)

    def source_indices(source_count: int, target_count: int) -> NDArray[np.int64]:
        normalized_centers = (np.arange(target_count) + 0.5) / target_count
        return np.minimum(
            (normalized_centers * source_count).astype(np.int64),
            source_count - 1,
        )

    ix = source_indices(source_mesh.nelx, target_mesh.nelx)
    iy = source_indices(source_mesh.nely, target_mesh.nely)
    iz = source_indices(source_mesh.nelz, target_mesh.nelz)
    return source_grid[np.ix_(ix, iy, iz)].reshape(-1).copy()


def expand_design(
    mesh: StructuredMesh,
    densities: NDArray[np.float64],
    volume_fraction: float,
) -> NDArray[np.float64]:
    """Thicken a binary design with the physically nearest void elements."""

    if not 0 < volume_fraction <= 1:
        raise ValueError("Volume fraction must be in (0, 1]")
    design = np.asarray(densities, dtype=float).reshape(-1) > 0.5
    if design.shape != (mesh.num_elements,):
        raise ValueError("Design requires one density per element")

    target_count = round(volume_fraction * mesh.num_elements)
    current_count = int(design.sum())
    if target_count <= current_count:
        return design.astype(float)
    if current_count == 0:
        raise ValueError("Cannot expand an empty design")

    grid = design.reshape(mesh.nelx, mesh.nely, mesh.nelz)
    distances = distance_transform_edt(~grid, sampling=mesh.element_size)
    void_indices = np.flatnonzero(~design)
    nearest_void = void_indices[
        np.argsort(distances.reshape(-1)[void_indices], kind="stable")
    ]
    design[nearest_void[: target_count - current_count]] = True
    return design.astype(float)

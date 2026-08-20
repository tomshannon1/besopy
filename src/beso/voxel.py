from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from .mesh import StructuredMesh


@dataclass(frozen=True)
class VoxelDomain:
    mesh: StructuredMesh
    occupied: NDArray[np.float64]
    source_bounds: NDArray[np.float64]
    source_to_local_offset: NDArray[np.float64]
    source_vertices: int
    source_faces: int
    source_bodies: int
    source_volume: float
    source_watertight: bool
    body_masks: tuple[NDArray[np.bool_], ...]
    body_bounds: tuple[NDArray[np.float64], ...]
    body_volumes: tuple[float, ...]
    pitch: float

    @property
    def occupied_elements(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def occupied_fraction(self) -> float:
        return float(self.occupied.mean())


def voxelize_stl(
    path: str | Path,
    pitch: float,
    *,
    require_watertight: bool = True,
) -> VoxelDomain:
    if pitch <= 0:
        raise ValueError("Voxel pitch must be positive")

    loaded = trimesh.load(Path(path), force="scene")
    geometry = loaded.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh) or geometry.is_empty:
        raise ValueError("STL does not contain a usable triangle mesh")
    if require_watertight and not geometry.is_watertight:
        raise ValueError("STL must be watertight before solid voxelization")

    voxel_grid = geometry.voxelized(pitch).fill()
    occupancy = np.asarray(voxel_grid.matrix, dtype=float)
    shape = occupancy.shape
    mesh = StructuredMesh(
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        length=float(shape[0] * pitch),
        width=float(shape[1] * pitch),
        height=float(shape[2] * pitch),
    )
    first_center = np.asarray(
        voxel_grid.indices_to_points(np.array([[0, 0, 0]]))[0],
        dtype=float,
    )
    local_first_center = 0.5 * np.asarray(mesh.element_size)
    components = sorted(
        geometry.split(only_watertight=False),
        key=lambda component: abs(component.volume),
        reverse=True,
    )
    body_masks: list[NDArray[np.bool_]] = []
    for component in components:
        component_grid = component.voxelized(pitch).fill()
        indices = voxel_grid.points_to_indices(component_grid.points)
        valid = np.all(
            (indices >= 0) & (indices < np.asarray(shape)),
            axis=1,
        )
        mask = np.zeros(shape, dtype=bool)
        valid_indices = indices[valid]
        mask[tuple(valid_indices.T)] = True
        body_masks.append(mask.reshape(-1))

    return VoxelDomain(
        mesh=mesh,
        occupied=occupancy.reshape(-1),
        source_bounds=np.asarray(geometry.bounds, dtype=float),
        source_to_local_offset=local_first_center - first_center,
        source_vertices=len(geometry.vertices),
        source_faces=len(geometry.faces),
        source_bodies=int(geometry.body_count),
        source_volume=float(abs(geometry.volume)),
        source_watertight=bool(geometry.is_watertight),
        body_masks=tuple(body_masks),
        body_bounds=tuple(
            np.asarray(component.bounds, dtype=float) for component in components
        ),
        body_volumes=tuple(float(abs(component.volume)) for component in components),
        pitch=float(pitch),
    )

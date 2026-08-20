from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mesh import StructuredMesh


@dataclass(frozen=True)
class LoadCase:
    forces: NDArray[np.float64]
    fixed_dofs: NDArray[np.int64]
    load_dof: int


def cantilever_load_case(
    mesh: StructuredMesh,
    load: float = -1.0,
    direction: int = 2,
) -> LoadCase:
    """Clamp x=0 and apply one load at the center of the opposite face."""

    if direction not in (0, 1, 2):
        raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")
    target = np.array([mesh.length, mesh.width / 2, mesh.height / 2])
    tip_nodes = mesh.nodes_on_x(mesh.length)
    distances = np.linalg.norm(mesh.coordinates[tip_nodes] - target, axis=1)
    load_node = int(tip_nodes[np.argmin(distances)])
    load_dof = 3 * load_node + direction

    forces = np.zeros(mesh.num_dofs, dtype=float)
    forces[load_dof] = load
    return LoadCase(
        forces=forces,
        fixed_dofs=mesh.fixed_dofs_at_x(),
        load_dof=load_dof,
    )

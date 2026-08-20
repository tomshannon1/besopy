from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StructuredMesh:
    """Regular hexahedral mesh with three translational DOFs per node."""

    nelx: int
    nely: int
    nelz: int
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self) -> None:
        values = (self.nelx, self.nely, self.nelz)
        if any(value < 1 for value in values):
            raise ValueError("Element counts must all be positive")
        if any(value <= 0 for value in (self.length, self.width, self.height)):
            raise ValueError("Physical dimensions must all be positive")

    @property
    def num_elements(self) -> int:
        return self.nelx * self.nely * self.nelz

    @property
    def num_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)

    @property
    def num_dofs(self) -> int:
        return 3 * self.num_nodes

    def node_index(self, ix: int, iy: int, iz: int) -> int:
        if not (0 <= ix <= self.nelx and 0 <= iy <= self.nely and 0 <= iz <= self.nelz):
            raise IndexError("Node index lies outside the mesh")
        return (ix * (self.nely + 1) + iy) * (self.nelz + 1) + iz

    @cached_property
    def coordinates(self) -> NDArray[np.float64]:
        x = np.linspace(0.0, self.length, self.nelx + 1)
        y = np.linspace(0.0, self.width, self.nely + 1)
        z = np.linspace(0.0, self.height, self.nelz + 1)
        return np.array(
            [
                (x[ix], y[iy], z[iz])
                for ix in range(self.nelx + 1)
                for iy in range(self.nely + 1)
                for iz in range(self.nelz + 1)
            ],
            dtype=float,
        )

    @cached_property
    def connectivity(self) -> NDArray[np.int64]:
        elements: list[list[int]] = []
        for ix in range(self.nelx):
            for iy in range(self.nely):
                for iz in range(self.nelz):
                    elements.append(
                        [
                            self.node_index(ix, iy, iz),
                            self.node_index(ix + 1, iy, iz),
                            self.node_index(ix + 1, iy + 1, iz),
                            self.node_index(ix, iy + 1, iz),
                            self.node_index(ix, iy, iz + 1),
                            self.node_index(ix + 1, iy, iz + 1),
                            self.node_index(ix + 1, iy + 1, iz + 1),
                            self.node_index(ix, iy + 1, iz + 1),
                        ]
                    )
        return np.asarray(elements, dtype=np.int64)

    @cached_property
    def element_dofs(self) -> NDArray[np.int64]:
        offsets = np.arange(3, dtype=np.int64)
        return (3 * self.connectivity[:, :, None] + offsets).reshape(-1, 24)

    @cached_property
    def element_centers(self) -> NDArray[np.float64]:
        return self.coordinates[self.connectivity].mean(axis=1)

    @cached_property
    def element_size(self) -> tuple[float, float, float]:
        return (
            self.length / self.nelx,
            self.width / self.nely,
            self.height / self.nelz,
        )

    def nodes_on_x(self, x: float, *, atol: float = 1e-12) -> NDArray[np.int64]:
        return np.flatnonzero(np.isclose(self.coordinates[:, 0], x, atol=atol))

    def fixed_dofs_at_x(self, x: float = 0.0) -> NDArray[np.int64]:
        nodes = self.nodes_on_x(x)
        return (3 * nodes[:, None] + np.arange(3)).reshape(-1)

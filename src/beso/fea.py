from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .mesh import StructuredMesh


def elasticity_matrix(
    young_modulus: float, poisson_ratio: float
) -> NDArray[np.float64]:
    if young_modulus <= 0:
        raise ValueError("Young's modulus must be positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson's ratio must be between -1 and 0.5")

    coefficient = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    nu = poisson_ratio
    return coefficient * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ],
        dtype=float,
    )


def hexahedron_stiffness(
    size: tuple[float, float, float],
    young_modulus: float,
    poisson_ratio: float,
) -> NDArray[np.float64]:
    """Return the 24x24 stiffness matrix for an eight-node brick element."""

    hx, hy, hz = size
    coordinates = np.array(
        [
            [0, 0, 0],
            [hx, 0, 0],
            [hx, hy, 0],
            [0, hy, 0],
            [0, 0, hz],
            [hx, 0, hz],
            [hx, hy, hz],
            [0, hy, hz],
        ],
        dtype=float,
    )
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    constitutive = elasticity_matrix(young_modulus, poisson_ratio)
    stiffness = np.zeros((24, 24), dtype=float)

    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:
                derivatives = np.empty((8, 3), dtype=float)
                for node, (sx, sy, sz) in enumerate(signs):
                    derivatives[node] = 0.125 * np.array(
                        [
                            sx * (1 + sy * eta) * (1 + sz * zeta),
                            sy * (1 + sx * xi) * (1 + sz * zeta),
                            sz * (1 + sx * xi) * (1 + sy * eta),
                        ]
                    )

                jacobian = derivatives.T @ coordinates
                determinant = np.linalg.det(jacobian)
                if determinant <= 0:
                    raise ValueError("Element has a non-positive Jacobian")
                gradients = derivatives @ np.linalg.inv(jacobian)

                strain_displacement = np.zeros((6, 24), dtype=float)
                for node, (dx, dy, dz) in enumerate(gradients):
                    column = 3 * node
                    strain_displacement[:, column : column + 3] = np.array(
                        [
                            [dx, 0, 0],
                            [0, dy, 0],
                            [0, 0, dz],
                            [dy, dx, 0],
                            [0, dz, dy],
                            [dz, 0, dx],
                        ]
                    )
                stiffness += (
                    strain_displacement.T
                    @ constitutive
                    @ strain_displacement
                    * determinant
                )

    return 0.5 * (stiffness + stiffness.T)


@dataclass(frozen=True)
class StaticResult:
    displacements: NDArray[np.float64]
    compliance: float
    element_strain_energy: NDArray[np.float64]
    assembly_seconds: float
    linear_solve_seconds: float
    postprocess_seconds: float


@dataclass
class FiniteElementSolver:
    mesh: StructuredMesh
    young_modulus: float
    poisson_ratio: float = 0.3
    penalization: float = 3.0
    minimum_density: float = 1e-6

    def __post_init__(self) -> None:
        if self.penalization < 1:
            raise ValueError("Penalization must be at least one")
        if not 0 < self.minimum_density <= 1:
            raise ValueError("Minimum density must be in (0, 1]")

    @cached_property
    def element_stiffness(self) -> NDArray[np.float64]:
        return hexahedron_stiffness(
            self.mesh.element_size, self.young_modulus, self.poisson_ratio
        )

    @cached_property
    def _assembly_indices(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        dofs = self.mesh.element_dofs
        rows = np.repeat(dofs, 24, axis=1).ravel()
        columns = np.tile(dofs, (1, 24)).ravel()
        return rows, columns

    def assemble(self, densities: NDArray[np.float64]) -> sparse.csc_matrix:
        densities = np.asarray(densities, dtype=float).reshape(-1)
        if densities.shape != (self.mesh.num_elements,):
            raise ValueError("One density is required per element")
        if np.any((densities < 0) | (densities > 1)):
            raise ValueError("Densities must lie between zero and one")

        factors = self.minimum_density + (1 - self.minimum_density) * (
            densities**self.penalization
        )
        values = (factors[:, None] * self.element_stiffness.ravel()).ravel()
        rows, columns = self._assembly_indices
        matrix = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(self.mesh.num_dofs, self.mesh.num_dofs),
        )
        return matrix.tocsc()

    def solve(
        self,
        densities: NDArray[np.float64],
        forces: NDArray[np.float64],
        fixed_dofs: NDArray[np.int64],
    ) -> StaticResult:
        forces = np.asarray(forces, dtype=float).reshape(-1)
        fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=np.int64).reshape(-1))
        if forces.shape != (self.mesh.num_dofs,):
            raise ValueError("Force vector has the wrong size")
        if np.any((fixed_dofs < 0) | (fixed_dofs >= self.mesh.num_dofs)):
            raise ValueError("A fixed DOF lies outside the mesh")

        free_dofs = np.setdiff1d(
            np.arange(self.mesh.num_dofs, dtype=np.int64),
            fixed_dofs,
            assume_unique=True,
        )
        assembly_started_at = time.perf_counter()
        stiffness = self.assemble(densities)
        assembly_seconds = time.perf_counter() - assembly_started_at

        linear_solve_started_at = time.perf_counter()
        displacements = np.zeros(self.mesh.num_dofs, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            displacements[free_dofs] = spsolve(
                stiffness[free_dofs][:, free_dofs], forces[free_dofs]
            )
        if not np.all(np.isfinite(displacements)):
            raise np.linalg.LinAlgError("The finite-element system is singular")
        linear_solve_seconds = time.perf_counter() - linear_solve_started_at

        postprocess_started_at = time.perf_counter()
        element_displacements = displacements[self.mesh.element_dofs]
        element_energy = np.einsum(
            "ei,ij,ej->e",
            element_displacements,
            self.element_stiffness,
            element_displacements,
        )
        return StaticResult(
            displacements=displacements,
            compliance=float(forces @ displacements),
            element_strain_energy=element_energy,
            assembly_seconds=assembly_seconds,
            linear_solve_seconds=linear_solve_seconds,
            postprocess_seconds=time.perf_counter() - postprocess_started_at,
        )

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from .mesh import StructuredMesh


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    @property
    def is_watertight(self) -> bool:
        edges = np.concatenate(
            (
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            )
        )
        edges.sort(axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(len(counts) and np.all(counts == 2))

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))


def read_vtk_densities(
    path: str | Path, number_of_elements: int
) -> NDArray[np.float64]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("LOOKUP_TABLE default") + 1
    except ValueError as error:
        raise ValueError("VTK file has no density lookup table") from error
    values = np.asarray(lines[start : start + number_of_elements], dtype=float)
    if values.shape != (number_of_elements,):
        raise ValueError("VTK density field has the wrong number of elements")
    return values


def _largest_component(surface: SurfaceMesh) -> SurfaceMesh:
    face_count = len(surface.faces)
    vertex_count = len(surface.vertices)
    face_ids = np.repeat(np.arange(face_count), 3)
    incidence = sparse.csr_matrix(
        (np.ones(3 * face_count), (face_ids, surface.faces.ravel())),
        shape=(face_count, vertex_count),
    )
    adjacency = incidence @ incidence.T
    _, labels = connected_components(adjacency, directed=False)
    largest = np.argmax(np.bincount(labels))
    faces = surface.faces[labels == largest]

    used_vertices, inverse = np.unique(faces, return_inverse=True)
    return SurfaceMesh(
        vertices=surface.vertices[used_vertices],
        faces=inverse.reshape(-1, 3),
    )


def _taubin_smooth(
    surface: SurfaceMesh,
    iterations: int,
    *,
    forward: float = 0.5,
    backward: float = -0.53,
) -> SurfaceMesh:
    if iterations < 0:
        raise ValueError("Smoothing iterations cannot be negative")
    if iterations == 0:
        return surface

    edges = np.concatenate(
        (
            surface.faces[:, [0, 1]],
            surface.faces[:, [1, 2]],
            surface.faces[:, [2, 0]],
        )
    )
    edges = np.vstack((edges, edges[:, ::-1]))
    adjacency = sparse.csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(surface.vertices), len(surface.vertices)),
    )
    adjacency.data[:] = 1.0
    row_sums = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    averaging = sparse.diags(1.0 / row_sums) @ adjacency

    vertices = surface.vertices.copy()
    for _ in range(iterations):
        vertices += forward * (averaging @ vertices - vertices)
        vertices += backward * (averaging @ vertices - vertices)
    return SurfaceMesh(vertices=vertices, faces=surface.faces)


def density_surface(
    mesh: StructuredMesh,
    densities: NDArray[np.float64],
    *,
    sigma: float = 0.7,
    level: float = 0.5,
    smoothing_iterations: int = 10,
    keep_largest_component: bool = True,
) -> SurfaceMesh:
    if sigma < 0:
        raise ValueError("Gaussian sigma cannot be negative")
    if not 0 < level < 1:
        raise ValueError("Surface level must be in (0, 1)")
    values = np.asarray(densities, dtype=float).reshape(-1)
    if values.shape != (mesh.num_elements,):
        raise ValueError("One density is required per element")

    occupancy = (values > 0.5).astype(float).reshape(mesh.nelx, mesh.nely, mesh.nelz)
    padding = max(2, int(np.ceil(3 * sigma)) + 1)
    field = np.pad(occupancy, padding, mode="constant")
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma, mode="constant")
    if field.max() <= level:
        raise ValueError("Smoothing removed every printable feature")

    vertices, faces, _normals, _values = marching_cubes(
        field,
        level=level,
        spacing=mesh.element_size,
        allow_degenerate=False,
    )
    offset = (padding - 0.5) * np.asarray(mesh.element_size)
    surface = SurfaceMesh(
        vertices=vertices - offset,
        faces=np.asarray(faces, dtype=np.int64),
    )
    if keep_largest_component:
        surface = _largest_component(surface)
    return _taubin_smooth(surface, smoothing_iterations)


def write_binary_stl(path: str | Path, surface: SurfaceMesh) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    triangles = surface.vertices[surface.faces]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross_products, axis=1)
    normals = np.divide(
        cross_products,
        lengths[:, None],
        out=np.zeros_like(cross_products),
        where=lengths[:, None] > 0,
    )

    with destination.open("wb") as stl:
        stl.write(b"BESO printable isosurface".ljust(80, b"\0"))
        stl.write(struct.pack("<I", len(triangles)))
        for normal, triangle in zip(normals, triangles, strict=True):
            stl.write(struct.pack("<12fH", *normal, *triangle.ravel(), 0))
    return destination

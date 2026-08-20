from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mesh import StructuredMesh


def write_vtk(
    path: str | Path,
    mesh: StructuredMesh,
    densities: NDArray[np.float64],
    *,
    active_only: bool = False,
) -> Path:
    """Write a legacy ASCII VTK unstructured grid readable by ParaView."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    densities = np.asarray(densities, dtype=float).reshape(-1)
    if densities.shape != (mesh.num_elements,):
        raise ValueError("One density is required per element")

    mask = densities > 0.5 if active_only else np.ones(mesh.num_elements, dtype=bool)
    cells = mesh.connectivity[mask]
    cell_densities = densities[mask]

    with destination.open("w", encoding="utf-8") as vtk:
        vtk.write("# vtk DataFile Version 3.0\n")
        vtk.write("BESO topology optimization\n")
        vtk.write("ASCII\n")
        vtk.write("DATASET UNSTRUCTURED_GRID\n")
        vtk.write(f"POINTS {mesh.num_nodes} double\n")
        for x, y, z in mesh.coordinates:
            vtk.write(f"{x:.12g} {y:.12g} {z:.12g}\n")

        vtk.write(f"CELLS {len(cells)} {9 * len(cells)}\n")
        for cell in cells:
            vtk.write("8 " + " ".join(str(int(node)) for node in cell) + "\n")
        vtk.write(f"CELL_TYPES {len(cells)}\n")
        vtk.write("12\n" * len(cells))
        vtk.write(f"CELL_DATA {len(cells)}\n")
        vtk.write("SCALARS density double 1\n")
        vtk.write("LOOKUP_TABLE default\n")
        for density in cell_densities:
            vtk.write(f"{density:.12g}\n")
    return destination

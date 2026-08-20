import struct

import numpy as np

from beso import StructuredMesh
from beso.io import write_vtk
from beso.surface import density_surface, read_vtk_densities, write_binary_stl


def test_marching_cubes_surface_is_watertight() -> None:
    mesh = StructuredMesh(3, 3, 3, length=3.0, width=3.0, height=3.0)

    surface = density_surface(
        mesh,
        np.ones(mesh.num_elements),
        sigma=0.5,
        smoothing_iterations=2,
    )

    assert len(surface.vertices) > 0
    assert len(surface.faces) > 0
    assert surface.is_watertight


def test_vtk_density_round_trip_and_binary_stl(tmp_path) -> None:
    mesh = StructuredMesh(2, 2, 2)
    densities = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    vtk_path = write_vtk(tmp_path / "design.vtk", mesh, densities)

    restored = read_vtk_densities(vtk_path, mesh.num_elements)
    surface = density_surface(
        mesh,
        restored,
        sigma=0,
        smoothing_iterations=0,
    )
    stl_path = write_binary_stl(tmp_path / "design.stl", surface)

    np.testing.assert_array_equal(restored, densities)
    contents = stl_path.read_bytes()
    triangle_count = struct.unpack("<I", contents[80:84])[0]
    assert triangle_count == len(surface.faces)
    assert len(contents) == 84 + 50 * triangle_count

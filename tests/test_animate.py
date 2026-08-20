import numpy as np
import trimesh

from beso.animate import render_mesh, subsample_indices


def test_subsample_indices_keeps_ends() -> None:
    picked = subsample_indices(10, 4)
    assert picked[0] == 0
    assert picked[-1] == 9
    assert len(picked) == 4


def test_render_mesh_draws_geometry_on_dark_background() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    bounds = np.array([[-1.0, -0.5, -0.5], [1.0, 0.5, 0.5]])
    image = render_mesh(
        mesh,
        bounds=bounds,
        width=80,
        height=50,
        azimuth=40.0,
        elevation=25.0,
    )
    assert image.shape == (50, 80, 3)
    assert image[0, 0].tolist() == [27, 29, 33]
    assert image.mean() > 30

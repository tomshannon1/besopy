import numpy as np
import pytest

from beso import StructuredMesh


def test_structured_mesh_counts_and_connectivity() -> None:
    mesh = StructuredMesh(2, 3, 4, length=2.0, width=3.0, height=4.0)

    assert mesh.num_elements == 24
    assert mesh.num_nodes == 60
    assert mesh.connectivity.shape == (24, 8)
    assert mesh.element_dofs.shape == (24, 24)
    np.testing.assert_allclose(mesh.coordinates.min(axis=0), [0, 0, 0])
    np.testing.assert_allclose(mesh.coordinates.max(axis=0), [2, 3, 4])


def test_mesh_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="Element counts"):
        StructuredMesh(0, 1, 1)

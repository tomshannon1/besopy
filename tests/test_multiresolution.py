import numpy as np
import pytest

from beso import StructuredMesh
from beso.multiresolution import coarsen_shape, expand_design, resample_densities


def test_coarsen_shape_rounds_up_odd_axes() -> None:
    mesh = StructuredMesh(30, 15, 8)

    assert coarsen_shape(mesh, 2) == (15, 8, 4)


def test_resample_densities_preserves_constant_and_binary_designs() -> None:
    coarse = StructuredMesh(2, 1, 1)
    fine = StructuredMesh(4, 2, 2)

    constant = resample_densities(coarse, fine, np.ones(coarse.num_elements))
    split = resample_densities(coarse, fine, np.array([1.0, 0.0]))

    np.testing.assert_array_equal(constant, np.ones(fine.num_elements))
    assert set(np.unique(split)) == {0.0, 1.0}
    assert split.mean() == 0.5


def test_resample_rejects_wrong_source_size() -> None:
    mesh = StructuredMesh(2, 2, 2)

    with pytest.raises(ValueError, match="one density"):
        resample_densities(mesh, mesh, np.ones(2))


def test_expand_design_reaches_volume_and_preserves_existing_solids() -> None:
    mesh = StructuredMesh(4, 2, 2)
    initial = np.zeros(mesh.num_elements)
    initial[0] = 1.0

    expanded = expand_design(mesh, initial, 0.5)

    assert expanded.mean() == 0.5
    assert expanded[0] == 1.0
    assert set(np.unique(expanded)) == {0.0, 1.0}

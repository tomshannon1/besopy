import json

import numpy as np
import pytest
import trimesh

from beso.import_cli import main
from beso.voxel import voxelize_stl


def _write_box(path) -> None:
    trimesh.creation.box(extents=(4.0, 3.0, 2.0)).export(path)


def test_voxelize_watertight_stl(tmp_path) -> None:
    source = tmp_path / "box.stl"
    _write_box(source)

    domain = voxelize_stl(source, pitch=1.0)

    assert domain.source_watertight
    assert domain.source_bodies == 1
    assert len(domain.body_masks) == 1
    assert domain.body_masks[0].sum() == domain.occupied_elements
    assert domain.occupied_elements > 0
    assert domain.occupied.shape == (domain.mesh.num_elements,)
    assert set(np.unique(domain.occupied)) <= {0.0, 1.0}


def test_voxelize_rejects_invalid_pitch(tmp_path) -> None:
    source = tmp_path / "box.stl"
    _write_box(source)

    with pytest.raises(ValueError, match="pitch"):
        voxelize_stl(source, pitch=0)


def test_import_cli_writes_domain_and_metadata(tmp_path, capsys) -> None:
    source = tmp_path / "box.stl"
    output = tmp_path / "imported"
    _write_box(source)

    assert (
        main(
            [
                str(source),
                "--pitch",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads((output / "import-summary.json").read_text())
    assert summary["source_watertight"]
    assert summary["occupied_elements"] > 0
    assert (output / "design-domain.npy").exists()
    assert (output / "body-masks.npz").exists()
    assert (output / "domain.vtk").exists()
    assert (output / "domain-solid.vtk").exists()
    assert "occupied=" in capsys.readouterr().out

import json

import trimesh

from beso.cli import main as optimize_cantilever
from beso.domain_cli import main as optimize_domain
from beso.history import frame_record
from beso.import_cli import main as import_stl


def test_frame_record_always_includes_chart_fields() -> None:
    blank = frame_record(
        None, "000.stl", iteration=0, volume_fraction=1.0, compliance=None
    )
    assert blank["iteration"] == 0
    assert blank["volume_fraction"] == 1.0
    assert blank["compliance"] is None
    filled = frame_record(
        None, "001.stl", iteration=1, volume_fraction=0.42, compliance=1.25
    )
    assert filled["compliance"] == 1.25
    assert filled["volume_fraction"] == 0.42


def test_domain_optimize_writes_stl_history(tmp_path) -> None:
    source = tmp_path / "box.stl"
    imported = tmp_path / "imported"
    output = tmp_path / "history"
    trimesh.creation.box(extents=(4.0, 3.0, 2.0)).export(source)
    assert import_stl([str(source), "--pitch", "1", "--output", str(imported)]) == 0

    assert (
        optimize_domain(
            [
                str(imported),
                "--output",
                str(output),
                "--volume",
                "0.75",
                "--load-body",
                "1",
                "--protect-bottom-layers",
                "0",
                "--evolution-rate",
                "0.5",
                "--max-iterations",
                "2",
                "--snapshot-every",
                "1",
                "--surface-sigma",
                "0.2",
                "--surface-smoothing-iterations",
                "1",
                "--quiet",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert summary["history_frames"] >= 2
    assert (output / "index.html").exists()
    assert (output / "iterations" / "000.stl").exists()
    assert history["frames"][0]["stl"] == "000.stl"
    assert history["frames"][0]["iteration"] == 0
    assert history["overlay"]["loads"]
    assert (output / "overlay.json").exists()
    first_sens = next(frame for frame in history["frames"] if frame.get("sensitivity"))
    assert (output / "iterations" / first_sens["sensitivity"]).exists()


def test_cantilever_writes_stl_history(tmp_path) -> None:
    output = tmp_path / "cantilever"
    assert (
        optimize_cantilever(
            [
                "--nelx",
                "4",
                "--nely",
                "2",
                "--nelz",
                "2",
                "--volume",
                "0.5",
                "--evolution-rate",
                "0.5",
                "--max-iterations",
                "2",
                "--filter-mode",
                "node",
                "--snapshot-every",
                "1",
                "--surface-sigma",
                "0.2",
                "--surface-smoothing-iterations",
                "1",
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert summary["history_frames"] >= 2
    assert (output / "index.html").exists()
    assert (output / "design-smoothed.stl").exists()
    assert (output / "iterations" / "000.stl").exists()
    assert history["frames"][0]["iteration"] == 0
    assert history["overlay"]["support_plane"]["axis"] == "x"
    assert len(history["overlay"]["loads"]) >= 3
    assert any(item.get("resultant") for item in history["overlay"]["loads"])
    assert history["overlay"]["sensitivity"]["kind"] == "node"
    first_sens = next(frame for frame in history["frames"] if frame.get("sensitivity"))
    assert first_sens["sensitivity_shape"] == [5, 3, 3]
    assert (output / "iterations" / first_sens["sensitivity"]).exists()
    assert all("volume_fraction" in frame for frame in history["frames"])
    assert history["frames"][0]["volume_fraction"] == 1.0
    assert any(frame.get("compliance") is not None for frame in history["frames"])
    html = (output / "index.html").read_text()
    assert 'id="show-chart"' in html
    assert 'id="chart-panel"' in html
    assert "drawChart" in html

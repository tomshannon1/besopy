import json

from beso.cli import main


def test_cli_writes_vtk_and_summary_quietly(tmp_path, capsys) -> None:
    output = tmp_path / "result"
    exit_code = main(
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
            "--output",
            str(output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert (output / "design.vtk").exists()
    assert (output / "design-solid.vtk").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["volume_fraction"] == 0.5
    assert summary["iterations"] == 2
    assert summary["mesh"]["element_count"] == 16
    assert summary["mesh"]["node_count"] == 45
    assert summary["mesh"]["dof_count"] == 135
    assert summary["optimization_elapsed_seconds"] > 0
    assert summary["total_elapsed_seconds"] > 0
    assert summary["history"][0]["duration_seconds"] > 0
    assert summary["timings"]["linear_solve_seconds"] > 0
    assert capsys.readouterr().out == ""


def test_cli_reports_runtime_progress(tmp_path, capsys) -> None:
    main(
        [
            "--nelx",
            "2",
            "--nely",
            "1",
            "--nelz",
            "1",
            "--volume",
            "0.5",
            "--evolution-rate",
            "0.5",
            "--max-iterations",
            "2",
            "--output",
            str(tmp_path / "visible"),
        ]
    )

    output = capsys.readouterr().out
    assert "mesh=2x1x1 elements=2 nodes=12 dofs=36" in output
    assert "iter=001/002" in output
    assert "volume=" in output
    assert "compliance=" in output
    assert "changed=" in output
    assert "assembly=" in output
    assert "solve=" in output
    assert "step=" in output
    assert "elapsed=" in output
    assert "eta=" in output
    assert "total=" in output


def test_cli_runs_multiresolution_warm_start(tmp_path, capsys) -> None:
    output = tmp_path / "multiresolution"
    main(
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
            "--coarse-factor",
            "2",
            "--output",
            str(output),
            "--quiet",
        ]
    )

    summary = json.loads((output / "summary.json").read_text())
    assert summary["mode"] == "multiresolution"
    assert summary["coarse"]["mesh"] == [2, 1, 1]
    assert (output / "coarse-design-solid.vtk").exists()
    assert (output / "design-solid.vtk").exists()
    assert capsys.readouterr().out == ""


def test_cli_runs_bridge_problem(tmp_path) -> None:
    output = tmp_path / "bridge"
    exit_code = main(
        [
            "--problem",
            "bridge",
            "--nelx",
            "8",
            "--nely",
            "2",
            "--nelz",
            "3",
            "--volume",
            "0.5",
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
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert summary["problem"] == "bridge"
    assert summary["load"]["placement"] == "convoy"
    assert history["overlay"]["support_piers"]
    assert history["overlay"]["deck"]
    assert len(history["overlay"]["story_loads"]) == 3
    assert (output / "iterations" / "000.stl").exists()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mesh import StructuredMesh
from .optimizer import IterationRecord
from .surface import density_surface, write_binary_stl

_VIEWER_HTML = Path(__file__).with_name("viewer.html")


def snapshot_surface(
    path: str | Path,
    mesh: StructuredMesh,
    densities: NDArray[np.float64],
    *,
    sigma: float,
    smoothing_iterations: int,
) -> Path:
    surface = density_surface(
        mesh,
        densities,
        sigma=sigma,
        smoothing_iterations=smoothing_iterations,
    )
    return write_binary_stl(path, surface)


def should_snapshot(iteration: int, every: int) -> bool:
    if every <= 0:
        return False
    return iteration <= 1 or iteration % every == 0


def overlay_load_arrows(
    mesh: StructuredMesh,
    forces: NDArray[np.float64],
    *,
    max_arrows: int = 16,
) -> list[dict]:
    arrows: list[dict] = []
    for node in range(mesh.num_nodes):
        vector = forces[3 * node : 3 * node + 3]
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 0.0:
            continue
        arrows.append(
            {
                "position": mesh.coordinates[node].tolist(),
                "direction": (vector / magnitude).tolist(),
                "magnitude": magnitude,
            }
        )
    if len(arrows) <= 1:
        return arrows
    magnitudes = np.array([item["magnitude"] for item in arrows], dtype=float)
    positions = np.array([item["position"] for item in arrows], dtype=float)
    directions = np.array([item["direction"] for item in arrows], dtype=float)
    weights = magnitudes / magnitudes.sum()
    centroid = weights @ positions
    mean_direction = weights @ directions
    scale = float(np.linalg.norm(mean_direction)) or 1.0
    displayed = (
        arrows
        if len(arrows) <= max_arrows
        else [arrows[index] for index in np.argsort(magnitudes)[::-1][:max_arrows]]
    )
    displayed.insert(
        0,
        {
            "position": centroid.tolist(),
            "direction": (mean_direction / scale).tolist(),
            "magnitude": float(magnitudes.sum()),
            "resultant": True,
        },
    )
    return displayed


def overlay_support_plane(
    *,
    axis: str,
    value: float,
    width: float,
    height: float,
) -> dict:
    if axis == "x":
        center = [value, width / 2.0, height / 2.0]
    elif axis == "y":
        center = [width / 2.0, value, height / 2.0]
    else:
        center = [width / 2.0, height / 2.0, value]
    return {
        "axis": axis,
        "value": value,
        "center": center,
        "extent_u": width,
        "extent_v": height,
    }


def overlay_support_points(
    points: NDArray[np.float64],
    *,
    max_points: int = 80,
) -> list[list[float]]:
    if len(points) == 0:
        return []
    if len(points) <= max_points:
        return np.asarray(points, dtype=float).tolist()
    stride = max(1, len(points) // max_points)
    return np.asarray(points[::stride][:max_points], dtype=float).tolist()


def sensitivity_overlay(mesh: StructuredMesh, kind: str) -> dict:
    if kind == "node":
        shape = [mesh.nelx + 1, mesh.nely + 1, mesh.nelz + 1]
    else:
        shape = [mesh.nelx, mesh.nely, mesh.nelz]
    return {
        "kind": kind,
        "shape": shape,
        "colormap": "turbo",
    }


def build_overlay(
    mesh: StructuredMesh,
    *,
    forces: NDArray[np.float64],
    sensitivity_kind: str,
    support_plane: dict | None = None,
    support_piers: list[dict] | None = None,
    support_points: NDArray[np.float64] | None = None,
    deck: dict | None = None,
    story_loads: list[dict] | None = None,
    max_arrows: int = 16,
    story: dict | None = None,
) -> dict:
    overlay = {
        "mesh": {
            "elements": [mesh.nelx, mesh.nely, mesh.nelz],
            "dimensions": [mesh.length, mesh.width, mesh.height],
        },
        "loads": overlay_load_arrows(mesh, forces, max_arrows=max_arrows),
        "sensitivity": sensitivity_overlay(mesh, sensitivity_kind),
    }
    if support_plane is not None:
        overlay["support_plane"] = support_plane
    if support_piers:
        overlay["support_piers"] = support_piers
    if deck is not None:
        overlay["deck"] = deck
    if story_loads:
        overlay["story_loads"] = story_loads
    if support_points is not None:
        overlay["support_points"] = overlay_support_points(support_points)
    if story is not None:
        overlay["story"] = story
    return overlay


def write_sensitivity_field(path: str | Path, field: NDArray[np.float64]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(field, dtype=np.float32, order="C").tofile(destination)
    return destination


def write_history_index(
    output: str | Path,
    frames: list[dict],
    *,
    overlay: dict | None = None,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict = {"frames": frames}
    existing = destination / "history.json"
    if overlay is not None:
        payload["overlay"] = overlay
        (destination / "overlay.json").write_text(
            json.dumps(overlay, indent=2) + "\n",
            encoding="utf-8",
        )
    elif existing.exists():
        try:
            previous = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if "overlay" in previous:
            payload["overlay"] = previous["overlay"]
    (destination / "history.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    html = _VIEWER_HTML.read_text(encoding="utf-8")
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "view.html").write_text(html, encoding="utf-8")
    return destination / "index.html"


def frame_record(
    record: IterationRecord | None,
    stl_name: str,
    *,
    iteration: int,
    volume_fraction: float,
    compliance: float | None,
    sensitivity: str | None = None,
    sensitivity_shape: list[int] | None = None,
) -> dict:
    payload = {
        "iteration": iteration,
        "volume_fraction": volume_fraction,
        "compliance": compliance if record is None else record.compliance,
        "changed_elements": 0 if record is None else record.changed_elements,
        "stl": stl_name,
    }
    if sensitivity is not None:
        payload["sensitivity"] = sensitivity
        payload["sensitivity_shape"] = sensitivity_shape
    return payload


class SurfaceHistory:
    def __init__(
        self,
        output: str | Path,
        mesh: StructuredMesh,
        *,
        every: int,
        sigma: float,
        smoothing_iterations: int,
        overlay: dict | None = None,
    ) -> None:
        self.output = Path(output)
        self.mesh = mesh
        self.every = every
        self.sigma = sigma
        self.smoothing_iterations = smoothing_iterations
        self.overlay = overlay
        self.frames: list[dict] = []
        self.iteration_dir = self.output / "iterations"
        if every > 0:
            self.output.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def capture(
        self,
        iteration: int,
        densities: NDArray[np.float64],
        *,
        volume_fraction: float,
        record: IterationRecord | None = None,
        compliance: float | None = None,
        sensitivities: NDArray[np.float64] | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if not force and not should_snapshot(iteration, self.every):
            return
        if self.frames and self.frames[-1]["iteration"] == iteration:
            return
        stl_name = f"{iteration:03d}.stl"
        snapshot_surface(
            self.iteration_dir / stl_name,
            self.mesh,
            densities,
            sigma=self.sigma,
            smoothing_iterations=self.smoothing_iterations,
        )
        sensitivity_name = None
        sensitivity_shape = None
        if sensitivities is not None:
            kind = (self.overlay or {}).get("sensitivity", {}).get("kind", "element")
            if kind == "node":
                sensitivity_shape = [
                    self.mesh.nelx + 1,
                    self.mesh.nely + 1,
                    self.mesh.nelz + 1,
                ]
            else:
                sensitivity_shape = [self.mesh.nelx, self.mesh.nely, self.mesh.nelz]
            field = np.asarray(sensitivities, dtype=float).reshape(sensitivity_shape)
            sensitivity_name = f"{iteration:03d}.sens"
            write_sensitivity_field(self.iteration_dir / sensitivity_name, field)
        self.frames.append(
            frame_record(
                record,
                stl_name,
                iteration=iteration,
                volume_fraction=volume_fraction,
                compliance=compliance,
                sensitivity=sensitivity_name,
                sensitivity_shape=sensitivity_shape,
            )
        )
        write_history_index(self.output, self.frames, overlay=self.overlay)

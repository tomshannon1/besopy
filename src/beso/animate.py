from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from PIL import Image

BACKGROUND = np.array([27, 29, 33], dtype=np.uint8)
SURFACE = np.array([210, 218, 226], dtype=np.float32)
AMBIENT = 0.22
LIGHT = np.array([0.35, -0.55, 0.75], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)


def subsample_indices(count: int, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_frames:
        return list(range(count))
    picked = np.linspace(0, count - 1, max_frames).round().astype(int)
    return [int(index) for index in np.unique(picked)]


def history_stl_paths(folder: Path) -> list[Path]:
    payload = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    paths = []
    for frame in frames:
        stl = frame.get("stl")
        if not stl:
            continue
        path = folder / "iterations" / Path(stl).name
        if path.exists():
            paths.append(path)
    if paths:
        return paths
    return sorted((folder / "iterations").glob("*.stl"))


def domain_bounds(folder: Path, meshes: list[trimesh.Trimesh]) -> NDArray[np.float64]:
    overlay_path = folder / "overlay.json"
    if overlay_path.exists():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        dims = overlay.get("mesh", {}).get("dimensions")
        if dims and len(dims) == 3:
            return np.array([[0.0, 0.0, 0.0], dims], dtype=float)
    valid = [mesh for mesh in meshes if not mesh.is_empty and len(mesh.vertices)]
    if not valid:
        return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    mins = np.vstack([mesh.bounds[0] for mesh in valid])
    maxs = np.vstack([mesh.bounds[1] for mesh in valid])
    return np.vstack((mins.min(axis=0), maxs.max(axis=0)))


def _camera_basis(azimuth_deg: float, elevation_deg: float) -> NDArray[np.float64]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    forward = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=float,
    )
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    scale = np.linalg.norm(right)
    if scale < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= scale
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return np.vstack((right, up, forward))


def _project(
    vertices: NDArray[np.float64],
    bounds: NDArray[np.float64],
    basis: NDArray[np.float64],
    width: int,
    height: int,
) -> NDArray[np.float64]:
    center = bounds.mean(axis=0)
    local = (vertices - center) @ basis.T
    corners = (np.array(
        [
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
        ],
        dtype=float,
    ) - center) @ basis.T
    span = np.abs(corners[:, :2]).max() * 1.18
    span = max(span, 1e-6)
    mapped = np.empty((len(local), 3), dtype=float)
    mapped[:, 0] = (local[:, 0] / span * 0.5 + 0.5) * (width - 1)
    mapped[:, 1] = (0.5 - local[:, 1] / span * 0.5) * (height - 1)
    mapped[:, 2] = local[:, 2]
    return mapped


def _shade(normals: NDArray[np.float64]) -> NDArray[np.uint8]:
    intensity = AMBIENT + (1.0 - AMBIENT) * np.clip(normals @ LIGHT, 0.0, 1.0)
    return np.clip(SURFACE * intensity[:, None], 0, 255).astype(np.uint8)


def render_mesh(
    mesh: trimesh.Trimesh,
    *,
    bounds: NDArray[np.float64],
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
) -> NDArray[np.uint8]:
    image = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    if mesh.is_empty or len(mesh.faces) == 0:
        return image
    basis = _camera_basis(azimuth, elevation)
    projected = _project(np.asarray(mesh.vertices, dtype=float), bounds, basis, width, height)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=float)
    colors = _shade(normals)
    depths = projected[faces, 2]
    order = np.argsort(depths.mean(axis=1))
    zbuf = np.full((height, width), -np.inf, dtype=np.float32)
    for index in order:
        tri = projected[faces[index]]
        color = colors[index]
        _fill_triangle(image, zbuf, tri, color)
    return image


def _fill_triangle(
    image: NDArray[np.uint8],
    zbuf: NDArray[np.float32],
    tri: NDArray[np.float64],
    color: NDArray[np.uint8],
) -> None:
    height, width = zbuf.shape
    min_x = max(int(np.floor(tri[:, 0].min())), 0)
    max_x = min(int(np.ceil(tri[:, 0].max())), width - 1)
    min_y = max(int(np.floor(tri[:, 1].min())), 0)
    max_y = min(int(np.ceil(tri[:, 1].max())), height - 1)
    if max_x < min_x or max_y < min_y:
        return
    x0, y0, z0 = tri[0]
    x1, y1, z1 = tri[1]
    x2, y2, z2 = tri[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-8:
        return
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not np.any(mask):
        return
    depth = w0 * z0 + w1 * z1 + w2 * z2
    window = zbuf[min_y : max_y + 1, min_x : max_x + 1]
    closer = mask & (depth >= window)
    window[closer] = depth[closer]
    image[min_y : max_y + 1, min_x : max_x + 1][closer] = color


def encode_gif(frames: list[NDArray[np.uint8]], destination: Path, fps: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and len(frames) > 1:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = Path(scratch)
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(scratch_path / f"{index:04d}.png")
            palette = scratch_path / "palette.png"
            pattern = str(scratch_path / "%04d.png")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    pattern,
                    "-vf",
                    "palettegen=max_colors=64",
                    str(palette),
                ],
                check=True,
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    pattern,
                    "-i",
                    str(palette),
                    "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=3",
                    str(destination),
                ],
                check=True,
            )
        return destination
    images = [Image.fromarray(frame) for frame in frames]
    duration = max(1, round(1000 / fps))
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return destination


def animate_history(
    folder: Path,
    destination: Path,
    *,
    width: int = 480,
    height: int = 300,
    max_frames: int = 40,
    fps: int = 12,
    azimuth: float = 42.0,
    elevation: float = 28.0,
    yaw_span: float = 16.0,
) -> Path:
    paths = history_stl_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No STL frames in {folder / 'iterations'}")
    chosen = [paths[index] for index in subsample_indices(len(paths), max_frames)]
    meshes = [trimesh.load_mesh(path, process=False) for path in chosen]
    bounds = domain_bounds(folder, meshes)
    frames: list[NDArray[np.uint8]] = []
    count = max(len(meshes) - 1, 1)
    for index, mesh in enumerate(meshes):
        yaw = azimuth - yaw_span / 2 + yaw_span * index / count
        frames.append(
            render_mesh(
                mesh,
                bounds=bounds,
                width=width,
                height=height,
                azimuth=yaw,
                elevation=elevation,
            )
        )
    return encode_gif(frames, destination, fps=fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beso-animate",
        description="Render a looping GIF from BESO STL iteration history.",
    )
    parser.add_argument("history", type=Path, help="Output folder with history.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=12)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    path = animate_history(
        args.history,
        args.output,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

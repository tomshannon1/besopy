from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mesh import StructuredMesh


@dataclass(frozen=True)
class LoadCase:
    forces: NDArray[np.float64]
    fixed_dofs: NDArray[np.int64]
    load_dof: int
    load_nodes: NDArray[np.int64]


def cantilever_load_case(
    mesh: StructuredMesh,
    load: float = -1.0,
    direction: int = 2,
    *,
    placement: str = "bottom-edge",
) -> LoadCase:
    """Clamp x=0 and apply a total load on the free end.

    ``bottom-edge`` spreads the load uniformly across the nodes on the
    free-end bottom edge (x = length, z = 0). ``face-center`` applies the
    whole load at the centre of the free face, matching the classic point
    load used for Euler-Bernoulli checks.
    """

    if direction not in (0, 1, 2):
        raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")
    if placement not in {"bottom-edge", "face-center"}:
        raise ValueError("Load placement must be 'bottom-edge' or 'face-center'")

    tip_nodes = mesh.nodes_on_x(mesh.length)
    tip_coordinates = mesh.coordinates[tip_nodes]
    if placement == "bottom-edge":
        z_min = float(tip_coordinates[:, 2].min())
        load_nodes = tip_nodes[np.isclose(tip_coordinates[:, 2], z_min)]
        order = np.argsort(mesh.coordinates[load_nodes, 1])
        load_nodes = load_nodes[order]
    else:
        target = np.array([mesh.length, mesh.width / 2.0, mesh.height / 2.0])
        load_node = int(
            tip_nodes[np.argmin(np.linalg.norm(tip_coordinates - target, axis=1))]
        )
        load_nodes = np.array([load_node], dtype=np.int64)

    share = load / len(load_nodes)
    forces = np.zeros(mesh.num_dofs, dtype=float)
    forces[3 * load_nodes + direction] = share
    mid = load_nodes[len(load_nodes) // 2]
    return LoadCase(
        forces=forces,
        fixed_dofs=mesh.fixed_dofs_at_x(),
        load_dof=int(3 * mid + direction),
        load_nodes=np.asarray(load_nodes, dtype=np.int64),
    )


CANTILEVER_PLACEMENTS = ("bottom-edge", "face-center")
BRIDGE_PLACEMENTS = ("convoy", "deck", "eccentric", "midspan")
ALL_LOAD_PLACEMENTS = CANTILEVER_PLACEMENTS + BRIDGE_PLACEMENTS


def resolve_load_placement(problem: str, placement: str | None) -> str:
    if problem == "cantilever":
        chosen = placement or "bottom-edge"
        if chosen not in CANTILEVER_PLACEMENTS:
            raise ValueError(
                "Cantilever load placement must be 'bottom-edge' or 'face-center'"
            )
        return chosen
    chosen = placement or "convoy"
    if chosen not in BRIDGE_PLACEMENTS:
        raise ValueError(
            "Bridge load placement must be 'convoy', 'deck', 'eccentric', or 'midspan'"
        )
    return chosen


def _nearest_x(mesh: StructuredMesh, target: float) -> float:
    xs = np.unique(mesh.coordinates[:, 0])
    return float(xs[np.argmin(np.abs(xs - target))])


def _top_nodes(mesh: StructuredMesh) -> NDArray[np.int64]:
    return mesh.nodes_on_z(mesh.height)


def _nodes_at_x(mesh: StructuredMesh, nodes: NDArray[np.int64], x: float) -> NDArray[np.int64]:
    return nodes[np.isclose(mesh.coordinates[nodes, 0], x)]


def _center_strip(mesh: StructuredMesh, nodes: NDArray[np.int64], fraction: float = 0.45) -> NDArray[np.int64]:
    y = mesh.coordinates[nodes, 1]
    half = mesh.width * fraction / 2.0
    mid = mesh.width / 2.0
    return nodes[(y >= mid - half) & (y <= mid + half)]


def _edge_strip(mesh: StructuredMesh, nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    y = mesh.coordinates[nodes, 1]
    return nodes[np.isclose(y, float(y.min()))]


def _force_direction(direction: int, load: float) -> list[float]:
    vector = [0.0, 0.0, 0.0]
    vector[direction] = 1.0 if load >= 0.0 else -1.0
    return vector


def _station_groups(
    mesh: StructuredMesh, nodes: NDArray[np.int64], count: int = 5
) -> list[NDArray[np.int64]]:
    stations = np.unique(mesh.coordinates[nodes, 0])
    n_stations = min(count, len(stations))
    indices = np.unique(
        np.linspace(0, len(stations) - 1, n_stations).round().astype(int)
    )
    return [_nodes_at_x(mesh, nodes, float(stations[index])) for index in indices]


def _group_story_loads(
    mesh: StructuredMesh,
    groups: list[NDArray[np.int64]],
    *,
    load: float,
    direction: int,
) -> list[dict]:
    unit = _force_direction(direction, load)
    total = sum(len(group) for group in groups if len(group))
    stories = []
    for group in groups:
        if len(group) == 0:
            continue
        stories.append(
            {
                "position": mesh.coordinates[group].mean(axis=0).tolist(),
                "direction": unit,
                "magnitude": abs(load) * len(group) / max(total, 1),
                "resultant": True,
            }
        )
    return stories


def bridge_load_case(
    mesh: StructuredMesh,
    load: float = -1.0,
    direction: int = 2,
    *,
    placement: str = "convoy",
) -> tuple[LoadCase, list[dict]]:
    """Abutments at both bottom ends, with a cooler deck traffic pattern.

    ``convoy`` — three axle lines at L/4, L/2, 3L/4 on the centre roadway.
    ``deck`` — distributed traffic along the full-length centre strip.
    ``eccentric`` — all traffic on one fascia (torsion).
    ``midspan`` — a single axle line at mid-span.
    """

    if direction not in (0, 1, 2):
        raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")
    if placement not in BRIDGE_PLACEMENTS:
        raise ValueError(
            "Bridge load placement must be 'convoy', 'deck', 'eccentric', or 'midspan'"
        )
    bottom = mesh.nodes_on_z(0.0)
    left = np.intersect1d(mesh.nodes_on_x(0.0), bottom, assume_unique=True)
    right = np.intersect1d(mesh.nodes_on_x(mesh.length), bottom, assume_unique=True)
    support_nodes = np.unique(np.concatenate((left, right)))
    top = _top_nodes(mesh)
    if placement == "midspan":
        load_groups = [_nodes_at_x(mesh, top, _nearest_x(mesh, mesh.length / 2.0))]
        story_groups = load_groups
    elif placement == "deck":
        strip = _center_strip(mesh, top)
        load_groups = [strip]
        story_groups = _station_groups(mesh, strip)
    elif placement == "eccentric":
        strip = _edge_strip(mesh, top)
        load_groups = [strip]
        story_groups = _station_groups(mesh, strip)
    else:
        strip = _center_strip(mesh, top)
        load_groups = [
            _nodes_at_x(mesh, strip, _nearest_x(mesh, mesh.length * fraction))
            for fraction in (0.25, 0.50, 0.75)
        ]
        story_groups = load_groups
    load_nodes = np.unique(
        np.concatenate([group for group in load_groups if len(group)])
    )
    if load_nodes.size == 0:
        raise ValueError("Bridge load case found no loaded deck nodes")
    share = load / len(load_nodes)
    forces = np.zeros(mesh.num_dofs, dtype=float)
    forces[3 * load_nodes + direction] = share
    mid = load_nodes[len(load_nodes) // 2]
    load_case = LoadCase(
        forces=forces,
        fixed_dofs=(3 * support_nodes[:, None] + np.arange(3, dtype=np.int64)).reshape(
            -1
        ),
        load_dof=int(3 * mid + direction),
        load_nodes=np.asarray(load_nodes, dtype=np.int64),
    )
    return load_case, _group_story_loads(
        mesh, story_groups, load=load, direction=direction
    )


def _elements_touching(
    mesh: StructuredMesh, nodes: NDArray[np.int64]
) -> NDArray[np.bool_]:
    return np.isin(mesh.connectivity, nodes).any(axis=1)


def _bridge_piers(mesh: StructuredMesh) -> list[dict]:
    dx, _, dz = mesh.element_size
    height = max(dz * 1.6, mesh.height * 0.12)
    width = mesh.width * 1.15
    thickness = max(dx * 1.3, mesh.length * 0.03)
    return [
        {
            "center": [0.0, mesh.width / 2.0, height / 2.0],
            "size": [thickness, width, height],
        },
        {
            "center": [mesh.length, mesh.width / 2.0, height / 2.0],
            "size": [thickness, width, height],
        },
    ]


def _top_layer_elements(mesh: StructuredMesh) -> NDArray[np.bool_]:
    return (np.arange(mesh.num_elements) % mesh.nelz) == (mesh.nelz - 1)


def _bridge_deck(mesh: StructuredMesh, placement: str) -> dict:
    _, dy, dz = mesh.element_size
    if placement == "eccentric":
        return {
            "center": [mesh.length / 2.0, dy * 0.6, mesh.height],
            "size": [mesh.length, min(mesh.width * 0.4, 3 * dy), dz * 0.4],
        }
    return {
        "center": [mesh.length / 2.0, mesh.width / 2.0, mesh.height],
        "size": [mesh.length, mesh.width * 0.5, dz * 0.4],
    }


_BRIDGE_STORY = {
    "convoy": {
        "load_label": "TRAFFIC  ·  three axles",
        "caption": "Piers are fixed. Three axle loads sit on a live roadway — a convoy, not one mid-span punch.",
    },
    "deck": {
        "load_label": "TRAFFIC  ·  distributed deck",
        "caption": "Piers are fixed. Traffic is spread along the whole centre lane of the deck.",
    },
    "eccentric": {
        "load_label": "OFFSET TRAFFIC  ·  one fascia",
        "caption": "Piers are fixed. All traffic is on one edge, so the span has to carry torsion as well as bending.",
    },
    "midspan": {
        "load_label": "DECK LOAD  ·  mid-span",
        "caption": "Piers cannot move. The hanging mass is traffic on the mid-span deck.",
    },
}


@dataclass(frozen=True)
class OptimizationProblem:
    name: str
    load_case: LoadCase
    overlay_kwargs: dict
    passive_solid: NDArray[np.bool_] | None = None
    connectivity_seeds: NDArray[np.bool_] | None = None
    require_connectivity: bool = False


def make_problem(
    name: str,
    mesh: StructuredMesh,
    *,
    load: float,
    load_placement: str | None,
) -> OptimizationProblem:
    placement = resolve_load_placement(name, load_placement)
    if name == "cantilever":
        load_case = cantilever_load_case(mesh, load=load, placement=placement)
        return OptimizationProblem(
            name=name,
            load_case=load_case,
            overlay_kwargs={
                "support_plane": {
                    "axis": "x",
                    "value": 0.0,
                    "center": [0.0, mesh.width / 2.0, mesh.height / 2.0],
                    "extent_u": mesh.width,
                    "extent_v": mesh.height,
                },
                "story": {
                    "support_label": "FIXED WALL  ·  nodes clamped",
                    "load_label": "APPLIED LOAD  ·  free end",
                    "caption": "Wall = nodes fixed in place. Hanging mass = where the force is applied.",
                },
            },
        )
    if name == "bridge":
        load_case, story_loads = bridge_load_case(
            mesh, load=load, placement=placement
        )
        deck = _top_layer_elements(mesh)
        seeds = deck | _elements_touching(
            mesh,
            np.unique(
                np.concatenate((load_case.load_nodes, load_case.fixed_dofs // 3))
            ),
        )
        copy = _BRIDGE_STORY[placement]
        return OptimizationProblem(
            name=name,
            load_case=load_case,
            overlay_kwargs={
                "support_piers": _bridge_piers(mesh),
                "deck": _bridge_deck(mesh, placement),
                "story_loads": story_loads,
                "max_arrows": 32,
                "story": {
                    "support_label": "ABUTMENTS  ·  piers fixed",
                    "load_label": copy["load_label"],
                    "caption": copy["caption"],
                },
            },
            passive_solid=deck,
            connectivity_seeds=seeds,
            require_connectivity=True,
        )
    raise ValueError("Problem must be 'cantilever' or 'bridge'")

"""Readers for the raw PathSnap files (one folder per trip, four headerless whitespace files).

    <trip>.track   lon lat t          GPS observations, t in seconds
    <trip>.nodes   lon lat            road-network nodes, row index = node id (0-based)
    <trip>.arcs    from_node to_node  directed road segments, row index = arc id (0-based)
    <trip>.route   arc_id             reference route, chronological (training / student_test only)

Development code only ever reads the `train` and `student_test` partitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    nodes: np.ndarray          # (n_nodes, 2) lon, lat
    arcs: np.ndarray           # (n_arcs, 2) from_node, to_node
    route: np.ndarray | None   # reference arc ids, or None when not loaded

    @property
    def n_gps(self) -> int:
        return len(self.t)


def _table(path: Path, ncols: int, dtype) -> np.ndarray:
    try:
        arr = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
    except pd.errors.EmptyDataError:
        return np.empty((0, ncols), dtype=dtype)
    return arr


def list_trips(root: str | Path, partition: str) -> list[str]:
    return sorted(p.name for p in (Path(root) / partition).iterdir() if p.is_dir())


def load_trip_dir(trip_dir: str | Path, trip_id: str, partition: str = "", with_route: bool = True) -> Trip:
    """Load one trip from a folder laid out as <trip_dir>/<trip_id>.{track,nodes,arcs,route}.

    The API uses this with with_route=False for inference; the reference route is only
    read server-side, after prediction, to score it.
    """
    base = Path(trip_dir) / trip_id
    track = _table(base.with_suffix(".track"), 3, np.float64)
    nodes = _table(base.with_suffix(".nodes"), 2, np.float64)
    arcs = _table(base.with_suffix(".arcs"), 2, np.int64)
    route = None
    if with_route and base.with_suffix(".route").exists():
        route = _table(base.with_suffix(".route"), 1, np.int64).ravel()
    return Trip(trip_id, partition, track[:, 0], track[:, 1], track[:, 2], nodes, arcs, route)


def load_route(trip_dir: str | Path, trip_id: str) -> np.ndarray | None:
    """Reference route only (server-side scoring). None if the trip has no .route file."""
    path = Path(trip_dir) / f"{trip_id}.route"
    return _table(path, 1, np.int64).ravel() if path.exists() else None


def load_trip(root: str | Path, partition: str, trip_id: str, with_route: bool = True) -> Trip:
    return load_trip_dir(Path(root) / partition / trip_id, trip_id, partition, with_route)

"""Readers for the raw PathSnap files (one folder per trip, four headerless whitespace files).

    <trip>.track   lon lat t          GPS observations, t in seconds
    <trip>.nodes   lon lat            road-network nodes, row index = node id (0-based)
    <trip>.arcs    from_node to_node  directed road segments, row index = arc id (0-based)
    <trip>.route   arc_id             reference route, chronological (training / student_test only)

Development code only ever reads the `train` and `student_test` partitions.

Robustness: GPS rows with a missing/non-finite value are dropped; unreadable or inconsistent files raise a ValueError whose
message names the file (never its path) so the API can answer with a clear 422 instead of crashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test")


class BadTripData(ValueError):
    """A trip's input files are unreadable or inconsistent (the API answers 422 with the message)."""


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


def trip_file(trip_dir: str | Path, trip_id: str, ext: str) -> Path:
    """<trip_dir>/<trip_id>.<ext>. Built by string join (not with_suffix) so ids that contain dots still work."""
    return Path(trip_dir) / f"{trip_id}.{ext}"


def _table(path: Path, ncols: int, dtype) -> np.ndarray:
    try:
        arr = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
    except pd.errors.EmptyDataError:
        return np.empty((0, ncols), dtype=dtype)
    except (ValueError, pd.errors.ParserError):
        raise BadTripData(f"cannot read {path.name}: it is not a numeric table") from None
    if arr.shape[1] < ncols:
        raise BadTripData(f"cannot read {path.name}: expected {ncols} columns, found {arr.shape[1]}")
    return arr


def list_trips(root: str | Path, partition: str) -> list[str]:
    return sorted(p.name for p in (Path(root) / partition).iterdir() if p.is_dir())


def load_trip_dir(trip_dir: str | Path, trip_id: str, partition: str = "", with_route: bool = True) -> Trip:
    """Load one trip from a folder laid out as <trip_dir>/<trip_id>.{track,nodes,arcs,route}.

    The API uses this with with_route=False for inference; the reference route is only
    read server-side, after prediction, to score it.
    """
    track = _table(trip_file(trip_dir, trip_id, "track"), 3, np.float64)[:, :3]
    track = track[np.isfinite(track).all(axis=1)]                      # drop rows with a missing / non-finite value
    nodes = _table(trip_file(trip_dir, trip_id, "nodes"), 2, np.float64)[:, :2]
    arcs = _table(trip_file(trip_dir, trip_id, "arcs"), 2, np.int64)[:, :2]
    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise BadTripData("the .arcs file refers to node ids that are not in the .nodes file")
    route = load_route(trip_dir, trip_id) if with_route else None
    return Trip(trip_id, partition, track[:, 0], track[:, 1], track[:, 2], nodes, arcs, route)


def load_route(trip_dir: str | Path, trip_id: str) -> np.ndarray | None:
    """Reference route only (server-side scoring). None if the trip has no usable .route file."""
    path = trip_file(trip_dir, trip_id, "route")
    if not path.exists():
        return None
    try:
        arr = _table(path, 1, np.float64).ravel()                        # tolerate "12.0"-style ids
    except BadTripData:
        return None
    arr = arr[np.isfinite(arr)]
    return arr.astype(np.int64)


def load_trip(root: str | Path, partition: str, trip_id: str, with_route: bool = True) -> Trip:
    return load_trip_dir(Path(root) / partition / trip_id, trip_id, partition, with_route)

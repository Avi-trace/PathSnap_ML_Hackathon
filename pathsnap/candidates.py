"""Candidate generation: for every GPS observation, the directed arcs within `radius` metres.

Arcs are straight segments between two nodes, and some are long (highways), so indexing arc midpoints alone
would miss arcs that pass close to a point but have a far-away midpoint. Each arc is therefore split into
pieces of at most `piece_m` metres; a KD-tree over piece midpoints is queried with radius + max piece
half-length (a superset), and the exact point-to-segment distance filters the result. Nothing within
`radius` of an observation can be missed by construction.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment


@dataclass
class ArcIndex:
    tree: cKDTree
    piece_arc: np.ndarray
    pad: float
    N: np.ndarray       # node xy (m)
    A: np.ndarray       # arc from-node
    B: np.ndarray       # arc to-node

    @property
    def n_arcs(self) -> int:
        return len(self.A)


def build_arc_index(N: np.ndarray, A: np.ndarray, B: np.ndarray, piece_m: float = 40.0) -> ArcIndex:
    a, b = N[A], N[B]
    length = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(length / piece_m).astype(np.int64))
    arc = np.repeat(np.arange(len(A)), n)
    start = np.cumsum(n) - n
    j = np.arange(n.sum()) - np.repeat(start, n)
    f = (j + 0.5) / np.repeat(n, n)
    mids = a[arc] + (b[arc] - a[arc]) * f[:, None]
    pad = float((length / n / 2.0).max()) if len(A) else 0.0
    return ArcIndex(cKDTree(mids), arc, pad, N, A, B)


def generate(G: np.ndarray, idx: ArcIndex, radius: float):
    """Return (obs, arc, dist_m, frac) for every (observation, arc) pair with dist <= radius."""
    if len(G) == 0 or idx.n_arcs == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0), np.empty(0)
    lists = idx.tree.query_ball_point(G, r=radius + idx.pad, workers=-1)
    lens = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
    total = int(lens.sum())
    if total == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0), np.empty(0)
    obs = np.repeat(np.arange(len(G), dtype=np.int64), lens)
    pieces = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=total)
    key = np.unique(obs * idx.n_arcs + idx.piece_arc[pieces])     # one row per (obs, arc)
    obs, arc = key // idx.n_arcs, key % idx.n_arcs
    dist, frac = point_segment(G[obs], idx.N[idx.A[arc]], idx.N[idx.B[arc]])
    keep = dist <= radius
    return obs[keep], arc[keep], dist[keep], frac[keep]

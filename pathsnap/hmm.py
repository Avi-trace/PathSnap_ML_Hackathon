"""HMM / Viterbi route reconstruction on top of the candidate classifier.

Hidden state      the candidate directed arc of an observation (one state per candidate row, so the
                  candidate -> road-segment correspondence is preserved by construction)
Emission          log q_ic,   q_ic = p_ic / sum_c p_ic   (classifier probability normalised per observation),
                  scaled by `alpha`
Transition        Newson-Krumm: log T = -| d_gps - d_route | / beta, where d_gps is the straight-line distance
                  between consecutive observations and d_route is the distance along the road network between the
                  two candidate positions (forward-only along directed arcs, bounded Dijkstra)
Unreachable pair  no directed path within the search limit -> finite penalty `invalid_pen` (never -inf, so Viterbi
                  can always continue and a missing candidate cannot dead-end the decode)
GPS gap           consecutive observations > gap_s seconds or > reset_d metres apart: transitions are uniform, i.e.
                  the chain restarts from the emissions alone
Observation       with no candidate is dropped from the chain and reported as unmatched (-1)

Viterbi: delta_1 = emit_1; delta_{k+1}(j) = emit_{k+1}(j) + max_i [ delta_k(i) + logT_k(i, j) ]; backtrack the argmax.
The expensive part (network distances) depends only on the candidates, so it is computed once per trip
(`prepare_trip`) and reused for every parameter setting during decoder tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

BACKWARD_TOL_M = 15.0     # allowed backward projection jitter on the same arc (GPS noise when slow / stopped)


@dataclass
class Prepared:
    obs: np.ndarray        # observation index of each chain step (m,)
    starts: np.ndarray     # row offsets of each step's candidates (m+1,)
    dg: np.ndarray         # straight-line distance between consecutive chain observations (m-1,)
    D: list                # per step: float32 (K_i, K_{i+1}) route distances (inf = unreachable) or None (gap -> reset)
    graph: object = None   # local directed road graph (csr), kept to gap-fill the output route
    new_id: object = None  # original node id -> local graph index (-1 = outside)


def local_graph(N: np.ndarray, A: np.ndarray, B: np.ndarray, G: np.ndarray, buffer_m: float = 300.0):
    """Directed length-weighted graph restricted to nodes within buffer_m of the GPS trace (keeps Dijkstra cheap)."""
    d, _ = cKDTree(G).query(N, distance_upper_bound=buffer_m, workers=-1)
    keep = np.isfinite(d)
    new_id = np.full(len(N), -1, dtype=np.int64)
    new_id[keep] = np.arange(int(keep.sum()))
    m = keep[A] & keep[B]
    a, b = new_id[A[m]], new_id[B[m]]
    w = np.maximum(np.linalg.norm(N[B[m]] - N[A[m]], axis=1), 1e-3)
    n = int(keep.sum())
    order = np.argsort(w)                                   # duplicate arcs: keep the shortest
    _, first = np.unique(a[order] * n + b[order], return_index=True)
    sel = order[first]
    return csr_matrix((w[sel], (a[sel], b[sel])), shape=(n, n)), new_id


def prepare_trip(G, t, obs, arc, frac, alen, N, A, B, gap_s: float = 15.0, reset_d: float = 500.0) -> Prepared:
    """obs/arc/frac/alen: candidate rows sorted by observation. Computes all network distances once."""
    graph, new_id = local_graph(N, A, B, G)
    uo, starts = np.unique(obs, return_index=True)
    starts = np.r_[starts, len(obs)]
    m = len(uo)
    dg = np.linalg.norm(np.diff(G[uo], axis=0), axis=1) if m > 1 else np.empty(0)
    D = []
    for s in range(m - 1):
        dt = t[uo[s + 1]] - t[uo[s]]
        if dt > gap_s or dg[s] > reset_d:
            D.append(None)
            continue
        i1, i2 = slice(starts[s], starts[s + 1]), slice(starts[s + 1], starts[s + 2])
        a1, a2, f1, f2, L1, L2 = arc[i1], arc[i2], frac[i1], frac[i2], alen[i1], alen[i2]
        src, inv = np.unique(new_id[B[a1]], return_inverse=True)
        tgt = new_id[A[a2]]
        if (src < 0).any() or (tgt < 0).any():              # candidate endpoint outside the local graph
            D.append(None)
            continue
        sp = dijkstra(graph, directed=True, indices=src, limit=3.0 * dg[s] + 300.0)[inv][:, tgt]
        d = (1.0 - f1)[:, None] * L1[:, None] + sp + (f2 * L2)[None, :]
        same = a1[:, None] == a2[None, :]
        if same.any():
            fwd = (f2[None, :] - f1[:, None]) * L1[:, None]
            d = np.where(same, np.where(fwd >= -BACKWARD_TOL_M, np.maximum(fwd, 0.0), np.inf), d)
        D.append(d.astype(np.float32))
    return Prepared(uo, starts, dg, D, graph, new_id)


def decode(prep: Prepared, p: np.ndarray, alpha: float = 1.0, beta: float = 10.0, invalid_pen: float = 20.0) -> np.ndarray:
    """Viterbi. Returns, per chain step, the chosen row index into the candidate arrays."""
    st, m = prep.starts, len(prep.obs)
    q = p / np.maximum(np.add.reduceat(p, st[:-1]), 1e-12).repeat(np.diff(st))
    emit = alpha * np.log(np.clip(q, 1e-9, 1.0))
    score = emit[st[0]:st[1]]
    back = []
    for s in range(m - 1):
        D = prep.D[s]
        k1, k2 = st[s + 1] - st[s], st[s + 2] - st[s + 1]
        if D is None:
            logT = np.zeros((k1, k2))
        else:
            logT = np.where(np.isfinite(D), -np.abs(prep.dg[s] - D) / beta, -invalid_pen)
        cand = score[:, None] + logT
        bp = cand.argmax(axis=0)
        score = cand[bp, np.arange(k2)] + emit[st[s + 1]:st[s + 2]]
        back.append(bp)
    path = np.empty(m, dtype=np.int64)
    path[-1] = int(score.argmax())
    for s in range(m - 2, -1, -1):
        path[s] = back[s][path[s + 1]]
    return st[:-1] + path

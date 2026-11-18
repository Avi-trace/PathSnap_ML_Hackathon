"""Per-observation reference labels from the ordered reference route.

The data only gives the route as an ordered list of arc ids, not which arc each GPS point belongs to. We
align the observations to the route with a monotone dynamic programme:

    minimise   sum_i  min(dist(G_i, route_arc[j_i]), CAP)  +  eps * (j_i - j_{i-1})
    subject to j_i >= j_{i-1}            (the vehicle never moves backwards along its own route)

The monotone constraint resolves routes that revisit or run parallel to themselves, which a plain
"nearest route arc" would get wrong. `eps` is a tiny per-step penalty that prefers staying on an arc
over jumping ahead when distances tie. The recurrence is a prefix-minimum, so each observation costs
O(route length) with numpy and the whole alignment is O(n_obs * n_route).
"""
from __future__ import annotations

import numpy as np

CAP_M = 200.0


def _row_costs(P: np.ndarray, a: np.ndarray, ab: np.ndarray, den: np.ndarray) -> np.ndarray:
    t = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(-1) / den[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.minimum(np.linalg.norm(P[:, None, :] - proj, axis=-1), CAP_M)


def align_to_route(G: np.ndarray, N: np.ndarray, A: np.ndarray, B: np.ndarray, route: np.ndarray,
                   eps: float = 0.2, chunk: int = 128):
    """Return (ref_arc, ref_dist_m) for every observation: the aligned route arc and its distance."""
    n, m = len(G), len(route)
    a, b = N[A[route]], N[B[route]]
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-9)
    ar = np.arange(m)
    adj = eps * ar
    bp = np.zeros((n, m), dtype=np.int32 if m > 32000 else np.int16)

    prev = None
    for s in range(0, n, chunk):
        cost = _row_costs(G[s:s + chunk], a, ab, den)
        for k in range(len(cost)):
            i = s + k
            if i == 0:
                prev = cost[0]
                continue
            v = prev - adj
            pm = np.minimum.accumulate(v)
            bp[i] = np.maximum.accumulate(np.where(v <= pm, ar, 0))
            prev = cost[k] + adj + pm

    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]
    ref = route[j]
    d = np.empty(n)
    for s in range(0, n, 4096):
        sl = slice(s, s + 4096)
        t = np.clip(((G[sl] - a[j[sl]]) * ab[j[sl]]).sum(1) / den[j[sl]], 0.0, 1.0)
        d[sl] = np.linalg.norm(G[sl] - (a[j[sl]] + t[:, None] * ab[j[sl]]), axis=1)
    return ref, d

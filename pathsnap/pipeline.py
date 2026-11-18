"""End-to-end inference: raw trip files -> candidates -> classifier -> HMM/Viterbi -> route (+ server-side scoring).

`predict_trip` never sees the reference route. `score_prediction` is the only place that does, and it runs
server-side after the prediction exists; its output is aggregate numbers only (ground truth is never sent to the UI).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from .features import ELIGIBLE_M, FEATURES, RADIUS, candidate_table
from .geo import Projector
from .hmm import Prepared, decode, prepare_trip
from .io import Trip
from .labels import align_to_route
from .metrics import transition_flags

DEFAULT_HMM = dict(alpha=1.0, beta=10.0, invalid_pen=20.0, gap_s=15.0, reset_d=500.0)


@dataclass
class Bundle:
    """Everything inference needs: classifier, feature list, candidate radius, decoder parameters."""
    model: object
    features: list = field(default_factory=lambda: list(FEATURES))
    radius: float = RADIUS
    hmm: dict = field(default_factory=lambda: dict(DEFAULT_HMM))
    meta: dict = field(default_factory=dict)


def save_bundle(bundle: Bundle, path) -> None:
    joblib.dump(bundle, path, compress=3)


def load_bundle(path) -> Bundle:
    return joblib.load(path)


@dataclass
class Prediction:
    trip_id: str
    n_obs: int
    clf_arc: np.ndarray        # classifier-only arc per observation (-1 = no candidate)
    hmm_arc: np.ndarray        # HMM/Viterbi arc per observation (-1 = unmatched)
    cand_obs: np.ndarray       # candidate rows, kept for scoring the candidate miss rate
    cand_arc: np.ndarray
    G: np.ndarray              # projected observations (m)
    N: np.ndarray              # projected nodes (m)
    prep: Prepared | None
    timings_ms: dict


def predict_trip(bundle: Bundle, trip: Trip) -> Prediction:
    t0 = time.perf_counter()
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    G = proj.xy(trip.lon, trip.lat)
    N = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    rows = candidate_table(G, trip.t, N, A, B, bundle.radius)
    t1 = time.perf_counter()
    n = trip.n_gps
    clf, hmm = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    if len(rows) == 0:
        return Prediction(trip.trip_id, n, clf, hmm, np.empty(0, np.int64), np.empty(0, np.int64), G, N, None,
                          {"candidates": (t1 - t0) * 1e3, "classifier": 0.0, "hmm": 0.0})
    p = bundle.model.predict_proba(rows[bundle.features].to_numpy(np.float32))[:, 1]
    t2 = time.perf_counter()
    obs, arc = rows["obs"].to_numpy(), rows["arc"].to_numpy()
    win = pd.Series(p).groupby(obs).idxmax().to_numpy()
    clf[obs[win]] = arc[win]
    h = bundle.hmm
    prep = prepare_trip(G, trip.t, obs, arc, (rows["along_m"] / rows["arc_len"]).to_numpy(np.float64),
                        rows["arc_len"].to_numpy(np.float64), N, A, B, h["gap_s"], h["reset_d"])
    choice = decode(prep, p, h["alpha"], h["beta"], h["invalid_pen"])
    hmm[prep.obs] = arc[choice]
    t3 = time.perf_counter()
    return Prediction(trip.trip_id, n, clf, hmm, obs, arc, G, N, prep,
                      {"candidates": (t1 - t0) * 1e3, "classifier": (t2 - t1) * 1e3, "hmm": (t3 - t2) * 1e3})


def _continuity(arcs: np.ndarray, A, B) -> float:
    a = arcs[arcs >= 0]
    flags, _ = transition_flags(a, A, B)
    return float(flags.mean()) if len(flags) else 1.0


def score_prediction(pred: Prediction, trip: Trip) -> dict:
    """Server-side scoring against the reference route (needs trip.route). Aggregate numbers only."""
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    ref, rd = align_to_route(pred.G, pred.N, A, B, trip.route)
    elig = rd <= ELIGIBLE_M
    n_elig = int(elig.sum())
    n_arcs = len(A)
    cand_keys = np.unique(pred.cand_obs.astype(np.int64) * n_arcs + pred.cand_arc)
    ref_keys = np.arange(pred.n_obs, dtype=np.int64) * n_arcs + ref
    in_cands = np.isin(ref_keys, cand_keys)
    ref_set = set(trip.route.tolist())

    def f1(arcs):
        s = set(arcs[arcs >= 0].tolist())
        tp = len(s & ref_set)
        pr, rc = tp / max(len(s), 1), tp / max(len(ref_set), 1)
        return 2 * pr * rc / max(pr + rc, 1e-12)

    return {
        "n_obs": n_elig,
        "pointwise_acc_classifier": float((pred.clf_arc[elig] == ref[elig]).mean()),
        "pointwise_acc_hmm": float((pred.hmm_arc[elig] == ref[elig]).mean()),
        "route_continuity_classifier": _continuity(pred.clf_arc, A, B),
        "route_continuity_hmm": _continuity(pred.hmm_arc, A, B),
        "route_f1_classifier": f1(pred.clf_arc),
        "route_f1_hmm": f1(pred.hmm_arc),
        "candidate_miss_rate": float((~in_cands[elig]).mean()),
        "latency_ms": {k: int(round(v)) for k, v in pred.timings_ms.items()},
    }


# ---------------------------------------------------------------- geometry for the UI (no ground truth involved)
def _r5(x) -> list:
    return np.round(np.asarray(x, dtype=np.float64), 5).tolist()


def _fill_path(prep: Prepared, u: int, v: int) -> list:
    """Original node ids of the shortest directed path u -> v inside the local graph ([] if none)."""
    su, sv = prep.new_id[u], prep.new_id[v]
    if su < 0 or sv < 0:
        return []
    dist, pred = dijkstra(prep.graph, directed=True, indices=int(su), return_predecessors=True, limit=1500.0)
    if not np.isfinite(dist[sv]):
        return []
    inv = np.flatnonzero(prep.new_id >= 0)
    path, cur = [], int(sv)
    while cur != su and cur >= 0:
        path.append(int(inv[cur]))
        cur = int(pred[cur])
    path.append(int(inv[int(su)]))
    return path[::-1]


def route_polyline(arcs_per_obs: np.ndarray, trip: Trip, prep: Prepared | None) -> list:
    """Ordered lon/lat polyline of a predicted arc sequence; breaks between arcs are bridged by shortest paths."""
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    a = arcs_per_obs[arcs_per_obs >= 0]
    if len(a) == 0:
        return []
    seq = a[np.r_[True, a[1:] != a[:-1]]]
    pts: list = []

    def add(nid):
        p = [round(float(trip.nodes[nid, 0]), 5), round(float(trip.nodes[nid, 1]), 5)]
        if not pts or pts[-1] != p:
            pts.append(p)

    prev = None
    for arc in seq:
        if prev is not None and B[prev] != A[arc] and prep is not None and prep.graph is not None:
            for nid in _fill_path(prep, int(B[prev]), int(A[arc])):
                add(nid)
        add(A[arc])
        add(B[arc])
        prev = arc
    return pts


def ui_payload(pred: Prediction, trip: Trip, metrics: dict) -> dict:
    """JSON-ready response for the UI. Contains predictions, road geometry and aggregate scores only."""
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    node = trip.nodes
    seg = lambda arcs: _r5(np.c_[node[A[arcs], 0], node[A[arcs], 1], node[B[arcs], 0], node[B[arcs], 1]]) if len(arcs) else []
    # road network near the trace: every arc that was a candidate, de-duplicated across directions
    ua = np.unique(pred.cand_arc)
    key = np.minimum(A[ua], B[ua]).astype(np.int64) * len(node) + np.maximum(A[ua], B[ua])
    ua = ua[np.unique(key, return_index=True)[1]]
    matched = np.flatnonzero(pred.clf_arc >= 0)
    arcs_m = pred.clf_arc[matched]
    flags, pos = transition_flags(arcs_m, A, B)
    run_starts = np.r_[0, pos]
    breaks = matched[pos[~flags]] if len(pos) else np.empty(0, dtype=np.int64)
    return {
        "trip_id": trip.trip_id,
        "gps": _r5(np.c_[trip.lon, trip.lat]),
        "network": seg(ua),
        "classifier_only": {"arcs": seg(arcs_m[run_starts]) if len(arcs_m) else [],
                            "breaks": _r5(np.c_[trip.lon[breaks], trip.lat[breaks]]) if len(breaks) else []},
        "hmm": {"route": route_polyline(pred.hmm_arc, trip, pred.prep)},
        "metrics": metrics,
    }

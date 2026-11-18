"""Shared helpers for 10_folium_prediction_maps.py and 10b_hmm_diagnostics.py (real pipeline runs, student_test only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import ELIGIBLE_M                       # noqa: E402
from pathsnap.io import load_trip                              # noqa: E402
from pathsnap.labels import align_to_route                     # noqa: E402
from pathsnap.metrics import transition_flags                  # noqa: E402
from pathsnap.paths import DATA_ROOT                           # noqa: E402
from pathsnap.pipeline import load_bundle, predict_trip, score_prediction  # noqa: E402

BUNDLE = ROOT / "artifacts" / "final" / "bundle.joblib"


def run_trip(bundle, tid):
    trip = load_trip(DATA_ROOT, "student_test", tid)
    pred = predict_trip(bundle, trip)
    metrics = score_prediction(pred, trip)
    ref, rd = align_to_route(pred.G, pred.N, trip.arcs[:, 0], trip.arcs[:, 1], trip.route)
    elig = rd <= ELIGIBLE_M
    return trip, pred, metrics, ref, elig


def seg_dist(G, N, A, B, arcs):
    """Distance (m) from each observation to its arc `arcs` (projected metres); nan where arc < 0."""
    out = np.full(len(arcs), np.nan)
    ok = arcs >= 0
    a, b = N[A[arcs[ok]]], N[B[arcs[ok]]]
    ab = b - a
    t = np.clip(((G[ok] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    out[ok] = np.hypot(*(G[ok] - (a + t[:, None] * ab)).T)
    return out


def connected(u, v, A, B, cache):
    """True if arcs u,v are the same, or either is reachable from the other in <=3 directed hops (metrics.transition_flags)."""
    if u < 0 or v < 0:
        return False
    if u == v:
        return True
    k = (u, v)
    if k not in cache:
        f1, _ = transition_flags(np.array([u, v]), A, B)
        f2, _ = transition_flags(np.array([v, u]), A, B)
        cache[k] = bool(f1[0] or f2[0])
    return cache[k]


def groups(pred, ref, elig):
    c = pred.clf_arc == ref
    h = pred.hmm_arc == ref
    return {"both_right": int((elig & c & h).sum()), "clf_right_hmm_wrong": int((elig & c & ~h).sum()),
            "clf_wrong_hmm_right": int((elig & ~c & h).sum()), "both_wrong": int((elig & ~c & ~h).sum())}, c, h


def ranges(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return [(int(idx[s]), int(idx[e])) for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(idx) - 1])]

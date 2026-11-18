"""Evaluation metrics for the candidate classifier and for the reconstructed route.

Observation-level Top-1 accuracy: per GPS observation take the highest-scoring candidate and compare it with
the reference arc. It is reported two ways:
  top1       over every labelled (eligible) observation; a reference arc missing from the candidate set
             counts as a miss, so top1 <= 1 - candidate_miss_rate
  top1_cond  over observations whose reference arc is in the candidate set (pure classifier ranking quality)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score


def obs_keys(trip: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return trip.astype(np.int64) * 100_000 + obs.astype(np.int64)


def top1_hits(trip, obs, y, p) -> tuple[int, np.ndarray]:
    """Number of observations whose best-scoring candidate is the reference arc, and the winning row positions."""
    df = pd.DataFrame({"k": obs_keys(trip, obs), "p": p})
    win = df.groupby("k", sort=False)["p"].idxmax().to_numpy()
    return int(np.asarray(y)[win].sum()), win


def clf_metrics(y, p, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
            "logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _successors(A: np.ndarray, B: np.ndarray):
    order = np.argsort(A, kind="stable")
    sorted_from = A[order]
    n = int(max(A.max(), B.max())) + 1
    start = np.searchsorted(sorted_from, np.arange(n + 1))
    return order, start


def route_continuity(pred_arcs: np.ndarray, A: np.ndarray, B: np.ndarray, max_hops: int = 3) -> tuple[int, int]:
    """(connected transitions, transitions) between consecutive *distinct* predicted arcs.

    A transition u -> v is connected if v can be reached from u by following at most `max_hops` directed arcs
    (1 Hz GPS at road speeds can skip a short arc between two observations, so strict adjacency would be unfair).
    """
    flags, _ = transition_flags(pred_arcs, A, B, max_hops)
    return int(flags.sum()), len(flags)


def transition_flags(pred_arcs: np.ndarray, A: np.ndarray, B: np.ndarray, max_hops: int = 3):
    """Per transition between consecutive distinct arcs: (connected? flags, index in pred_arcs where the new arc starts)."""
    if len(pred_arcs) < 2:
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)
    order, start = _successors(A, B)
    run_starts = np.flatnonzero(np.r_[True, pred_arcs[1:] != pred_arcs[:-1]])
    seq = pred_arcs[run_starts]
    flags = np.zeros(len(seq) - 1, dtype=bool)
    for i, (u, v) in enumerate(zip(seq[:-1], seq[1:])):
        frontier = {int(u)}
        for _ in range(max_hops):
            nxt = set()
            for a in frontier:
                node = B[a]
                nxt.update(order[start[node]:start[node + 1]].tolist())
            if int(v) in nxt:
                flags[i] = True
                break
            frontier = nxt
            if not frontier:
                break
    return flags, run_starts[1:]

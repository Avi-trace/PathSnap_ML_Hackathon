"""Candidate table + per-candidate features (the same code is used for training and for API inference).

Row = (GPS observation, candidate directed arc within RADIUS metres). Features are the ones agreed for the
project: per-point geometry plus a short +/-2 observation heading window. The classifier sees no
neighbouring observations' candidates, so route-level coherence is left for the HMM to add.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import build_arc_index, generate
from .geo import Projector
from .labels import align_to_route

RADIUS = 75.0        # candidate search radius (m); measured miss rate 0.67 % on clean trips
ELIGIBLE_M = 150.0   # an observation gets a reference label only if its aligned arc is this close
WINDOW = 2           # heading/speed estimated from observations i-2 .. i+2
MOVING_M = 4.0       # minimum displacement over the window for the heading to be trusted
MAX_WINDOW_DT = 10.0 # seconds; a wider window straddles a GPS gap

FEATURES = ["dist", "dist_gap", "dist_rank", "n_cand", "along_m", "remain_m", "arc_len", "lateral",
            "cos_dh", "abs_dh", "cos_gap", "speed_w", "step_speed", "dt_prev", "dt_next",
            "outdeg_end", "indeg_start", "two_way"]


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def gps_kinematics(G: np.ndarray, t: np.ndarray) -> dict:
    n = len(G)
    i = np.arange(n)
    lo, hi = np.clip(i - WINDOW, 0, n - 1), np.clip(i + WINDOW, 0, n - 1)
    d = G[hi] - G[lo]
    dist = np.linalg.norm(d, axis=1)
    dtw = t[hi] - t[lo]
    ok = (dtw > 0) & (dtw <= MAX_WINDOW_DT)
    dt = np.diff(t)
    dt_prev = np.r_[dt[0], dt]
    dt_next = np.r_[dt, dt[-1]]
    step = np.r_[0.0, np.linalg.norm(np.diff(G, axis=0), axis=1)]
    step[0] = step[1] if n > 1 else 0.0
    step_speed = np.where(dt_prev > 0, step / np.maximum(dt_prev, 1e-3), 0.0)
    return {"heading": np.arctan2(d[:, 1], d[:, 0]),
            "moving": ok & (dist >= MOVING_M),
            "speed_w": np.where(ok, dist / np.maximum(dtw, 1e-3), 0.0).clip(0, 60),
            "step_speed": step_speed.clip(0, 60),
            "dt_prev": np.clip(dt_prev, 0, 60), "dt_next": np.clip(dt_next, 0, 60)}


def candidate_table(G: np.ndarray, t: np.ndarray, N: np.ndarray, A: np.ndarray, B: np.ndarray,
                    radius: float = RADIUS) -> pd.DataFrame:
    idx = build_arc_index(N, A, B)
    obs, arc, dist, frac = generate(G, idx, radius)
    k = gps_kinematics(G, t)
    n_nodes = len(N)
    outdeg, indeg = np.bincount(A, minlength=n_nodes), np.bincount(B, minlength=n_nodes)
    two_way = np.isin(B.astype(np.int64) * n_nodes + A, A.astype(np.int64) * n_nodes + B)

    vec = N[B[arc]] - N[A[arc]]
    alen = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
    dh = _wrap(k["heading"][obs] - np.arctan2(vec[:, 1], vec[:, 0]))
    mv = k["moving"][obs]
    P, a0 = G[obs], N[A[arc]]
    df = pd.DataFrame({
        "obs": obs, "arc": arc,
        "dist": dist, "along_m": frac * alen, "remain_m": (1 - frac) * alen, "arc_len": alen,
        "lateral": (vec[:, 0] * (P[:, 1] - a0[:, 1]) - vec[:, 1] * (P[:, 0] - a0[:, 0])) / alen,
        "cos_dh": np.where(mv, np.cos(dh), 0.0), "abs_dh": np.where(mv, np.degrees(np.abs(dh)), 90.0),
        "speed_w": k["speed_w"][obs], "step_speed": k["step_speed"][obs],
        "dt_prev": k["dt_prev"][obs], "dt_next": k["dt_next"][obs],
        "outdeg_end": outdeg[B[arc]], "indeg_start": indeg[A[arc]], "two_way": two_way[arc].astype(np.float32),
    })
    g = df.groupby("obs", sort=False)
    df["dist_rank"] = g["dist"].rank(method="first") - 1
    df["dist_gap"] = df["dist"] - g["dist"].transform("min")
    df["n_cand"] = g["dist"].transform("size")
    df["cos_gap"] = g["cos_dh"].transform("max") - df["cos_dh"]
    df[FEATURES] = df[FEATURES].astype(np.float32)
    return df


def build_trip_table(trip, radius: float = RADIUS, with_labels: bool = True) -> dict:
    """Return {'rows': candidate rows (eligible observations only when labelled), 'obs': per-observation info}."""
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    G = proj.xy(trip.lon, trip.lat)
    N = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    rows = candidate_table(G, trip.t, N, A, B, radius)
    obs = pd.DataFrame({"obs": np.arange(trip.n_gps), "t": trip.t})
    if with_labels and trip.route is not None:
        ref, rd = align_to_route(G, N, A, B, trip.route)
        rows["label"] = (rows["arc"].to_numpy() == ref[rows["obs"].to_numpy()]).astype(np.int8)
        ref_in = np.zeros(trip.n_gps, bool)
        ref_in[rows.loc[rows["label"] == 1, "obs"].to_numpy()] = True
        obs["ref_arc"], obs["ref_dist"], obs["eligible"], obs["ref_in_cands"] = ref, rd, rd <= ELIGIBLE_M, ref_in
        rows = rows[obs["eligible"].to_numpy()[rows["obs"].to_numpy()]].reset_index(drop=True)
    rows["trip"] = np.int32(int(trip.trip_id))
    return {"rows": rows, "obs": obs, "trip_id": trip.trip_id, "epsg": proj.epsg}

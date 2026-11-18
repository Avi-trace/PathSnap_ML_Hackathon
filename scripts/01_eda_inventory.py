"""Dataset inventory + data-quality checks + candidate-radius coverage.

Reads only the `train` and `student_test` partitions. Writes reports/eda_trip_summary.csv and prints a summary.
usage: python scripts/01_eda_inventory.py --data-root "<.../Student folder/Student folder>"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.candidates import build_arc_index, generate      # noqa: E402
from pathsnap.geo import Projector                              # noqa: E402
from pathsnap.io import PARTITIONS, list_trips, load_trip       # noqa: E402

RADII = (25, 50, 75, 100, 150)


def summarise(trip) -> dict:
    row = {"partition": trip.partition, "trip_id": trip.trip_id}
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    G, N = proj.xy(trip.lon, trip.lat), proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    n_nodes, n_arcs = len(N), len(A)

    # ---- GPS observations
    dt = np.diff(trip.t)
    step = np.linalg.norm(np.diff(G, axis=0), axis=1)
    speed = step[dt > 0] / dt[dt > 0]
    row.update(epsg=proj.epsg, n_gps=trip.n_gps, duration_s=trip.t[-1] - trip.t[0],
               nan_gps=int(np.isnan(np.c_[trip.lon, trip.lat, trip.t]).sum()),
               coord_implausible=int(((np.abs(trip.lon) > 180) | (np.abs(trip.lat) > 90)).sum()),
               t_nonincreasing=int((dt <= 0).sum()), dt_eq_1s=float((dt == 1).mean()), dt_max=float(dt.max()),
               n_gaps_gt5s=int((dt > 5).sum()), speed_p99_ms=float(np.percentile(speed, 99)),
               n_speed_gt_50ms=int((speed > 50).sum()),
               gps_extent_x_km=np.ptp(G[:, 0]) / 1e3, gps_extent_y_km=np.ptp(G[:, 1]) / 1e3)

    # ---- road network
    used = np.zeros(n_nodes, bool)
    used[A] = True
    used[B] = True
    keys = A.astype(np.int64) * n_nodes + B
    g = coo_matrix((np.ones(n_arcs, np.int8), (A, B)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(g, directed=False)
    comp = np.bincount(labels[used]) if used.any() else np.zeros(1, int)
    alen = np.linalg.norm(N[B] - N[A], axis=1)
    row.update(n_nodes=n_nodes, n_arcs=n_arcs, isolated_nodes=int((~used).sum()), self_loops=int((A == B).sum()),
               duplicate_arcs=int(n_arcs - len(np.unique(keys))),
               two_way_frac=float(np.isin(B.astype(np.int64) * n_nodes + A, keys).mean()),
               n_components=int((comp > 0).sum()), largest_component_frac=float(comp.max() / max(used.sum(), 1)),
               arc_len_median_m=float(np.median(alen)), arc_len_max_m=float(alen.max()),
               arc_ids_in_range=bool(A.min() >= 0 and B.max() < n_nodes))

    # ---- reference route
    route = trip.route
    ok = route is not None and len(route) > 0 and route.min() >= 0 and route.max() < n_arcs
    row.update(route_len=0 if route is None else len(route), route_ids_in_range=bool(ok))
    if ok:
        row.update(route_head_to_tail=float((B[route[:-1]] == A[route[1:]]).mean()) if len(route) > 1 else 1.0,
                   route_unique_arc_frac=float(len(np.unique(route)) / len(route)),
                   route_km=float(alen[route].sum() / 1e3))

    # ---- candidate coverage at the largest radius, derived for every smaller radius
    obs, arc, d, _ = generate(G, build_arc_index(N, A, B), max(RADII))
    dmin = np.full(trip.n_gps, np.inf)
    if ok:
        in_route = np.zeros(n_arcs, bool)
        in_route[route] = True
        m = in_route[arc]
        np.minimum.at(dmin, obs[m], d[m])
    fin = dmin[np.isfinite(dmin)]
    row.update(gps_to_route_p50=float(np.percentile(dmin[np.isfinite(dmin)], 50)) if len(fin) else np.nan,
               gps_to_route_p90=float(np.percentile(fin, 90)) if len(fin) else np.nan,
               gps_to_route_p99=float(np.percentile(fin, 99)) if len(fin) else np.nan,
               gps_to_route_gt150=int((~np.isfinite(dmin)).sum()))
    for R in RADII:
        row[f"miss_R{R}"] = float((dmin > R).mean())                 # nearest reference arc farther than R
        row[f"cand_rows_R{R}"] = int((d <= R).sum())
        row[f"cand_per_obs_R{R}"] = float((d <= R).sum() / trip.n_gps)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports" / "eda_trip_summary.csv"))
    a = ap.parse_args()
    rows, t0 = [], time.time()
    for part in PARTITIONS:
        for tid in list_trips(a.data_root, part):
            rows.append(summarise(load_trip(a.data_root, part, tid)))
            print(f"[{len(rows):3d}] {part}/{tid} done  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n===== INVENTORY =====")
    for part, g in df.groupby("partition"):
        print(f"{part}: trips={len(g)} gps={g.n_gps.sum()} gps/trip(min,med,max)=({g.n_gps.min()},{int(g.n_gps.median())},{g.n_gps.max()}) "
              f"nodes(med,max)=({int(g.n_nodes.median())},{g.n_nodes.max()}) arcs(med,max)=({int(g.n_arcs.median())},{g.n_arcs.max()}) "
              f"route_len(med,max)=({int(g.route_len.median())},{g.route_len.max()})")
    print("epsg zones:", df.epsg.value_counts().to_dict())
    print("\n===== DATA QUALITY (all 90 trips) =====")
    print("NaN GPS values:", int(df.nan_gps.sum()), "| implausible coords:", int(df.coord_implausible.sum()),
          "| non-increasing timestamps:", int(df.t_nonincreasing.sum()))
    print(f"1 s sampling: mean share of steps = {df.dt_eq_1s.mean():.3f}; trips with a gap >5 s: {(df.n_gaps_gt5s > 0).sum()}; max gap = {df.dt_max.max():.0f}s")
    print("speed >50 m/s steps:", int(df.n_speed_gt_50ms.sum()), "| trips:", int((df.n_speed_gt_50ms > 0).sum()))
    print("network: isolated nodes", int(df.isolated_nodes.sum()), "| self-loops", int(df.self_loops.sum()), "| duplicate arcs", int(df.duplicate_arcs.sum()),
          f"| two-way share mean {df.two_way_frac.mean():.3f}", "| trips with >1 component:", int((df.n_components > 1).sum()),
          f"| min largest-component share {df.largest_component_frac.min():.3f}")
    print("route: ids in range for", int(df.route_ids_in_range.sum()), "of", len(df),
          f"| head-to-tail mean {df.route_head_to_tail.mean():.4f}, min {df.route_head_to_tail.min():.4f}",
          f"| trips with revisited arcs: {(df.route_unique_arc_frac < 1).sum()}")
    print("\n===== CANDIDATE COVERAGE (observation-weighted) =====")
    w = df.n_gps
    for R in RADII:
        print(f"R={R:>3} m: reference-arc miss rate {np.average(df[f'miss_R{R}'], weights=w):.4f} | candidates/obs {df[f'cand_rows_R{R}'].sum() / w.sum():5.1f} "
              f"| train rows {df.loc[df.partition == 'train', f'cand_rows_R{R}'].sum():,}")
    print("\ntrips with worst GPS-to-route p90 (m):")
    print(df.sort_values("gps_to_route_p90", ascending=False)[["partition", "trip_id", "n_gps", "gps_to_route_p50", "gps_to_route_p90", "gps_to_route_p99", "gps_to_route_gt150"]].head(8).to_string(index=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

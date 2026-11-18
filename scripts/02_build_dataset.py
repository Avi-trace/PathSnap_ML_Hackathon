"""Build the labelled candidate/feature tables for train + student_test trips.

usage: python scripts/02_build_dataset.py --data-root "<.../Student folder/Student folder>"
Writes artifacts/data/<partition>/<trip>.joblib and prints a baseline summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import RADIUS, build_trip_table       # noqa: E402
from pathsnap.io import PARTITIONS, list_trips, load_trip    # noqa: E402

# Reference route sits ~120 m / ~67 m from the GPS trace on average with a wide spread (see reports/eda_trip_summary.csv),
# i.e. the labels are unusable. They are built but excluded from fitting; see docs.
EXCLUDE_FROM_FIT = {"00000033", "00000054"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--radius", type=float, default=RADIUS)
    a = ap.parse_args()
    t0, stats = time.time(), []
    for part in PARTITIONS:
        out = ROOT / "artifacts" / "data" / part
        out.mkdir(parents=True, exist_ok=True)
        for tid in list_trips(a.data_root, part):
            tab = build_trip_table(load_trip(a.data_root, part, tid), radius=a.radius)
            joblib.dump(tab, out / f"{tid}.joblib", compress=3)
            rows, obs = tab["rows"], tab["obs"]
            el = obs[obs.eligible]
            near_ok = rows[(rows.dist_rank == 0)].label.sum()
            stats.append(dict(part=part, trip=tid, n_obs=len(obs), n_elig=len(el), rows=len(rows), miss=int((~el.ref_in_cands).sum()),
                              nearest_ok=int(near_ok), fit=tid not in EXCLUDE_FROM_FIT or part != "train"))
            print(f"{part}/{tid}: obs={len(obs)} eligible={len(el)} rows={len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    print("\n===== DATASET SUMMARY (radius %.0f m) =====" % a.radius)
    for part in PARTITIONS:
        s = [x for x in stats if x["part"] == part and x["fit"]]
        e, r, m, nn = (sum(x[k] for x in s) for k in ("n_elig", "rows", "miss", "nearest_ok"))
        print(f"{part} (fit trips only): trips={len(s)} eligible_obs={e:,} candidate_rows={r:,} ({r / e:.1f}/obs) "
              f"candidate_miss_rate={m / e:.4f} nearest-arc Top-1={nn / e:.4f}")
    print("excluded from fitting:", sorted(EXCLUDE_FROM_FIT))


if __name__ == "__main__":
    main()

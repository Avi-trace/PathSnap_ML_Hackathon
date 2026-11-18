"""Run the full inference pipeline on raw student_test files and check it against the training-time numbers.

usage: python scripts/05_pipeline_check.py [--model artifacts/models/RandomForest.joblib]
The classifier-only Top-1 produced here (raw files -> candidates -> model) must match the Top-1 that
03_train_models.py measured from the cached candidate tables for the same model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.io import list_trips, load_trip                                    # noqa: E402
from pathsnap.pipeline import Bundle, predict_trip, score_prediction, ui_payload  # noqa: E402

from pathsnap.paths import DATA_ROOT                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "models" / "RandomForest.joblib"))
    ap.add_argument("--name", default="RandomForest")
    a = ap.parse_args()
    bundle = Bundle(model=joblib.load(a.model))
    rows = []
    for tid in list_trips(DATA_ROOT, "student_test"):
        trip = load_trip(DATA_ROOT, "student_test", tid)
        t0 = time.time()
        pred = predict_trip(bundle, trip)
        m = score_prediction(pred, trip)
        payload = ui_payload(pred, trip, m)
        rows.append({"trip": tid, "n_obs": m["n_obs"], **{k: v for k, v in m.items() if k not in ("n_obs", "latency_ms")},
                     "ms_total": sum(m["latency_ms"].values()), **{f"ms_{k}": v for k, v in m["latency_ms"].items()},
                     "route_pts": len(payload["hmm"]["route"]), "breaks": len(payload["classifier_only"]["breaks"]), "wall_s": time.time() - t0})
        r = rows[-1]
        print(f"{tid}: obs={r['n_obs']:5d} acc clf {r['pointwise_acc_classifier']:.4f} -> hmm {r['pointwise_acc_hmm']:.4f} | continuity {r['route_continuity_classifier']:.3f} -> {r['route_continuity_hmm']:.3f} "
              f"| routeF1 {r['route_f1_classifier']:.3f} -> {r['route_f1_hmm']:.3f} | {r['ms_total']:5d} ms", flush=True)
    df = pd.DataFrame(rows)
    w = df.n_obs
    wavg = lambda c: float(np.average(df[c], weights=w))
    print("\n===== AGGREGATE over 10 student_test trips (observation-weighted) =====")
    for c in ("pointwise_acc_classifier", "pointwise_acc_hmm", "route_continuity_classifier", "route_continuity_hmm", "route_f1_classifier", "route_f1_hmm", "candidate_miss_rate"):
        print(f"{c:30s} {wavg(c):.4f}")
    print(f"latency per trip (ms): mean {df.ms_total.mean():.0f}, max {df.ms_total.max()}")
    ref = pd.read_csv(ROOT / "reports" / "model_comparison.csv").set_index("model").loc[a.name]
    print(f"\nCHECK  classifier-only Top-1 from raw files {wavg('pointwise_acc_classifier'):.4f}  vs  training-time student Top-1 {ref.st_top1:.4f}  "
          f"(diff {abs(wavg('pointwise_acc_classifier') - ref.st_top1):.5f})")
    print(f"CHECK  continuity (clf) {wavg('route_continuity_classifier'):.4f}  vs training-time {ref.st_route_continuity:.4f}  (trip-pooled vs observation-weighted, may differ slightly)")
    df.to_csv(ROOT / "reports" / "pipeline_check_student.csv", index=False)


if __name__ == "__main__":
    main()

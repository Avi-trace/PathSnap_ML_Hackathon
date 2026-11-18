"""Tune the HMM decoder parameters (Bayesian, Optuna TPE) and compare classifier-only vs classifier+HMM.

Parameters tuned   alpha        emission power on the normalised classifier probability   (0.3 - 3)
                   beta         Newson-Krumm transition scale in metres                    (2 - 60, log)
                   invalid_pen  log-penalty for an unreachable candidate pair              (3 - 40)
Objective          observation-level Top-1 of the Viterbi path over a fixed random sample of 30 train trips,
                   using OUT-OF-FOLD classifier probabilities (so the emissions are as noisy as on unseen trips).
Evaluation         the 10 student_test trips through the complete pipeline (raw files -> API code path),
                   with the final classifier trained on all fit trips. The evaluator partition is never read.
usage: python scripts/07_tune_decoder.py [--trials 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.experiment import DATA_ROOT, EXCLUDE_FROM_FIT, SEED, load_part     # noqa: E402
from pathsnap.geo import Projector                                                # noqa: E402
from pathsnap.hmm import decode, prepare_trip                                     # noqa: E402
from pathsnap.io import list_trips, load_trip                                     # noqa: E402
from pathsnap.pipeline import DEFAULT_HMM, Bundle, predict_trip, score_prediction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "models" / "RandomForest_tuned.joblib"))
    ap.add_argument("--oof", default=str(ROOT / "artifacts" / "oof_p_rf_tuned.joblib"))
    a = ap.parse_args()

    train, tr_obs = load_part("train", EXCLUDE_FROM_FIT)
    oof = joblib.load(a.oof)
    assert len(oof) == len(train)
    rng = np.random.default_rng(SEED)
    trips = [t for t in np.unique(train.trip.to_numpy()) if tr_obs[f"{t:08d}"].shape[0] <= 4000]
    sample = sorted(rng.choice(trips, size=min(30, len(trips)), replace=False).tolist())
    print(f"decoder tuning sample: {len(sample)} train trips", flush=True)

    t0, prepared = time.time(), []
    for t in sample:
        tid = f"{t:08d}"
        idx = np.flatnonzero(train.trip.to_numpy() == t)
        rows, p = train.iloc[idx], oof[idx]
        trip = load_trip(DATA_ROOT, "train", tid, with_route=False)
        pr = Projector(trip.lon.mean(), trip.lat.mean())
        G, N = pr.xy(trip.lon, trip.lat), pr.xy(trip.nodes[:, 0], trip.nodes[:, 1])
        prep = prepare_trip(G, trip.t, rows.obs.to_numpy(), rows.arc.to_numpy(), (rows.along_m / rows.arc_len).to_numpy(np.float64),
                            rows.arc_len.to_numpy(np.float64), N, trip.arcs[:, 0], trip.arcs[:, 1])
        prepared.append((prep, p, rows.label.to_numpy(), int(tr_obs[tid].eligible.sum())))
    print(f"network distances precomputed in {time.time() - t0:.0f}s", flush=True)

    def top1(alpha, beta, pen):
        hits = n = 0
        for prep, p, y, n_elig in prepared:
            hits += int(y[decode(prep, p, alpha, beta, pen)].sum())
            n += n_elig
        return hits / n

    def clf_only_hits(prep, p, y):       # highest-scoring candidate per observation, no sequence model
        win = np.array([s + int(np.argmax(p[s:e])) for s, e in zip(prep.starts[:-1], prep.starts[1:])])
        return int(y[win].sum())

    clf_top1 = sum(clf_only_hits(prep, p, y) for prep, p, y, _ in prepared) / sum(x[3] for x in prepared)
    d0 = top1(**{k: DEFAULT_HMM[k] for k in ("alpha", "beta")}, pen=DEFAULT_HMM["invalid_pen"])
    print(f"sample OOF Top-1: classifier-only {clf_top1:.4f} | HMM default params {d0:.4f}", flush=True)

    def objective(trial):
        return top1(trial.suggest_float("alpha", 0.3, 3.0), trial.suggest_float("beta", 2.0, 60.0, log=True), trial.suggest_float("invalid_pen", 3.0, 40.0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=12))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    study.trials_dataframe().to_csv(ROOT / "reports" / "hmm_trials.csv", index=False)
    best = {**DEFAULT_HMM, **{k: float(v) for k, v in study.best_params.items() if k in DEFAULT_HMM}, "invalid_pen": float(study.best_params["invalid_pen"])}
    json.dump({**best, "sample_trips": sample, "best_sample_top1": study.best_value, "default_sample_top1": d0, "classifier_only_sample_top1": clf_top1,
               "n_trials": len(study.trials), "tuning_seconds": time.time() - t0}, open(ROOT / "reports" / "hmm_best.json", "w"), indent=2)
    print(f"best sample Top-1 {study.best_value:.4f} (default {d0:.4f}, classifier-only {clf_top1:.4f}) in {time.time() - t0:.0f}s\nbest: {best}", flush=True)

    # ---- student_test through the whole pipeline: classifier-only vs classifier+HMM, tuned vs default decoder
    model = joblib.load(a.model)
    rows = []
    for tag, hp in (("default decoder", dict(DEFAULT_HMM)), ("tuned decoder", best)):
        bundle = Bundle(model=model, hmm=hp)
        for tid in list_trips(DATA_ROOT, "student_test"):
            trip = load_trip(DATA_ROOT, "student_test", tid)
            m = score_prediction(predict_trip(bundle, trip), trip)
            rows.append({"decoder": tag, "trip": tid, **{k: v for k, v in m.items() if k != "latency_ms"}})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "classifier_vs_hmm_per_trip.csv", index=False)
    agg = []
    for tag, g in df.groupby("decoder", sort=False):
        w = g.n_obs
        agg.append({"decoder": tag, **{c: float(np.average(g[c], weights=w)) for c in g.columns if c.startswith(("pointwise", "route_", "candidate"))}})
    agg = pd.DataFrame(agg)
    agg.to_csv(ROOT / "reports" / "classifier_vs_hmm.csv", index=False)
    print("\n===== student_test, observation-weighted =====\n" + agg.round(4).T.to_string(), flush=True)
    t = df[df.decoder == "tuned decoder"]
    worse = t[t.pointwise_acc_hmm < t.pointwise_acc_classifier]
    print("\ntrips where the tuned HMM did NOT improve pointwise accuracy:", worse[["trip", "pointwise_acc_classifier", "pointwise_acc_hmm"]].round(4).to_dict("records"))


if __name__ == "__main__":
    main()

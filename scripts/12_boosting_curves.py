"""Per-boosting-round train/validation logloss curves for LightGBM and XGBoost.

This is a SEPARATE trip-level 80/20 split (seed 0), not the 3-fold group-CV folds used in
scripts/03_train_models.py / reports/model_comparison.csv -- it exists only to show each
boosting round's train vs validation logloss (sklearn CV does not expose per-round histories).

Hyper-parameters are EXACTLY those in scripts/03_train_models.py: LightGBM n_estimators=300,
learning_rate=0.1, num_leaves=63, min_child_samples=50; XGBoost n_estimators=300, learning_rate=0.1,
max_depth=8, tree_method="hist". To keep this fast, the fit rows are subsampled to a fixed random 30%
of observations (same obs-level idea as script 03: all candidates of a sampled observation are kept);
the validation set (20% of trips) is used in full and is never subsampled.

Writes: reports/boosting_curves.json, reports/figures/model_boosting_curves.png
usage: python scripts/12_boosting_curves.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.experiment import EXCLUDE_FROM_FIT, SEED, SUB, load_part  # noqa: E402
from pathsnap.features import FEATURES                                  # noqa: E402
from pathsnap.metrics import obs_keys                                   # noqa: E402

REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
N_JOBS = 4
SPLIT_SEED = 0
VAL_FRAC = 0.20
N_ROUNDS = 300


def main():
    t0 = time.time()
    train, _ = load_part("train", EXCLUDE_FROM_FIT)
    trips = np.sort(train["trip"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(trips))
    n_val_trips = int(round(VAL_FRAC * len(trips)))
    val_trips = set(trips[perm[:n_val_trips]].tolist())
    fit_trips = set(trips[perm[n_val_trips:]].tolist())
    print(f"trips: {len(trips)} total -> {len(fit_trips)} fit / {len(val_trips)} validation "
          f"(trip-level 80/20 split, seed {SPLIT_SEED})", flush=True)

    trip_arr = train["trip"].to_numpy()
    fit_mask = np.isin(trip_arr, list(fit_trips))
    val_mask = np.isin(trip_arr, list(val_trips))

    # fixed observation-level fit subsample, same idea (seed, fraction) as scripts/03_train_models.py
    uk, inv = np.unique(obs_keys(trip_arr, train["obs"].to_numpy()), return_inverse=True)
    sampled_obs = (np.random.default_rng(SEED).random(len(uk)) < SUB)[inv]
    fit_idx = np.flatnonzero(fit_mask & sampled_obs)
    val_idx = np.flatnonzero(val_mask)
    print(f"fit rows {len(fit_idx):,} (30% obs subsample of fit trips) | validation rows {len(val_idx):,} (full)", flush=True)

    X = train[FEATURES].to_numpy(np.float32)
    y = train["label"].to_numpy()
    Xfit, yfit = X[fit_idx], y[fit_idx]
    Xval, yval = X[val_idx], y[val_idx]

    curves = {}

    # ---------------------------------------------------------------- LightGBM
    t_m = time.time()
    evals_result: dict = {}
    m = lgb.LGBMClassifier(n_estimators=N_ROUNDS, learning_rate=0.1, num_leaves=63, min_child_samples=50,
                           n_jobs=N_JOBS, verbose=-1, random_state=SEED)
    m.fit(Xfit, yfit, eval_set=[(Xfit, yfit), (Xval, yval)], eval_names=["train", "validation"],
          eval_metric="binary_logloss", callbacks=[lgb.record_evaluation(evals_result)])
    curves["LightGBM"] = {"train_logloss": [float(v) for v in evals_result["train"]["binary_logloss"]],
                          "val_logloss": [float(v) for v in evals_result["validation"]["binary_logloss"]]}
    print(f"LightGBM: {N_ROUNDS} rounds fit in {time.time() - t_m:.0f}s, "
          f"final train ll {curves['LightGBM']['train_logloss'][-1]:.4f}, "
          f"val ll {curves['LightGBM']['val_logloss'][-1]:.4f}", flush=True)

    # ---------------------------------------------------------------- XGBoost
    t_m = time.time()
    m = xgb.XGBClassifier(n_estimators=N_ROUNDS, learning_rate=0.1, max_depth=8, tree_method="hist",
                          n_jobs=N_JOBS, random_state=SEED, eval_metric="logloss")
    m.fit(Xfit, yfit, eval_set=[(Xfit, yfit), (Xval, yval)], verbose=False)
    ev = m.evals_result()
    curves["XGBoost"] = {"train_logloss": [float(v) for v in ev["validation_0"]["logloss"]],
                        "val_logloss": [float(v) for v in ev["validation_1"]["logloss"]]}
    print(f"XGBoost: {N_ROUNDS} rounds fit in {time.time() - t_m:.0f}s, "
          f"final train ll {curves['XGBoost']['train_logloss'][-1]:.4f}, "
          f"val ll {curves['XGBoost']['val_logloss'][-1]:.4f}", flush=True)

    meta = {"split_seed": SPLIT_SEED, "val_frac": VAL_FRAC, "n_fit_trips": len(fit_trips), "n_val_trips": len(val_trips),
            "fit_trips": sorted(int(t) for t in fit_trips), "val_trips": sorted(int(t) for t in val_trips),
            "fit_obs_subsample": SUB, "fit_rows": len(fit_idx), "val_rows": len(val_idx), "n_rounds": N_ROUNDS,
            "note": "Separate trip-level 80/20 split (seed 0), NOT the 3-fold group-CV folds used elsewhere in this study.",
            "curves": curves}
    json.dump(meta, open(REPORTS / "boosting_curves.json", "w"), indent=2)
    print(f"wrote {REPORTS / 'boosting_curves.json'}", flush=True)

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"LightGBM": "#54a24b", "XGBoost": "#e45756"}
    for name, c in curves.items():
        rounds = np.arange(1, len(c["train_logloss"]) + 1)
        ax.plot(rounds, c["train_logloss"], "-", color=colors[name], label=f"{name} train")
        ax.plot(rounds, c["val_logloss"], "--", color=colors[name], label=f"{name} validation")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss")
    ax.set_title("LightGBM & XGBoost: per-round train (solid) vs validation (dashed) logloss", fontsize=12)
    fig.text(0.5, 0.945, f"Separate trip-level 80/20 split (seed {SPLIT_SEED}), NOT the 3-fold CV folds; "
             f"fit rows subsampled to {SUB:.0%} of observations, as in script 03.",
             ha="center", va="top", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGDIR / "model_boosting_curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'model_boosting_curves.png'}")
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

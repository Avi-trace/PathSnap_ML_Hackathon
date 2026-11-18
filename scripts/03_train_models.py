"""Train and compare 9 models from 4 algorithm families on the candidate-ranking task.

Protocol (fixed before looking at any result):
  * Fit data: the 78 usable train trips (2 trips with unusable reference routes are excluded, see 02_build_dataset.py).
  * 3-fold GroupKFold by trip. Within a CV fold, models are fitted on a fixed random 30 % of the *observations*
    of the training folds (all candidates of a sampled observation are kept) to save time, and scored on the
    FULL held-out fold. Final refit: all fit trips, all rows. Final scoring: the 10 student_test trips.
  * Selection metric: observation-level Top-1 accuracy (mean over CV folds); log loss breaks ties.
  * The evaluator partition is never read.
usage: python scripts/03_train_models.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES                                          # noqa: E402
from pathsnap.io import load_trip                                                # noqa: E402
from pathsnap.metrics import clf_metrics, obs_keys, route_continuity, top1_hits  # noqa: E402
from pathsnap.mlp import TorchMLP                                                # noqa: E402

from pathsnap.paths import DATA_ROOT                                             # noqa: E402
EXCLUDE_FROM_FIT = {"00000033", "00000054"}
SEED, SUB, N_JOBS, FOLDS = 0, 0.30, 8, 3
torch.set_num_threads(N_JOBS)

MODELS = {   # name -> (family, config string, factory)
    "GaussianNB": ("Probabilistic", "var_smoothing=1e-9", lambda: GaussianNB()),
    "LogisticRegression": ("Linear", "StandardScaler + C=1.0, lbfgs, max_iter=300",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))),
    "DecisionTree": ("Tree", "max_depth=12, min_samples_leaf=50",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)),
    "RandomForest": ("Bagging ensemble", "50 trees, max_depth=16, min_samples_leaf=50",
                     lambda: RandomForestClassifier(n_estimators=50, max_depth=16, min_samples_leaf=50, n_jobs=N_JOBS, random_state=SEED)),
    "ExtraTrees": ("Bagging ensemble", "50 trees, max_depth=16, min_samples_leaf=50",
                   lambda: ExtraTreesClassifier(n_estimators=50, max_depth=16, min_samples_leaf=50, n_jobs=N_JOBS, random_state=SEED)),
    "HistGradientBoosting": ("Boosting", "300 iters, lr=0.1, max_leaf_nodes=63",
                             lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=SEED)),
    "LightGBM": ("Boosting", "300 rounds, lr=0.1, num_leaves=63, min_child_samples=50",
                 lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=50, n_jobs=N_JOBS, verbose=-1, random_state=SEED)),
    "XGBoost": ("Boosting", "300 rounds, lr=0.1, max_depth=8, hist",
                lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, tree_method="hist", n_jobs=N_JOBS, random_state=SEED)),
    "MLP(PyTorch)": ("Neural network", "128-64 ReLU, Adam lr=1e-3, wd=1e-5, batch 4096, 15 epochs, BCE",
                     lambda: TorchMLP(hidden=(128, 64), epochs=15, lr=1e-3, weight_decay=1e-5, batch=4096, seed=SEED)),
}


def load_part(part: str, exclude=()):
    rows, obs = [], {}
    for f in sorted((ROOT / "artifacts" / "data" / part).glob("*.joblib")):
        tab = joblib.load(f)
        if tab["trip_id"] in exclude:
            continue
        rows.append(tab["rows"])
        obs[tab["trip_id"]] = tab["obs"]
    return pd.concat(rows, ignore_index=True), obs


def score(part_rows: pd.DataFrame, p: np.ndarray, obs_tabs: dict):
    y = part_rows["label"].to_numpy()
    trips = np.unique(part_rows["trip"].to_numpy())
    tabs = [obs_tabs[f"{t:08d}"] for t in trips]
    n_elig = sum(int(o.eligible.sum()) for o in tabs)
    n_ref = sum(int((o.eligible & o.ref_in_cands).sum()) for o in tabs)
    hits, win = top1_hits(part_rows["trip"].to_numpy(), part_rows["obs"].to_numpy(), y, p)
    m = clf_metrics(y, p)
    m.update(top1=hits / n_elig, top1_cond=hits / n_ref, n_obs=n_elig)
    return m, win


def main():
    t_start = time.time()
    train, tr_obs = load_part("train", EXCLUDE_FROM_FIT)
    test, te_obs = load_part("student_test")
    print(f"train rows {len(train):,} ({train.trip.nunique()} trips) | student_test rows {len(test):,} ({test.trip.nunique()} trips)", flush=True)
    Xtr, ytr = train[FEATURES].to_numpy(np.float32), train["label"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(np.float32), test["label"].to_numpy()

    # fixed observation-level subsample used for CV fitting
    uk, inv = np.unique(obs_keys(train["trip"].to_numpy(), train["obs"].to_numpy()), return_inverse=True)
    sampled = (np.random.default_rng(SEED).random(len(uk)) < SUB)[inv]
    folds = list(GroupKFold(n_splits=FOLDS).split(Xtr, groups=train["trip"].to_numpy()))
    print(f"CV: {FOLDS} group folds; fit subsample {SUB:.0%} of observations ({sampled.sum():,} rows)", flush=True)

    # student trips' graphs, for route continuity
    graphs = {t: load_trip(DATA_ROOT, "student_test", f"{t:08d}", with_route=False).arcs for t in test["trip"].unique()}
    (ROOT / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    fold_rows, summary, mlp_hist = [], [], {}
    for name, (family, cfg, make) in MODELS.items():
        t_m = time.time()
        cv = []
        for k, (tr_i, te_i) in enumerate(folds):
            fit_i = tr_i[sampled[tr_i]]
            m = make()
            t0 = time.time()
            m.fit(Xtr[fit_i], ytr[fit_i])
            fit_s = time.time() - t0
            p = m.predict_proba(Xtr[te_i])[:, 1]
            met, _ = score(train.iloc[te_i], p, tr_obs)
            met.update(model=name, fold=k, fit_s=fit_s, n_fit_rows=len(fit_i))
            cv.append(met)
            fold_rows.append(met)
            print(f"  {name} fold {k}: top1={met['top1']:.4f} cond={met['top1_cond']:.4f} f1={met['macro_f1']:.4f} ll={met['logloss']:.4f} fit={fit_s:.0f}s", flush=True)
        # final refit on every fit trip (all rows), score on student_test
        m = make()
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_full = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        infer_100k = (time.time() - t0) / len(Xte) * 1e5
        met, win = score(test, p, te_obs)
        pred_trip, pred_arc = test["trip"].to_numpy()[win], test["arc"].to_numpy()[win]
        ok = tot = 0
        for t, arcs in graphs.items():
            a_t = pred_arc[pred_trip == t]
            o, n = route_continuity(a_t, arcs[:, 0], arcs[:, 1])
            ok, tot = ok + o, tot + n
        cvdf = pd.DataFrame(cv)
        summary.append({"model": name, "family": family, "config": cfg,
                        "cv_top1_mean": cvdf.top1.mean(), "cv_top1_std": cvdf.top1.std(), "cv_top1_cond_mean": cvdf.top1_cond.mean(),
                        "cv_macro_f1_mean": cvdf.macro_f1.mean(), "cv_logloss_mean": cvdf.logloss.mean(), "cv_fit_s_mean": cvdf.fit_s.mean(),
                        "st_top1": met["top1"], "st_top1_cond": met["top1_cond"], "st_macro_f1": met["macro_f1"], "st_precision": met["precision"],
                        "st_recall": met["recall"], "st_logloss": met["logloss"], "st_route_continuity": ok / max(tot, 1),
                        "fit_s_full": fit_full, "infer_s_per_100k_rows": infer_100k, "st_tn": met["tn"], "st_fp": met["fp"], "st_fn": met["fn"], "st_tp": met["tp"]})
        if isinstance(m, TorchMLP):
            mlp_hist = m.history_
        joblib.dump(m, ROOT / "artifacts" / "models" / f"{name.replace('(', '_').replace(')', '')}.joblib", compress=0)
        s = summary[-1]
        print(f"[{name}] CV top1 {s['cv_top1_mean']:.4f}+-{s['cv_top1_std']:.4f} | student top1 {s['st_top1']:.4f} (cond {s['st_top1_cond']:.4f}) "
              f"F1 {s['st_macro_f1']:.4f} ll {s['st_logloss']:.4f} cont {s['st_route_continuity']:.4f} | full fit {fit_full:.0f}s, infer {infer_100k:.2f}s/100k | {time.time() - t_m:.0f}s total", flush=True)
        del m

    pd.DataFrame(summary).to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "reports" / "cv_fold_results.csv", index=False)
    json.dump({"seed": SEED, "cv_subsample": SUB, "folds": FOLDS, "features": FEATURES, "excluded_from_fit": sorted(EXCLUDE_FROM_FIT),
               "models": {k: {"family": v[0], "config": v[1]} for k, v in MODELS.items()}, "mlp_history": mlp_hist},
              open(ROOT / "reports" / "model_configs.json", "w"), indent=2)
    df = pd.DataFrame(summary).sort_values(["cv_top1_mean", "cv_logloss_mean"], ascending=[False, True])
    print("\n===== MODEL COMPARISON (sorted by CV Top-1) =====")
    print(df[["model", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "st_top1", "st_macro_f1", "st_logloss", "st_route_continuity", "fit_s_full", "infer_s_per_100k_rows"]].round(4).to_string(index=False))
    print(f"\nbest by CV Top-1: {df.iloc[0].model} | total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model, Random Forest, then tuned-vs-untuned.

Why Random Forest: it ranked first on the pre-agreed selection metric (mean 3-fold group-CV observation-level
Top-1) in scripts/03_train_models.py and also had the best student_test log loss; its probabilities feed the HMM.
Objective      maximise mean observation-level Top-1 over the same 3 group folds and fit subsample as script 03,
               so the untuned CV score is directly comparable.
Search space   n_estimators 30-80        ensemble size (variance reduction vs cost)
               max_depth 10-20           tree complexity
               min_samples_leaf 10-200 (log)  leaf size: regularises and smooths the probability estimates
               max_features 0.2-1.0      share of the 18 features tried per split (untuned 'sqrt' ~ 0.24)
               max_samples 0.3-0.7       bootstrap share per tree (decorrelates trees, cuts cost)
Data           78 usable train trips. student_test is used afterwards, once, to compare tuned vs untuned.
usage: python scripts/04_tune_random_forest.py [--trials 16 --timeout 900]
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
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.experiment import EXCLUDE_FROM_FIT, SEED, cv_setup, load_part, score   # noqa: E402
from pathsnap.features import FEATURES                                              # noqa: E402

N_JOBS = 8
UNTUNED = dict(n_estimators=50, max_depth=16, min_samples_leaf=50, max_features="sqrt", max_samples=None)


def make(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params, bootstrap=True, n_jobs=N_JOBS, random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds for the whole search")
    a = ap.parse_args()

    train, tr_obs = load_part("train", EXCLUDE_FROM_FIT)
    test, te_obs = load_part("student_test")
    X, y = train[FEATURES].to_numpy(np.float32), train["label"].to_numpy()
    folds, sampled = cv_setup(train)

    def objective(trial: optuna.Trial) -> float:
        params = dict(n_estimators=trial.suggest_int("n_estimators", 30, 80),
                      max_depth=trial.suggest_int("max_depth", 10, 20),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                      max_features=trial.suggest_float("max_features", 0.2, 1.0),
                      max_samples=trial.suggest_float("max_samples", 0.3, 0.7))
        t0, tops, lls = time.time(), [], []
        for tr_i, te_i in folds:
            fit_i = tr_i[sampled[tr_i]]
            m = make(params).fit(X[fit_i], y[fit_i])
            met, _ = score(train.iloc[te_i], m.predict_proba(X[te_i])[:, 1], tr_obs)
            tops.append(met["top1"])
            lls.append(met["logloss"])
        trial.set_user_attr("cv_logloss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(tops))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
                                storage=f"sqlite:///{(ROOT / 'reports' / 'optuna_rf.db').as_posix()}", study_name="rf_top1", load_if_exists=True)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d}: CV top1={t.value:.4f} ll={t.user_attrs['cv_logloss']:.4f} ({t.user_attrs['seconds']:.0f}s) best={s.best_value:.4f}", flush=True)])
    tune_s = time.time() - t0
    (ROOT / "reports").mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(ROOT / "reports" / "optuna_trials.csv", index=False)
    best = study.best_params
    json.dump({"model": "RandomForest", "best_params": best, "best_cv_top1": study.best_value, "n_trials": len(study.trials), "tuning_seconds": tune_s,
               "sampler": "TPE (seed 0, 5 random start-up trials)", "objective": "mean 3-fold group-CV observation-level Top-1", "untuned_params": UNTUNED},
              open(ROOT / "reports" / "optuna_best.json", "w"), indent=2)
    print(f"\nbest CV Top-1 {study.best_value:.4f} after {len(study.trials)} trials in {tune_s:.0f}s\nbest params: {best}", flush=True)

    # ---- refit on all fit trips; compare tuned with untuned on student_test
    Xte = test[FEATURES].to_numpy(np.float32)
    rows = []
    for tag, params in (("untuned", UNTUNED), ("tuned", best)):
        t0 = time.time()
        m = make(params).fit(X, y)
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        inf = (time.time() - t0) / len(Xte) * 1e5
        met, _ = score(test, p, te_obs)
        rows.append({"variant": tag, **{f"st_{k}": met[k] for k in ("top1", "top1_cond", "macro_f1", "precision", "recall", "logloss")},
                     "fit_s_full": fit_s, "infer_s_per_100k_rows": inf})
        joblib.dump(m, ROOT / "artifacts" / "models" / f"RandomForest_{tag}.joblib", compress=0)
        joblib.dump(p, ROOT / "artifacts" / f"student_p_rf_{tag}.joblib")
        print(f"{tag}: student top1 {met['top1']:.4f} F1 {met['macro_f1']:.4f} ll {met['logloss']:.4f} fit {fit_s:.0f}s", flush=True)
    cmp = pd.DataFrame(rows)
    cmp["cv_top1"] = [pd.read_csv(ROOT / "reports" / "model_comparison.csv").set_index("model").loc["RandomForest", "cv_top1_mean"], study.best_value]
    cmp.to_csv(ROOT / "reports" / "tuned_vs_untuned.csv", index=False)
    print("\n===== TUNED vs UNTUNED (student_test) =====\n" + cmp.round(4).to_string(index=False), flush=True)

    # ---- out-of-fold probabilities of the tuned configuration (fit on all rows of the training folds) for decoder tuning
    oof = np.zeros(len(train))
    for tr_i, te_i in folds:
        oof[te_i] = make(best).fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
    joblib.dump(oof, ROOT / "artifacts" / "oof_p_rf_tuned.joblib")
    met, _ = score(train, oof, tr_obs)
    print(f"OOF (tuned, full-data folds) train Top-1 {met['top1']:.4f}, logloss {met['logloss']:.4f}", flush=True)


if __name__ == "__main__":
    main()

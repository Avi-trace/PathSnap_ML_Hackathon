"""Cross-validation Macro-F1 (and Top-1, log loss) of the TUNED Random Forest, per fold, next to the untuned one.

Script 04 tuned on CV Top-1 and saved only that number for the tuned model. The report (Deliverables.pdf, section 4) asks for the
untuned-vs-tuned comparison on cross-validation Macro-F1, so this script repeats the SAME protocol as scripts 03/04
(78 usable train trips, the same 3 trip-grouped folds, the same fixed 30% fit subsample, scoring on the full held-out fold)
for both configurations. The untuned row must reproduce the Random Forest rows of reports/cv_fold_results.csv (checked below).

usage: python scripts/04b_tuned_cv_macro_f1.py        -> reports/tuned_vs_untuned_cv.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
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
    best = json.load(open(ROOT / "reports" / "optuna_best.json"))["best_params"]
    train, tr_obs = load_part("train", EXCLUDE_FROM_FIT)
    X, y = train[FEATURES].to_numpy(np.float32), train["label"].to_numpy()
    folds, sampled = cv_setup(train)
    rows = []
    for tag, params in (("untuned", UNTUNED), ("tuned", best)):
        for k, (tr_i, te_i) in enumerate(folds):
            fit_i = tr_i[sampled[tr_i]]
            t0 = time.time()
            m = make(params).fit(X[fit_i], y[fit_i])
            met, _ = score(train.iloc[te_i], m.predict_proba(X[te_i])[:, 1], tr_obs)
            rows.append(dict(variant=tag, fold=k, macro_f1=met["macro_f1"], top1=met["top1"], logloss=met["logloss"], fit_s=time.time() - t0))
            print(f"{tag} fold {k}: macro-F1 {met['macro_f1']:.4f} top1 {met['top1']:.4f} logloss {met['logloss']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "tuned_vs_untuned_cv.csv", index=False)

    # consistency check against script 03's stored fold results for the untuned forest
    ref = pd.read_csv(ROOT / "reports" / "cv_fold_results.csv")
    ref = ref[ref.model == "RandomForest"].sort_values("fold")
    mine = df[df.variant == "untuned"].sort_values("fold")
    print("max |macro-F1 difference| vs cv_fold_results.csv (untuned RF):", float(np.abs(ref.macro_f1.to_numpy() - mine.macro_f1.to_numpy()).max()))
    print(df.groupby("variant")[["macro_f1", "top1", "logloss"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()

"""Over/under-fitting evidence: score each saved final model on its OWN training rows.

The models in artifacts/models/*.joblib were fitted on ALL rows of the 78 usable train trips
(see scripts/03_train_models.py, final refit). This script reloads the same cached training table
(pathsnap.experiment.load_part) and scores each model on it with pathsnap.experiment.score, giving a
training-set Top-1/macro-F1/log loss to put next to the CV Top-1 and student Top-1 already in
reports/model_comparison.csv. That gap is the over/under-fitting evidence requested by
team_packages/PIECE_2_MODELS_TUNING.md ("Known gap you must close").

Writes: reports/train_vs_cv_vs_student.csv, reports/figures/model_overfitting.png
usage: python scripts/11_train_accuracy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.experiment import EXCLUDE_FROM_FIT, load_part, score  # noqa: E402
from pathsnap.features import FEATURES                              # noqa: E402
import pathsnap.mlp  # noqa: E402,F401  (needed to unpickle TorchMLP)

REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "artifacts" / "models"

# file stem -> display name matching reports/model_comparison.csv "model" column
MODEL_FILES = {
    "GaussianNB": "GaussianNB",
    "LogisticRegression": "LogisticRegression",
    "DecisionTree": "DecisionTree",
    "RandomForest": "RandomForest",
    "ExtraTrees": "ExtraTrees",
    "HistGradientBoosting": "HistGradientBoosting",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "MLP_PyTorch": "MLP(PyTorch)",
}
CHUNK = 200_000  # rows per predict_proba call, to keep memory down (~3-4 GB free)


def predict_in_chunks(m, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), CHUNK):
        out[i:i + CHUNK] = m.predict_proba(X[i:i + CHUNK])[:, 1]
    return out


def main():
    t0 = time.time()
    train, tr_obs = load_part("train", EXCLUDE_FROM_FIT)
    print(f"train rows {len(train):,} ({train.trip.nunique()} trips), loaded in {time.time() - t0:.0f}s", flush=True)
    X = train[FEATURES].to_numpy(np.float32)

    cmp = pd.read_csv(REPORTS / "model_comparison.csv").set_index("model")

    rows = []
    for stem, name in MODEL_FILES.items():
        f = MODELS_DIR / f"{stem}.joblib"
        if not f.exists():
            print(f"!! missing {f}, skipping {name}", flush=True)
            continue
        t_m = time.time()
        m = joblib.load(f)
        p = predict_in_chunks(m, X)
        met, _ = score(train, p, tr_obs)
        del m
        cv_top1 = cmp.loc[name, "cv_top1_mean"]
        st_top1 = cmp.loc[name, "st_top1"]
        cv_f1 = cmp.loc[name, "cv_macro_f1_mean"]
        st_f1 = cmp.loc[name, "st_macro_f1"]
        rows.append({"model": name, "train_top1": met["top1"], "train_top1_cond": met["top1_cond"],
                     "train_macro_f1": met["macro_f1"], "train_logloss": met["logloss"],
                     "cv_top1_mean": cv_top1, "st_top1": st_top1,
                     "train_minus_cv_top1": met["top1"] - cv_top1, "train_minus_st_top1": met["top1"] - st_top1,
                     "cv_macro_f1_mean": cv_f1, "st_macro_f1": st_f1, "train_minus_cv_macro_f1": met["macro_f1"] - cv_f1})
        print(f"[{name}] train top1={met['top1']:.4f} f1={met['macro_f1']:.4f} ll={met['logloss']:.4f} "
              f"(cv={cv_top1:.4f} st={st_top1:.4f} gap-vs-cv={met['top1'] - cv_top1:+.4f}) "
              f"{time.time() - t_m:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "train_vs_cv_vs_student.csv", index=False)
    print(f"\nwrote {REPORTS / 'train_vs_cv_vs_student.csv'}")

    # ---------------------------------------------------------------- grouped bar chart per model
    order = [n for n in MODEL_FILES.values() if n in out["model"].values]
    out = out.set_index("model").loc[order].reset_index()
    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, out["train_top1"], width=w, label="Train Top-1 (own training rows)", color="#54a24b")
    ax.bar(x, out["cv_top1_mean"], width=w, label="CV Top-1 (held-out folds)", color="#4c78a8")
    ax.bar(x + w, out["st_top1"], width=w, label="Student-test Top-1", color="#e45756")
    ax.set_xticks(x)
    short = {"GaussianNB": "GaussianNB", "LogisticRegression": "LogReg", "DecisionTree": "DecTree",
             "RandomForest": "RandForest", "ExtraTrees": "ExtraTrees", "HistGradientBoosting": "HistGB",
             "LightGBM": "LightGBM", "XGBoost": "XGBoost", "MLP(PyTorch)": "MLP"}
    ax.set_xticklabels([short[m] for m in order], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("observation-level Top-1")
    ax.set_title("Train vs CV vs student-test Top-1 per model (over/underfitting evidence)\n"
                  "Train: own training rows (final refit, all 78 train trips). CV: held-out 3-fold group CV "
                  "(train). Student: 10 student_test trips, final refit.", fontsize=11)
    for xi, v in zip(x, out["train_top1"]):
        ax.text(xi - w, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    for xi, v in zip(x, out["cv_top1_mean"]):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    for xi, v in zip(x, out["st_top1"]):
        ax.text(xi + w, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_overfitting.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'model_overfitting.png'}")

    print("\n===== train-minus-CV Top-1 gap (overfitting evidence) =====")
    print(out[["model", "train_top1", "cv_top1_mean", "st_top1"]].assign(
        gap_vs_cv=lambda d: d.train_top1 - d.cv_top1_mean).round(4).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

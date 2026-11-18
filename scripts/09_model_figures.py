"""Model-comparison figures and tables for report section 3 (and the Optuna evidence for section 4).

Reads ONLY existing result files (never retrains, never retypes a number):
  reports/model_comparison.csv   reports/cv_fold_results.csv   reports/model_configs.json
  reports/optuna_trials.csv      reports/optuna_best.json      reports/tuned_vs_untuned.csv

Writes:
  reports/tables/model_comparison_table.csv / .md
  reports/tables/tuned_vs_untuned_table.md
  reports/figures/model_cv_macro_f1.png
  reports/figures/model_student_top1_f1.png
  reports/figures/model_logloss.png
  reports/figures/model_fit_infer_time.png
  reports/figures/model_confusion_matrices.png
  reports/figures/model_mlp_loss_curve.png
  reports/figures/model_optuna_trials.png
  reports/figures/model_optuna_params.png

usage: python scripts/09_model_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
TABDIR = REPORTS / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# consistent model order: by family then the order used in model_configs.json / report
MODEL_ORDER = ["GaussianNB", "LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees",
               "HistGradientBoosting", "LightGBM", "XGBoost", "MLP(PyTorch)"]
SHORT = {"GaussianNB": "GaussianNB", "LogisticRegression": "LogReg", "DecisionTree": "DecTree",
         "RandomForest": "RandForest", "ExtraTrees": "ExtraTrees", "HistGradientBoosting": "HistGB",
         "LightGBM": "LightGBM", "XGBoost": "XGBoost", "MLP(PyTorch)": "MLP"}

PALETTE = {"Probabilistic": "#8a8a8a", "Linear": "#4c78a8", "Tree": "#72b7b2",
           "Bagging ensemble": "#54a24b", "Boosting": "#e45756", "Neural network": "#b279a2"}


def to_md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids the optional 'tabulate' dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def colors_for(models: pd.Series, family: pd.Series) -> list[str]:
    return [PALETTE.get(f, "#999999") for f in family]


def main():
    cmp = pd.read_csv(REPORTS / "model_comparison.csv").set_index("model").loc[MODEL_ORDER].reset_index()
    cv = pd.read_csv(REPORTS / "cv_fold_results.csv")
    configs = json.load(open(REPORTS / "model_configs.json"))
    optuna_trials = pd.read_csv(REPORTS / "optuna_trials.csv")
    optuna_best = json.load(open(REPORTS / "optuna_best.json"))
    tuned_vs = pd.read_csv(REPORTS / "tuned_vs_untuned.csv")

    cv_std = cv.groupby("model")["macro_f1"].std().reindex(MODEL_ORDER)
    cv_f1_mean = cv.groupby("model")["macro_f1"].mean().reindex(MODEL_ORDER)

    # ---------------------------------------------------------------- 1) consolidated table (report 3.1)
    table = pd.DataFrame({
        "Model": cmp["model"],
        "Algorithm family": cmp["family"],
        "Key configuration": cmp["config"],
        "CV macro-F1 (mean +/- std)": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(cv_f1_mean.values, cv_std.values)],
        "CV Top-1": cmp["cv_top1_mean"].map(lambda v: f"{v:.4f}"),
        "Route continuity (student)": cmp["st_route_continuity"].map(lambda v: f"{v:.4f}"),
        "Student-test macro-F1": cmp["st_macro_f1"].map(lambda v: f"{v:.4f}"),
        "Log loss": cmp["st_logloss"].map(lambda v: f"{v:.4f}"),
        "Inference time (s/100k rows)": cmp["infer_s_per_100k_rows"].map(lambda v: f"{v:.4f}"),
    })
    table.to_csv(TABDIR / "model_comparison_table.csv", index=False)
    with open(TABDIR / "model_comparison_table.md", "w") as f:
        f.write("# Model comparison (report section 3.1)\n\n")
        f.write("Data: CV columns are the mean (+/- std) over the 3 trip-grouped CV folds "
                "(`cv_fold_results.csv`); all other columns are from the final refit scored on the "
                "10 student_test trips (`model_comparison.csv`).\n\n")
        f.write(to_md_table(table))
        f.write("\n")
    print("wrote", TABDIR / "model_comparison_table.csv", "and .md")

    # ---------------------------------------------------------------- 2a) CV macro-F1 bar chart with std error bars
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODEL_ORDER))
    cols = colors_for(cmp["model"], cmp["family"])
    ax.bar(x, cv_f1_mean.values, yerr=cv_std.values, capsize=4, color=cols, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel("CV macro-F1 (mean +/- std over 3 folds)")
    ax.set_title("3-fold group-CV macro-F1 per model\n(partition: train, 3 trip-grouped folds; fit on 30% obs subsample)")
    ax.set_ylim(0, 1.0)
    for xi, v in zip(x, cv_f1_mean.values):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_cv_macro_f1.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 2b) student Top-1 and macro-F1 side by side
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.38
    ax.bar(x - w / 2, cmp["st_top1"], width=w, label="Student-test Top-1", color="#4c78a8")
    ax.bar(x + w / 2, cmp["st_macro_f1"], width=w, label="Student-test macro-F1", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Student-test Top-1 and macro-F1 per model (partition: student_test, final refit)")
    for xi, v in zip(x, cmp["st_top1"]):
        ax.text(xi - w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    for xi, v in zip(x, cmp["st_macro_f1"]):
        ax.text(xi + w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_student_top1_f1.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 2c) log loss (log scale, GaussianNB dominates)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, cmp["st_logloss"], color=cols, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Student-test log loss (log scale)")
    ax.set_title("Student-test log loss per model (log scale; GaussianNB dominates the linear scale)")
    for xi, v in zip(x, cmp["st_logloss"]):
        ax.text(xi, v * 1.08, f"{v:.4f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_logloss.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 2d) fit time and inference time
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, cmp["fit_s_full"], color=cols, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right")
    axes[0].set_ylabel("fit time, full refit (s, log scale)")
    axes[0].set_title("Final-refit fit time (all 78 train trips, all rows)")
    for xi, v in zip(x, cmp["fit_s_full"]):
        axes[0].text(xi, v * 1.08, f"{v:.0f}s", ha="center", fontsize=7)

    axes[1].bar(x, cmp["infer_s_per_100k_rows"], color=cols, edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right")
    axes[1].set_ylabel("inference time (s per 100k rows)")
    axes[1].set_title("Student-test inference time")
    for xi, v in zip(x, cmp["infer_s_per_100k_rows"]):
        axes[1].text(xi, v + v * 0.04, f"{v:.3f}", ha="center", fontsize=7)
    fig.suptitle("Fit and inference time per model (partition: train fit / student_test inference)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_fit_infer_time.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 3) confusion matrices grid (student-test)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (_, row) in zip(axes.flat, cmp.iterrows()):
        tn, fp, fn, tp = row.st_tn, row.st_fp, row.st_fn, row.st_tp
        counts = np.array([[tn, fp], [fn, tp]], dtype=float)
        row_sum = counts.sum(axis=1, keepdims=True)
        norm = counts / row_sum
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                txt_color = "white" if norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{norm[i, j]:.3f}\n(n={int(counts[i, j]):,})", ha="center", va="center",
                        fontsize=8, color=txt_color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(SHORT[row.model], fontsize=10)
    fig.suptitle("Student-test confusion matrices (row-normalised, counts annotated); threshold 0.5 on candidate "
                 "probability; partition: student_test, final refit", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "model_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 4) MLP train vs val loss
    hist = configs["mlp_history"]
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], "o-", label="train loss", color="#4c78a8")
    ax.plot(epochs, hist["val_loss"], "s-", label="validation loss", color="#e45756")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("MLP (PyTorch): train vs validation loss per epoch", fontsize=13)
    fig.text(0.5, 0.955, "Note: only the MLP has an epoch-wise history in this study; the other 8 models fit in a "
             "single call and have no comparable curve here.", ha="center", va="top", fontsize=8.5, wrap=True)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGDIR / "model_mlp_loss_curve.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 5) Optuna trial history + best-so-far
    n_startup = 5  # from optuna_best.json "sampler": "TPE (seed 0, 5 random start-up trials)"
    optuna_trials = optuna_trials.sort_values("number").reset_index(drop=True)
    is_startup = optuna_trials["number"] < n_startup
    best_so_far = optuna_trials["value"].cummax()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(optuna_trials.loc[is_startup, "number"], optuna_trials.loc[is_startup, "value"],
               color="#4c78a8", label="random start-up trial (5)", s=70, zorder=3)
    ax.scatter(optuna_trials.loc[~is_startup, "number"], optuna_trials.loc[~is_startup, "value"],
               color="#e45756", label="TPE-guided trial (4)", s=70, zorder=3)
    ax.plot(optuna_trials["number"], best_so_far, "k--", alpha=0.6, label="best-so-far")
    ax.set_xlabel("trial number")
    ax.set_ylabel("mean 3-fold CV Top-1 (objective)")
    ax.set_title(f"Optuna TPE trial history, Random Forest tuning (n={len(optuna_trials)} trials)", fontsize=12)
    fig.text(0.5, 0.955, "Objective: mean 3-fold group-CV Top-1. Partition: train, same CV folds as script 03.",
              ha="center", va="top", fontsize=9)
    ax.set_xticks(optuna_trials["number"])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGDIR / "model_optuna_trials.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 5b) per-hyperparameter scatter vs CV Top-1
    params = ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, p in zip(axes.flat, params):
        col = f"params_{p}"
        ax.scatter(optuna_trials.loc[is_startup, col], optuna_trials.loc[is_startup, "value"],
                   color="#4c78a8", label="random start-up", s=60)
        ax.scatter(optuna_trials.loc[~is_startup, col], optuna_trials.loc[~is_startup, "value"],
                   color="#e45756", label="TPE-guided", s=60)
        ax.set_xlabel(p)
        ax.set_ylabel("CV Top-1")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Optuna: hyper-parameter value vs CV Top-1 (n={len(optuna_trials)} trials)", fontsize=13)
    fig.text(0.5, 0.945, "Honest scatter, NOT a parameter-importance plot -- too few trials for that. "
             "Partition: train, same CV folds as script 03.", ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGDIR / "model_optuna_params.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 5c) tuned vs untuned table
    tv = tuned_vs.set_index("variant")
    with open(TABDIR / "tuned_vs_untuned_table.md", "w") as f:
        f.write("# Tuned vs untuned Random Forest (report section 4)\n\n")
        f.write(f"Tuning: Optuna TPE, {optuna_best['n_trials']} trials (5 random start-up, 4 sampler-guided), "
                f"{optuna_best['tuning_seconds']:.0f}s total tuning time. Objective: mean 3-fold group-CV Top-1 "
                "(partition: train). Student-test rows are the final refit on all 78 train trips, scored on the "
                "10 student_test trips.\n\n")
        out = pd.DataFrame({
            "Metric": ["CV Top-1", "Student Top-1", "Student macro-F1", "Student log loss",
                       "Inference time (s/100k rows)", "Fit time, full refit (s)", "Tuning cost (trials / seconds)"],
            "Untuned": [f"{tv.loc['untuned', 'cv_top1']:.4f}", f"{tv.loc['untuned', 'st_top1']:.4f}",
                        f"{tv.loc['untuned', 'st_macro_f1']:.4f}", f"{tv.loc['untuned', 'st_logloss']:.4f}",
                        f"{tv.loc['untuned', 'infer_s_per_100k_rows']:.4f}", f"{tv.loc['untuned', 'fit_s_full']:.1f}",
                        "n/a (default params)"],
            "Tuned": [f"{tv.loc['tuned', 'cv_top1']:.4f}", f"{tv.loc['tuned', 'st_top1']:.4f}",
                      f"{tv.loc['tuned', 'st_macro_f1']:.4f}", f"{tv.loc['tuned', 'st_logloss']:.4f}",
                      f"{tv.loc['tuned', 'infer_s_per_100k_rows']:.4f}", f"{tv.loc['tuned', 'fit_s_full']:.1f}",
                      f"{optuna_best['n_trials']} trials / {optuna_best['tuning_seconds']:.0f}s"],
        })
        f.write(to_md_table(out))
        f.write("\n\nBest params (tuned): " + json.dumps(optuna_best["best_params"]) + "\n")
    print("wrote", TABDIR / "tuned_vs_untuned_table.md")

    print("done:", FIGDIR)


if __name__ == "__main__":
    main()

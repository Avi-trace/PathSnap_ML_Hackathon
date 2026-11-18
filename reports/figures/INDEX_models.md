# Model-comparison outputs index (scripts/09_model_figures.py, 11_train_accuracy.py, 12_boosting_curves.py)

Every number comes from `reports/model_comparison.csv`, `cv_fold_results.csv`, `model_configs.json`, `optuna_trials.csv`, `optuna_best.json`, `tuned_vs_untuned.csv`, or a fresh scoring run against the cached candidate tables / saved models (`reports/train_vs_cv_vs_student.csv`, `reports/boosting_curves.json`). No number is hand-typed.

- **`reports/tables/model_comparison_table.csv / .md`** -- Consolidated comparison table (report section 3.1): Model, Algorithm family, Key configuration, CV macro-F1 (mean +/- std over 3 folds), CV Top-1, Route continuity (student), Student-test macro-F1, Log loss, Inference time.
  - source: model_comparison.csv (cv_top1_mean, st_*, infer_s_per_100k_rows, config) + cv_fold_results.csv (per-fold macro_f1, for mean/std)
  - partition: CV columns from train (3 trip-grouped folds); all other columns from the final refit (train, all 78 trips) scored on student_test (10 trips)
- **`reports/figures/model_cv_macro_f1.png`** -- Bar chart: 3-fold group-CV macro-F1 per model with std error bars.
  - source: cv_fold_results.csv (grouped by model)
  - partition: train, 3 trip-grouped CV folds, fit on the fixed 30% observation subsample
- **`reports/figures/model_student_top1_f1.png`** -- Bar chart: student-test Top-1 and macro-F1 side by side, per model.
  - source: model_comparison.csv (st_top1, st_macro_f1)
  - partition: student_test, final refit
- **`reports/figures/model_logloss.png`** -- Bar chart: student-test log loss per model, log scale (GaussianNB dominates the linear scale).
  - source: model_comparison.csv (st_logloss)
  - partition: student_test, final refit
- **`reports/figures/model_fit_infer_time.png`** -- Two bar charts: final-refit fit time (log scale) and student-test inference time per model.
  - source: model_comparison.csv (fit_s_full, infer_s_per_100k_rows)
  - partition: train (fit) / student_test (inference), final refit
- **`reports/figures/model_confusion_matrices.png`** -- 3x3 grid of student-test confusion matrices, one per model, row-normalised with raw counts annotated.
  - source: model_comparison.csv (st_tn, st_fp, st_fn, st_tp); threshold 0.5 on the candidate probability
  - partition: student_test, final refit
- **`reports/figures/model_mlp_loss_curve.png`** -- MLP (PyTorch) train vs validation BCE loss per epoch. Caption states that only the MLP has an epoch-wise history in this study; the other 8 models fit in a single call and have no comparable curve.
  - source: model_configs.json -> mlp_history (train_loss, val_loss)
  - partition: train (internal 95/5 split inside the MLP's own fit call on the final refit)
- **`reports/figures/model_optuna_trials.png`** -- Optuna TPE trial history for Random Forest tuning: objective value per trial and best-so-far, with the 5 random start-up trials and 4 sampler-guided trials labelled separately.
  - source: optuna_trials.csv (value, number); optuna_best.json (n_startup = 5)
  - partition: train, same 3 CV folds as scripts/03_train_models.py
- **`reports/figures/model_optuna_params.png`** -- Scatter of each tuned hyper-parameter (n_estimators, max_depth, min_samples_leaf, max_features, max_samples) vs CV Top-1, start-up vs sampler-guided trials coloured separately. Honest scatter, not a parameter-importance plot (only 9 trials).
  - source: optuna_trials.csv
  - partition: train, same 3 CV folds as scripts/03_train_models.py
- **`reports/tables/tuned_vs_untuned_table.md`** -- Tuned vs untuned Random Forest: CV Top-1, student Top-1/macro-F1/log loss, inference time, fit time, and tuning cost (n_trials, tuning_seconds).
  - source: tuned_vs_untuned.csv + optuna_best.json (n_trials, tuning_seconds, best_params)
  - partition: CV Top-1 from train (3 folds); student_test columns from the final refit on train, scored on student_test
- **`reports/train_vs_cv_vs_student.csv`** -- Training-set Top-1/Top-1-cond/macro-F1/log loss for each of the 9 saved final models, scored on their OWN training rows, alongside the existing CV Top-1 and student Top-1 (closes the "known gap" in PIECE_2_MODELS_TUNING.md: training accuracy was not recorded by script 03).
  - source: fresh scoring run -- `pathsnap.experiment.load_part("train", EXCLUDE_FROM_FIT)` + `pathsnap.experiment.score` against each `artifacts/models/*.joblib`, joined with model_comparison.csv's cv_top1_mean/st_top1
  - partition: train, all 78 trips, all rows (each model's own fit data)
- **`reports/figures/model_overfitting.png`** -- Grouped bar chart per model: train Top-1 (own rows) vs CV Top-1 (held-out folds) vs student-test Top-1 -- the over/under-fitting evidence.
  - source: train_vs_cv_vs_student.csv
  - partition: train (train + CV columns) / student_test (final refit)
- **`reports/boosting_curves.json`** -- Per-boosting-round train/validation logloss for LightGBM and XGBoost (300 rounds each), hyper-parameters identical to scripts/03_train_models.py.
  - source: fresh fit with eval_set=[(fit),(validation)], eval_metric logloss, on a SEPARATE trip-level 80/20 fit/validation split (seed 0) of the 78 train trips -- NOT the 3-fold CV folds. Fit rows subsampled to a fixed random 30% of observations (same idea as script 03); validation used in full, never subsampled.
  - partition: train, 62 fit trips (679,696 fit rows) / 16 validation trips (625,735 rows)
- **`reports/figures/model_boosting_curves.png`** -- Plot of the above: LightGBM and XGBoost, train (solid) vs validation (dashed) logloss per round. Caption states this is a separate trip-level split, not the CV folds.
  - source: boosting_curves.json
  - partition: train, same 80/20 trip split as above

## Over/under-fitting gap (train Top-1 minus CV Top-1 mean), from `reports/train_vs_cv_vs_student.csv`

| Model | Train Top-1 | CV Top-1 | Student Top-1 | Train - CV gap |
|---|---|---|---|---|
| GaussianNB | 0.8023 | 0.7989 | 0.7901 | +0.0034 |
| LogisticRegression | 0.8575 | 0.8545 | 0.8556 | +0.0030 |
| DecisionTree | 0.8907 | 0.8593 | 0.8712 | +0.0314 |
| RandomForest | 0.8904 | 0.8658 | 0.8730 | +0.0246 |
| ExtraTrees | 0.8718 | 0.8615 | 0.8612 | +0.0103 |
| HistGradientBoosting | 0.8945 | 0.8575 | 0.8643 | +0.0370 |
| LightGBM | 0.8948 | 0.8582 | 0.8585 | +0.0365 |
| XGBoost | 0.9369 | 0.8635 | 0.8707 | +0.0734 |
| MLP(PyTorch) | 0.8896 | 0.8634 | 0.8680 | +0.0262 |

Reading it: the linear/probabilistic models (GaussianNB, LogisticRegression) barely overfit (gap <0.004) -- they are too constrained to memorise 2.88M training rows. XGBoost shows the largest train-CV gap (+0.073, train Top-1 0.937) despite early stopping not being used and default depth 8 over 300 rounds -- consistent with scripts/12's boosting-curve plot, where XGBoost's train logloss keeps dropping while validation logloss flattens. LightGBM and HistGradientBoosting show similar-sized gaps (~0.036-0.037); script 12's curve shows LightGBM's validation logloss is visibly noisier (spikes) than XGBoost's smoother validation curve, at the same nominal hyper-parameters, on a different (80/20 trip) split. Random Forest and Decision Tree sit in between (+0.025 to +0.031); their min_samples_leaf=50 constraint limits memorisation. Despite these gaps, every model's student-test Top-1 is close to (and for several models, above) its own CV Top-1 -- the CV folds (grouped by trip) are a fair, slightly pessimistic estimate of unseen-trip performance, and none of the gaps change the model ranking from `model_comparison.csv`.

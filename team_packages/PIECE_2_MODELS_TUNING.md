# Piece 2: Models & Bayesian tuning (report sections 3 and 4)

## Your part
Own the model experiments: nine models from four algorithm families, how they are compared, which one is selected and why, and the Bayesian hyper-parameter optimisation of the selected one.
You answer, **for every model**: why suitable, strengths/limits, hyper-parameters and loss, student-test results, over/under-fitting evidence.

## What is already built (read these files)
| File | What it does |
|---|---|
| `pathsnap/experiment.py` | loads the cached candidate tables, builds the fixed CV folds, scores predictions |
| `pathsnap/metrics.py` | observation-level Top-1, macro-F1, precision/recall, log loss, confusion matrix, route continuity |
| `pathsnap/mlp.py` | PyTorch MLP with an epoch-wise train/validation loss history |
| `scripts/03_train_models.py` | trains and compares the 9 models -> `reports/model_comparison.csv`, `cv_fold_results.csv`, `model_configs.json`, `logs/03_train_models.log` |
| `scripts/04_tune_random_forest.py` | Optuna (TPE) tuning of the selected model, tuned vs untuned -> `reports/optuna_trials.csv`, `optuna_best.json`, `tuned_vs_untuned.csv` (already in `reports/`; re-running takes about 25 minutes) |

Run: first `scripts/02_build_dataset.py` (see TEAM_README), then `python scripts/03_train_models.py` (about 10 min). Seeds are fixed (0).

## Explain it (plain language, with the real numbers)
**The task for the classifier.** For each GPS point there are about 16 candidate arcs; exactly one is the reference arc. The model gives each candidate a probability of being the right one (18 features, see piece 1).
Training rows: 2.88 M candidates from 78 trips (two trips with unusable reference routes are excluded).

**Protocol (fixed before looking at results).** 3-fold cross-validation **grouped by trip** (a trip never appears in both training and validation, otherwise neighbouring GPS points leak). To save time, CV fits use a fixed random 30 % of the *observations*
(all candidates of a sampled observation are kept) and are scored on the **full** held-out fold. The final refit uses all fit trips and all rows and is scored on the 10 `student_test` trips.
Selection metric chosen up front: **observation-level Top-1** (for each GPS point, is the highest-scoring candidate the reference arc?). It matches the real goal and needs no probability threshold; log loss breaks ties because the HMM uses the probabilities.

**The nine models** (config in `model_configs.json`):
| Model | Family | Key configuration |
|---|---|---|
| Gaussian Naive Bayes | probabilistic | default smoothing |
| Logistic Regression | linear | standardised inputs, C=1, lbfgs |
| Decision Tree | tree | depth 12, min leaf 50 |
| Random Forest | bagging ensemble | 50 trees, depth 16, min leaf 50 |
| Extra Trees | bagging ensemble | 50 trees, depth 16, min leaf 50 |
| HistGradientBoosting | boosting | 300 iterations, lr 0.1, 63 leaves |
| LightGBM | boosting | 300 rounds, lr 0.1, 63 leaves |
| XGBoost | boosting | 300 rounds, lr 0.1, depth 8, hist |
| MLP (PyTorch) | neural network | 128-64 ReLU, Adam 1e-3, 15 epochs, BCE loss |

**Results** (3-fold CV Top-1 / student Top-1 / student macro-F1 / log loss): Random Forest 0.8658 / 0.8730 / 0.9300 / 0.0345 (best on every one); XGBoost 0.8635 / 0.8707 / 0.9280 / 0.0360; MLP 0.8634 / 0.8680 / 0.9227 / 0.0385;
Extra Trees 0.8615 / 0.8612 / 0.9175 / 0.0468; Decision Tree 0.8593 / 0.8712 / 0.9276 / 0.0394; LightGBM 0.8582 / 0.8585 / 0.9248 / 0.0506; HistGB 0.8575 / 0.8643 / 0.9276 / 0.0407; Logistic Regression 0.8545 / 0.8556 / 0.9056 / 0.0524; Gaussian NB 0.7989 / 0.7901 / 0.8163 / 0.3551.

**How to read them honestly.** The top seven are within one fold-standard-deviation (about 0.03: fold 2 is an easier set of trips than fold 0), so do not claim large differences. Random Forest wins on the agreed metric and has the best probability quality.
Gaussian NB is clearly worst: it assumes independent features, but distance, rank and gap are strongly related. LightGBM's high log loss (0.0506) with default settings suggests over-confident probabilities. A ceiling near 0.87 for all of them says the per-point features limit accuracy
(direction/junction ambiguity), which is what the HMM then fixes by using the sequence.

**Bayesian optimisation.** Optuna's TPE sampler (a Bayesian method: it models which parameter regions gave good scores and samples there) on the Random Forest. Objective: mean CV Top-1 on the same folds, so tuned and untuned are comparable.
Space: trees 30-80, depth 10-20, min leaf 10-200 (log), max features 0.2-1.0, bootstrap share 0.3-0.7. 9 trials were run (5 random start-up, then 4 guided by the sampler; a 20-minute budget limited it). Best CV Top-1 0.8684 (65 trees, depth 17, min leaf 10, max features 0.33, bootstrap share 0.64) vs 0.8658 untuned (+0.26 points).
Every trial was between 0.8650 and 0.8684, so the model is not very sensitive to these settings. On student-test the tuned forest is a **tie, not a win**: Top-1 0.8720 vs 0.8730, macro-F1 0.9295 vs 0.9300, log loss 0.0342 vs 0.0345 (fit 156 s vs 190 s; inference 0.15 vs 0.11 s per 100k rows).
Report it honestly: the gain is within noise, and only 9 trials were possible, so a larger search is unlikely to change the conclusion. Files: `reports/optuna_trials.csv`, `optuna_best.json`, `tuned_vs_untuned.csv`.

## Your deliverables
`scripts/09_model_figures.py` already produces the items below under `reports/figures/` and `reports/tables/` **from the CSV/JSON files only** (see `reports/figures/INDEX_models.md`).
**Your job: run it, read the script until you can explain every plot, check the figures, extend where needed, and write the per-model report text yourself.** The items:
1. The consolidated comparison table of report section 3.1 (Model, Family, Key configuration, CV macro-F1, Top-1, Route continuity, Student-test macro-F1, Log loss, Inference time).
2. Bar charts: CV macro-F1 per model (with fold std), student Top-1 and macro-F1, log loss (where applicable).
3. Confusion matrices on student-test (counts are in `model_comparison.csv`: `st_tn st_fp st_fn st_tp`).
4. MLP training vs validation loss per epoch (`model_configs.json` -> `mlp_history`). **No fake curves** for models without epoch histories; say "N/A" and why.
5. Optuna: trial history, best-so-far, parameter importance, and the tuned-vs-untuned table (CV macro-F1/Top-1, student Top-1, macro-F1, log loss, inference time, tuning cost).
6. For each of the nine models, a subsection answering the questions A-C of the brief (why suitable; strengths/limits; hyper-parameters and loss; student results; over/under-fitting) using your own words and the numbers.

**Over/under-fitting evidence (now recorded):** `scripts/11_train_accuracy.py` scores each saved final model on its own training rows (`reports/train_vs_cv_vs_student.csv`, figure `model_overfitting.png`).
Train-minus-CV Top-1 gap: Gaussian NB +0.003 and Logistic Regression +0.003 (barely over-fit, but also the weakest or simplest), Extra Trees +0.010, Random Forest +0.025, MLP +0.026, Decision Tree +0.031, LightGBM +0.037, HistGradientBoosting +0.037, **XGBoost +0.073** (train 0.937 vs CV 0.864, the largest).
`scripts/12_boosting_curves.py` (separate 80/20 trip split, not the CV folds) shows both boosters' validation log loss stops improving after roughly 40-60 of 300 rounds and then drifts up (LightGBM's is also spiky), so 300 rounds over-trains them. Discuss what that means for each model.
Also compute CV macro-F1 per fold if you want error bars on F1 (already in `cv_fold_results.csv`).

## Likely viva questions
Why group the folds by trip? Why is Top-1 the selection metric and what are its weaknesses? Why did Gaussian NB fail? What does TPE do differently from random search? Why are the top models so close and what does that mean? What is log loss and why does the HMM care?
What did the MLP's validation loss show? What would you try with more time?

## Hand back
New scripts and everything under `reports/figures/`, plus your written subsections.

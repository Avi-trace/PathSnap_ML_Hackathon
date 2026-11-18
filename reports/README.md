# reports/: every result behind the technical report

All numbers and figures here are produced by scripts in `../scripts/` from the `train` and `student_test` partitions (the evaluator partition is never read).
Figures and tables were generated from the CSV/JSON files below, not typed by hand. Each folder has an index file saying what every figure shows and which file it came from.

## Where to find what, by report section

| Report section | Files |
|---|---|
| **2.2 Dataset understanding and EDA** | `eda_trip_summary.csv` (one row per trip); `tables/eda_inventory`, `eda_missing_invalid`, `eda_network_connectivity`, `eda_data_quality_log` (.csv/.md); `figures/eda_01 ... eda_07` (`INDEX_eda.md`); Folium map `maps/eda_trip_00000085.html` + `screenshots/eda_map.png` |
| **3.1 Overall model comparison table** | `tables/model_comparison_table.md/.csv` (from `model_comparison.csv`, `cv_fold_results.csv`) |
| **3.2 Per-model answers** | `model_comparison.csv`, `cv_fold_results.csv`, `model_configs.json` (configs, seeds, MLP epoch history), `train_vs_cv_vs_student.csv` (over/under-fitting), `logs/03_train_models.log` |
| **3.3 Model visualisations** | `figures/model_cv_macro_f1`, `model_student_top1_f1`, `model_logloss`, `model_fit_infer_time`, `model_confusion_matrices`, `model_mlp_loss_curve`, `model_boosting_curves` (LightGBM/XGBoost per round), `model_overfitting` (`figures/INDEX_models.md`) |
| **4 Bayesian optimisation** | `optuna_trials.csv`, `optuna_best.json`, `tuned_vs_untuned.csv`, `tables/tuned_vs_untuned_table.md`, `figures/model_optuna_trials`, `model_optuna_params`, `logs/04_tune_random_forest.log` |
| **5 HMM and Viterbi** | `hmm_best.json`, `hmm_trials.csv` (decoder tuning), `classifier_vs_hmm.csv`, `classifier_vs_hmm_per_trip.csv`, `hmm_diagnostics*.csv` (where the HMM helps or hurts), `logs/07_tune_decoder.log` |
| **6 Folium maps** | `maps/pred_trip_<id>.html` (+ `_zoom` variants) for trips 00000052, 00000009, 00000040, 00000007, 00000077; screenshots in `screenshots/map_pred_*.png`; `maps/INDEX_maps.md` |
| **7 API, UI and deployment** | UI screenshots `screenshots/ui_01 ... ui_05` (connect, intro, trip list, finished run, error state). **Still to capture at the demo: the API server console showing the request log lines (the brief asks for the receiving laptop's console).** |
| Pipeline consistency check | `pipeline_check_student.csv` (raw files through the API code path reproduce the training-time Top-1) |

## Headline results (student_test, development data)

- Selected model: Random Forest (best CV Top-1 0.8658 among nine; tuned CV 0.8684; student-test untuned 0.8730 vs tuned 0.8720, a tie).
- Classifier only -> + HMM with tuned decoder: pointwise accuracy 87.20 % -> 91.39 %, route continuity 89.17 % -> 99.16 %, route F1 87.93 % -> 93.54 %.
- Worst case for the HMM: trip `00000007` (92.54 % -> 90.10 %); hardest trip for both: `00000077`.
- Two training trips (`00000033`, `00000054`) were excluded from fitting because their reference routes are displaced from their own GPS traces.

## Reproduce

See the root `README.md` ("Reproduce the experiments"). `reports/optuna_rf.db` (the saved Optuna study) is not committed; `optuna_trials.csv` holds the same trials.

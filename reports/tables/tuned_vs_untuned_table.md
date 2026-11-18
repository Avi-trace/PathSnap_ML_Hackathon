# Tuned vs untuned Random Forest (report section 4)

Tuning: Optuna TPE, 9 trials (5 random start-up, 4 sampler-guided), 1236s total tuning time. Objective: mean 3-fold group-CV Top-1 (partition: train). Student-test rows are the final refit on all 78 train trips, scored on the 10 student_test trips.

| Metric | Untuned | Tuned |
| --- | --- | --- |
| CV Top-1 | 0.8658 | 0.8684 |
| Student Top-1 | 0.8730 | 0.8720 |
| Student macro-F1 | 0.9300 | 0.9295 |
| Student log loss | 0.0345 | 0.0342 |
| Inference time (s/100k rows) | 0.1090 | 0.1533 |
| Fit time, full refit (s) | 190.3 | 155.7 |
| Tuning cost (trials / seconds) | n/a (default params) | 9 trials / 1236s |

Best params (tuned): {"n_estimators": 65, "max_depth": 17, "min_samples_leaf": 10, "max_features": 0.32789310482836637, "max_samples": 0.6420380135006861}

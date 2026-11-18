# Map and diagnostics index

All files come from real runs of the final bundle (`artifacts/final/bundle.joblib`: tuned Random Forest + tuned decoder) on the `student_test` partition through `predict_trip` + `score_prediction`. The reference route layer is offline-report only; the evaluator UI never shows it.

## Interactive maps (Folium, OSM tiles; script `scripts/10_folium_prediction_maps.py`, helpers `scripts/_diag_common.py`)
Layers (toggle in LayerControl): 1 GPS (blue), 2 road network = candidate arcs de-duplicated across directions (grey), 3 reference route (green, offline only), 4 classifier-only route (orange dashed + red break markers), 5 HMM/Viterbi route (purple, start/end markers). Title box carries the per-trip metrics from the run.

| File | Trip | Shows |
|---|---|---|
| pred_trip_00000052.html | 00000052 | small trip, HMM helps |
| pred_trip_00000009.html | 00000009 | HMM clearly helps (many classifier breaks removed) |
| pred_trip_00000040.html | 00000040 | HMM clearly helps |
| pred_trip_00000007.html | 00000007 | HMM made pointwise accuracy worse |
| pred_trip_00000077.html | 00000077 | hard for both |
| pred_trip_<id>_zoom.html | same | same layers, view fitted to the best 80-observation window for that trip's disagreement type (00000007: HMM wrong and classifier right; 00000077: both wrong; 00000052/009/040: classifier wrong and HMM right) |

## Screenshots (`reports/screenshots/`, headless Chrome 1400x900, tiles rendered)
map_pred_<id>.png (full map) and map_pred_<id>_zoom.png (zoom html) for the five trips.

## Diagnostics (script `scripts/10b_hmm_diagnostics.py`; definitions in its docstring)
- `reports/hmm_diagnostics.csv`: per student trip (10), counts of eligible observations: both right / classifier right HMM wrong / classifier wrong HMM right / both wrong, plus pointwise accuracies.
- `reports/hmm_diagnostics_00000007.csv`, `reports/hmm_diagnostics_00000077.csv`: contiguous observation-index ranges (kind `hmm_wrong_clf_right` or `clf_wrong_hmm_right`) with time range, point count, median GPS-to-chosen-arc distance for classifier and HMM, and HMM-arc vs reference-arc connectivity (same arc or reachable either way within 3 directed hops via `metrics.transition_flags`).

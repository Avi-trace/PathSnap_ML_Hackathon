# PathSnap: map-matching noisy GPS to a road network

A candidate-arc classifier plus an HMM/Viterbi decoder reconstructs the road route of a vehicle from raw 1 Hz GPS.
Two laptops: **Laptop 1** runs the model + FastAPI backend, **Laptop 2** runs the evaluator UI.

```
raw trip files -> candidate arcs (<= 75 m) -> features -> classifier -> P(arc | observation)
              -> HMM / Viterbi (emission = classifier probability, transition = GPS vs network distance) -> route
```

## Repository layout

| Path | What it is |
|---|---|
| `pathsnap/io.py, geo.py` | file readers; per-trip UTM projection (metres) and point-to-segment geometry |
| `pathsnap/candidates.py` | exact radius search for candidate arcs (arcs are split into pieces so long arcs are not missed) |
| `pathsnap/labels.py` | per-observation reference labels: monotone dynamic-programming alignment to the `.route` file |
| `pathsnap/features.py` | 18 per-candidate features (same code for training and the API) |
| `pathsnap/metrics.py` | observation-level Top-1, Macro-F1, log loss, route continuity |
| `pathsnap/hmm.py` | HMM hidden states, emissions, transitions, Viterbi, gap handling |
| `pathsnap/pipeline.py` | raw files in, route out, plus server-side scoring |
| `pathsnap/api.py` | FastAPI backend |
| `scripts/01_eda_inventory.py` | dataset inventory, data-quality checks, candidate-radius coverage |
| `scripts/02_build_dataset.py` | labelled candidate tables for every trip |
| `scripts/03_train_models.py` | 9 models, 4 families, group CV + student-test comparison |
| `scripts/04_tune_random_forest.py` | Bayesian (Optuna TPE) tuning of the selected model, tuned vs untuned |
| `scripts/05_pipeline_check.py` | end-to-end check: raw files through the API code path |
| `scripts/06_export_bundle.py` | packages classifier + decoder parameters into `artifacts/final/bundle.joblib` |
| `scripts/07_tune_decoder.py` | Bayesian tuning of the HMM parameters; classifier-only vs classifier+HMM |
| `scripts/08_eda_figures.py` | EDA figures and tables, plus the Folium map of a raw trip |
| `scripts/09_model_figures.py` | model-comparison table, bar charts, confusion matrices, MLP loss curve, Optuna plots |
| `scripts/10_folium_prediction_maps.py`, `10b_hmm_diagnostics.py` | Folium maps of the predictions (GPS, network, reference, classifier-only, HMM) and per-trip HMM diagnostics |
| `scripts/11_train_accuracy.py`, `12_boosting_curves.py` | training-vs-CV-vs-student accuracy (over-fitting evidence) and LightGBM/XGBoost per-round loss curves |
| `RUNBOOK.md` | demo-day checklist for the two-laptop evaluation |
| `reports/` | all result tables (CSV/JSON) and experiment logs behind the technical report |
| `web/` | evaluator UI (Vite + React + TypeScript) |
| `client_handoff/` | spec, mock server and recorded sample responses for the client-side teammate |
| `team_packages/` | per-teammate briefs (what each piece does and how to explain it) and the script that zips them |

## Setup

```
conda create -n pathsnap python=3.11 && conda activate pathsnap
pip install -r requirements.txt
```

## Reproduce the experiments (train + student_test only; the evaluator partition is never read)

```
python scripts/01_eda_inventory.py  --data-root "<.../Student folder/Student folder>"
python scripts/02_build_dataset.py  --data-root "<.../Student folder/Student folder>"
python scripts/03_train_models.py
python scripts/04_tune_random_forest.py
python scripts/07_tune_decoder.py
python scripts/06_export_bundle.py --model artifacts/models/RandomForest_tuned.joblib --hmm reports/hmm_best.json --name "RandomForest (Optuna-tuned) + HMM"
python scripts/05_pipeline_check.py                      # raw files -> API code path must reproduce the training-time Top-1
# report figures, maps and evidence (outputs in reports/figures, reports/maps, reports/tables, reports/screenshots)
python scripts/08_eda_figures.py && python scripts/09_model_figures.py && python scripts/11_train_accuracy.py && python scripts/12_boosting_curves.py
python scripts/10_folium_prediction_maps.py && python scripts/10b_hmm_diagnostics.py
```

All random seeds are fixed (seed 0) and every configuration is written to `reports/model_configs.json`.

## Run the system (final evaluation)

A step-by-step demo-day checklist is in **`RUNBOOK.md`**. In short:

**Laptop 1 (API).** Point `PATHSNAP_TRIPS_DIR` at the folder holding the ten evaluation trips
(`<trip>/<trip>.track|nodes|arcs|route`; the `.route` files stay on this machine and are only used for scoring).

```
set PATHSNAP_TRIPS_DIR=<folder with the evaluation trips>
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP 8000 in the Windows firewall for the network profile in use (a Wi-Fi marked "Public" needs a Public/any-profile rule; see `RUNBOOK.md`).
The console shows one log line per request.

**Laptop 2 (UI).**

```
cd web && npm install && npm run dev        # or: npm run build && npm run preview
```

Open `http://<laptop-2>:5173/?api=http://<laptop-1-ip>:8000`.

API: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. The response contains predictions, road geometry
and aggregate scores only; the reference route is never sent to the UI.

## Notes

- Two training trips (`00000033`, `00000054`) have reference routes that sit 70-125 m from their own GPS traces with a
  wide spread, so their labels are unusable. They are excluded from fitting and kept in the EDA tables.
- The submitted UI contains no bundled or mock results: every number comes from the live API. For UI development without the server,
  `client_handoff/mock_server.py` serves recorded sample responses and flags itself as a mock; the UI shows a red banner when it sees that flag.

## AI assistance disclosure

Parts of this code base were written with Claude Code (Anthropic) under the team's direction. The team made the design
decisions (model families, selection metric, HMM design, label construction, features) and is responsible for the
results. *Edit this section to match the event's disclosure policy.*

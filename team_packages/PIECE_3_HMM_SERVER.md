# Piece 3: HMM / Viterbi, pipeline & server (report sections 5, 6 and 7 for the API)

## Your part
Own the sequence decoder, the end-to-end inference pipeline, the FastAPI backend (Laptop 1), the geographic (Folium) visualisations of the predictions, and the deployment test.
You answer: *What are the hidden states? How are emissions and transitions computed? How are gaps/missing candidates/disconnected segments handled? Did the HMM help?*

## What is already built (read these files)
| File | What it does |
|---|---|
| `pathsnap/hmm.py` | HMM states, emission, transition, Viterbi, gap handling |
| `pathsnap/pipeline.py` | raw trip files -> candidates -> classifier -> HMM -> route; server-side scoring; JSON for the UI |
| `pathsnap/api.py` | FastAPI: `GET /health`, `GET /trips`, `POST /predict` |
| `scripts/05_pipeline_check.py` | runs the whole pipeline on the 10 student trips and checks it against the training-time numbers |
| `scripts/06_export_bundle.py` | packs classifier + decoder parameters into `artifacts/final/bundle.joblib` (included in your zip) |
| `scripts/07_tune_decoder.py` | Bayesian tuning of the decoder parameters + classifier-vs-HMM comparison tables |

## Explain it (plain language)
**Why an HMM.** The classifier scores every GPS point on its own, so consecutive picks can be on roads that are not connected (a "jump"). A hidden Markov model adds the rule that the vehicle moves along connected roads, and Viterbi finds the single most likely route.

**Hidden state.** For each GPS observation, one state per candidate arc (one state = one candidate row), so every state keeps its link to a real road segment.

**Emission** (how well a state explains the observation): the classifier probability, normalised over that observation's candidates, then raised to a power `alpha`: `log e = alpha * log(p / sum p)`.

**Transition** (how plausible moving from candidate i to candidate j): Newson-Krumm. Compare the straight-line distance between the two GPS points `d_gps` with the distance along the road network between the two candidate positions `d_route`
(forward-only along directed arcs, shortest path by bounded Dijkstra on a local graph around the trace; on the same arc it is the position difference, with 15 m tolerance for GPS jitter).
`log T = -|d_gps - d_route| / beta`. A vehicle that moved 20 m should be on candidates about 20 m apart *along the road*.

**Viterbi recurrence.** `delta_1(j) = emit_1(j)`; `delta_{k+1}(j) = emit_{k+1}(j) + max_i [ delta_k(i) + logT_k(i, j) ]`; store the best `i`; at the end take the best final state and backtrack.

**Edge cases.** *No candidates* for a point: the point is dropped from the chain and reported unmatched. *Unreachable pair* (no path within the search limit `3*d_gps + 300 m`): a finite penalty `invalid_pen` (default 20 in log units), never minus infinity, so decoding can always continue.
*GPS gap* (more than 15 s or 500 m between consecutive points): transitions become uniform, i.e. the chain restarts from the emissions alone. *Disconnected segments chosen by the classifier*: this is exactly what the HMM penalises; the output route bridges any remaining break with the shortest path.

**Parameters** (defaults): `alpha=1`, `beta=10 m`, `invalid_pen=20`, `gap_s=15`, `reset_d=500`. `07_tune_decoder.py` tunes `alpha`, `beta`, `invalid_pen` with Optuna (TPE, 60 trials) on 30 random train trips using **out-of-fold** classifier probabilities, then evaluates on student-test.
**Tuned values:** `alpha=0.975`, `beta=2.97 m`, `invalid_pen=39.1` (sample Top-1 86.65 % classifier-only, 91.88 % default HMM, 92.33 % tuned HMM). The small `beta` means the decoder trusts the GPS-vs-road-distance agreement strongly.

**Results** (tuned Random Forest, 10 student trips, raw files through the real pipeline, observation-weighted) in `reports/classifier_vs_hmm.csv` and `classifier_vs_hmm_per_trip.csv`:
| | classifier only | HMM default decoder | HMM tuned decoder |
|---|---|---|---|
| Pointwise accuracy | 87.20 % | 90.82 % | 91.39 % |
| Route continuity | 89.17 % | 99.07 % | 99.16 % |
| Route F1 | 87.93 % | 92.97 % | 93.54 % |

The tuned HMM improved nine of ten trips and made trip `00000007` worse (92.54 % -> 90.10 %); `00000077` is hard for both. Analyse why for the report ("cases where the HMM did not help"). A sanity check: classifier-only Top-1 recomputed from raw files equals the training-time figure exactly.

**API design.** `POST /predict {"trip_id"}` runs the full pipeline and returns the predicted routes, road geometry and aggregate scores; **the reference route is read only after predicting, only to score, and is never in a response**. Trip ids are checked against an allow-list (no path traversal);
CORS is limited to localhost and private LAN addresses; every request prints a log line (the brief requires showing the receiving laptop's log). Nothing is cached or hardcoded. Typical 0.3 to 3.5 s per trip (mean 1.2 s).

## Your deliverables
1. The **final bundle** (tuned Random Forest + tuned decoder) is already in `artifacts/final/`. To reproduce it you need the outputs of `04_tune_random_forest.py` (tuned model and out-of-fold file, about 25 minutes), then `python scripts/07_tune_decoder.py` and `06_export_bundle.py --model artifacts/models/RandomForest_tuned.joblib --hmm reports/hmm_best.json`. Review the two comparison CSVs above.
2. **Folium maps (mandatory, section 6)**: write `scripts/10_folium_prediction_maps.py` producing interactive `reports/maps/pred_trip_<id>.html` for at least two student trips with layers: original GPS, road network, reference route (offline report only; the UI must never show it), classifier-only route, HMM route; distinct colours, legend, layer control, good zoom. Include one trip where the HMM helped and `00000007` where it did not. Save screenshots for the PDF.
3. Written answers to the six questions of report section 5 plus the classifier-only vs classifier+HMM comparison (pointwise accuracy, sequence-level F1, continuity) and a discussion of failures.
4. **API and deployment evidence (section 7):** screenshots of successful responses **in the console of the receiving laptop**; the two-laptop test (below).

**Run the server (Laptop 1):**
```
set PATHSNAP_TRIPS_DIR=<folder with the ten evaluation trips: <id>\<id>.track|nodes|arcs|route>
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```
Find the LAN address with `ipconfig`; allow inbound TCP 8000 for the private network (admin terminal: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000 profile=private`).
From the client laptop: `curl http://<ip>:8000/health`. Default trips folder is the `student_test` partition (development only). More in `client_handoff/CLIENT_README.md`, section 10.

## Likely viva questions
What are the hidden states and why one per candidate? Write the Viterbi recurrence. What do `alpha` and `beta` do? What happens with a GPS gap, an unreachable pair, no candidates? Why out-of-fold probabilities for tuning?
Why is ground truth never sent to the UI? How do you prevent path traversal in `/predict`? Where did the HMM make things worse and why?

## Hand back
New scripts, maps, screenshots, tables and your written section.

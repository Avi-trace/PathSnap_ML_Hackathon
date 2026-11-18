# PathSnap: progress report

*Status at the first push to GitHub (hackathon day, afternoon). Every number below is read from `reports/` and reproducible with the scripts in `scripts/`.
Results on the 10 `student_test` trips are development results, not the final evaluation.*

## 1. Problem and approach

Noisy 1 Hz GPS is matched to a road network to recover the route. The system is a **candidate-arc classifier** (scores each
nearby directed road segment for each GPS point) followed by an **HMM / Viterbi decoder** that enforces a connected route.

```
GPS point -> candidate arcs (<=75 m) -> 18 features -> classifier P(arc) -> HMM/Viterbi -> route
```

## 2. Done

**Data understanding (EDA)** `scripts/01_eda_inventory.py`
- 90 trips (80 train, 10 student-test), 227,577 GPS observations, trips from 25 different UTM zones, so each trip is projected to its own metric zone.
- Quality checks: no NaN or implausible coordinates; 96.7 % of all steps are exactly 1 s apart (the per-trip share averages 94.3 %) but 47 trips contain a gap > 5 s (largest 153,064 s, a single gap inside trip `00000090`) and 1,538 steps imply > 50 m/s;
  networks have 24 self-loops, 1,903 duplicate arcs and almost all have several disconnected components; reference routes are valid and connected (head-to-tail share 99.94 %).
- Finding: two training trips (`00000033`, `00000054`, 12 % of train observations) have reference routes 70-125 m away from their own GPS trace
  with a wide spread, so their labels are unusable. They are excluded from fitting and documented.

**Labels and candidates** `labels.py`, `candidates.py`, `features.py`
- Only the ordered route is given, not per-point labels, so labels come from a monotone dynamic-programming alignment of GPS points to the route.
- Candidate radius 75 m: candidate miss rate **0.45 %** (train), **0.09 %** (student-test), 15.8 candidates per observation, 2.88 M training rows.
- A "nearest arc" baseline gets only **55.7 %** Top-1 (every road has two directed arcs at the same distance), so direction features matter.

**9 models, 4 algorithm families, 3-fold group CV by trip** `scripts/03_train_models.py` (selection metric agreed beforehand: observation-level Top-1)

| Model (family) | CV Top-1 | Student Top-1 | Student macro-F1 | Student log loss | Fit time |
|---|---|---|---|---|---|
| **Random Forest** (bagging) | **0.8658** | **0.8730** | **0.9300** | **0.0345** | 104 s |
| XGBoost (boosting) | 0.8635 | 0.8707 | 0.9280 | 0.0360 | 25 s |
| MLP, PyTorch (neural) | 0.8634 | 0.8680 | 0.9227 | 0.0385 | 43 s |
| Extra Trees (bagging) | 0.8615 | 0.8612 | 0.9175 | 0.0468 | 29 s |
| Decision Tree (tree) | 0.8593 | 0.8712 | 0.9276 | 0.0394 | 75 s |
| LightGBM (boosting) | 0.8582 | 0.8585 | 0.9248 | 0.0506 | 17 s |
| HistGradientBoosting (boosting) | 0.8575 | 0.8643 | 0.9276 | 0.0407 | 36 s |
| Logistic Regression (linear) | 0.8545 | 0.8556 | 0.9056 | 0.0524 | 12 s |
| Gaussian NB (probabilistic) | 0.7989 | 0.7901 | 0.8163 | 0.3551 | 0.7 s |

Honest reading: the top seven models are within about one fold-standard-deviation (about 0.03) of each other. Random Forest ranks first on the agreed metric and also has the best log loss
(probability quality matters because the HMM consumes the probabilities). Gaussian NB is clearly worst (its independence assumption fails on correlated features).

**HMM / Viterbi decoder** `hmm.py`: hidden state = candidate arc; emission = classifier probability; transition = Newson-Krumm
(GPS distance vs network distance, bounded Dijkstra); GPS gaps reset the chain; unreachable pairs get a finite penalty.
Decoder parameters (emission power `alpha`, transition scale `beta`, unreachable-pair penalty) were tuned with Optuna (TPE, 60 trials) on 30 random **train** trips using
out-of-fold classifier probabilities: sample Top-1 86.65 % (classifier only) -> 91.88 % (HMM, default parameters) -> 92.33 % (HMM, tuned: alpha 0.97, beta 2.97 m, penalty 39).
End-to-end result on the 10 student trips (tuned Random Forest, raw files through the real pipeline, observation-weighted):

| | classifier only | HMM, default decoder | HMM, tuned decoder |
|---|---|---|---|
| Pointwise accuracy | 87.20 % | 90.82 % | **91.39 %** |
| Route continuity | 89.17 % | 99.07 % | **99.16 %** |
| Route F1 | 87.93 % | 92.97 % | **93.54 %** |

The HMM improved nine of ten trips; it made trip `00000007` worse (92.54 % to 90.10 %), which will be analysed in the report. Consistency check: the classifier-only
Top-1 recomputed from raw files equals the training-time figure exactly (0.8730 for the untuned forest).

**Backend** `pathsnap/api.py` (FastAPI): `GET /health`, `GET /trips`, `POST /predict`. The reference route stays server-side and is used only to compute aggregate
scores; it is never in a response. Trip ids are allow-listed, CORS is restricted to localhost / private LAN, every request prints a log line. Mean 1.2 s per trip (max 3.2 s).

**Front end** `web/` (React + TypeScript + Vite): boot screen, 600vh scroll-driven intro (5 stages, driven by a live API response), trip picker for the 10 trips, offline canvas map
(no tiles) with layer toggles, results panel (classifier vs HMM), request log, error handling, WebGL post-processing with automatic fallback. All 10 trips were run in a browser against the live API during development (accuracy, continuity and jump counts matched the API response); four of them were re-checked independently.

**Client teammate hand-off** `client_handoff/`: a self-contained spec, a mock server and three recorded sample responses so the client laptop can be built independently.

## 3. Bayesian hyper-parameter optimisation (done)

Optuna TPE on the selected Random Forest, objective = mean 3-fold group-CV Top-1, 9 trials (5 random start-up, then 4 guided; limited by a 20-minute budget).
Best CV Top-1 **0.8684** against 0.8658 untuned (+0.26 points). On student-test the tuned forest is a **tie, not a win**: Top-1 0.8720 vs 0.8730, macro-F1 0.9295 vs 0.9300, log loss 0.0342 vs 0.0345.
Every trial landed between 0.8650 and 0.8684, so the model is not sensitive to these settings and the gain is within noise. The final inference bundle (`artifacts/final/bundle.joblib`) is the tuned forest plus the tuned decoder.

## 4. Report assets, front end and deployment (done)

- **Report assets** (`reports/`, indexed in `reports/README.md`): 7 EDA figures + 4 EDA tables + an interactive Folium map of a raw trip; model comparison table, 9 model figures (CV and student-test scores, log loss, confusion matrices, MLP loss curve, LightGBM/XGBoost per-round curves, Optuna plots);
  Folium prediction maps for 5 student trips (GPS, road network, reference route, classifier-only route, HMM route, layer control) with screenshots; HMM diagnostics (where the HMM helps or hurts).
- **Over-fitting evidence:** train-minus-CV Top-1 gap is about 0.003 for Gaussian NB and Logistic Regression and 0.010 to 0.037 for the tree, forest, neural and boosting models, and **0.073 for XGBoost** (train 0.937 vs CV 0.864);
  validation loss of both boosters stops improving after roughly 40-60 of 300 rounds.
- **Front end:** mock data and the `?mock=1` mode are removed (a red banner appears if a server says it is the dev mock); scores show a dash instead of NaN when the server sends none; web README written; five UI screenshots with the final model.
- **Deployment:** API restarted on the final bundle and reached over the machine's LAN address with the client IP in the server log; inbound firewall rule for this Python exists on the Public profile; `RUNBOOK.md` written (demo-day checklist, troubleshooting).
- **Team hand-off:** four per-teammate briefs with explanations and viva questions (`team_packages/`) and a client spec with mock server for the front-end teammate (`client_handoff/`).

## 5. Still to do

1. **Two-laptop test on the real second laptop** with the real evaluation folder. The API was verified over its own LAN address on one machine; whether the venue network lets laptops reach each other (client isolation) is untested.
2. **Screenshot of the API server console** during the demo (report section 7 asks for the receiving laptop's console).
3. **The written technical report (PDF):** the team's own analysis and explanations; each member must be able to explain their piece. The tables, figures and maps are ready.
4. **AI-assistance disclosure** text to match the event policy (placeholder in `README.md`).
5. Possible improvements if time allows: a larger Bayesian search budget, early stopping for the boosted models, per-trip analysis of why trip `00000077` is hard.

## 6. Risks

- The evaluation trips are unseen; two training trips had unusable reference routes, and similar issues could occur there.
- WebGL effects were verified on a CPU-rendered browser only (an automatic fallback exists); smoothness on the demo GPU is unverified.
- All student-test numbers are development results. The decoder was tuned only on train trips (with out-of-fold probabilities); the evaluator partition was never read.

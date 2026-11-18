# PathSnap: team package (read this first)

The project is split into **four pieces**, one per teammate. Each piece matches sections of the required technical report,
so the person who owns it is also the person who must explain it in the viva.

| Piece | Owner (fill in) | Report sections | What you own |
|---|---|---|---|
| **1. Data & Features** | | 2.2 EDA | raw-data inspection, projection, labels, candidate generation, 18 features |
| **2. Models & Bayesian tuning** | | 3 and 4 | 9-model comparison, metrics, Optuna tuning |
| **3. HMM / Viterbi, pipeline & server** | | 5, 6 and 7 (API) | decoder, end-to-end pipeline, FastAPI backend, Folium maps, deployment |
| **4. Client UI** | | 7 (UI) | the evaluator-facing web interface on Laptop 2 |

Open **`START_HERE.md`** in your zip: it is your piece's brief (what is built, how to explain it, what you still have to deliver).

## How the pieces fit

```
raw files ─► [1] candidates + labels + features ─► [2] classifier (trained, compared, tuned)
                                                        │  P(arc | GPS point)
                                                        ▼
                         [3] HMM/Viterbi ─► route + scores ─► FastAPI ─► [4] web UI
```

## Setup (pieces 1, 2, 3)

```
conda create -n pathsnap python=3.11 && conda activate pathsnap
pip install -r requirements.txt
set PATHSNAP_DATA_ROOT=<folder that contains train\ and student_test\>      (PowerShell: $env:PATHSNAP_DATA_ROOT="...")
python scripts/02_build_dataset.py --data-root "%PATHSNAP_DATA_ROOT%"        (about 1 minute; needed by pieces 2 and 3)
```
The dataset is not in the zip: use your own copy. Training everything (`03_train_models.py`) takes about 10 minutes and about 4-6 GB of RAM.

## Rules that apply to everyone (from the hackathon brief)

1. **Never read, copy or guess the evaluator partition.** Development code only uses `train` and `student_test`.
2. **Ground truth never reaches the UI or the model's inputs.** The reference route is used only to build training labels and to score on the server.
3. **No hardcoded results.** Every number in the report must come from a script run you can repeat; reports are checked against the repo.
4. **Independent work.** Code and reports are compared across teams: do not share these files outside the team.
5. **You must be able to explain your piece**, line by line, in the viva. Read your `START_HERE.md` "Explain it" section until you could say it without notes.
6. **AI-assistance disclosure.** Parts of this code were written with Claude Code. Disclose it as the event requires (the project README has a placeholder) and write your own report text from your own understanding of your results.
7. No secrets in the repository; never commit the dataset.

## Numbers you will see

All numbers in `PROGRESS.md` and in `reports/` are **development results** on the 10 `student_test` trips, produced by the final bundle (tuned Random Forest + tuned decoder).
The real evaluation uses a separate hidden partition, so expect the final numbers to differ. If the team lead re-runs anything, ask for the refreshed `reports/` folder before you write numbers into the report.

## Hand-back

Send back **only your deliverables** (listed in your `START_HERE.md`): new scripts, figures, maps, tables, notes. Do not send the whole zip back,
the dataset, or large model files. Put new scripts in `scripts/` with the next free number so they merge cleanly.

## Glossary

- **Arc**: a directed road segment (two nodes). A two-way road is two arcs.
- **Candidate**: an arc within 75 m of a GPS point. The classifier scores each candidate.
- **Reference route**: the ordered list of arcs the vehicle really took (from the `.route` file); an evaluation reference, not perfect ground truth.
- **Top-1 (observation level)**: share of GPS points whose highest-scoring candidate is the reference arc.
- **Route continuity**: share of jumps between consecutive different predicted arcs that are connected (reachable within 3 arcs).

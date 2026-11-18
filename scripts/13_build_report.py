"""Build the technical report (HTML + PDF) from the real result files in reports/.

usage: python scripts/13_build_report.py
Writes report/PathSnap_Technical_Report.html (self-contained: images are embedded, so it opens in Word or any browser and can be edited)
and report/PathSnap_Technical_Report.pdf (printed with headless Chrome).

Every table is read from reports/*.csv|json. Every number quoted in the text is filled in from the same files (or recomputed here from the
cached tables / the final bundle), never typed in by hand. Figures and screenshots are the files in reports/figures, reports/maps and
reports/screenshots (made by scripts 01-12).
"""
from __future__ import annotations

import base64
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "reports"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
REPO_URL = "https://github.com/Avi-trace/PathSnap_ML_Hackathon"
BAD_TRIPS = ["00000033", "00000054"]

# ------------------------------------------------------------------ data
cmp_ = pd.read_csv(R / "model_comparison.csv").set_index("model")
folds = pd.read_csv(R / "cv_fold_results.csv")
tv = pd.read_csv(R / "train_vs_cv_vs_student.csv").set_index("model")
tuned = pd.read_csv(R / "tuned_vs_untuned.csv").set_index("variant")
ob = json.load(open(R / "optuna_best.json"))
trials = pd.read_csv(R / "optuna_trials.csv")
hb = json.load(open(R / "hmm_best.json"))
hmm_agg = pd.read_csv(R / "classifier_vs_hmm.csv").set_index("decoder")
hmm_trip = pd.read_csv(R / "classifier_vs_hmm_per_trip.csv", dtype={"trip": str})
diag = pd.read_csv(R / "hmm_diagnostics.csv", dtype={"trip": str})
cfg = json.load(open(R / "model_configs.json"))
eda = pd.read_csv(R / "eda_trip_summary.csv", dtype={"trip_id": str})
T = {n: pd.read_csv(R / "tables" / f"eda_{n}.csv") for n in ("inventory", "missing_invalid", "network_connectivity", "data_quality_log")}
MODELS = list(cmp_.index)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def fig(path: Path, caption: str, width: str = "100%") -> str:
    return f'<figure><img src="data:image/png;base64,{b64(path)}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def pair(items: list[tuple[Path, str]], width: str = "100%") -> str:
    return '<div class="grid2">' + "".join(fig(p, c, width) for p, c in items) + "</div>"


def tbl(df: pd.DataFrame, caption: str = "", cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    cap = f"<div class='tcap'>{caption}</div>" if caption else ""
    return f"{cap}<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


f3, f4 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")
pc = lambda v, d=1: f"{100 * v:.{d}f}%"

# ------------------------------------------------------------------ numbers recomputed here so the text is verifiable
# 1) dataset summary from the cached candidate tables (artifacts/data is local; the result is stored in reports/dataset_summary.json)
DS = R / "dataset_summary.json"
if (ROOT / "artifacts" / "data" / "train").exists():
    import joblib
    ds = {}
    for part in ("train", "student_test"):
        rows = elig = miss = pos = near = 0
        n_trips = 0
        for f in sorted((ROOT / "artifacts" / "data" / part).glob("*.joblib")):
            tab = joblib.load(f)
            if part == "train" and tab["trip_id"] in BAD_TRIPS:
                continue
            r_, o_ = tab["rows"], tab["obs"]
            n_trips += 1
            rows += len(r_)
            elig += int(o_.eligible.sum())
            miss += int((o_.eligible & ~o_.ref_in_cands).sum())
            pos += int(r_["label"].sum())
            near += int(r_.loc[r_["dist_rank"] == 0, "label"].sum())
        ds[part] = dict(trips=n_trips, eligible_obs=elig, rows=rows, rows_per_obs=rows / elig, miss_rate=miss / elig, positive_rate=pos / rows, nearest_arc_top1=near / elig)
    json.dump(ds, open(DS, "w"), indent=2)
DSUM = json.load(open(DS))

# 2) candidate-radius coverage (clean trips only), observation weighted
clean = eda[~eda.trip_id.isin(BAD_TRIPS)]
w = clean.n_gps
cover = pd.DataFrame({"Radius (m)": [25, 50, 75, 100, 150],
                      "Reference arc missed": [pc(np.average(clean[f"miss_R{r}"], weights=w), 2) for r in (25, 50, 75, 100, 150)],
                      "Candidates per GPS point": [f"{clean[f'cand_rows_R{r}'].sum() / w.sum():.1f}" for r in (25, 50, 75, 100, 150)]})
dt_pool = float((eda.dt_eq_1s * (eda.n_gps - 1)).sum() / (eda.n_gps - 1).sum())
n_gps_all = int(eda.n_gps.sum())
n_zones = int(eda.epsg.nunique())

# 3) final-bundle latency per student trip (the real pipeline, run now)
LAT = R / "latency_final_bundle.csv"
try:
    from pathsnap.io import load_trip
    from pathsnap.paths import DATA_ROOT
    from pathsnap.pipeline import load_bundle, predict_trip, score_prediction
    bundle = load_bundle(ROOT / "artifacts" / "final" / "bundle.joblib")
    rows_ = []
    for tid in sorted(hmm_trip.trip.unique()):
        trip = load_trip(DATA_ROOT, "student_test", tid)
        t0 = time.perf_counter()
        pred = predict_trip(bundle, trip)
        pipe_ms = (time.perf_counter() - t0) * 1e3
        m = score_prediction(pred, trip)
        rows_.append(dict(trip=tid, gps_points=trip.n_gps, candidates_ms=pred.timings_ms["candidates"], classifier_ms=pred.timings_ms["classifier"],
                          hmm_ms=pred.timings_ms["hmm"], pipeline_ms=pipe_ms, acc_clf=m["pointwise_acc_classifier"], acc_hmm=m["pointwise_acc_hmm"]))
    pd.DataFrame(rows_).to_csv(LAT, index=False)
except Exception as e:                                                      # noqa: BLE001
    print("latency run skipped:", type(e).__name__, e)
lat = pd.read_csv(LAT, dtype={"trip": str}) if LAT.exists() else None

# 4) facts about trip 00000007 (where the HMM hurts): where the errors are and how fast the vehicle moves there
t7 = {}
try:
    from pathsnap.geo import Projector
    d7 = pd.read_csv(R / "hmm_diagnostics_00000007.csv", comment="#", dtype={"trip": str})
    bad = d7[d7.kind == "hmm_wrong_clf_right"]
    tr = load_trip(DATA_ROOT, "student_test", "00000007", with_route=False)
    G = Projector(tr.lon.mean(), tr.lat.mean()).xy(tr.lon, tr.lat)
    step = np.r_[0, np.linalg.norm(np.diff(G, axis=0), axis=1)] / np.r_[1, np.diff(tr.t)]
    mask = np.zeros(len(G), bool)
    for s, e in zip(bad.obs_start, bad.obs_end):
        mask[s:e + 1] = True
    t7 = dict(n_ranges=len(bad), n_points=int(bad.n_points.sum()), first_obs=int(bad.obs_start.min()), n_obs=len(G), max_len=int(bad.n_points.max()),
              speed_bad=float(np.median(step[mask])), speed_other=float(np.median(step[~mask])), slow_share=float((step[mask] < 1).mean()))
except Exception as e:                                                      # noqa: BLE001
    print("trip 7 facts skipped:", type(e).__name__, e)

# 5) extra figure: classifier-only vs HMM accuracy per student trip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ct = hmm_trip[hmm_trip.decoder == "tuned decoder"].sort_values("trip")
FIG_HMM = R / "figures" / "report_hmm_per_trip.png"
fig_, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(ct))
ax.bar(x - 0.2, 100 * ct.pointwise_acc_classifier, 0.4, label="classifier only", color="#9aa5ad")
ax.bar(x + 0.2, 100 * ct.pointwise_acc_hmm, 0.4, label="classifier + HMM (tuned)", color="#e69a1f")
ax.set_xticks(x, ct.trip, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("pointwise accuracy (%)")
ax.set_ylim(60, 100)
ax.set_title("Pointwise accuracy per trip: classifier only vs classifier + HMM (partition: student_test, 10 trips)", fontsize=9)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig_.tight_layout()
fig_.savefig(FIG_HMM, dpi=170)
plt.close(fig_)

# 6) extras used by the text: family count, code version, tuned-vs-untuned CV table, feature importances, software list
n_fam = int(cmp_.family.nunique())
try:
    GIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception:                                                           # noqa: BLE001
    GIT = ""
tcv = pd.read_csv(R / "tuned_vs_untuned_cv.csv")                              # made by scripts/04b_tuned_cv_macro_f1.py
FI = None
try:
    import joblib as _jl
    from pathsnap.features import FEATURES as _FEATS
    _rf = _jl.load(ROOT / "artifacts" / "models" / "RandomForest_tuned.joblib")
    FI = pd.Series(_rf.feature_importances_, index=_FEATS).sort_values(ascending=False)
except Exception as e:                                                      # noqa: BLE001
    print("feature importance skipped:", type(e).__name__, e)
BUNDLE_MB = (ROOT / "artifacts" / "final" / "bundle.joblib").stat().st_size / 1e6
_req = [ln.split("=")[0].split(">")[0].strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
_pkg = json.load(open(ROOT / "web" / "package.json"))
SOFTWARE_PY = ", ".join(_req)
SOFTWARE_JS = ", ".join(list(_pkg.get("dependencies", {})))

exec(open(ROOT / "scripts" / "13b_report_text.py", encoding="utf-8").read())   # the narrative and page assembly (kept in a second file for length)

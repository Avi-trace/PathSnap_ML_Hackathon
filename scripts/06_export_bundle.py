"""Package the final classifier + decoder parameters into one inference bundle used by the API.

usage: python scripts/06_export_bundle.py [--model artifacts/models/RandomForest_tuned.joblib]
                                          [--hmm reports/hmm_best.json] [--name "..."]
Without --hmm the default decoder parameters are used (dev bundle).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES, RADIUS                       # noqa: E402
from pathsnap.pipeline import DEFAULT_HMM, Bundle, save_bundle        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "models" / "RandomForest.joblib"))
    ap.add_argument("--hmm", default=None, help="JSON file with the decoder parameters")
    ap.add_argument("--name", default="RandomForest (untuned, dev bundle)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "final" / "bundle.joblib"))
    a = ap.parse_args()
    hmm = dict(DEFAULT_HMM)
    if a.hmm:
        hmm.update({k: v for k, v in json.load(open(a.hmm)).items() if k in DEFAULT_HMM})
    meta = {"name": a.name, "source_model": Path(a.model).name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": 0,
            "excluded_train_trips": ["00000033", "00000054"], "candidate_radius_m": RADIUS}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(Bundle(model=joblib.load(a.model), features=list(FEATURES), radius=RADIUS, hmm=hmm, meta=meta), a.out)
    json.dump({"meta": meta, "features": FEATURES, "hmm": hmm}, open(Path(a.out).with_suffix(".json"), "w"), indent=2)
    print("saved", a.out, f"({Path(a.out).stat().st_size / 1e6:.1f} MB)", "| decoder:", hmm)


if __name__ == "__main__":
    main()

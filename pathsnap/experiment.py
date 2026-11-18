"""Shared experiment plumbing: load the cached candidate tables, build the fixed CV folds, score predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import clf_metrics, obs_keys, top1_hits

ROOT = Path(__file__).resolve().parents[1]
from .paths import DATA_ROOT  # noqa: E402,F401  (re-exported for the scripts)
EXCLUDE_FROM_FIT = {"00000033", "00000054"}
SEED, SUB, FOLDS = 0, 0.30, 3


def load_part(part: str, exclude=()):
    rows, obs = [], {}
    for f in sorted((ROOT / "artifacts" / "data" / part).glob("*.joblib")):
        tab = joblib.load(f)
        if tab["trip_id"] in exclude:
            continue
        rows.append(tab["rows"])
        obs[tab["trip_id"]] = tab["obs"]
    return pd.concat(rows, ignore_index=True), obs


def cv_setup(train: pd.DataFrame):
    """Same folds and fixed observation-level fit subsample as scripts/03_train_models.py."""
    uk, inv = np.unique(obs_keys(train["trip"].to_numpy(), train["obs"].to_numpy()), return_inverse=True)
    sampled = (np.random.default_rng(SEED).random(len(uk)) < SUB)[inv]
    folds = list(GroupKFold(n_splits=FOLDS).split(np.zeros(len(train)), groups=train["trip"].to_numpy()))
    return folds, sampled


def score(part_rows: pd.DataFrame, p: np.ndarray, obs_tabs: dict):
    y = part_rows["label"].to_numpy()
    tabs = [obs_tabs[f"{t:08d}"] for t in np.unique(part_rows["trip"].to_numpy())]
    n_elig = sum(int(o.eligible.sum()) for o in tabs)
    n_ref = sum(int((o.eligible & o.ref_in_cands).sum()) for o in tabs)
    hits, win = top1_hits(part_rows["trip"].to_numpy(), part_rows["obs"].to_numpy(), y, p)
    m = clf_metrics(y, p)
    m.update(top1=hits / n_elig, top1_cond=hits / n_ref, n_obs=n_elig)
    return m, win

"""FastAPI backend (Laptop 1).

    GET  /health   -> {"status", "model", "n_trips", ...}
    GET  /trips    -> the trips this server can serve: [{"trip_id", "n_gps", "duration_s"}]
    POST /predict  {"trip_id": "..."} -> predicted routes, road geometry and aggregate scores

Ground truth stays here: the reference route (.route) is read only after the prediction exists, only to compute
aggregate scores, and it is never part of any response. Nothing is cached or hardcoded: every /predict call runs the
whole pipeline (candidates -> classifier -> HMM/Viterbi) and logs a line to this console.

Robustness: one broken trip folder never stops the server (it is skipped and logged); trips added after start-up appear on the
next /trips call; unreadable or inconsistent trip data answers 422 with a plain message; unexpected failures answer 500 with a generic
message (details only in this console); scoring problems never lose a prediction (the trip is simply returned with scored:false).

Configuration (environment variables)
    PATHSNAP_BUNDLE     inference bundle (default artifacts/final/bundle.joblib)
    PATHSNAP_TRIPS_DIR  folder of <trip>/<trip>.{track,nodes,arcs[,route]} to serve (default: the student_test trips)
    PATHSNAP_CORS_REGEX allowed browser origins (default: localhost and private LAN addresses)
run:  python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .io import BadTripData, load_route, load_trip_dir, trip_file
from .paths import DATA_ROOT
from .pipeline import load_bundle, predict_trip, score_prediction, ui_payload

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = Path(os.environ.get("PATHSNAP_BUNDLE", ROOT / "artifacts" / "final" / "bundle.joblib"))
TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", DATA_ROOT / "student_test"))
CORS_REGEX = os.environ.get("PATHSNAP_CORS_REGEX", r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")
REQUIRED = ("track", "nodes", "arcs")
MAX_ID_LEN = 128

log = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")


def _trip_meta(d: Path, tid: str) -> dict:
    g = pd.read_csv(trip_file(d, tid, "track"), sep=r"\s+", header=None, usecols=[0, 1, 2], dtype=np.float64).to_numpy()
    t = g[np.isfinite(g).all(axis=1), 2]                                    # same rows the predictor will use
    return {"trip_id": tid, "n_gps": int(len(t)), "duration_s": float(t.max() - t.min()) if len(t) else 0.0}


def discover_trips(root: Path, known: dict | None = None) -> dict:
    """Trips are the sub-folders holding <id>.track/.nodes/.arcs. Only these ids can ever be requested.

    Every folder is read defensively: a broken one is skipped with a log line, never fatal. `known` (the previous result) lets a
    rescan reuse unchanged entries, so calling this again is cheap and picks up trips added later."""
    known, trips = known or {}, {}
    if not root.is_dir():
        return trips
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and len(p.name) <= MAX_ID_LEN):
        tid = d.name
        missing = [e for e in REQUIRED if not trip_file(d, tid, e).exists()]
        if missing:
            log.warning("skipping folder %r: missing %s", tid, ", ".join("." + m for m in missing))
            continue
        stamp = trip_file(d, tid, "track").stat().st_mtime_ns
        if tid in known and known[tid].get("_stamp") == stamp:
            trips[tid] = known[tid]
            continue
        try:
            meta = _trip_meta(d, tid)
        except Exception as e:                                              # noqa: BLE001
            log.warning("skipping trip %r: its .track file cannot be read (%s)", tid, type(e).__name__)
            continue
        if meta["n_gps"] == 0:
            log.warning("skipping trip %r: no usable GPS rows in its .track file", tid)
            continue
        meta["_stamp"] = stamp
        trips[tid] = meta
    return trips


def _refresh() -> None:
    app.state.trips = discover_trips(TRIPS_DIR, app.state.trips)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TRIPS_DIR.is_dir():
        raise RuntimeError(f"the trips folder does not exist: {TRIPS_DIR}  (set PATHSNAP_TRIPS_DIR to the folder holding the trips; "
                           "in PowerShell use $env:PATHSNAP_TRIPS_DIR=\"...\", in cmd use set \"PATHSNAP_TRIPS_DIR=...\")")
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(f"the model bundle was not found: {BUNDLE_PATH}  (set PATHSNAP_BUNDLE, or run scripts/06_export_bundle.py)")
    app.state.bundle = load_bundle(BUNDLE_PATH)
    app.state.trips = discover_trips(TRIPS_DIR)
    app.state.lock = threading.Lock()
    log.info("model   : %s (bundle %s)", app.state.bundle.meta.get("name"), BUNDLE_PATH)
    log.info("decoder : %s", app.state.bundle.hmm)
    log.info("serving : %d trips from %s", len(app.state.trips), TRIPS_DIR)
    if "PATHSNAP_TRIPS_DIR" not in os.environ:
        log.warning("PATHSNAP_TRIPS_DIR is not set: serving the DEVELOPMENT trips (student_test). For the evaluation, set it to the evaluation trips folder and restart.")
    if not app.state.trips:
        log.warning("NO USABLE TRIPS were found in %s. Each trip needs a folder <id>/ with <id>.track, <id>.nodes and <id>.arcs. "
                    "Trips added later appear automatically on the next /trips call.", TRIPS_DIR)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=CORS_REGEX, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class PredictRequest(BaseModel):
    trip_id: str


def _finite(x):
    """JSON cannot carry NaN/inf: replace them by null so a metric problem can never turn into a bare 500."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _fmt(v, spec: str) -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def _score(pred, trip, d: Path, tid: str) -> dict:
    """Scoring is optional: any problem with the reference route yields scored:false and the prediction is still returned."""
    try:
        trip.route = load_route(d, tid)                                    # server-side only, after predicting
        metrics = score_prediction(pred, trip)
    except Exception:                                                      # noqa: BLE001
        log.exception("scoring failed for trip=%s; returning the prediction unscored", tid)
        metrics = {"scored": False, "n_obs": int(trip.n_gps), "latency_ms": {k: int(round(v)) for k, v in pred.timings_ms.items()}}
    return _finite(metrics)


@app.get("/", include_in_schema=False)
def root():
    """Friendly landing answer so that opening the API address in a browser is not a bare 404."""
    return {"service": "PathSnap API", "status": "ok", "endpoints": ["GET /health", "GET /trips", "POST /predict"],
            "note": "This is the API, not the user interface. Open the UI on the client laptop, e.g. http://localhost:8080/?api=http://<this-laptop-address>:8000"}


@app.get("/health")
def health():
    b = app.state.bundle
    return {"status": "ok", "model": b.meta.get("name"), "n_trips": len(app.state.trips), "decoder": b.hmm}


@app.get("/trips")
def trips():
    _refresh()                                                             # picks up trips added after start-up
    return [{k: v for k, v in t.items() if not k.startswith("_")} for t in app.state.trips.values()]


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    tid = req.trip_id
    if tid not in app.state.trips:
        _refresh()
    if tid not in app.state.trips:                 # allow-list of real folder names: also rules out any path traversal
        log.warning("PREDICT rejected unknown trip_id=%r from %s", tid[:64], client)
        raise HTTPException(status_code=404, detail="unknown trip_id")
    t0 = time.perf_counter()
    d = TRIPS_DIR / tid
    try:
        with app.state.lock:                        # one heavy prediction at a time
            trip = load_trip_dir(d, tid, with_route=False)      # the predictor never sees the route
            if trip.n_gps == 0:
                raise BadTripData("the trip has no usable GPS points")
            pred = predict_trip(app.state.bundle, trip)
        metrics = _score(pred, trip, d, tid)
        payload = ui_payload(pred, trip, metrics)
    except BadTripData as e:                        # the trip's input files are the problem: say so plainly
        log.warning("PREDICT bad input trip=%s client=%s: %s", tid, client, e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:                           # noqa: BLE001
        log.exception("PREDICT failed trip=%s client=%s", tid, client)
        raise HTTPException(status_code=500, detail=f"internal error while processing this trip ({type(e).__name__}); see the server console") from e
    payload["model"] = app.state.bundle.meta.get("name")
    ms = (time.perf_counter() - t0) * 1e3
    if metrics.get("scored"):
        log.info("PREDICT trip=%s client=%s obs=%d | acc clf %s -> hmm %s | continuity clf %s -> hmm %s | server %.0f ms",
                 tid, client, metrics["n_obs"], _fmt(metrics.get("pointwise_acc_classifier"), ".4f"), _fmt(metrics.get("pointwise_acc_hmm"), ".4f"),
                 _fmt(metrics.get("route_continuity_classifier"), ".3f"), _fmt(metrics.get("route_continuity_hmm"), ".3f"), ms)
    else:
        log.info("PREDICT trip=%s client=%s obs=%d | not scored (no usable reference route on this server) | server %.0f ms",
                 tid, client, metrics["n_obs"], ms)
    return payload

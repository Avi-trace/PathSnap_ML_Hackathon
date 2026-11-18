"""FastAPI backend (Laptop 1).

    GET  /health   -> {"status", "model", "n_trips", ...}
    GET  /trips    -> the trips this server can serve: [{"trip_id", "n_gps", "duration_s"}]
    POST /predict  {"trip_id": "..."} -> predicted routes, road geometry and aggregate scores

Ground truth stays here: the reference route (.route) is read only after the prediction exists, only to compute
aggregate scores, and it is never part of any response. Nothing is cached or hardcoded: every /predict call runs the
whole pipeline (candidates -> classifier -> HMM/Viterbi) and logs a line to this console.

Configuration (environment variables)
    PATHSNAP_BUNDLE     inference bundle (default artifacts/final/bundle.joblib)
    PATHSNAP_TRIPS_DIR  folder of <trip>/<trip>.{track,nodes,arcs[,route]} to serve (default: the student_test trips)
    PATHSNAP_CORS_REGEX allowed browser origins (default: localhost and private LAN addresses)
run:  python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .io import load_route, load_trip_dir
from .paths import DATA_ROOT
from .pipeline import load_bundle, predict_trip, score_prediction, ui_payload

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = Path(os.environ.get("PATHSNAP_BUNDLE", ROOT / "artifacts" / "final" / "bundle.joblib"))
TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", DATA_ROOT / "student_test"))
CORS_REGEX = os.environ.get("PATHSNAP_CORS_REGEX", r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

log = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")


def discover_trips(root: Path) -> dict:
    """Trips are the sub-folders holding <id>.track/.nodes/.arcs. Only these ids can ever be requested."""
    trips = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and ID_RE.match(p.name)):
        if all((d / f"{d.name}.{ext}").exists() for ext in ("track", "nodes", "arcs")):
            t = pd.read_csv(d / f"{d.name}.track", sep=r"\s+", header=None, usecols=[2]).iloc[:, 0].to_numpy()
            trips[d.name] = {"trip_id": d.name, "n_gps": int(len(t)), "duration_s": float(t[-1] - t[0]) if len(t) else 0.0}
    return trips


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(BUNDLE_PATH)
    app.state.trips = discover_trips(TRIPS_DIR)
    app.state.lock = threading.Lock()
    log.info("model   : %s (bundle %s)", app.state.bundle.meta.get("name"), BUNDLE_PATH)
    log.info("decoder : %s", app.state.bundle.hmm)
    log.info("serving : %d trips from %s", len(app.state.trips), TRIPS_DIR)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=CORS_REGEX, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    b = app.state.bundle
    return {"status": "ok", "model": b.meta.get("name"), "n_trips": len(app.state.trips), "decoder": b.hmm}


@app.get("/trips")
def trips():
    return list(app.state.trips.values())


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in app.state.trips:         # allow-list: also rules out any path traversal
        log.warning("PREDICT rejected unknown trip_id=%r from %s", req.trip_id[:64], request.client.host if request.client else "?")
        raise HTTPException(status_code=404, detail="unknown trip_id")
    t0 = time.perf_counter()
    d = TRIPS_DIR / req.trip_id
    try:
        with app.state.lock:                        # one heavy prediction at a time
            trip = load_trip_dir(d, req.trip_id, with_route=False)      # the predictor never sees the route
            pred = predict_trip(app.state.bundle, trip)
        route = load_route(d, req.trip_id)           # server-side only, after predicting
        if route is not None and len(route):
            trip.route = route
            metrics = score_prediction(pred, trip) | {"scored": True}
        else:
            metrics = {"scored": False, "n_obs": int(trip.n_gps), "latency_ms": {k: int(round(v)) for k, v in pred.timings_ms.items()}}
        payload = ui_payload(pred, trip, metrics)
    except Exception as e:                           # noqa: BLE001
        log.exception("PREDICT failed trip=%s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    payload["model"] = app.state.bundle.meta.get("name")
    ms = (time.perf_counter() - t0) * 1e3
    if metrics.get("scored"):
        log.info("PREDICT trip=%s client=%s obs=%d | acc clf %.4f -> hmm %.4f | continuity clf %.3f -> hmm %.3f | server %.0f ms",
                 req.trip_id, request.client.host if request.client else "?", metrics["n_obs"], metrics["pointwise_acc_classifier"],
                 metrics["pointwise_acc_hmm"], metrics["route_continuity_classifier"], metrics["route_continuity_hmm"], ms)
    else:
        log.info("PREDICT trip=%s client=%s obs=%d | not scored (no reference route on server) | server %.0f ms",
                 req.trip_id, request.client.host if request.client else "?", metrics["n_obs"], ms)
    return payload

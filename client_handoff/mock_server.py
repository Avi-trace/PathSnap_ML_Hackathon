"""DEV-ONLY mock of the PathSnap API, so the client can be built and tested without the real server.

Serves three recorded real responses (a small, a medium and a large trip) from ./samples with the same contract as
the real API. Every response is labelled as a mock (`mock: true`, model name "MOCK SERVER ...") and the client MUST
show a visible banner when it sees that. It must never be used for the final demo: the evaluator's trips are served
only by the real API on the server laptop.

    pip install fastapi uvicorn
    python mock_server.py            # http://127.0.0.1:8000   (--port 8001 to change)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SAMPLES = Path(__file__).resolve().parent / "samples"
TRIPS = json.loads((SAMPLES / "trips.json").read_text())
IDS = {t["trip_id"] for t in TRIPS}
MODEL = "MOCK SERVER - recorded sample data, not a live model"

app = FastAPI(title="PathSnap MOCK API (dev only)")
app.add_middleware(CORSMiddleware, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                   allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "mock": True, "n_trips": len(TRIPS)}


@app.get("/trips")
def trips():
    return TRIPS


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in IDS:                       # same allow-list behaviour as the real API
        raise HTTPException(status_code=404, detail="unknown trip_id")
    time.sleep(1.0)                                  # imitate real server latency
    d = json.loads((SAMPLES / f"predict_{req.trip_id}.json").read_text())
    d["model"], d["mock"] = MODEL, True
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print("=" * 70, "\n MOCK API - recorded sample data only. NOT the real model.\n" + "=" * 70)
    uvicorn.run(app, host=a.host, port=a.port)

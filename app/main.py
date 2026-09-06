"""
main.py
-------
FastAPI service that serves the heart-disease model.

Observability: every prediction request is logged as a single structured
JSON line to stdout (input features + prediction + probability + a
request id + a UTC timestamp). In GKE, stdout from the container is
automatically ingested by Cloud Logging — no extra agent config is
needed — so this satisfies Deliverable 5's "per-sample logging +
observability via Cloud Logging" requirement.
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")

FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------------------
# Structured JSON logging -> stdout (picked up by GKE / Cloud Logging)
# ---------------------------------------------------------------------------
logger = logging.getLogger("heart_disease_api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_json(event: str, **fields):
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    logger.info(json.dumps(record, default=str))


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
_model = None


class PatientFeatures(BaseModel):
    age: float
    gender: str = Field(..., description="'male' or 'female'")
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    probability_heart_disease: float


@app.on_event("startup")
def load_model():
    global _model
    log_json("startup", model_path=MODEL_PATH)
    _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/ready")
def ready():
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    request_id = str(uuid.uuid4())
    start = time.time()

    row = pd.DataFrame([patient.dict()])[FEATURE_COLUMNS]
    try:
        proba = float(_model.predict_proba(row)[0, 1])
        pred = int(_model.predict(row)[0])
    except Exception as exc:  # noqa: BLE001
        log_json("prediction_error", request_id=request_id, error=str(exc),
                  input_features=patient.dict())
        raise HTTPException(status_code=400, detail=f"inference failed: {exc}") from exc

    latency_ms = (time.time() - start) * 1000
    label = "yes" if pred == 1 else "no"

    # Per-sample prediction log: input features + output + timestamp
    log_json(
        "prediction",
        request_id=request_id,
        input_features=patient.dict(),
        prediction=label,
        probability_heart_disease=proba,
        latency_ms=round(latency_ms, 2),
    )

    return PredictionResponse(
        request_id=request_id,
        prediction=label,
        probability_heart_disease=proba,
    )

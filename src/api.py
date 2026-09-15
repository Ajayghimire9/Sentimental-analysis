"""FastAPI inference service for the registered sentiment model."""

from __future__ import annotations

import os
import time

import mlflow
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

app = FastAPI(title="Sentiment Intelligence API", version="2.0.0")
REQUESTS = Counter("sentiment_requests_total", "Total sentiment inference requests")
LATENCY = Histogram("sentiment_request_latency_seconds", "Inference latency in seconds")
PREDICTIONS = Counter("sentiment_predictions_total", "Predictions by class", ["label"])
_model = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


def get_model():
    global _model
    if _model is not None:
        return _model
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must point to the model registry")
    mlflow.set_tracking_uri(uri)
    _model = mlflow.pyfunc.load_model("models:/sentiment-classifier@champion")
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": "sentiment-classifier"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(request: PredictionRequest):
    REQUESTS.inc()
    started = time.perf_counter()
    try:
        prediction = int(get_model().predict([request.text])[0])
        label = {0: "negative", 1: "neutral", 2: "positive"}[prediction]
        PREDICTIONS.labels(label=label).inc()
        return {"label": label, "class_id": prediction, "text_length": len(request.text)}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Approved model unavailable or inference failed"
        ) from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/ready")
def ready():
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Approved model unavailable") from exc
    return {"ready": True}

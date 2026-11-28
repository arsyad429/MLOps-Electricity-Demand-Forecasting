
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
import mlflow
from mlflow import MlflowClient
import pandas as pd
import requests

import math
import time

from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from backend.schemas import Features

MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "CISO-Demand-Forecaster-Compose")
MODEL_URI = f"models:/{MODEL_NAME}@champion"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URL)
    client = MlflowClient(tracking_uri=MLFLOW_URL)
    version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    signature = model.metadata.get_input_schema()
    if signature is not None and signature.input_names() != list(Features.model_fields):
        raise RuntimeError("Loaded MLflow model input names or order differ from the prediction API schema.")
    app.state.client = client
    app.state.model = model
    app.state.model_version = str(version)
    yield


app = FastAPI(title="CISO Forecasting — Docker Compose", lifespan=lifespan)

PREDICTION_REQUESTS = Counter(
    "ciso_prediction_requests_total",
    "Total prediction requests",
    ["status"],
)

PREDICTION_DURATION = Histogram(
    "ciso_prediction_duration_seconds",
    "Duration of model inference in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, float("inf")),
)

PREDICTED_DEMAND = Gauge(
    "ciso_predicted_demand_mwh",
    "Latest predicted electricity demand in MWh",
)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/mlflow-check")
def mlflow_check():
    """Proves connection to the *other container* over the custom bridge network."""
    try:
        response = requests.get(f"{MLFLOW_URL}/health", timeout=5)
        response.raise_for_status()
        version = app.state.client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {type(exc).__name__}") from exc
    return {
        "communication": "success",
        "mlflow_service_url": MLFLOW_URL,
        "registered_model": MODEL_NAME,
        "model_version": version.version,
        "loaded_model_version": app.state.model_version,
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict(payload: Features):
    start = time.perf_counter()
    status = "error"

    try:
        features = pd.DataFrame([payload.model_dump()])
        y = float(app.state.model.predict(features)[0])

        if not math.isfinite(y):
            raise ValueError("Model prediction is not finite.")

        PREDICTED_DEMAND.set(y)
        status = "success"

        return {
            "predicted_demand": y,
            "forecast_horizon_hours": 2,
            "model_uri": MODEL_URI,
            "model_version": app.state.model_version,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}"
        ) from exc

    finally:
        PREDICTION_DURATION.observe(
            time.perf_counter() - start
        )
        PREDICTION_REQUESTS.labels(status=status).inc()
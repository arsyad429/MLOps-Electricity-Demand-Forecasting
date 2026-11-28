"""Minimal FastAPI serving the model from the Compose-local MLflow registry.

POST /predict accepts engineered feature values, not raw EIA observations.
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
import mlflow
from mlflow import MlflowClient
import pandas as pd
import requests

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
    try:
        features = pd.DataFrame([payload.model_dump()])
        y = float(app.state.model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc
    return {
        "predicted_demand": y,
        "forecast_horizon_hours": 2,
        "model_uri": MODEL_URI,
        "model_version": app.state.model_version,
    }

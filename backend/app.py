"""Minimal FastAPI serving the model from the Compose-local MLflow registry.

POST /predict accepts engineered feature values, not raw EIA observations.
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel, Field
import pandas as pd
import requests

MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "CISO-Demand-Forecaster-Compose")
MODEL_URI = f"models:/{MODEL_NAME}@champion"


class Features(BaseModel):
    value: float = Field(ge=0)
    lag_1h: float = Field(ge=0)
    lag_2h: float = Field(ge=0)
    lag_24h: float = Field(ge=0)
    rolling_mean_3h: float = Field(ge=0)
    rolling_mean_24h: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URL)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    app.state.client = MlflowClient(tracking_uri=MLFLOW_URL)
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
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict(payload: Features):
    try:
        features = pd.DataFrame([payload.model_dump()])
        y = float(app.state.model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc
    return {"predicted_demand": y, "forecast_horizon_hours": 2, "model_uri": MODEL_URI}

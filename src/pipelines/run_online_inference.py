"""Run one live CISO t+2h forecast through the FastAPI prediction service."""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from numbers import Real
from time import perf_counter
from urllib.parse import urlsplit

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from backend.schemas import Features
from src.data.ingest_data import EIAError, EIATemporaryError, HOUR, PROJECT_ROOT
from src.features.feature_engineering import FEATURE_COLUMNS
from src.pipelines.check_inference_readiness import check_inference_readiness


FORECAST_HORIZON_HOURS = 2
PREDICTION_TIMEOUT = (3, 10)
MODEL_URI_PATTERN = re.compile(r"models:/[A-Za-z0-9_.-]+(?:@[A-Za-z0-9_.-]+|/[1-9][0-9]*)?")


@dataclass(frozen=True)
class OnlineInferenceResult:
    status: str
    data_source: str | None = None
    observation_timestamp: pd.Timestamp | None = None
    prediction_created_at: pd.Timestamp | None = None
    forecast_target_timestamp: pd.Timestamp | None = None
    latest_actual_demand: float | None = None
    predicted_demand: float | None = None
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS
    input_data_age_hours: float | None = None
    inference_latency_ms: float | None = None
    model_uri: str | None = None
    model_version: str | None = None
    http_status: int | None = None
    message: str | None = None

    def to_dict(self):
        return {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in asdict(self).items()
        }


class PredictionError(RuntimeError):
    """A secret-safe failure from the FastAPI request or response."""

    def __init__(self, status, message, *, http_status=None):
        super().__init__(message)
        self.status = status
        self.http_status = http_status


def _now_utc():
    return pd.Timestamp.now(tz="UTC")


def _utc_hour(value):
    if not isinstance(value, datetime) or pd.isna(value) or value.utcoffset() is None:
        raise ValueError("Observation and forecast target must be timezone-aware timestamps.")
    timestamp = pd.Timestamp(value).tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError("Observation and forecast target must fall on UTC hour boundaries.")
    return timestamp


def _is_finite_number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _observation_metadata(readiness):
    observation = _utc_hour(readiness.latest_observation)
    target = _utc_hour(readiness.target_timestamp)
    if target != observation + FORECAST_HORIZON_HOURS * HOUR:
        raise ValueError("Forecast target must equal the observation timestamp plus two hours.")

    observations = readiness.observations
    if (not isinstance(observations, pd.DataFrame) or observations.empty
            or not {"period", "value"}.issubset(observations.columns)):
        raise ValueError("Readiness did not supply the latest CISO observation.")
    latest = observations.iloc[-1]
    if _utc_hour(latest["period"]) != observation:
        raise ValueError("Readiness timestamp does not match the latest CISO observation.")
    demand = latest["value"]
    if not _is_finite_number(demand) or demand < 0:
        raise ValueError("Latest actual demand must be finite and nonnegative.")
    return observation, target, float(demand)


def _feature_payload(frame, latest_actual):
    if (not isinstance(frame, pd.DataFrame) or len(frame) != 1
            or list(frame.columns) != FEATURE_COLUMNS
            or list(Features.model_fields) != FEATURE_COLUMNS):
        raise ValueError("Inference requires one row with the ten model features in training order.")
    payload = frame.to_dict(orient="records")[0]
    if not all(_is_finite_number(value) for value in payload.values()):
        raise ValueError("All ten model features must be finite numbers.")
    try:
        payload = Features.model_validate(payload).model_dump()
    except ValidationError:
        raise ValueError("One or more model features are outside the API's allowed ranges.") from None
    if payload["value"] != latest_actual:
        raise ValueError("The value feature does not match the latest actual demand.")
    return payload


def _prediction_url():
    load_dotenv(PROJECT_ROOT / ".env")
    base_url = os.getenv("INFERENCE_API_URL", "http://127.0.0.1:8000")
    message = "INFERENCE_API_URL must be an HTTP(S) base URL without credentials, a query, or a fragment."
    try:
        url = urlsplit(base_url)
        _ = url.port
    except ValueError:
        raise ValueError(message) from None
    if (url.scheme not in {"http", "https"} or not url.hostname or "@" in url.netloc
            or "?" in base_url or "#" in base_url
            or any(character.isspace() or ord(character) < 32 for character in base_url)):
        raise ValueError(message)
    return f"{base_url.rstrip('/')}/predict"


def _parse_prediction(response):
    try:
        body = response.json()
    except ValueError:
        raise PredictionError("invalid_response", "Prediction API returned invalid JSON.") from None
    if not isinstance(body, dict) or not _is_finite_number(body.get("predicted_demand")):
        raise PredictionError("invalid_response", "Prediction API response requires finite numeric predicted_demand.")
    horizon = body.get("forecast_horizon_hours")
    if type(horizon) is not int or horizon != FORECAST_HORIZON_HOURS:
        raise PredictionError("invalid_response", "Prediction API response must specify a two-hour horizon.")

    model_uri = body.get("model_uri")
    if model_uri is not None and (
        not isinstance(model_uri, str) or not MODEL_URI_PATTERN.fullmatch(model_uri)
    ):
        raise PredictionError("invalid_response", "Prediction API returned an invalid registered-model URI.")
    version = body.get("model_version")
    if version is not None and not re.fullmatch(r"[1-9][0-9]*", str(version)):
        raise PredictionError("invalid_response", "Prediction API returned an invalid model version.")
    return float(body["predicted_demand"]), model_uri, None if version is None else str(version)


def _request_prediction(url, payload):
    try:
        response = requests.post(
            url, json=payload, timeout=PREDICTION_TIMEOUT, allow_redirects=False
        )
    except requests.Timeout:
        raise PredictionError("api_timeout", "Prediction API request timed out.") from None
    except requests.ConnectionError:
        raise PredictionError("api_connection_error", "Could not connect to the prediction API.") from None
    except requests.RequestException:
        raise PredictionError("api_request_error", "Prediction API request failed.") from None

    if response.status_code != 200:
        raise PredictionError(
            "api_http_error",
            f"Prediction API returned HTTP {response.status_code}.",
            http_status=response.status_code,
        )
    return _parse_prediction(response)


def run_online_inference():
    """Return one forecast, or a structured status explaining why it was skipped."""
    try:
        readiness = check_inference_readiness()
    except EIATemporaryError as error:
        return OnlineInferenceResult("ingestion_unavailable", message=str(error))
    except EIAError as error:
        return OnlineInferenceResult("ingestion_error", message=str(error))

    result = OnlineInferenceResult(readiness.status, data_source=readiness.source)
    if readiness.status == "no_observations":
        return replace(result, message="EIA returned no actual demand observations.")

    try:
        observation, target, demand = _observation_metadata(readiness)
    except ValueError as error:
        return replace(result, status="invalid_observation", message=str(error))
    now = _now_utc()
    if observation > now:
        return replace(result, status="invalid_observation", message="Latest observation is in the future.")

    result = replace(
        result,
        observation_timestamp=observation,
        forecast_target_timestamp=target,
        latest_actual_demand=demand,
        input_data_age_hours=(now - observation) / HOUR,
    )
    if readiness.status == "stale" or target <= now:
        return replace(result, status="stale", message="The t+2h forecast target has already arrived.")
    if readiness.status != "ready":
        return replace(result, message="EIA observations are missing the required consecutive hourly context.")

    try:
        payload = _feature_payload(readiness.features, demand)
    except ValueError as error:
        return replace(result, status="invalid_features", message=str(error))
    try:
        url = _prediction_url()
    except ValueError as error:
        return replace(result, status="configuration_error", message=str(error))

    start = perf_counter()
    try:
        prediction, model_uri, model_version = _request_prediction(url, payload)
    except PredictionError as error:
        return replace(
            result,
            status=error.status,
            http_status=error.http_status,
            message=str(error),
            inference_latency_ms=round((perf_counter() - start) * 1000, 3),
        )
    latency_ms = round((perf_counter() - start) * 1000, 3)
    created_at = _now_utc()
    result = replace(
        result,
        input_data_age_hours=(created_at - observation) / HOUR,
        inference_latency_ms=latency_ms,
    )
    if target <= created_at:
        return replace(result, status="stale", message="The t+2h forecast target arrived during inference.")

    return replace(
        result,
        status="success",
        prediction_created_at=created_at,
        predicted_demand=prediction,
        model_uri=model_uri,
        model_version=model_version,
    )


def main():
    result = run_online_inference()
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

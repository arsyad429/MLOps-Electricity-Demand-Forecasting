import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow.pyfunc
import pandas as pd
import pytest
import requests
from fastapi import FastAPI

import backend.app as serving
from backend.schemas import Features
from src.features.feature_engineering import FEATURE_COLUMNS
from src.pipelines import run_online_inference as online
from src.pipelines.check_inference_readiness import _assess_readiness


OBSERVATION = pd.Timestamp("2026-11-28T02:00:00Z")
CREATE_TIME = pd.Timestamp("2026-11-28T03:07:00Z")
TARGET = pd.Timestamp("2026-11-28T04:00:00Z")
MODEL_URI = "models:/CISO-Demand-Forecaster-Compose@champion"
PREDICTION_REPLY = {
    "predicted_demand": 25_196.375,
    "forecast_horizon_hours": 2,
    "model_uri": MODEL_URI,
    "model_version": "3",
}


@pytest.fixture
def ready():
    observations = pd.DataFrame({
        "period": pd.date_range(end=OBSERVATION, periods=25, freq="h"),
        "value": list(range(23_000, 23_025)),
    })
    observations.attrs["source"] = "dashboard"
    return _assess_readiness(observations, now=CREATE_TIME)


@pytest.fixture
def mocked_cycle(monkeypatch, ready):
    response = Mock(status_code=200)
    response.json.return_value = PREDICTION_REPLY.copy()
    post = Mock(return_value=response)
    readiness = Mock(return_value=ready)
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:9000/")
    monkeypatch.setattr(online, "check_inference_readiness", readiness)
    monkeypatch.setattr(online.requests, "post", post)
    monkeypatch.setattr(online, "_now_utc", Mock(return_value=CREATE_TIME))
    monkeypatch.setattr(online, "perf_counter", Mock(side_effect=[5, 5.125]))
    return readiness, post, response


def test_online_prediction_sends_the_training_features_and_returns_metadata(mocked_cycle, ready):
    readiness, post, _ = mocked_cycle

    result = online.run_online_inference()

    readiness.assert_called_once_with()
    post.assert_called_once()
    url, = post.call_args.args
    payload = post.call_args.kwargs["json"]
    assert url == "http://localhost:9000/predict"
    assert list(payload) == FEATURE_COLUMNS == list(Features.model_fields)
    assert payload == ready.features.to_dict(orient="records")[0]
    assert type(payload["hour"]) is int
    assert post.call_args.kwargs["timeout"] == (3, 10)
    assert post.call_args.kwargs["allow_redirects"] is False
    assert result.status == "success"
    assert result.data_source == "dashboard"
    assert result.latest_actual_demand == 23_024
    assert result.predicted_demand == 25_196.375
    assert result.model_uri == MODEL_URI
    assert result.model_version == "3"
    assert result.observation_timestamp == OBSERVATION
    assert result.prediction_created_at == CREATE_TIME
    assert result.forecast_target_timestamp == TARGET == OBSERVATION + pd.Timedelta(hours=2)
    assert result.forecast_horizon_hours == 2
    assert result.input_data_age_hours == pytest.approx(67 / 60)
    assert result.inference_latency_ms == 125
    assert result.to_dict()["forecast_target_timestamp"] == "2026-11-28T04:00:00+00:00"


@pytest.mark.parametrize("change", [
    lambda frame: None,
    lambda frame: frame.drop(columns="lag_24h"),
    lambda frame: frame.iloc[:, ::-1],
    lambda frame: pd.concat([frame, frame]),
    lambda frame: frame.assign(lag_24h=float("nan")),
    lambda frame: frame.assign(value=float("inf")),
    lambda frame: frame.assign(hour=True),
    lambda frame: frame.assign(hour=2.5),
    lambda frame: frame.assign(month=0),
    lambda frame: frame.assign(lag_1h="23001"),
    lambda frame: frame.assign(value=-1.0),
    lambda frame: frame.assign(value=15_000.0),
])
def test_invalid_features_never_reach_fastapi(mocked_cycle, ready, change):
    readiness, post, _ = mocked_cycle
    readiness.return_value = replace(ready, features=change(ready.features))

    result = online.run_online_inference()

    assert result.status == "invalid_features"
    assert result.predicted_demand is None
    post.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"latest_observation": OBSERVATION.tz_localize(None)},
    {"target_timestamp": pd.NaT},
    {"target_timestamp": TARGET + pd.Timedelta(hours=1)},
    {"target_timestamp": TARGET + pd.Timedelta(minutes=1)},
    {"latest_observation": OBSERVATION - pd.Timedelta(hours=1),
     "target_timestamp": TARGET - pd.Timedelta(hours=1)},
])
def test_invalid_observation_or_target_never_reaches_fastapi(mocked_cycle, ready, changes):
    readiness, post, _ = mocked_cycle
    readiness.return_value = replace(ready, **changes)

    result = online.run_online_inference()

    assert result.status == "invalid_observation"
    post.assert_not_called()


@pytest.mark.parametrize("readiness_status", ["ready", "stale"])
def test_stale_observations_never_reach_fastapi(mocked_cycle, ready, monkeypatch, readiness_status):
    readiness, post, _ = mocked_cycle
    readiness.return_value = replace(ready, status=readiness_status)
    monkeypatch.setattr(online, "_now_utc", Mock(return_value=TARGET))

    result = online.run_online_inference()

    assert result.status == "stale"
    assert result.forecast_target_timestamp == TARGET
    assert result.input_data_age_hours == 2
    assert result.prediction_created_at is None
    post.assert_not_called()


def test_response_crossing_target_time_is_stale(mocked_cycle, monkeypatch):
    _, post, _ = mocked_cycle
    monkeypatch.setattr(online, "_now_utc", Mock(side_effect=[CREATE_TIME, TARGET]))

    result = online.run_online_inference()

    post.assert_called_once()
    assert result.status == "stale"
    assert result.predicted_demand is None
    assert result.inference_latency_ms == 125
    assert result.input_data_age_hours == 2


@pytest.mark.parametrize("status", ["no_observations", "incomplete_context"])
def test_not_ready_preserves_readiness_status(mocked_cycle, ready, status):
    readiness, post, _ = mocked_cycle
    readiness.return_value = replace(ready, status=status, features=None)

    assert online.run_online_inference().status == status
    post.assert_not_called()


@pytest.mark.parametrize("payload", [
    None,
    [],
    {},
    {"predicted_demand": 22_000},
    {"forecast_horizon_hours": 2},
    PREDICTION_REPLY | {"predicted_demand": "23000"},
    PREDICTION_REPLY | {"predicted_demand": True},
    PREDICTION_REPLY | {"predicted_demand": float("inf")},
    PREDICTION_REPLY | {"forecast_horizon_hours": 1},
    PREDICTION_REPLY | {"forecast_horizon_hours": "2"},
    PREDICTION_REPLY | {"model_uri": "https://example.invalid/?token=fake-secret"},
    PREDICTION_REPLY | {"model_version": "champion"},
])
def test_incomplete_or_invalid_prediction_response(mocked_cycle, payload):
    _, _, response = mocked_cycle
    response.json.return_value = payload

    result = online.run_online_inference()

    assert result.status == "invalid_response"
    assert result.inference_latency_ms == 125
    assert result.predicted_demand is None
    assert "fake-secret" not in json.dumps(result.to_dict())


def test_non_json_prediction_response_is_secret_safe(mocked_cycle):
    _, _, response = mocked_cycle
    response.json.side_effect = ValueError("JSON contained fake-secret")

    result = online.run_online_inference()

    assert result.status == "invalid_response"
    assert "fake-secret" not in result.message


def test_older_server_without_model_version_is_supported(mocked_cycle):
    _, _, response = mocked_cycle
    response.json.return_value.pop("model_version")

    result = online.run_online_inference()

    assert result.status == "success"
    assert result.model_uri == MODEL_URI
    assert result.model_version is None


@pytest.mark.parametrize(("error", "status"), [
    (requests.ConnectionError("url contained fake-secret"), "api_connection_error"),
    (requests.ConnectTimeout("url contained fake-secret"), "api_timeout"),
    (requests.ReadTimeout("url contained fake-secret"), "api_timeout"),
])
def test_api_transport_failures_are_bounded_and_secret_safe(mocked_cycle, error, status):
    _, post, _ = mocked_cycle
    post.side_effect = error

    result = online.run_online_inference()

    post.assert_called_once()
    assert result.status == status
    assert result.inference_latency_ms == 125
    assert "fake-secret" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("status_code", [307, 400, 422, 500, 503])
def test_http_errors_report_the_status_without_exposing_the_body(mocked_cycle, status_code):
    _, _, response = mocked_cycle
    response.status_code = status_code
    response.text = "request body contained fake-secret"

    result = online.run_online_inference()

    assert result.status == "api_http_error"
    assert result.http_status == status_code
    assert str(status_code) in result.message
    assert "fake-secret" not in json.dumps(result.to_dict())
    response.json.assert_not_called()


@pytest.mark.parametrize("url", [
    "http://user:fake-secret@localhost:8000",
    "http://localhost:8000?token=fake-secret",
    "http://localhost:8000/#fake-secret",
    "not-a-url",
])
def test_invalid_api_url_is_rejected_without_echoing_credentials(mocked_cycle, monkeypatch, url):
    _, post, _ = mocked_cycle
    monkeypatch.setenv("INFERENCE_API_URL", url)

    result = online.run_online_inference()

    assert result.status == "configuration_error"
    assert "fake-secret" not in result.message
    post.assert_not_called()


@pytest.mark.parametrize(("status", "exit_code"), [("success", 0), ("stale", 1)])
def test_cli_emits_a_json_record_and_status_exit_code(monkeypatch, capsys, status, exit_code):
    result = online.OnlineInferenceResult(status, observation_timestamp=OBSERVATION)
    monkeypatch.setattr(online, "run_online_inference", Mock(return_value=result))

    assert online.main() == exit_code
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == status
    assert output["observation_timestamp"] == "2026-11-28T02:00:00+00:00"


@pytest.mark.parametrize("matching_signature", [True, False])
def test_fastapi_pins_loaded_version_even_if_the_champion_alias_moves(monkeypatch, ready, matching_signature):
    registry = Mock()
    registry.get_model_version_by_alias.side_effect = [
        SimpleNamespace(version="3"), SimpleNamespace(version="4"),
    ]
    model = Mock()
    model.metadata.get_input_schema.return_value.input_names.return_value = (
        FEATURE_COLUMNS if matching_signature else FEATURE_COLUMNS[::-1]
    )
    model.predict.return_value = [25_196.375]
    load_model = Mock(return_value=model)
    api = FastAPI()
    monkeypatch.setattr(serving, "app", api)
    monkeypatch.setattr(serving, "MlflowClient", Mock(return_value=registry))
    monkeypatch.setattr(serving.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_model)
    monkeypatch.setattr(serving.requests, "get", Mock())

    async def exercise_app():
        async with serving.lifespan(api):
            check = serving.mlflow_check()
            assert check["model_version"] == "4"
            assert check["loaded_model_version"] == "3"
            reply = serving.predict(Features.model_validate(ready.features.to_dict(orient="records")[0]))
            assert reply == PREDICTION_REPLY
            assert list(model.predict.call_args.args[0].columns) == FEATURE_COLUMNS

    if matching_signature:
        asyncio.run(exercise_app())
    else:
        with pytest.raises(RuntimeError, match="input names or order"):
            asyncio.run(exercise_app())

    load_model.assert_called_once_with(f"models:/{serving.MODEL_NAME}/3")

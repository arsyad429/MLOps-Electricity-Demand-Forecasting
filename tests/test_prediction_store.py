import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.data.ingest_data import EIATemporaryError
from src.pipelines import check_inference_readiness as readiness
from src.pipelines import run_online_inference as online
from src.storage import prediction_store as storage


OBSERVATION = pd.Timestamp("2026-11-28T02:00:00Z")
CREATED = pd.Timestamp("2026-11-28T03:00:00Z")
COMPLETED = pd.Timestamp("2026-11-28T03:00:01Z")
TARGET = pd.Timestamp("2026-11-28T04:00:00Z")
MODEL_URI = "models:/CISO-Demand-Forecaster-Compose@champion"


@pytest.fixture
def store(tmp_path):
    return storage.PredictionStore(tmp_path / "worker#data.sqlite3")


@pytest.fixture
def prediction():
    return storage.Prediction(
        observation_timestamp=OBSERVATION,
        forecast_target_timestamp=TARGET,
        prediction_created_at=CREATED,
        predicted_demand=27_000,
        actual_input_demand=25_500,
        data_source="dashboard",
        forecast_horizon_hours=2,
        model_uri=MODEL_URI,
        model_version="1",
        inference_latency_ms=20.5,
    )


def rows(store, kind):
    return storage.recent_records(kind, path=store.path, limit=100)


def add_observation(store, *, demand=25_500, timestamp=OBSERVATION, source="dashboard", seen=CREATED):
    store.upsert_observations([(timestamp, demand)], source=source, ingested_at=seen)


def complete_prediction(store, prediction, execution_id=None):
    if execution_id is None:
        execution_id = store.start_run(started_at=CREATED)
    return store.complete_run(
        execution_id,
        status="success",
        source="dashboard",
        latest_observation=OBSERVATION,
        prediction=prediction,
        completed_at=COMPLETED,
    )


def test_initializes_schema_wal_busy_timeout_and_foreign_keys(store):
    with closing(storage._connect(store.path)) as connection:
        tables = {
            row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert tables == {"observations", "observation_revisions", "predictions", "pipeline_runs"}
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert rows(store, "observations") == []


def test_database_path_uses_environment_and_project_runtime_default(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "load_dotenv", Mock())
    monkeypatch.delenv("PREDICTION_DB_PATH", raising=False)
    assert storage.prediction_db_path() == storage.PROJECT_ROOT / "runtime/predictions.sqlite3"
    monkeypatch.setenv("PREDICTION_DB_PATH", str(tmp_path / "configured.sqlite3"))
    assert storage.PredictionStore().path == tmp_path / "configured.sqlite3"


def test_newer_schema_is_rejected_without_changing_version(store):
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(storage.StorageError, match="newer schema"):
        storage.PredictionStore(store.path)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_observation_upsert_normalizes_utc_and_keeps_first_ingestion(store):
    add_observation(store, timestamp="2026-11-27T18:00:00-08:00", seen="2026-11-27T19:00:00-08:00")
    add_observation(store, timestamp="2026-11-28T02:00:00Z", seen=COMPLETED)

    observation, = rows(store, "observations")
    assert observation["observation_timestamp"] == "2026-11-28T02:00:00.000000Z"
    assert observation["actual_demand"] == 25_500
    assert observation["data_source"] == "dashboard"
    assert observation["ingested_at"] == observation["updated_at"] == "2026-11-28T03:00:00.000000Z"
    assert observation["last_seen_at"] == "2026-11-28T03:00:01.000000Z"
    assert observation["revision_count"] == 0
    assert rows(store, "revisions") == []


def test_observation_revisions_preserve_both_values_and_ignore_older_fetches(store):
    add_observation(store)
    add_observation(store, demand=25_750, source="api", seen=COMPLETED)
    add_observation(store, demand=24_500, seen=CREATED)
    add_observation(store, demand=25_750, source="api", seen=COMPLETED)

    current, = rows(store, "observations")
    revision, = rows(store, "revisions")
    assert current["actual_demand"] == revision["revised_actual_demand"] == 25_750
    assert current["data_source"] == revision["revised_data_source"] == "api"
    assert revision["previous_actual_demand"] == 25_500
    assert revision["previous_data_source"] == "dashboard"
    assert revision["revision_number"] == current["revision_count"] == 1
    assert current["updated_at"] == revision["revised_at"] == "2026-11-28T03:00:01.000000Z"


@pytest.mark.parametrize(("timestamp", "demand", "source"), [
    (None, 12, "api"),
    ("2026-11-28T02:00:00", 12, "api"),
    ("2026-11-28T02:01:00Z", 12, "api"),
    ("2026-11-28T04:00:00Z", 12, "api"),
    (OBSERVATION, None, "api"),
    (OBSERVATION, float("nan"), "api"),
    (OBSERVATION, float("inf"), "api"),
    (OBSERVATION, -1, "dashboard"),
    (OBSERVATION, False, "dashboard"),
    (OBSERVATION, 12, "imputed"),
])
def test_rejects_invalid_observations(store, timestamp, demand, source):
    with pytest.raises(ValueError):
        add_observation(store, timestamp=timestamp, demand=demand, source=source)
    assert rows(store, "observations") == []


def test_invalid_row_rolls_back_the_entire_batch_and_its_revisions(store):
    add_observation(store)
    with pytest.raises(ValueError):
        store.upsert_observations(
            [(OBSERVATION, 25_505), (OBSERVATION + pd.Timedelta(hours=1), None)],
            source="dashboard",
            ingested_at=COMPLETED,
        )
    observation, = rows(store, "observations")
    assert observation["actual_demand"] == 25_500
    assert observation["revision_count"] == 0
    assert rows(store, "revisions") == []


def test_successful_prediction_survives_reopening_and_observation_revisions(store, prediction):
    add_observation(store)
    outcome = complete_prediction(store, prediction)
    add_observation(store, demand=26_000, seen=COMPLETED)

    reopened = storage.PredictionStore(store.path)
    stored, = rows(reopened, "predictions")
    run, = rows(reopened, "runs")
    assert outcome.status == "success"
    assert stored["prediction_id"] == run["prediction_id"] == outcome.prediction_id
    assert stored["predicted_demand"] == 27_000
    assert stored["actual_input_demand"] == 25_500
    assert stored["data_source"] == "dashboard"
    assert stored["model_version"] == "1"
    assert stored["model_uri"] == MODEL_URI
    assert stored["forecast_horizon_hours"] == 2
    assert stored["forecast_target_timestamp"] == "2026-11-28T04:00:00.000000Z"
    assert stored["prediction_created_at"] == "2026-11-28T03:00:00.000000Z"
    assert stored["created_at"] == "2026-11-28T03:00:01.000000Z"
    assert stored["inference_latency_ms"] == 20.5
    assert rows(reopened, "observations")[0]["actual_demand"] == 26_000


def test_predictions_deduplicate_across_aliases_and_allow_different_loaded_versions(store, prediction):
    add_observation(store)
    original = complete_prediction(store, prediction)
    duplicate = complete_prediction(
        store, replace(prediction, model_uri=MODEL_URI.replace("@champion", "/1"), predicted_demand=18_000)
    )
    another_model = complete_prediction(store, replace(prediction, model_uri="models:/other@champion"))
    another_version = complete_prediction(store, replace(prediction, model_version="2"))

    assert duplicate.status == "duplicate"
    assert duplicate.prediction_id == original.prediction_id
    assert duplicate.existing_prediction["predicted_demand"] == prediction.predicted_demand
    assert another_model.status == another_version.status == "success"
    assert len(rows(store, "predictions")) == 3
    assert [run["status"] for run in rows(store, "runs")] == ["success", "success", "duplicate", "success"]


@pytest.mark.parametrize(("first_version", "second_version", "first_uri", "second_uri"), [
    (None, None, MODEL_URI, MODEL_URI),
    (None, "1", MODEL_URI, MODEL_URI),
    ("1", None, MODEL_URI, MODEL_URI),
    (None, "1", None, MODEL_URI),
    ("1", None, MODEL_URI, None),
])
def test_missing_model_metadata_cannot_bypass_duplicate_detection(
    store, prediction, first_version, second_version, first_uri, second_uri
):
    add_observation(store)
    first = complete_prediction(store, replace(prediction, model_version=first_version, model_uri=first_uri))
    second = complete_prediction(store, replace(prediction, model_version=second_version, model_uri=second_uri))

    assert second.status == "duplicate"
    assert first.prediction_id == second.prediction_id
    assert len(rows(store, "predictions")) == 1


@pytest.mark.parametrize("changes", [
    {"predicted_demand": float("inf")},
    {"predicted_demand": None},
    {"inference_latency_ms": -1},
    {"actual_input_demand": -1},
    {"forecast_horizon_hours": 3},
    {"forecast_target_timestamp": OBSERVATION},
    {"prediction_created_at": TARGET},
    {"model_uri": "https://example.invalid/?token=fake-secret"},
    {"model_version": "latest"},
])
def test_invalid_prediction_keeps_execution_pending_and_writes_no_forecast(store, prediction, changes):
    add_observation(store)
    execution = store.start_run(started_at=CREATED)
    with pytest.raises(ValueError) as error:
        complete_prediction(store, replace(prediction, **changes), execution)
    assert "fake-secret" not in str(error.value)
    assert rows(store, "predictions") == []
    assert rows(store, "runs")[0]["status"] == "running"


def test_prediction_requires_an_existing_observation(store, prediction):
    execution = store.start_run(started_at=CREATED)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        complete_prediction(store, prediction, execution)
    assert rows(store, "predictions") == []
    assert rows(store, "runs")[0]["completed_at"] is None


def test_prediction_insert_rolls_back_if_run_completion_fails(store, prediction):
    add_observation(store)
    execution = store.start_run(started_at=COMPLETED + pd.Timedelta(seconds=1))
    with pytest.raises(sqlite3.IntegrityError, match="completed_at >= started_at"):
        complete_prediction(store, prediction, execution)

    assert rows(store, "predictions") == []
    run, = rows(store, "runs")
    assert run["status"] == "running"
    assert run["completed_at"] is None


def test_run_records_safe_error_status_and_cannot_be_completed_twice(store):
    execution = store.start_run(started_at="2026-11-28T10:00:00+07:00")
    store.complete_run(
        execution, status="api_http_error", source="dashboard", latest_observation=OBSERVATION,
        completed_at=COMPLETED, http_status=503,
    )
    run, = rows(store, "runs")
    assert run["started_at"] == "2026-11-28T03:00:00.000000Z"
    assert run["completed_at"] == "2026-11-28T03:00:01.000000Z"
    assert run["latest_observation"] == "2026-11-28T02:00:00.000000Z"
    assert run["error_category"] == "api_http_error"
    assert run["error_message"] == "The prediction API returned HTTP 503."
    assert run["prediction_id"] is None
    with pytest.raises(storage.StorageError, match="running"):
        store.complete_run(execution, status="no_observations")


def test_concurrent_initialization_and_writes_create_one_prediction(tmp_path, prediction):
    path = tmp_path / "shared.sqlite3"
    start_together = Barrier(6, timeout=10)
    write_together = Barrier(6, timeout=10)

    def worker(_):
        start_together.wait()
        db = storage.PredictionStore(path)
        add_observation(db)
        execution = db.start_run(started_at=CREATED)
        write_together.wait()
        return complete_prediction(db, prediction, execution)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker, range(6)))

    assert [result.status for result in results].count("success") == 1
    assert [result.status for result in results].count("duplicate") == 5
    assert len({result.prediction_id for result in results}) == 1
    db = storage.PredictionStore(path)
    assert len(rows(db, "observations")) == len(rows(db, "predictions")) == 1
    assert len(rows(db, "runs")) == 6


def test_inspection_is_read_only_and_does_not_create_a_missing_database(tmp_path, store, monkeypatch, capsys):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        storage.recent_records("predictions", path=missing)
    assert not missing.exists()

    add_observation(store)
    monkeypatch.setattr("sys.argv", [
        "prediction_store", "observations", "--db", str(store.path), "--limit", "1"
    ])
    storage.main()
    assert json.loads(capsys.readouterr().out)[0]["actual_demand"] == 25_500


@pytest.fixture
def live_window(store, monkeypatch):
    observations = pd.DataFrame({
        "period": pd.date_range(end=OBSERVATION, periods=25, freq="h"),
        "value": list(range(25_476, 25_501)),
    })
    observations.attrs["source"] = "dashboard"
    fetch = Mock(return_value=observations)
    post = Mock()
    monkeypatch.setattr(readiness, "fetch_ciso_observations", fetch)
    monkeypatch.setattr(online, "fetch_inference_window", lambda: readiness.fetch_inference_window(now=CREATED))
    monkeypatch.setattr(online, "_now_utc", lambda: CREATED)
    monkeypatch.setattr(online, "_prediction_url", lambda: "http://localhost:8000/predict")
    monkeypatch.setattr(online.requests, "post", post)
    original_timestamp = storage._timestamp
    monkeypatch.setattr(storage, "_timestamp", lambda value=None: original_timestamp(
        COMPLETED if value is None else value
    ))
    return observations, fetch, post


def test_online_persistence_writes_observations_before_features_and_returns_original_on_duplicate(
    store, live_window, monkeypatch
):
    _, fetch, post = live_window
    assess = online._assess_readiness

    def verify_observations_already_committed(observations, *, now):
        assert len(rows(store, "observations")) == 25
        return assess(observations, now=now)

    def predict(demand):
        response = Mock(status_code=200)
        response.json.return_value = {
            "predicted_demand": demand, "forecast_horizon_hours": 2,
            "model_uri": MODEL_URI, "model_version": "1",
        }
        return response

    post.side_effect = [predict(27_000), predict(28_000)]
    monkeypatch.setattr(online, "_assess_readiness", verify_observations_already_committed)
    first = online.run_persisted_inference(store=store)
    second = online.run_persisted_inference(store=store)

    assert first.status == "success" and second.status == "duplicate"
    assert first.prediction_id == second.prediction_id
    assert first.predicted_demand == second.predicted_demand == 27_000
    assert first.prediction_created_at == second.prediction_created_at
    assert first.execution_id != second.execution_id
    assert fetch.call_count == post.call_count == 2
    assert len(rows(store, "predictions")) == 1
    assert [run["status"] for run in rows(store, "runs")] == ["duplicate", "success"]


def test_failed_inference_preserves_observations_and_stores_only_sanitized_error(store, live_window):
    _, fetch, post = live_window
    post.side_effect = requests.ReadTimeout("request to https://example.invalid/?token=fake-secret")

    result = online.run_persisted_inference(store=store)

    assert result.status == "api_timeout"
    fetch.assert_called_once()
    assert len(rows(store, "observations")) == 25
    assert rows(store, "predictions") == []
    run, = rows(store, "runs")
    assert run["status"] == run["error_category"] == "api_timeout"
    assert run["error_message"] == "The prediction API timed out."
    assert "fake-secret" not in str(run)


@pytest.mark.parametrize("status", ["stale", "incomplete_context"])
def test_skipped_inference_still_saves_observations(store, live_window, monkeypatch, status):
    observations, fetch, post = live_window
    if status == "stale":
        monkeypatch.setattr(online, "fetch_inference_window", lambda: readiness.fetch_inference_window(now=TARGET))
        monkeypatch.setattr(online, "_now_utc", lambda: TARGET)
    else:
        fetch.return_value = observations.drop(index=10)

    result = online.run_persisted_inference(store=store)

    assert result.status == rows(store, "runs")[0]["status"] == status
    assert len(rows(store, "observations")) == (25 if status == "stale" else 24)
    assert rows(store, "predictions") == []
    fetch.assert_called_once()
    post.assert_not_called()


def test_ingestion_failure_records_a_run_without_data(store, live_window):
    _, fetch, post = live_window
    fetch.side_effect = EIATemporaryError("EIA Grid Monitor failed after 3 attempts.")

    result = online.run_persisted_inference(store=store)

    assert result.status == "ingestion_unavailable"
    assert rows(store, "runs")[0]["error_category"] == "ingestion_unavailable"
    assert rows(store, "observations") == rows(store, "predictions") == []
    post.assert_not_called()


def test_failed_prediction_commit_records_storage_error_and_keeps_observations(store, live_window):
    _, _, post = live_window
    response = Mock(status_code=200)
    response.json.return_value = {"predicted_demand": 27_000, "forecast_horizon_hours": 2}
    post.return_value = response
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("""
            CREATE TRIGGER reject_success BEFORE UPDATE ON pipeline_runs
            WHEN NEW.status = 'success'
            BEGIN SELECT RAISE(ABORT, 'fake-secret'); END
        """)

    result = online.run_persisted_inference(store=store)

    assert result.status == "storage_error"
    assert rows(store, "predictions") == []
    assert len(rows(store, "observations")) == 25
    run, = rows(store, "runs")
    assert run["status"] == run["error_category"] == "storage_error"
    assert run["data_source"] == "dashboard"
    assert run["latest_observation"] == "2026-11-28T02:00:00.000000Z"
    assert "fake-secret" not in str(run) + str(result)

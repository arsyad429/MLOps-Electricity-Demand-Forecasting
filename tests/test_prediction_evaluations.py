import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

from src.pipelines import evaluate_predictions as evaluator
from src.pipelines import run_online_inference as online
from src.storage import prediction_store as storage


DAY = datetime(2025, 1, 16, tzinfo=timezone.utc)
MODEL = "models:/CISO-Demand-Forecaster-Compose@champion"


def hour(value):
    return DAY + timedelta(hours=value)


@pytest.fixture
def store(tmp_path):
    return storage.PredictionStore(tmp_path / "predictions.sqlite3")


def record_prediction(store, *, origin=2, demand=27_000, version="1", feature_payload=None):
    observation, created, target = hour(origin), hour(origin + 1), hour(origin + 2)
    store.upsert_observations([(observation, 25_500)], source="dashboard", ingested_at=created)
    execution_id = store.start_run(started_at=created)
    return store.complete_run(
        execution_id,
        status="success",
        source="dashboard",
        latest_observation=observation,
        completed_at=created + timedelta(seconds=1),
        prediction=storage.Prediction(
            observation_timestamp=observation,
            forecast_target_timestamp=target,
            prediction_created_at=created,
            predicted_demand=demand,
            actual_input_demand=25_500,
            data_source="dashboard",
            forecast_horizon_hours=2,
            model_uri=MODEL,
            model_version=version,
            inference_latency_ms=12.0,
            feature_payload=feature_payload,
        ),
    ).prediction_id


def recent(store, kind):
    return storage.recent_records(kind, path=store.path)


def model_reports(store, as_of, *, window_hours=None):
    report = storage.evaluation_summary(path=store.path, as_of=as_of, window_hours=window_hours)
    return {model["model_version"]: model for model in report["models"]}


def test_only_the_exact_target_is_scored_and_only_after_it_is_ingested(store):
    prediction_id = record_prediction(store)
    store.upsert_observations([(hour(3), 25_800), (hour(5), 27_500)], source="dashboard", ingested_at=hour(6))

    assert store.evaluate_predictions(as_of=hour(3) + timedelta(minutes=30)) == 0
    assert model_reports(store, hour(3) + timedelta(minutes=30))["1"]["future"] == 1
    assert store.evaluate_predictions(as_of=hour(6)) == 0
    assert model_reports(store, hour(6))["1"]["pending_actual"] == 1
    assert recent(store, "evaluations") == []

    store.upsert_observations([(hour(4), 26_000)], source="api", ingested_at=hour(7))
    assert store.evaluate_predictions(as_of=hour(6)) == 0
    assert model_reports(store, hour(7))["1"]["ready_to_evaluate"] == 1
    assert store.evaluate_predictions(as_of=hour(7)) == 1
    assert store.evaluate_predictions(as_of=hour(8)) == 0

    evaluation, = recent(store, "evaluations")
    assert evaluation["prediction_id"] == prediction_id
    assert evaluation["forecast_target_timestamp"] == "2025-01-16T04:00:00.000000Z"
    assert evaluation["actual_demand"] == 26_000
    assert evaluation["actual_data_source"] == "api"
    assert evaluation["actual_revision_count"] == 0
    assert evaluation["evaluated_at"] == "2025-01-16T07:00:00.000000Z"
    assert evaluation["error_mwh"] == evaluation["absolute_error_mwh"] == 1_000
    assert model_reports(store, hour(8))["1"]["evaluated"] == 1


def test_revises_evaluation_without_mutating_the_original_forecast(store):
    record_prediction(store)
    original = recent(store, "predictions")
    store.upsert_observations([(hour(4), 26_000)], source="dashboard", ingested_at=hour(5))
    assert store.evaluate_predictions(as_of=hour(5)) == 1
    store.upsert_observations([(hour(4), 27_500)], source="dashboard", ingested_at=hour(6))

    awaiting_update = model_reports(store, hour(6))["1"]
    assert awaiting_update["ready_to_evaluate"] == 1
    assert awaiting_update["evaluated"] == 0
    assert awaiting_update["mae_mwh"] is None
    assert store.evaluate_predictions(as_of=hour(6)) == 1
    assert store.evaluate_predictions(as_of=hour(7)) == 0

    evaluation, = recent(store, "evaluations")
    revision, = recent(store, "revisions")
    assert evaluation["actual_revision_count"] == revision["revision_number"] == 1
    assert revision["previous_actual_demand"] == 26_000
    assert evaluation["actual_demand"] == 27_500
    assert evaluation["error_mwh"] == -500
    assert evaluation["absolute_error_mwh"] == 500
    assert recent(store, "predictions") == original


def test_summary_groups_model_versions_and_uses_target_time_for_window(store):
    record_prediction(store, origin=2)
    record_prediction(store, origin=3, demand=23_000)
    record_prediction(store, origin=2, demand=26_000, version="2")
    store.upsert_observations(
        [(hour(4), 26_000), (hour(5), 25_000)], source="dashboard", ingested_at=hour(5)
    )
    cutoff = hour(5) + timedelta(minutes=10)
    assert store.evaluate_predictions(as_of=cutoff) == 3

    report = model_reports(store, cutoff)
    v1 = report["1"]
    assert v1["model_name"] == "CISO-Demand-Forecaster-Compose"
    assert v1["evaluated"] == v1["predictions"] == 2
    assert v1["mae_mwh"] == 1_500
    assert v1["rmse_mwh"] == pytest.approx(math.sqrt((1_000**2 + 2_000**2) / 2))
    assert v1["wape_pct"] == pytest.approx(3_000 / 51_000 * 100)
    assert report["2"]["rmse_mwh"] == 0
    assert report["2"]["evaluated"] == 1
    filtered = model_reports(store, cutoff, window_hours=1)
    assert list(filtered) == ["1"]
    assert filtered["1"]["mae_mwh"] == 2_000


def test_zero_actual_leaves_wape_undefined_and_other_metrics_finite(store):
    record_prediction(store)
    store.upsert_observations([(hour(4), 0)], source="dashboard", ingested_at=hour(5))
    assert store.evaluate_predictions(as_of=hour(5)) == 1
    report = model_reports(store, hour(5))["1"]
    assert report["mae_mwh"] == report["rmse_mwh"] == 27_000
    assert report["wape_pct"] is None


def test_schema_migration_scores_existing_v1_forecasts(tmp_path, monkeypatch):
    path = tmp_path / "previous.sqlite3"
    with monkeypatch.context() as legacy:
        legacy.setattr(storage, "SCHEMA_VERSION", 1)
        old_store = storage.PredictionStore(path)
        prediction_id = record_prediction(old_store)
        old_store.upsert_observations([(hour(4), 26_250)], source="dashboard", ingested_at=hour(5))
        original_forecast = recent(old_store, "predictions")
        original_run = recent(old_store, "runs")

    new_store = storage.PredictionStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert new_store.evaluate_predictions(as_of=hour(6)) == 1
    assert recent(new_store, "evaluations")[0]["prediction_id"] == prediction_id
    assert recent(new_store, "predictions") == original_forecast
    assert recent(new_store, "runs") == original_run


def test_v2_migration_preserves_historical_records_and_scores_legacy_and_new_forecasts(tmp_path, monkeypatch):
    path = tmp_path / "schema-v2.sqlite3"
    cutoff = hour(7)
    with monkeypatch.context() as legacy:
        legacy.setattr(storage, "SCHEMA_VERSION", 2)
        old_store = storage.PredictionStore(path)
        legacy_id = record_prediction(old_store)
        old_store.upsert_observations([(hour(4), 26_000)], source="dashboard", ingested_at=hour(5))
        assert old_store.evaluate_predictions(as_of=cutoff) == 1
        saved = {kind: recent(old_store, kind) for kind in ("observations", "predictions", "evaluations")}
        before = model_reports(old_store, cutoff)
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 2

    migrated = storage.PredictionStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {kind: recent(migrated, kind) for kind in saved} == saved
    assert model_reports(migrated, cutoff) == before
    assert recent(migrated, "features") == []

    payload = {
        "value": 25_500.0, "lag_1h": 25_400.0, "lag_2h": 25_300.0, "lag_24h": 25_200.0,
        "rolling_mean_3h": 25_400.0, "rolling_mean_24h": 25_300.0,
        "hour": 18, "day_of_week": 2, "month": 1, "is_weekend": 0,
    }
    assert record_prediction(migrated, feature_payload=payload) == legacy_id
    assert recent(migrated, "features") == []
    new_id = record_prediction(migrated, origin=3, demand=23_000, feature_payload=dict(payload, hour=19))
    migrated.upsert_observations([(hour(5), 25_000)], source="dashboard", ingested_at=hour(6))
    assert migrated.evaluate_predictions(as_of=cutoff) == 1
    assert {row["prediction_id"] for row in recent(migrated, "evaluations")} == {legacy_id, new_id}
    snapshot, = recent(migrated, "features")
    assert snapshot["prediction_id"] == new_id

    model = model_reports(migrated, cutoff)["1"]
    assert model["evaluated"] == 2
    assert model["mae_mwh"] == 1_500
    assert model["rmse_mwh"] == pytest.approx(math.sqrt((1_000**2 + 2_000**2) / 2))
    assert model["wape_pct"] == pytest.approx(3_000 / 51_000 * 100)


def test_evaluation_cli_summary_is_read_only_and_backfill_writes(store, capsys):
    record_prediction(store)
    store.upsert_observations([(hour(4), 26_000)], source="dashboard", ingested_at=hour(5))

    assert evaluator.main(["--db", str(store.path), "--summary-only"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["evaluations_updated"] is None
    assert stats["models"][0]["ready_to_evaluate"] == 1
    assert recent(store, "evaluations") == []

    assert evaluator.main(["--db", str(store.path)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["evaluations_updated"] == 1
    assert stats["models"][0]["evaluated"] == 1


def test_online_cycle_scores_mature_forecast_even_if_new_inference_is_skipped(store, monkeypatch):
    prediction_id = record_prediction(store)
    frame = pd.DataFrame({"period": [pd.Timestamp(hour(4))], "value": [26_000.0]})
    frame.attrs["source"] = "dashboard"
    fetch = Mock(return_value=(frame, pd.Timestamp(hour(5))))
    post = Mock()
    monkeypatch.setattr(online, "fetch_inference_window", fetch)
    monkeypatch.setattr(online, "_now_utc", lambda: pd.Timestamp(hour(5)))
    monkeypatch.setattr(online.requests, "post", post)

    result = online.run_persisted_inference(store=store)

    assert result.status == "incomplete_context"
    assert result.evaluations_updated == 1
    assert recent(store, "evaluations")[0]["prediction_id"] == prediction_id
    assert len(recent(store, "predictions")) == 1
    fetch.assert_called_once_with()
    post.assert_not_called()


def test_evaluation_error_does_not_change_the_inference_status(store, monkeypatch, caplog):
    frame = pd.DataFrame({"period": [], "value": []})
    frame.attrs["source"] = "dashboard"
    monkeypatch.setattr(online, "fetch_inference_window", lambda: (frame, pd.Timestamp(hour(5))))
    error = sqlite3.OperationalError("https://example.invalid/?token=fake-secret")
    monkeypatch.setattr(store, "evaluate_predictions", Mock(side_effect=error))

    result = online.run_persisted_inference(store=store)

    assert result.status == recent(store, "runs")[0]["status"] == "no_observations"
    assert result.evaluations_updated is None
    assert "Delayed evaluation could not complete (OperationalError)" in caplog.text
    assert "fake-secret" not in caplog.text

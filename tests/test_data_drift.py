import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from scripts import demo_data_drift as demo
from src.monitoring import drift_metrics
from src.monitoring.drift import DriftStatus, ModelIdentity, build_reference, evaluate_drift, load_reference
from src.monitoring.production_metrics import ProductionMetricsCollector
from src.storage.prediction_store import inference_feature_rows


MODEL = ModelIdentity("CISO-Test", "1")


def demand_records(low, high):
    return [(index, {"value": value}) for index, value in enumerate([10.0] * low + [20.0] * high)]


@pytest.mark.parametrize(("low", "high", "expected"), [
    (50, 50, DriftStatus.STABLE),
    (68, 32, DriftStatus.MODERATE_DRIFT),
    (90, 10, DriftStatus.SIGNIFICANT_DRIFT),
    (49, 50, DriftStatus.INSUFFICIENT_DATA),
])
def test_psi_classifies_known_distributions(low, high, expected):
    reference = build_reference(MODEL, {"value": [10.0] * 100 + [20.0] * 100})
    result = evaluate_drift(MODEL, reference, demand_records(low, high))

    assert result.status == expected
    assert result.production_samples == low + high
    if expected == DriftStatus.STABLE:
        assert result.features["value"].psi == pytest.approx(0)
    if expected == DriftStatus.INSUFFICIENT_DATA:
        assert result.features["value"].psi is None


def test_empty_bins_and_values_beyond_training_bounds_still_score():
    reference = build_reference(MODEL, {"value": list(range(10, 110))})
    result = evaluate_drift(MODEL, reference, [(hour, {"value": 500}) for hour in range(100)])

    feature = result.features["value"]
    assert feature.production_samples == 100
    assert feature.psi > 0.25
    assert result.status == DriftStatus.SIGNIFICANT_DRIFT


def test_constant_bins_detect_departure_and_allow_matching_constant():
    reference = build_reference(MODEL, {"value": [10.0] * 200})
    same = evaluate_drift(MODEL, reference, demand_records(100, 0))
    changed = evaluate_drift(MODEL, reference, demand_records(0, 100))

    assert same.features["value"].psi == pytest.approx(0)
    assert changed.status == DriftStatus.SIGNIFICANT_DRIFT
    assert changed.features["value"].psi > 0.25


def test_repeated_decile_boundaries_are_removed():
    data = [10.0] * 90 + list(range(11, 41))
    reference = build_reference(MODEL, {"value": data})
    boundaries = reference.features["value"].boundaries
    result = evaluate_drift(MODEL, reference, [(index, {"value": value}) for index, value in enumerate(data)])

    assert 0 < len(boundaries) < 9
    assert all(left < right for left, right in zip(boundaries, boundaries[1:]))
    assert result.features["value"].psi == pytest.approx(0)


def test_empty_reference_and_empty_production():
    with pytest.raises(ValueError, match="positive number"):
        build_reference(MODEL, {"value": []})
    reference = build_reference(MODEL, {"value": [10.0] * 100})
    report = evaluate_drift(MODEL, reference, [])
    assert report.production_samples == 0
    assert report.status == DriftStatus.INSUFFICIENT_DATA
    assert report.features["value"].psi is None


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), True, "10"])
def test_partial_feature_coverage_never_yields_overall_stable(invalid):
    reference = build_reference(MODEL, {"value": [10.0] * 100, "lag_1h": [20.0] * 100})
    production = [(t, {"value": 10.0, "lag_1h": invalid}) for t in range(100)]
    stable_feature = evaluate_drift(MODEL, reference, production)
    drifting = evaluate_drift(MODEL, reference, [(t, dict(payload, value=100.0)) for t, payload in production])

    assert stable_feature.production_samples == 100
    assert stable_feature.features["value"].status == DriftStatus.STABLE
    assert stable_feature.features["lag_1h"].status == DriftStatus.INSUFFICIENT_DATA
    assert stable_feature.features["lag_1h"].psi is None
    assert stable_feature.status == DriftStatus.INSUFFICIENT_DATA
    assert drifting.status == DriftStatus.SIGNIFICANT_DRIFT
    assert drifting.features["lag_1h"].psi is None


def test_replicated_predictions_and_incompatible_payloads_do_not_count():
    reference = build_reference(MODEL, {"value": [10.0] * 100})
    rows = [(1, {"value": 10.0})] * 1_000 + [(2, {"other": 10.0})] * 1_000
    result = evaluate_drift(MODEL, reference, rows)

    assert result.production_samples == 1
    assert result.invalid_payloads == 1
    assert result.features["value"].production_samples == 1
    assert result.status == DriftStatus.INSUFFICIENT_DATA


def test_reference_rejects_different_model_version_or_feature_order(tmp_path):
    doc = build_reference(MODEL, {"value": [10.0] * 100, "hour": [0] * 100}).to_dict()
    path = tmp_path / MODEL.reference_filename
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_reference(path, ModelIdentity(MODEL.name, "2"))
    doc["feature_names"].reverse()
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_reference(path, MODEL)


def test_sqlite_reader_filters_requested_version_and_recent_observation_origins(tmp_path):
    now = datetime(2026, 11, 28, 6, 30, tzinfo=timezone.utc)
    path = demo.prepare_demo(tmp_path, "stable", now=now)
    args = {
        "path": path, "model_name": demo.MODEL.name, "model_version": "1", "window_hours": 48,
        "as_of": now,
    }
    rows = inference_feature_rows(**args)
    assert len(rows) == 47
    assert len({row["observation_timestamp"] for row in rows}) == 47
    assert inference_feature_rows(**dict(args, model_version="2")) == []
    assert inference_feature_rows(**dict(args, model_name="CISO-Other")) == []


@pytest.mark.parametrize(("scenario", "status", "count"), [
    ("stable", 0, 100),
    ("significant", 2, 100),
    ("insufficient", -1, 40),
])
def test_end_to_end_exporter_uses_sqlite_payloads_and_suppresses_missing_scores(tmp_path, monkeypatch, scenario, status, count):
    path = demo.prepare_demo(tmp_path, scenario)
    config = drift_metrics.DriftConfig(reference_dir=tmp_path)
    monkeypatch.setattr(drift_metrics, "loaded_model", lambda _url: demo.MODEL)
    registry = CollectorRegistry()
    registry.register(ProductionMetricsCollector(drift=drift_metrics.DriftCollector(config, path=path), path=path))
    labels = {"model_name": demo.MODEL.name, "model_version": demo.MODEL.version}

    assert registry.get_sample_value("ciso_drift_status", labels) == status
    assert registry.get_sample_value("ciso_drift_production_samples", labels) == count
    assert registry.get_sample_value("ciso_production_db_read_success") == 1
    assert registry.get_sample_value("ciso_production_latest_predicted_demand_mwh", labels) is not None
    value = registry.get_sample_value("ciso_feature_psi", {**labels, "feature": "value"})
    if status == -1:
        assert value is None
        assert registry.get_sample_value("ciso_drift_max_psi", labels) is None
        assert b"ciso_drifted_features{" not in generate_latest(registry)
    elif status == 2:
        assert value >= 0.25
    else:
        assert value == pytest.approx(0)


def test_promotion_refreshes_cache_and_removes_previous_model_scores(tmp_path, monkeypatch):
    path = demo.prepare_demo(tmp_path, "significant")
    config = drift_metrics.DriftConfig(reference_dir=tmp_path, refresh_seconds=30)
    old_model = demo.MODEL
    next_model = replace(old_model, version="2")
    current = [old_model]
    clock = [100]
    monkeypatch.setattr(drift_metrics, "loaded_model", lambda _url: current[0])
    monkeypatch.setattr(drift_metrics.time, "monotonic", lambda: clock[0])
    registry = CollectorRegistry()
    registry.register(drift_metrics.DriftCollector(config, path=path))
    old_labels = {"model_name": old_model.name, "model_version": old_model.version}
    next_labels = {"model_name": next_model.name, "model_version": next_model.version}

    assert registry.get_sample_value("ciso_drift_max_psi", old_labels) > 0.25
    current[0] = next_model
    assert registry.get_sample_value("ciso_drift_max_psi", old_labels) > 0.25
    clock[0] += config.refresh_seconds
    assert registry.get_sample_value("ciso_drift_max_psi", old_labels) is None
    assert registry.get_sample_value("ciso_drift_status", next_labels) == -1
    assert registry.get_sample_value("ciso_drift_baseline_available", next_labels) == 0

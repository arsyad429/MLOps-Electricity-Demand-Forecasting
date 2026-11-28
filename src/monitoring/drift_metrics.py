"""Read the active model and cache its SQLite PSI report for Prometheus."""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from urllib.request import urlopen

from prometheus_client.core import GaugeMetricFamily

from src.monitoring.drift import DriftResult, DriftStatus, ModelIdentity, evaluate_drift, load_reference
from src.storage.prediction_store import inference_feature_rows


LOGGER = logging.getLogger(__name__)
REFERENCE_DIR = Path(__file__).with_name("references")
MODEL_LABELS = ["model_name", "model_version"]


@dataclass(frozen=True)
class DriftConfig:
    window_hours: int = 168
    minimum: int = 100
    moderate: float = 0.10
    significant: float = 0.25
    refresh_seconds: int = 60
    reference_dir: Path = REFERENCE_DIR
    model_check_url: str = "http://127.0.0.1:8000/mlflow-check"

    def __post_init__(self):
        if self.window_hours < 1 or self.refresh_seconds < 1:
            raise ValueError("DRIFT_WINDOW_HOURS and DRIFT_REFRESH_SECONDS must be positive.")
        evaluate_drift(None, None, (), minimum=self.minimum, moderate=self.moderate, significant=self.significant)

    @classmethod
    def from_env(cls):
        return cls(
            window_hours=int(os.getenv("DRIFT_WINDOW_HOURS", "168")),
            minimum=int(os.getenv("DRIFT_MIN_PRODUCTION_SAMPLES", "100")),
            moderate=float(os.getenv("DRIFT_MODERATE_THRESHOLD", "0.10")),
            significant=float(os.getenv("DRIFT_SIGNIFICANT_THRESHOLD", "0.25")),
            refresh_seconds=int(os.getenv("DRIFT_REFRESH_SECONDS", "60")),
            reference_dir=Path(os.getenv("DRIFT_REFERENCE_DIR", str(REFERENCE_DIR))),
            model_check_url=os.getenv("DRIFT_MODEL_CHECK_URL", cls.model_check_url),
        )


@dataclass(frozen=True)
class DriftSnapshot:
    model: ModelIdentity | None
    result: DriftResult
    evaluated_at: datetime
    sqlite_read: bool = False


def loaded_model(url):
    with urlopen(url, timeout=3) as response:
        body = json.loads(response.read(65_537))
    if type(body) is not dict or type(body.get("registered_model")) is not str:
        raise ValueError("Model check did not identify the registered model.")
    return ModelIdentity(body["registered_model"], str(body.get("loaded_model_version")))


def _decode_rows(rows):
    for row in rows:
        try:
            payload = json.loads(row["feature_payload"])
        except (ValueError, TypeError):
            payload = None
        yield row["observation_timestamp"], payload


class DriftCollector:
    def __init__(self, config=None, *, path=None):
        self.config = DriftConfig.from_env() if config is None else config
        self.path = path
        self._lock = Lock()
        self._snapshot = None
        self._refresh_at = 0.0

    def _evaluate(self):
        now = datetime.now(timezone.utc)
        config = self.config
        try:
            model = loaded_model(config.model_check_url)
        except (OSError, ValueError, HTTPException) as exc:
            LOGGER.warning("Active model unavailable for drift monitoring (%s).", type(exc).__name__)
            return DriftSnapshot(None, evaluate_drift(None, None, ()), now)

        try:
            reference = load_reference(config.reference_dir / model.reference_filename, model)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Drift reference unavailable for %s v%s (%s).", model.name, model.version, type(exc).__name__)
            return DriftSnapshot(model, evaluate_drift(model, None, ()), now)

        try:
            rows = inference_feature_rows(
                model_name=model.name, model_version=model.version, path=self.path,
                as_of=now, window_hours=config.window_hours,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            LOGGER.warning("SQLite features unavailable for drift monitoring (%s).", type(exc).__name__)
            return DriftSnapshot(model, evaluate_drift(model, reference, ()), now)

        report = evaluate_drift(
            model, reference, _decode_rows(rows), minimum=config.minimum,
            moderate=config.moderate, significant=config.significant,
        )
        return DriftSnapshot(model, report, now, sqlite_read=True)

    def snapshot(self):
        """Refresh at most once per interval, including when the current report is unavailable."""
        with self._lock:
            if self._snapshot is None or time.monotonic() >= self._refresh_at:
                self._snapshot = self._evaluate()
                self._refresh_at = time.monotonic() + self.config.refresh_seconds
            return self._snapshot

    def collect(self):
        snapshot = self.snapshot()
        labels = ([snapshot.model.name, snapshot.model.version] if snapshot.model else ["unknown", "unknown"])
        result = snapshot.result
        scored = [feature for feature in result.features.values() if feature.psi is not None]
        gauges = (
            ("status", "Overall drift status (-1 insufficient, 0 stable, 1 moderate, 2 significant)", result.status),
            ("reference_samples", "Training observations in the active PSI reference", result.reference_samples),
            ("production_samples", "Unique production origins with the reference feature schema", result.production_samples),
            ("min_production_samples", "Minimum finite production feature values required for PSI", self.config.minimum),
            ("moderate_threshold", "PSI cutoff for moderate drift", self.config.moderate),
            ("significant_threshold", "PSI cutoff for significant drift", self.config.significant),
            ("evaluated_features", "Number of features with a current PSI score", len(scored)),
            ("invalid_payloads", "Unique production payloads rejected for a schema mismatch", result.invalid_payloads),
            ("baseline_available", "Whether the loaded model has a valid PSI reference", bool(result.reference_samples)),
            ("sqlite_read_success", "Whether feature reads completed for the active PSI reference", snapshot.sqlite_read),
            ("evaluation_timestamp_seconds", "Unix time of the latest drift check, including insufficient data",
             snapshot.evaluated_at.timestamp()),
        )
        for name, help_text, value in gauges:
            metric = GaugeMetricFamily(f"ciso_drift_{name}", help_text, labels=MODEL_LABELS)
            metric.add_metric(labels, float(value))
            yield metric

        statuses = GaugeMetricFamily("ciso_feature_drift_status", "Drift status by feature", labels=[*MODEL_LABELS, "feature"])
        counts = GaugeMetricFamily(
            "ciso_feature_drift_production_samples", "Finite production values per feature", labels=[*MODEL_LABELS, "feature"],
        )
        for name, feature in result.features.items():
            statuses.add_metric([*labels, name], float(feature.status))
            counts.add_metric([*labels, name], feature.production_samples)
        yield statuses
        yield counts

        if not scored:
            return
        psi = GaugeMetricFamily("ciso_feature_psi", "Population Stability Index by feature", labels=[*MODEL_LABELS, "feature"])
        for name, feature in result.features.items():
            if feature.psi is not None:
                psi.add_metric([*labels, name], feature.psi)
        yield psi
        for name, help_text, value in (
            ("ciso_drifted_features", "Evaluated features with moderate or significant drift",
             sum(feature.status >= DriftStatus.MODERATE_DRIFT for feature in scored)),
            ("ciso_significant_drift_features", "Evaluated features with significant drift",
             sum(feature.status == DriftStatus.SIGNIFICANT_DRIFT for feature in scored)),
            ("ciso_drift_max_psi", "Maximum current PSI among evaluated features", max(feature.psi for feature in scored)),
        ):
            metric = GaugeMetricFamily(name, help_text, labels=MODEL_LABELS)
            metric.add_metric(labels, value)
            yield metric

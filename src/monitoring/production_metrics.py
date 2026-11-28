
"""Expose production CISO forecast metrics from SQLite."""

import logging
import os
import time

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily

from src.monitoring.drift_metrics import DriftCollector
from src.storage.prediction_store import (
    evaluation_summary,
    recent_records,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionMetricsCollector:
    def __init__(self, *, drift=None, path=None):
        self.path = path
        self.drift = DriftCollector(path=path) if drift is None else drift

    def collect(self):
        yield from self._performance_metrics()
        yield from self.drift.collect()

    def _performance_metrics(self):
        try:
            summary = evaluation_summary(path=self.path, window_hours=168)
            recent = recent_records("predictions", path=self.path, limit=1)
        except Exception as exc:
            logger.warning(
                "SQLite metrics unavailable: %s",
                type(exc).__name__,
            )
            yield GaugeMetricFamily(
                "ciso_production_db_read_success",
                "Whether SQLite metrics were read successfully",
                value=0,
            )
            return

        yield GaugeMetricFamily(
            "ciso_production_db_read_success",
            "Whether SQLite metrics were read successfully",
            value=1,
        )

        definitions = {
            "predictions": (
                "ciso_production_predictions",
                "Production predictions in the last 7 days",
            ),
            "evaluated": (
                "ciso_production_evaluated",
                "Evaluated predictions in the last 7 days",
            ),
            "pending_actual": (
                "ciso_production_pending_actual",
                "Predictions awaiting actual demand",
            ),
            "future": (
                "ciso_production_future",
                "Predictions whose target time is in the future",
            ),
            "mae_mwh": (
                "ciso_production_mae_mwh",
                "Seven-day production MAE in MWh",
            ),
            "rmse_mwh": (
                "ciso_production_rmse_mwh",
                "Seven-day production RMSE in MWh",
            ),
            "wape_pct": (
                "ciso_production_wape_percent",
                "Seven-day production WAPE percentage",
            ),
        }

        families = {
            key: GaugeMetricFamily(
                name,
                description,
                labels=["model_name", "model_version"],
            )
            for key, (name, description) in definitions.items()
        }

        for model in summary["models"]:
            labels = [
                model["model_name"] or "unknown",
                str(model["model_version"] or "unknown"),
            ]

            for key, metric in families.items():
                value = model.get(key)

                if value is not None:
                    metric.add_metric(labels, float(value))

        yield from families.values()

        latest = GaugeMetricFamily(
            "ciso_production_latest_predicted_demand_mwh",
            "Latest stored production prediction in MWh",
            labels=["model_name", "model_version"],
        )

        if recent:
            record = recent[0]
            latest.add_metric(
                [
                    record["model_name"] or "unknown",
                    str(record["model_version"] or "unknown"),
                ],
                float(record["predicted_demand"]),
            )

        yield latest


def main():
    port = int(os.getenv("METRICS_PORT", "9108"))

    registry = CollectorRegistry(auto_describe=False)
    registry.register(ProductionMetricsCollector())

    start_http_server(
        port,
        addr="0.0.0.0",
        registry=registry,
    )

    logger.info(
        "Production metrics exporter listening on port %d",
        port,
    )

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

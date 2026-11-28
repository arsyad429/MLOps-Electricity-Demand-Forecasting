"""Demonstrate PSI with temporary SQLite databases and a loopback model check."""

import json
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from src.features.feature_engineering import FEATURE_COLUMNS
from src.monitoring.drift import ModelIdentity, build_reference
from src.monitoring.drift_metrics import DriftCollector, DriftConfig
from src.storage.prediction_store import Prediction, PredictionStore


MODEL = ModelIdentity("CISO-Drift-Demo", "1")
SCENARIOS = {"stable": (50, 50), "significant": (90, 10), "insufficient": (20, 20)}


def payload(demand):
    return dict(zip(FEATURE_COLUMNS, [demand, 30.0, 30.0, 30.0, 30.0, 30.0, 12, 1, 1, 0]))


def prepare_demo(directory, scenario, *, now=None):
    directory.mkdir(parents=True, exist_ok=True)
    lower, upper = payload(10.0), payload(20.0)
    reference = build_reference(MODEL, {
        name: [lower[name]] * 100 + [upper[name]] * 100 for name in FEATURE_COLUMNS
    })
    path = directory / "demo.sqlite3"
    with (directory / MODEL.reference_filename).open("x", encoding="utf-8") as file:
        json.dump(reference.to_dict(), file, allow_nan=False)
    store = PredictionStore(path)
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    low, high = SCENARIOS[scenario]
    for index, demand in enumerate([10.0] * low + [20.0] * high, start=1):
        origin = now - timedelta(hours=index)
        created = origin + timedelta(minutes=30)
        store.upsert_observations([(origin, demand)], source="dashboard", ingested_at=created)
        execution_id = store.start_run(started_at=created)
        prediction = Prediction(
            origin, origin + timedelta(hours=2), created, demand, demand, "dashboard", 2,
            f"models:/{MODEL.name}/{MODEL.version}", MODEL.version, 1.0, payload(demand),
        )
        store.complete_run(
            execution_id, status="success", source="dashboard", latest_observation=origin,
            completed_at=created, prediction=prediction,
        )
    return path


class DemoModelCheck(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"registered_model": MODEL.name, "loaded_model_version": MODEL.version}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    with (
        TemporaryDirectory(prefix="ciso-drift-demo-") as directory,
        ThreadingHTTPServer(("127.0.0.1", 0), DemoModelCheck) as server,
    ):
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for scenario in SCENARIOS:
                root = Path(directory) / scenario
                path = prepare_demo(root, scenario)
                config = DriftConfig(reference_dir=root, model_check_url=f"http://127.0.0.1:{server.server_port}")
                result = DriftCollector(config, path=path).snapshot().result
                print(json.dumps({
                    "scenario": scenario, "status": result.status.name, "samples": result.production_samples,
                    "value_psi": result.features["value"].psi,
                }))
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()

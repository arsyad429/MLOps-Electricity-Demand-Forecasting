
import argparse
import json
import math
import os
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "ci_output"

CANDIDATE_PATH = OUTPUT_DIR / "candidate.json"
GATE_PATH = OUTPUT_DIR / "evaluation.json"
THRESHOLD_PATH = ROOT / "config/ci_thresholds.yaml"

MODEL_NAME = "CISO-Demand-Forecaster-CI"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def evaluate():
    candidate = load_json(CANDIDATE_PATH)

    config = yaml.safe_load(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )

    thresholds = config["evaluation"]
    checks = {}

    for metric, threshold_key in (
        ("rmse", "max_val_rmse"),
        ("wape", "max_val_wape"),
    ):
        maximum = thresholds.get(threshold_key)

        if maximum is None:
            continue

        actual = float(candidate["val_metrics"][metric])
        maximum = float(maximum)

        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError(
                f"Threshold tidak valid: {threshold_key}"
            )

        checks[metric] = {
            "actual": actual,
            "threshold": maximum,
            "passed": (
                math.isfinite(actual)
                and actual <= maximum
            ),
        }

    if not checks:
        raise ValueError(
            "Belum ada threshold LK-01 yang diisi "
            "pada config/ci_thresholds.yaml"
        )

    passed = all(
        item["passed"]
        for item in checks.values()
    )

    report = {
        "passed": passed,
        "run_id": candidate["run_id"],
        "git_sha": candidate["git_sha"],
        "checks": checks,
    }

    save_json(GATE_PATH, report)

    print(json.dumps(report, indent=2))

    if not passed:
        raise SystemExit(
            "EVALUATION FAILED: model tidak lolos threshold"
        )

    print("EVALUATION PASSED")


def register():
    candidate = load_json(CANDIDATE_PATH)
    evaluation = load_json(GATE_PATH)

    if not evaluation["passed"]:
        raise ValueError("Evaluation belum PASS")

    if evaluation["run_id"] != candidate["run_id"]:
        raise ValueError("Run ID candidate tidak cocok")

    if evaluation["git_sha"] != candidate["git_sha"]:
        raise ValueError("Git commit candidate tidak cocok")

    expected_sha = os.environ.get("GITHUB_SHA")
    if expected_sha and candidate["git_sha"] != expected_sha:
        raise ValueError("Candidate berasal dari commit lain")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI belum tersedia")

    mlflow.set_tracking_uri(tracking_uri)

    # Registrasi HANYA setelah evaluation PASS.
    version = mlflow.register_model(
        model_uri=candidate["model_uri"],
        name=MODEL_NAME,
        tags={
            "algorithm": candidate["algorithm"],
            "dataset_dvc_md5": candidate["dataset_dvc_md5"],
            "git_sha": candidate["git_sha"],
            "validation_rmse": str(
                candidate["val_metrics"]["rmse"]
            ),
            "validation_status": "passed",
        },
    )

    client = MlflowClient()

    # Stage legacy, sesuai persyaratan tugas.
    staged = client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # Alias modern untuk MLflow.
    client.set_registered_model_alias(
        MODEL_NAME,
        "staging",
        version.version,
    )

    report = {
        "model_name": MODEL_NAME,
        "version": str(version.version),
        "stage": staged.current_stage,
        "alias": "staging",
        "run_id": candidate["run_id"],
        "git_sha": candidate["git_sha"],
        "dataset_dvc_md5": candidate["dataset_dvc_md5"],
    }

    save_json(
        OUTPUT_DIR / "registration.json",
        report,
    )

    print("MODEL REGISTERED")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["evaluate", "register"],
    )

    args = parser.parse_args()

    if args.action == "evaluate":
        evaluate()
    else:
        register()

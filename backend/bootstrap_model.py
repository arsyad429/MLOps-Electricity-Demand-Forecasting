"""One-time import from DagsHub registry into the persistent Compose MLflow registry.

Only imports again with --force. Run before the FastAPI server starts.
"""

import argparse
import os

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature


LOCAL_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
SOURCE_TRACKING_URI = os.environ.get("SOURCE_MLFLOW_URI", "")
SOURCE_MODEL_URI = os.environ.get("SOURCE_MODEL_URI", "models:/CISO-Demand-Forecaster-CI@staging")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "CISO-Demand-Forecaster-Compose")


def example_features():
    """Synthetic feature values for *signature*, not model performance testing."""
    return pd.DataFrame([{
        "value": 25000.0,
        "lag_1h": 24900.0,
        "lag_2h": 24800.0,
        "lag_24h": 24300.0,
        "rolling_mean_3h": 24900.0,
        "rolling_mean_24h": 24400.0,
        "hour": 13,
        "day_of_week": 1,
        "month": 6,
        "is_weekend": 0,
    }])


def import_model(force=False):
    local_client = MlflowClient(tracking_uri=LOCAL_TRACKING_URI)
    if not force:
        try:
            existing = local_client.get_model_version_by_alias(LOCAL_MODEL_NAME, "champion")
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
        else:
            print(f"[bootstrap] Local champion v{existing.version} already exists. Skipping import.", flush=True)
            return

    username = os.environ.get("DAGSHUB_USERNAME", "").strip()
    token = os.environ.get("DAGSHUB_TOKEN", "").strip()
    if not SOURCE_TRACKING_URI or not username or not token:
        raise RuntimeError("SOURCE_MLFLOW_URI, DAGSHUB_USERNAME and DAGSHUB_TOKEN must be configured in .env.compose.")

    # These credentials are only needed for the external DagsHub download.
    old_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    old_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(SOURCE_TRACKING_URI)

        source_name = SOURCE_MODEL_URI.split("models:/", 1)[-1].split("@", 1)[0]
        source_alias = SOURCE_MODEL_URI.split("@", 1)[-1]
        remote_client = MlflowClient(tracking_uri=SOURCE_TRACKING_URI)
        source_version = remote_client.get_model_version_by_alias(source_name, source_alias)
        algorithm = source_version.tags.get("algorithm", "xgboost").lower()
        print(f"[bootstrap] Importing {source_name} v{source_version.version} ({algorithm}) from DagsHub.", flush=True)

        if algorithm == "xgboost":
            model = mlflow.xgboost.load_model(SOURCE_MODEL_URI)
        elif algorithm == "lightgbm":
            model = mlflow.lightgbm.load_model(SOURCE_MODEL_URI)
        else:
            raise ValueError(f"Unsupported registry algorithm: {algorithm}")
    finally:
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
        if old_username is None:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = old_username
        if old_password is None:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = old_password

    example = example_features()
    signature = infer_signature(example, model.predict(example))
    mlflow.set_experiment("CISO_Compose_Model_Import")
    with mlflow.start_run(run_name="Import_DagsHub_Model"):
        mlflow.set_tags({
            "source_registry": SOURCE_TRACKING_URI,
            "source_model_uri": SOURCE_MODEL_URI,
            "source_model_version": str(source_version.version),
            "algorithm": algorithm,
            "role": "compose_seed",
        })
        if algorithm == "xgboost":
            info = mlflow.xgboost.log_model(
                xgb_model=model, name="model", signature=signature, input_example=example
            )
        else:
            info = mlflow.lightgbm.log_model(
                lgb_model=model, name="model", signature=signature, input_example=example
            )
        v = mlflow.register_model(
            model_uri=info.model_uri,
            name=LOCAL_MODEL_NAME,
            tags={
                "algorithm": algorithm,
                "source_model_version": str(source_version.version),
                "source_model_uri": SOURCE_MODEL_URI,
            },
        )
        local_client.set_registered_model_alias(LOCAL_MODEL_NAME, "champion", v.version)
        print(f"[bootstrap] Registered local {LOCAL_MODEL_NAME} v{v.version} @champion", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-import model and move local champion alias")
    args = parser.parse_args()
    import_model(force=args.force)

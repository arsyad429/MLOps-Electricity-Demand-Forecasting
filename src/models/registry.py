import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.models.train import (
    DATA_PATH,
    DVC_PATH,
    FEATURE_COLUMNS,
    TARGET,
    build_model,
    calculate_metrics,
    load_dataset,
    split_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAME = "CISO-Demand-Forecaster"
SOURCE_VERSION = "3"  # LK-06 XGBoost champion seen in Registry
NEW_NAME = "CISO-Demand-Forecaster-LK07"
TRACKING_URI_DEFAULT = "http://127.0.0.1:5000"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/lk07/model_lineage.json"


def get_data_hash():
    with DVC_PATH.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)
    return meta["outs"][0]["md5"]


def check_source(client):
    version = client.get_model_version(SOURCE_NAME, SOURCE_VERSION)
    expected = get_data_hash()
    registered_hash = version.tags.get("dataset_dvc_md5")
    if registered_hash and registered_hash != expected:
        raise ValueError(
            "Hash DVC berbeda dari model LK-06. "
            f"Registry={registered_hash}, workspace={expected}. "
            "Pulihkan dataset yang sesuai terlebih dahulu."
        )
    if not version.run_id:
        raise ValueError("Source model version tidak memiliki MLflow run_id.")
    return version


def require_empty_registry(client):
    versions = list(client.search_model_versions(f"name = '{NEW_NAME}'"))
    if versions:
        raise RuntimeError(
            f"{NEW_NAME} sudah memiliki versi. Jangan registrasi ulang "
            "karena nomor v1/v2 akan bertambah. Jalankan 'status'."
        )


def register_v1(client):
    """Reuse the exact existing LK-06 champion artifact, no retraining."""
    require_empty_registry(client)
    source = check_source(client)
    uri = f"models:/{SOURCE_NAME}/{SOURCE_VERSION}"
    mv = client.copy_model_version(src_model_uri=uri, dst_name=NEW_NAME)
    client.set_model_version_tag(NEW_NAME, mv.version, "algorithm", "xgboost")
    client.set_model_version_tag(
        NEW_NAME, mv.version, "dataset_dvc_md5", get_data_hash()
    )
    client.set_model_version_tag(NEW_NAME, mv.version, "lk07_origin", uri)
    client.set_model_version_tag(NEW_NAME, mv.version, "lifecycle_status", "None")
    client.update_model_version(
        NEW_NAME, mv.version,
        "LK-07 v1: exact copy of the LK-06 XGBoost champion v3 artifact.",
    )
    print(f"Registered {NEW_NAME} v{mv.version}, sourced from {uri}")
    print(f"Initial stage: {client.get_model_version(NEW_NAME, mv.version).current_stage}")


def get_updated_params(source_run):
    """Change just one hyperparameter relative to the champion."""
    p = source_run.data.params
    needed = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "reg_lambda",
    ]
    missing = [key for key in needed if key not in p]
    if missing:
        raise ValueError(f"Missing champion hyperparameters: {missing}")

    before_estimators = int(p["n_estimators"])
    if before_estimators <= 100:
        after_estimators = before_estimators + 50
    else:
        after_estimators = before_estimators - 50

    return {
        "n_estimators": after_estimators,
        "max_depth": int(p["max_depth"]),
        "learning_rate": float(p["learning_rate"]),
        "subsample": float(p["subsample"]),
        "colsample_bytree": float(p["colsample_bytree"]),
        "reg_lambda": float(p["reg_lambda"]),
    }, before_estimators


def register_v2(client):
    """Retrain champion with modified n_estimators, record new model version."""
    v1 = client.get_model_version(NEW_NAME, "1")
    if v1.version != "1":
        raise RuntimeError("LK-07 v1 belum terdaftar.")
    if list(client.search_model_versions(f"name = '{NEW_NAME}'")) and any(
        int(v.version) >= 2
        for v in client.search_model_versions(f"name = '{NEW_NAME}'")
    ):
        raise RuntimeError("LK-07 v2 sudah ada. Jalankan 'status'.")

    source = check_source(client)
    source_run = client.get_run(source.run_id)
    params, old_estimators = get_updated_params(source_run)
    print(f"Update n_estimators: {old_estimators} -> {params['n_estimators']}")

    data, data_hash = load_dataset()
    train_df, val_df, _ = split_dataset(data)
    x_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    x_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET]

    mlflow.set_experiment("CISO_LK07_Model_Lifecycle")
    with mlflow.start_run(run_name="LK07_XGBoost_Updated_v2") as run:
        model = build_model("xgboost", params)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        metrics = calculate_metrics(y_val, prediction)

        mlflow.log_params(params)
        mlflow.log_param("forecast_horizon_hours", 2)
        mlflow.set_tags({
            "algorithm": "xgboost",
            "dataset_dvc_md5": data_hash,
            "source_model": f"{SOURCE_NAME}/{SOURCE_VERSION}",
            "experiment_role": "LK07_parameter_update",
        })
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(DVC_PATH), artifact_path="dataset_lineage")

        example = x_train.head(5)
        info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
        mv = mlflow.register_model(
            model_uri=info.model_uri,
            name=NEW_NAME,
            tags={
                "algorithm": "xgboost",
                "dataset_dvc_md5": data_hash,
                "change": f"n_estimators:{old_estimators}->{params['n_estimators']}",
                "validation_rmse": str(metrics["rmse"]),
                "lifecycle_status": "None",
            },
        )
        client.update_model_version(
            NEW_NAME, mv.version,
            "LK-07 v2: retrained XGBoost with a small n_estimators change.",
        )
    print(f"Registered {NEW_NAME} v{mv.version} | run_id={run.info.run_id}")
    print(f"Validation metrics for v2: {metrics}")
    print(f"Stage: {client.get_model_version(NEW_NAME, mv.version).current_stage}")
    old_rmse = source_run.data.metrics.get("val_rmse")
    print(f"Comparison to LK-06 champion (v1 artifact): RMSE={old_rmse}")
    print("Note: do not promote if v2 fails your quality/performance criteria.")


def show_status(client):
    versions = sorted(
        client.search_model_versions(f"name = '{NEW_NAME}'"),
        key=lambda mv: int(mv.version),
    )
    if not versions:
        print("No registered versions. Run register-v1 first.")
        return
    for mv in versions:
        print(
            f"v{mv.version} | stage={mv.current_stage} | "
            f"aliases={getattr(mv, 'aliases', [])} | "
            f"run_id={mv.run_id} | dvc_md5={mv.tags.get('dataset_dvc_md5')}"
        )


def promote(client, version, stage):
    mv = client.get_model_version(NEW_NAME, version)
    current_stage = mv.current_stage or "None"
    if stage == "Staging" and current_stage not in ("None", "Staging"):
        raise ValueError(f"Expected None -> Staging, found {current_stage}")
    if stage == "Production" and current_stage != "Staging":
        raise ValueError(f"Expected Staging -> Production, found {current_stage}")

    # Legacy API retained for compatibility with the LK requirement.
    result = client.transition_model_version_stage(
        name=NEW_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )
    client.set_model_version_tag(NEW_NAME, version, "lifecycle_status", stage)
    if stage == "Staging":
        client.set_registered_model_alias(NEW_NAME, "staging", version)
    else:
        client.set_registered_model_alias(NEW_NAME, "champion", version)
        try:
            staging = client.get_model_version_by_alias(NEW_NAME, "staging")
            if staging.version == str(version):
                client.delete_registered_model_alias(NEW_NAME, "staging")
        except Exception:
            # The legacy stage has already been updated; alias clean-up is best effort.
            pass
    print(f"v{version}: {current_stage} -> {result.current_stage}")
    print(f"Alias: {'staging' if stage == 'Staging' else 'champion'}")


def export_metadata(client):
    versions = sorted(
        client.search_model_versions(f"name = '{NEW_NAME}'"),
        key=lambda mv: int(mv.version),
    )
    if not versions:
        raise ValueError("No registry versions. Nothing to export.")

    entries = []
    for mv in versions:
        run = client.get_run(mv.run_id) if mv.run_id else None
        entries.append({
            "version": int(mv.version),
            "algorithm": mv.tags.get("algorithm"),
            "stage": mv.current_stage,
            "aliases": list(getattr(mv, "aliases", [])),
            "run_id": mv.run_id,
            "model_uri": f"models:/{NEW_NAME}/{mv.version}",
            "source_model_uri": mv.tags.get("lk07_origin"),
            "dataset_dvc_md5": mv.tags.get("dataset_dvc_md5"),
            "params": dict(run.data.params) if run else {},
            "metrics": dict(run.data.metrics) if run else {},
        })

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_model": NEW_NAME,
        "source_registered_model": f"{SOURCE_NAME}/{SOURCE_VERSION}",
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_dvc_pointer": str(DVC_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_dvc_md5": get_data_hash(),
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "model_versions": entries,
    }
    try:
        prod = client.get_model_version_by_alias(NEW_NAME, "champion")
        metadata["production_alias_uri"] = f"models:/{NEW_NAME}@champion"
        metadata["production_version"] = int(prod.version)
    except Exception:
        metadata["production_alias_uri"] = None
        metadata["production_version"] = None

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("Next: dvc add artifacts/lk07/model_lineage.json")


def verify(client):
    champion = client.get_model_version_by_alias(NEW_NAME, "champion")
    if champion.current_stage != "Production":
        raise ValueError(
            f"Champion version v{champion.version} is not in legacy Production stage: "
            f"{champion.current_stage}"
        )
    alias_uri = f"models:/{NEW_NAME}@champion"
    stage_uri = f"models:/{NEW_NAME}/Production"  # Legacy stage-based URI
    by_alias = mlflow.pyfunc.load_model(alias_uri)
    by_stage = mlflow.pyfunc.load_model(stage_uri)

    df = pd.read_csv(DATA_PATH)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    sample = df[FEATURE_COLUMNS].tail(1)
    if sample.isnull().any().any():
        raise ValueError("Inference sample contains missing features")
    p1 = np.asarray(by_alias.predict(sample), dtype=float)
    p2 = np.asarray(by_stage.predict(sample), dtype=float)
    if not np.isfinite(p1).all() or not np.allclose(p1, p2):
        raise ValueError("Model inference invalid or mismatched between alias/stage")
    print(f"Model Registry: {NEW_NAME}")
    print(f"Production version: {champion.version}")
    print(f"Loaded via alias:  {alias_uri}")
    print(f"Loaded via stage:  {stage_uri}")
    print(f"Predicted electricity demand (t+2h): {p1[0]:.3f} MWh")
    print("This is a historical sample smoke test, not live forecasting.")


def main():
    parser = argparse.ArgumentParser(description="LK-07 model lifecycle using MLflow + DVC")
    parser.add_argument("--tracking-uri", default=TRACKING_URI_DEFAULT)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("register-v1")
    sub.add_parser("register-v2")
    sub.add_parser("status")
    sp = sub.add_parser("promote")
    sp.add_argument("--version", type=int, default=2)
    sp.add_argument("--stage", choices=("Staging", "Production"), required=True)
    sub.add_parser("export")
    sub.add_parser("verify")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    actions = {
        "register-v1": register_v1,
        "register-v2": register_v2,
        "status": show_status,
        "export": export_metadata,
        "verify": verify,
    }
    if args.action == "promote":
        promote(client, args.version, args.stage)
    else:
        actions[args.action](client)


if __name__ == "__main__":
    main()

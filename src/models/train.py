"""Compare Optuna-tuned XGBoost and LightGBM for CISO t+2h demand forecasting.

Run from the project root:
    python -m src.models.train --trials 10

Requires a running MLflow tracking server, and a DVC-tracked processed dataset.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import json
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/historical/ciso_train_features.csv"
DVC_PATH = DATA_PATH.with_suffix(".csv.dvc")

EXPERIMENT_NAME = "CISO_Demand_Forecasting_2h_Optuna"
REGISTERED_MODEL_NAME = "CISO-Demand-Forecaster"
TARGET = "target_demand_2h"
HORIZON = pd.Timedelta(hours=2)
SEED = 42

FEATURE_COLUMNS = [
    "value",
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]


def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}. Jalankan dvc pull.")
    if not DVC_PATH.is_file():
        raise FileNotFoundError(f"DVC pointer tidak ditemukan: {DVC_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = ["period", *FEATURE_COLUMNS, TARGET]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing_columns}")

    df["period"] = pd.to_datetime(df["period"], utc=True, errors="raise")
    if df["period"].duplicated().any():
        raise ValueError("Terdapat timestamp duplikat dalam dataset.")

    df[FEATURE_COLUMNS + [TARGET]] = df[FEATURE_COLUMNS + [TARGET]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("period").dropna(subset=required).reset_index(drop=True)

    # feature_engineering.py memakai shift(1/2/24) dan rolling(24).
    has_24h_context = df["period"].diff(24).eq(pd.Timedelta(hours=24))
    discarded = int((~has_24h_context).sum())
    df = df.loc[has_24h_context].reset_index(drop=True)
    print(f"Baris dibuang karena konteks 24 jam tidak lengkap: {discarded}")

    if len(df) < 100:
        raise ValueError("Dataset terlalu kecil setelah validasi hourly continuity.")

    with DVC_PATH.open("r", encoding="utf-8") as f:
        dvc_md5 = yaml.safe_load(f)["outs"][0]["md5"]

    print(f"Dataset valid: {len(df)} baris | DVC md5: {dvc_md5}")
    return df, dvc_md5


def split_dataset(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    val_start = df.iloc[train_end]["period"]
    test_start = df.iloc[val_end]["period"]

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Purge data: training labels t+2 tidak boleh masuk waktu validation;
    # validation labels t+2 tidak boleh masuk waktu test.
    train_df = train_df.loc[train_df["period"] + HORIZON < val_start]
    val_df = val_df.loc[val_df["period"] + HORIZON < test_start]

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Dataset tidak cukup untuk chronological split.")

    print(
        f"Chronological split | train={len(train_df)} | "
        f"validation={len(val_df)} | test={len(test_df)}"
    )
    return train_df, val_df, test_df


def calculate_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / denominator * 100)
        if denominator > 0 else float("nan"),
    }


def suggest_params(trial, algorithm):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }
    if algorithm == "lightgbm":
        params["num_leaves"] = trial.suggest_int(
            "num_leaves", 8, min(2 ** params["max_depth"], 128)
        )
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 80)
    else:
        params["reg_lambda"] = trial.suggest_float(
            "reg_lambda", 0.01, 10.0, log=True
        )
    return params


def build_model(algorithm, params):
    if algorithm == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            **params,
        )
    if algorithm == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
            subsample_freq=1,
            **params,
        )
    raise ValueError(f"Algoritma tidak dikenal: {algorithm}")


def set_lineage_tags(algorithm, data_hash):
    mlflow.set_tags({
        "algorithm": algorithm,
        "dataset_dvc_md5": data_hash,
        "target": TARGET,
        "forecast_horizon_hours": "2",
        "split": "chronological_70_15_15_purged_2h",
    })


def tune_algorithm(algorithm, X_train, y_train, X_val, y_val, data_hash, n_trials):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3),
        study_name=f"{algorithm}_tplus2h",
    )

    with mlflow.start_run(run_name=f"{algorithm}_Optuna_Search"):
        set_lineage_tags(algorithm, data_hash)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optuna_sampler", "TPE")

        def objective(trial):
            params = suggest_params(trial, algorithm)
            with mlflow.start_run(
                run_name=f"{algorithm}_trial_{trial.number:03d}", nested=True
            ):
                set_lineage_tags(algorithm, data_hash)
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_params(params)
                model = build_model(algorithm, params)
                model.fit(X_train, y_train)
                metrics = calculate_metrics(y_val, model.predict(X_val))
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                return metrics["rmse"]

        study.optimize(objective, n_trials=n_trials)
        mlflow.log_metric("best_val_rmse", float(study.best_value))
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_dict(
            {"algorithm": algorithm, "best_params": study.best_params,
             "best_val_rmse": study.best_value},
            "optuna_best.json",
        )

    return study.best_params


def fit_and_register(
    algorithm,
    params,
    X_train,
    y_train,
    X_val,
    y_val,
    data_hash,
    register=True,
):
    with mlflow.start_run(run_name=f"{algorithm}_Optuna_Best") as run:
        set_lineage_tags(algorithm, data_hash)
        mlflow.log_params(params)
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        model = build_model(algorithm, params)
        model.fit(X_train, y_train)
        val_metrics = calculate_metrics(y_val, model.predict(X_val))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        example = X_train.head(5)
        signature = infer_signature(example, model.predict(example))
        if algorithm == "xgboost":
            info = mlflow.xgboost.log_model(
                xgb_model=model, name="model", signature=signature,
                input_example=example,
            )
        else:
            info = mlflow.lightgbm.log_model(
                lgb_model=model, name="model", signature=signature,
                input_example=example,
            )

        mlflow.log_artifact(str(DVC_PATH), artifact_path="dataset_lineage")
        mlflow.log_dict({"feature_columns": FEATURE_COLUMNS}, "features.json")

        # MLflow 3: info.model_uri dapat berupa models:/<logged_model_id>.
        
        model_version = None

        if register:
            model_version = mlflow.register_model(
                model_uri=info.model_uri,
                name=REGISTERED_MODEL_NAME,
                tags={
                    "algorithm": algorithm,
                    "dataset_dvc_md5": data_hash,
                    "validation_rmse": str(
                        val_metrics["rmse"]
                    ),
                },
            )

        version = (
            str(model_version.version)
            if model_version is not None
            else "unregistered"
        )

        print(
            f"{algorithm} | validation RMSE="
            f"{val_metrics['rmse']:.3f} | "
            f"registry version={version}"
        )

        return {
            "algorithm": algorithm,
            "model": model,
            "model_uri": info.model_uri,
            "version": version,
            "run_id": run.info.run_id,
            "val_metrics": val_metrics,
        }



def train(n_trials, tracking_uri, ci_mode=False):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, data_hash = load_dataset()
    train_df, val_df, test_df = split_dataset(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    results = []
    for algorithm in ["xgboost", "lightgbm"]:
        print(f"\n=== OPTUNA TUNING: {algorithm.upper()} ({n_trials} trials) ===")
        params = tune_algorithm(
            algorithm, X_train, y_train, X_val, y_val, data_hash, n_trials
        )
        result = fit_and_register(
            algorithm, 
            params, 
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            data_hash, 
            register=not ci_mode,
        )
        results.append(result)

    # Pilih pemenang dari validation, bukan test.
    if not ci_mode:
        champion = min(results, key=lambda r: r["val_metrics"]["rmse"])
        client = MlflowClient()
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, 
            "champion", 
            champion["version"]
        )

    # Test hanya diakses setelah pemilihan champion.
    # Kedua model dinilai satu kali untuk laporan, bukan untuk memilih pemenang.
    for result in results:
        metrics = calculate_metrics(y_test, result["model"].predict(X_test))
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("selected_champion", str(result is champion).lower())
        result["test_metrics"] = metrics


    if ci_mode:
        output_dir = PROJECT_ROOT / "ci_output"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate = {
            "algorithm": champion["algorithm"],
            "run_id": champion["run_id"],
            "model_uri": champion["model_uri"],
            "dataset_dvc_md5": data_hash,
            "git_sha": os.environ.get(
                "GITHUB_SHA",
                "local",
            ),
            "val_metrics": champion["val_metrics"],
            "test_metrics": champion["test_metrics"],
        }

        output_path = output_dir / "candidate.json"

        output_path.write_text(
            json.dumps(candidate, indent=2),
            encoding="utf-8",
        )

        print(f"CI candidate saved: {output_path}")


    print("\n=== PERBANDINGAN AKHIR ===")
    for result in results:
        v = result["val_metrics"]
        t = result["test_metrics"]
        print(
            f"{result['algorithm']:8} | v{result['version']} | "
            f"val RMSE={v['rmse']:.3f}, MAE={v['mae']:.3f}, WAPE={v['wape']:.2f}% | "
            f"test RMSE={t['rmse']:.3f}, MAE={t['mae']:.3f}, WAPE={t['wape']:.2f}%"
        )
    print(
        f"CHAMPION (menurut validation RMSE): {champion['algorithm']} "
        f"v{champion['version']}"
    )
    print(f"Load champion: models:/{REGISTERED_MODEL_NAME}@champion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna + MLflow: XGBoost vs LightGBM")
    parser.add_argument(
        "--trials", 
        type=int, 
        default=50, 
        help="Jumlah trial per algoritma"
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        ),
        help="MLflow Tracking Server URL",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Log model without registering it",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials harus minimal 1")
    train(
        args.trials,
        args.tracking_uri,
        ci_mode=args.ci,
    )

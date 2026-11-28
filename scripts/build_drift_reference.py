"""Build an immutable PSI reference from the training lineage of a registered model."""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values
from mlflow import MlflowClient

from src.models.train import FEATURE_COLUMNS, TARGET, split_dataset
from src.monitoring.drift import ModelIdentity, build_reference


REFERENCE_DIR = Path("src/monitoring/references")
SPLIT = "chronological_70_15_15_purged_2h"
POINTER_ARTIFACT = "dataset_lineage/ciso_train_features.csv.dvc"


class ReferenceBuildError(ValueError):
    """A build error with a message safe for the CLI."""


def training_version(local_version, local_client, source_env):
    source_uri = local_version.tags.get("source_model_uri")
    if not source_uri:
        return local_version, local_client
    name = source_uri.removeprefix("models:/").split("@", 1)[0].split("/", 1)[0]
    model = ModelIdentity(name, local_version.tags.get("source_model_version", ""))
    settings = dotenv_values(source_env)
    source_registry = local_client.get_run(local_version.run_id).data.tags.get("source_registry")
    tracking_uri = os.environ.get("SOURCE_MLFLOW_URI") or settings.get("SOURCE_MLFLOW_URI") or source_registry
    username = os.environ.get("DAGSHUB_USERNAME") or settings.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_TOKEN") or settings.get("DAGSHUB_TOKEN")
    if not tracking_uri or not username or not token:
        raise ReferenceBuildError("Configure the source MLflow tracking URI and DagsHub credentials.")
    source_url = urlsplit(tracking_uri)
    if (source_url.scheme != "https" or source_url.netloc != "dagshub.com"
            or source_url.query or source_url.fragment):
        raise ReferenceBuildError("The imported source registry must be a DagsHub HTTPS URL.")
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    client = MlflowClient(tracking_uri=tracking_uri)
    return client.get_model_version(model.name, model.version), client


def dataset_bytes(csv_path, remote, oid, size):
    if csv_path is not None:
        data = csv_path.read_bytes()
    else:
        from dvc.repo import Repo

        with Repo() as repo:
            database = repo.cloud.get_remote_odb(remote)
            with database.fs.open(database.oid_to_path(oid), "rb") as file:
                data = file.read()
    if len(data) != size or hashlib.md5(data).hexdigest() != oid:
        raise ReferenceBuildError("Dataset bytes do not match the MLflow run's archived DVC pointer.")
    return data


def training_partition(data):
    # Match load_dataset() in src/models/train.py before using its exact split.
    frame = pd.read_csv(io.BytesIO(data))
    required = ["period", *FEATURE_COLUMNS, TARGET]
    frame["period"] = pd.to_datetime(frame["period"], utc=True, errors="raise")
    if frame["period"].duplicated().any():
        raise ReferenceBuildError("Training dataset contains duplicate timestamps.")
    frame[FEATURE_COLUMNS + [TARGET]] = frame[FEATURE_COLUMNS + [TARGET]].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values("period").dropna(subset=required).reset_index(drop=True)
    frame = frame.loc[frame["period"].diff(24).eq(pd.Timedelta(hours=24))].reset_index(drop=True)
    if len(frame) < 100:
        raise ReferenceBuildError("Training data is too small after the hourly continuity filter.")
    return split_dataset(frame)[0]


def generate(args):
    model = ModelIdentity(args.model_name, args.model_version)
    mlflow.set_tracking_uri(args.tracking_uri)
    local_client = MlflowClient(tracking_uri=args.tracking_uri)
    registered = local_client.get_model_version(model.name, model.version)
    signature = mlflow.models.get_model_info(f"models:/{model.name}/{model.version}").signature
    if (signature is None or signature.inputs is None or signature.inputs.input_names() != FEATURE_COLUMNS
            or any(column.type.name not in {"integer", "long", "float", "double"} for column in signature.inputs)):
        raise ReferenceBuildError("The registered model signature does not match the numeric training features.")

    source, client = training_version(registered, local_client, args.source_env)
    tags = client.get_run(source.run_id).data.tags
    if tags.get("split") != SPLIT or tags.get("forecast_horizon_hours") != "2":
        raise ReferenceBuildError("The source run does not identify the supported chronological training split.")
    git_sha = source.tags.get("git_sha") or tags.get("mlflow.source.git.commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise ReferenceBuildError("The source model does not identify its training commit.")
    pointer = Path(client.download_artifacts(source.run_id, POINTER_ARTIFACT))
    item = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
    if item["md5"] != tags.get("dataset_dvc_md5") or item["md5"] != source.tags.get("dataset_dvc_md5"):
        raise ReferenceBuildError("The source run and version do not agree with the archived DVC pointer.")
    data = dataset_bytes(args.csv, args.dvc_remote, item["md5"], item["size"])
    train = training_partition(data)
    reference = build_reference(model, {name: train[name].tolist() for name in FEATURE_COLUMNS}, bins=args.bins)
    manifest = reference.to_dict()
    manifest["lineage"] = {
        "local_registry_run_id": registered.run_id,
        "source_model_name": source.name,
        "source_model_version": str(source.version),
        "source_training_run_id": source.run_id,
        "training_git_sha": git_sha,
        "dataset_path": "data/processed/historical/ciso_train_features.csv",
        "dataset_dvc_md5": item["md5"],
        "dataset_size_bytes": item["size"],
        "split": SPLIT,
        "training_start_utc": train["period"].min().isoformat(),
        "training_end_utc": train["period"].max().isoformat(),
    }
    output = args.output_dir / model.reference_filename
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise ReferenceBuildError("A different reference already exists for this version. Use a separate output directory.")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as file:
            file.write(rendered)
    return output, reference.sample_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True, help="Actual loaded version, as returned by /mlflow-check.")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--source-env", type=Path, default=Path(".env.compose"))
    parser.add_argument("--dvc-remote", default="dagshub")
    parser.add_argument("--csv", type=Path, help="Optional local copy of the exact DVC object.")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    try:
        output, count = generate(args)
    except ReferenceBuildError as exc:
        parser.exit(1, f"{exc}\n")
    except Exception as exc:
        # Library/network exceptions may include credentials, URIs or request details.
        parser.exit(1, f"Could not build the reference ({type(exc).__name__}). Check registry lineage and DVC access.\n")
    print(f"{output} | training rows: {count}")


if __name__ == "__main__":
    main()

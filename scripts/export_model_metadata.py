import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model_name = "CISO-Demand-Forecaster"

version = client.get_model_version_by_alias(
    model_name,
    "champion"
)

run = client.get_run(version.run_id)

metadata = {
    "model_name": model_name,
    "model_version": int(version.version),
    "model_alias": "champion",
    "model_uri": f"models:/{model_name}@champion",
    "algorithm": version.tags.get("algorithm"),
    "dataset_dvc_md5": version.tags.get("dataset_dvc_md5"),
    "best_hyperparameters": run.data.params,
    "metrics": run.data.metrics
}

output_path = Path("config/model_metadata.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Model metadata saved to {output_path}")
import pandas as pd
from pathlib import Path

from src.data.preprocess import formating_ciso_data


PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/raw/inference"
OUTPUT_DIR = PROJECT_ROOT / "data/interim/inference"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def preprocess_inference_data(file_path):
    df = pd.read_csv(file_path)

    processed_df = formating_ciso_data(
        df,
        train=False
    )

    output_path = (
        OUTPUT_DIR
        / f"processed_{Path(file_path).name}"
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    return output_path
import pandas as pd
from pathlib import Path

from src.data.preprocess import formating_ciso_data


PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/raw/historical"
OUTPUT_DIR = PROJECT_ROOT / "data/interim/historical"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def preprocess_train_data(file_path):
    df = pd.read_csv(file_path)

    processed_df = formating_ciso_data(
        df,
        train=True
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

if __name__ == "__main__":
    input_path = (
        RAW_DIR
        / "ciso_historical.csv"
    )

    output_path = preprocess_train_data(
        input_path
    )

    print(
        f"Preprocessed data saved to: "
        f"{output_path}"
    )
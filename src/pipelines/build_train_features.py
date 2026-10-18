import pandas as pd
from pathlib import Path

from src.features.feature_engineering import (
    add_features
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "historical"
    / "processed_ciso_historical.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def build_train_features():

    df = pd.read_csv(
        INPUT_PATH
    )

    df["period"] = pd.to_datetime(
        df["period"],
        utc=True
    )

    df = add_features(df)

    # Lag/rolling menyebabkan NaN
    df = df.dropna()

    output_path = (
        OUTPUT_DIR
        / "ciso_train_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Feature dataset saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    build_train_features()
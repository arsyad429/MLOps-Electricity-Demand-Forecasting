import os
from dotenv import load_dotenv
import requests
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def get_ciso_EIA_data(
    start_date=None,
    end_date=None,
    train=True
):
    load_dotenv()

    API_KEY = os.getenv("EIA_API_KEY")

    BASE_URL = "https://api.eia.gov/v2"

    DATA_URL = (
        f"{BASE_URL}/electricity/rto/"
        "region-data/data/"
    )

    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",

        # California ISO
        "facets[respondent][]": "CISO",

        # Actual demand
        "facets[type][]": "D",

        "sort[0][column]": "period",
        "sort[0][direction]": "asc",

        "offset": 0,
        "length": 5000
    }

    if train:
        if start_date is None or end_date is None:
            raise ValueError(
                "start_date dan end_date wajib "
                "diisi ketika train=True"
            )

        params["start"] = start_date
        params["end"] = end_date

    else:
        current_date_hour = (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H")
        )

        params["start"] = current_date_hour

    response = requests.get(
        DATA_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()

def save_raw_data(df, prefix):
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_path = (
        OUTPUT_DIR
        / f"{prefix}_{timestamp}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    return output_path
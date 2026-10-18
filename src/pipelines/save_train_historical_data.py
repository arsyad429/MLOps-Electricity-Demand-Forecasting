import time
import requests
import pandas as pd

from datetime import datetime, timezone
from pathlib import Path

from src.data.ingest_data import (
    get_ciso_EIA_data,
    save_raw_data
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "historical"
    / "ciso_historical.csv"
)


def fetch_incremental_data(
    start_timestamp,
    end_timestamp,
    max_retries=3
):
    all_data = []

    current_start = start_timestamp

    while current_start <= end_timestamp:

        # Ambil per 24 jam
        current_end = min(
            current_start + pd.Timedelta(hours=23),
            end_timestamp
        )

        start_date = current_start.strftime(
            "%Y-%m-%dT%H"
        )

        end_date = current_end.strftime(
            "%Y-%m-%dT%H"
        )

        print(
            f"Fetching chunk: "
            f"{start_date} -> {end_date}"
        )

        for attempt in range(max_retries):

            try:
                result = get_ciso_EIA_data(
                    start_date=start_date,
                    end_date=end_date,
                    train=True
                )

                rows = result[
                    "response"
                ]["data"]

                if rows:
                    all_data.extend(rows)

                break

            except requests.exceptions.HTTPError as e:

                status_code = (
                    e.response.status_code
                    if e.response is not None
                    else None
                )

                if (
                    status_code
                    in [500, 502, 503, 504]
                    and attempt < max_retries - 1
                ):
                    wait_time = (
                        5 * (attempt + 1)
                    )

                    print(
                        f"EIA returned "
                        f"{status_code}. "
                        f"Retrying in "
                        f"{wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:

                    wait_time = (
                        5 * (attempt + 1)
                    )

                    print(
                        f"Request failed. "
                        f"Retrying in "
                        f"{wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        # Lanjut ke jam setelah chunk terakhir
        current_start = (
            current_end
            + pd.Timedelta(hours=1)
        )

    return pd.DataFrame(all_data)


def save_ciso_train_df():

    # =========================
    # 1. Baca dataset existing
    # =========================
    historical_df = pd.read_csv(
        DATA_PATH
    )

    # =========================
    # 2. Cari timestamp terakhir
    # =========================
    last_timestamp = pd.to_datetime(
        historical_df["period"],
        utc=True
    ).max()

    print(
        f"Last stored timestamp: "
        f"{last_timestamp}"
    )

    # =========================
    # 3. Mulai dari +1 jam
    # =========================
    start_timestamp = (
        last_timestamp
        + pd.Timedelta(hours=1)
    )

    # =========================
    # 4. Sampai latest possible
    # =========================
    end_timestamp = (
        pd.Timestamp.now(tz="UTC")
        .floor("h")
        - pd.Timedelta(hours=1)
    )

    print(
        f"Fetching new data: "
        f"{start_timestamp} "
        f"-> {end_timestamp}"
    )

    # Kalau dataset sudah up-to-date
    if start_timestamp > end_timestamp:
        print(
            "Historical dataset "
            "is already up-to-date."
        )
        return

    # =========================
    # 5. Request incremental
    # =========================
    new_df = fetch_incremental_data(
        start_timestamp,
        end_timestamp
    )

    # =========================
    # 6. Kalau belum ada data baru
    # =========================
    if new_df.empty:
        print(
            "No new data available "
            "from EIA."
        )
        return

    print(
        f"New rows fetched: "
        f"{len(new_df)}"
    )

    # =========================
    # 7. Gabungkan
    # =========================
    updated_df = pd.concat(
        [
            historical_df,
            new_df
        ],
        ignore_index=True
    )

    # Hindari timestamp duplicate
    updated_df = (
        updated_df
        .drop_duplicates(
            subset=["period"],
            keep="last"
        )
        .sort_values("period")
        .reset_index(drop=True)
    )

    # =========================
    # 8. Simpan ke file yang sama
    # =========================
    output_path = save_raw_data(
        updated_df,
        "historical/ciso_historical"
    )

    print(
        f"Dataset updated: "
        f"{output_path}"
    )

    print(
        f"Previous rows : "
        f"{len(historical_df)}"
    )

    print(
        f"New rows      : "
        f"{len(new_df)}"
    )

    print(
        f"Current rows  : "
        f"{len(updated_df)}"
    )


if __name__ == "__main__":
    save_ciso_train_df()
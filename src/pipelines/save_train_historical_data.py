"""Refresh the existing historical CISO dataset with a rolling EIA overlap."""

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

from src.data.ingest_data import (
    EIAConfig,
    EIAError,
    EIAResponseError,
    HOUR,
    fetch_ciso_observations,
    normalize_observations,
    write_raw_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/raw/historical/ciso_historical.csv"


@dataclass(frozen=True)
class HistoricalUpdate:
    status: str
    source: str | None
    latest_observation: pd.Timestamp | None
    new_rows: int = 0
    revised_rows: int = 0
    path: Path | None = None


def fetch_incremental_data(start_timestamp, end_timestamp, max_retries=None, *, config=None):
    """Retrieve a range through the common client and a single retry budget."""
    if max_retries is not None:
        config = replace(config or EIAConfig.from_env(), max_attempts=max_retries)
    return fetch_ciso_observations(start_timestamp, end_timestamp, config=config)


def save_ciso_train_df(*, path=DATA_PATH, now=None, overlap_hours=72, config=None):
    """Merge observed hours and revisions into the existing raw history."""
    if overlap_hours < 1:
        raise ValueError("Historical overlap must include at least one hour.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Historical seed CSV not found at {path}.")
    end = (pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)).floor("h") - HOUR

    # Serialize writers so an older in-flight response cannot overwrite a later run.
    try:
        with FileLock(f"{path}.lock", timeout=30):
            return _refresh_history(path, end, overlap_hours, config)
    except Timeout:
        raise EIAError("Another historical ingestion run holds the dataset lock.") from None


def _refresh_history(path, end, overlap_hours, config):
    stored = normalize_observations(pd.read_csv(path), allow_missing_values=True)
    watermark = stored.loc[stored["value"].notna(), "period"].max()
    if pd.isna(watermark):
        watermark = None
    if watermark is not None and watermark > end + HOUR:
        raise EIAResponseError("Historical CSV contains an observation after the current UTC hour.")
    anchor = watermark if watermark is not None else end
    start = anchor - (overlap_hours - 1) * HOUR
    if not stored.empty:
        start = max(start, stored["period"].min())
    if start > end:
        return HistoricalUpdate("no_new_observations", None, watermark)

    incoming = fetch_incremental_data(start, end, config=config)
    source = incoming.attrs["source"]
    if incoming.empty:
        return HistoricalUpdate("no_new_observations", source, watermark)

    previous = stored.set_index("period")["value"]
    observed = incoming.set_index("period")["value"]
    previous_for_incoming = previous.reindex(observed.index)
    new_rows = int(previous_for_incoming.isna().sum())
    revised_rows = int((previous_for_incoming.notna() & previous_for_incoming.ne(observed)).sum())
    if not new_rows and not revised_rows:
        return HistoricalUpdate("no_new_observations", source, watermark)

    updated = (
        pd.concat([stored, incoming], ignore_index=True)
        .drop_duplicates(subset="period", keep="last")
        .sort_values("period")
        .reset_index(drop=True)
    )
    latest = updated.loc[updated["value"].notna(), "period"].max()
    return HistoricalUpdate("updated", source, latest, new_rows, revised_rows,
                            write_raw_data(updated, path))


def main():
    try:
        result = save_ciso_train_df()
    except EIAError as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from None
    print(f"Status: {result.status}")
    print(f"Source: {result.source}")
    print(f"Latest observation (UTC): {result.latest_observation}")
    print(f"New observations: {result.new_rows}; revised observations: {result.revised_rows}")
    if result.path:
        print(f"Dataset updated: {result.path}")


if __name__ == "__main__":
    main()

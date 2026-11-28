from unittest.mock import Mock

import pandas as pd
import pytest

from src.data import ingest_data as eia
from src.pipelines import save_train_historical_data as historical


START = pd.Timestamp("2026-11-26T00:00Z")
HOUR = pd.Timedelta(hours=1)


def seed(path):
    path.write_text("period,value\n2026-11-26T00,10\n2026-11-26T01,20\n2026-11-26T02,\n")
    return path


def test_incremental_backfill_after_an_outage_with_overlap_and_deduplication(tmp_path, monkeypatch):
    path = seed(tmp_path / "ciso.csv")
    original = path.read_bytes()
    calls = []
    available = pd.DataFrame({
        "period": pd.date_range(START, periods=6, freq="h"),
        "value": [10, 21, 22, 23, 24, 25],
    })

    def fetch(start, end, *, config):
        calls.append((start, end))
        if len(calls) == 1:
            raise eia.EIATemporaryError("EIA temporarily unavailable.")
        result = available.loc[available["period"].between(start, end)].copy()
        result.attrs["source"] = "dashboard"
        return result

    monkeypatch.setattr(historical, "fetch_incremental_data", fetch)
    now = START + 6 * HOUR

    with pytest.raises(eia.EIATemporaryError):
        historical.save_ciso_train_df(path=path, now=now, overlap_hours=2)
    assert path.read_bytes() == original

    result = historical.save_ciso_train_df(path=path, now=now, overlap_hours=2)
    stored = pd.read_csv(path)
    assert result.status == "updated"
    assert result.new_rows == 4
    assert result.revised_rows == 1
    assert result.latest_observation == START + 5 * HOUR
    assert stored["value"].tolist() == [10, 21, 22, 23, 24, 25]
    assert pd.to_datetime(stored["period"], utc=True).tolist() == available["period"].tolist()

    previous_bytes = path.read_bytes()
    repeat = historical.save_ciso_train_df(path=path, now=now, overlap_hours=2)
    assert repeat.status == "no_new_observations"
    assert repeat.latest_observation == result.latest_observation
    assert path.read_bytes() == previous_bytes
    assert calls == [(START, START + 5 * HOUR), (START, START + 5 * HOUR),
                     (START + 4 * HOUR, START + 5 * HOUR)]


def test_historical_empty_result_preserves_observed_watermark(tmp_path, monkeypatch):
    path = seed(tmp_path / "ciso.csv")
    original = path.read_bytes()
    empty = pd.DataFrame(columns=["period", "value"])
    empty.attrs["source"] = "api"
    monkeypatch.setattr(historical, "fetch_incremental_data", Mock(return_value=empty))

    result = historical.save_ciso_train_df(path=path, now=START + 5 * HOUR)

    assert result.status == "no_new_observations"
    assert result.latest_observation == START + HOUR
    assert result.path is None
    assert path.read_bytes() == original


def test_interrupted_csv_write_preserves_existing_history(tmp_path, monkeypatch):
    path = seed(tmp_path / "ciso.csv")
    original = path.read_bytes()

    def fail(self, staging_path, **kwargs):
        staging_path.write_text("partial CSV")
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        eia.write_raw_data(pd.DataFrame({"period": [START], "value": [10]}), path)

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]

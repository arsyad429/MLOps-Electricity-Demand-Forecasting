"""Validated CISO actual demand from EIA API v2 and the EIA Grid Monitor."""

import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/raw"
API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DASHBOARD_URL = "https://www.eia.gov/electricity/930-api/region_data/data"
HOUR = pd.Timedelta(hours=1)
CISO_METADATA = {
    "respondent": "CISO",
    "respondent-name": "California Independent System Operator",
    "type": "D",
    "type-name": "Demand",
    "value-units": "megawatthours",
}
OBSERVATION_COLUMNS = ["period", "respondent", "respondent-name", "type",
                       "type-name", "value", "value-units"]
API_HOUR_PREFIX = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}")
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
LOGGER = logging.getLogger(__name__)


class EIAError(RuntimeError):
    """An ingestion error with a message safe for terminal output."""


class EIATemporaryError(EIAError):
    """EIA or the network remained unavailable within the retry budget."""


class EIARequestError(EIAError):
    """A configuration or request error that retries cannot correct."""


class EIAResponseError(EIAError):
    """A response failed the expected schema or observation checks."""


@dataclass(frozen=True)
class EIAConfig:
    source: str = "auto"
    connect_timeout: float = 5
    read_timeout: float = 12
    max_attempts: int = 3
    page_size: int = 5_000
    backoff_base: float = 1
    max_backoff: float = 10

    def __post_init__(self):
        if self.source not in ("auto", "api", "dashboard"):
            raise EIARequestError("EIA source must be auto, api, or dashboard.")
        timings = (self.connect_timeout, self.read_timeout, self.backoff_base, self.max_backoff)
        if not all(math.isfinite(value) and value > 0 for value in timings):
            raise EIARequestError("EIA timeouts and backoff limits must be positive and finite.")
        if self.max_attempts < 1 or not 1 <= self.page_size <= 5_000:
            raise EIARequestError("EIA attempts must be positive and page size must be 1–5000.")

    @classmethod
    def from_env(cls):
        load_dotenv(PROJECT_ROOT / ".env")
        try:
            return cls(
                source=os.getenv("EIA_SOURCE", "auto"),
                connect_timeout=float(os.getenv("EIA_CONNECT_TIMEOUT", "5")),
                read_timeout=float(os.getenv("EIA_READ_TIMEOUT", "12")),
                max_attempts=int(os.getenv("EIA_MAX_ATTEMPTS", "3")),
            )
        except ValueError:
            raise EIARequestError("EIA timeout and attempt settings must be numeric.") from None


def _retry_after_seconds(value):
    if not value:
        return 0
    if value.isascii() and value.isdecimal():
        return int(value)
    try:
        retry_time = parsedate_to_datetime(value)
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        return max(0, (retry_time - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0


def _request_json(session, url, params, config, context):
    backoff = min(config.backoff_base, config.max_backoff)
    for attempt in range(1, config.max_attempts + 1):
        retry_after = 0
        try:
            response = session.get(
                url,
                params=params,
                timeout=(config.connect_timeout, config.read_timeout),
                allow_redirects=False,
            )
        except requests.exceptions.SSLError:
            raise EIARequestError(f"{context} TLS verification failed.") from None
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as error:
            failure = type(error).__name__
        except requests.RequestException as error:
            raise EIARequestError(f"{context} request failed ({type(error).__name__}).") from None
        else:
            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError:
                    raise EIAResponseError(f"{context} returned invalid JSON.") from None
            failure = f"HTTP {status}"
            if status not in TRANSIENT_STATUS_CODES:
                raise EIARequestError(f"{context} rejected the request ({failure}).")
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))

        if attempt == config.max_attempts:
            raise EIATemporaryError(f"{context} failed after {attempt} attempt(s) ({failure}).")
        if retry_after > config.max_backoff:
            raise EIATemporaryError(f"{context} returned {failure} with Retry-After beyond the retry budget.")

        delay = min(config.max_backoff, backoff + random.uniform(0, backoff / 2))
        delay = max(delay, retry_after)
        LOGGER.warning("%s returned %s on attempt %d; retrying in %.1fs.", context, failure, attempt, delay)
        time.sleep(delay)
        backoff = min(config.max_backoff, backoff * 2)


def normalize_observations(frame, *, allow_missing_values=False):
    """Parse a raw observation frame without filling gaps or hiding duplicates."""
    if not {"period", "value"}.issubset(frame.columns):
        raise EIAResponseError("CISO observations require period and value columns.")
    frame = frame.copy()
    try:
        periods = pd.to_datetime(frame["period"], format="mixed", utc=True, errors="raise")
        values = pd.to_numeric(frame["value"], errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError):
        raise EIAResponseError("CISO observations contain an invalid period or demand value.") from None

    if periods.isna().any() or not periods.eq(periods.dt.floor("h")).all():
        raise EIAResponseError("CISO observations contain a missing or non-hourly timestamp.")
    if periods.duplicated().any():
        raise EIAResponseError("CISO observations contain duplicate UTC timestamps.")
    if (not allow_missing_values and values.isna().any()) or not values.dropna().map(
        lambda value: math.isfinite(value) and value >= 0
    ).all():
        raise EIAResponseError("CISO demand must be finite, nonnegative, and available.")

    frame["period"] = periods
    frame["value"] = values
    return frame.sort_values("period").reset_index(drop=True)


def _as_observations(pairs, start, end, source):
    frame = pd.DataFrame(pairs, columns=["period", "value"])
    # A published null means EIA has no observation for that hour.
    frame = normalize_observations(frame, allow_missing_values=True)
    if not frame["period"].between(start, end).all():
        raise EIAResponseError(f"{source} returned a timestamp outside the requested UTC range.")
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    for column, value in CISO_METADATA.items():
        frame[column] = value
    frame = frame[OBSERVATION_COLUMNS]
    frame.attrs["source"] = source
    return frame


def _observation_pair(row, time_key, value_key):
    period = row.get(time_key)
    if not isinstance(period, str) or not period or value_key not in row:
        raise EIAResponseError("EIA returned an observation without its timestamp or demand field.")
    demand = row[value_key]
    if demand is not None and (isinstance(demand, bool) or not isinstance(demand, (int, float, str))):
        raise EIAResponseError("EIA returned an invalid demand value.")
    if demand is not None:
        try:
            demand = float(demand)
        except (ValueError, OverflowError):
            raise EIAResponseError("EIA returned a nonnumeric demand value.") from None
        if not math.isfinite(demand) or demand < 0:
            raise EIAResponseError("EIA returned nonfinite or negative demand.")
    return period, demand


def _time_span(start, end):
    try:
        start, end = (pd.to_datetime(value, utc=True) for value in (start, end))
    except (TypeError, ValueError, OverflowError):
        raise EIARequestError("EIA date bounds must be valid UTC hours.") from None
    if any(not isinstance(value, pd.Timestamp) or value != value.floor("h") for value in (start, end)) or start > end:
        raise EIARequestError("EIA requires an ordered, inclusive range of UTC hours.")
    return start, end


def _api_pairs(session, start, end, config, api_key):
    span = (end - start) // HOUR + 1
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": "D",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": min(config.page_size, span),
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
    }
    context = f"EIA API v2 {params['start']} to {params['end']} UTC"
    pairs = []
    total = None
    while total is None or len(pairs) < total:
        payload = _request_json(session, API_URL, params | {"offset": len(pairs)}, config, context)
        page = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise EIAResponseError(f"{context} returned an invalid response envelope.")
        if page.get("frequency", "hourly") != "hourly":
            raise EIAResponseError(f"{context} returned a non-hourly frequency.")
        count = str(page.get("total"))
        if not count.isascii() or not count.isdecimal() or len(count) > len(str(span)):
            raise EIAResponseError(f"{context} returned an invalid pagination total.")
        count = int(count)
        rows = page["data"]
        if (count > span or (total is not None and count != total)
                or len(rows) > params["length"] or len(pairs) + len(rows) > count
                or (not rows and len(pairs) < count)):
            raise EIAResponseError(f"{context} returned inconsistent page counts.")
        total = count
        for row in rows:
            if (not isinstance(row, dict) or row.get("respondent") != "CISO"
                    or row.get("type") != "D" or row.get("value-units") != "megawatthours"):
                raise EIAResponseError(f"{context} returned the wrong demand series or units.")
            period, demand = _observation_pair(row, "period", "value")
            if not API_HOUR_PREFIX.match(period):
                raise EIAResponseError(f"{context} returned an invalid hourly timestamp.")
            pairs.append((period, demand))
    return pairs


def _dashboard_pairs(session, start, end, config):
    pairs = []
    while start <= end:
        # The dashboard caps result size. A 31-day window has at most 744 rows.
        chunk_end = min(start + 743 * HOUR, end)
        params = {
            "respondent[0]": "CISO",
            "type[0]": "D",
            "frequency": "hourly",
            "timezone": "UTC",
            "start": start.strftime("%m%d%Y %H:%M:%S"),
            "end": chunk_end.strftime("%m%d%Y %H:%M:%S"),
        }
        context = f"EIA Grid Monitor {start:%Y-%m-%dT%H} to {chunk_end:%Y-%m-%dT%H} UTC"
        payload = _request_json(session, DASHBOARD_URL, params, config, context)
        page = payload[0] if isinstance(payload, list) and len(payload) == 1 else None
        rows = page.get("data") if isinstance(page, dict) else None
        if (not isinstance(rows, list) or page.get("recordCount") != len(rows)
                or page.get("totalCount") != len(rows) or len(rows) > (chunk_end - start) // HOUR + 1):
            raise EIAResponseError(f"{context} returned an invalid or truncated response.")
        for row in rows:
            if not isinstance(row, dict) or row.get("RESPONDENT_ID") != "CISO" or row.get("TYPE_ID") != "D":
                raise EIAResponseError(f"{context} returned the wrong demand series.")
            # REPORTED_VAL is the balancing authority's actual demand. VAL can be imputed.
            period, demand = _observation_pair(row, "TIMESTAMP_", "REPORTED_VAL")
            try:
                parsed = datetime.strptime(period, "%m/%d/%Y %H:%M:%S")
            except ValueError:
                raise EIAResponseError(f"{context} returned an invalid UTC timestamp.") from None
            if not start <= pd.Timestamp(parsed, tz="UTC") <= chunk_end:
                raise EIAResponseError(f"{context} returned a timestamp outside its UTC window.")
            pairs.append((parsed.isoformat(), demand))
        start = chunk_end + HOUR
    return pairs


def fetch_ciso_observations(start, end, *, config=None):
    """Retrieve validated UTC hours and numeric MWh. DataFrame attrs identify the source."""
    start, end = _time_span(start, end)
    config = config or EIAConfig.from_env()
    with requests.Session() as session:
        if config.source != "dashboard":
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.getenv("EIA_API_KEY")
            if not api_key:
                raise EIARequestError("Set EIA_API_KEY or select EIA_SOURCE=dashboard.")
            try:
                pairs = _api_pairs(session, start, end, config, api_key)
            except EIATemporaryError as error:
                if config.source == "api":
                    raise
                LOGGER.warning("%s Trying the EIA Grid Monitor.", error)
            else:
                return _as_observations(pairs, start, end, "api")

        pairs = _dashboard_pairs(session, start, end, config)
        return _as_observations(pairs, start, end, "dashboard")


def get_ciso_EIA_data(start_date=None, end_date=None, train=True, *, config=None):
    """Keep the existing response.data interface for callers using the raw JSON shape."""
    if not train:
        end_date = pd.Timestamp.now(tz="UTC").floor("h")
        start_date = end_date - pd.Timedelta(hours=3)
    frame = fetch_ciso_observations(start_date, end_date, config=config)
    rows = frame.assign(period=frame["period"].dt.strftime("%Y-%m-%dT%H"))
    return {"response": {"data": rows.to_dict("records"), "total": str(len(rows)),
                         "frequency": "hourly"}, "source": frame.attrs["source"]}


def write_raw_data(df, output_path):
    """Stage CSV output on the same filesystem and publish it with a single replace."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as file:
        staging_path = Path(file.name)
    try:
        df.to_csv(staging_path, index=False)
        staging_path.replace(output_path)
    finally:
        staging_path.unlink(missing_ok=True)
    return output_path


def save_raw_data(df, prefix):
    return write_raw_data(df, OUTPUT_DIR / f"{prefix}.csv")

import traceback
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.data import ingest_data as eia
from src.features.feature_engineering import FEATURE_COLUMNS
from src.pipelines import check_inference_readiness as readiness
from src.pipelines import save_inference_data as inference


START = pd.Timestamp("2026-11-26T00:00Z")
HOUR = pd.Timedelta(hours=1)
TEST_KEY = "example-credential"


def api_row(period=START, demand="23000", **fields):
    return {
        "period": period.isoformat() if isinstance(period, pd.Timestamp) else period,
        "value": demand,
        "respondent": "CISO",
        "type": "D",
        "value-units": "megawatthours",
        **fields,
    }


def api_page(rows, total=None):
    return {"response": {"frequency": "hourly", "total": str(len(rows) if total is None else total),
                         "data": rows}}


def dashboard_page(periods, *, missing_report=False):
    rows = [
        {
            "RESPONDENT_ID": "CISO",
            "TYPE_ID": "D",
            "TIMESTAMP_": period.strftime("%m/%d/%Y %H:%M:%S"),
            "REPORTED_VAL": None if missing_report else 23_000 + i,
            "VAL": 23_000 + i,
        }
        for i, period in enumerate(periods)
    ]
    return [{"data": rows, "recordCount": len(rows), "totalCount": len(rows)}]


def response(payload=None, *, status=200, retry_after=None):
    result = Mock(status_code=status, headers={}, url=f"https://example.invalid/?api_key={TEST_KEY}")
    result.json.return_value = payload
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    return result


@pytest.fixture
def http(monkeypatch):
    get = Mock()
    sleep = Mock()
    monkeypatch.setenv("EIA_API_KEY", TEST_KEY)
    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(eia.time, "sleep", sleep)
    monkeypatch.setattr(eia.random, "uniform", lambda low, high: high)
    return get, sleep


def test_api_paginates_and_parses_utc_numeric_data(http):
    get, _ = http
    get.side_effect = [
        response(api_page([api_row(**{"period": "2026-11-25T16:00:00-08:00"})], total=2)),
        response(api_page([api_row(START + HOUR, 24_000)], total=2)),
    ]
    config = eia.EIAConfig(source="api", connect_timeout=2, read_timeout=8, page_size=1)

    result = eia.fetch_ciso_observations(START, START + HOUR, config=config)

    assert result.attrs["source"] == "api"
    assert result["period"].tolist() == [START, START + HOUR]
    assert result["value"].tolist() == [23_000, 24_000]
    assert [call.kwargs["params"]["offset"] for call in get.call_args_list] == [0, 1]
    assert get.call_args.kwargs["timeout"] == (2, 8)
    assert get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_status_retries(http, status):
    get, sleep = http
    get.side_effect = [
        response(status=status, retry_after="4" if status == 429 else None),
        response(api_page([api_row()])),
    ]
    result = eia.fetch_ciso_observations(START, START, config=eia.EIAConfig(source="api"))

    assert len(result) == 1
    assert get.call_count == 2
    sleep.assert_called_once_with(4 if status == 429 else 1.5)


def test_network_retry_backoff_and_error_are_bounded_and_secret_safe(http, caplog):
    get, sleep = http
    get.side_effect = [
        requests.ReadTimeout(f"request contained {TEST_KEY}"),
        requests.ConnectionError(f"max retries at https://example.invalid/?api_key={TEST_KEY}"),
        response(status=504),
    ]

    with pytest.raises(eia.EIATemporaryError, match=r"3 attempt\(s\) \(HTTP 504\)") as error:
        eia.fetch_ciso_observations(START, START, config=eia.EIAConfig(source="api"))

    assert [call.args[0] for call in sleep.call_args_list] == [1.5, 3]
    assert get.call_count == 3
    assert TEST_KEY not in "".join(traceback.format_exception(error.value))
    assert TEST_KEY not in caplog.text


def test_retry_after_beyond_budget_stops_without_retrying(http):
    get, sleep = http
    get.return_value = response(status=429, retry_after="60")

    with pytest.raises(eia.EIATemporaryError, match="Retry-After beyond the retry budget"):
        eia.fetch_ciso_observations(START, START, config=eia.EIAConfig(source="api"))

    assert get.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("status", [301, 400, 401, 403])
def test_permanent_api_errors_are_safe_and_do_not_fall_back(http, status):
    get, sleep = http
    get.return_value = response(status=status)
    with pytest.raises(eia.EIARequestError, match=f"HTTP {status}") as error:
        eia.fetch_ciso_observations(START, START, config=eia.EIAConfig())
    assert TEST_KEY not in "".join(traceback.format_exception(error.value))
    assert get.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("payload", [api_page([]), api_page([api_row(demand=None)])])
def test_explicitly_empty_or_unpublished_observations_are_valid(http, payload):
    get, _ = http
    get.return_value = response(payload)

    result = eia.fetch_ciso_observations(START, START, config=eia.EIAConfig())

    assert result.empty
    assert str(result["period"].dtype).endswith("UTC]")
    assert result.attrs["source"] == "api"


@pytest.mark.parametrize("payload, message", [
    ({}, "envelope"),
    ({"response": {"data": [], "total": True}}, "pagination total"),
    (api_page([], 1), "inconsistent page counts"),
    (api_page([api_row(period=START + 2 * HOUR)]), "outside the requested"),
    (api_page([api_row(period=START + pd.Timedelta(minutes=1))]), "non-hourly"),
    (api_page([api_row(period="2026-11-26")]), "invalid hourly timestamp"),
    (api_page([api_row(**{"period": ""})]), "without its timestamp"),
    (api_page([api_row(**{"period": None})]), "without its timestamp"),
    (api_page([{key: value for key, value in api_row().items() if key != "period"}]), "without its timestamp"),
    (api_page([{key: value for key, value in api_row().items() if key != "value"}]), "demand field"),
    (api_page([api_row(demand="invalid")]), "nonnumeric"),
    (api_page([api_row(demand="NaN")]), "nonfinite"),
    (api_page([api_row(demand=-1)]), "negative"),
    (api_page([api_row(demand=False)]), "invalid demand"),
    (api_page([api_row(type="DF")]), "wrong demand series"),
    (api_page([api_row(respondent="PJM")]), "wrong demand series"),
    (api_page([api_row(**{"value-units": "megawatts"})]), "wrong demand series or units"),
    (api_page([api_row(), api_row(**{"period": "2026-11-25T16:00:00-08:00"})]), "duplicate UTC"),
])
def test_invalid_responses_fail_before_fallback(http, payload, message):
    get, _ = http
    get.return_value = response(payload)
    with pytest.raises(eia.EIAResponseError, match=message):
        eia.fetch_ciso_observations(START, START + HOUR, config=eia.EIAConfig())
    assert get.call_count == 1


def test_invalid_json_exception_does_not_expose_response_body(http):
    get, _ = http
    result = response()
    result.json.side_effect = ValueError(f"raw response contained {TEST_KEY}")
    get.return_value = result

    with pytest.raises(eia.EIAResponseError, match="invalid JSON") as error:
        eia.fetch_ciso_observations(START, START, config=eia.EIAConfig())

    assert TEST_KEY not in "".join(traceback.format_exception(error.value))


def test_auto_recovers_with_dashboard_without_sending_it_the_api_key(http):
    get, _ = http
    get.side_effect = [response(status=504), response(dashboard_page([START]))]

    result = eia.fetch_ciso_observations(START, START, config=eia.EIAConfig(max_attempts=1))

    assert result.attrs["source"] == "dashboard"
    assert result["value"].tolist() == [23_000]
    api_call, dashboard_call = get.call_args_list
    assert api_call.args[0] == eia.API_URL
    assert api_call.kwargs["params"]["api_key"] == TEST_KEY
    assert dashboard_call.args[0] == eia.DASHBOARD_URL
    assert "api_key" not in dashboard_call.kwargs["params"]
    assert dashboard_call.kwargs["params"]["timezone"] == "UTC"


def test_dashboard_imputation_does_not_become_an_observation(http):
    get, _ = http
    get.return_value = response(dashboard_page([START], missing_report=True))
    assert eia.fetch_ciso_observations(START, START, config=eia.EIAConfig(source="dashboard")).empty


def test_dashboard_rejects_unexpected_truncation(http):
    get, _ = http
    payload = dashboard_page([START])
    payload[0]["totalCount"] = 2
    get.return_value = response(payload)
    with pytest.raises(eia.EIAResponseError, match="truncated"):
        eia.fetch_ciso_observations(START, START + HOUR, config=eia.EIAConfig(source="dashboard"))


@pytest.mark.parametrize("variant, status", [
    ("fresh", "ready"),
    ("gap", "incomplete_context"),
    ("stale", "stale"),
    ("empty", "no_observations"),
])
def test_inference_readiness_from_a_validated_window(http, variant, status):
    get, _ = http
    periods = pd.date_range(START, periods=26, freq="h")
    now = periods[-1] + (2 * HOUR if variant == "stale" else HOUR)
    if variant == "gap":
        periods = periods.delete(10)
    if variant == "empty":
        periods = []
    get.return_value = response(dashboard_page(periods))

    result = readiness.check_inference_readiness(config=eia.EIAConfig(source="dashboard"), now=now)

    assert result.status == status
    assert result.has_hourly_context == (variant in ("fresh", "stale"))
    if variant == "fresh":
        assert result.latest_observation == periods[-1]
        assert result.target_timestamp == periods[-1] + 2 * HOUR
        assert result.age == HOUR
        assert result.features.columns.tolist() == FEATURE_COLUMNS
        features = result.features.iloc[0]
        assert features["lag_24h"] == 23_001
        assert features["rolling_mean_24h"] == 23_013.5
    else:
        assert result.features is None


def test_inference_persists_full_live_context_and_rejects_a_stale_snapshot(http, monkeypatch):
    get, _ = http
    periods = pd.date_range(START, periods=25, freq="h")
    get.return_value = response(dashboard_page(periods))
    save = Mock()
    monkeypatch.setattr(inference, "save_raw_data", save)
    config = eia.EIAConfig(source="dashboard")

    ready = inference.save_ciso_inference_df(config=config, now=periods[-1] + HOUR)
    stale = inference.save_ciso_inference_df(config=config, now=periods[-1] + 2 * HOUR)

    assert ready.status == "ready"
    assert stale.status == "stale"
    assert save.call_count == 1
    assert len(save.call_args.args[0]) == 25

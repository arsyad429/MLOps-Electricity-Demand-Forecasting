"""Check whether fresh EIA observations can produce the existing t+2h input."""

from dataclasses import dataclass

import pandas as pd

from src.data.ingest_data import HOUR, EIAError, fetch_ciso_observations
from src.features.feature_engineering import FEATURE_COLUMNS, add_features


@dataclass(frozen=True)
class InferenceReadiness:
    status: str
    source: str
    observations: pd.DataFrame
    checked_at: pd.Timestamp
    latest_observation: pd.Timestamp | None = None
    target_timestamp: pd.Timestamp | None = None
    has_hourly_context: bool = False
    features: pd.DataFrame | None = None

    @property
    def age(self):
        if self.latest_observation is not None:
            return self.checked_at - self.latest_observation


def _assess_readiness(observations, *, now):
    """Use only the latest 25 hours and require real hourly spacing for lag 24."""
    source = observations.attrs["source"]
    if observations.empty:
        return InferenceReadiness("no_observations", source, observations, now)

    recent = observations.tail(25)
    latest = recent["period"].iloc[-1]
    target = latest + 2 * HOUR
    consecutive = len(recent) == 25 and recent["period"].diff().iloc[1:].eq(HOUR).all()
    features = None
    if target <= now:
        status = "stale"
    elif not consecutive:
        status = "incomplete_context"
    else:
        features = add_features(recent).iloc[[-1]][FEATURE_COLUMNS]
        status = "ready"
    return InferenceReadiness(status, source, observations, now, latest, target, consecutive, features)


def check_inference_readiness(*, config=None, now=None, lookback_hours=72):
    """Fetch a live window and return the model inputs only if it is ready."""
    if lookback_hours < 24:
        raise ValueError("Inference lookback must cover at least 24 hours.")
    requested_at = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    end = requested_at.floor("h")
    observations = fetch_ciso_observations(end - lookback_hours * HOUR, end, config=config)
    checked_at = pd.Timestamp.now(tz="UTC") if now is None else requested_at
    return _assess_readiness(observations, now=checked_at)


def print_readiness(result):
    print(f"Status: {result.status}")
    print(f"Source: {result.source}")
    print(f"Valid CISO demand observations: {len(result.observations)}")
    if result.latest_observation is not None:
        print(f"Latest observation (UTC): {result.latest_observation}")
        print(f"Latest demand (MWh): {result.observations['value'].iloc[-1]:g}")
        print(f"Age: {result.age / HOUR:.2f} hours")
        print(f"Target t+2h (UTC): {result.target_timestamp}")
        print(f"Latest 25 hours consecutive: {result.has_hourly_context}")
    if result.features is not None:
        print(result.features.iloc[0].to_string())


def main():
    try:
        result = check_inference_readiness()
    except EIAError as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from None
    print_readiness(result)
    if result.status != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

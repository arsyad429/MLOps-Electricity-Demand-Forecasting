import pandas as pd


def add_features(df):
    df = df.copy()

    # ==========================
    # California local time
    # ==========================
    period_california = (
        df["period"]
        .dt
        .tz_convert("America/Los_Angeles")
    )

    # ==========================
    # Calendar features
    # ==========================
    df["hour"] = (
        period_california.dt.hour
    )

    df["day_of_week"] = (
        period_california.dt.dayofweek
    )

    df["month"] = (
        period_california.dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ==========================
    # Lag features
    # ==========================
    df["lag_1h"] = (
        df["value"].shift(1)
    )

    df["lag_2h"] = (
        df["value"].shift(2)
    )

    df["lag_24h"] = (
        df["value"].shift(24)
    )

    # ==========================
    # Rolling features
    # ==========================
    df["rolling_mean_3h"] = (
        df["value"]
        .rolling(window=3)
        .mean()
    )

    df["rolling_mean_24h"] = (
        df["value"]
        .rolling(window=24)
        .mean()
    )

    return df
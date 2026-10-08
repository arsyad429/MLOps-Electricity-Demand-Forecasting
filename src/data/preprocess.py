import pandas as pd


def add_target_2h(df):
    target_df = df[
        ["period", "value"]
    ].copy()

    target_df["period"] = (
        target_df["period"]
        - pd.Timedelta(hours=2)
    )

    target_df = target_df.rename(
        columns={
            "value": "target_demand_2h"
        }
    )

    df = df.merge(
        target_df,
        on="period",
        how="left"
    )

    return df


def formating_ciso_data(df, train=True):
    df["period"] = pd.to_datetime(
        df["period"],
        utc=True
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.drop_duplicates()
    df = df.dropna()

    df = (
        df
        .sort_values("period")
        .reset_index(drop=True)
    )

    if train:
        df = add_target_2h(df)

    return df
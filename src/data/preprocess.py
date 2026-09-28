import pandas as pd

def formating_ciso_data(df, train =True):
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
        df["target_demand_1h"] = (
            df["value"].shift(-1)
        )

    return df
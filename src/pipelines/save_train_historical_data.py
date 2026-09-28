import pandas as pd
from src.data.ingest_data import get_ciso_EIA_data, save_raw_data
from datetime import datetime, timezone


def save_ciso_train_df(start_year = 2019, end_year= datetime.now(timezone.utc).year):
    full_df = pd.DataFrame()
    for year in range (start_year, end_year+1):
        result_01_07 = get_ciso_EIA_data(start_date=f"{year}-01-01T00", end_date=f"{year}-07-28T07")
        result_07_12 = get_ciso_EIA_data(start_date=f"{year}-07-28T08", end_date=f"{year}-12-31T13")
        df1 = pd.DataFrame(result_01_07["response"]["data"])
        df2 = pd.DataFrame(result_07_12["response"]["data"])
        df3 = pd.concat([df1, df2], ignore_index=True, axis=0)
        full_df = pd.concat([full_df, df3], ignore_index=True, axis=0)

    save_raw_data(
        full_df,
        "historical/ciso_historical"
    )

if __name__ == "__main__":
    save_ciso_train_df()
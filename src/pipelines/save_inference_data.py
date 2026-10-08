import pandas as pd
from src.data.ingest_data import get_ciso_EIA_data, save_raw_data

def save_ciso_inference_df():
    result = get_ciso_EIA_data(train=False)
    inference_df = pd.DataFrame(
        result["response"]["data"]
    )

    inference_df = (
        inference_df
        .sort_values("period")
        .tail(1)
    )
    save_raw_data(
        inference_df,
        "inference/ciso_inference"
    )

if __name__ == "__main__":
    save_ciso_inference_df()
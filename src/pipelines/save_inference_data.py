"""Persist a validated live CISO window for downstream inference."""

from src.data.ingest_data import EIAError, save_raw_data
from src.pipelines.check_inference_readiness import check_inference_readiness, print_readiness


def save_ciso_inference_df(*, config=None, now=None):
    result = check_inference_readiness(config=config, now=now)
    if result.status == "ready":
        save_raw_data(result.observations, "inference/ciso_inference")
    return result


def main():
    try:
        result = save_ciso_inference_df()
    except EIAError as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from None
    print_readiness(result)
    if result.status != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

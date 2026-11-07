
import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    build_model,
    calculate_metrics,
    split_dataset,
)
from src.features.feature_engineering import add_features


def test_metrics():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 190.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["wape"] == pytest.approx(6.6666667)


def test_chronological_split():
    df = pd.DataFrame({
        "period": pd.date_range(
            "2025-01-01",
            periods=120,
            freq="h",
            tz="UTC",
        )
    })

    train, val, test = split_dataset(df)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    assert (
        train["period"].max() + pd.Timedelta(hours=2)
        < val["period"].min()
    )

    assert (
        val["period"].max() + pd.Timedelta(hours=2)
        < test["period"].min()
    )


def test_feature_engineering():
    df = pd.DataFrame({
        "period": pd.date_range(
            "2025-01-01",
            periods=30,
            freq="h",
            tz="UTC",
        ),
        "value": np.arange(30, dtype=float),
    })

    result = add_features(df)

    assert result.loc[24, "lag_1h"] == 23.0
    assert result.loc[24, "lag_2h"] == 22.0
    assert result.loc[24, "lag_24h"] == 0.0
    assert result.loc[24, "rolling_mean_3h"] == 23.0


@pytest.mark.parametrize(
    "algorithm",
    ["xgboost", "lightgbm"],
)
def test_model_training(algorithm):
    X = pd.DataFrame({
        "value": np.arange(50, dtype=float)
    })
    y = np.arange(50, dtype=float)

    model = build_model(
        algorithm,
        {
            "n_estimators": 5,
            "max_depth": 2,
        },
    )

    model.fit(X, y)
    prediction = model.predict(X)

    assert len(prediction) == 50
    assert np.isfinite(prediction).all()

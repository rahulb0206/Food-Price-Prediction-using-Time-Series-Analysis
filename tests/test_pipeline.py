"""
test_pipeline.py
----------------
Basic unit tests for the preprocessing and evaluation functions.

Run with:
    python -m pytest tests/test_pipeline.py -v

These tests verify that the pipeline produces correct shapes, no data leakage
in the train/test split, and that metric computations are mathematically correct.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add src/ to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import load_raw, engineer_features, get_train_test_split


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    return load_raw()


@pytest.fixture
def engineered_df(raw_df):
    return engineer_features(raw_df)


# ── Preprocessing tests ────────────────────────────────────────────────────

def test_raw_shape(raw_df):
    """Raw dataset should have exactly 1000 rows and 17 columns."""
    assert raw_df.shape == (1000, 17), f"Expected (1000, 17), got {raw_df.shape}"


def test_no_missing_raw(raw_df):
    """Raw dataset is synthetic — should have zero missing values."""
    assert raw_df.isnull().sum().sum() == 0, "Unexpected null values in raw dataset"


def test_engineered_shape(engineered_df):
    """Engineered dataset should have 27 columns (16 features + target + 10 engineered)."""
    assert engineered_df.shape[1] == 27, f"Expected 27 columns, got {engineered_df.shape[1]}"


def test_engineered_no_missing(engineered_df):
    """After dropna(), no NaNs should remain in the engineered dataset."""
    assert engineered_df.isnull().sum().sum() == 0, "Null values found after engineering"


def test_previous_lag_column_exists(engineered_df):
    """Previous_Lag should be present in the engineered dataset."""
    assert "Previous_Lag" in engineered_df.columns, "Previous_Lag column missing"


def test_interaction_features_present(engineered_df):
    """All 10 engineered features should be present."""
    expected = [
        "Previous_Lag", "Rainfall_3m_avg", "Temperature_3m_avg",
        "TransportCost_3m_avg", "GDP_Growth_diff", "Demand_Trend",
        "Rainfall_Demand", "Temp_Crop", "Rainfall_Temp", "Yield_Demand",
    ]
    for col in expected:
        assert col in engineered_df.columns, f"Missing engineered feature: {col}"


def test_temporal_split_no_leakage(engineered_df):
    """
    Temporal split must not leak: max index in train < min index in test.
    This is the single most important correctness check for time-series data.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(engineered_df)
    assert X_train.index.max() < X_test.index.min(), \
        "Data leakage detected: train indices overlap with test indices"


def test_split_sizes(engineered_df):
    """80/20 split should give approximately correct sizes."""
    X_train, X_test, _, _ = get_train_test_split(engineered_df, test_fraction=0.2)
    total = len(engineered_df)
    assert abs(len(X_train) - int(total * 0.8)) <= 1, "Train set size incorrect"
    assert abs(len(X_test) - (total - int(total * 0.8))) <= 1, "Test set size incorrect"


def test_price_range(raw_df):
    """Food_Price should stay within expected synthetic data bounds."""
    assert raw_df["Food_Price"].min() >= 0.5, "Price below expected minimum"
    assert raw_df["Food_Price"].max() <= 15.0, "Price above expected maximum"


def test_demand_index_range(raw_df):
    """Demand_Index should be between 0.5 and 1.5 by design."""
    assert raw_df["Demand_Index"].min() >= 0.5, "Demand_Index below 0.5"
    assert raw_df["Demand_Index"].max() <= 1.5, "Demand_Index above 1.5"


def test_rainfall_demand_interaction(engineered_df):
    """
    Rainfall_Demand should equal Rainfall * Demand_Index exactly.
    Tests that interaction feature computation is correct.
    """
    expected = (engineered_df["Rainfall"] * engineered_df["Demand_Index"]).values
    actual = engineered_df["Rainfall_Demand"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-5,
                                err_msg="Rainfall_Demand interaction feature is incorrect")


# ── Metric tests ───────────────────────────────────────────────────────────

def test_perfect_prediction_r2():
    """R² of a perfect prediction should be exactly 1.0."""
    from sklearn.metrics import r2_score
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r2_score(y, y) == 1.0, "Perfect prediction should give R²=1.0"


def test_mean_prediction_r2_zero():
    """Predicting the mean for all observations should give R²=0.0."""
    from sklearn.metrics import r2_score
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_pred = np.full_like(y, y.mean())
    assert abs(r2_score(y, mean_pred)) < 1e-10, "Mean prediction should give R²≈0"


def test_rmse_non_negative():
    """RMSE must always be non-negative."""
    from sklearn.metrics import mean_squared_error
    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.1, 2.1, 3.1])
    rmse = np.sqrt(mean_squared_error(y, preds))
    assert rmse >= 0, "RMSE cannot be negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

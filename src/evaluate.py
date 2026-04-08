"""
evaluate.py
-----------
Loads saved models and computes evaluation metrics for all three approaches:
ARIMA, LSTM, and XGBoost. Prints a side-by-side comparison table.

Usage:
    python src/evaluate.py

Requires models saved by train.py in outputs/models/.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
DATA_PATH  = ROOT / "data" / "processed" / "engineered_dataset.csv"


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    """
    Compute RMSE, MAE, MAPE, and R² for a set of predictions.

    Args:
        actual:    Ground-truth values.
        predicted: Model predictions.
        name:      Model label used in output.

    Returns:
        Dict with keys: model, RMSE, MAE, MAPE_pct, R2.

    Example:
        >>> m = compute_metrics(np.array([1,2,3]), np.array([1.1,2.1,2.9]), "Test")
        >>> 0 < m["R2"] < 1
        True
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2   = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")

    return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2}


def evaluate_arima(df: pd.DataFrame) -> dict:
    """
    Evaluate the saved ARIMA model on the full monthly price series.

    ARIMA operates on an aggregated univariate series (monthly averages),
    so metrics here reflect its ability to track the overall trend —
    not individual daily records.

    Args:
        df: Engineered DataFrame with Year, Month, Food_Price columns.

    Returns:
        Metrics dict (see compute_metrics).
    """
    ts = (
        df.groupby(["Year", "Month"])["Food_Price"]
        .mean()
        .reset_index()
    )
    ts["date"] = pd.to_datetime(ts[["Year", "Month"]].assign(DAY=1))
    ts = ts.sort_values("date").set_index("date")

    model  = joblib.load(MODELS_DIR / "arima_model.pkl")
    actual = ts["Food_Price"].values
    preds  = model.predict(start=0, end=len(actual) - 1).values

    return compute_metrics(actual, preds, "ARIMA(5,1,0)")


def evaluate_lstm(df: pd.DataFrame, window: int = 12) -> dict:
    """
    Evaluate the saved LSTM model on the monthly price series.

    Predictions start at index `window` because the first window of
    observations is used as the seed input — there's nothing to predict
    for the first 12 months.

    Args:
        df:     Engineered DataFrame.
        window: Same lookback window used during training (default 12).

    Returns:
        Metrics dict (see compute_metrics).
    """
    ts = (
        df.groupby(["Year", "Month"])["Food_Price"]
        .mean()
        .reset_index()
    )
    ts["date"] = pd.to_datetime(ts[["Year", "Month"]].assign(DAY=1))
    ts = ts.sort_values("date").set_index("date")

    series = ts["Food_Price"].values.reshape(-1, 1)
    scaler = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
    scaled = scaler.transform(series)

    generator      = TimeseriesGenerator(scaled, scaled, length=window, batch_size=1)
    model          = load_model(str(MODELS_DIR / "lstm_model.h5"), compile=False)
    preds_scaled   = model.predict(generator, verbose=0)
    preds          = scaler.inverse_transform(preds_scaled).flatten()
    actual         = series[window:].flatten()

    return compute_metrics(actual, preds, "LSTM (window=12)")


def evaluate_xgboost(df: pd.DataFrame) -> dict:
    """
    Evaluate the saved XGBoost pipeline on the held-out test set (last 20%).

    The 80/20 temporal split mirrors the training split — same ordering,
    no data leakage. Full-dataset metrics are also available but the
    test-set numbers are what matters for honest reporting.

    Args:
        df: Engineered DataFrame.

    Returns:
        Metrics dict (see compute_metrics).
    """
    df = df.copy()
    df["Previous_Lag"] = df["Previous_Price"].shift(1).bfill()
    df.dropna(inplace=True)

    split   = int(len(df) * 0.8)
    X_test  = df.drop(columns=["Food_Price"]).iloc[split:]
    y_test  = df["Food_Price"].iloc[split:]

    model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    preds = model.predict(X_test)

    return compute_metrics(y_test.values, preds, "XGBoost (test set)")


def main():
    print("Loading engineered dataset...")
    df = pd.read_csv(DATA_PATH)

    results = []
    results.append(evaluate_arima(df))
    results.append(evaluate_lstm(df))
    results.append(evaluate_xgboost(df))

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    summary = pd.DataFrame(results).set_index("Model")
    summary["RMSE"]     = summary["RMSE"].round(4)
    summary["MAE"]      = summary["MAE"].round(4)
    summary["MAPE (%)"] = summary["MAPE (%)"].round(2)
    summary["R²"]       = summary["R²"].round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()

"""
train.py
--------
Trains ARIMA, LSTM, and XGBoost models on the engineered food price dataset.
Each model is saved to outputs/models/ for evaluation and inference.

Usage:
    python src/train.py

Outputs:
    outputs/models/arima_model.pkl
    outputs/models/lstm_model.h5
    outputs/models/lstm_scaler.pkl
    outputs/models/xgboost_model.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from preprocess import load_raw, engineer_features


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame, scale_for_nn: bool = False) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that one-hot encodes Season and
    scales all numeric features.

    Why StandardScaler for XGBoost (not MinMaxScaler): tree models are
    scale-invariant, but StandardScaler prevents extreme outliers from
    dominating distance-based steps in the pipeline. MinMaxScaler is used
    for LSTM since it was trained that way.

    Args:
        X:           Feature DataFrame (no target column).
        scale_for_nn: Use MinMaxScaler if True (for LSTM), else StandardScaler.

    Returns:
        Unfitted ColumnTransformer.
    """
    categorical = ["Season"]
    numerical   = X.drop(columns=categorical).columns.tolist()
    scaler      = MinMaxScaler() if scale_for_nn else StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("cat",   OneHotEncoder(drop="first"), categorical),
            ("scale", scaler,                      numerical),
        ]
    )


def train_arima(df: pd.DataFrame) -> None:
    """
    Fit ARIMA(5,1,0) on monthly-averaged food price series and save.

    ARIMA is our baseline — it sees only one variable (the price series)
    and ignores all 16 other features. It's here to show what a classical
    univariate model can and cannot do with this kind of data.

    Args:
        df: Engineered DataFrame with Year, Month, Food_Price columns.
    """
    ts = (
        df.groupby(["Year", "Month"])["Food_Price"]
        .mean()
        .reset_index()
    )
    ts["date"] = pd.to_datetime(ts[["Year", "Month"]].assign(DAY=1))
    ts = ts.sort_values("date").set_index("date")

    model_fit = ARIMA(ts["Food_Price"], order=(5, 1, 0)).fit()
    joblib.dump(model_fit, MODELS_DIR / "arima_model.pkl")
    print("  ARIMA saved.")


def train_lstm(df: pd.DataFrame, window: int = 12) -> None:
    """
    Train a two-layer LSTM on the monthly price series and save model + scaler.

    Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)
    Window size of 12 was chosen to capture a full seasonal cycle.
    EarlyStopping prevents overfitting on a small time series.

    Args:
        df:     Engineered DataFrame.
        window: Lookback window in months (default 12).
    """
    ts = (
        df.groupby(["Year", "Month"])["Food_Price"]
        .mean()
        .reset_index()
    )
    ts["date"] = pd.to_datetime(ts[["Year", "Month"]].assign(DAY=1))
    ts = ts.sort_values("date").set_index("date")

    series = ts["Food_Price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    split_idx    = int(len(scaled) * 0.9)
    train_data   = scaled[:split_idx]
    val_data     = scaled[split_idx - window:]

    train_gen = TimeseriesGenerator(train_data, train_data, length=window, batch_size=1)
    val_gen   = TimeseriesGenerator(val_data,   val_data,   length=window, batch_size=1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss=MeanSquaredError())
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )

    model.save(str(MODELS_DIR / "lstm_model.h5"))
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    print("  LSTM saved.")


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Train an XGBoost regressor inside a sklearn Pipeline with preprocessing.

    Hyperparameters were chosen to prevent overfitting on a 1,000-row dataset:
    - max_depth=3:         shallow trees generalise better on small data
    - colsample_bytree=0.5: feature subsampling adds regularisation
    - gamma=0.2:           minimum loss reduction required to split
    - reg_lambda=1:        L2 regularisation on leaf weights

    TimeSeriesSplit(n_splits=5) respects temporal ordering during cross-validation.

    Args:
        X: Feature DataFrame (includes Season column).
        y: Target Series (Food_Price).

    Returns:
        Fitted Pipeline (preprocessor + XGBRegressor).
    """
    preprocessor = build_preprocessor(X)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    XGBRegressor(
            random_state=42,
            max_depth=3,
            colsample_bytree=0.5,
            reg_lambda=1,
            gamma=0.2,
        )),
    ])

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_root_mean_squared_error")
    print(f"  XGBoost CV RMSE per fold: {(-scores).round(4)}")
    print(f"  Mean CV RMSE: {(-scores).mean():.4f} ± {(-scores).std():.4f}")

    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
    print("  XGBoost saved.")
    return model


def main():
    print("Loading and engineering data...")
    from preprocess import load_raw, engineer_features
    df = engineer_features(load_raw())
    print(f"  Dataset shape: {df.shape}")

    X = df.drop(columns=["Food_Price"])
    y = df["Food_Price"]

    print("\nTraining ARIMA...")
    train_arima(df)

    print("\nTraining LSTM...")
    train_lstm(df)

    print("\nTraining XGBoost...")
    train_xgboost(X, y)

    print("\nAll models saved to outputs/models/")


if __name__ == "__main__":
    main()

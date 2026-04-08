"""
preprocess.py
-------------
Loads the raw food price dataset, engineers all features, and saves
the processed dataset ready for model training.

Usage:
    python src/preprocess.py

Output:
    data/processed/engineered_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA  = ROOT / "data" / "raw" / "food_price_prediction_dataset.csv"
PROCESSED = ROOT / "data" / "processed" / "engineered_dataset.csv"


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """
    Load the raw food price dataset from CSV.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with 1000 rows and 17 columns (16 features + target).

    Example:
        >>> df = load_raw()
        >>> df.shape
        (1000, 17)
    """
    df = pd.read_csv(path)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 engineered features capturing lag effects, rolling averages,
    and interaction terms between weather, demand, and supply variables.

    Why these features:
    - Lag/rolling features capture price momentum and climate continuity.
    - Interaction terms (Temp × Crop, Rainfall × Demand) encode non-linear
      supply-demand dynamics that linear correlations miss entirely.

    Args:
        df: Raw DataFrame (output of load_raw).

    Returns:
        DataFrame with 10 additional columns (27 total).

    Example:
        >>> df_eng = engineer_features(load_raw())
        >>> df_eng.shape[1]
        27
    """
    df = df.copy()

    # Lag — yesterday's price is the single strongest price signal
    df["Previous_Lag"] = df["Previous_Price"].shift(1).bfill()

    # Rolling averages — smooth out daily noise in weather and costs
    df["Rainfall_3m_avg"]      = df["Rainfall"].rolling(window=3, min_periods=1).mean()
    df["Temperature_3m_avg"]   = df["Temperature"].rolling(window=3, min_periods=1).mean()
    df["TransportCost_3m_avg"] = df["Transport_Cost"].rolling(window=3, min_periods=1).mean()

    # First differences — capture acceleration, not just level
    df["GDP_Growth_diff"] = df["GDP_Growth"].diff().bfill()
    df["Demand_Trend"]    = df["Demand_Index"].diff().bfill()

    # Interaction terms — where the non-linearity lives
    df["Rainfall_Demand"] = df["Rainfall"] * df["Demand_Index"]
    df["Temp_Crop"]       = df["Temperature"] * df["Crop_Yield"]
    df["Rainfall_Temp"]   = df["Rainfall"] * df["Temperature"]
    df["Yield_Demand"]    = df["Crop_Yield"] * df["Demand_Index"]

    df.dropna(inplace=True)
    return df


def get_train_test_split(
    df: pd.DataFrame,
    target: str = "Food_Price",
    test_fraction: float = 0.2,
):
    """
    Temporal train/test split — keeps chronological order intact.

    Random split is wrong here. The dataset is time-ordered; a random split
    would let future observations bleed into training, inflating all metrics.

    Args:
        df:            Engineered DataFrame.
        target:        Name of the target column.
        test_fraction: Fraction of rows held out as test set.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> X_tr, X_te, y_tr, y_te = get_train_test_split(df)
        >>> len(X_tr) > len(X_te)
        True
    """
    split = int(len(df) * (1 - test_fraction))
    X = df.drop(columns=[target])
    y = df[target]
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"  Raw shape: {df.shape}")

    print("Engineering features...")
    df_eng = engineer_features(df)
    print(f"  Engineered shape: {df_eng.shape}")

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df_eng.to_csv(PROCESSED, index=False)
    print(f"  Saved → {PROCESSED}")


if __name__ == "__main__":
    main()

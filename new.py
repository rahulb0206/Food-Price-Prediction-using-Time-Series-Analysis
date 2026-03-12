import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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
import joblib


def load_data(path):
    data = pd.read_csv(path)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data['Previous_Lag'] = data['Previous_Price'].shift(1).bfill()

    # Feature Engineering
    data['Rainfall_3m_avg'] = data['Rainfall'].rolling(window=3, min_periods=1).mean()
    data['GDP_Growth_diff'] = data['GDP_Growth'].diff().bfill()
    data['Demand_Trend'] = data['Demand_Index'].diff().bfill()
    data['Temperature_3m_avg'] = data['Temperature'].rolling(window=3, min_periods=1).mean()
    data['TransportCost_3m_avg'] = data['Transport_Cost'].rolling(window=3, min_periods=1).mean()
    data['Rainfall_Demand'] = data['Rainfall'] * data['Demand_Index']
    data['Temp_Crop'] = data['Temperature'] * data['Crop_Yield']

    data.dropna(inplace=True)
    return data


def preprocess_data(data, scale_for_nn=False):
    X = data.drop(columns=["Food_Price"])
    y = data["Food_Price"]

    categorical = ["Season"]
    numerical = X.drop(columns=categorical).columns.tolist()

    scaler = MinMaxScaler() if scale_for_nn else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical),
            ("scale", scaler, numerical)
        ]
    )

    return X, y, preprocessor


def train_arima(data):
    ts = data.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)

    model = ARIMA(ts['Food_Price'], order=(5, 1, 0))
    model_fit = model.fit()

    joblib.dump(model_fit, "models/arima_model_new.pkl")
    print("\n arima model saved as 'models/arima_model_new.pkl'")


def train_lstm(data):
    ts = data.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)

    series = ts['Food_Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    window = 12
    split_idx = int(len(scaled) * 0.9)

    train_data = scaled[:split_idx]
    val_data = scaled[split_idx - window:]

    train_generator = TimeseriesGenerator(train_data, train_data, length=window, batch_size=1)
    val_generator = TimeseriesGenerator(val_data, val_data, length=window, batch_size=1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanSquaredError())

    model.fit(train_generator, validation_data=val_generator, epochs=50, verbose=1, callbacks=[EarlyStopping(patience=5)])
    model.save("models/lstm_model_new.h5")
    joblib.dump(scaler, "models/lstm_scaler_new.pkl")

    print("\n lstm model and scaler saved as 'models/lstm_model_final.h5' and 'models/lstm_scaler_final.pkl'")


def train_xgboost_time_split(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42, reg_lambda=1, gamma=0.2, colsample_bytree=0.5))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')

    print("TimeSeries Cross-Validation RMSE scores:", -scores)
    print("Average CV RMSE:", -scores.mean())

    model.fit(X, y)
    joblib.dump(model, "models/xgboost_model_new.pkl")
    print("\n XGBoost model saved with engineered features as 'models/xgboost_model_new.pkl'")


def show_feature_importance():
    model = joblib.load("models/xgboost_model_new.pkl")
    xgb = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    importances = xgb.feature_importances_
    importances_percent = (importances / importances.sum()) * 100
    sorted_idx = np.argsort(importances_percent)[::-1]

    print("\nTop 10 Features by Relative Importance (%):")
    for i in range(10):
        print(f"{feature_names[sorted_idx[i]]}: {importances_percent[sorted_idx[i]]:.2f}%")

    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances_percent[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.title("Top Features Influencing Food Price (XGBoost)")
    plt.xlabel("Relative Importance Score (%)")
    plt.ylabel("Features (sorted by importance)")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data("food_price_prediction_dataset.csv")

    print("\nTraining ARIMA...")
    train_arima(df)

    print("\nTraining LSTM...")
    train_lstm(df)

    print("\nPreparing data and training XGBoost...")
    X, y, preprocessor = preprocess_data(df)
    train_xgboost_time_split(X, y, preprocessor)


if __name__ == "__main__":
    main()
    show_feature_importance()

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

warnings.filterwarnings("ignore")

def load_data(path):
    data = pd.read_csv(path)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data['Food_Type'] = data['Food_Type'].astype(str)

    # Feature Engineering
    data['Previous_Lag'] = data['Previous_Price'].shift(1).bfill()
    data['Rainfall_3m_avg'] = data['Rainfall'].rolling(window=3, min_periods=1).mean()
    data['GDP_Growth_diff'] = data['GDP_Growth'].diff().bfill()
    data['Demand_Trend'] = data['Demand_Index'].diff().bfill()
    data['Temperature_3m_avg'] = data['Temperature'].rolling(window=3, min_periods=1).mean()
    data['TransportCost_3m_avg'] = data['Transport_Cost'].rolling(window=3, min_periods=1).mean()
    data['Rainfall_Demand'] = data['Rainfall'] * data['Demand_Index']
    data['Temp_Crop'] = data['Temperature'] * data['Crop_Yield']
    data['Rainfall_Temp'] = data['Rainfall'] * data['Temperature']
    data['Yield_Demand'] = data['Crop_Yield'] * data['Demand_Index']

    data.dropna(inplace=True)
    data.to_csv(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/final_dataset.csv", index=False)
    print("**********Saved final dataset with engineered features!!***********")
    return data

def filter_numerical_features_only(preprocessor, importances):
    feature_names = preprocessor.get_feature_names_out()
    numerical_features = [name for name in feature_names if name.startswith("scale__")]
    importance_dict = {name: imp for name, imp in zip(feature_names, importances) if name in numerical_features and imp > 0.0}
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return zip(*sorted_features) 

def plot_filtered_feature_importance(preprocessor, importances):
    top_features, top_importances = filter_numerical_features_only(preprocessor, importances)
    top_features = list(top_features)
    top_importances = list(top_importances)

    importances_percent = (np.array(top_importances) / np.sum(top_importances)) * 100

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=importances_percent, y=top_features, palette="crest")
    plt.title("Feature Importance Scores (XGBoost) - Only Engineered/Original Features")
    plt.xlabel("Relative Importance Score (%)")
    plt.ylabel("Features")
    plt.xlim(0, max(importances_percent) * 1.1)

    for i, v in enumerate(importances_percent):
        ax.text(v + 0.5, i, f"{v:.2f}%", color='black', va='center')

    plt.tight_layout()
    plt.savefig(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/filtered_feature_importance_top10.png")
    plt.show()

def preprocess_data(data, scale_for_nn=False):
    X = data.drop(columns=["Food_Price"])
    y = data["Food_Price"]

    categorical = ["Season", "Food_Type"]
    numerical = X.drop(columns=categorical).columns.tolist()

    scaler = MinMaxScaler() if scale_for_nn else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical),
            ("scale", scaler, numerical)
        ]
    )
    return X, y, preprocessor

def train_xgboost(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            random_state=42,
            max_depth=3,
            colsample_bytree=0.5,
            reg_lambda=1,
            gamma=0.2
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')
    print("XGBoost TimeSeries CV RMSE:", -scores.mean())

    model.fit(X, y)
    joblib.dump(model, r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/xgboost_model.pkl")
    print("XGBoost model saved.")
    return model

def show_filtered_feature_importance(model):
    xgb = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    importances = xgb.feature_importances_
    plot_filtered_feature_importance(preprocessor, importances)

def main():
    df = load_data(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/final_dataset.csv")
    X, y, preprocessor = preprocess_data(df)
    model = train_xgboost(X, y, preprocessor)
    show_filtered_feature_importance(model)

if __name__ == "__main__":
    main()

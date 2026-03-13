import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/new_engineered_dataset.csv")
df['Previous_Lag'] = df['Previous_Price'].shift(1).bfill()
df.dropna(inplace=True)


print("\nDescriptive Statistics:\n")
print(df.describe())

# Correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/Corr_matrix.png")
plt.show()

# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.histplot(df['Food_Price'], kde=True)
plt.title("Distribution of Food Price")
plt.savefig(r"/Users/rahulbalasubramani/Desktop/MSIM Study Materials/2nd Sem/IS_MDS/MoDS_FinalProject/final_output/Distribution of the target variable.png")
plt.show()


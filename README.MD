# Food Price Prediction using Time Series Analysis

A time series regression project built for MoDS (Modeling of Dynamic Systems) that predicts food prices for 10 common fruits and vegetables using economic, agricultural, and weather-based features.

---

## Team

| Name | GitHub |
|---|---|
| Rahul Balasubramani | [@rahulb6](https://github.com/rahulb6) |
| Rohith Jagadeesan | [@rohithj2](https://github.com/rohithj2) |
| Ramprasath Loganda Sureshbabu | [@rl43](https://github.com/rl43) |

---

## Why Food Prices?

Food price volatility hits everyone — farmers, retailers, policymakers, and everyday consumers. We wanted to work on something that had real stakes, not just a clean textbook dataset. The idea was to build a model that could realistically forecast prices for common grocery items given a mix of economic and environmental conditions.

We chose a regression approach since the target — food price in USD per unit — is continuous.

---

## The Data (and the hardest part of this project)

Since no single real-world dataset had all the features we wanted, we generated our own synthetic dataset simulating 14 years (2010–2024) of price data across 10 food types: Tomato, Onion, Potato, Carrot, Beans, Apple, Strawberry, Banana, Orange, and Grapes.

This turned out to be the biggest challenge of the project. Choosing the right features, defining realistic value ranges, and then convincing our professor that our data design made sense took a lot of back and forth. Every feature had to be justified — why this range, why this distribution, how does it interact with the target. It pushed us to actually understand the real-world dynamics behind each variable rather than just plugging in numbers.

In the end, the feature design was something our professor appreciated, and it made the modeling phase a lot more grounded.

**Target:** Food Price (USD/kg or USD/gal) — range $0.50 to $15.00

**Features across five categories:**

| Category | Features |
|---|---|
| Time | Day (1–28), Month, Year, Season |
| Economic | Inflation Rate (%), Exchange Rate (USD), GDP Growth (%) |
| Agricultural | Crop Yield (tons/ha), Fertilizer Price (USD/ton), Water Availability Index |
| Weather | Temperature (°C), Rainfall (mm), Humidity (%) |
| Market | Transport Cost (USD/ton), Demand Index, Previous Price |

A few design decisions worth noting:
- Day range was capped at 1–28 to avoid complications from varying month lengths
- Exchange rate range (0.1–150) was chosen to cover real extremes like JPY and KWD
- Demand Index (0.5–1.5) acts as a multiplier to simulate festive season surges or crisis drops

---

## Feature Engineering

Before modeling, we engineered 10 new features to capture lag effects, rolling trends, and interaction terms between variables.

| Feature | Logic | Why it matters |
|---|---|---|
| Previous_Lag | `Previous_Price.shift(1)` | Price momentum — yesterday's price predicts today's |
| Rainfall_3m_avg | `Rainfall.rolling(3).mean()` | Smooths out seasonal rainfall spikes |
| GDP_Growth_diff | `GDP_Growth.diff()` | Month-to-month economic shifts |
| Demand_Trend | `Demand_Index.diff()` | Captures demand acceleration/deceleration |
| Temperature_3m_avg | `Temperature.rolling(3).mean()` | Rolling climate context for crops |
| TransportCost_3m_avg | `Transport_Cost.rolling(3).mean()` | Smoothed logistics costs |
| Rainfall_Demand | `Rainfall × Demand_Index` | How weather affects demand behavior |
| Temp_Crop | `Temperature × Crop_Yield` | Temperature impact on productivity |
| Rainfall_Temp | `Rainfall × Temperature` | Combined climate interaction |
| Yield_Demand | `Crop_Yield × Demand_Index` | Supply-demand balance |

The correlation matrix post-engineering confirmed that `Previous_Price` had the strongest linear relationship with food price (~0.99), while engineered interaction terms like `Rainfall_Temp` and `Temp_Crop` captured non-linear dynamics that raw features couldn't.

---

## Models

We tested three approaches to compare classical, deep learning, and ensemble methods:

**ARIMA** — our baseline. Works well for univariate trends but struggled badly here. With 13+ input variables and non-linear dynamics, it consistently predicted near a mean value and missed any real variability. Good for understanding the floor, not the ceiling.

**LSTM** — promising in theory since food prices have sequential dependencies, but it underfit in our setup. Predictions clustered around a flat average regardless of input. Likely needs longer sequences and more tuning to shine here.

**XGBoost** — the clear winner. Predictions aligned tightly with actual values, and it handled both linear and non-linear feature interactions well. The engineered features helped it significantly. This is the model we used for the final two-week forward prediction.

---

## Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| ARIMA | High | High | High | Low |
| LSTM | Moderate | Moderate | Moderate | Moderate |
| **XGBoost** | **Lowest** | **Lowest** | **Lowest** | **Highest (~1.0)** |

**Top features by importance (XGBoost):** `Previous_Price`, `Previous_Lag`, `Demand_Index`, `Temp_Crop`, `Rainfall_Temp`

---

## Challenges

- Designing and justifying synthetic data from scratch — every feature range had to be backed by real-world reasoning
- ARIMA's inability to incorporate external variables made it a weak baseline
- LSTM required significant tuning that we couldn't fully explore within project scope
- Overfitting early on before feature engineering and proper train-test splitting
- Coordinating across three team members on a shared codebase — managing that cleanly was its own challenge

---

## Future Scope

- Plug in real-world data from USDA, FAO, or World Bank to validate against actual price trends
- Build a real-time forecasting API for retailers or supply chain teams
- Explore Transformer-based models (like Temporal Fusion Transformer) for longer-range forecasting
- Extend to supply-and-demand simulation with policy levers

---

## Tech Stack

- Python — pandas, numpy, matplotlib, seaborn, scikit-learn
- statsmodels (ARIMA), TensorFlow/Keras (LSTM), XGBoost
- Preprocessing — MinMaxScaler, StandardScaler, rolling windows, lag features

---

## References

1. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control.*
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation.*
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
4. FAO (Food and Agriculture Organization). *Reports on Global Food Prices.*
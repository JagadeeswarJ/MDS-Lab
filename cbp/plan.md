# Stock Price Prediction Project Plan

## 1. Problem Statement

* Stock price prediction is challenging due to:

  * High volatility and uncertainty.
  * Non-linear, time-dependent patterns.
  * External factors (news, economy, politics).
* Traditional models (e.g., Moving Averages, ARIMA) are limited.
* Goal: Compare simple ML methods vs Deep Learning for stock prediction.

---

## 2. Methodology

### Data Collection

* Source: Yahoo Finance / Kaggle.
* Data: Open, High, Low, Close, Volume.
* Derived features: Moving Averages (e.g., 10-day, 30-day).

### Preprocessing

* Handle missing values.
* Normalize/scale data.
* Train-test split (e.g., 80:20).

### Models

* **Baseline:** Linear Regression.
* **Advanced:** LSTM (Long Short-Term Memory network).
* (Optional): ARIMA for classical time-series benchmark.

### Evaluation

* Metrics: RMSE, MAE, MAPE.
* Compare baseline vs LSTM performance.
* Visualization: Actual vs Predicted prices.

---

## 3. Implementation Flow

1. **Data Collection** – download stock data.
2. **Preprocessing** – clean, scale, feature engineering.
3. **Baseline Model** – Linear Regression.
4. **Advanced Model** – LSTM sequence model.
5. **Comparison & Analysis** – metrics + graphs.
6. **Conclusion** – highlight improvements and limitations.

---

## 4. Key Insights

* Linear Regression provides a baseline but fails on temporal dependencies.
* LSTM captures non-linear, sequential patterns better.
* Still, prediction accuracy is limited by external unpredictable factors.

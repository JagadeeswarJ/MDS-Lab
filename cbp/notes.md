# old methods:
## Moving Average(MA)
- for time dependent patterns
- average of last n day is calculated, and plotted as a line
- gives us big picture of trends, and smoothens out short term price fluctuations
- *Negatives
    - Lagging Indicator => reacts late, by the rime signal form, the price often changes
## ARIMA = Autoregressive Integrated MA - past values + past errors , assume stationarity
- in time series analysis lag means past price;
    - P<sub>t-k</sub> is the price from k days ago
    - these are used as features for T-S analysis
- for arima, the data should be stionary -if not convert it
    - means: statistical properties don’t change over time.
        - The mean is constant.
        - The variance is constant.
        - The autocovariance (correlation with lags) depends only on the lag, not on time.
    - if not
        - The model might pick up trends instead of real dependencies.
        - Forecasts will be unreliable


# Our Project
1. data set => Yahoo Finance API
    - pip install yfinance
2. Baseline Models (Regression)

Train a Linear Regression and Polynomial Regression model.

Features: Open, High, Low, Volume, MA7, MA30.

Target: Close price.

Evaluate with RMSE, MAE, R².

3. Deep Learning Model (LSTM)

Use past 60 days of closing prices as input → predict next day close.

Reshape data for LSTM: (samples, timesteps, features).

Build LSTM using TensorFlow/Keras:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
```


use lstm and compare with linear regresion
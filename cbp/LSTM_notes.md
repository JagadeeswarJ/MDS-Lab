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

# LSTM - Long Short Term Memory
is an RNN => recurring neural networking 

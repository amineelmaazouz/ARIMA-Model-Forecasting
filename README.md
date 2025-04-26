# Arima Model Forecasting

## Description
This project aims to develop an **ARIMA model** for time series forecasting using a financial dataset, specifically for forecasting liquidity data. The project includes several important steps such as data visualization, statistical tests for stationarity, seasonal decomposition, model fitting, forecasting, and backtesting.

## Features
- **Data Importing**: Reads Excel files containing financial time series data.
- **Data Visualization**: Plots time series data for better understanding.
- **Seasonal Decomposition**: Analyzes the strength of seasonality in the data.
- **Stationarity Testing**: Performs Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity.
- **ARIMA Model**: Builds and fits an ARIMA model to forecast future liquidity values.
- **Model Evaluation**: Computes evaluation metrics such as RMSE and MAPE.
- **Backtesting**: Evaluates the model using historical data with a rolling window approach.
- **Forecasting**: Forecasts future values based on the ARIMA model.

## Requirements

- Python 3.7 or higher
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `scipy`
- `pmdarima`


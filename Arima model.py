import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

file_path = r"C:\\Users\\X280\\Desktop\\Python projects\\Forecasting project\\Classeur1.xlsx"

def evaluate():
    df = import_data(file_path)
    visualize_data_time_series(df)
    ets_decomposition_plot(df)
    stationarity_test(df)
    acf_pcf_plot(df)
    model = arima_model(df)
    forecast(df, model)
    backtesting(df)

def import_data(file_path):
    """
    :param file_path: path of dataframe
    :return: dataframe
    """
    df = pd.read_excel(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df = df.asfreq('b')
    df = df.fillna(method='bfill')
    return df

def visualize_data_time_series(df):
    """
    :param df: df to visualize
    :return: None
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df['Liquidity'], label='Liquidity', color='blue')
    plt.title('MAD Liquidity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value (in ‰)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def ets_decomposition_plot(df, model="additive", threshold=0.8):
    """
    :param df: df
    :param model: additive or multiplicative
    :param threshold: 0.8
    :return: None
    """
    decomposition = seasonal_decompose(df[['Liquidity']], model=model)
    seasonal = decomposition.seasonal
    seasonality_strength = np.var(seasonal) / (np.var(seasonal) + np.var(decomposition.resid))
    print("Strength of seasonality:", seasonality_strength)
    if seasonality_strength > threshold:
        print("Strong seasonality spotted")
    elif seasonality_strength < 1 - threshold:
        print("Really weak seasonality spotted")
    decomposition.plot()
    plt.show()

def stationarity_test(df, threshold=0.05, test="adfuller"):
    """
    :param df: df
    :param threshold: 0.05
    :param test: adfuller test to check stationnarity
    :return: None
    """
    if test == "adfuller":
        ad_fuller_test = adfuller(df['Liquidity'])
        print('ADF Statistic:', ad_fuller_test[0])
        print('p-value:', ad_fuller_test[1])
        if ad_fuller_test[1] < threshold:
            print("The series is stationary, indicating mean reversion.")
        else:
            print("The series is not stationary, no mean reversion.")
    else:
        statistic, p_value, _, _ = kpss(df["Liquidity"], regression='c')
        if p_value > threshold:
            print("The series is stationary, indicating mean reversion.")
        else:
            print("The series is not stationary, no mean reversion.")

def acf_pcf_plot(df):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(df['Liquidity'], lags=40, ax=plt.gca())
    plt.title('ACF Liquidity')
    plt.subplot(122)
    plot_pacf(df['Liquidity'], lags=40, ax=plt.gca())
    plt.title('PACF Liquidity')
    plt.tight_layout()
    plt.show()

def arima_model(df):
    df['Liquidity'] = df['Liquidity'].fillna(method='ffill')
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    model = sm.tsa.ARIMA(train['Liquidity'], order=(1, 1, 1))
    model_fit = model.fit()

    test_forecast = model_fit.forecast(steps=len(test))

    plt.figure(figsize=(12, 6))
    plt.plot(train['Liquidity'], label='Train')
    plt.plot(test['Liquidity'], label='Test')
    plt.plot(test.index, test_forecast, label='Forecast', color='red')
    plt.legend()
    plt.title('Backtest Liquidity Forecast')
    plt.show()

    rmse = math.sqrt(mean_squared_error(test['Liquidity'], test_forecast))
    mape = mean_absolute_percentage_error(test['Liquidity'], test_forecast)
    print(f'Backtest RMSE: {rmse}')
    print(f'Backtest MAPE: {mape * 100}%')

    final_model = sm.tsa.ARIMA(df['Liquidity'], order=(1, 1, 1)).fit()
    print(final_model.summary())
    final_model.plot_diagnostics(figsize=(15, 8))
    plt.show()

    return final_model

def forecast(df, model, n_periods=100):
    forecast = model.forecast(steps=n_periods)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='b')

    plt.figure(figsize=(12, 6))
    plt.plot(df['Liquidity'], label='Historical')
    plt.plot(forecast_dates, forecast, label='Forecast', color='red')
    plt.legend()
    plt.title('Liquidity Forecast')
    plt.show()

def backtesting(df, train_length=600, validation_length=150, test_length=50, step_size=30):
    pearson_r_list = []
    rmse_list = []
    mda_list = []
    samples_list = []

    def compute_mda(true_values, pred_values):
        direction_true = np.sign(true_values.diff().dropna())
        direction_pred = np.sign(pd.Series(pred_values).diff().dropna())
        return (direction_true == direction_pred).mean()

    for start in range(0, len(df) - train_length - validation_length - test_length, step_size):
        train_end = start + train_length
        validation_end = train_end + validation_length
        test_end = validation_end + test_length

        train = df.iloc[start:train_end]
        validation = df.iloc[train_end:validation_end]
        test = df.iloc[validation_end:test_end]

        train_validation = pd.concat([train, validation])

        model = sm.tsa.ARIMA(train_validation['Liquidity'], order=(1, 1, 1))
        model_fit = model.fit()

        test_forecast = model_fit.forecast(steps=len(test))

        mse = mean_squared_error(test['Liquidity'], test_forecast)
        rmse = np.sqrt(mse)
        pearson_r, _ = pearsonr(test['Liquidity'], test_forecast)
        mda = compute_mda(test['Liquidity'], test_forecast)

        pearson_r_list.append(pearson_r)
        rmse_list.append(rmse)
        mda_list.append(mda)
        samples_list.append((train_validation, test, test_forecast))

    print(f'Pearson-R: {np.mean(pearson_r_list):.4f} ± {np.std(pearson_r_list):.4f}')
    print(f'RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}')
    print(f'MDA: {np.mean(mda_list):.4f} ± {np.std(mda_list):.4f}')

    samples_to_plot = [1, int(len(samples_list) / 2), int(len(samples_list) - 1)]
    for i, sample_index in enumerate(samples_to_plot):
        if sample_index < len(samples_list):
            train_validation, test, test_forecast = samples_list[sample_index]
            plt.figure(figsize=(12, 6))
            plt.plot(train_validation['Liquidity'], label='Train + Validation')
            plt.plot(test.index, test['Liquidity'], label='Test')
            plt.plot(test.index, test_forecast, label='Forecast', color='red')
            plt.legend()
            plt.title(f'Sample {sample_index} Liquidity Forecast')
            plt.show()


if __name__ == "__main__":
    evaluate()
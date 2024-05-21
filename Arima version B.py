import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score


# 1. 加载数据
data = pd.read_csv("output2.csv")  # 替换为你的数据文件路径

# 2. 准备数据
dates = pd.to_datetime(data["Date"])
open_prices = data["Open"]
close_prices = data["Close"]
max_prices = data["Max"]
min_prices = data["Min"]


# 3. 时间序列平稳性检验
def test_stationarity(timeseries):
    # 进行ADF检验
    adf_test = adfuller(timeseries)
    adf_results = pd.Series(adf_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in adf_test[4].items():
        adf_results['Critical Value (%s)'%key] = value
    print(adf_results)

print("Open Prices Stationarity Test:")
test_stationarity(open_prices)
print("\nClose Prices Stationarity Test:")
test_stationarity(close_prices)
print("\nMax Prices Stationarity Test:")
test_stationarity(max_prices)
print("\nMin Prices Stationarity Test:")
test_stationarity(min_prices)
print("\n")

# 4. 拟合ARIMA模型并预测每天的开盘价、闭市价、最大值和最小值
def fit_arima_and_forecast(series, train_size):
    # 用指定大小的训练集拟合模型
    train_data = series[:train_size]
    test_data = series[train_size:]
    
    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(1,1,0))
    fitted_model = model.fit()
    
    # 预测剩余的数据
    forecast = fitted_model.forecast(steps=len(test_data))
    return forecast

train_size = 4561  # 增加训练集大小

open_forecast = fit_arima_and_forecast(open_prices, train_size)
close_forecast = fit_arima_and_forecast(close_prices, train_size)
max_forecast = fit_arima_and_forecast(max_prices, train_size)
min_forecast = fit_arima_and_forecast(min_prices, train_size)

rmse_open = np.sqrt(mean_squared_error(open_prices[train_size:], open_forecast))
rmse_close = np.sqrt(mean_squared_error(close_prices[train_size:], close_forecast))
rmse_max = np.sqrt(mean_squared_error(max_prices[train_size:], max_forecast))
rmse_min = np.sqrt(mean_squared_error(min_prices[train_size:], min_forecast))

r2_open = r2_score(open_prices[train_size:], open_forecast)
r2_close = r2_score(close_prices[train_size:], close_forecast)
r2_max = r2_score(max_prices[train_size:], max_forecast)
r2_min = r2_score(min_prices[train_size:], min_forecast)


# 5. 可视化预测结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(dates, open_prices, label="Actual Open Price", color="blue")
plt.plot(dates[train_size:], open_forecast, label="Predicted Open Price", color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Open Price Forecast")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(dates, close_prices, label="Actual Close Price", color="green")
plt.plot(dates[train_size:], close_forecast, label="Predicted Close Price", color="orange", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Close Price Forecast")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(dates, max_prices, label="Actual Max Price", color="blue")
plt.plot(dates[train_size:], max_forecast, label="Predicted Max Price", color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Max Price Forecast")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(dates, min_prices, label="Actual Min Price", color="green")
plt.plot(dates[train_size:], min_forecast, label="Predicted Min Price", color="orange", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Min Price Forecast")
plt.legend()
plt.grid(True)

print("Predicted Open Price:")
print(open_forecast)
print("\nPredicted Close Price:")
print(close_forecast)
print("\nPredicted Max Price:")
print(max_forecast)
print("\nPredicted Min Price:")
print(min_forecast)

forecast_df = pd.DataFrame({
    "Open_Forecast": open_forecast,
    "Close_Forecast": close_forecast,
    "Max_Forecast": max_forecast,
    "Min_Forecast": min_forecast
})

# 将DataFrame保存为CSV文件
forecast_df.to_csv("forecast_4values C.csv", index=False)

print("RMSE for Open Price:", rmse_open)
print("R^2 for Open Price:", r2_open)
print("\nRMSE for Close Price:", rmse_close)
print("R^2 for Close Price:", r2_close)
print("\nRMSE for Max Price:", rmse_max)
print("R^2 for Max Price:", r2_max)
print("\nRMSE for Min Price:", rmse_min)
print("R^2 for Min Price:", r2_min)

plt.tight_layout()
plt.show()

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 读取数据
net_df = pd.read_csv("/Users/huashenglong/Desktop/output2.csv", index_col="Date", parse_dates=True)

# 划分训练集和测试集
train_data = net_df.iloc[:int(len(net_df) * 0.9)]
test_data = net_df.iloc[int(len(net_df) * 0.9):]

# 初始化 history
history = [x for x in train_data['Max']]
y = test_data['Max']

# 进行第一次预测
predictions = []
model = ARIMA(history, order=(1, 1, 0))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y.iloc[0])

# 滚动预测
for i in range(1, len(y)):
    # 预测
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # 反转转换预测值
    predictions.append(yhat)
    # 观察结果
    obs = y.iloc[i]
    history.append(obs)

# 报告性能
mse = mean_squared_error(y, predictions)
print('MSE: ' + str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: ' + str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: ' + str(rmse))


import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(net_df.index[-600:], net_df['Max'].tail(600), color='green', label = 'Train Stock Price')
plt.plot(test_data.index, y, color = 'red', label = 'Real Stock Price')
plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.savefig('arima_model.pdf')
plt.show()




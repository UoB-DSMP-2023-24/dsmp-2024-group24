import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# load data
net_df = pd.read_csv("/Users/huashenglong/Desktop/output2.csv", index_col="Date", parse_dates=True)

# Divide the training set and test set
train_data = net_df.iloc[:int(len(net_df) * 0.9)]
test_data = net_df.iloc[int(len(net_df) * 0.9):]

# Initialise history
history = [x for x in train_data['Max']]
y = test_data['Max']

# Conducting the first forecast
predictions = []
model = ARIMA(history, order=(1, 1, 0))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y.iloc[0])

# rolling forecast
for i in range(1, len(y)):
    # forecast
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # Inverted conversion forecasts
    predictions.append(yhat)
    # result
    obs = y.iloc[i]
    history.append(obs)

# Reporting Performance
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




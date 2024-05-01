import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#lstm v1

gstock_data = pd.read_csv('E:/mini/output2.csv')

gstock_data = gstock_data[['Date', 'Max', 'Min']]

gstock_data.set_index('Date', drop=True, inplace=True)

print(gstock_data.head())

Ms = MinMaxScaler()


gstock_data[gstock_data.columns] = Ms.fit_transform(gstock_data)



training_size = round(len(gstock_data) * 0.80)


train_data = gstock_data[:training_size]
test_data = gstock_data[training_size:]

def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(50, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return (np.array(sequences), np.array(labels))



train_seq, train_label = create_sequence(train_data)
print(train_seq.shape)


test_seq, test_label = create_sequence(test_data)
print(test_seq.shape)


model = Sequential()


model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))


model.add(Dropout(0.1))


model.add(LSTM(units=50))


model.add(Dense(2))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])


model.summary()


model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)


test_predicted = model.predict(test_seq)

test_inverse_predicted = Ms.inverse_transform(test_predicted)


gs_slic_data = pd.concat([gstock_data.iloc[-1450:].copy(), pd.DataFrame(test_inverse_predicted, columns=['max_predicted', 'min_predicted'], index=gstock_data.iloc[-1450:].index)], axis=1)


gs_slic_data[['Max', 'Min']] = Ms.inverse_transform(gs_slic_data[['Max', 'Min']])


gs_slic_data.head()


gs_slic_data[['Max', 'max_predicted']].plot(figsize=(10, 6))
plt.xticks(rotation=45)
plt.xlabel('date', size=15)
plt.ylabel('price', size=15)
plt.title('actual vs predicted', size=15)
plt.show()


gs_slic_data[['Min', 'min_predicted']].plot(figsize=(10, 6))
plt.xticks(rotation=45)
plt.xlabel('date', size=15)
plt.ylabel('price', size=15)
plt.title('actual vs predicted', size=15)
plt.show()

mse_max = mean_squared_error(gs_slic_data['Max'][-len(test_inverse_predicted):], gs_slic_data['max_predicted'][-len(test_inverse_predicted):])
mse_min = mean_squared_error(gs_slic_data['Min'][-len(test_inverse_predicted):], gs_slic_data['min_predicted'][-len(test_inverse_predicted):])


rmse_max = np.sqrt(mse_max)
rmse_min = np.sqrt(mse_min)

print(f'MSE for Max: {mse_max}, RMSE for Max: {rmse_max}')
print(f'MSE for Min: {mse_min}, RMSE for Min: {rmse_min}')

test_predicted = model.predict(test_seq)


test_inverse_predicted = Ms.inverse_transform(test_predicted)


actual_max = gs_slic_data['Max'][-len(test_inverse_predicted):]
actual_min = gs_slic_data['Min'][-len(test_inverse_predicted):]


predicted_max = gs_slic_data['max_predicted'][-len(test_inverse_predicted):]
predicted_min = gs_slic_data['min_predicted'][-len(test_inverse_predicted):]


r2_max = r2_score(actual_max, predicted_max)
r2_min = r2_score(actual_min, predicted_min)


print(f"R平方值（最大价格预测）: {r2_max}")
print(f"R平方值（最小价格预测）: {r2_min}")
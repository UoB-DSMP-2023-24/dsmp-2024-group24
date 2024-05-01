import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import matplotlib.pyplot as plt

#lstm v2

gstock_data = pd.read_csv('E:/mini/nn6239.csv')
gstock_data = gstock_data[['Date', 'Max', 'Min']]
gstock_data.set_index('Date', drop=True, inplace=True)
if np.isinf(gstock_data).values.any() or gstock_data.isnull().values.any():

    gstock_data.replace([np.inf, -np.inf], np.nan, inplace=True)


    gstock_data.fillna(gstock_data.mean(), inplace=True)

print(gstock_data.head())

Ms = MinMaxScaler()


gstock_data[gstock_data.columns] = Ms.fit_transform(gstock_data)



train_data = gstock_data[:10641]
test_data = gstock_data[10641:]

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


model.fit(train_seq, train_label, epochs=50, validation_data=(test_seq, test_label), verbose=1)


test_predicted = model.predict(test_seq)


test_inverse_predicted = Ms.inverse_transform(test_predicted)


gs_slic_data = pd.concat([gstock_data.iloc[-3971:].copy(), pd.DataFrame(test_inverse_predicted, columns=['max_predicted', 'min_predicted'], index=gstock_data.iloc[-3971:].index)], axis=1)


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

gs_slic_data.to_csv('E:/mini/f2/nn6239.csv', index=True)
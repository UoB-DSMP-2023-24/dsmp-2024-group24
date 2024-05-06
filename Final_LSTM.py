import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import random
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '0'

# load data
gstock_data = pd.read_csv('E:/mini/output_segments2.csv', index_col='Date')

# choose 'Open' 'Close', 'Max', 'Min' 
gstock_data = gstock_data[['Open', 'Close', 'Max', 'Min']]

# Checking and handling infinite or missing values
if np.isinf(gstock_data).values.any() or gstock_data.isnull().values.any():
    gstock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    gstock_data.fillna(gstock_data.mean(), inplace=True)

# normalisation
Ms = MinMaxScaler()
gstock_data[gstock_data.columns] = Ms.fit_transform(gstock_data)


# Divide the training set and test set
train_data = gstock_data[:4500]
test_data = gstock_data[4500:]


# Creating sequences and labels
def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(60, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx].values)
        labels.append(dataset.iloc[stop_idx].values)
        start_idx += 1

    return (np.array(sequences), np.array(labels))

# Creating sequence data and labels
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# Building the LSTM model
model = Sequential()
# The first LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.2))
# Add a second LSTM layer, note that we set return_sequences=True here as well
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
# Add the last LSTM layer without setting the return_sequences parameter
model.add(LSTM(units=50 ))
model.add(Dropout(0.2))
# output
model.add(Dense(4))

# compilation model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# fit
history = model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)

# pred
test_predicted = model.predict(test_seq)

# inverse normalisation
test_inverse_predicted = Ms.inverse_transform(test_predicted)

# Creating a new DataFrame to store actual and predicted values
gs_slic_data = pd.concat([
    gstock_data.iloc[-len(test_seq):],
    pd.DataFrame(test_inverse_predicted, columns=['Open_predicted', 'Close_predicted', 'Max_predicted', 'Min_predicted'], index=gstock_data.iloc[-len(test_seq):].index)
], axis=1)

# Inverse normalised primitive features
gs_slic_data[gstock_data.columns] = Ms.inverse_transform(gs_slic_data[gstock_data.columns])

# Display data
print(gs_slic_data.head())



# save data
gs_slic_data.to_csv('E:/mini/f2/baoshen_predictions5.csv', index=True)

for feature in ['Open', 'Close', 'Max', 'Min']:
    mse = mean_squared_error(gs_slic_data[feature], gs_slic_data[f'{feature}_predicted'])
    rmse = np.sqrt(mse)
    print(f'RMSE for {feature}: {rmse}')

# R²
for feature in ['Open', 'Close', 'Max', 'Min']:
    r2 = r2_score(gs_slic_data[feature], gs_slic_data[f'{feature}_predicted'])
    print(f'R² for {feature}: {r2}')


# Plotting actual and projected values
features = ['Open', 'Close', 'Max', 'Min']
for feature in features:
    plt.figure(figsize=(10, 4))
    plt.plot(gs_slic_data[feature], label=f'Actual {feature}')
    plt.plot(gs_slic_data[f'{feature}_predicted'], label=f'Predicted {feature}')
    plt.legend()
    plt.title(f'{feature} Actual vs Predicted')
    plt.show()

# Plotting training loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

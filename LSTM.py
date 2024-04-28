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

# 加载数据
gstock_data = pd.read_csv('E:/mini/output_segments2.csv', index_col='Date')

# 选择 'Close', 'Max', 'Min' 特征
gstock_data = gstock_data[['Open', 'Close', 'Max', 'Min']]

# 检查并处理无限值或缺失值
if np.isinf(gstock_data).values.any() or gstock_data.isnull().values.any():
    gstock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    gstock_data.fillna(gstock_data.mean(), inplace=True)

# 归一化处理
Ms = MinMaxScaler()
gstock_data[gstock_data.columns] = Ms.fit_transform(gstock_data)


# 划分训练集和测试集
train_data = gstock_data[:4500]
test_data = gstock_data[4500:]


# 创建序列和标签
def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(60, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx].values)
        labels.append(dataset.iloc[stop_idx].values)
        start_idx += 1

    return (np.array(sequences), np.array(labels))

# 创建序列数据和标签
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# 建立LSTM模型
model = Sequential()
# 第一个LSTM层
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.2))
# 添加第二个LSTM层，注意我们在这里也设置了 return_sequences=True
model.add(LSTM(units=50, return_sequences=True))  # 可以调整units的数量
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  # 可以调整units的数量
model.add(Dropout(0.2))
# 添加最后一个LSTM层，此时不需要设置 return_sequences 参数
model.add(LSTM(units=50 ))
model.add(Dropout(0.2))
# 输出层，3个单元对应 'Close', 'Max', 'Min' 的预测
model.add(Dense(4))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# 训练模型
history = model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)

# 预测
test_predicted = model.predict(test_seq)

# 逆归一化
test_inverse_predicted = Ms.inverse_transform(test_predicted)

# 创建新的DataFrame存储实际值和预测值
gs_slic_data = pd.concat([
    gstock_data.iloc[-len(test_seq):],
    pd.DataFrame(test_inverse_predicted, columns=['Open_predicted', 'Close_predicted', 'Max_predicted', 'Min_predicted'], index=gstock_data.iloc[-len(test_seq):].index)
], axis=1)

# 逆归一化原始特征
gs_slic_data[gstock_data.columns] = Ms.inverse_transform(gs_slic_data[gstock_data.columns])

# 显示数据
print(gs_slic_data.head())



# 保存数据
gs_slic_data.to_csv('E:/mini/f2/baoshen_predictions5.csv', index=True)

for feature in ['Open', 'Close', 'Max', 'Min']:
    mse = mean_squared_error(gs_slic_data[feature], gs_slic_data[f'{feature}_predicted'])
    rmse = np.sqrt(mse)
    print(f'RMSE for {feature}: {rmse}')

# 计算R²
for feature in ['Open', 'Close', 'Max', 'Min']:
    r2 = r2_score(gs_slic_data[feature], gs_slic_data[f'{feature}_predicted'])
    print(f'R² for {feature}: {r2}')


# 绘制实际值和预测值
features = ['Open', 'Close', 'Max', 'Min']
for feature in features:
    plt.figure(figsize=(10, 4))
    plt.plot(gs_slic_data[feature], label=f'Actual {feature}')
    plt.plot(gs_slic_data[f'{feature}_predicted'], label=f'Predicted {feature}')
    plt.legend()
    plt.title(f'{feature} Actual vs Predicted')
    plt.show()

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
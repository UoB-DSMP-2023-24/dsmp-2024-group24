import os
import pandas as pd
from datetime import datetime

# 假设文件存储在这个目录下
directory_path = 'D:/数据科学/学期二/mini project/JPMorgan_Set01/Tapes'

# 用于存储提取数据的列表
data_list = []

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):  # 确保处理CSV文件
        file_path = os.path.join(directory_path, filename)

        # 从文件名提取日期
        # 假设文件名遵循 "UoB_Set01_YYYY-MM-DDtapes.csv" 的格式
        date_str = filename.split('_')[2]
        date_str = date_str.replace('tapes.csv', '')  # 从字符串中移除非日期部分
        date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # 读取CSV文件
        df = pd.read_csv(file_path, header=None)  # 假设文件没有列标题

        # 提取开盘价和闭市价
        opening_price = df.iloc[0, 1]  # 第一条记录的第二列
        closing_price = df.iloc[-1, 1]  # 最后一条记录的第二列

        # 将提取的数据添加到列表中
        data_list.append((date, opening_price, closing_price))

# 将数据转换为DataFrame并根据日期排序
data_df = pd.DataFrame(data_list, columns=['Date', 'Opening Price', 'Closing Price']).sort_values(by='Date')

# 输出结果
print(data_df)

# 可选：将结果保存到CSV文件
output_file_path = r'D:\数据科学\学期二\mini project\DATA\output.csv'
data_df.to_csv(output_file_path, index=False)
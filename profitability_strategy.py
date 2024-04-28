import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


# data = pd.read_csv('C:\\Users\\bht\\Desktop\\combine1_no_decimals.csv', index_col='Date', parse_dates=True)
data = pd.read_csv('C:\\Users\\bht\\Desktop\\baoshen\\baoshen\\true1.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\baoshen\\baoshen\\combine1.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\baoshen\\baoshen\\combine3.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\baoshen\\baoshen\\combine4.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\baoshen\\baoshen\\combine5.csv', index_col='Date', parse_dates=True)
data.rename(columns={'Highest Price': 'High', 'Lowest Price': 'Low','Opening Price': 'Open','Closing Price': 'Close'}, inplace=True)
# data.rename(columns={'Opening_Price': 'Open', 'Closing_Price': 'Close','Max_Value': 'High','Min_Value': 'Low'}, inplace=True)

class SmaCross(Strategy):
    n1 = 5
    n2 = 7

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(data, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True)

output = bt.run()
print(output)

bt.plot()

# 获取交易记录的 DataFrame
trades = output['_trades']

print(type(trades))
print(trades)


# 创建一个 DataFrame 来存储买卖时间和类型
trades_df = pd.DataFrame({
    'Entry Time': trades['EntryTime'],
    'Exit Time': trades['ExitTime'],
    'Type': trades['Size'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
})

# 打印交易时间表格
print(trades_df)

# 保存交易时间表格为 CSV 文件
trades_df.to_csv('C:\\Users\\bht\\Desktop\\baoshen\\true.csv', index=False)
trades_df.to_csv('C:\\Users\\bht\\Desktop\\baoshen\\combine1.csv', index=False)
trades_df.to_csv('C:\\Users\\bht\\Desktop\\baoshen\\combine3.csv', index=False)
trades_df.to_csv('C:\\Users\\bht\\Desktop\\baoshen\\combine4.csv', index=False)
trades_df.to_csv('C:\\Users\\bht\\Desktop\\baoshen\\combine5.csv', index=False)
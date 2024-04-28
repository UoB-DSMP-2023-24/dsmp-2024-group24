import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


# data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\true.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\combine1.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\combine2.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\combine3.csv', index_col='Date', parse_dates=True)
# data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\combine4.csv', index_col='Date', parse_dates=True)
data = pd.read_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\combine5.csv', index_col='Date', parse_dates=True)
data.rename(columns={'Opening_Price': 'Open', 'Closing_Price': 'Close','Max_Value': 'High','Min_Value': 'Low'}, inplace=True)

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

# Get the DataFrame of the transaction record
trades = output['_trades']

print(type(trades))
print(trades)


# Create a DataFrame to store the time and type of purchase and sale
trades_df = pd.DataFrame({
    'Entry Time': trades['EntryTime'],
    'Exit Time': trades['ExitTime'],
    'Type': trades['Size'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
})

# Printable Trading Hours Form
print(trades_df)

# Save the trading hours table as a CSV file
# trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\true_trade.csv', index=False)
# trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\c1_trade.csv', index=False)
# trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\c2_trade.csv', index=False)
# trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\c3_trade.csv', index=False)
# trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\c4_trade.csv', index=False)
trades_df.to_csv('C:\\Users\\bht\\Desktop\\Mini-project\\result\\c5_trade.csv', index=False)

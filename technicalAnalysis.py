import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import ta

df = pd.read_csv('input/5 Min/aapl.csv', sep=',')
close = df['Close']

df['bb_high_indicator'] = ta.bollinger_hband_indicator(close)
df['bb_low_indicator'] = ta.bollinger_lband_indicator(close)
df['rsi'] = ta.rsi(close, n=12)

portfolio = []
money = 0

class stock():
    def __init__(self, price_bought):
        self.price_bought = price_bought

def buy(df, index, money, portfolio):
    price_of_new_stock = df['Close'].iloc[i]
    portfolio.insert(0,stock(price_of_new_stock))
    money -= price_of_new_stock
    return money

def sell(money, portfolio):
    while (len(portfolio) > 0):
        stock_to_sell = portfolio.pop(0)
        money += stock_to_sell.price_bought
    return money

for i in range(len(df['rsi'])):

    rsi = df['rsi'].iloc[i]
    bb_high = df['bb_high_indicator'].iloc[i]
    bb_low = df['bb_low_indicator'].iloc[i]

    if (rsi < 30):
        money = buy(df, i, money, portfolio)
        #print("RSI is {}, Bought stock, balance: {}".format(rsi, money))

    elif (rsi > 70) & (len(portfolio) > 0):
        money = sell(money, portfolio)
        #print("RSI is {}, Sold stock, balance: {}".format(rsi, money))

print('Technical Analysis\n----------------')
# money from buying/selling stocks
print('Money: {}'.format(round(money, 2)))

# portfolio value (monetary value of all stocks held in portfolio)
portfolio_value = len(portfolio) * df['Close'].iloc[-1]
print('Portfolio value: {}'.format(round(portfolio_value, 2)))

# total value = money + portfolio value
print('Total value: {}'.format(round(money+portfolio_value, 2)))


print('\n\nBuy and Hold\n----------------')
price_difference = df['Close'].iloc[-1] - df['Close'].iloc[0]
print('Total value: {}'.format(round(price_difference, 2)))

'''
df['bb_high'] = ta.bollinger_hband(close)
df['bb_low'] = ta.bollinger_lband(close)

df['bb_high'].plot(label='Bollinger Upper')
df['bb_low'].plot(label='Bollinger Lower')
df['Close'].plot(label='AAPL', figsize=(16,8), title='AAPL Closing Price')


# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.legend()
plt.show()'''
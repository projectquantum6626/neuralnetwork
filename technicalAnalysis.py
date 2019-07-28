import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import ta

df = pd.read_csv('input/aapl/6month_1hour.csv', sep=',')
close = df['Close']

def addTechnicalAnalysisIndicators(df):
    ''' #bollinger indicators (1 or 0)
    df['bb_high_indicator'] = ta.bollinger_hband_indicator(close)
    df['bb_low_indicator'] = ta.bollinger_lband_indicator(close)'''

    # rsi with time period (for 5 min intervals, n=12 -> 1 hour)
    df['rsi'] = ta.rsi(close, n=4)

    df['sma14'] = ta.bollinger_mavg(close, n=14)
    df['sma30'] = ta.bollinger_mavg(close, n=30)
    return df

# add technical analysis indicators
df = addTechnicalAnalysisIndicators(df)

# portfolio will hold stock objects. starts empty.
portfolio = []

# start with $100,000. net_worth = cash + len(portfolio) * current_stock_price
cash = 100000

buy_orders = 0
sell_orders = 0

class stock():
    def __init__(self, price_bought):
        self.price_bought = price_bought

# buy one stock
def buy(current_stock_price, cash, portfolio):
    portfolio.insert(0, stock(current_stock_price))
    cash -= current_stock_price
    return round(cash, 2)

# sell all stocks
def sell(current_stock_price, cash, portfolio):
    while (len(portfolio) > 0):
        stock_to_sell = portfolio[0]
        cost = stock_to_sell.price_bought
        # if loss is less than 5%, sell
        if ((current_stock_price - cost)/cost < 0.05):
            portfolio.pop(0)
            cash += current_stock_price
    return round(cash, 2)

# conditional trading (based on technical analysis)
for i in range(len(df['rsi'])):
    if i < len(df['rsi']) - 1:

        # define values to be used for conditions
        rsi = round(df['rsi'].iloc[i], 2)
        rsi_next = round(df['rsi'].iloc[i+1], 2)
        sma1 = round(df['sma14'].iloc[i], 2)
        sma2 = round(df['sma30'].iloc[i], 2)

        # get the current stock price
        current_stock_price = close.iloc[i]

        # buy conditions
        if (rsi < 50) & (rsi_next > 50) & (sma1 > sma2):
            cash = buy(current_stock_price, cash, portfolio)
            buy_orders += 1
            print("{} - BUY @ ${}, rsi: {}, money_spent: {}, money_earned: {}".format(df['Date'].iloc[i], current_stock_price, rsi, money_spent, money_earned))
        
        # sell conditions
        elif (rsi > 50) & (rsi_next < 50) & (sma1 < sma2) & (len(portfolio) > 0):
            money_earned = sell(current_stock_price, money_earned, portfolio)
            sell_orders += 1
            print("{} - SELL @ ${}, rsi: {}, money_spent: {}, money_earned: {}".format(df['Date'].iloc[i], current_stock_price, rsi, money_spent, money_earned))


# printing out results

print('Technical Analysis\n----------------')

# portfolio value (monetary value of all stocks held in portfolio)
portfolio_value = len(portfolio) * close.iloc[-1]

# money represents cash flows used to buy/sell stock
ROI = ((portfolio_value + money_earned - money_spent) / money_spent) * 100
# total value = money + portfolio value
print('Return on Investment: {}%'.format(round(ROI,4)))
print('Buy orders: {}, Sell orders: {}'.format(buy_orders, sell_orders))


print('\n\nBuy and Hold\n----------------')
start = close.iloc[0]
end = close.iloc[-1]
ROI = ((end - start) / start) * 100
print('Total value: {}%'.format(round(ROI, 4)))

'''
# graphing things 

df['bb_high'] = ta.bollinger_hband(close)
df['bb_low'] = ta.bollinger_lband(close)

df['bb_high'].plot(label='Bollinger Upper')
df['bb_low'].plot(label='Bollinger Lower')
df['Close'].plot(label='AAPL', figsize=(16,8), title='AAPL Closing Price')


# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.legend()
plt.show()'''
import pandas as pd

df = pd.read_csv('input/SILVER_USD_DAILY_3.csv', sep=',')
# Date column contains the dates that the data came with
originalDays = df['Date'].values

# Date2 column contains the dates that I'm interested in
allDays = df['Date2'].values

# Bid column contains the prices
bid = df['Bid'].values

#inserts into a column and pushes everything below downwards
def insert(df, column, row, value):
    indeces = df.iloc[row:].index.tolist()
    df.loc[indeces,[column]] = df.loc[indeces,[column]].shift()
    df.loc[[row], [column]] = value

#removes from a column and pushes everything below upwards
def remove(df, column, row):
    indeces = df.iloc[row:].index.tolist()
    df.loc[indeces,[column]] = df.loc[indeces,[column]].shift(-1)

#removes dates available in original days that are not desired in all days
for i in range(len(originalDays)):
    while (df['Date'].iloc[i] not in allDays) and not (pd.isna(df['Date'].iloc[i])):
        remove(df, 'Date', i)
        remove(df, 'Bid', i)

#inserts dates that aren't originally there (constant replacement)
for i in range(len(allDays)):
    while (df['Date2'].iloc[i] not in originalDays) and not (pd.isna(df['Date'].iloc[i])):
        insert(df, 'Date', i, df['Date2'].iloc[i])
        insert(df, 'Bid', i, df['Bid'].iloc[i-1])

#inserts prices that aren't originally there (constant replacement)
for i in range(len(bid)):
    if pd.isna(df['Bid'].iloc[i]) and not pd.isna(df['Date'].iloc[i]):
        df['Bid'].iloc[i] = df['Bid'].iloc[i-1]
    
df.to_csv('input/processed/SILVER_USD_DAILY_P.csv', index=False)
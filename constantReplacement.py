import pandas as pd


#inserts into a column and pushes everything below downwards
def insert(df, column, row, value):
    indeces = df.iloc[row:].index.tolist()
    df.loc[indeces,[column]] = df.loc[indeces,[column]].shift()
    df.loc[[row], [column]] = value

#removes from a column and pushes everything below upwards
def remove(df, column, row):
    indeces = df.iloc[row:].index.tolist()
    df.loc[indeces,[column]] = df.loc[indeces,[column]].shift(-1)

# fills in empty spots in Everything.csv
def fillInEmpty():
        everything_df = pd.read_csv('input/processed/Everything.csv', sep=',')
        for column in everything_df:
                print("Filling "+column)
                for i in range(len(everything_df[column].values)):
                        row = everything_df[column].iloc[i]
                        if (pd.isna(row) or row == 'ND' or row == 'NA' or
                        row == '' or row == ' ' or row == '.'):
                                everything_df[column].iloc[i] = everything_df[column].iloc[i-1]
        everything_df.to_csv('input/processed/Everything2.csv', index=False)


# format an individual file (add missing dates, remove extra dates)
def individualFile(filename):
        df = pd.read_csv('input/'+filename, sep=',')
        everything_df = pd.read_csv('input/processed/Everything.csv', sep=',')

        everything_df = everything_df['Date']
        df = pd.concat((df, everything_df.rename('Everything_Date')), axis=1)

        # Date column contains the dates that the data came with
        originalDays = df['Date'].values

        # Everything's date column contains the dates that I'm interested in
        allDays = df['Everything_Date'].values

        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        #removes dates available in original days that are not desired in all days
        for i in range(len(originalDays)):
                while (df['Date'].iloc[i] not in allDays) and not (pd.isna(df['Date'].iloc[i])):
                        remove(df, 'Date', i)
                        for col in columns:
                                remove(df, col, i)

        #inserts dates that aren't originally there (constant replacement)
        for i in range(len(allDays)):
                while (df['Everything_Date'].iloc[i] not in df['Date'].values):
                        insert(df, 'Date', i, df['Everything_Date'].iloc[i])
                        for col in columns:
                                insert(df, col, i, df[col].iloc[i-1])
        #inserts prices that aren't originally there (constant replacement)
        """for i in range(len(bid)):
        if pd.isna(df['Bid'].iloc[i]) and not pd.isna(df['Date'].iloc[i]):
                df[''].iloc[i] = df[''].iloc[i-1]"""
        df = df.drop(columns=['Everything_Date'])
        df.to_csv('input/processed/P_'+filename, index=False)

fillInEmpty()
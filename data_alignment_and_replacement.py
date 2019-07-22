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

# fills in empty spots
def fillInEmpty(filename, replacement):
        df = pd.read_csv(filename, sep=',')
        for col in df:
                if col != 'Date':
                        print("Filling "+col)
                        for i in range(len(df[col].values)):
                                row = df[col].iloc[i]
                                empty = ['ND', 'NA', 'null', '#VALUE!', '#DIV/0!', '', ' ', '.']
                                if (pd.isna(row) or (row in empty)):
                                        if not (pd.isna(df['Date'].iloc[i])):
                                                if replacement == 'constant':
                                                        df.at[i, col] = df[col].iloc[i-1]
                                                elif replacement == 'linear':
                                                        empty_cell = 1
                                                        while pd.isna(df[col].iloc[i+empty_cell]) or (df[col].iloc[i+empty_cell] in empty):
                                                                empty_cell += 1
                                                        for cell in range(empty_cell):
                                                                #print('col:{}, i: {}, empty_cell:{}'.format(col, i, empty_cell))
                                                                start = df[col].iloc[i-1]
                                                                end = df[col].iloc[i+empty_cell]
                                                                df.at[i+cell, col] = float(df[col].iloc[i+cell-1]) + (float(end)-float(start))/float(empty_cell+1)
        df.to_csv(filename, index=False)


# format an individual file (add missing dates, remove extra dates)
def individualFile(filename, replacement, columns = ['Open', 'High', 'Low', 'Close', 'Volume']):
        df = pd.read_csv(filename, sep=',')
        columns_to_keep = ['Date'] + columns
        df = df[columns_to_keep]

        everything_df = pd.read_csv('input/processed/Stanford.csv', sep=',')
        everything_df = everything_df['Date']

        df = pd.concat((df, everything_df.rename('Everything_Date')), axis=1)

        # Date column contains the dates that the data came with
        originalDays = df['Date'].values

        # Everything's date column contains the dates that I'm interested in
        allDays = df['Everything_Date'].values

        print('Removing unnecessary dates')

        #removes dates available in original days that are not desired in all days
        for i in range(len(originalDays)):
                while (df['Date'].iloc[i] not in allDays) and not (pd.isna(df['Date'].iloc[i])):
                        remove(df, 'Date', i)
                        for col in columns:
                                remove(df, col, i)
        
        print('Inserting necessary dates (values based on {} interpolation)'.format(replacement))

        #inserts dates that aren't originally there (constant replacement)
        for i in range(len(allDays)):
                while ((df['Everything_Date'].iloc[i] not in df['Date'].values) and not (pd.isna(df['Everything_Date'].iloc[i]))):
                        insert(df, 'Date', i, df['Everything_Date'].iloc[i])
                        for col in columns:
                                if replacement == 'constant':
                                        insert(df, col, i, df[col].iloc[i-1])
                                if replacement == 'linear':
                                        insert(df, col, i, (df[col].iloc[i-1]+df[col].iloc[i])/2)


        #inserts prices that aren't originally there (constant replacement)
        """for i in range(len(bid)):
        if pd.isna(df['Bid'].iloc[i]) and not pd.isna(df['Date'].iloc[i]):
                df[''].iloc[i] = df[''].iloc[i-1]"""
        df = df.drop(columns=['Everything_Date'])
        new_filename = filename.replace('input/','input/processed/P_')
        df.to_csv(new_filename, index=False)

        print('Filling in dates that were originally there with missing data')

        fillInEmpty(new_filename, replacement)

#individualFile('input/PLATINUM_DAILY.csv', 'linear', columns=['Price'])
fillInEmpty('input/processed/Stanford_DJIA.csv', 'linear')
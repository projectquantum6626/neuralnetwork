import pandas as pd
import numpy as np
import os
from datetime import datetime

def dateProcessing(file):
        df = pd.read_csv(file, sep=',')
        # date to unix time (ms since 1970)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df['Date']]
        df.to_csv(file.replace('everything/','everything/P_'), index=False)

def separateByYear(period, path, df, file, startingYear = 2000, endingYear = 2019):
        currentYear = startingYear
        yearList = []
        while currentYear <= endingYear:
                yearList += [currentYear]
                currentYear += period

        df['Date'] = pd.to_datetime(df['Date'])
        # where the actual separation occurs
        for year in yearList:
                df_temp = df.loc[((str(year+period)+'-1-1') > df['Date'] ) & (df['Date'] > (str(year)+'-1-1'))]
                df_temp['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df_temp['Date']]
                df_temp.to_csv(path + str(period) + 'year/P_' + str(year) + '_' + str(year + period) + '_' + file, index=False)

def preprocess(path, files):
        for file in files:
                print(file)
                if file.endswith('.csv') or file.endswith('.txt'):
                        df = pd.read_csv(path+file, sep=',')

                        # yearly periods I'm interested in separating the data
                        '''periods = [1,2,3,4,5,6,7,8,9]
                        for period in periods:
                                folder = 'input/processed/everything/' + str(period) + 'year'
                                if not os.path.exists(folder):
                                        os.mkdir(folder)
                                        separateByYear(period, path, df, file)'''

                        # Generate up/down
                        #if 'Movement' not in df:
                        #        df['Movement'] = np.where(df["Close"]>df["Open"], '1', '0')


if __name__ == "__main__":
        #preprocess('input/processed/everything/', ['Everything.csv'])
        dateProcessing('input/processed/everything/Everything_old.csv')

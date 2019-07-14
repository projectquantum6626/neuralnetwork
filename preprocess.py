import pandas as pd
import numpy as np
import os
from datetime import datetime

def preprocess(path, files):
        for file in files:
                print(file)
                if file.endswith('.csv') or file.endswith('.txt'):
                        df = pd.read_csv(path+file, sep=',')
                        # Generate up/down
                        #if 'Movement' not in df:
                        #        df['Movement'] = np.where(df["Close"]>df["Open"], '1', '0')

                        # date to unix time (ms since 1970)
                        #try:
                        df['Date'] = pd.to_datetime(df['Date'])
                        for year in [2000, 2004, 2008, 2012, 2016]:
                                print(str(year+4))
                                df_temp = df.loc[((str(year+4)+'-1-1') > df['Date'] ) & (df['Date'] > (str(year)+'-1-1'))]
                                df_temp['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df_temp['Date']]
                                df_temp.to_csv(path + '4year/P_' + str(year) + '_' + str(year+4) + '_' + file, index=False)
                        '''except:
                                print('error')
                                pass'''
        

if __name__ == "__main__":
        preprocess('input/processed/everything/', ['Everything.csv'])

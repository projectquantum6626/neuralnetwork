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
                        if 'Movement' not in df:
                                df['Movement'] = np.where(df["Close"]>df["Open"], '1', '0')

                        # drop OpenInt column
                        if 'OpenInt' in df:
                                df = df.drop(columns=['OpenInt'])

                        # date to unix time (ms since 1970)
                        try:
                                df['Date'] = pd.to_datetime(df['Date'])
                                df = df.loc[(df['Date']>'2019-5-9')]
                                df['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df['Date']]
                        except:
                                pass

                        df.to_csv(path + 'processed/' + file, index=False)
        

if __name__ == "__main__":
        preprocess('input/', ['S&P.csv'])

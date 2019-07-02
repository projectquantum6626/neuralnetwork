import pandas as pd
import numpy as np
import os

def preprocess(path):
        for file in os.listdir(path):
                if file.endswith('.csv') or file.endswith('.txt'):
                        df = pd.read_csv(path+file, sep=',')

                        # Generate up/down
                        if 'Movement' not in df:
                                df['Movement'] = np.where(df["Close"]>=df["Open"], '1', '0')

                        # drop OpenInt column
                        if 'OpenInt' in df:
                                df = df.drop(columns=['OpenInt'])
                        
                        df.to_csv(path + file, index=False)
        

if __name__ == "__main__":
        preprocess('input/')

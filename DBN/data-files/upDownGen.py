import pandas as pd
import numpy as np

SP_Data = pd.read_csv("S&P.csv", sep=',')
SP_Data["Movement"] = np.where(SP_Data["Close"]>=SP_Data["Open"], '1', '0')
SP_Data.to_csv("S&P_Movement.csv", index=False)
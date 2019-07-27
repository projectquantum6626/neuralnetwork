"""
Dependencies:
pip install matplotlib statsmodels numpy pandas sklearn ploty keras
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

def classifySVM(data_file, output_col, period=None, Date=False, processed=False):
    df = pd.read_csv(data_file, sep=',')

    # Choosing whether to take in Date column or not
    if not Date:
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
    elif not processed:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df['Date']]

    # Define inputs and outputs
    X = df.drop(columns=[output_col]).values
    Y = df[[output_col]].values

    # Split testing and training
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

    # Scale everything
    '''scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)'''

    #X_train = np.nan_to_num(X_train)
    # Define MLP model and train
    for kernel in ['rbf', 'poly', 'linear']:
        classifier = svm.SVC(cache_size=1000, kernel=kernel, gamma='auto')
        classifier.fit(X_train,Y_train.ravel())
        predicted = classifier.predict(X_test)
        print('{}, accuracy: {}'.format(kernel, accuracy_score(Y_test, predicted)))

# Test with popular data set
# classifySVM('../input/banknote_authentication.csv', 'Class')

classifySVM('../input/S&P_Daily_Bollinger.csv', 'Movement')


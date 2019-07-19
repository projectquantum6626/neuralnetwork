"""
Dependencies:
pip install matplotlib statsmodels numpy pandas sklearn ploty keras
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from joblib import dump, load # model persistence

#if (not os.path.exists(output_col + '_MLP.joblib')):
# Parameters for banknote authentication data set
#data_file, output_col, hidden_layer = '../input/banknote_authentication.csv', 'Class', (8,4)
def classifySVM(data_file, output_col, period=None, Date=False):
    df = pd.read_csv(data_file, sep=',')

    if not Date:
        df = df.drop(columns=['Date'])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timestamp() for x in df['Date']]

    # Define inputs and outputs
    X = df.drop(columns=[output_col]).values
    Y = df[[output_col]].values

    # Split testing and training
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

    # Scale everything
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define MLP model and train
    for kernel in ['rbf', 'poly', 'linear']:
        classifier = svm.SVC(kernel=kernel, gamma='auto')
        classifier.fit(X_train,Y_train.ravel())
        predicted = classifier.predict(X_test)
        print('{} {}'.format(kernel, accuracy_score(Y_test, predicted)))

    """dump(classifier, output_col + '_MLP.joblib')
        print('Dumped model')

    else:
        print('Loaded model')
        classifier = load(output_col + '_MLP.joblib')"""

    #log = open('log.csv', 'a+')
    #log.write(data_file.replace('../input/processed/everything/','') + '\n')

# folder = '../input/processed/everything/'
classifySVM('../input/processed/everything/Everything_delta.csv', 'S&P_Movement')

# FULL 19 years
# classifyMLP('../input/processed/everything/Everything.csv', 'S&P_Movement', (3,6), 'sgd')

# Periods 1-9
'''
for period in [1,2,3,4,5,6,7,8,9]:
    for solver in ['sgd', 'adam', 'lbfgs']:
        folder = '../input/processed/everything/'+str(period)+'year'
        for file in os.listdir(folder):
            if file.endswith('old.csv'):
                classifyMLP(folder+'/'+file, 'S&P_Movement', (3,6), solver, period)
'''

# Just 1 year periods
'''
for period in [1]:
    for solver in ['sgd', 'adam', 'lbfgs']:
        folder = '../input/processed/everything/'+str(period)+'year'
        for file in os.listdir(folder):
            if file.endswith('old.csv'):
                classifyMLP(folder+'/'+file, 'S&P_Movement', (3,6), solver, period)
'''
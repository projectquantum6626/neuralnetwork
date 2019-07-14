"""
Dependencies:
pip install matplotlib statsmodels numpy pandas sklearn ploty keras
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
from joblib import dump, load # model persistence

#if (not os.path.exists(output_col + '_MLP.joblib')):
# Parameters for banknote authentication data set
#data_file, output_col, hidden_layer = '../input/banknote_authentication.csv', 'Class', (8,4)
def classifyMLP(data_file, output_col, hidden_layer):
    df = pd.read_csv(data_file, sep=',')

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
    classifier=MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=20000, alpha=0.00001,
                                solver='lbfgs', verbose=10,  random_state=21, tol=0.0000001)
    classifier.fit(X_train,Y_train)

    """dump(classifier, output_col + '_MLP.joblib')
        print('Dumped model')

    else:
        print('Loaded model')
        classifier = load(output_col + '_MLP.joblib')"""

    # Print results
    accuracies=cross_val_score(estimator=classifier,X=X_test,y=Y_test,cv=10)

    log = open('log.txt', 'a+')
    log.write("\nAccuracies: {}".format(accuracies))
    log.write("\nMean Accuracy: {}".format(accuracies.mean()))

for file in os.listdir('../input/processed/everything/2year'):
    classifyMLP('../input/processed/everything/2year/'+file, 'S&P_Movement', (3,6))
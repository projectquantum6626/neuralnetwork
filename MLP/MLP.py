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

# Parameters for S&P 500
data_file, output_col, hidden_layer = '../input/processed/P_Everything.csv', 'S&P_Movement', (3,6)

# Parameters for banknote authentication data set
#data_file, output_col, hidden_layer = '../input/banknote_authentication.csv', 'Class', (8,4)

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
classifier=MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=20000, alpha=0.001,
                            solver='sgd', verbose=10,  random_state=21, tol=0.0000001)
classifier.fit(X_train,Y_train)

# Print results
accuracies=cross_val_score(estimator=classifier,X=X_test,y=Y_test,cv=10)
print("Accuracies: {}".format(accuracies))
print("Mean Accuracy: {}".format(accuracies.mean()))
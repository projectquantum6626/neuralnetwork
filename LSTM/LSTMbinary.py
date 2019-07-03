"""
Dependencies:
pip install matplotlib statsmodels numpy pandas sklearn ploty keras
"""
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from joblib import dump, load # model persistence

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

# Parameters for S&P 500
#data_file, output_col = '../input/processed/S&P.csv', 'Movement'

# Parameters for banknote authentication data set
data_file, output_col = '../input/banknote_authentication.csv', 'Class'

df = pd.read_csv(data_file, sep=',')

inputs = df.drop(columns=[output_col]).values
outputs = df[[output_col]].values

# Split the training and test set
train_inputs, test_inputs = train_test_split(inputs, test_size=0.3)
train_outputs, test_outputs = train_test_split(outputs, test_size=0.3)
print("Train entries: " + str(len(train_inputs)))
print("Test entries: " + str(len(test_inputs)))

scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.fit_transform(test_inputs)

def reshapeInput(array):
    nrows, ncols = array.shape
    array = array.reshape(nrows, ncols, 1)
    return array

# reshape
train_inputs = reshapeInput(train_inputs)
test_inputs = reshapeInput(test_inputs)

# to categorical (OneHotEncoding)
"""print("SHAPE")
print(test_outputs.shape)
print("SHAPE")
print(test_outputs.shape)"""

# test_outputs = to_categorical(test_outputs, 2)
# train_outputs = to_categorical(train_outputs, 2)

# ----- LSTM MODEL DEFINITION -----
def build_model(inputs, output_size):
    
    model = Sequential()
    model.add(LSTM(32))
    # model.add(Dropout(0.10))
    model.add(Dense(units = 1, activation='sigmoid'))
    #model.add(Activation('relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ----- TRAINING OF THE LSTM MODEL -----
if (not os.path.exists('LSTMupdown.joblib')):
    # initialise model architecture
    nn_model = build_model(train_inputs, output_size=1)

    nn_history = nn_model.fit(train_inputs, train_outputs, 
                                epochs=10, batch_size=2, verbose=2)
    
    # Saving the model to prevent re-training
    # dump(nn_model, 'LSTMupdown.joblib')
    # print('Dumped model')

else:
    print('Loaded model')
    nn_model = load('LSTMupdown.joblib')

# ----- Plot of prediction one data point ahead -----
nn_predict = nn_model.predict(test_inputs)
"""plt.plot(test_outputs, label = "actual")
plt.plot(nn_predict, label = "predicted")
plt.legend()
plt.show()"""
print("PREDICT")
print(nn_predict[:5])
print("ACTUAL")
print(test_outputs[:5])
#print("Accuracy score: {}".format(accuracy_score(test_outputs, nn_predict)))
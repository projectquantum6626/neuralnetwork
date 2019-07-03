"""
Dependencies:
pip install matplotlib statsmodels numpy pandas sklearn ploty keras
"""
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as smt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datetime as dt
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import plotly
import os
from joblib import dump, load # model persistence

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

df = pd.read_csv('input/processed/S&P.csv', sep=',')

inputs = df.drop(columns=['Movement']).values
outputs = df[['Movement']].values

# Split the training and test set
train_inputs, test_inputs = train_test_split(inputs, test_size=0.3)
train_outputs, test_outputs = train_test_split(outputs, test_size=0.3)
print("Train entries: " + str(len(train_inputs)))
print("Test entries: " + str(len(test_inputs)))

def reshape(array):
    nrows, ncols = array.shape
    array = array.reshape(nrows, ncols, 1)
    return array

# reshape
train_inputs = reshape(train_inputs)
test_inputs = reshape(test_inputs)


# ----- LSTM MODEL DEFINITION -----
def build_model(inputs, output_size, neurons, activ_func='sigmoid',
                dropout=0.10, loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy']):
    
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
# ----- TRAINING OF THE LSTM MODEL -----
if (not os.path.exists('LSTMupdown.joblib')):
    # initialise model architecture
    nn_model = build_model(train_inputs, output_size=1, neurons = 32)
    # model output is next price normalised to 10th previous closing price
    # train model on data
    # note: eth_history contains information on the training error per epoch
    nn_history = nn_model.fit(train_inputs, train_outputs, 
                                epochs=100, batch_size=1, verbose=2, shuffle=True)
    
    dump(nn_model, 'LSTMupdown.joblib')
    print('Dumped model')

else:
    print('Loaded model')
    nn_model = load('LSTMupdown.joblib')

# ----- Plot of prediction one data point ahead -----
nn_predict = nn_model.predict(test_inputs)
plt.plot(test_outputs, label = "actual")
plt.plot(nn_predict, label = "predicted")
plt.legend()
plt.show()
print("PREDICT")
print(nn_predict)
print("ACTUAL")
print(test_outputs)
print("Accuracy score: {}".format(accuracy_score(test_outputs, nn_predict)))


# ----- Plot of prediction 10 time steps ahead -----
"""
#https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/blob/master/lstm.py
def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

predictions = predict_sequence_full(nn_model, LSTM_test_inputs, 10)

plt.plot(LSTM_test_outputs, label="actual")
plt.plot(predictions, label="predicted")
plt.legend()
plt.show()
MAE = mean_absolute_error(LSTM_test_outputs, predictions)
print('The Mean Absolute Error is: {}'.format(MAE))
"""

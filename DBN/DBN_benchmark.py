# This program implements the DBN network for the S&P 500 Index
import pandas as pd
import numpy as np
import multiprocessing as mp
import os.path
import os.mkdir
import os.remove
import time
np.random.seed(1337)
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression


# This function formats a dates dataframe as the day of the year that it is in
def format_year(df):
    for i, row in df.iterrows():
        testDate = df.iloc[i,0]
        splitDate = testDate.split("/")
        year = int(splitDate[2])
        if year % 4 == 0:
            # This is a leap year so the days are different
            dates = pd.read_csv("data-files/dates_leap.csv", sep=',')
            testMonthDay = splitDate[0] + "/" + splitDate[1]
            index = dates.index[dates["Date"] == testMonthDay].values
            value = dates.iloc[index[0], 1]
            df.at[i, "Date"] = value
        else:
            # This is a normal year so the days are numbered normally
            dates = pd.read_csv("data-files/dates.csv", sep=',')
            testMonthDay = splitDate[0] + "/" + splitDate[1]
            index = dates.index[dates["Date"] == testMonthDay].values
            value = dates.iloc[index[0], 1]
            df.at[i, "Date"] = value
    return df


# This function runs a general DBN network on a given filename in deep-belief-network
# file must have the attributes: open, high, low, volume, and close to function properly and must also be a csv file
# The function take in:
# hidden_layers_structure = [int array]
# learning rate rbm = double
# learning rate = double
# number of epochs = int
# number of iterations of backprogagation = int
# batch size = int
# activation functions = 'relu'
def DBN_Run(hidden_layers_struc, learnin_rate_rbm, learnin_rate, num_epochs_rbm, num_iter_backprop, batchSize, activation_func):
    # This is a parameter for the start date of the model, it must be entered in the format Month/Day/Year like 2/6/2019
    # The program will find the next closest date to the one listed by iterating forware in the calendar (no leap years)
    # Enter post 1/3/1950 for Volume data to be there
    startDate = "6/26/2000"

    # defines a start time for the project job
    startTime = time.time()
    # Creates the output file director if it is not already created
    if not os.path.isdir("benchmark-output"):
        os.mkdir("benchmark-output")

    # Loading dataset
    SP_Data_Full = pd.read_csv("data-files/S&P_Movement.csv", sep=',')

    # Change start index based on the given start date and read in new dataframe
    startIndex = SP_Data_Full.index[SP_Data_Full["Date"] == startDate].values
    SP_Data = SP_Data_Full.iloc[startIndex[0]:-1]
    SP_Data.reset_index(inplace = True)

    # Shift the data set so that the model is reading in the previous day's information on the High, Low, Close, Volume
    # and Movement
    SP_Data["PrevHigh"] = SP_Data["High"].shift(-1)
    SP_Data["PrevLow"] = SP_Data["Low"].shift(-1)
    SP_Data["PrevClose"] = SP_Data["Close"].shift(-1)
    SP_Data["PrevVolume"] = SP_Data["Volume"].shift(-1)
    SP_Data["PrevMovement"] = SP_Data["Movement"].shift(-1)

    # split the new dataframe into features and targets
    target = SP_Data[["Close"], ["Movement"]]
    features = SP_Data[["Date", "Open", "PrevHigh", "PrevLow", "PrevVolume", "PrevMovement"]]

    features = format_year(features)

    # format the dataframe as a numpy array for the tensorflow functions
    target_array = target.values
    features_array = features.values
    X, Y = features_array, target_array

    # Splitting data into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

    # Data scaling
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)

    # Training
    regressor = SupervisedDBNRegression(hidden_layers_structure=hidden_layers_struc,
                                        learning_rate_rbm=learnin_rate_rbm,
                                        learning_rate=learnin_rate,
                                        n_epochs_rbm=num_epochs_rbm,
                                        n_iter_backprop=num_iter_backprop,
                                        batch_size=batchSize,
                                        activation_function=activation_func)
    regressor.fit(X_train, Y_train)

    trainingTime = time.time() - startTime
    # Test
    X_test = min_max_scaler.transform(X_test)
    Y_pred = regressor.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    testingTime = time.time() - startTime - trainingTime

    totalRunTime = time.time() - startTime
    # Text output file for more user friendly reading
    # formated based on the input into this function as the hyperparameters in order delimited by the _
    file = open("benchmark-output/result_" + str(hidden_layers_struc) + "_" + str(learnin_rate_rbm)  + "_" +
                str(learnin_rate) + "_" + str(num_epochs_rbm) + "_" + str(num_iter_backprop) + "_" + str(batchSize) +
                "_" + str(activation_func) + ".txt", "w+")
    file.write('Done.\nR-squared: %f\nMSE: %f' % (r2, mse) + "\n")
    file.write("Training Time: " + str(trainingTime) + "\n")
    file.write("Testing Time: " + str(testingTime) + "\n")
    file.write("Total Run Time: " + str(totalRunTime) + "\n\n")
    file.write("Network Information:")
    file.write("Hidden Layer Structure: " + str(hidden_layers_struc) + "\n")
    file.write("Learning Rate RBM: " + str(learnin_rate_rbm) + "\n")
    file.write("Learning Rate: " + str(learnin_rate) + "\n")
    file.write("Number of Epochs: " + str(num_epochs_rbm) + "\n")
    file.write("Number of Iterative Backpropogations: " + str(num_iter_backprop) + "\n")
    file.write("Batch Size: " + str(batchSize) + "\n")
    file.write("Activation Function: " + str(activation_func) + "\n")
    file.close()
    # CSV file output for use in data visualization
    hiddenlayerNumNodes = hidden_layers_struc[0]
    hiddenlayerNum = hidden_layers_struc[1]
    file = open("benchmark-output/result_" + str(hidden_layers_struc) + "_" + str(learnin_rate_rbm) + "_" + str(
        learnin_rate) + "_" + str(num_epochs_rbm) + "_" + str(num_iter_backprop) + "_" + str(batchSize) + "_" + str(
        activation_func) + ".csv", "w+")
    file.write("R-squared,MSE,trainingTime,testingTime,totalRunTime,hiddenlayerNumNodes,hiddenlayerNum,learnin_rate_rbm"
               ",learnin_rate,num_epochs_rbm,num_iter_backprop,batchSize,activation_func\n")
    file.write(str(r2) + "," + str(mse) + "," + str(trainingTime) + "," + str(testingTime) + "," + str(totalRunTime) +
               "," + str(hiddenlayerNumNodes) + "," + str(hiddenlayerNum) + "," + str(learnin_rate_rbm) + "," + str(learnin_rate) + "," +
               str(num_epochs_rbm) + "," + str(num_iter_backprop) + "," + str(batchSize) + "," + str(activation_func))
    file.close()


# Start of multiprocessing benchmarking code
# The different benchmarking tests are as follows
# Iterate over each hyperparameter: hidden_layer_strucutre [1,1] to [100,100] in increments of one for each total of 10,000 tests
# learning_rate_rbm: .001 to .1 in increments of .001
# learning_rate: .001 to .1 in increments of .001
# n_epochs_rbm: 10 to 100 in increments of 10
# n_iter_backprop: 10 to 1000 in increments of 10
# batch_size: 10 to 100 in increments of 10
# Do a nested for loop of all of these
# activation function: will try all of these different benchmarks on all the different activation functions
if os.path.exists("data-files/reset.csv"):
    reset_df = pd.read_csv("data-files/reset.csv")
    activation_func = reset_df.at[0, "activation_func"]
    if activation_func == "relu":
        for learnRate in range(reset_df.at[0, "learnRate"], 101):
            for learnRateRBM in range(reset_df.at[0, "learnRateRBM"], 101):
                for hiddenlayerNum in range(reset_df.at[0, "hiddenlayerNum"], 101):
                    for hiddenlayerNumNodes in range(reset_df.at[0, "hiddenlayerNumNodes"], 101):
                        for numEpochs in range(reset_df.at[0, "numEpochs"], 11):
                            for backpropNum in range(reset_df.at[0, "backpropNum"], 101):
                                pool = mp.Pool(10)
                                job_list = []
                                for batchSize in range(1, 11):
                                    job_list.append(([hiddenlayerNumNodes, hiddenlayerNum], learnRateRBM / 1000,
                                                     learnRate / 1000, numEpochs * 10, backpropNum * 10, batchSize * 10,
                                                     activation_func))
                                pool.starmap(DBN_Run, job_list)
                                pool.close()
                                pool.join()
                                # write the reset file
                                file = open("data-files/reset.csv", "w")
                                file.write(str(learnRate) + "," + str(learnRateRBM) + "," + str(hiddenlayerNum) + "," +
                                           str(hiddenlayerNumNodes) + "," + str(numEpochs) + "," + str(backpropNum) + "," +
                                           str(activation_func) + "\n")
                                file.close()
        # Activation function of sigmoid
        activation_func = 'sigmoid'
        for learnRate in range(1, 101):
            for learnRateRBM in range(1, 101):
                for hiddenlayerNum in range(1, 101):
                    for hiddenlayerNumNodes in range(1, 101):
                        for numEpochs in range(1, 11):
                            for backpropNum in range(1, 101):
                                pool = mp.Pool(10)
                                job_list = []
                                for batchSize in range(1, 11):
                                    job_list.append(([hiddenlayerNumNodes, hiddenlayerNum], learnRateRBM / 1000,
                                                     learnRate / 1000, numEpochs * 10, backpropNum * 10,
                                                     batchSize * 10, activation_func))
                                pool.starmap(DBN_Run, job_list)
                                pool.close()
                                pool.join()
                                # write the reset file
                                file = open("data-files/reset.csv", "w")
                                file.write(
                                    str(learnRate) + "," + str(learnRateRBM) + "," + str(hiddenlayerNum) + "," +
                                    str(hiddenlayerNumNodes) + "," + str(numEpochs) + "," + str(backpropNum) + "," +
                                    str(activation_func) + "\n")
                                file.close()
        if os.path.exists("data-files/reset.csv"):
            os.remove("data-files/reset.csv")
    else:
        for learnRate in range(1, 101):
            for learnRateRBM in range(1, 101):
                for hiddenlayerNum in range(1, 101):
                    for hiddenlayerNumNodes in range(1, 101):
                        for numEpochs in range(1, 11):
                            for backpropNum in range(1, 101):
                                pool = mp.Pool(10)
                                job_list = []
                                for batchSize in range(1, 11):
                                    job_list.append(([hiddenlayerNumNodes, hiddenlayerNum], learnRateRBM / 1000,
                                                     learnRate / 1000, numEpochs * 10, backpropNum * 10,
                                                     batchSize * 10, activation_func))
                                pool.starmap(DBN_Run, job_list)
                                pool.close()
                                pool.join()
                                # write the reset file
                                file = open("data-files/reset.csv", "w")
                                file.write(
                                    str(learnRate) + "," + str(learnRateRBM) + "," + str(hiddenlayerNum) + "," +
                                    str(hiddenlayerNumNodes) + "," + str(numEpochs) + "," + str(backpropNum) + "," +
                                    str(activation_func) + "\n")
                                file.close()
        if os.path.exists("data-files/reset.csv"):
            os.remove("data-files/reset.csv")
else:
    # create the reset file
    file = open("data-files/reset.csv", "w+")
    file.write("learnRate,learnRateRBM,hiddenlayerNum,hiddenlayerNumNodes,numEpochs,backpropNum,activation_func\n")
    file.close()
    # Activation function of relu
    activation_func = 'relu'
    for learnRate in range(1, 101):
        for learnRateRBM in range(1, 101):
            for hiddenlayerNum in range (1, 101):
                for hiddenlayerNumNodes in range(1, 101):
                    for numEpochs in range (1, 11):
                        for backpropNum in range(1, 101):
                            pool = mp.Pool(10)
                            job_list = []
                            for batchSize in range(1, 11):
                                job_list.append(([hiddenlayerNumNodes, hiddenlayerNum], learnRateRBM/1000,
                                                 learnRate/1000, numEpochs*10, backpropNum*10, batchSize*10, activation_func))
                            pool.starmap(DBN_Run, job_list)
                            pool.close()
                            pool.join()
                            # write the reset file
                            file = open("data-files/reset.csv", "w")
                            file.write(str(learnRate) + "," + str(learnRateRBM) + "," + str(hiddenlayerNum) + "," +
                                       str(hiddenlayerNumNodes) + "," + str(numEpochs) + "," + str(backpropNum) + "," +
                                       str(activation_func) + "\n")
                            file.close()
    # Activation function of sigmoid
    activation_func = 'sigmoid'
    for learnRate in range(1, 101):
        for learnRateRBM in range(1, 101):
            for hiddenlayerNum in range (1, 101):
                for hiddenlayerNumNodes in range(1, 101):
                    for numEpochs in range (1, 11):
                        for backpropNum in range(1, 101):
                            pool = mp.Pool(10)
                            job_list = []
                            for batchSize in range(1, 11):
                                job_list.append(([hiddenlayerNumNodes, hiddenlayerNum], learnRateRBM/1000,
                                                 learnRate/1000, numEpochs*10, backpropNum*10, batchSize*10, activation_func))
                            pool.starmap(DBN_Run, job_list)
                            pool.close()
                            pool.join()
                            # write the reset file
                            file = open("data-files/reset.csv", "w")
                            file.write(str(learnRate) + "," + str(learnRateRBM) + "," + str(hiddenlayerNum) + "," +
                                       str(hiddenlayerNumNodes) + "," + str(numEpochs) + "," + str(backpropNum) + "," +
                                       str(activation_func) + "\n")
                            file.close()
    if os.path.exists("data-files/reset.csv"):
        os.remove("data-files/reset.csv")
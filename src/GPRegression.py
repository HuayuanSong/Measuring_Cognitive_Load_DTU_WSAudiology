#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:19:41 2020

@author: hso
"""
# Import dependencies
from os import chdir, path
from data_load import getData
from scipy import signal
from signal_processing import normalize, shift
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF
from data_load import getData
from sklearn.model_selection import KFold
from sklearn import tree
from low_pass_filter import low_pass
import matplotlib.pyplot as plt
import pickle


def transform_to_SNR(data):
    """
    Input:
        data = Dataframe from dataLoad()

    Output:
        df   = Transformed dataframe with each trial as a single sample point
    """
    new_cols = ["trial", "SNR", "TA"]
    df = pd.DataFrame(columns = new_cols)
    df["trial"] = df["trial"].astype(object)
    conditions = np.unique(data["SNR"])
    TAs = np.unique(data["TA"])
    print("Transforming data...")
    for TA in TAs:
        print("TA: %i / %i" %(TA + 1, len(TAs)))
        for cond in conditions:
            print("\tCondition: %i / %i" %(cond + 1, len(conditions)))
            TA_data = data[(data["SNR"] == cond) & (data["TA"] == TA)]
            trials = np.unique(TA_data["trial"])
            for trial in trials:
                temp_data = []
                # Get the trial     data with all channels
                sample = TA_data[TA_data["trial"] == trial]
                # Transform the data into 1 dimension (this is to change)
                #sample = np.sum(sample[data.columns[:16]].values, axis = 1)

                # shift target 150 ms back
                EEG_shifted = shift(sample[sample.columns[:16]].T, lag = -150, freq = 64)
                sample = sample.iloc[:len(EEG_shifted.T)]
                sample[sample.columns[:16]] = EEG_shifted.T

                temp_row=[]

                for i in range(0,16):

                    crosscor = np.mean(signal.correlate(sample[data.columns[i]],sample['target'], mode = 'same')/len(sample['target']))
                    temp_row.append(crosscor)

                # Collect the data into a temporary list
                temp_data.append(temp_row)
                temp_data.append(cond)
                temp_data.append(TA)
                # Add it all into a dataframe row
                row = pd.DataFrame([temp_data], columns = new_cols)
                # Add the row to the total dataframe
                df = pd.concat([row, df], ignore_index = True)
    return df

def classify_SNR(data, TA = 0, DT_config = (2,21,1), name = None, kernel = None):
    """
    Parameters
    ----------
    data : Pandas dataframe
        Load data from getData().
    TA : Int, optional
        Test subject number. The default is 0.
    tr : Tuple, optional
        Range of max depth values for decision tree classifier. The default is (2,21,1).

    Returns
    -------
    Gaussian process regression with two-layer 15-fold CV. Decision tree classifier trained
    on validation set in inner loops; the DTCs are saved to pickle files. DTC with highest
    accuracy is loaded for evaluation on GPR performance on test data in outer loops.

    """
    if (name == None):
        name = 'data_gp_%i.csv' %TA

    # Values for tree max depth
    DT_params = np.arange(DT_config[0],
                          DT_config[1],
                          DT_config[2])

    # For reproduciblity
    random_state = 999

    data = data[data['TA']==TA]

    # Restructure data matrix
    data = transform_to_SNR(data)

    # Shuffle data
    data = data.sample(frac=1,random_state=random_state)

    # Define X and y
    X = data['trial']
    y = data['SNR']

    # Convert labels to [-5,0,5] for SNRs -5, 0, +5 DB, respectively
    y = y.replace(0,-5)
    y = y.replace(1,0)
    y = y.replace(2,5)

    # Initiate kernel for GPR
    if kernel == None:
        kernel = DotProduct()
    #kernel_lin = DotProduct()
    #kernel_sqd = RBF(length_scale = len(X[0]))
    #kernel_sum = kernel_lin + kernel_sqd

    ## 2-layer 15-fold cross-validation ##
    kf = KFold(n_splits=15)

    scores = []

    # Outer fold
    i = 0
    for out_train_idx, out_test_idx in kf.split(X, y):
        print("Outer fold %i / %i" %(i + 1, kf.get_n_splits(X,y)))

        X_train = X.iloc[out_train_idx]
        y_train = y.iloc[out_train_idx]

        X_test = X.iloc[out_test_idx]
        y_test = y.iloc[out_test_idx]

        # Initiate errors for inner fold validations
        vals = np.zeros((kf.get_n_splits(X_train,y_train), len(DT_params)))

        # Inner fold
        j = 0
        for inn_train_idx, inn_test_idx in kf.split(X_train, y_train):
            print("\t Inner fold %i / %i" %(j + 1, kf.get_n_splits(X_train,y_train)))

            inn_X_train = X_train.iloc[inn_train_idx]
            inn_y_train = y_train.iloc[inn_train_idx]

            inn_X_test = X_train.iloc[inn_test_idx]
            inn_y_test = y_train.iloc[inn_test_idx]

            inn_X_train = np.reshape(list(inn_X_train), ((len(inn_X_train), len(X_train.iloc[0]))))
            inn_y_train = inn_y_train.values.reshape(inn_X_train.shape[0], 1)

            inn_X_test = np.reshape(list(inn_X_test), ((len(inn_X_test), len(X_test.iloc[0]))))
            inn_y_test = inn_y_test.values.reshape(inn_X_test.shape[0], 1)

            GP_model = GaussianProcessRegressor(kernel = kernel, random_state = random_state)
            GP_model.fit(inn_X_train, inn_y_train)
            DT_train = GP_model.predict(inn_X_train, return_std=False)
            DT_test = GP_model.predict(inn_X_test, return_std=False)

            k = 0
            # Loop through max depth values for DTC
            for param in DT_params:
                # Decision tree classifier
                DT_model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = param)

                # Fit DTC
                DT_model.fit(DT_train, inn_y_train)

                # Save DTC model for possible later use (choose model with highest acc. later)
                #DT_file = open("model_%s.sav" % k, 'wb')
                #pickle.dump(DT_model, DT_file)
                #DT_file.close()

                # Test DTC on y_pred for GPR and hold it up against y_test (inner)
                val = DT_model.score(DT_test, inn_y_test)

                # Add score to matrix
                vals[j, k] = val

                k += 1
            j += 1

        X_train = np.reshape(list(X_train), ((len(X_train), len(X.iloc[0]))))
        y_train = y_train.values.reshape(X_train.shape[0], 1)

        X_test = np.reshape(list(X_test), ((len(X_test), len(X.iloc[0]))))
        y_test = y_test.values.reshape(X_test.shape[0], 1)

        # Get optimal DTC model
        param_score = np.sum(vals, axis = 0)
        idx = np.argmax(param_score)
        best_param = DT_params[idx]

        opt_DT_model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = best_param)

        GP_model = GaussianProcessRegressor(kernel = kernel, random_state = random_state)
        GP_model.fit(X_train, y_train)
        DT_train = GP_model.predict(X_train, return_std = False)
        DT_test = GP_model.predict(X_test, return_std = False)

        opt_DT_model.fit(DT_train, y_train)
        preds = opt_DT_model.predict(DT_test)
        # Test DTC on y_pred for GPR and hold it up against y_test (outer)
        #score = opt_DT_model.score(DT_test, y_test)

        # Append score to list
        #scores.append(score)
        for j in range(len(preds)):
            scores.append([preds[j], y_test[j][0]])

        i += 1

    # Make data matrix
    #data_matrix = np.array(['TA:',TA,'Score:',np.mean(scores)]).T
    return scores
    # Save as CSV in working directory
    #np.savetxt(name,data_matrix, delimiter= "," ,fmt='%s' )


if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))

    data = getData()

    # Apply lowpass filter
    data = low_pass(data,64, 8)
    print(data)
    print(transform_to_SNR(data))
    #results = []

    kern_lin = DotProduct()
    kern_sqd = RBF(length_scale = 16)
    kern_sum = kern_lin + kern_sqd

    TAs = np.uniqe(data["TA"])

    for i in TAs:
        res_lin = classify_SNR(data, TA = i, kernel = kern_lin)
        res_sqd = classify_SNR(data, TA = i, kernel = kern_sqd)
        res_sum = classify_SNR(data, TA = i, kernel = kern_sum)

        n_sample = len(res_lin)
        plt.plot(np.arange(n_sample), np.asarray(res_lin)[:, 1], label = "True")
        plt.scatter(np.arange(n_sample), np.asarray(res_lin)[:, 0]-0.1, label = "lin")
        plt.scatter(np.arange(n_sample), np.asarray(res_sqd)[:, 0]-0.2, label = "sqd")
        plt.scatter(np.arange(n_sample), np.asarray(res_sum)[:, 0]-0.3, label = "sum")
        plt.legend()
        plt.grid(True)
        plt.title("Kernel perfomance, TA = %i" %i)
        plt.show()

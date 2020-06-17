#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:34:36 2020

"""

# Load dependencies
from os import path, chdir
from data_load import getData, random_trial
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.layers.normalization import BatchNormalization

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Pearson R correlation Loss function for ANN model from Taillez et al (2017)
# https://github.com/tdetaillez/neural_networks_auditory_attention_decoding
def corr_loss(act,pred):          
    cov=(K.mean((act-K.mean(act))*(pred-K.mean(pred))))
    return 1-(cov/(K.std(act)*K.std(pred)+K.epsilon()))

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


def cross_validate(data, TA = None, node_min=4,node_max=16,n_nodes=7):
    """

    Parameters
    ----------
    data : Pandas Dataframe
        load dataframe using from data_load, using getData function.
    lambda_config : tuple, optional
        Lambda range for Ridge regression. The default is (2e0, 2e20, 11). Range from
        Cross et al 2016 publication.
    t_config : tuple, optional
        Jitter lag range for MNE in ms. The default is (-.25, .1).


    Returns
    -------
    NA

    """

    np.random.seed(999)

    # Define range of nodes to optimize for training models
    nodes = np.linspace(node_min,node_max,n_nodes)
    nodes = nodes.astype(int)

    # Define result DataFrame
    df_cols = ["corr_true", "corr_mask", "corr_rand", "TA", "SNR","Optimal Nr. Nodes"]
    df = pd.DataFrame(columns = df_cols)

    if TA == None:
        TAs = np.unique(data["TA"])
    else:
        TAs = np.array([TA])

    for TA in TAs:
        data_sub = data[data["TA"] == TA]
        data_train = data_sub

        SNRs = np.unique(data_sub["SNR"])
        SNR_order = []

        for SNR in SNRs:
            trials = data_sub[data_sub["SNR"] == SNR]["trial"] # Get the trials
            trials = np.unique(trials) # Get the unique trial indicies
            np.random.shuffle(trials) # Shuffle the order of the trials
            SNR_order.append(trials) # Store the order

        # Get the lowest possible k for k-fold
        K = np.inf
        for order in SNR_order:
            if len(order) < K:
                K = len(order)

        # Outer fold
        for k in range(K):
            # Split into test and training
            data_train = data_sub
            data_test = pd.DataFrame()
            # Filter the test data away
            for i in range(len(SNR_order)):
                data_test = pd.concat([data_test, data_train[(data_sub["SNR"] == i) & (data_train["trial"] == SNR_order[i][k])]], ignore_index = True)
                data_train = data_train.drop(data_train[(data_train["SNR"] == i) & (data_train["trial"] == SNR_order[i][k])].index)

            # Initiate errors for inner fold validations
            vals = np.zeros((K-1, n_nodes))

            # Get the list of validation trials
            SNR_valid_order = SNR_order.copy()
            for i in range(len(SNR_order)):
                SNR_valid_order[i] = np.delete(SNR_valid_order[i],k)

            # Inner fold
            for j in range(K-1):
                print("TA: %i / %i\n\tFold: %i / %i\n\tInner fold: %i / %i" %(TA + 1, len(TAs), k + 1, K, j + 1, K-1))
                # Find optimal hyperparameter

                data_valid_train = data_train
                data_valid_test = pd.DataFrame()

                for i in range(len(SNR_order)):
                    data_valid_test = pd.concat([data_valid_test, data_valid_train[(data_valid_train["SNR"] == i) & (data_valid_train["trial"] == SNR_valid_order[i][j])]], ignore_index = True)
                    data_valid_train = data_valid_train.drop(data_valid_train[(data_valid_train["SNR"] == i) & (data_valid_train["trial"] == SNR_valid_order[i][j])].index)

                i = 0
                for l in nodes:
                # Define model with l parameter
                   model = Sequential()
                   model.add(BatchNormalization())
                   model.add(Dense(units=l, activation='tanh',input_shape=(np.asarray(data_valid_train[data.columns[:16]]).shape[0], np.asarray(data_valid_train[data.columns[:16]]).shape[1])))
                   model.add(Dropout(0.2))
                
                   # Batch normalization
                   model.add(BatchNormalization())
                   model.add(Dense(units=2, activation='tanh'))
                   model.add(Dropout(0.5))
                    
                   # Batch normalization
                   model.add(BatchNormalization())
                   model.add(Dense(units=1, activation='linear'))
                   model.compile(optimizer='rmsprop', loss=corr_loss)
                   model.fit(np.asarray(data_valid_train[data.columns[:16]]), np.asarray(data_valid_train["target"]), epochs=30, verbose=2, shuffle=False)     

                   # Compute cross correlation for regressional value
                   val = np.zeros(len(SNR_order))
                   for i_ in range(len(SNR_order)):
                       val[i_] = pearsonr(np.asarray(data_valid_test[data_valid_test["SNR"] == i_]["target"]).reshape(-1, 1),model.predict(np.asarray(data_valid_test[data_valid_test["SNR"] == i_][data.columns[:16]])))[0]
                   # Add score to matrix
                   vals[j, i] = np.mean(val)

                   i += 1
                j += 1

            # Get optimal parameter
            param_score = np.sum(vals, axis = 0)

            node_opt = nodes[np.argmax(param_score)]
            print("Optimal lambda = %f" %node_opt)

            # Train optimal model        
            model_opt = Sequential()
            model_opt.add(BatchNormalization())
            model_opt.add(Dense(units=node_opt, activation='tanh',input_shape=(np.asarray(data_train[data.columns[:16]]).shape[0], np.asarray(data_train[data.columns[:16]]).shape[1])))
            model_opt.add(Dropout(0.2))
                
            # Batch normalization
            model_opt.add(BatchNormalization())
            model_opt.add(Dense(units=2, activation='tanh'))
            model_opt.add(Dropout(0.5))
                
            # Batch normalization
            model_opt.add(BatchNormalization())
            model_opt.add(Dense(units=1, activation='linear'))
            model_opt.compile(optimizer='rmsprop', loss=corr_loss)

            # Fit model to train data
            model_opt.fit(np.asarray(data_train[data.columns[:16]]), np.asarray(data_train["target"]), epochs=30, verbose=2, shuffle=False)     

            for i in range(len(SNR_order)):
                # Predict envelope
                data_test_SNR = data_test[data_test["SNR"] == i]
                y_pred = model_opt.predict(np.asarray(data_test_SNR[data.columns[:16]]))
                y_rand = random_trial(data, TA = TA, trial = SNR_order[i][k])["target"]

                corr_true = pearsonr(y_pred, np.asarray(data_test_SNR["target"]).reshape(-1, 1))
                corr_mask = pearsonr(y_pred, np.asarray(data_test_SNR["mask"]).reshape(-1, 1))
                corr_rand = pearsonr(y_pred, np.asarray(y_rand).reshape(-1, 1))

                # Convert to DataFrame
                data_results = np.zeros((1, len(df_cols)))
                data_results[:, 0] = corr_true[0]
                data_results[:, 1] = corr_mask[0]
                data_results[:, 2] = corr_rand[0]
                data_results[:, 3] = TA
                data_results[:, 4] = node_opt
                data_results[:, 5] = i

                df_ = pd.DataFrame(data = data_results, columns = df_cols)

                # Concatenate
                df = pd.concat([df, df_], ignore_index = True)
            df.to_pickle("local_data/results/ALL_SNR_2_layers_ANN_result_%i_%i.pkl" %(TA, k))
    print("Done")
    return df

if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))

    # Get the data
    data = getData()

    # Run the script
    results = cross_validate(data,node_min=4,node_max=16,n_nodes=7)


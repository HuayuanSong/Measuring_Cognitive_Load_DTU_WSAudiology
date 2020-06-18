  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict speech envelope from EEG using ANN

Build ANN model with 2 hidden layers using Keras; one for every SNR value; I.e., -5, 0 & 5 DB, respectively. 
Trains using 2-layer leave-trial-out CV, to find number of nodes in the first hidden layer. 
Pearson's R correlation coefficient is used as loss function; and for validation.
    
"""

# Load dependencies
from os import path, chdir
from sklearn.model_selection import LeaveOneGroupOut
from data_load import getData, random_trial
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
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

# Modified above function to use for Pearson R correlation coefficient calculation
def corr(act,pred):          
    cov=(K.mean((act-K.mean(act))*(pred-K.mean(pred))))
    return (cov/(K.std(act)*K.std(pred)+K.epsilon()))

def cross_validate(data,node_min=4,node_max=16,n_nodes=7):
    
    """

    Parameters
    ----------
    data : Pandas Dataframe
        load dataframe using from data_load, using getData function.
    node_min : Int, optional
        Min nr. of nodes in range of nodes to loop over in inner CV loop
    node_max : Int, optional
        Max nr. of nodes in range of nodes to loop over in inner CV loop
    n_nodes : Int, optional
        Number, N, of nodes to loop over in inner CV loop
        
    Returns
    -------
    Pickle file with dataframe of results;

    """

    # Define range of nodes to optimize for training models
    nodes = np.linspace(node_min,node_max,n_nodes)
    nodes = nodes.astype(int)

    # Split into train/testing with leave-one-group-out
    logo = LeaveOneGroupOut()

    # Define result DataFrame
    df_cols = ["corr_true", "corr_mask", "corr_rand", "TA", "SNR","Optimal Nr. Nodes"]
    df = pd.DataFrame(columns = df_cols)

    TAs = np.unique(data["TA"])

    for TA in TAs:
        SNRs = np.unique(data[data["TA"] == TA]["SNR"])

        for SNR in SNRs:
            data_sub = data[(data["TA"] == TA) & (data["SNR"] == SNR)]

            # Assign X, y and group variable (trial, as to do leave-trial-out)
            X = data_sub[data.columns[:16]]
            
            # Attended audio
            y = data_sub["target"]
            
            # Unattended audio
            masks = data_sub["mask"]
            
            groups = data_sub["trial"]
            n_outer_groups = len(np.unique(groups))

            ### Two-layer CV starts here ###
            # Outer fold
            i = 0
            for out_train_idx, out_test_idx in logo.split(X, y, groups):

                X_train = X.iloc[out_train_idx]
                y_train = y.iloc[out_train_idx]

                X_test = X.iloc[out_test_idx]
                y_test = y.iloc[out_test_idx]

                # Define inner groups, these are n - 1 of n total groups
                inner_groups = data["trial"].iloc[out_train_idx]
                n_inner_groups = len(np.unique(inner_groups))

                # Initiate errors for inner fold validations
                vals = np.zeros((n_inner_groups, n_nodes))

                # Inner fold
                j = 0
                for inn_train_idx, inn_test_idx in logo.split(X_train, y_train, inner_groups):
                    print("TA = %i / %i\tSNR = %i / %i\nOuter fold %i / %i\t Inner fold %i / %i" %(TA + 1, len(TAs), SNR + 1, len(SNRs), i + 1, n_outer_groups, j + 1, n_inner_groups))

                    inn_X_train = X_train.iloc[inn_train_idx]
                    inn_y_train = y_train.iloc[inn_train_idx]

                    inn_X_test = X_train.iloc[inn_test_idx]
                    inn_y_test = y_train.iloc[inn_test_idx]

                    # Validate model with all parameters
                    k = 0
                    for l in nodes:
                        # Define model with l parameter
                        model = Sequential()
                        
                        # Batch normalization
                        model.add(BatchNormalization())
                        model.add(Dense(units=l, activation='tanh',input_shape=(inn_X_train.shape[0], inn_X_train.shape[1])))
                        model.add(Dropout(0.2))
                        
                        # Batch normalization
                        model.add(BatchNormalization())
                        model.add(Dense(units=2, activation='tanh'))
                        model.add(Dropout(0.5))
                        
                        # Batch normalization
                        model.add(BatchNormalization())
                        model.add(Dense(units=1, activation='linear'))
                        model.compile(optimizer='rmsprop', loss=corr_loss)
                        model.fit(np.asarray(inn_X_train), np.asarray(inn_y_train), epochs=30, verbose=2, shuffle=False)     

                        results = model.evaluate(np.asarray(inn_X_test), np.asarray(inn_y_test))
                        # Compute Pearson R correlation for regressional value
                        val = results
                        vals[j, k] = val

                        k += 1

                    j += 1

                # Get optimal parameter
                param_score = np.sum(vals, axis = 0)

                node_opt = nodes[np.argmax(param_score)]
                print("Optimal nodes = %f" %node_opt)

                # Train optimal model               
                model_opt = Sequential()
                
                # Batch normalization
                model_opt.add(BatchNormalization())
                model_opt.add(Dense(units=node_opt, activation='tanh',input_shape=(X_train.shape[0], X_train.shape[1])))
                model_opt.add(Dropout(0.2))
                
                # Batch normalization
                model_opt.add(BatchNormalization())
                model_opt.add(Dense(units=2, activation='tanh'))
                model_opt.add(Dropout(0.5))
                
                # Batch normalization
                model_opt.add(BatchNormalization())
                model_opt.add(Dense(units=1, activation='linear'))
                model_opt.compile(optimizer='rmsprop', loss=corr_loss)
                model_opt.fit(np.asarray(X_train), np.asarray(y_train), epochs=30, verbose=2, shuffle=False)     

                # Predict envelope
                y_pred = model_opt.predict(np.asarray(X_test))
                
                """
                plt.style.use('ggplot')
                plt.plot(y_test, label="True value")
                plt.plot(y_pred, label="Predicted value")
                plt.legend()
                plt.show()
                """

                trial_test = np.unique(data_sub.iloc[out_test_idx]["trial"])[0]
                
                # Random speech 
                y_rand = random_trial(data, TA = TA, trial = trial_test)["target"]
                
                # Compute Pearson R between predicted envelope and attended speech
                corr_true = corr(K.constant(np.asarray(y_test)),K.constant(y_pred))
                
                # Compute Pearson R between predicted envelope and unattended speech
                corr_mask = corr(K.constant(np.asarray(masks.iloc[out_test_idx])),K.constant(y_pred))
                
                # Compute Pearson R between predicted envelope and random speech
                corr_rand = corr(K.constant(np.asarray(y_rand)),K.constant(y_pred))

                # Evaluate envelope, compare with random trial
                ### Add correlations to dataframe ###
                # Convert to DataFrame
                data_results = np.zeros((1, len(df_cols)))
                data_results[:, 0] = np.asarray(corr_true)
                data_results[:, 1] = np.asarray(corr_mask)
                data_results[:, 2] = np.asarray(corr_rand)
                data_results[:, 3] = TA
                data_results[:, 4] = SNR
                data_results[:, 5] = node_opt

                df_ = pd.DataFrame(data = data_results, columns = df_cols)

                # Concatenate
                df = pd.concat([df, df_], ignore_index = True)
                print(df)

                i += 1
            df.to_pickle("local_data/results/Seperate_SNR_2_layer_ANN_result_%i_%i.pkl" %(TA, SNR))
    return df


if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))

    # Get the data
    data = getData()

    # Run the script
    results = cross_validate(data, node_min=4,node_max=16,n_nodes=7)

    #results = pd.read_pickle("local_data/ANN_1_Hidden_FCs_results.pkl")

    results["SNR"] = results["SNR"] * 5 - 5
    results["SNR"] = results["SNR"].astype("category")
    results["TA"] = results["TA"].astype("category")
    results = results.groupby(['SNR', 'TA']).mean().reset_index()

    dd=pd.melt(results,id_vars=['SNR','TA'],value_vars=['corr_true','corr_rand'],var_name='envelopes')
    dd["envelopes"] = dd["envelopes"].replace("corr_true","Target")
    dd["envelopes"] = dd["envelopes"].replace("corr_rand","Random")
    
    ## Boxplots ##
    sns.set(style="darkgrid")
    sns.boxplot(x='SNR',y='value',data=dd,hue='envelopes')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Score (corrcoef)")
    plt.title("Boxplot of results")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.show()

    sns.lineplot(x='TA',y='value',data=dd[dd["envelopes"]=="Target"], label="Target")
    sns.lineplot(x='TA',y='value',data=dd[dd["envelopes"]=="Random"], label="Random")
    plt.xlabel("TA")
    plt.ylabel("Score (corrcoef)")
    plt.title("Performance per TA")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.show()

# Load dependencies
from os import path, chdir
from sklearn.model_selection import LeaveOneGroupOut
from data_load import getData, random_trial
import numpy as np
from mne.decoding import ReceptiveField
from scipy import signal

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import seaborn as sns

def cross_validate(data, lambda_config = (2**0, 2**20, 11), t_config = (-.25, .1)):
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

    # Define lambda values for training models
    lambdas = np.linspace(lambda_config[0],
                          lambda_config[1],
                          lambda_config[2])

    # Parameters for MNE
    tmin, tmax = t_config
    sfreq = 64

    # Split into train/testing with leave-one-group-out
    logo = LeaveOneGroupOut()

    # Define result DataFrame
    df_cols = ["corr_true", "corr_mask", "corr_rand", "TA", "SNR"]
    df = pd.DataFrame(columns = df_cols)

    TAs = np.unique(data["TA"])
    #TAs = np.array([4, 5, 6])

    for TA in TAs:
        SNRs = np.unique(data[data["TA"] == TA]["SNR"])

        for SNR in SNRs:
            data_sub = data[(data["TA"] == TA) & (data["SNR"] == SNR)]

            # Assign X, y and group variable (trial, as to do leave-trial-out)
            X = data_sub[data.columns[:16]]
            y = data_sub["target"]
            masks = data_sub["mask"]
            groups = data_sub["trial"]
            n_outer_groups = len(np.unique(groups))

            ### Two-layer CV starts here ###
            # Outer fold
            i = 0
            for out_train_idx, out_test_idx in logo.split(X, y, groups):
                #print("Outer fold %i / %i" %(i + 1, n_outer_groups))

                X_train = X.iloc[out_train_idx]
                y_train = y.iloc[out_train_idx]

                X_test = X.iloc[out_test_idx]
                y_test = y.iloc[out_test_idx]

                # Define inner groups, these are n - 1 of n total groups
                inner_groups = data["trial"].iloc[out_train_idx]
                n_inner_groups = len(np.unique(inner_groups))

                # Initiate errors for inner fold validations
                vals = np.zeros((n_inner_groups, lambda_config[2]))

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
                    for l in lambdas:
                        # Define model with l parameter
                        model = ReceptiveField(tmin, tmax, sfreq, feature_names = list(data.columns[:16]),
        					estimator = l,
        					scoring = "corrcoef")

                        # Fit model to inner fold training data
                        model.fit(np.asarray(inn_X_train), np.asarray(inn_y_train))

                        # Compute cross correlation for regressional value
                        val = model.score(np.asarray(inn_X_test), np.asarray(inn_y_test))
                        # Add score to matrix
                        vals[j, k] = val

                        k += 1
                    j += 1

                # Get optimal parameter
                param_score = np.sum(vals, axis = 0)

                #plt.title("Lambda scores, TA: %i, SNR: %i" %(TA, SNR))
                #plt.plot(lambdas, param_score)
                #plt.xlabel("Lambda value")
                #plt.ylabel("Correlation score")
                #plt.show()

                lambda_opt = lambdas[np.argmax(param_score)]
                print("Optimal lambda = %f" %lambda_opt)

                # Train optimal model
                model_opt = ReceptiveField(tmin, tmax, sfreq, feature_names = list(data.columns[:16]),
        					estimator = lambda_opt,
        					scoring = "corrcoef")

                # Fit model to train data
                model_opt.fit(np.asarray(X_train), np.asarray(y_train))
                # Predict envelope
                y_pred = model_opt.predict(np.asarray(X_test))

                #plt.plot(y_pred)
                #plt.plot(y_test)
                #plt.show()

                trial_test = np.unique(data_sub.iloc[out_test_idx]["trial"])[0]
                y_rand = random_trial(data, TA = TA, trial = trial_test)["target"]

                corr_true = pearsonr(y_pred, np.asarray(y_test))
                corr_mask = pearsonr(y_pred, np.asarray(masks.iloc[out_test_idx]))
                corr_rand = pearsonr(y_pred, np.asarray(y_rand))

                # Evaluate envelope, compare with random trial
                ### Add correlations to dataframe ###
                # Convert to DataFrame
                data_results = np.zeros((1, len(df_cols)))
                data_results[:, 0] = corr_true[0]
                data_results[:, 1] = corr_mask[0]
                data_results[:, 2] = corr_rand[0]
                data_results[:, 3] = TA
                data_results[:, 4] = SNR

                df_ = pd.DataFrame(data = data_results, columns = df_cols)

                # Concatenate
                df = pd.concat([df, df_], ignore_index = True)

                i += 1
            df.to_pickle("local_data/results/result_%i_%i.pkl" %(TA, SNR))
    return df


if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))

    # Get the data
    data = getData()

    # Run the script
    results = cross_validate(data, lambda_config = (2**0, 2**20, 11), t_config = (-.25, .1))

    #results = pd.read_pickle("local_data/results.pkl")

    results["SNR"] = results["SNR"] * 5 - 5
    results["SNR"] = results["SNR"].astype("category")
    results["TA"] = results["TA"].astype("category")
    results = results.groupby(['SNR', 'TA']).mean().reset_index()


    dd=pd.melt(results,id_vars=['SNR','TA'],value_vars=['corr_true','corr_rand'],var_name='envelopes')
    dd["envelopes"] = dd["envelopes"].replace("corr_true","Target")
    dd["envelopes"] = dd["envelopes"].replace("corr_rand","Random")

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

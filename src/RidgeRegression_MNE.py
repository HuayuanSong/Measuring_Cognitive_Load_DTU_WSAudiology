"""
Ridge Regression with MNE package. 

EEG is shifted 150 MS back.

Performs ridge regression with MNE package built-in model. Trains using 2-layer
leave-trial-out CV, to find optimal lambda parameter.
    
The optimal lambda model is used to train a ridge regression model,
on 80/20 split data for training/testing. Tested data with purely -5 DB, 0, +5 DB SNR & Random speech
to compare cross correlation between predicted and target envelopes
"""

# Load dependencies
from os import path, chdir
from sklearn.model_selection import LeaveOneGroupOut
from data_load import getData
import numpy as np
from mne.decoding import ReceptiveField
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from low_pass_filter import low_pass
from scipy import signal
import pandas as pd
from signal_processing import shift
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

def cross_validate(data, lambda_config = (2e0, 2e20, 11),TA=0,name="data_0"):
    """

    Parameters
    ----------
    data : Pandas Dataframe
        load dataframe using from data_load, using getData function.
    lambda_config : tuple, optional
        Lambda range for Ridge regression. The default is (2e0, 2e20, 11). Range from 
        Cross et al 2016 publication.
    TA : int., optional
        Test subject nr. The default is 0.
    name : str., optional
        Name of txt file with model performance. The default is "data_0".

    Returns
    -------
    CSV-file with results
    Plots

    """
    
    # For reproducibility
    random_state = 9999
    
    # Train one model per TA
    data = data[data['TA']==TA]
    
    # Define lambda values for training models
    lambdas = np.linspace(lambda_config[0],
                          lambda_config[1],
                          lambda_config[2])

    # Split into train/testing with leave-one-group-out
    logo = LeaveOneGroupOut()
    
    # Shift EEG 150 ms back
    EEG_shifted = shift(data[data.columns[:16]].T, lag = -150, freq = 64)
    data = data.iloc[:len(EEG_shifted.T)]
    data[data.columns[:16]] = EEG_shifted.T
    
    # Assign X, y and group variable
    X = data[data.columns[:16]]   
    y = data['target']

    groups = data["trial"]
    n_outer_groups = len(np.unique(groups))

    # Initiate test errors
    MSEs = []
    MSEdummies = []
    
    # For cross correlation
    scores = []
    opt_lambda_list = []
    
    # Parameters for MNE
    tmin = -.25
    tmax = .1
    sfreq = 64
    
    ## Leave-trial-out CV ##
    
    # Outer fold
    i = 0
    for out_train_idx, out_test_idx in logo.split(X, y, groups):
        print("Outer fold %i / %i" %(i + 1, n_outer_groups))

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
            print("\t Inner fold %i / %i" %(j + 1, n_inner_groups))

            inn_X_train = X_train.iloc[inn_train_idx]
            inn_y_train = y_train.iloc[inn_train_idx]

            inn_X_test = X_train.iloc[inn_test_idx]
            inn_y_test = y_train.iloc[inn_test_idx]

            # Validate model with all parameters
            k = 0
            for l in lambdas:
                # Define model with l parameter
                model = ReceptiveField(tmin, tmax, sfreq, feature_names = None,
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
        lambda_opt = lambdas[np.argmax(param_score)]
        print("Optimal lambda = %f" %lambda_opt)
        
        # Store optimal lambda parameter
        opt_lambda_list.append(lambda_opt)

        # Train optimal model
        model_opt = ReceptiveField(tmin, tmax, sfreq, feature_names = None,
					estimator = lambda_opt,
					scoring = "corrcoef")
        
        # Fit model to inner fold training data
        model_opt.fit(np.asarray(X_train), np.asarray(y_train))
        
        # Compute error of optimal model
        score = model_opt.score(np.asarray(X_test), np.asarray(y_test))
        print('Score:')
        print(score)
        
        # Fit dummy model
        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(np.asarray(X_train), np.asarray(y_train))
        
        # Add error to list
        scores.append(score)
        MSE = mean_squared_error(np.asarray(y_test), model_opt.predict(np.asarray(X_test)), squared=True)
        MSEs.append(MSE)
        
        MSEdummy = mean_squared_error(np.asarray(y_test),dummy_regr.predict(np.asarray(X_test)), squared=True)
        MSEdummies.append(MSEdummy)

        i += 1
    
    ## Training and testing for optimal model ##
    
    # Making dataframes for each SNR cond
    data_0 = data[data['SNR']==0]
    data_1 = data[data['SNR']==1]
    data_2 = data[data['SNR']==2]
    
    # Shuffle data 
    data_0 = data_0.sample(frac=1,random_state=random_state)
    data_1 = data_1.sample(frac=1,random_state=random_state)
    data_2 = data_2.sample(frac=1,random_state=random_state)
    
    # Split data 80/20 for training/testing 
    train_0, test_0 = train_test_split(data_0, test_size=0.2,random_state=random_state)
    train_1, test_1 = train_test_split(data_1, test_size=0.2,random_state=random_state)
    train_2, test_2 = train_test_split(data_2, test_size=0.2,random_state=random_state)
    
    # Combine training dataframes into one
    data = train_0.append(train_1, ignore_index=True)
    data = data.append(train_2, ignore_index=True)    
    
    # Combine testing dataframes into one
    data_test = test_0.append(test_1,ignore_index=True)
    data_test = data_test.append(test_2, ignore_index=True)   
    
    # Mean score across all folds
    mu_score = np.mean(scores)
    print("Mean score = %f" %mu_score)

    best_fold = np.argmax(scores) + 1
    print("Best fold = %i" %best_fold)
    
    # Optimal model
    model_optimal = ReceptiveField(tmin, tmax, sfreq, feature_names = None,
					estimator = opt_lambda_list[np.argmax(scores)],
					scoring = "corrcoef")
                
    # Fit optimal model to training data
    model_optimal.fit(np.asarray(data[data.columns[:16]]), np.asarray(data["target"]))
    
    # Dummy classifier
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(np.asarray(data[data.columns[:16]]), np.asarray(data["target"]))
                
    # Compute cross correlation scores on test data for all three SNR conds
    score_0 = signal.correlate(model_optimal.predict(np.asarray(test_0[test_0.columns[:16]])),np.asarray(test_0['target']),mode='same')/len(test_0['target'])
    score_1 = signal.correlate(model_optimal.predict(np.asarray(test_1[test_1.columns[:16]])),np.asarray(test_1['target']),mode='same')/len(test_1['target'])
    score_2 = signal.correlate(model_optimal.predict(np.asarray(test_2[test_2.columns[:16]])),np.asarray(test_2['target']),mode='same')/len(test_2['target'])
    
    # Compute cross correlation scores on test data for all three SNR conds with dummy regressor
    score_0_dummy = signal.correlate(dummy_regr.predict(np.asarray(test_0[test_0.columns[:16]])),np.asarray(test_0['target']),mode='same')/len(test_0['target'])
    score_1_dummy = signal.correlate(dummy_regr.predict(np.asarray(test_1[test_1.columns[:16]])),np.asarray(test_1['target']),mode='same')/len(test_1['target'])
    score_2_dummy = signal.correlate(dummy_regr.predict(np.asarray(test_2[test_2.columns[:16]])),np.asarray(test_2['target']),mode='same')/len(test_2['target'])
    
    # Cross correlate with random speech
    random_corr = signal.correlate(model_optimal.predict(np.asarray(test_1[test_1.columns[:16]])),np.random.uniform(low=min(test_1['target']), high=max(test_1['target']), size=(np.asarray(test_1).shape[0],)),mode='same')/len(np.random.uniform(low=min(test_1['target']), high=max(test_1['target']), size=(np.asarray(test_1).shape[0],)))   
        
        ### SHOW RESULTS IN PLOTS ###
    
    ## Make line plots ##
    
    # Define x-axes
    x_axis_0 = np.linspace(0,len(test_0['target']),num=len(test_0['target']))
    x_axis_1 = np.linspace(0,len(test_1['target']),num=len(test_1['target']))
    x_axis_2 = np.linspace(0,len(test_2['target']),num=len(test_2['target']))
    x_all = np.linspace(0,len(data_test['target']),num=len(data_test['target']))
    
     ## All SNRs ##
    # For True
    plt.plot(x_all,np.asarray(data_test['target']),color='sandybrown', label='True')
    
    # For MNE Ridge regression
    plt.plot(x_all,model_optimal.predict(np.asarray(data_test[data_test.columns[:16]])),color='deepskyblue', label='Predicted')
    
    # For baseline dummy 
    plt.plot(x_all,dummy_regr.predict(np.asarray(data_test[data_test.columns[:16]])),color='rebeccapurple',dashes=[6, 2],label='Baseline (mean)')
    plt.grid()
    plt.title(f'TA: {TA} · Predicted and True Speech Envelopes · All SNRs')
    plt.xlabel('Samples')
    plt.ylabel('Speech Envelope')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(f'Figure All - TA {TA}.png')
    
     ## -5 DB SNR ##
    # For True
    plt.plot(x_axis_0,np.asarray(test_0['target']),color='sandybrown', label = 'True')
    
    # For MNE Ridge regression
    plt.plot(x_axis_0,model_optimal.predict(np.asarray(test_0[test_0.columns[:16]])),color='deepskyblue', label='Predicted')
    
    # For baseline dummy 
    plt.plot(x_axis_0,dummy_regr.predict(np.asarray(test_0[test_0.columns[:16]])),color='rebeccapurple',dashes=[6, 2],label='Baseline (mean)')
    plt.grid()
    plt.title(f'TA: {TA} · Predicted and True Speech Envelopes · -5 DB SNR')
    plt.xlabel('Samples')
    plt.ylabel('Speech Envelope')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(f'Figure -5 DB - TA {TA}.png')
    
     ## 0 DB SNR ##
    # For True
    plt.plot(x_axis_1,np.asarray(test_1['target']),color='sandybrown', label='True')
    
    # For MNE Ridge regression
    plt.plot(x_axis_1,model_optimal.predict(np.asarray(test_1[test_1.columns[:16]])),color='deepskyblue', label='Predicted')

    # For baseline dummy 
    plt.plot(x_axis_1,dummy_regr.predict(np.asarray(test_1[test_1.columns[:16]])),color='rebeccapurple',dashes=[6, 2],label='Baseline (mean)')
    plt.grid()
    plt.title(f'TA: {TA} · Predicted and True Speech Envelopes · 0 DB SNR')
    plt.xlabel('Samples')
    plt.ylabel('Speech Envelope')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(f'Figure 0 DB - TA {TA}.png')
    
     ## +5 DB SNR ##
    # For True
    plt.plot(x_axis_2,np.asarray(test_2['target']),color='sandybrown', label='True')
    
    # For MNE Ridge regression
    plt.plot(x_axis_2,model_optimal.predict(np.asarray(test_2[test_2.columns[:16]])),color='deepskyblue', label='Predicted') 
    
    # For baseline dummy 
    plt.plot(x_axis_2,dummy_regr.predict(np.asarray(test_2[test_2.columns[:16]])),color='rebeccapurple',dashes=[6, 2],label='Baseline (mean)')
    plt.grid()
    plt.title(f'TA: {TA} · Predicted and True Speech Envelopes · +5 DB SNR')
    plt.xlabel('Samples')
    plt.ylabel('Speech Envelope')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(f'Figure +5 DB - TA {TA}.png')   
    
    ## Bar chart to compare MSE for L2 and baseline ##
    
    measure = [np.mean(MSEs),np.mean(MSEdummies)]
    variance = [np.var(MSEs),np.var(MSEdummies)]
    x_labels = ['L2 (MNE)', 'Baseline']

    x_pos = [i for i, _ in enumerate(x_labels)]
    
    plt.bar(x_pos, measure, color='sandybrown', yerr=variance)
    plt.grid()
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.title("MSEs of L-2 and Baseline Model Compared")
    plt.xticks(x_pos, x_labels)
    plt.show()
    plt.savefig(f'MSEs compared - TA {TA}.png')    

    ## Make boxplot of CrossCorrs ##
    
    ticks = ['-5 DB', '0 DB','+5 DB','Random']
    
    # Function to set the colors of the boxplots
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    bpl = plt.boxplot([score_0,score_1,score_2], positions=np.array(range(len([score_0,score_1,score_2])))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot([score_0_dummy,score_1_dummy,score_2_dummy], positions=np.array(range(len([score_0_dummy,score_1_dummy,score_2_dummy])))*2.0+0.4, sym='', widths=0.6)
    bpu = plt.boxplot(random_corr,positions=[6],sym='', widths=0.6)
    set_box_color(bpl, 'deepskyblue') 
    set_box_color(bpr, 'rebeccapurple')
    set_box_color(bpu, 'sandybrown')

    # Draw temporary purple and blue lines and use them to create a legend
    plt.plot([], c='deepskyblue', label='L2 MNE')
    plt.plot([], c='rebeccapurple', label='Baseline')
    plt.plot([], c='sandybrown', label='Random')
    plt.title("Cross-correlation Between L-2-Predicted and True Envelopes in All SNR Levels & Random")
    plt.legend()
    plt.ylabel("Cross-correlation")
    plt.grid()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'Boxplot over CrossCorr - TA {TA}.png')
    
    
    # Make data matrix
    data_matrix = np.array(['TA:',TA,'Optimal Lambda Value:',opt_lambda_list[np.argmax(scores)], 'Best Score (Pearson´s R):',scores[np.argmax(scores)],'Mean Score (Pearson´s R):',np.mean(scores),
                            'Best MSE:',min(MSEs),'Mean MSE:',np.mean(MSEs),'Best MSE Dummy:',min(MSEdummies),'Mean MSE Dummy',np.mean(MSEdummies),'CrossCorr for -5DB SNR:',np.mean(score_0),
                            'CrossCorr for 0DB SNR:',np.mean(score_1)
                            ,'CrossCorr for +5DB SNR:',np.mean(score_2),'CrossCorr for random:',np.mean(random_corr),'Dummy CrossCorr for -5DB SNR:',np.mean(score_0_dummy),
                            'Dummy CrossCorr for 0DB SNR:',np.mean(score_1_dummy)
                            ,'Dummy CrossCorr for +5DB SNR:',np.mean(score_2_dummy)]).T

    # Save as CSV in working directory
    np.savetxt(name,
           data_matrix, delimiter= "," ,
           fmt='%s' ) 

if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))

    # Get the data
    data = getData()
    
    # Apply lowpass filter
    data = low_pass(data,64, 8)
    print(data)

    # Run cross validation for each TA
    #TAs = np.unique(data["TA"])
    #for TA in TAs:
    #    cross_validate(data, lambda_config = (2e0, 2e20, 11),TA = TA, name = ("data_%i.csv" %TA))




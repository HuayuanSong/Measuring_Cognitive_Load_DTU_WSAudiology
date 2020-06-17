# Fagprojekt
# Load data script

# Load dependencies
import numpy as np
import pandas as pd
from os import listdir, path    # OS pathing functions
from scipy.io import loadmat    # Load matlab data file
from signal_processing import * # Signal processing functions
from scipy import stats
#from mne.filter import filter_data

def getData(params = None, data_path = "local_data/trials.pkl", TA_path = "local_data/EEG/", speech_path = "local_data/audio/"):
    """
    Input:
        params      = List of parameters for the analysis (check function for default)
        data_pat    = Path for the saved dataframe (default: local_data/trials.pkl)
        TA_path     = Path for the .mat files for the TAs (default: local_data/EEG/)
        speech_path = Path for the .wav audio files for speech stimulus (default: local_data/audio/)

    Loads the data into a dataframe
    """

    # Set default params if none is defined
    if params == None:
        params = {
            "full" : True,          # Run a full analysis on all TAs
            "conditions" : [0, 1, 2] # Trial conditions, 0 = -5, 1 = 0 and 2 = +5 SNR
        }
    # Check if data already exists
    if path.exists(data_path):
        data = pd.read_pickle(data_path)
        return data

    print("No data found, loading new data and saving to %s" %data_path)

    # Initial setup
    df = pd.DataFrame()
    full_analysis = params["full"]
    conditions = params["conditions"]
    TA_list = listdir(TA_path)                           # Lists all .mat files in the TA_path
    TA_list = TA_list if full_analysis else [TA_list[0]] # Take the first TA if not running full analysis
    extra_col = ["target", "mask", "TA", "SNR","trial"]

    for TA in TA_list:

        print("Loading from: %s" %TA)
        TA_index = TA_list.index(TA)
        file_EEG = TA_path + TA
        data = loadmat(file_EEG)     # Load the .mat file
        data = data["trial_cond"][0] # Remove the global, version and header entries of the dict from loadmat()

        for cond in conditions:

            print("\tCondition %d / %d" %(cond + 1, len(conditions)))

            sub_data = data[cond][0][0]                   # Select sub data, per trial condition, i.e. make it easier from here
            n_trials = len(sub_data["trial"][0])          # Get number of trials
            fsample = sub_data["fsample"][0][0]           # Get the sample rate
            labels = [x[0][0] for x in sub_data["label"]] # Get electrode label names
            n_channels = len(labels)                      # Number of electrodes (channels)
            audio_targets = sub_data["targetaudio"][0]    # Lists all target speeches indexed by trial no.
            audio_masks = sub_data["maskeraudio"][0]      # Lists all mask speeched indexed by trial no.

            # Set labels if first time
            if len(df) == 0:
                df = pd.DataFrame(columns = labels + extra_col)

            for trial in range(n_trials):

                print("\tTrial %d / %d" %(trial + 1, n_trials))

                # Define the time axis (in seconds)
                t = sub_data["time"][0][trial][0]

                # Use time between 0 and 60 to mask EEG
                EEG_mask = (t >= 0) & (t < 60)
                t = t[EEG_mask]
                #t = np.linspace(0, 60, 64*60)
                t_length = len(t)
                print(t_length)

                target = audio_targets[trial][0]             # Target speech, this is the attended audio
                mask = audio_masks[trial][0]                 # Mask speech, this is ignored by user
                mask_delay = sub_data["trialinfo"][trial][2] # Delay between target and mask speech

                # Load speech audio files, get sample rate and data
                #print("\tLoading audio files:\n\t\tTarget: %s\n\t\tMask: %s" %(target, mask))
                target_fs, target_data = load_audio(speech_path + target)
                mask_fs, mask_data = load_audio(speech_path + mask)

                # Compute envelopes and downsampling to fsample
                #print("\tComputing envelopes...")
                target_envelope = compute_envelope(target_data)
                mask_envelope = compute_envelope(mask_data)

                # Perform low pass filtering
                #target_envelope = filter_data(target_envelope, sfreq = 64, l_freq = None, h_freq = 8, verbose = False)
                #mask_envelope = filter_data(mask_envelope, sfreq = 64, l_freq = None, h_freq = 8, verbose = False)
                target_envelope = low_pass(target_envelope, target_fs, cutoff = 8, order = 3)
                mask_envelope = low_pass(mask_envelope, mask_fs, cutoff = 8, order = 3)

                #print("\tDownsampling...")
                target_envelope = resample(target_envelope, target_fs, fsample)[0]
                mask_envelope = resample(mask_envelope, mask_fs, fsample)[0]

                # Check if we have enough data for this trial, target should include length of delay
                if len(target_envelope) + mask_delay < t_length or len(mask_envelope) < t_length:
                    print("\tAn envelope is too short, skipping trial...")
                    continue

                # Apply mask to synchronize data
                target_envelope = target_envelope[mask_delay:t_length + mask_delay]
                mask_envelope = mask_envelope[:t_length]

                target_envelope = resample(target_envelope, fsample, 64)[0]
                mask_envelope = resample(mask_envelope, fsample, 64)[0]

                # Give sample length when cutting off 1 second at the start and .5 seconds at the end (ERP)
                n_cutoff = (int(1.0 * 64), int(0.5 * 64))
                sample_length = len(target_envelope) - n_cutoff[0] - n_cutoff[1]

                # Initialize matrix to append data
                data_EEG = np.zeros((sample_length, n_channels + 5))

                for c in range(n_channels):
                    # The signal from the EEG channel
                    channel_data = sub_data["trial"][0][trial][c][EEG_mask][:t_length]

                    # Low pass filter the EEG at 8 Hz
                    channel_data = low_pass(channel_data, 256, cutoff = 8, order = 3)

                    # Add it to the matrix
                    channel_data = resample(channel_data, 256, 64)[0] # Downsample from 256 to 64 bitrate

                    # Cut off 1 and .5 seconds
                    channel_data = channel_data[n_cutoff[0]:len(channel_data)-n_cutoff[1]]

                    # Standardize the signal
                    channel_data = stats.zscore(channel_data)
                    data_EEG[:,c] = channel_data

                # Cut 1 second of the start, and .5 seconds of the end because of ERP
                target_envelope = target_envelope[n_cutoff[0]:len(target_envelope) - n_cutoff[1]]
                mask_envelope = mask_envelope[n_cutoff[0]:len(mask_envelope) - n_cutoff[1]]

                # Standardize target and mask
                target_envelope = stats.zscore(target_envelope)
                mask_envelope = stats.zscore(mask_envelope)

                # Append to data matrix
                data_EEG[:, n_channels] = target_envelope
                data_EEG[:, n_channels + 1] = mask_envelope
                data_EEG[:, n_channels + 2] = [TA_index]*sample_length
                data_EEG[:, n_channels + 3] = [cond]*sample_length
                data_EEG[:, n_channels + 4] = [trial]*sample_length

                # Convert to DataFrame
                df_ = pd.DataFrame(data = data_EEG, columns = labels + extra_col)

                # Concatenate
                df = pd.concat([df, df_], ignore_index = True)

    # Clean categorical data in dataframe
    df["TA"] = df["TA"].astype("category")
    df["SNR"] = df["SNR"].astype("category")

    # Save the dataframe for next time
    df.to_pickle(data_path)

    return df

def random_trial(data, TA = None, trial = None):
    # Pick random TA if none is specified
    if TA == None:
        TAs = np.unique(data["TA"])
        TA = TAs[np.random.randint(0, len(TAs))]
    # Pick random trial if the blacklisted trial is not specified
    if trial == None:
        trials = np.unique(data[data["TA"] == TA]["trial"])
        trial = trials[np.random.randint(0, len(trials))]
    # If blacklisted trial is specified, pick random trial that is NOT the specified one
    else:
        trials = np.unique(data[(data["TA"] == TA) & (data["trial"] != trial)]["trial"])
        trial = trials[np.random.randint(0, len(trials))]
    samples = data[(data["TA"] == TA) & (data["trial"] == trial)]
    SNRs = np.unique(samples["SNR"])
    SNR = SNRs[np.random.randint(0, len(SNRs))]
    return data[(data["TA"] == TA) & (data["trial"] == trial) & (data["SNR"] == SNR)]

if __name__ == "__main__":

    data = getData()

    print(random_trial(data, TA = 0, trial = 5))

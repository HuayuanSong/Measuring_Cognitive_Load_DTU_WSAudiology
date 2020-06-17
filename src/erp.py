# Fagprojekt
# ERP testing
from os import listdir, path
import os
path = "/Users/kathr/Documents/Fagprojekt/project-in-ai-and-data/src"
os.chdir(path)
# Load dependencies
import matplotlib.pyplot as plt # Plot data
import numpy as np              # Numpy array features
from os import listdir, path    # OS pathing functions
from scipy.io import loadmat    # Load matlab data file
from utilities import get_cycle # Fancy color map for plots
from matplotlib.lines import Line2D


def runTest(params, TA_path = "local_data/EEG/"):
    """
    Input:
        params      = List of parameters for the test
        TA_path     = Path for the .mat files for the TAs (default: local_data/EEG/)

    Runs the ERP to test for triggers
    """
    # Initial setup
    full_analysis = params["full"]
    conditions = params["conditions"]
    TA_list = listdir(TA_path)                           # Lists all .mat files in the TA_path
    TA_list = TA_list if full_analysis else [TA_list[0]] # Take the first TA if not running full analysis

    print("Running %s analysis..." %("full" if full_analysis else "quick"))

    for TA in TA_list:

        print("Analysis on: %s" %TA)

        file_EEG = TA_path + TA
        #global data # Debugging
        data = loadmat(file_EEG)     # Load the .mat file
        data = data["trial_cond"][0] # Remove the global, version and header entries of the dict from loadmat()

        colors = ["red", "blue", "green"]
        fig, axes = plt.subplots(3, 3, figsize = (15, 10))


        for cond in conditions:
            sub_data = data[cond][0][0]                # Select sub data, per trial condition, i.e. make it easier from here
            n_trials = len(sub_data["trial"][0])       # Get number of trials
            fsample = sub_data["fsample"][0][0]        # Get the sample rate
            labels = sub_data["label"][0][0]           # Get electrode label names
            n_channels = len(labels)                   # Number of electrodes (channels)

            data_target_on = []
            data_mask_on = []
            data_audio_off = []

            t_target_on = [-1,1]
            t_mask_on = [-1,1]
            t_audio_off = [-1,1]

            for trial in range(n_trials):
                # Define the time axis (in seconds) and delay
                t = sub_data["time"][0][trial][0]
                mask_delay = sub_data["trialinfo"][trial][2] # Delay between target and mask speech
                mask_delay_s = mask_delay / fsample

                target_on = (t >= t_target_on[0]) & (t < t_target_on[1])
                mask_on   = (t >= mask_delay_s + t_mask_on[0]) & (t < mask_delay_s + t_mask_on[1])
                audio_off = (t >= 60 + t_audio_off[0]) & (t < 60 + t_audio_off[1])

                """
                To get the EEG data, use the following:
                    sub_data["trial"][0][trial][idx][EEG_mask][:t_length]
                where idx is the channel index (0-15 for 16 channels)
                """

                # Compute mean
                trial_mean = np.mean(sub_data["trial"][0][trial], axis = 0)

                # Append data
                data_target_on.append(trial_mean[target_on])
                data_mask_on.append(trial_mean[mask_on])
                data_audio_off.append(trial_mean[audio_off])

            # Compute mean of trials
            mean_target_on = np.mean(data_target_on, axis = 0)
            mean_mask_on   = np.mean(data_mask_on, axis = 0)
            mean_audio_off = np.mean(data_audio_off, axis = 0)

            # Show for target on
            x = np.linspace(t_target_on[0], t_target_on[1], len(mean_target_on))
            for i in range(n_trials):
                axes[cond, 0].plot(x, data_target_on[i], label = str(i), color = "grey", alpha = .5)
            axes[cond, 0].plot(x, mean_target_on, label = "mean", color = colors[cond])
            axes[cond, 0].grid(True)
            axes[cond, 0].axvline(0, color = "black")
            axes[cond, 0].set_title("Target on")

            # Show for mask on
            x = np.linspace(t_mask_on[0], t_mask_on[1], len(mean_mask_on))
            for i in range(n_trials):
                axes[cond, 1].plot(x, data_mask_on[i], label = str(i), color = "grey", alpha = .5)
            axes[cond, 1].plot(x, mean_mask_on, label = "mean", color = colors[cond])
            axes[cond, 1].grid(True)
            axes[cond, 1].axvline(0, color = "black")
            axes[cond, 1].set_title("Mask on")

            # Show for audio off
            x = np.linspace(t_audio_off[0], t_audio_off[1], len(mean_audio_off))
            for i in range(n_trials):
                axes[cond, 2].plot(x, data_audio_off[i], label = str(i), color = "grey", alpha = .5)
            axes[cond, 2].plot(x, mean_audio_off, label = "mean", color = colors[cond])
            axes[cond, 2].grid(True)
            axes[cond, 2].axvline(0, color = "black")
            axes[cond, 2].set_title("Audio off")

        #plt.show()
        lines = [Line2D([0], [0], color = colors[0], lw=4),
            Line2D([0], [0], color = colors[1], lw=4),
            Line2D([0], [0], color = colors[2], lw=4)]
        fig.legend(lines, ["-5 dB SNR", "0 dB SNR", "+5 dB SNR"], loc = "center right")
        fig.suptitle("ERP on %s" %TA, fontsize = 16)
        plt.show()
        #fig.tight_layout()

if __name__ == "__main__":
    params = {
        "full" : False,          # Run a full analysis on all TAs
        "conditions" : [0, 1, 2] # Trial conditions, 0 = -5, 1 = 0 and 2 = +5 SNR
    }
    runTest(params)
# Importing dependencies
import numpy as np
import pandas as pd
from data_load import getData
from os import chdir, path
from mne.filter import filter_data


def low_pass(data, sfreq, hfreq):
	channels = data.columns[:16].values
	for i in channels:
		print("Low-pass filtering channel %i / %i: %s" %(np.where(channels == i)[0] + 1, len(channels), i))
		data[i] = filter_data(data[i], sfreq = sfreq, l_freq = None, h_freq = hfreq, verbose = False)
	return data


if __name__ == "__main__":
    # Set working directory
    chdir(path.dirname(__file__))
    data = getData()
    data_low_pass = low_pass(data, 64, 8)
    print(data.describe())
    print(data_low_pass.describe())
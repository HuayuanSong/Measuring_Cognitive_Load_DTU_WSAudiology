# Fagprojekt
# Data analysis, Main script file

# Load dependencies
from os import path, chdir      # Navigate directories
import matplotlib.pyplot as plt # Plot data
import numpy as np              # Numpy array features
import pandas as pd             # Pandas dataframe features
from data_load import getData   # Load the .mat data and structurize into pandas dataframe
from utilities import get_cycle # Fancy color map for plots

if __name__ == "__main__":
    # Set working directory
    wd = path.dirname(__file__)
    chdir(wd)

    # Get the data
    data = getData()

    # Example correlation plot
    EEG_channels = data.columns[0:16]
    corr = data[EEG_channels].corr()
    plt.matshow(corr)
    plt.xticks(np.arange(16), data.columns[0:16], rotation = "vertical")
    plt.yticks(np.arange(16), data.columns[0:16])
    plt.colorbar()
    plt.title("Correlation matrix, %s samples" %format(len(data), ",d"), y = 1.2, fontsize = 10)
    plt.show()

    # Plot highest correlation
    corr_ranked = (corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool)).stack().sort_values(ascending = False))
    plt.scatter(data[corr_ranked.keys()[0][0]], data[corr_ranked.keys()[0][1]], alpha = .01, color = "black", s = 2)
    plt.title("Scatter plot, %s and %s, correlation = %f" %(corr_ranked.keys()[0][0], corr_ranked.keys()[0][1], corr_ranked[0]))
    plt.xlabel(corr_ranked.keys()[0][0])
    plt.ylabel(corr_ranked.keys()[0][1])
    plt.show()
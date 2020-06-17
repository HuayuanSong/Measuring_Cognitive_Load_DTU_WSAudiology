# Fagprojekt
# Speech computations

# Load dependencies
from scipy.io import wavfile
from scipy.signal import hilbert
from mne.filter import resample
import numpy as np

def loadAudio(file):
    """
    Input:
        file = path to audio (.wav) file

    Output:
        fs   = audio sample rate
        data = audio signal
    """
    fs, data = wavfile.read(file)
    return fs, data

def reSample(signal, current, target):
    """
    Input:
        signal  = The signal to resample
        current = The current sample rate
        target  = The target sample rate

    Output:
        signal_resampled
        target
    """
    newRate = current / target
    signal_resampled = resample(signal, down = newRate, npad = "auto")
    return signal_resampled, target

def normalize(signal):
    """
    Input:
        signal = The signal to normalize

    Output:
        The normalized signal, max distance from 0 is 1
    """
    amp = max(abs(signal))
    return signal / amp

def computeEnvelope(audio):
    """
    Input:
        audio = Audio data

    Output:
        envelope (with eventual downsampling) computed using absolute value of hilbert transformation
    """
    return np.abs(hilbert(audio))

if __name__ == "speech":
    print("speech.py is deprecated, use signal_processing.py instead (beware of new function naming)")
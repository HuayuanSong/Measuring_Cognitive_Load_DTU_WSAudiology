# Fagprojekt
# Signal processing

# Load dependencies
from scipy.io import wavfile
from scipy.signal import hilbert, butter, sosfiltfilt
import mne.filter
import numpy as np

def load_audio(file):
    """
    Input:
        file = path to audio (.wav) file

    Output:
        fs   = audio sample rate
        data = audio signal
    """
    fs, data = wavfile.read(file)
    return fs, data

def resample(signal, current, target):
    """
    Input:
        signal  = The signal to resample
        current = The current sample rate
        target  = The target sample rate

    Output:
        signal_resampled
        target
    """
    ratio = current / target
    signal_resampled = mne.filter.resample(signal, down = ratio, npad = "auto")
    return signal_resampled, target

def normalize(signal):
    """
    Input:
        signal = The signal to normalize

    Output:
        The normalized signal, max abs amplitude is 1
    """
    amp = max(abs(signal))
    return signal / amp

def compute_envelope(audio):
    """
    Input:
        audio = Audio data

    Output:
        envelope (with eventual downsampling) computed using absolute value of hilbert transformation
    """
    return np.abs(hilbert(audio))

def low_pass(signal, sfreq, cutoff, order = 1):
    """
    Performs a low pass butterworth filter of nth order

    Parameters
    ----------
    signal_ : array
        Array type of the signal to be filtered
    sfreq : float
        The sampled frequency of the signal
    cutoff : float
        The cutoff frequency in the same unit as the signal
    order : int, optional
        The order to perform the filtering at. The default is 1.

    Returns
    -------
    signal_filtered : array
        Array type of the filtered signal

    """
    # Create the filter
    nyq = 0.5 * sfreq
    fc = cutoff / nyq
    sos = butter(order, fc, output = "sos")
    return sosfiltfilt(sos, signal)



def shift(signal, lag = 0, freq = 1):
    """

    Parameters
    ----------
    signal : float array
        The signal to shift
    lag : int, optional
        The time in ms to shift the signal. The default is 0.
    freq : float, optional
        The signal's frequency. The default is 1.

    Returns
    -------
    signal : float array
        The signal shifted by the designated lag, has the samples shifted outside the scope cut-off.

    """
    shape = signal.shape
    if len(signal.shape) == 1:
        signal = signal.reshape(1, -1)
    n_shift = int(freq * lag / 1000)

    if np.abs(n_shift) > len(signal):
        raise ValueError("Can't shift signal beyond signal length")

    return np.roll(signal, n_shift)[:,:n_shift]
''' Module for generating and operating on vibration spectra.'''

import numpy as np
import numpy.ma as ma
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import os

import aggregate_utils
import plotting_utils

def get_next_pow2(n):
    '''
    Returns next power of 2 greater than or equal to a number.

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    k : int
        Net power of 2 greater than or equal to the input.
    '''
    k = 1
    while k < n:
        k = k << 1
    return k

def normalize(f, dx):
    '''
    Scales the input such that it has unit area.

    Parameters
    ----------
    f : array-like, 1D
        Input signal.
    dx : float
        Distance between samples.

    Returns
    -------
    f_ : array-like, same size as input
        Normalized signal.
    '''
    return f / np.trapz(f, dx=dx)
        
def get_spectra_periodogram(ts, fps, freqmin, freqmax, nfft=None, window='boxcar', detrend='constant', verbose=True):
    ''' 
    Computes power spectral density for each vibration signal using periodogram.

    Computes the power spectrum for each signal and returns the portion of the power 
    spectrum between freqmin and freqmax. 
    
    Parameters
    ----------
    ts : arraylike (1d, 2d, or 3d)
        Time series with axis 0 equal to time axis.
    fps : float
        Frame rate of video in frames per second.
    freqmin : float
        Lower bound for frequency thresholding in Hz.
    freqmax : float
        Upper bound for frequency thresholding in Hz.
    nfft : int
        Size of underlying DFT. If greater than the length of
        the input signal, zero pads signal to desired length.
    window : string
        Window to apply to time series prior to fft.
    detrend : string
        See scipy.signal.periodogram for details.
    verbose : boolean
        Print progress statements when true.

    Returns
    -------
    freq : array, 1d
        Frequency range (bins), bounded by freqmin and freqmax.
    pxx : array, (1d, 2d, or 3d)
        Array containing the bin magnitudes for each of the input signals 
        (for frequencies between freqmin and freqmax).
    '''
    if verbose:
        print('Computing pixel spectra')
    
    if nfft is None:
        nfft = 8 * get_next_pow2(ts.shape[0])
    
    freq, pxx = scipy.signal.periodogram(ts, 
                                         fps, 
                                         window=window, 
                                         nfft=nfft,
                                         detrend=detrend, 
                                         axis=0)
    if (verbose):
        print('Finished computing spectra')
     
    # filter by freq range
    low = np.where(freq > freqmin)[0][0]
    high = np.where(freq > freqmax)[0][0]
    pxx = pxx[low:high]
    freq = freq[low:high]
    
    return freq, pxx

def get_spectra_periodogram_int_cycles(ts, 
                                       step, 
                                       n, 
                                       fps, 
                                       freqmin, 
                                       freqmax, 
                                       window='boxcar', 
                                       detrend='constant', 
                                       verbose=False, 
                                       plot=False):
    
    ''' 
    Iteratively trims the input and computes the power spectral density using the periodogram.

    Attempts to compute the power spectral density for an interger number of cycles by iteratively
    trimming the input and computing the power spectral density with the periodogram (of length
    equal to the trimmed input). Returns spectra whose corresponding average spectrum's peak frequency
    has the greatest magnitude.
    
    Parameters
    ----------
    ts : arraylike (1d, 2d, or 3d)
        Time series with axis 0 equal to time axis.
    step: int
        Number of frames to remove from begining
        during each iteration.
    n : int
        Number of trimming iterations.
    fps : float
        Frame rate of video in frames per second.
    freqmin : float
        Lower bound for frequency thresholding in Hz.
    freqmax : float
        Upper bound for frequency thresholding in Hz.
    Window : string
        Window to apply to time series prior to fft.
    detrend : string
        See scipy.signal.periodogram for details.
    verbose : boolean
        Print progress statements when true.
    plot : boolean
        Plots average spectrum of each iteration when true.

    Returns
    -------
    freq : 1d array 
        Frequency range (bins), bounded by freqmin and freqmax.
    pxx : array, (1d, 2d, or 3d)
        Array containing the bin magnitudes for each of the input signals
        (for frequencies between freqmin and freqmax).
    '''
    freq_max = None
    pxx_max = None
    mag_max = -1
    i_max = 0

    if verbose:
        print('Iteratively computing spectrum for trimmed time series')
        print(f'Step size: {step}')
        print(f'Number of iteratoins: {n}')

    for i in range(n):
        if verbose:
            print(f'i = {i}')

        ts_trim = ts[i * step:]

        freq, pxx = get_spectra_periodogram(ts, fps, freqmin, freqmax, nfft=len(ts_trim), 
                                            window=window, detrend=detrend, verbose=False)

        pxx_avg = aggregate_utils.average_spectra(pxx)
        
        idx, mag, prom = find_dominant_peak(pxx_avg)

        if verbose:
            print(f'Peak Magnitude: {mag}')

        if mag > mag_max:
            mag_max = mag
            pxx_max = pxx
            freq_max = freq
            i_max = i

        if plot:
            plotting_utils.plot_spectrum(freq, pxx, peak_idx=idx)
            plt.show()

    if verbose:
        print(f'Best spectrum when i = {i_max}')

    return freq_max, pxx_max

def get_spectra_welch(ts, 
                      fs, 
                      freqmin, 
                      freqmax, 
                      nperseg=4096,
                      nfft=None, 
                      window='boxcar', 
                      detrend='constant',
                      noverlap=None,
                      verbose=True):
    '''
    Computes spectra using Welch's method.

    Parameters
    ----------
    ts : array-like, (1d, 2d, or 3d)
        Time series with axis 0 equal to time axis.
    fps : float
        Frame rate of video in frames per second.
    freqmin : float
        Lower bound for frequency thresholding in Hz.
    freqmax : float
        Upper bound for frequency thresholding in Hz.
    nperseg : int
        Size of each segment.
    window : string
        Window to apply to time series prior to fft.
    detrend : string
        See scipy.signal.periodogram for details.
    verbose : boolean
        Print progress statements when true.

    Returns
    -------
    freq : array, 1d
        Frequency range (bins), bounded by freqmin and freqmax.
    pxx : array, (1d, 2d, or 3d)
        Array containing the bin magnitudes for each of the input signals 
        (for frequencies between freqmin and freqmax).
    '''
    if verbose:
        print('Computing pixel spectra')
    
    freq, pxx = scipy.signal.welch(ts, fs=fs, window=window, nperseg=nperseg, nfft=nfft, detrend=detrend, noverlap=noverlap, axis=0)
    
    if verbose:
        print('Finished computing spectra')
    
    # filter by freq range
    low = np.where(freq > freqmin)[0][0]
    high = np.where(freq > freqmax)[0][0]
    pxx = pxx[low:high]
    freq = freq[low:high]
    
    return freq, pxx

def get_dom_freq_max_power(freq, pxx):
    '''
    Returns array containing the frequencies with the greatest power for each input signal.
    
    Parameters
    ----------
    freq  : array, 1d
        Spectrum frequencies.
    pxx : 1d, 2d, or 3d array
        Power magnitudes for spectrum of each input signal.

    Returns
    -------
    freq_max_power : array, 1d smaller than input
        Frequencies with the max power for each input signal.
    power : array 
        Power corresponding to each frequency in freq_max_power.
    _ : None
        Empty variable added so that the function has three return values
        and can be substitude anywhere get_dom_freq_peak_finder is used.
    '''
    max_power_idx = np.argmax(pxx, axis=0)
    power = np.max(pxx, axis=0)
    freq_max_power = freq[max_power_idx]
    return freq_max_power, power, None

def find_dominant_peak(x, criteria='magnitude'):
    ''' 
    Returns index and prominence of most dominant peak. 

    Choose dominant peak based on selection criteria: either the peak with the greatest
    magnitude or the greatest prominence.

    Parameters
    ----------
    x : array
        Input signal.
    criteria : string
        Selection criteria: either 'magnitude' or 'prominence'.

    Returns
    -------
    dom_peak_idx: int
        Index of peak.
    magnitude : float
        Magnitude of dominant peak.
    prominence  : float
        Prominence of dominant peak.
    '''
    if criteria != 'magnitude' and criteria != 'prominence':
        print("Invalid filter parameter given. Should be either 'magnitude' or 'prominence'")
        return None
    
    # find peaks and corresponding prominences
    peak_idxs, properties = scipy.signal.find_peaks(x, prominence=0)
    
    if len(peak_idxs) == 0:
        return -1, 0, 0
    
    # find index (w.r.t peak_idxs) of peak with most prominence
    max_prom_idx = np.argmax(properties['prominences'])
    
    # find index (w.r.t peak_idxs) of peak with greatest magnitude
    max_mag_idx = np.argmax(x[peak_idxs])
    
    # choose dominant peak index based on filter type
    peak_idx = max_prom_idx if criteria == 'prominence' else max_mag_idx 
    
    # return index of dominant peak and corresponding prominence
    dom_peak_idx = peak_idxs[peak_idx]
    return dom_peak_idx, x[dom_peak_idx], properties['prominences'][peak_idx]

def get_dom_freq_peak_finder(freq, pxx, criteria='magnitude'):
    ''' 
    Finds dominant frequency in each spectrum and returns index of peak and corresponding prominence.

    For each spectrum, finds the dominant frequency based on the given criteria. When 'magnitude' is passed,
    the peak with the greatest magnitude will be selected. When 'prominence' is passed, the peak with the greatest
    magnitude will be selected.

    Paramters
    ---------
    freq : 1d array
        Spectrum frequencies.
    pxx : 1d, 2d, or 3d array 
        Power magnitudes for each vibration signal spectrum.
    criteria : string
        Either 'magnitude' or 'prominence'.

    Returns
    -------
    dom_freq : array dimension = dim(pxx) - 1
        The dominant frequency for each spectrum in the input.
    magnitudes : array, same shape as dom_freq
        Magnitude of each peak.
    prominences : array, same shape as dom_freq
        Prominence of each peak.
    '''
    # find indices of most prominent peaks
    peak_idxs, magnitudes, prominences = np.apply_along_axis(find_dominant_peak, 0, pxx, criteria)
    peak_idxs = peak_idxs.astype(int)
    
    if -1 in peak_idxs.ravel():
        peak_idxs = ma.masked_equal(peak_idxs, -1)

    # use indices to get frequencies
    dom_freq = freq[peak_idxs]
    
    # return peaks with maximum prominence
    # return corresponding prominences
    return dom_freq, magnitudes, prominences

def save_spectra(prefix, freq, pxx):
    '''
    Saves spectra data structures in numpy binary files.

    Parameters
    ----------
    prefix : string
        Path to the directory where the files should be stored.
    freq : array
        Spectra frequencies.
    pxx : array
        Spectra magnitudes.
    '''
    freq_path = os.path.join(prefix, 'freq')
    pxx_path = os.path.join(prefix, 'pxx')
    np.save(freq_path, freq)
    np.save(pxx_path, pxx)

def save_dom_freq(prefix, dom_freq):
    '''
    Saves array storing dominant frequency for each vibration signal.

    Parameters
    ----------
    prefix : string
        Path to the directory where the files should be stored.
    dom_freq : array
        Dominant frequency for each vibration signal.
    '''
    dom_freq_path = os.path.join(prefix, 'dom_freq')
    np.save(dom_freq_path, dom_freq)

def load_spectra(prefix):
    '''
    Loads freq and pxx data describing spectra previously saved.

    Parameters
    ----------
    prefix : string
        Path to the directory where the files are stored.
    
    Returns
    -------
    freq : array
        Spectra frequencies.
    pxx : array
        Spectra magnitudes.
    '''
    freq = np.load(os.path.join(prefix, 'freq'))
    pxx = np.load(os.path.join(prefix, 'pxx'))
    return freq, pxx

def load_dom_freq(prefix):
    '''
    Loads dominant frequency data previously saved.

    Parameters
    ----------
    prefix : string
        Path to the directory where the files are stored.

    Returns
    -------
    dom_freq : array
        Dominant frequency for each vibration signal.
    '''
    dom_freq = np.load(os.path.join(prefix, 'dom_freq'))
    return dom_freq
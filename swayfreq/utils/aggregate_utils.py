'''Module for aggregating frequency content from multiple spectra'''

import numpy as np
import numpy.ma as ma
import scipy.stats

def average_spectra(pxx):
    ''' 
    Averages spectra binwise.

    Averages the magnitudes at each frequency bin across all the spectra.
    This function assumes axis zero of pxx is the frequency axis and that other axes
    represent the position of the different vibration spectra. 

    This function also assumes each spectra has the same underlying frequency bins.

    Parameters
    ----------
    pxx : 1d, 2d, or 3d array 
        Spectra magnitudes (axis zero is the frequency bin axis).

    Returns
    -------
    pxx_avg : 1d array
        Binwise average of pxx.
    '''
    # average over positional dimension (first axis assumed to be frequency dimension)
    ax = tuple(np.arange(1, len(pxx.shape)))
    return np.nanmean(pxx, axis=ax)

def hist_spectrum(dom_freq, freq, masked=True):
    '''
    Creates histogram for the dominant frequency array.

    When freq is equal to the bins of the associated spectum, the edge of each bin corresponds 
    with a frquency in the spectrum such that each histogram bin counts the multiplicity of 
    one spectrum frequency.

    Parameters
    ----------
    dom_freq : 1d or 2d array
        Dominat frequency of each vibration signal.
    freq : 1d array
        Bins to be used for histogram (in most cases frequency.
        bins of underlying spectra).
    masked : boolean
        True when dom_freq has been masked.

    Returns
    -------
    count : array
        Count associated with each histogram bin.
    bins : array
        Histogram bins.
    '''
    data = dom_freq.compressed().ravel() if masked else dom_freq.ravel()
    freq = np.append(freq, freq[-1] + (freq[1] - freq[0]))
    count, bins = np.histogram(data, bins=freq, density=True)
    return count, bins[:-1]

def mask_dom_freq(dom_freq, weight, threshold=None, stat_reduc='median', nstd=1, percentile=75):
    ''' 
    Masks dominant frequency array uisng the provided weights and either a threshold or statisitical reduction.

    When a threshold is provided, masks values in dom_freq  whose corresponding weight is less than the threshold.
    When no threshold is provided, masks values in dom_freq whose corresponding weight is less than value obtained by
    performing the operation specified in stat_reduc to weight (either median, mean, or percentile). I.e. masks values in dom_freq
    whose weight is less than the median or mean of all the weights.
    
    Parameters
    ----------
    dom_freq : 1d or 2d array
        Dominant frequency for each vibration signal.
    weight : 1d or 2d array (same shape as dom_freq),
        Weight for each dominant frequency (typically power or prominence).
    threshold :  float
        Custom masking threshold, optional. When provided stat_reduc is ignored.
    stat_reduc : string
        Reduction applied to weight to obtain masking threshold.
    nstd : int
        Number of standard deviations to add to mean when masking with mean.
    percentile :  int in (0, 100)
        Percentile of data to estimate (when stat_reduc = 'percentile').
        
    Returns
    -------
    dom_freq_masked : masked array with same shape as dom_freq
        Masked dominant frequencies.
    '''
    mask = None

    if threshold is not None:
        mask = weight < threshold
    elif stat_reduc == 'median':
        mask = weight < np.median(weight.ravel())
    elif stat_reduc == 'mean':
        mask = weight < weight.ravel().mean() + (nstd * np.std(weight.ravel()))
    elif stat_reduc == 'percentile':
        mask = weight < np.percentile(weight.ravel(), percentile)
    
    else:
        print('No valid threshold or stat_filter provided')
        return None
    
    dom_freq_masked = ma.array(dom_freq, mask=mask)
    return dom_freq_masked

def masked_mode(masked):
    '''
    Returns the mode of a masked array.
    
    Parameters
    ----------
    masked : masked array

    Returns
    -------
    mode : float
        Mode of masked array.
    '''
    return scipy.stats.mode(ma.filled(masked, np.nan).ravel(), nan_policy='omit')

def mode(unmasked):
    '''
    Returns the mode of a unmasked numpy array.
    
    Parameters
    ----------
    unmasked : unmasked numpy array

    Returns
    -------
    mode : float
        Mode of unmasked array.
    '''
    return scipy.stats.mode(unmasked.ravel())
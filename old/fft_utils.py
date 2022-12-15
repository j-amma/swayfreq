''' Module containing functions useful for FFT pixel analysis. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics

def compute_frequency(data, fps, freqlow, freqhigh, nfreq=1, normAmp=False, nsamp=2048, verbose = True, every = 100):
    ''' Computes the top n frequencies in each pixel time series.
    
    Performs the fft on each pixel time series in the given array and returns
    and returns the nfreq frequency components between freqlow and freqhigh
    with the greatest amplitudes.
    
    Params:
        data        -- 3d array [y, x, t] storing a single channel uncompressed video
        fps         -- frame rate of the video in frames per second
        freqlow     -- lower bound of target frequencies
        freqhigh    -- upper bound of target frequencies
        nfreq       -- number of frequencies to return per pixel (default 1)
        normAmp     -- boolean for normalizing spectral amplitudes of a pixel (default false)
        nsamp       -- number of samples for FFT (default 2048, slightly extends 60s 30fps vid)
        verbose     -- print helpful progress statement when true
        every       -- frequency (number of rows) progress statement should be printed
        
    Returns:
        maxf        -- 3d array with dims [x,y,nfreq] storing the top nfreq frequencies 
                       or each pixel time series
        f_ample     -- 3d array with dims [x,y,nfreq] storing the corresponding amplitudes
        frequencies -- freq range between freqlow and freqhigh with the appropriate
                       frequency resolution
        mean        -- mean of amplitudes of all pixel bins; if normAmp = True, gives mean
                       of normalized amplitudes
        std         -- standard deviation of amplitudes of all pixel bins; if normAmp = True,
                       givs mean of normalized amplitudes
    '''
    
    if (verbose):
        print('Starting to compute frequencies for each pixel')
    
    # Initialize output arrays
    maxf = np.zeros((nfreq, data.shape[0], data.shape[1]))
    f_ample = np.zeros((nfreq, data.shape[0], data.shape[1]))
    
    # For each pixel in the video
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            # calculate fft response
            freq_space = np.fft.fft(data[y, x, :], n = nsamp)
            frequencies = np.fft.fftfreq(nsamp, d = 1 / fps)
            
            # Isolate data between target min/max frequencies
            low = np.where(frequencies > freqlow)[0][0]
            high = np.where(frequencies > freqhigh)[0][0]
            freq_space = abs(freq_space[low:high])
            frequencies = frequencies[low:high]
            
            # norm amplitude by amplitude sum
            if (normAmp):
                freq_space_sum = np.sum(freq_space)
                freq_space /= freq_space_sum
                freq_space *= 100
                
            # calculate median and standard deviation for masking
            mean = np.median(np.ndarray.flatten(freq_space))
            std = np.std(np.ndarray.flatten(freq_space))
            
            # Create dataframe to facilitate sorting
            # sort entries by amplitude
            freq_df = pd.DataFrame({'f_ample' : freq_space, 'freq' : frequencies})
            freq_df_sorted = freq_df.sort_values(by='f_ample', ascending=False)
            freq_df_sorted = freq_df_sorted.reset_index()
            
            # store top nfreq frequencies
            for i in range(nfreq):
                maxf[i, y, x] = freq_df_sorted['freq'][i]
                f_ample[i, y, x] = freq_df_sorted['f_ample'][i]
    
        if(verbose and (y % every == 0)):
            print("   Frequencies computed for all pixels in row " + str(y))
    
    if(verbose):
        print ('Frequency Computation Complete')
    return maxf, f_ample, frequencies, mean, std


def filter_by_amp(maxf, f_ample, threshold):
    ''' Masks frequencies according to an amplitude threshold.
    
    Params:
        maxf      -- 3d array containing top frequencies per pixel
        f_ample   -- 3d array containing corresponding amplitudes
        threshold -- amplitude threshold value
    
    Returns:
        masked maxf array
    '''
    return np.ma.array(maxf, mask = f_ample < threshold)

def top_nfreq(masked, maxf, frequencies, nfreq):
    ''' Returns the cumulative top n freqs in the ROI.
        
    In other words, choses the top n frequncies fro the
    cumulative histogram.
    
    Params:
        masked      -- 3d array containing all frequencies above some
                       amplitude threshold
        maxf        -- 3d array containing top n frequencies per pixel
        frequencies -- frequency bins under analysis
    
    Returns:
        topn        -- list of top n frequencies
    
    '''
    freqs_from_masked = np.extract(masked, maxf)
    all_freqs = np.ndarray.flatten(freqs_from_masked)
    hist, bin_edges = np.histogram(all_freqs, frequencies)

    # Create dataframe to facilitate sorting
    # sort entries by amplitude
    freq_df = pd.DataFrame({'count' : hist, 'bins' : frequencies[:-1]})
    freq_df_sorted = freq_df.sort_values(by='count', ascending=False)
    freq_df_sorted = freq_df_sorted.reset_index()

    topn = []
    for i in range(nfreq):
        topn.append(freq_df_sorted['bins'][i])
    
    return topn


def plot_freq(data, maxf, nfreq, low, high, dscrp):
    ''' Plots map and histogram of top frequency content.
    
    Params:
        data  -- 3d array containing frequency information (possibly
                 masked
        maxf  -- 3d array containing top frequencies per pixel
        nfreq -- number of frequencies stored per pixel
        low   -- lower bound of frequency content
        high  -- upper bound of frequency content
        dscrp -- description of frequency content (string)
    '''
    fig, axs = plt.subplots(nfreq,2, figsize=(16,16))
    fig.suptitle('Frequency Content, ' + dscrp, fontsize=16)
    for ax1, i in zip(axs[:,0], range(nfreq)):
        im = ax1.imshow(data[i], vmin = low, vmax= high)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Freq. (Hz)')

    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    for ax2, j in zip(axs[:,1], range(nfreq)):
        freq_value = np.extract(data[j], maxf[j])
        ax2.hist(freq_value, 100, [low, high]) # TODO: make this match frequencies!
        ax2.set_ylabel("Num Pixels")
        ax2.set_xlabel("Freq. (Hz)")
        
def plot_cumul_freq_hist(maxf, frequencies, masked_maxf = None, dscrp = ""):
    ''' Plots a histogram using all pixel frequencies.
    
    If a masked_maxf is given, produces a histogram for the 
    masked frequecy content.
    
    Params:
        maxf        -- 3d array containing unmasked frequency content
        frequencies -- freq range between freqlow and freqhigh with the appropriate
                       frequency resolution
        dscrp       -- description of data
        masked_maxf -- masked 3d array containing masked frequency content
    
    '''
    
    fig = plt.figure(figsize=(14,7))

    freq_value = maxf
    if (masked_maxf is not None):
        freq_value = np.extract(masked_maxf, maxf)
    freq_value = np.ndarray.flatten(freq_value)

    plt.hist(freq_value, bins=frequencies)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.legend()

    plt.title("Pixel Frequency Histogram, " + dscrp +"\n Top Frequency = " + str(statistics.mode(freq_value)) + " Hz")

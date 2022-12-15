import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import cv2
import vid2vib_utils

def plot_image(image, low=None, high=None, ax=None, title=None, cmap='magma', edgecolor=None, gscale_image=False, colorbar=True, colorbarlabel=None, no_ticks=True, figsize=None):    
    '''Plots an image or heat map
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(image, cmap=cmap, vmin=low, vmax=high)
    
    if no_ticks:
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.1)
        plt.colorbar(im, cax=cax, label=colorbarlabel)
        clb = plt.colorbar(im, cax=cax, label=colorbarlabel)
        if gscale_image:
            ticks = clb.get_ticks()
            ticks[-1] = vid2vib_utils.CHANNEL_MAX
            clb.set_ticks(ticks)

    if edgecolor is not None:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            ax.spines[axis].set_color(edgecolor)
    
    ax.set_title(title)

    return ax

def plot_grayscale_image(image, ax=None, title=None, edgecolor=None, colorbar=True, colorbarlabel=None, no_ticks=True, figsize=None):
    '''
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plot_image(image,
                low=vid2vib_utils.CHANNEL_MIN, 
                high=vid2vib_utils.CHANNEL_MAX, 
                ax=ax, 
                title=title, 
                cmap='gray', 
                edgecolor=edgecolor, 
                gscale_image=True, 
                colorbar=colorbar, 
                colorbarlabel=colorbarlabel, 
                no_ticks=no_ticks, 
                figsize=figsize)
    return ax

def plot_truecolor_image(image, ax=None, title=None, edgecolor=None, no_ticks=True, figsize=None):
    '''
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plot_image(image,
                ax=ax, 
                title=title, 
                cmap=None, 
                edgecolor=edgecolor, 
                gscale_image=False, 
                colorbar=False, 
                colorbarlabel=None, 
                no_ticks=no_ticks, 
                figsize=figsize)

    return ax

def plot_frame(path, ax=None, title=None, figsize=None):
    '''
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    frame = vid2vib_utils.get_frame(path)
    plot_truecolor_image(frame, ax=ax, title=title, figsize=figsize)

    return ax

def plot_rois(ax, rois, roicolors=['fuchsia']):
    '''
    '''
    if len(roicolors) != len(rois):
        roicolors = [roicolors[0] for i in range(len(rois))]

    for i, roi in enumerate(rois):
        anchor = (roi[2], roi[0])
        w = roi[3] - roi[2]
        h = roi[1] - roi[0]
        
        rect = patches.Rectangle(anchor, w, h, linewidth=2, edgecolor=roicolors[i], facecolor='none')
        ax.add_patch(rect)

def plot_spectrum(freq, pxx, ax=None, peak_idx=None, title=None, ylabel='Power Spectral Density', xlabel='Frequency (Hz)', yscale='linear', color='maroon', figsize=None):
    '''Plots a single power spectral density
    
    Params:
        freq     -- 1d array with spectrum frequencies
        pxx      -- 1d, 2d, or 3d array storing power magnitudes for each time series spectrum
        ax       -- axis to plot spectrum on, creates new axis if None
        peak_idx -- int, index of peak frequency
        title    -- string, plot title
        ylabel   -- string, x axis label, defaults to 'Power Spectral Density'
        xlabel   -- string, y axis label, defaults to 'Frequency (Hz)'
        yscale   -- string, y axis scale, either 'linear' or 'log', defaults to 'linear'
        figsize  -- tuple, figure size

    Returns:
        None
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(freq, pxx, color=color)
    if peak_idx is not None:
        ax.scatter(freq[peak_idx], pxx[peak_idx], marker='.', color='k', s=50)
        peak_label_xpad = (freq[-1] - freq[0]) / 100
        ax.text(freq[peak_idx] + peak_label_xpad, 
                pxx[peak_idx], 
                f'\n Peak Frequency: {freq[peak_idx]:.2f} Hz')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_ylim(top=1.1 * ax.get_ylim()[1])

def plot_timeseries(signal, t=None, ax=None, title=None, color='k', ylabel='', xlabel='', figsize=None):
    '''Plots an individual time series (signal in time)
    
    Params:
        signal  -- 1d arraylike representing amplitude of signal
        t       -- 1d arraylike with time index, if t=None, the t is set to range(len(signal))
        ax      -- axis to plot spectrum on, creates new axis if None
        title   -- string, plot title
        ylabel  -- string, x axis label, defaults to 'Power Spectral Density'
        xlabel  -- string, y axis label, defaults to 'Frequency (Hz)'
        figsize -- tuple, figure size

    Returns:
        None
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if t is None:
        t = range(len(signal))
    ax.plot(t, signal, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

def plot_freq_histogram(dom_freq, freq, ax=None, title='Dominant Frequency Histogram', color='royalblue', figsize=None):
    ''' Plots histogram for frequency heat map

    Params:
        dom_freq -- 1d or 2d array storing dominant frequencies
        freq     -- 1d array with spectrum frequencies (used as bins)
        ax       -- axis to plot spectrum on, creates new axis if None
        title    -- string, title of plot, defaults to 'Dominant Frequency Histogram'
        figsize  -- tuple, figure size

    Returns:
        None
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n, bins = np.histogram(dom_freq.ravel(), bins=freq, density=True)
    ax.plot(bins[:-1], n, color=color)

    ax.set_title(title)
    ax.set_ylabel('Pixel Count')
    ax.set_xlabel('Frequency Bin (Hz)')
    
    argmax = np.argmax(n)
    bin_width = bins[1] - bins[0]
    peak_label_xpad = bin_width
    ax.text(bins[argmax] + peak_label_xpad, 
            n[argmax], 
            f'\n Peak Frequency: {bins[argmax]:.2f} Hz')
    ax.scatter(bins[argmax], n[argmax], marker='.', color='k', s=50)
    ax.set_ylim(top=1.1 * ax.get_ylim()[1])

def plot_mbt_timeseries(levels, ts, axs=None, title='MBT Timeseries', figsize=None):
    '''Plots timeseries obtained from vid2vib_utils.mlt
    
    Params:
        levels  -- 1d array of applied binary thresholds
        ts      -- 1d array of resultant time series
        axs     -- arraylike, axes to plot spectrum on, creates new axis if None
        title   -- string, title of plot, defaults to 'MLT Timeseries'
        figsize -- tuple, figure size
    
    Returns:
        None
    '''
    
    if axs is None:
        fig, axs = plt.subplots(len(levels), 1, sharex=True, figsize=figsize)
    for i, level in enumerate(levels):
        plot_timeseries(ts[:, i], ax=axs[i], title=f'Threshold: {level}')
    fig.suptitle(title)
    fig.supxlabel('Frame')
    fig.supylabel('Num Pixels < Threshold')
    plt.subplots_adjust(hspace=0.9)
    
def plot_mbt_spectra(levels, freq, pxx, peak_idxs=None, axs=None, title='MBT Spectra', yscale='linear', color='maroon', figsize=None):
    '''Plots spectra obtained from multilevel binary thresholding
    
    Params:
        levels    -- 1d array of applied binary thresholds
        freq      -- 1d array with spectrum frequencies
        pxx       -- 2d array storing power magnitudes for each time series spectrum
        peak_idxs -- 1d array with index of dominant frequency for each spectrum
        axs       -- arraylike, axes to plot spectrum on, creates new axis if None
        title     -- string, title of plot, defaults to 'MLT Spectra'
        figsize   -- tuple, figure size
    
    Returns:
        None
    '''
    if axs is None:
        fig, axs = plt.subplots(len(levels), 1, sharex=True, figsize=figsize)
    if peak_idxs is None:
        peak_idxs = [None for i in range(len(levels))]
    for i, level in enumerate(levels):
        plot_spectrum(freq, pxx[:, i], ax=axs[i], peak_idx=peak_idxs[i], title=f'Threshold: {level}', color=color, ylabel=None, xlabel=None, yscale=yscale)
        ylim = axs[i].get_ylim()
        axs[i].set_ylim(ylim[0], ylim[1] * 1.2)
    fig.suptitle(title)
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Power Spectral Density')
    plt.subplots_adjust(hspace=0.9)
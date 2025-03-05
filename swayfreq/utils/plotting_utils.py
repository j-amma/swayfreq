'''Module for effeciently visualizing video processing input and output'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import cv2

from swayfreq.utils import vid2vib_utils

def plot_image(image,
               low=None, 
               high=None, 
               ax=None, 
               title=None, 
               cmap='magma', 
               gscale_image=False, 
               colorbar=True, 
               colorbarlabel=None, 
               edgecolor=None, 
               ticks=False, 
               figsize=None):    
    '''
    Displays a raster image (a wrapper for imshow)
    
    Parameters
    ----------
    image : 1d, 2d, or 3d array
        Input raster image.
    low : float
        Lower bound of displated brightnesses.
    high : float
        Upper bound of displated brightnesses.
    ax : matplotlib.axes.Axes
        Axis to plot image on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot.
    cmap : string
        Name of matplotlib colormap to use. 
        Defaults to magma.
    gscale_image : boolean
        When true, changes the upper bound of the colorbar ticks to be 255.
        Defaults to false.
    colorbar : boolean
        When true, adds colorbar to right of raster image.
        Defaults to true.
    colorbarlabel : string
        Label for the colorbar. 
        Defaults to no label.
    edgecolor : string
        Color to outline the raster with (using matplotlib color labels).
        Default to no outline.
    ticks : boolean
        Adds ticks to raster grid when true.
        Defaults to false.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(image, cmap=cmap, vmin=low, vmax=high)
    
    if not ticks:
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

def plot_grayscale_image(image, ax=None, title=None, edgecolor=None, colorbar=True, colorbarlabel=None, ticks=False, figsize=None):
    '''
    Plots and formats a grayscale image.
    
    Parameters
    ----------
    image : 2d array
        Input raster image.
    ax : matplotlib.axes.Axes
        Axis to plot image on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot.
    edgecolor : string
        Color to outline the raster with (using matplotlib color labels).
        Default to no outline.
    colorbar : boolean
        When true, adds colorbar to right of raster image.
        Defaults to true.
    colorbarlabel : string
        Label for the colorbar. 
        Defaults to no label.
    ticks : boolean
        Adds ticks to raster grid when true.
        Defaults to false.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
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
                ticks=ticks, 
                figsize=figsize)
    return ax

def plot_truecolor_image(image, ax=None, title=None, edgecolor=None, ticks=False, figsize=None):
    '''
    Plots and formats a true color image.
    
    Parameters
    ----------
    image : 3d array
        Input raster image.
    ax : matplotlib.axes.Axes
        Axis to plot image on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot.
    edgecolor : string
        Color to outline the raster with (using matplotlib color labels).
        Default to no outline.
    ticks : boolean
        Adds ticks to raster grid when true.
        Defaults to false.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
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
                ticks=ticks, 
                figsize=figsize)

    return ax

def plot_frame(path, ax=None, title=None, ticks=True, figsize=None):
    '''
    Plots the first frame of the video with the given path.
    
    Parameters
    ----------
    path : string
        Path to the video.
    ax : matplotlib.axes.Axes
        Axis to plot image on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot.
    ticks : boolean
        Adds ticks to raster grid when true.
        Defaults to false.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    frame = vid2vib_utils.get_frame(path)
    plot_truecolor_image(frame, ax=ax, title=title, ticks=ticks, figsize=figsize)

    return ax

def plot_rois(ax, rois, roicolors=['fuchsia']):
    '''
    Plots each roi provided in the given list on the given axis.

    ROIs are defined using the (ymin, ymax, xmin, xmax) convention.
    If a color is not specified for each roi, each ROI will be colored with the first provided color.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot ROIs on. If one is not provided (default),
        a new axis is created.
    rois : listlike
        List of ROIs, where each ROI is defined as (ymin, ymax, xmin, xmax)
    roicolors : listlike
        Color (string) of each ROI

    Returns
    -------
    None
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
    '''
    Plots and formats the power spectral density.

    When peak_idx is provided, additionally plots and labels a marker at the peak of the spectrum.
    
    Parameters
    ----------
    freq : 1d array
        Spectrum frequency bins.
    pxx : 1d, 2d, or 3d array
        PSD magnitudes
    ax : matplotlib.axes.Axes
        Axis to plot ROIs on. If one is not provided (default),
        a new axis is created.
    peak_idx : int
        Index of freq corresponding to peak of pxx. Defaults to None.
    title : string
        Title of the plot. Defaults to None (no title).
    ylabel : string
        Y-axis label, defaults to 'Power Spectral Density'.
    xlabel : string
        X-axis label, defaults to 'Frequency (Hz)'.
    yscale : string
        Scale of y-axis ('liner' or 'log').
    color : string
        Color of PSD curve. Defaults to 'maroon'.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
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

    return ax

def plot_timeseries(signal, t=None, ax=None, title=None, color='k', ylabel='', xlabel='', figsize=None):
    '''
    Plots an arbitrary time series signal.
    
    Parameters
    ----------
    signal : 1-d array
        Time series signal.
    t : 1-d array
        Time index. If t=None, the t is set to range(len(signal)).
    ax : matplotlib.axes.Axes
        Axis to plot ROIs on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot. Defaults to None (no title).
    color : string
        Color of signal. Defaults to 'k' (black).
    ylabel : string
        Y-axis label, defaults to '' (empty).
    xlabel : string
        X-axis label, defaults to '' (empty).    
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if t is None:
        t = range(len(signal))
    ax.plot(t, signal, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return ax

def plot_freq_histogram(dom_freq, freq, ax=None, title='Dominant Frequency Histogram', label_peak=True, density=True, color='royalblue', figsize=None):
    ''' 
    Plots a histogram for the dominant frequency array.

    Each bin the histogram corresponds to one bin in the power spectrum.
    For simplified comparison with power spectra, the histogram is plotted as a line plot instead of using bars.

    Parameters
    ----------
    dom_freq : 1d or 2d array
        Dominant frequency of each input signal.
    freq : 1d array
        Spectrum frequency bins.
    ax : matplotlib.axes.Axes
        Axis to plot ROIs on. If one is not provided (default),
        a new axis is created.
    title : string
        Title of the plot. Defaults to 'Dominant Frequency Histogram'.
    label_peak : boolean
        When true, labels the histogram bin with the greatest count.
    density : boolean
        When true, normalized the histogram to have unit area.
    color : string
        Color of histogram. Defaults to 'royalblue'.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n, bins = np.histogram(dom_freq.ravel(), bins=freq, density=density)
    ax.plot(bins[:-1], n, color=color)

    ax.set_title(title)
    ax.set_ylabel('Pixel Count')
    ax.set_xlabel('Frequency Bin (Hz)')
    
    if label_peak:
        argmax = np.argmax(n)
        bin_width = bins[1] - bins[0]
        peak_label_xpad = bin_width
        ax.text(bins[argmax] + peak_label_xpad, 
                n[argmax], 
                f'\n Peak Frequency: {bins[argmax]:.2f} Hz')
        ax.scatter(bins[argmax], n[argmax], marker='.', color='k', s=50)
        ax.set_ylim(top=1.1 * ax.get_ylim()[1])
    
    return ax

def plot_mbt_timeseries(levels, ts, axs=None, title='MBT Timeseries', figsize=None):
    '''
    Plots MBT timeseries (those obtained from vid2vib_utils.mbt).
    
    Parameters
    ----------
    levels : 1d array
        Binary threshold used to generate signals.
    ts : 2d array
        MBT Time series.
    axs : list of matplotlib.axes.Axes
        Axes to plot signals on (one axis per signal).
    title : string
        Title of the plot. Defaults to 'MBT Timeseries'.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.
    
    Returns
    -------
    axs : list of matplotlib.axes.Axes
        Axes used to plot signals.
    '''
    
    if axs is None:
        fig, axs = plt.subplots(len(levels), 1, sharex=True, figsize=figsize)
    for i, level in enumerate(levels):
        plot_timeseries(ts[:, i], ax=axs[i], title=f'Threshold: {level}')
    fig.suptitle(title)
    fig.supxlabel('Frame')
    fig.supylabel('Num Pixels < Threshold')
    plt.subplots_adjust(hspace=0.9)

    return axs
    
def plot_mbt_spectra(levels, freq, pxx, peak_idxs=None, axs=None, title='MBT Spectra', yscale='linear', color='maroon', figsize=None):
    '''
        Plots power spectrum associated with each MBT signal.
    
    Parameters
    ----------
    levels : 1d array
        Binary threshold used to generate signals.
    freq : 1d array
        Spectra frequency bins (same for all spectra).
    pxx : 2d array
        PSD magnitudes of each MBT signal.
    axs : list of matplotlib.axes.Axes
        Axes to plot signals on (one axis per signal).
    title : string
        Title of the plot. Defaults to 'MBT Spectra'.
    yscale : string
        Scale of y-axis ('liner' or 'log').
    color : string
        Color of PSD curve. Defaults to 'maroon'.
    figsize : tuple
        Size of the figure in inches. Defaults to matplotlib default.
    
    Returns
    -------
    axs : list of matplotlib.axes.Axes
        Axes used to plot spectra.
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
    return axs
'''Class for analyzing the frequency contrent of videos using the Virtual Vision Sensor'''

import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from swayfreq.utils import vid2vib_utils
from swayfreq.utils import spectra_utils
from swayfreq.utils import aggregate_utils
from swayfreq.utils import plotting_utils

class VVSAnalyzer:

    def __init__(self, vid_path, roi, config_path=None):
        '''
        Constructs a VVSAnalyzer object for the given video and ROI.

        Parameters
        ----------
        vid_path : string
            Path to the video of interest.
        roi : listlike
            Region of interest to process (ymin, ymax, xmin, xmax).
        config_path : string
            Path to .json file storing processing parameters.

        Returns
        -------
        None
        '''
        if config_path is not None:
            # load saved processing context
            # TODO
            pass
        self.path = vid_path
        self.roi = roi
        self.suffix = 'vvsanalyzer'
    
    def vid2vib(self, vid2vib_kwargs):
        '''
        Generate vibration signals using the VVS video to vibration translation.

        Parameters
        ----------
        vid2vib_kwargs : dictionary
            Keyword arguments for vid2vib_utils.uncompressed_vid. 
                reduction : string
                    Reduction method (either "gray" for grayscale or "r", "g", or "b" for a particular color channel).
                verbose : boolean
                    Prints progress statements when true.

        Returns
        -------
        vib : 3d array
            Uncompressed video with reduction applied.
        fps : float
            Video framerate.
        '''
        self.vid2vib_kwargs = vid2vib_kwargs
        
        # Read video into array
        self.vib, self.fps = vid2vib_utils.uncompressed_vid(self.path, self.roi, **vid2vib_kwargs)

        # TODO: plot vibration signals if verbose

        return self.vib, self.fps

    def compute_spectra(self, freqmin, freqmax, spectra_func, spectra_func_kwargs):
        '''
        Computes vibration power spectra for each vibration signal.

        Trims the resultant spectra to a range of plausible frequencies between
        freqmin and freqmax.

        Parameters
        ----------
        freqmin : float
            Lower bound for frequency thresholding in Hz.
        freqmax : float
            Upper bound for frequency thresholding in Hz.
        spectra_func : function
            Function from spectra_utils (or custom) for computing spectra.
        spectra_func_kwargs : dictionary
            Keyword arguments for spectra_func.

        Returns
        -------
        freq : array, 1d
            Frequency range (bins), bounded by freqmin and freqmax.
        pxx : array, (1d, 2d, or 3d)
            Array containing the bin magnitudes for each of the input signals 
            (for frequencies between freqmin and freqmax).
        '''
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.spectra_func = spectra_func
        self.spectra_func_kwargs = spectra_func_kwargs
        
        # Compute spectrums
        self.freq, self.pxx = spectra_func(self.vib, self.fps, freqmin, freqmax, **spectra_func_kwargs)
        return self.freq, self.pxx

    def aggregate(self, dom_freq_func, dom_freq_func_kwargs, masks):
        '''
        Aggregates and optionally masks the vibration signals using a user-provided function.

        Returns masked output for each provided mask.
        Masks are defined using a dictionary 
        {'name' : <string>, 'weight' : <string>, 'kwargs' : {'stat_reduc':<tring>, 'threshold' : <float>, 'percentile' : <float>}}

        name : string
            Name of the mask.
        weight : string
            Variable used for create mask. Either 'prominence' or 'magnitude'.
        kwargs : dictionary
            Keyword arguments for the dom_freq_function.

        Parameters
        ----------
        dom_freq_func : function
            Function from aggregate_utils for aggregating vibration spectra.
        dom_freq_func_kwargs : 
            Keyword arguments for dom_freq_func
        masks : list of dictionaries
            Masks to apply to the output.

        Returns
        -------
        agg_df : Pandas.DataFrame
            Dataframe storing extracted frequencies for the unmasked data and each mask.
        pxx_avg : 1d array
            Average power spectrum magnitude.
        pxx_avg_peak_idx : int
            Index of the peak frequency of pxx_avg.
        dom_freq : 2d array
            Unmasked frequency heat map (pixel wise peak frequency).
        masked : list of np.ma.MaskedArray
            Masked frequency heat map (pixel wise peak frequency).
        masked_avg_spectrums : list of 1d arrays
            Average power spectrum magnitude corresponding to each mask.
        masked_avg_spectrums_peak_idxs
            Index of the peak frequency of pxx_avg for each mask.
        '''
        # save processing parameter choices
        self.dom_freq_func = dom_freq_func
        self.dom_freq_func_kwargs = dom_freq_func_kwargs
        self.masks = masks
        
        # UNMASKED DATA
        
        # Find average spectrum and peak
        self.pxx_avg = aggregate_utils.average_spectra(self.pxx)
        self.pxx_avg_peak_idx, _, _ = spectra_utils.find_dominant_peak(self.pxx_avg)
        self.pxx_avg_peak = self.freq[self.pxx_avg_peak_idx]
        
        # Find Peaks
        # here prominence can also include magnitude but is named as such to avoid
        # being confused with the magnitude variable, which only represents power
        self.dom_freq, magnitude, prominence = dom_freq_func(self.freq, self.pxx, **dom_freq_func_kwargs)
        self.unmasked_mode = aggregate_utils.mode(self.dom_freq)
        
        # MASKED DATA

        # Apply masks
        self.masked=[]
        self.masked_modes = []
        self.masked_avg_spectrums = []
        self.masked_avg_spectrums_peak_idxs = []
        self.masked_spectrum_peaks = []
        for i, mask in enumerate(masks):  # masks is a dictionary describing the mask containing at least 'weight' and 'kwargs' keys
            # Assigned mask weight criteria
            w = magnitude if mask['weight'] == 'magnitude' else prominence
            
            # Mask dom freq array
            masked = aggregate_utils.mask_dom_freq(self.dom_freq, w, **mask['kwargs'])
            self.masked.append(masked)
            self.masked_modes.append(aggregate_utils.masked_mode(masked))
            
            # Compute average spectrum and peak of masked pxx
            pxx_masked = ma.masked_array(self.pxx, mask=np.broadcast_to(masked.mask, self.pxx.shape))
            self.masked_avg_spectrums.append(aggregate_utils.average_spectra(pxx_masked))
            
            pxx_avg_peak_idx, _, _ = spectra_utils.find_dominant_peak(self.masked_avg_spectrums[i])
            self.masked_avg_spectrums_peak_idxs.append(pxx_avg_peak_idx)
            self.masked_spectrum_peaks.append(self.freq[pxx_avg_peak_idx])
        
        # COMPILE OUTPUT
        masked_labels = [[f'mask{i}_avg_spectrum_peak', f'mask{i}_mode'] for i in range(len(masks))]
        masked_values = [[self.masked_spectrum_peaks[i], self.masked_modes[i]] for i in range(len(masks))]
        
        labels = ['unmasked_avg_spectrum_peak', 'unmasked_mode']
        values = [self.pxx_avg_peak, self.unmasked_mode]
        for i in range(len(masks)):
            labels = labels + masked_labels[i]  
            values = values + masked_values[i]
        
        agg_dict = {key:value for key, value in zip(labels, values)}
        self.agg_df = pd.DataFrame(agg_dict, index=[0])

        return self.agg_df, self.pxx_avg, self.pxx_avg_peak_idx, self.dom_freq, self.masked, self.masked_avg_spectrums, self.masked_avg_spectrums_peak_idxs

    def report(self, figsize=None):
        '''
        Plots the frame and ROI, frequency heatmaps, and average spectra and heatmap histograms.

        Parameters
        ----------
        figsize : tuple
            Size of the figure in inches. Defaults to matplotlib default.

        Returns
        -------
        agg_df : Pandas.DataFrame
            Dataframe storing extracted frequencies for the unmasked data and each mask.
        '''
        # Plot frame with ROI
        ax = plotting_utils.plot_frame(self.path, title='Camera FOV and ROIs', figsize=figsize)
        plotting_utils.plot_rois(ax, [self.roi])
        ax.legend(['ROI'], loc='upper left')

        # Plot Grayscale ROI, unmasked and masked frequency heat maps
        fig, axs = plt.subplots(1, 2 + len(self.masks), figsize=figsize)
        
        plotting_utils.plot_grayscale_image(self.vib[0], 
                                            ax=axs[0], 
                                            title='Grayscale ROI', 
                                            edgecolor='Fuchsia', 
                                            colorbar=True, 
                                            colorbarlabel='Brightness', 
                                            ticks=False)
        
        plotting_utils.plot_image(self.dom_freq,
                                  low=self.freqmin, 
                                  high=self.freqmax, 
                                  ax=axs[1], 
                                  title='Unmasked Peak Frequency', 
                                  cmap='magma', 
                                  colorbar=True, 
                                  colorbarlabel='Hz',
                                  ticks=False)
        
        for i, mask in enumerate(self.masks):
            name = mask['name']
            plotting_utils.plot_image(self.masked[i], 
                                      low=self.freqmin, 
                                      high=self.freqmax, 
                                      ax=axs[2 + i], 
                                      title=f'Masked Dominant Frequency\n{name}', 
                                      cmap='magma', 
                                      colorbar=True, 
                                      colorbarlabel='Hz',
                                      ticks=False) 
        plt.subplots_adjust(wspace=0.5)

        # Plot average spectra and histograms for unmasked and masked
        # Left column average spectrum, right column histogram
        fig, axs = plt.subplots(1 + len(self.masks), 2, sharex=True, figsize=figsize)
        
        plotting_utils.plot_spectrum(self.freq, self.pxx_avg, ax=axs[0,0], peak_idx=self.pxx_avg_peak_idx)
        plotting_utils.plot_freq_histogram(self.dom_freq, self.freq, ax=axs[0, 1])
       
        pad= 10
        axs[0,0].annotate('Unmasked', xy=(0, 0.5), xytext=(axs[0,0].yaxis.labelpad - pad, 0), xycoords=axs[0,0].yaxis.label, size='large', textcoords='offset points', ha='right', va='center')
        
        for i, mask in enumerate(self.masks):
            name = mask['name']
            plotting_utils.plot_spectrum(self.freq, self.masked_avg_spectrums[i], ax=axs[1 + i,0], peak_idx=self.masked_avg_spectrums_peak_idxs[i])
            plotting_utils.plot_freq_histogram(self.masked[i].compressed(), self.freq, ax=axs[1 + i, 1])
            axs[1 + i,0].annotate(f'Masked:\n{name}', xy=(0, 0.5), xytext=(-axs[1 + i,0].yaxis.labelpad - pad, 0), xycoords=axs[1 + i,0].yaxis.label, size='large', textcoords='offset points', ha='right', va='center')
        plt.subplots_adjust(wspace=0.5)

        cols = ['Average Spectrum', 'Peak Frequency Histogram']
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)
        
        # keep track of figures
        fig_nums = plt.get_fignums()
        self.figs = [plt.figure(n) for n in fig_nums]
        
        # Format agg_df
        self.agg_df.style

        return self.agg_df
        

    def save(self, out_prefix):
        '''
        Save important variables and the processing context.

        Creates files with the given directory.

        Saves:
        * Vibration power spectra
        * Aggregated frequency data
        * Plots
        * Processing variables

        Parameters
        ----------
        out_prefix : string
            Path to a directory to save all files.
        '''
        ### make output directory and subdirectory
        if not os.path.exists(out_prefix):
            os.makedirs(out_prefix, exist_ok=True)
        
        ### save data
        # save raw spectra
        array_dict = {'freq' : self.freq,
                        'pxx' : self.pxx}
        for data_name in array_dict:
            out_fn = os.path.join(out_prefix, data_name + f'_{self.suffix}')
            np.save(out_fn, array_dict[data_name])

        # save aggregated frequency data
        out_fn = os.path.join(out_prefix, 'dominant_frequencies' + f'_{self.suffix}.csv')
        self.agg_df.to_csv(out_fn)

        ### save plots
        plot_names = ['frame_and_roi', 'roi_and_heatmaps', 'spectra']
        
        for fig, plot_name in zip(self.figs, plot_names):
            out_fn = os.path.join(out_prefix , plot_name + f'_{self.suffix}')
            fig.savefig(out_fn)

        ### save metadata
        config_dict = {'vid_path' : self.path,
                       'roi' : self.roi,
                       'vid2vib_kwargs': self.vid2vib_kwargs,
                       'freqmin' : self.freqmin,
                       'freqmax' : self.freqmax,
                       'spectra_func' : str(self.spectra_func),
                       'spectra_kwargs' : self.spectra_func_kwargs,
                       'dom_freq_func' : str(self.dom_freq_func),
                       'dom_freq_func_kwargs' : self.dom_freq_func_kwargs,
                       'masks' : self.masks}
        with open(os.path.join(out_prefix, 'config' + f'_{self.suffix}'), 'w') as outfile:
            json.dump(config_dict, outfile)
        
    
    def analyze(self, 
                vid2vib_kwargs, 
                freqmin, 
                freqmax, 
                spectra_func, 
                spectra_func_kwargs, 
                dom_freq_func, 
                dom_freq_func_kwargs, 
                masks,
                figsize):
        '''
        Complete processing pipeline for analyzing the sway of a ROI in a video.

        Generates vibration signals from a video using VVS, computes power spectra for each
        vibration signal (pixel), aggregates the power spectra, and reports/plots the results.

        Parameters
        ----------
        vid2vib_kwargs : dictionary
            Keyword arguments for vid2vib_utils.uncompressed_vid. 
                reduction : string
                    Reduction method (either "gray" for grayscale or "r", "g", or "b" for a particular color channel).
                verbose : boolean
                    Prints progress statements when true.
        freqmin : float
            Lower bound for frequency thresholding in Hz.
        freqmax : float
            Upper bound for frequency thresholding in Hz.
        spectra_func : function
            Function from spectra_utils (or custom) for computing spectra.
        spectra_func_kwargs : dictionary
            Keyword arguments for spectra_func.
        dom_freq_func : function
            Function from aggregate_utils for aggregating vibration spectra.
        dom_freq_func_kwargs : 
            Keyword arguments for dom_freq_func
        masks : list of dictionaries
            Masks to apply to the output.
        figsize : tuple
            Size of the figure in inches. Defaults to matplotlib default.

        Returns
        -------
        agg_df : Pandas.DataFrame
            Dataframe storing extracted frequencies for the unmasked data and each mask.
        '''
        self.vid2vib(vid2vib_kwargs)
        self.compute_spectra(freqmin, freqmax, spectra_func, spectra_func_kwargs)
        self.aggregate(dom_freq_func, dom_freq_func_kwargs, masks)
        return self.report(figsize)



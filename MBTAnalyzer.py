import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import vid2vib_utils
import spectra_utils
import aggregate_utils
import plotting_utils

class MBTAnalyzer:

    def __init__(self, vid_path, roi, nlevels, config_path=None):
        if config_path is not None:
            # load saved processing context
            pass
        self.path = vid_path
        self.roi = roi
        self.nlevels = nlevels
        self.suffix = 'mbtanalyzer'
    
    def vid2vib(self, vid2vib_kwargs):
        '''
        '''
        self.vid2vib_kwargs = vid2vib_kwargs
        
        # Read video into array
        self.vib, self.fps, self.levels = vid2vib_utils.mbt(self.path, self.roi, nlevels=self.nlevels, **vid2vib_kwargs)
        return self.vib, self.fps, self.levels

    def compute_spectra(self, freqmin, freqmax, spectra_func, spectra_func_kwargs):
        '''
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
        '''
        self.dom_freq_func = dom_freq_func
        self.dom_freq_func_kwargs = dom_freq_func_kwargs
        self.masks = masks

        # Find peaks of spectra
        peak_idxs, _, _ = np.apply_along_axis(spectra_utils.find_dominant_peak, 0, self.pxx)
        self.peak_idxs = peak_idxs.astype(int)
        
        # Find average spectrum and peak
        self.pxx_avg = aggregate_utils.average_spectra(self.pxx)
        self.pxx_avg_peak_idx, _, _ = spectra_utils.find_dominant_peak(self.pxx_avg)
        self.pxx_avg_peak = self.freq[self.pxx_avg_peak_idx]
        
        # Apply masks
        if masks is not None:
            # TODO: implement sensible masking scheme
            pass

        agg_dict = {'avg_spectrum_peak': self.pxx_avg_peak}
        self.agg_df = pd.DataFrame(agg_dict, index=[0])

        return self.agg_df, self.pxx_avg, self.pxx_avg_peak_idx

    def report(self, figsize=None):
        '''
        '''
        ### Plots

        # Plot frame with ROI
        ax = plotting_utils.plot_frame(self.path, title='Camera FOV and ROIs', figsize=figsize)
        plotting_utils.plot_rois(ax, [self.roi])
        ax.legend(['ROI'], loc='upper left')

        # Plot MBT Time series
        plotting_utils.plot_mbt_timeseries(self.levels, self.vib, axs=None, figsize=figsize)
        
        # Plot MBT Spectra
        plotting_utils.plot_mbt_spectra(self.levels, self.freq, self.pxx, self.peak_idxs, figsize=figsize)
    
        # Plot Stacked spectra and average spectrum
        plotting_utils.plot_spectrum(self.freq, self.pxx_avg, peak_idx=self.pxx_avg_peak_idx, yscale='linear', figsize=figsize)
        
        # Save ref to figures
        fig_nums = plt.get_fignums()
        self.figs = [plt.figure(n) for n in fig_nums]
        
        # Format agg_df
        self.agg_df.style

        return self.agg_df
        

    def save(self, out_prefix):
        '''
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
        plot_names = ['frame_and_roi', 'vibration_signals', 'spectra', 'average_spectrum']
        
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

        self.vid2vib(vid2vib_kwargs)
        self.compute_spectra(freqmin, freqmax, spectra_func, spectra_func_kwargs)
        self.aggregate(dom_freq_func, dom_freq_func_kwargs, masks)
        self.report(figsize)
        return self.agg_df
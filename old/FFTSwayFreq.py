""" Class for managing FFT Pixel Frequency Analysis """

import numpy as np
import pandas as pd
import UncompressVid as uv
import vid_preproc as vp
import fft_utils as fu

class FFTSwayFreq():
    
    def __init__(self, path, dims, freqlow, freqhigh, nfreq, rgb = 2, verbose = True):
        self.path = path
        self.dims = dims
        self.fps = None
        self.resolution = None
        self.freqlow = freqlow
        self.freqhigh = freqhigh
        self.nfreq = nfreq
        self.rgb = rgb
        self.fftnsamp = None
        self.prep_data = None
        self.window = False
        self.maxf = None
        self.f_ample = None
        self.frequencies = None
        self.mean = None
        self.std = None
        self.masked = None
        self.mask = None
        self.mask_type = 'Unmasked'
        self.top_n_freq = None
        self.verbose = verbose
        
    def init_video(self):
        # read video
        vid_reader = uv.UncompressVid(self.path)
        self.fps = vid_reader.get_fps()
        self.resolution = vid_reader.get_resolution()
        vid = vid_reader.convert_to_arr(self.dims[0], self.dims[1], self.dims[2], self.dims[3], self.verbose) 
        
        # isolate RGB channel
        self.prep_data = vp.isolate_rbg(vid, self.rgb)

                  
    def window_data(self):
        self.window = True
        self.prep_data = vp.window_data(self.prep_data, verbose = self.verbose)
        
    def map_to_grayscale(self):
        # map all pixels to one of n colors
        # e.g. all pixels in tree black, all other pixels white
        print('Not yet implemented')
    
    def compute_frequency(self, normalize=False, mask = None, nsamp=2048):
        self.fftsamp = nsamp
        self.maxf, self.f_ample, self.frequencies, self.mean, self.std = fu.compute_frequency(self.prep_data, 
                                                                                              self.fps, 
                                                                                              self.freqlow, 
                                                                                              self.freqhigh, 
                                                                                              self.nfreq,
                                                                                              normalize,
                                                                                              nsamp,
                                                                                              self.verbose)
        
    
    def mask_freq(self, mask_type='median', mask_val=None):
        self.mask = 0
        self.mask_type = mask_type
        if (mask_type == 'median'):
            self.mask = self.mean
        elif ((mask_type == 'std_above') or (mask_type == '2std_above') or (mask_type == '3std_above')):
            if (mask_type == 'std_above'):
                self.mask = self.mean + self.std
            elif (mask_type == '2std_above'): 
                self.mask = self.mean + 2 * self.std
            elif (mask_type == '3std_above'):
                self.mask = self.mean + 3 * self.std
        elif(mask_type == 'custom' and mask_val is not None):
            self.mask = mask_val
        else:
            print('Invalid params')
            return
        
        self.masked = fu.filter_by_amp(self.maxf, self.f_ample, self.mask)
        
    
    def get_top_nfreq(self, masked = True):
        # TODO: build support for nonmasked varriant
        self.top_n_freq = fu.top_nfreq(self.masked, self.maxf, self.frequencies, self.nfreq)
        return self.top_n_freq
            
    def plot_freq(self, path = None, unmasked = False, masked = True, cumul = True):
        if (unmasked):
            dscrp = 'Unmasked'
            fu.plot_freq(data, maxf, nfreq, low, high, dscrp)
        if (masked):
            dscrp = 'Mask = ' + self.mask_type
            fu.plot_freq(self.masked, self.maxf, self.nfreq, self.freqlow, self.freqhigh, dscrp)
        if (cumul):
            dscrp = 'Mask = ' + self.mask_type
            fu.plot_cumul_freq_hist(self.maxf, self.frequencies, self.masked, dscrp)
        
    def get_algo_specs(path = None):
        print('Not yet implemented')
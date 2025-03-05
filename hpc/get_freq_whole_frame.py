#!/usr/bin/env python

import numpy as np

from swayfreq.analyzers import vvs_lyzer
from swayfreq.utils import spectra_utils


# name of video sample to process
vid_name = 'manitou-b'

# path to video sample
vid_path = f'.data/{vid_name}.MP4'

# regions of interest
rois = {'region1' : [0, 1079, 0, 1919]}
roi = 'region1'

out_prefix = f'./output/{vid_name}/'

# keyword arguments for VVS video to vibration translation (which uses vid2vib.uncompressed_vid)
vid2vib_vvs_kwargs = {'reduction':'gray'}

# keyword argumetns for MBT video to vibration translation (which uses vid2vib.mbt)
vid2vib_mbt_kwargs = {'reduction':'gray'}

# lower bound of canidate frequencies
freqmin = 0.15

# upper bound of candidate frequencies
freqmax = 0.5

# function used to estimate PSD of vibration signals
spectra_func = spectra_utils.get_spectra_periodogram

# kwargs for spectra_func
spectra_func_kwargs = {'window':'hann', 
                       'nfft':None}

# load region1 and calculate frequencies
vvs_r1 = vvs_lyzer.VVSAnalyzer(vid_path, rois[roi], out_prefix)
vib_vvs_r1, fps = vvs_r1.vid2vib(vid2vib_vvs_kwargs)
freq_vvs_r1, pxx_vvs_r1 = vvs_r1.compute_spectra(freqmin, freqmax, spectra_func, spectra_func_kwargs)

# save output
np.save('./data/freq_out', freq_vvs_r1)
np.save('./data/pxx_out', pxx_vvs_r1)
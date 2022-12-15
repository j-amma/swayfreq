import numpy as np
import datetime
import spectra_utils


def get_accel_spectra(df,
                      fs,
                      t0, 
                      freqmin, 
                      freqmax, 
                      length,
                      spectra_func, 
                      spectra_func_kwargs,
                      date_format='%Y-%m-%d %H:%M:%S',
                      center=False):
    '''
    Estimates spectra for each vibration signal in df.

    Trims each vibration signal in the df to the given start datetime and length.
    Estimates spectra for each trimmed signal using the provided function and kwargs. 
    Trims the output spectra to [freqmin, freqmax].
    Computes average spectrum.

    Parameters
    ----------
    df : DataFrame
        Input signals (Datetime index, each column a signal)
    fs : float
        Sampling frequency in Hz
    t0 : string
        Data start time
    freqmin : float
        Lower bound of candidate frequencies
    freqmax : float
        Upper bound of candidate frequencies
    length : float
        Length of trimmed input signals in minutes
    spectra_func : function
        Function used to estimate spectra
    spectra_func_kwargs : dict
        kwargs for spectra_func
    date_format : string
        How t0 is formatted (using standard Datetime conventions)
    center: boolean
        When true, the data window will have the given length and be centered
        on the start time

    Returns
    -------
    df_trim : Dataframe
        Trimmed input signals (have given start time and length)
    freq : 1d array
        Frequency range (bins), bounded by freqmin and freqmax.
    pxx : array
        Power magnitudes for each spectrum
    pxx_avg : array
        average spectrum power magnitudes
    pxx_avg_peak_idx : int
        Index of average spectrum peak frequency
    '''
    t0_datetime = datetime.datetime.strptime(t0, date_format)
    
    start = t0_datetime - datetime.timedelta(minutes=length/2) if center else t0_datetime
    end = start + datetime.timedelta(minutes=length)
    mask = (df.index > start) & (df.index < end)
    df_trim = df[mask]
    freq, pxx = spectra_func(df_trim, fs, freqmin, freqmax, **spectra_func_kwargs)

    pxx_avg = np.mean(pxx, axis=1)
    pxx_avg_peak_idx, _, _ = spectra_utils.find_dominant_peak(pxx_avg, criteria='magnitude')

    return df_trim, freq, pxx, pxx_avg, pxx_avg_peak_idx
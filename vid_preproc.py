''' Useful preprocessing functions for an uncompressed video. '''
import numpy as np
from scipy.signal.windows import tukey 

def isolate_rbg(data, channel):
    ''' Isolates one RGB channel of the given uncompressed video. 
    
    Param:
        data    -- 4d uncompressed video array with dims [time, y, x, RGB]
        channel -- int in range [0,2] corresponding to RGB
        
    Returns:
        RGB isolated uncompressed video with dims [y,x,time]
    '''
    return np.transpose(data[:,:,:, channel],axes=[1,2,0])

def window_data(data, normalize = True, verbose = True, every = 100):    
    ''' Applies a Tukey window to each pixel time series.
    
    Optionally subtracts the mean from each time series before
    windowing.
    
    Params:
        data      -- 3d array of single channel video data
        normalize -- subtracts mean from each time series when true
        verbose   -- print helpful progress statements
        every     -- column interval for progress statements
    
    Returns:
        windowd video array
    '''
    if (verbose):
        print('Starting Windowing')
    
    window = tukey(len(data[0, 0, :]), 0.1)
    dims = np.shape(data)
    for y in range(dims[0]):
        for x in range(dims[1]):
            # Normalize Data (subtract mean from each pixel time series)
            if (normalize):
                series_avg = np.mean(data[y, x, :])
                data[y, x, :] -= series_avg
            
            # Window
            data[y, x, :] = data[y, x, :] * window
        if(verbose and (y % every == 0)):
            print("   Windowing row " + str(y) + " complete")
    if (verbose):
        print('Windowng Complete')
    return data
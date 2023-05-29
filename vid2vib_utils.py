'''Module for translating videos into vibration signals'''

import cv2
import numpy as np
import sys
import datetime
import ffmpeg

import VideoReader

CHANNEL_MIN = 0  # mininum brightness of channel
CHANNEL_MAX = 255  # maximum brightness of channel

def get_frame(path):
    '''
    Returns the first frame of the video with the given path.
    
    Parameters
    ----------
    path : string
        Path to video, can be absolute or relative.

    Returns
    -------
    frame : array-like, 3d
        First frame of video.
    '''
    vid_capture = cv2.VideoCapture(path)
    ret, frame = vid_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vid_capture.release()
    return frame

def get_video_datetime(path):
    '''
    Given the path to a video, returns its start time as a datetime object.

    Returns start time of the video in video local time.
    
    Shared by Sidney Bush.
    
    Parameters
    ----------
    path : string
        Path to video, can be absolute or relative.

    Returns
    -------
    dt : datetime object
        Video start time.
    '''
    try: 
        probe = ffmpeg.probe(path)
        datetime_str = probe['streams'][0]['tags']['creation_time']
        datetime_str = ' '.join(datetime_str.split('.')[0].split('T'))
        dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return dt
    except ffmpeg.Error as e: 
        print(e.stderr.decode('utf8'))
        raise e

def uncompressed_vid(path, roi, reduction='gray', verbose=True):
    ''' 
    Reads the video roi into an array.

    Applies a user specified reduction to translate the 4d data (time, y, x, rgb)
    into 3d data (time, y, x) where each value in the array represents a pixel brightness.

    For more details about using this approach to get vibration signals, see

    Schumacher, T., & Shariati, A. (2013). Monitoring of Structures and Mechanical Systems Using 
    Virtual Visual Sensors for Video Analysis: Fundamental Concept and Proof of Feasibility. 
    Sensors, 13(12), Article 12. https://doi.org/10.3390/s131216551
    
    Parameters
    ----------
    path: string
        Path to video, can be absolute or relative.
    roi : list-like
        Region of interest (ymin, ymax, xmin, xmax).
    reduction : string  
        Reduction method (either "gray" for grayscale or 
        "r", "g", or "b" for a particular color channel).
    verbose : boolean
        Prints progress statements when true.
        
    Returns
    -------
    outputdata : array, 3d
        3d numpy array [time, y, x].
    fps : float
        Framerate in frames per second.
    '''
    
    if (verbose):
        print("Reading video into array")
    
    # initialize reader (initializes new one each time in case user wants
    # to reread video using same Vid2Vib object)
    vid = VideoReader.VideoReader(path)
    
    ymin, ymax, xmin, xmax = roi
    
    # initialize output
    fps = get_fps(vid)
    num_frame = int(vid.metadata.get('nb_frames'))
    outputdata = np.zeros((num_frame, ymax-ymin, xmax-xmin))
    
    # read frames one by one
    for i, v in enumerate(vid):
        # extract roi from frame
        roi_raw = v[ymin:ymax, xmin:xmax]
        
        # reduce RGB data into single channel
        roi_reduced = reduce_channels(roi_raw, reduction)

        # save reduced roi
        outputdata[i] = roi_reduced
        
        if(verbose): 
            sys.stdout.write('\r{0}'.format(i))
            sys.stdout.flush()

    if (verbose):
        print('\nFinished reading video into array')
    
    return outputdata, fps

def mbt(path, roi, reduction='gray', nlevels=8, verbose=True):
    '''
    Uses multilevel binary thresholding to generate nlevels vibration signals.
    
    Applies to channel reduction and generates nlevel vibration signals using nlevel thresholds
    evenly spaced between the mininum and maximum brightnesses for the roi across all frames.
    
    For each threshold, a time series is generated where the value at each time step
    represents the number of pixels below the threshold.
    
    Note: this is the preferred method for converting a video to a vibration signal
    when memory consumption is a concern.

    See the below paper for more details:

    Ferrer, B., Espinosa, J., Roig, A. B., Perez, J., & Mas, D. (2013). Vibration frequency 
    measurement using a local multithreshold technique. Optics Express, 21(22), 26198–26208. 
    https://doi.org/10.1364/OE.21.026198
    
    Parameters
    ----------
    path : string
        Path to video, can be absolute or relative.
    roi : listlike      
        Region of interest (ymin, ymax, xmin, xmax).
    reduction : string
        Reduction method (either "gray" for grayscale or 
        "r", "g", or "b" for a particular color channel).
    nlevels : int
        Number of thresholds to apply (number of signals to generate).
        Defaults to 8 based on suggestions from authors.
    verbose : boolean
        Print progress statements when true.
    
    Returns
    -------
    outputdata : array, 2d
        Vibration signals [time, level].
    fps : float        
        Framerate in frames per second.
    levels : array, 1d     
        Thresholds used to generate the signals.
    '''
    if (verbose):
        print('Generating vibration time series using multilevel binary thresholding')
        print('Reading roi into array')
    

    vid_roi, fps = uncompressed_vid(path, roi, reduction=reduction, verbose=True)

    if (verbose):
        print('Done reading roi into array')
    
    brightness_min = np.min(vid_roi)
    brightness_max = np.max(vid_roi)

    # compute levels
    levels = np.linspace(brightness_min, 
                         brightness_max, 
                         nlevels + 1, 
                         endpoint=False, 
                         dtype=np.uint8)[1:]
    
    # initialize output array
    vibs = np.zeros((vid_roi.shape[0], nlevels))
    
    if (verbose):
        print('Counting pixels below each threshold')
    
    # iterate over all frames
    for i, frame in enumerate(vid_roi):

        # append number of pixels with intensity below level to appropriate time series
        for j, level in enumerate(levels):
            vibs[i, j] = (np.ravel(frame) < level).sum()
        
        # progress message
        if(verbose): 
            sys.stdout.write('\r{0}'.format(i))
            sys.stdout.flush()
    
    if (verbose):
        print("\nFinished generating signals")
        
    return vibs, fps, levels

def get_fps(vid):
    ''' 
    Returns the frame rate of the video in frames per second. 
    
    Parameters
    ----------
    vid : VideoReader object
        Input video.

    Returns
    -------
    fps : float
        Frame rate in frames per second.
    '''
    fps_str = vid.fps
    fps = fps_str.split('/')
    f1 = int(fps[0])
    f2 = int(fps[1])
    return f1 / f2

def reduce_channels(im, reduction):
    '''
    Reduces 3d image array to 2d array
    
    Returns 2d array [y, x] where each value represents the channel
    brightness at that coordinate. The user can either combine the RGB channels
    using a grayscale conversion or isolate a specific RGB channel.
    
    Parameters
    ----------
    vid : array-like, 3d
        Image array ( [y, x, channel]).
    reduction : string
        Reduction method (either "gray" for grayscale or 
        "r", "g", or "b" for a particular color channel).

    Returns
    -------
    vid_reduced : array-like, 2d
        Reduced image array where each value represents the 
        pixel brightness (either grayscale or isolated channel).
    '''
    # by convention, opencv uses the bgr order
    if reduction == 'gray':
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif reduction == 'b':
        return im[:,:,0]
    elif reduction == 'g':
        return im[:,:,1]
    elif reduction == 'r':
        return im[:,:,2]
    else:
        print("did not pass 'gray', 'b', 'g', or 'r'")
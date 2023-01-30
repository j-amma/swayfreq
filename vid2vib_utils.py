import cv2
import math
import numpy as np
import sys
import datetime
import ffmpeg
import VideoReader

CHANNEL_MIN = 0
CHANNEL_MAX = 255


def get_frame(path):
    vid_capture = cv2.VideoCapture(path)
    ret, frame = vid_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vid_capture.release()
    return frame

def get_video_datetime(path):
    '''
    Extracts datetime from video metadata
    '''
    vr = VideoReader.VideoReader(path)
    metadata = vr.get_metadata()
    
    datetime_str = metadata['tags']['creation_time']
    datetime_str = ' '.join(datetime_str.split('.')[0].split('T'))
    
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    
    return dt

def get_video_datetime(filename):
    """ Given a gopro.mp4 filename, return its start time as a datetime objects
    
    Written by Austin
    """
    try: 
        probe = ffmpeg.probe(filename)
        datetime_str = probe['streams'][0]['tags']['creation_time']
        datetime_str = ' '.join(datetime_str.split('.')[0].split('T'))
        dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return dt
    except ffmpeg.Error as e: 
        print(e.stderr.decode('utf8'))
        raise e

def uncompressed_vid(path, roi, reduction='gray', verbose=True, edge=False, edge_kwargs={'threshold1':200, 'threshold2':400}):
        ''' Reads video roi into an array.
        
        Params:
            roi         -- region of interest (ymin, ymax, xmin, xmax)
            reduction   -- reduction method (either grayscale or isolated rgb channel)
            get_edges   -- applies Canny filter to each frame when True
            canny_params -- dictionary of keyword arguments for Canny filter
            verbose     -- print progress statements when true
            
        Returns:
            outputdata -- 3d numpy array [time, y, x]
            fps        -- framerate in frames per second
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

            if edge:
                roi_reduced = cv2.Canny(roi_reduced, **edge_kwargs)
            
            # save reduced roi
            outputdata[i] = roi_reduced
            
            if(verbose): 
                sys.stdout.write('\r{0}'.format(i))
                sys.stdout.flush()

        if (verbose):
            print("\nFinished reading video into array")
        
        return outputdata, fps

def mbt(vid_path, roi, reduction='gray', nlevels=8, thresh_min=CHANNEL_MIN, thresh_max=CHANNEL_MAX, verbose=True):
    ''' Uses multilevel binary thresholding to generate nlevels vibration signals.
    
    Generates nlevel vibration signals using nlevel thresholds evenly spaced between
    vid2vib_utils.CHANNEL_MIN and vid2vib_utils.CHANNEL_MAX.
    
    For each threshold, a time series is generated where the value at each time step
    represents the number of pixels below the threshold.
    
    Note: this is the preferred method for converting a video to a vibration signal
    when memory consumption is a concern.
    
    Params:
        roi       -- region of interest (ymin, ymax, xmin, xmax)
        reduction -- reduction method (either grayscale or isolated rgb channel)
        nlevels   -- number of thresholds to apply (number of signals to generate)
        verbose   -- print progress statements when true
    
    Returns:
        outputdata -- 2d numpy array [time, level]
        fps        -- framerate in frames per second
        levels     -- 1d numpy array with the thresholds used
    '''
    if (verbose):
        print("Generating vibration time series using multilevel binary thresholding")
        print('Reading roi into array')
    

    vid_roi, fps = uncompressed_vid(vid_path, roi, reduction=reduction, verbose=True, edge=False)

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
        ''' Returns the frame rate of the video in frames per second. '''
        fps_str = vid.fps
        fps = fps_str.split('/')
        f1 = int(fps[0])
        f2 = int(fps[1])
        return math.floor(f1 / f2)

def reduce_channels(im, reduction):
    '''Reduces 3d image array to 2d array
    
    Returns 2d array [y, x] where each value represents the channel
    brightness at that coordinate. The user can either combine the RGB channels
    using a grayscale conversion or isolate a specific RGB channel.
    
    Params:
        vid       -- arraylike, image array (3d array [y, x, channel])
        reduction -- string, reduction method: 'gray' for composite value or 'b', 'g', or 'r' for single channel

    Returns:
        vid_reduced -- 2d array where each value represents the pixel brightness (either grayscale or isolated channel)
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
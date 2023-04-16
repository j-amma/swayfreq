import math
import numpy as np
import sys

import VideoReader
import vid2vib_utils


class Vid2Vib:
    ''' Class for generating vibration signals from a video'''
    
    def __init__(self, path):
        self.path = path
        self.vid = VideoReader.VideoReader(path)
    
    def get_resolution(self):
        ''' Returns the resolution of the video. '''
        return self.vid.resolution
    
    def get_fps(self):
        ''' Returns the frame rate of the video in frames per second. '''
        fps_str = self.vid.fps
        fps = fps_str.split('/')
        f1 = int(fps[0])
        f2 = int(fps[1])
        return math.floor(f1 / f2)
    
    def mbt(self, roi, reduction='gray', nlevels=8, thresh_min=vid2vib_utils.CHANNEL_MIN, thresh_max=vid2vib_utils.CHANNEL_MAX, verbose=True):
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
            levels     -- 1d numpy array
        '''
        if (verbose):
            print("Generating vibration time series using multilevel binary thresholding")
        
        # initialize reader
        vid = VideoReader.VideoReader(self.path)
        
        ymin, ymax, xmin, xmax = roi
        
        # compute levels
        levels = np.linspace(thresh_min, 
                             thresh_max, 
                             nlevels + 1, 
                             endpoint=False, 
                             dtype=np.uint8)[1:]
        
        # initialize output array
        num_frame = int(self.vid.metadata.get('nb_frames'))
        vibs = np.zeros((num_frame, nlevels))
        
        # read frame one by one
        for i, v in enumerate(self.vid):
            # extract roi from frame
            roi_raw = v[ymin:ymax, xmin:xmax]
            
            # reduce RGB data into single channel
            roi_reduced = vid2vib_utils.reduce_channels(roi_raw, reduction)
            
            # append number of pixels with intensity below level to appropriate time series
            for j, level in enumerate(levels):
                vibs[i, j] = (np.ravel(roi_reduced) < level).sum()
            
            # progress message
            if(verbose): 
                sys.stdout.write('\r{0}'.format(i))
                sys.stdout.flush()
        
        if (verbose):
            print("\nFinished generating signals")
            
        return vibs, levels
    
    def uncompressed_vid(self, roi, reduction='gray', get_edges=False, canny_params=None, verbose=True):
        ''' Reads video roi into an array.
        
        Params:
            roi         -- region of interest (ymin, ymax, xmin, xmax)
            reduction   -- reduction method (either grayscale or isolated rgb channel)
            get_edges   -- applies Canny filter to each frame when True
            canny_params -- dictionary of keyword arguments for Canny filter
            verbose     -- print progress statements when true
            
        Returns:
            outputdata -- 3d numpy array [time, y, x]
        '''
        
        if (verbose):
            print("Reading video into array")
        
        # initialize reader (initializes new one each time in case user wants
        # to reread video using same Vid2Vib object)
        vid = VideoReader.VideoReader(self.path)
        
        ymin, ymax, xmin, xmax = roi
        
        # initialize output array
        num_frame = int(self.vid.metadata.get('nb_frames'))
        outputdata = np.zeros((num_frame, ymax-ymin, xmax-xmin))
        
        # read frames one by one
        for i, v in enumerate(vid):
            # extract roi from frame
            roi_raw = v[ymin:ymax, xmin:xmax]
            
            # reduce RGB data into single channel
            roi_reduced = vid2vib_utils.reduce_channels(roi_raw, reduction)
            if get_edges:
                roi_reduced = cv2.Canny(roi_reduced, edge_params['threshold1'], edge_params['threshold2'])
            
            # save reduced roi
            outputdata[i] = roi_reduced
            
            if(verbose): 
                sys.stdout.write('\r{0}'.format(i))
                sys.stdout.flush()

        if (verbose):
            print("\nFinished reading video into array")
        
        return outputdata

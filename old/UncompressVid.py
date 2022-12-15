import numpy as np
import VideoReader as vr
import math

class UncompressVid:
    ''' Class loading a video into an array and collecting metadata.'''
    
    def __init__(self, path):
        self.vid = vr.VideoReader(path)
    
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
    
    def convert_to_arr(self, ymin, ymax, xmin, xmax, verbose = True, every = 500):
        ''' Reads frame subregion of the video into an array.
        
        Assumes the given path is valid.
        
        Params:
            ymin    -- lower y value of frame subregion
            ymax    -- upper y value of frame subregion
            xmin    -- lower x value of frame subregion
            xmax    -- upper x value of frame subregion
            verbose -- print progress statements when true
            every   -- frame print interval for progress statements
        
        Returns:
            outputdata -- 4d numpy array [time, y, x, RGB]
        '''
        
        if (verbose):
            print("Reading video into array")
        num_frame = int(self.vid.metadata.get('nb_frames'))
        outputdata = np.zeros((num_frame, ymax-ymin, xmax-xmin, 3))  # 3 for RGB channels
        frame = 0
        for v in self.vid:
            if(frame < num_frame):
                outputdata[frame] = v[ymin:ymax, xmin:xmax]
                frame += 1
            if((frame % every == 0) and verbose): 
                print("   Completed appending frame ", frame)
        if (verbose):
            print("Finished reading video")
        return outputdata
    





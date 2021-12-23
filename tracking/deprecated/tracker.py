################################################################################
#                                                                              #
#   Ant tracking code for python3                                              #
#                                                                              #
#       The Code takes a .mp4 video file as input and detects features,        #
#       and saves all detections into hdf5 format.                             #
#                                                                              #
#   Version 1.2.4                                                              #
#   Code written by Dawith Lim                                                 #
#   First written on 2019/09/18                                                #
#   Last modified: 2021/02/03                                                  #
#                                                                              #
#   Packages used                                                              #
#   -   argparse: Argument parser, allows me to use required & optional        #
#                 inputs with various flags                                    #
#   -   cv2: OpenCV module, used for image type conversion                     #
#   -   matplotlib: Plotting. Required for montage frame generation            #
#   -   numba: LLVM Machine code JIT compiler for numpy operations. This       #
#              is supposedly faster than regular compilation for numpy.        #
#   -   numpy: Useful for array manipulation and general calculations          #
#   -   os: Used for directory navigation/creation.                            #
#   -   pandas: Scientific data formatting package. Dependency for trackpy     #
#               code because that's how it saves the data.                     #
#   -   sys: Used for the sys.exit() to terminate the code                     #
#   -   trackpy: Soft-matter particle tracker; main module that handles        #
#                tracking.                                                     #
################################################################################


import argparse
import cv2 as cv
import h5py
import matplotlib as mpl
# Sets non-GUI backend for matplotlib to suppress plot output; for some reason
# trackpy pops open tracking progress on graphics always and I couldn't find
# a built in option to turn it off. Necessary when using on terminal that 
# doesn't have access to the GUI/window handling processes on the system
# (e.g. WSL, ssh or screen) because the code seems to get stuck after trying 
# to display stuff.
mpl.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import os
import pandas as pd
import sys
import trackpy as tp


class AntTracker:

    def __init__(self, exp):
#  exp = experiment id, name of the video file without the extension
        self.exp = exp
        filepath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(filepath)
        self.vidpath = '../../data/videos/'
        self.outpath = '../../data/trajectories/'
        self.filename = '{}data.hdf5'.format(exp)
        #self.datafile = h5py.File(self.filename, 'w')


        self.vid = cv.VideoCapture('{}{}.mp4'.format(self.vidpath, exp))
        self.check_video_capture()

        self.create_output_dir()

# Create a background substractor object and load prepared background image
        self.backsub = cv.createBackgroundSubtractorMOG2()
        bgnd = cv.imread('background.png')
        self.bgnd = cv.cvtColor(bgnd,cv.COLOR_BGRA2GRAY)


# Parameters

# Background subtraction parameters
        self.tozero_thresh1 = 64
        self.tozero_thresh2 = 200
        self.exponent = 1.02
        self.iter = 12
        self.clipval = 20

# Detection parameters
        self.antsize = 31
        self.minmass = 33000

# Auxiliary parameters for testing only
        self.skip = True
        self.skiplength = 0
        self.test = True
        self.testlength = 2

# Learning rate 0-1 determines how much to modify the background
# based on changes between previous and current frame. 
# In this case, background is externally supplied via import, so
# it is set to 1, but then changed to 0 (i.e. subsequent frames
# does not modify the background mask at all).
        empty = np.empty(np.shape(bgnd))


    def check_video_capture(self):
        success, frame = self.vid.read()
        if not success:
            raise IOError('Video file read failed: {}.mp4'.format(self.exp))

        print('Successfully opened {}.mp4.\n'.format(self.exp))
        size = np.shape(frame)
        print('Frame size: {}x{}'.format(size[0],size[1]))
        print('Channel number: {}'.format(size[2]))


    def create_output_dir(self):
        if os.path.isdir(self.exp):
            print('Directory [{}] already exists'.format(self.exp))
        else:
            os.mkdir(self.exp)
            print('Created directory [{}]'.format(self.exp))


    def proc_frame(self, frame):
# Main routine; convert to greyscale, subtract background, and then detect
# features.

        frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
        #frame = self.backsub.apply(frame, learningRate = 0)

        ret, frame = cv.threshold(255 - frame, self.tozero_thresh1,
                                  255, cv.THRESH_TOZERO)
        cv.imwrite('musk.png', frame)
        cont = np.clip(((255 - frame) ** self.exponent), 0, 255)
        for i in range(self.iter):
            cont = np.clip((cont - self.clipval), 0, 255)
            cont = np.clip((cont ** self.exponent), 0, 255)
        cv.imwrite('mask.png', cont)
        mask = 255 - cont
        ret, frame = cv.threshold(mask, self.tozero_thresh2,
                                  255, cv.THRESH_TOZERO)
        kernel = np.ones((3,3))
        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, iterations = 1)
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations = 2)
        

        feature = tp.locate(
                    frame, # Input image mask
                    self.antsize, # Estimated size of features in pixels.
                    invert = False, # Color inversion
                    noise_size = 0, # Size of Gaussian noise kernel
                    minmass = self.minmass, # Minimum optical mass of features
                    max_iterations = 50,
                    separation = 1,
                    preprocess = True)

        return feature


    def run(self):
        count = 0
        while self.skip and count < self.skiplength:
            success, frame = self.vid.read()
            count += 1
        count = 0

        success, frame = self.vid.read()
        frame = self.backsub.apply(self.bgnd, learningRate = 1)
        success, frame = self.vid.read()
# Dataset shape is set in a way that allows me to append rows of data
# Rather than having to define a predetermined shape, because the code
# Doesn't know the total length of the video input in advance.
# This is slower than having a predetermined size, so in the future
# It may be good for experiment code to write a params file that
# passes the number of frames to this code. 
        '''dataset = {}'''
        first = True

        
        while success:
            feature = self.proc_frame(frame)
            feature.loc[:, 'frame'] = pd.Series(count, index = feature.index)
            #print(feature.head())                
            
            ''' 
            try:
                iterr = tp.link_df_iter((old, feature), 20)
                labeled = pd.concat(iterr)
                print(labeled)
                if (count % 2) != 0:
                    for entry in labeled.to_numpy():
                        key = int(entry[-1])
                        try:
                            dset = dataset[key]
                        except KeyError:
                            dataset[key] = spawn_dataset(key)
                            dset = dataset[key]
                        feat = np.zeros(10)
                        feat[:] = entry
                        dset[-1] = feat
                        dset.resize((dset.shape[0] + 1, 10))
                        datafile.flush()
# Using except: pass is bad practice, change it to the 'proper' way later
            except IndexError:
                print('Index Error; something is wrong')
                pass
                #dataset[count] = np.full((1, 9), np.nan, dtype = np.float32)
            '''
            if first:
                dframe = feature
                first = False
            else:
                dframe = dframe.append(feature)
            count += 1
            print('Frame {} processed.'.format(count))

            #datafile.flush()
            success, frame = self.vid.read()
            old = feature
            
# Premature termination condition for testing
            if (self.test) and (count == self.testlength):
                success = False

# Dump the data in the unmodified, original structure
        dframe.to_hdf('{}{}'.format(self.outpath, self.filename),'raw',
                      mode = 'w')
        print('Linking ant trajectorees')


def spawn_dataset(key):
    dset = datafile.create_dataset(
                            'antdata{}'.format(key),
                            (1, 10), # Initial dataset shape
                            dtype = np.float32, 
                            maxshape = (None, 10),
                            chunks = (1, 10))
    return dset


if __name__ == '__main__':
# Enable numba JIT compiler for trackpy
    tp.enable_numba()
    tp.linking.Linker.MAX_SUB_NET_SIZE = 100

# Create an argument parser object
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-f', '--file', required = True, 
                    help = '.mp4 video file name without the extension')
    args = vars(ap.parse_args())
   
    at = AntTracker(args['file'])
# Open a data file to save the trajectory

    print('Initialized.')
    print('Running ant tracker')
    at.run()

    print('Process completed. Exiting')
    sys.exit(0)

# EOF

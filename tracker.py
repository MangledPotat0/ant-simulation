################################################################################
#                                                                              #
#   Ant tracking code for python3                                              #
#                                                                              #
#       The Code takes a .mp4 video file as input and detects features, forms  #
#       a trajectory and saved in hdf5 format. The code also produces a        #
#       montage so that the user can visually assess the quality of the        #
#       detection.                                                             #
#                                                                              #
#   Version 1.0.0.0.1.0                                                        #
#   Code written by Dawith Lim                                                 #
#   First written on 2019/09/18                                                #
#   Last modified: 2020/06/23                                                  #
#                                                                              #
#   Packages used                                                              #
#   -   argparse: Argument parser, allows me to use required & optional        #
#                 inputs with various flags                                    #
#   -   cv2: OpenCV module, used for image type conversion                     #
#   -   matplotlib: Plotting. Required for montage frame generation            #
#   -   numba: LLVM Machine code JIT compiler for numpy operations             #
#   -   numpy: Useful for array manipulation and general calculations          #
#   -   os: Used solely to do mkdir                                            #
#   -   pandas: Scientific data formatting package. Dependency for trackpy     #
#                 code because that's how it saves the data.                   #
#   -   pims: Image handler for trackpy                                        #
#   -   sys: Used for the sys.exit() to terminate the code                     #
#   -   trackpy: Soft-matter particle tracker                                  #
#                                                                              #
################################################################################

import argparse
import cv2 as cv
import h5py
import matplotlib as mpl
mpl.use('Agg') # Sets non-GUI backend for matplotlib to suppress plot output
import matplotlib.pyplot as plt
import numba
import numpy as np
import os
import pandas as pd
import pims
import sys
import trackpy as tp

class AntTracker:
    def __init__(self, exp, vfn):
        self.vfn = vfn
        self.exp = exp
        filepath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(filepath)
        self.vidpath = '../data/videos/'
        self.outpath = '../data/trajectories/{}'.format(exp)
        self.datafile = '{}data.hdf5'.format(exp)

        self.vid = cv.VideoCapture('{}{}'.format(vidpath,vfn))
        self.check_video_capture()

        self.create_output_dir()

        # Create a background substractor object
        self.backsub = cv.createBackgroundSubtractorMOG2()
        bgnd = cv.imread('background.tiff')
        bgnd = cv.cvtColor(bgnd,cv.COLOR_BGRA2GRAY)
        bgnd = self.backsub.apply(bgnd, learningRate=1)

    def check_video_capture(self):
        success, frame = self.vid.read()
        if not success:
            raise IOError('Video file read failed: {}'.format(self.vfn))

        print('Successfully opened {}.\n'.format(self.vfn))
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
        mask = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
        mask = self.backsub.apply(mask, learningRate=0)
        feature = tp.locate(mask, 43, invert=False, noise_size=3,
                            minmass=15000, max_iterations=100)
        return feature

    def run(self, datafile):
        count = 0
        success, frame = self.vid.read()
        success, frame = self.vid.read()
        self.backsub.apply(cv.cvtColor(frame, cv.COLOR_BGRA2GRAY))
        dataset = datafile.create_dataset('antdata', (1,9),dtype = np.float32, 
                                     maxshape=(None,9), chunks=(1,9))
        while success:
            feature = self.proc_frame(frame)
            feature.loc[:,'frame'] = pd.Series(count, index=feature.index)
            print(feature.head())                
            try:
                dataset[count] = feature
            except:
                dataset[count] = np.full((1,9),np.nan,dtype=np.float32)

            print('Frame {} processed.'.format(count))
            count += 1
            dataset.resize((dataset.shape[0]+1,9))
            datafile.flush()
            success, frame = self.vid.read()
        datafile.close()

def main():
    # Enable numba JIT compiler for trackpy
    tp.enable_numba()

    # Create an argument parser object
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-f', '--file', required=True, 
                    help='.mp4 video file name without the extension')
    args = vars(ap.parse_args())
    vfn = args['file'] + '.mp4'
   
    datafile = h5py.File('{}{}data.hdf5'.format(outpath,args['file']), 'w')

    print('Initialized.')

    at = AntTracker(args['file'], vfn)
    print('Running ant tracker')
    at.run(datafile)
    print('Linking ant trajectorees')
    #at.link_trajectory()
    print('Process exited normally.')
    sys.exit(0)

if __name__ == '__main__':
    main()

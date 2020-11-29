###############################################################################
#                                                                             #
#   Density distribution plot code                                            #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version 1.1.3                                                             #
#   First written on 2020/06/24                                               #
#   Last modified: 2020/11/28                                                 #
#                                                                             #
#   Description:                                                              #
#     This code divides input image stream into square bins, and integrates   #
#     pixel brightness inside each bins to obtain the optical density of      #
#     the ants in the imaging setup.                                          #
#                                                                             #
#   Packages used:                                                            #
#   -   argparse: Input argument parsing                                      #
#   -   cv2: OpenCV library; used for convenient image processing             #
#   -   math: Used here to call the ceiling function                          #
#   -   numpy: Used for array manipulation                                    #
#   -   os: Directory navigation                                              #
#                                                                             #
###############################################################################

import matplotlib.animation as ani
import argparse
import cv2 as cv
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys

class Processor():

    def __init__(self,args):

#  self.fileid is the identifier of the experiment; basically the filename
#  without the file extension.
#  binsize is the size of the bin (in pixels!) along the linear dimension
#  of the image. The bins are squares in shapes.
#  antcount = number of ants in the video that's being processed.
        self.fileid = args['file']
        self.binsize = args['binsize']
        self.antcount = args['antcount']
        
        codepath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(codepath)
        
#  Default directory of where the video files are located, relative to the
#  working directory (where codebase is located).
        self.filepath = '../data/videos/'

#  Create a background subtractor object.
#  The background subtractor is fed the pre-produced background image with
#  learning rate of 1 (i.e. makes a carbon copy of the backgnd image), and
#  then in the subsequent calls for background subtraction the learning rate
#  is set to 0. This was done to avoid inactive ant from being erroneously
#  marked as background and get removed.

        self.backsub = cv.createBackgroundSubtractorMOG2()
        self.bgnd = cv.cvtColor(cv.imread('background.tiff'),
                                cv.COLOR_BGRA2GRAY)
        self.frame = self.backsub.apply(self.bgnd, learningRate = 1)
        
#  Call the video file. I set the extension to mp4 specifically because
#  this format seem to have the best support in general. The original file
#  output from experiment is in .h264 raw format and I coudn't find opencv
#  support for any of the raw formats. (It throws errors when attempted)
        self.video = cv.VideoCapture('{}{}.mp4'.format(self.filepath,
                                     self.fileid))
        success, frame = self.video.read()
        if not success:
            raise IOError('Failed to read video file {}.mp4'.format(
                          self.fileid))
        else:
            print ('Successfully opened video file {}.mp4.\n'.format(
                   self.fileid))
            self.size = np.shape(frame)

#  Check the size and length of the videos is consistent with expectation
            print('Frame size: {}x{}\n'.format(self.size[0], self.size[1]))
            print('Color channels: {}\n'.format(self.size[2]))
            print('Process initialized\n')

#  Load the first frame before entering the main loop
        self.frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
        self.frame = self.backsub.apply(self.frame, learningRate = 0)
        ret, self.frame = cv.threshold(self.frame, 245, 255, cv.THRESH_BINARY)

    def run(self):

#  Total optical mass is computed, then frame is normalized so that the 
#  code gives density as a fraction in the unit of ants.
        sumval = np.sum(self.frame)
        self.frame = (self.antcount / sumval) * self.frame

        imstack = np.array([proc_frame(self.binsize, self.frame)])
        success, frame = self.video.read()
        success, frame = self.video.read()
        success = True

#  If the process for first frame happened without failing, then proceed
#  with the rest of the video until frame load fails.
        while success:
            self.frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
            self.frame = self.backsub.apply(self.frame, learningRate = 0)
            ret, self.frame = cv.threshold(self.frame, 245, 255, 
                                           cv.THRESH_BINARY)
            sumval = np.sum(self.frame)
            self.frame = (self.antcount / sumval) * self.frame
            imstack=np.vstack((imstack, [proc_frame(self.binsize, self.frame)]))
            success, frame = self.video.read()

        return imstack

    def export(self, imstack):
#  Export the ant density by bins as a time series in a data file.
#  Data shape is (nframes * xbins * ybins)
        imstack = imstack / 255
        datafile = h5py.File('{}.hdf5'.format(self.fileid), 'w')
        dset = datafile.create_dataset('{}x{}'.format(self.binsize,
                                                      self.binsize),
                                       data = imstack,
                                       dtype = np.float32)
        datafile.close()

    def plot(self,imstack):
        plotstack = []
        ct = 0
        try:
            os.makedirs('../data/density/{}{}'.format(self.fileid, 
                                                      self.binsize))
        except:
            print('Directory already exists.\n')
        
        maxval = np.max(imstack)
        
        for frame in imstack:
            plt.figure(figsize=(5.5, 5.5))
            plt.imshow(frame, cmap = 'Blues',
                       interpolation = 'nearest', 
                       vmin = 0, vmax = maxval)
            plt.colorbar()
            plt.savefig(
                    '../data/density/{}{}/{}{}{}.png'.format(
                            self.fileid, self.binsize,
                            self.fileid, self.binsize, ct),
                    bbox_inches = 'tight')
            plt.close()
            ct += 1


def proc_frame(binsize, frame):
#  This function bins optical mass into corresponding spatial bins.
    size = np.shape(frame)
    integrated = np.empty((math.ceil(size[0] / binsize) - 1,
                math.ceil(size[1] / binsize) - 1))
    integrated.fill(np.nan)

    i, j = 1, 1
    for row in integrated:
        for column in integrated:
            integrated[i - 1, j - 1] = np.sum(
                    frame[i * binsize:(i + 1) * binsize - 1,
                    j * binsize : (j + 1) * binsize - 1]
            j += 1
        j = 0
        i += 1

    return integrated

def main():
    
# Handle input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument(
                '-f', '--file', required = True,
                help = 'mpeg-4 video file name, without the file extension')
    ap.add_argument(
                '-b', '--binsize', required = True,
                type = np.int8, help = 'Bin size')
    ap.add_argument(
                '-n', '--antcount', required = True,
                type = np.int8, help = 'Number of ants in video')
    args = vars(ap.parse_args())
    
    proc = Processor(args)
    imstack = proc.run()
    proc.export(imstack)
    proc.plot(imstack)

    sys.exit(0)

if __name__ == '__main__':
    main()

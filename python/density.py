###############################################################################
#                                                                             #
#   Density distribution plot code                                            #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version 1.2.10                                                            #
#   First written on 2020/06/24                                               #
#   Last modified: 2021/01/15                                                 #
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
import time as tt

class Processor():

    def __init__(self, args):

#  self.fileid is the identifier of the experiment; basically the filename
#  without the file extension.
#  binsize is the size of the bin (in pixels!) along the linear dimension
#  of the image. The bins are squares in shapes.
#  antcount = number of ants in the video that's being processed.
        self.fileid = args['file']
        self.bincount = args['bincount']
        self.antcount = args['antcount']
        print(self.antcount)
        
        codepath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(codepath)
        
#  Default directory of where the video files are located, relative to the
#  working directory (where codebase is located).
        self.filepath = '../../data/videos/'
        self.outpath = '../../data/density/'

#  Create a background subtractor object.
#  The background subtractor is fed the pre-produced background image with
#  learning rate of 1 (i.e. makes a carbon copy of the backgnd image), and
#  then in the subsequent calls for background subtraction the learning rate
#  is set to 0. This was done to avoid inactive ant from being erroneously
#  marked as background and get removed.

        self.backsub = cv.createBackgroundSubtractorMOG2()
        self.bgnd = cv.cvtColor(cv.imread('background.png'),
                                cv.COLOR_BGRA2GRAY)
        self.frame = self.backsub.apply(self.bgnd, learningRate = 1)
        self.binsize = np.shape(self.frame) / self.bincount
        
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

        imstack = np.array([proc_frame(self.bincount,
                                       self.binsize,
                                       self.frame)])
        success, frame = self.video.read()
        success, frame = self.video.read()
        success = True

        kernel = np.ones((3, 3), dtype = np.uint8)

#  If the process for first frame happened without failing, then proceed
#  with the rest of the video until frame load fails. Typically this will
#  happen when the reader reaches the end of the video file, but this is not
#  necessarily guaranteed!

#  Premature termination condition for testing
        pkill = 0
        target = 100

        while success:
            self.frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
            self.frame = self.backsub.apply(self.frame, learningRate = 0)
            self.frame = 255. - self.frame
            ret, self.frame = cv.threshold(self.frame, 60, 255, 
                                           cv.THRESH_TOZERO)
            cont = np.clip(((255 - self.frame) ** 1.06), 0, 255)
            for i in range(4):
                cont = np.clip((cont - 12), 0, 255)
                cont = np.clip((cont ** 1.05), 0, 255)
            self.frame = cont.astype('float64')
            self.frame = cv.morphologyEx(self.frame, cv.MORPH_OPEN, kernel, 
                                         iterations = 2)
            sumval = np.sum(self.frame)
            self.frame = (self.antcount / sumval) * self.frame
            imstack = np.vstack((imstack, [proc_frame(self.bincount,
                                                      self.binsize,
                                                      self.frame)]))
            
            success, frame = self.video.read()
            pkill += 1
            #if pkill == target:
            #    success = False
            #    print('Process killed at {} frames. Comment out {}'.format(
            #            target, 'lines 139-142 to avoid this.'))

        return imstack

    def export(self, imstack):
#  Export the ant density by bins as a time series in a data file.
#  Data shape is (nframes * xbins * ybins)
        imstack = imstack / 255
        datafile = h5py.File('{}{}{}.hdf5'.format(self.outpath,
                                                  self.fileid,
                                                  self.bincount), 'w')
        dset = datafile.create_dataset('{}x{}'.format(self.bincount,
                                                      self.bincount),
                                       data = imstack,
                                       dtype = np.float32)
        datafile.close()

    def plot(self,imstack):
        print(np.shape(imstack))
        plotstack = []
        ct = 0
        try:
            os.makedirs('{}{}{}'.format(self.outpath,
                                        self.fileid, 
                                        self.bincount))
        except:
            print('Directory already exists.\n')
        
        maxval = np.max(imstack)
        fig = plt.figure(figsize=(5.5, 5.5))
        ims = []
        for frame in imstack:
            ims.append((plt.pcolor(frame,
                                   norm = plt.Normalize(0, self.antcount),
                                   cmap = 'Blues'), ))

        anim = ani.ArtistAnimation(fig, ims)
        #fig.colorbar()
            
        anim.save('../../data/density/{}{}/{}{}.mp4'.format(
                                self.fileid, self.bincount,
                                self.fileid, self.bincount),
                 fps = 10)
        plt.close(fig)


def proc_frame(bincount, binsize, frame):
#  This function bins optical mass into corresponding spatial bins.
    size = np.shape(frame)
    integrated = np.empty((bincount, bincount))
    integrated.fill(999)

    for i in range(len(integrated)):
        for j in range(len(integrated[i])):
            xlow = int(np.round(i * binsize[1]))
            xhigh = int(np.round( (i + 1) * binsize[1] - 1))
            ylow = int(np.round(j * binsize[1]))
            yhigh = int(np.round((j + 1) * binsize[1] - 1))
            integrated[i, j] = np.sum(frame[xlow:xhigh,ylow:yhigh])

    return integrated


if __name__ == '__main__':
    start = tt.time()
# Handle input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument(
                '-f', '--file', required = True,
                help = 'mpeg-4 video file name, without the file extension')
    ap.add_argument(
                '-b', '--bincount', required = True,
                type = np.int8, help = 'Number of bins along one direction')
    ap.add_argument(
                '-n', '--antcount', required = True,
                type = np.int16, help = 'Number of ants in video')
    args = vars(ap.parse_args())
    
    proc = Processor(args)
    imstack = proc.run()
    proc.export(imstack)
    proc.plot(imstack)

    end = tt.time()

    print('Optical density processing completed.\n Runtime: {}'.format(
            end - start))
    sys.exit(0)


# EOF

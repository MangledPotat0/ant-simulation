################################################################################
#                                                                              #
#   Density distribution plot code                                             #
#   Code written by Dawith Lim                                                 #
#                                                                              #
#   Version 1.1.2.1.2.0                                                        #
#   First written on 2020/06/24                                                #
#   Last modified: 2020/06/24                                                  #
#                                                                              #
#   Description:                                                               #
#     This code divides input image stream into square bins, and integrates    #
#     pixel brightness inside each bins to obtain the optical density of       #
#     the ants in the imaging setup.                                           #
#                                                                              #
#   Packages used:                                                             #
#   -   argparse: Input argument parsing                                       #
#   -   cv2: OpenCV library; used for convenient image processing              #
#   -   math: Used here to call the ceiling function                           #
#   -   numpy: Used for array manipulation                                     #
#   -   os: Directory navigation                                               #
#                                                                              #
################################################################################

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
        self.fileid = args['file']
        self.binsize = args['binsize']
        self.antcount = args['antcount']
        codepath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(codepath)
        self.filepath = '../data/videos/'

        self.backsub = cv.createBackgroundSubtractorMOG2()
        self.bgnd = cv.cvtColor(cv.imread('background.tiff'),cv.COLOR_BGRA2GRAY)
        self.frame = self.backsub.apply(self.bgnd,learningRate=1)
        
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

            print('Frame size: {}x{}\n'.format(self.size[0],self.size[1]))
            print('Color channels: {}\n'.format(self.size[2]))
            print('Process initialized\n')
        self.frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
        self.frame = self.backsub.apply(self.frame, learningRate=0)
        ret, self.frame = cv.threshold(self.frame,245,255,cv.THRESH_BINARY)

    def run(self):
        sumval = np.sum(self.frame)
        self.frame = (self.antcount/sumval) * self.frame
        imstack = np.array([proc_frame(self.binsize,self.frame)])
        success, frame = self.video.read()
        success, frame = self.video.read()
        success = True
        while success:
            self.frame = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)
            self.frame = self.backsub.apply(self.frame, learningRate=0)
            ret, self.frame = cv.threshold(self.frame,245,255,cv.THRESH_BINARY)
            sumval = np.sum(self.frame)
            self.frame = (self.antcount/sumval) * self.frame
            imstack=np.vstack((imstack,[proc_frame(self.binsize,self.frame)]))
            success, frame = self.video.read()

        return imstack

    def export(self,imstack):
        print(np.shape(imstack))
        imstack = imstack/255
        datafile = h5py.File('{}.hdf5'.format(self.fileid),'w')
        dset = datafile.create_dataset('{}x{}'.format(
            self.binsize,self.binsize),data=imstack,dtype=np.float32)
        datafile.close()

    def plot(self,imstack):
        plotstack = []
        ct=0
        try:
            os.makedirs('../data/density/{}{}'.format(self.fileid,self.binsize))
        except:
            print('Directory already exists.\n')
        maxval = np.max(imstack)
        for frame in imstack:
            plt.figure(figsize=(5.5,5.5))
            plt.imshow(frame, cmap='Blues',interpolation='nearest',vmin=0,
                    vmax=maxval)
            plt.colorbar()
            plt.savefig('../data/density/{}{}/{}{}{}.png'.format(
                        self.fileid,self.binsize,self.fileid,self.binsize,ct))
            plt.close()
            ct += 1


def proc_frame(binsize,frame):
    size = np.shape(frame)
    integrated = np.empty((math.ceil(size[0]/binsize)-1,
                math.ceil(size[1]/binsize)-1))
    integrated.fill(np.nan)

    i,j= 1,1
    for row in integrated:
        for column in integrated:
            integrated[i-1,j-1]=np.sum(frame[i*binsize:(i+1)*binsize-1,
                j*binsize:(j+1)*binsize-1])
            j += 1
        j=0
        i += 1

    return integrated

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help=
            'mpeg-4 video file name, without the file extension')
    ap.add_argument('-b', '--binsize', required=True, type=int, help='Bin size')
    ap.add_argument('-n', '--antcount', required=True, type=int,
                    help='Number of ants in video')
    args = vars(ap.parse_args())
    
    proc = Processor(args)
    imstack = proc.run()
    proc.export(imstack)
    proc.plot(imstack)

    sys.exit(0)

if __name__=='__main__':
    main()

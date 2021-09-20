###############################################################################
#   Ant labeled montage generator                                             #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 2.1.0                                                            #
#   First written on: 2020/12/20                                              #
#   Last modified: 2021/01/20                                                 #
#                                                                             #
#   Packages used                                                             #
#   -   numpy: Useful for array manipulation and general calculations         #
#   -   pims: Image handler for trackpy                                       #
#   -   trackpy: Soft-matter particle tracker                                 #
#   -   cv2: OpenCV module                                                    #
#   -   sys: Only really used for the sys.exit() to terminate the code        #
#   -   argparse: Argument parser, allows me to use required & optional       #
#                 inputs with various flags                                   #
#                                                                             #
###############################################################################

import argparse
import cv2 as cv
import h5py
import numpy as np
import os
import random as rand
import sys
import trackpy as tp

# Create an argument parser object

ap = argparse.ArgumentParser()

# Add arguments needed for the code
# -f filename.ext -> Video file to be analyzed

ap.add_argument(
        "-f", "--file",
        required = True,
        help = "Experiment file ID"
        )

args = vars(ap.parse_args())

codepath = os.path.dirname(os.path.realpath(__file__))
os.chdir(codepath)

montpath = '../../data/montages/{}'.format(args['file'])

try:
    os.mkdir(montpath)
except OSError:
    print('Failed to create new directory')

vidpath = '../../data/videos/'
trajpath = '../../data/sleap/'

video = cv.VideoCapture('{}{}.mp4'.format(vidpath,args['file']))
trajfile = h5py.File('{}{}.hdf5'.format(trajpath,args['file']),'r')
trajectories = {}

for key in trajfile:
    trajectories[key] = trajfile[key]

offset = 1
length = 500
ct = 0
while ct < offset:
    success, frame = video.read()
    ct += 1
radius = 3
thickness = 2
color = {}

plotstack = []
ct = 0
try:
    os.makedirs('{}{}{}'.format(self.outpath,
                                self.fileid, 
                                self.bincount))
except:
    print('Directory already exists.\n')


h = frame.shape[0]
w = frame.shape[1]

fps = 25.0
fourcc = cv.VideoWriter_fourcc(*'mp4v')
api = cv.CAP_ANY
out = cv.VideoWriter('./file.mp4',
                    apiPreference = api,
                    fourcc = fourcc,
                    fps = float(fps),
                    frameSize = (w, h),
                    isColor = True)
for keys in trajectories:
    color[keys] = (rand.randint(0,255), # B
                   rand.randint(0,255), # G
                   rand.randint(0,255)) # R

while success:
    for keys in trajectories:
        try:
            traj = trajectories[keys]
            coords = tuple(traj[:,0,ct])
            frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                              radius, color[keys], thickness)
            coords = tuple(traj[:,1,ct])
            frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                              radius, color[keys], thickness)
            coords = tuple(traj[:,2,ct])
            frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                              radius, color[keys], thickness)
        except IndexError:
            print('foo')
            pass
        except ValueError:
            print('Missing body segment')
            pass

    out.write(frame) 
    success, frame = video.read()
    ct += 1
    if ct > length:
        success = False

out.release()
sys.exit(0)

#EOF

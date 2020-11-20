###############################################################################
#   Ant labeled montage generator                                             #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 2.0.0                                                            #
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

montpath = '../data/montages/{}'.format(args['file'])
try:
    os.mkdir(montpath)
except OSError:
    print('Failed to create new directory')

vidpath = '../data/videos/'
trajpath = '../data/trajectories/'

video = cv.VideoCapture('{}{}.mp4'.format(vidpath,args['file']))
trajfile = h5py.File('{}{}data.hdf5'.format(trajpath,args['file']),'r')
trajectory = trajfile['antdata'][1:,:2]

video.read()
video.read()
success, _ = video.read()
ct = 0
radius = 3
thickness = 2
color = [0,255,0]

while success:
    coords = tuple(trajectory[ct])
    success, frame = video.read()
    frame = cv.circle(
            frame, (coords[1],coords[0]),
            radius, color,
            thickness )
    cv.imwrite('{}/{}f{}.png'.format(
            montpath,args['file'],
            str(ct).zfill(5)
            ),
        frame
        )
    ct += 1
sys.exit(0)

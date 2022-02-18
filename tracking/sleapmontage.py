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

# Create an argument parser object

ap = argparse.ArgumentParser()

# Add arguments needed for the code
# -f filename.ext -> Video file to be analyzed

ap.add_argument('-f', '--file', required = True,
                help = 'Trajectory file name')
ap.add_argument('-v', '--video', help = 'Video ID')

args = vars(ap.parse_args())

if len(args['video']) == 0:
    vidname = args['file']
else:
    vidname = args['video']

codepath = os.path.dirname(os.path.realpath(__file__))
os.chdir(codepath)

montpath = '../../data/montages/{}'.format(args['file'])

try:
    os.mkdir(montpath)
except OSError:
    print('Failed to create new directory')

vidpath = '../../data/videos/'
trajpath = '../../data/trajectories/'

video = cv.VideoCapture('{}{}.mp4'.format(vidpath,vidname))
trajfile = h5py.File('{}{}.hdf5'.format(trajpath,args['file']),'r')
trajectories = {}

for key in trajfile:
    trajectories[key] = trajfile[key]

offset = 1
length = 27030
ct = 0
while ct < offset:
    success, frame = video.read()
    ct += 1
radius = 3
thickness = 2
color = {}

plotstack = []

try:
    os.makedirs('{}{}{}'.format(self.outpath,
                                self.fileid, 
                                self.bincount))
except:
    print('Directory already exists.\n')


h = frame.shape[0]
w = frame.shape[1]

fps = 15.0
fourcc = cv.VideoWriter_fourcc(*'mp4v')
api = cv.CAP_ANY
out = cv.VideoWriter('{}/{}.mp4'.format(montpath,vidname),
                    apiPreference = api,
                    fourcc = fourcc,
                    fps = float(fps),
                    frameSize = (w, h),
                    isColor = True)
for key in trajectories:
    color[key] = (rand.randint(0,255), # B
                   rand.randint(0,255), # G
                   rand.randint(0,255)) # R

ct = 0
while success:
    for key in trajectories:
        try:
            traj = trajectories[key]
            if ct == traj[ct,0,0]:
                coords = traj[ct,1]
                frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                                  radius, color[key], thickness)
                coords = traj[ct,2]
                frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                                  radius, color[key], thickness)
                coords = traj[ct,3]
                frame = cv.circle(frame, (int(coords[0]),int(coords[1])), 
                                  radius, color[key], thickness)
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

###############################################################################
#                                                                             #
#   Ant imaging experiment background isolation code for python3              #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 2.1.1                                                            #
#   First Written on 2020/02/13                                               #
#   Last modified: 2021/01/15                                                 #
#                                                                             #
#   Description:                                                              #
#     Alternative (and somewhat improved) version of the old code. Instead    #
#     of performing N x w x h different instances of pairwise max() function  #
#     calls, it performs w x h different function calls of max(arr) for       #
#     arrays of size N. As such, it requires the full image stack loaded to   #
#     memory, which will be a problem for continuous stream. Also uses a lot  #
#     of memory.                                                              #
#                                                                             #
#   Packages used:                                                            #
#   -   argparse: Input argument parsing                                      #
#   -   cv2: Used for reading video and saving output background.             #
#   -   numpy: Used for all of the array handling                             #
#   -   os: Needed for filepath search                                        #
#   -   sys: Only needed for exit code                                        #
#   -   time: Used to check code runtime                                      #
#                                                                             #
###############################################################################

import argparse
import cv2 as cv
import numpy as np
import os
import sys
import time as tt

start = tt.time()

ap = argparse.ArgumentParser()
#  Import and load file

ap.add_argument('-f', '--file', required = True, help = 'Video file name')
# How 'deep' into the video stack we want to go for finding the background.
# For less than 10 ants, between 50~500 frames seems to be sufficient; for
# Large experiment, using the full video stack is recommended (use -1)
ap.add_argument('-d', '--depth', required = True, 
                help = 'Temporal depth from frame 0')

args = vars(ap.parse_args())

codepath = os.path.dirname(os.path.realpath(__file__))
os.chdir(codepath)

#  Set file path and import the video
filepath = '../../data/videos/'
filename = args['file']
video = cv.VideoCapture('{}{}.mp4'.format(filepath, filename))

depth = int(args['depth'])

forcebreak = True
if depth == -1:
    forcebreak = False

arr = [] #np.empty(0)
success, frame = video.read()
x,y,_ = np.shape(frame)

ct = 0

while success and ((not forcebreak) or (ct < depth)):
    ct += 1
    skip = 10
    for i in range(skip):
        video.read()
    success, frame = video.read()
    arr.append(frame) #(np.append(arr,frame))

arr = np.array(arr, dtype = np.uint8)
arr = arr.reshape(ct, x, y, 3)

setupdone = tt.time()
print('Setup done; image shape: {}'.format((x, y)))
print('Setup time: {}\n'.format(setupdone - start))

# Process image
bgnd = np.empty((x, y, 3),dtype = np.uint8)

for i in range(x):
    for j in range(y):
        bgnd[i, j] = [max(arr[:, i, j, 0]),
                      max(arr[:, i, j, 1]),
                      max(arr[:, i, j, 2])]

cv.imwrite('background.png', bgnd)

computedone = tt.time()
print('Setup time: {}\nTotal time: {}'.format(setupdone - start,
                                              computedone - start))

sys.exit(0)

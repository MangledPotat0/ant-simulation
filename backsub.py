################################################################################
#                                                                              #
#   Ant imaging experiment background subtraction code for python3             #
#   Code written by Dawith Lim                                                 #
#                                                                              #   
#   Version: 1.1.0.0.0.0                                                       #
#   First Written on 2020/02/13                                                #
#   Last Modified: 2020/06/23                                                  #
#                                                                              #
#   Description:                                                               #
#     Since video from experiment has white background with dark features      #
#     of interest, the code simply goes through successive frames and          #
#     obtains the maximum brightness of each pixel among the frames, which     #
#     is how the frame looks like when part of the feature (i.e. ant) is       #
#     not present in that pixel.                                               #
#                                                                              #
#   Packages used                                                              #
#   -   argparse: Input argument parsing                                       #
#   -   array: Array handling                                                  #
#   -   cv2: OpenCV version 2; Used for image manipulation and property        #
#            extraction in the code                                            #
#   -   itertools: To make iteration through every array elements easier       #
#   -   os: Directory organization                                             #
#   -   numpy: Array handling                                                  #
#   -   pims: Image handling library                                           #
#   -   sys: System tools library, used here for exit codes                    #
#                                                                              #
################################################################################

import argparse
from array import array as arr
import cv2 as cv
import itertools
import numpy as np
import os
import pims
import sys
ap = argparse.ArgumentParser()
nthreads = 4

ap.add_argument('-f', '--file', required=True, help='Video file')
ap.add_argument('-d', '--depth', required=True, help='Backsub depth')

args = vars(ap.parse_args())

codepath = os.path.dirname(os.path.realpath(__file__))
os.chdir(codepath)

filepath = ('../data/videos/')

filename = args['file']
depth = int(args['depth'])

forcebreak = True
if depth == -1:
    forcebreak = False
thsh = depth

print('{}{}.mp4'.format(filepath,filename))
video = cv.VideoCapture('{}{}.mp4'.format(filepath,filename))
success, capture = video.read()
#if not capture.isOpened:
#    print('Error')
#    exit(0)
#print(len(capture))

if not success:
    print("Fileread failed.")
    print('File {} does not exist'.format(filename))
    sys.exit(0)
print("fileread successful.")

dim1, dim2, dim3 = np.shape(capture)

print('{},{},{}'.format(dim1,dim2,dim3))

count=0
backgnd = capture

while success:
    count += 1
    if forcebreak and count > thsh:
        print("Forced break at frame = "+str(thsh))
        break
    for i,j,k in itertools.product(range(dim1),range(dim2),range(dim3)):
        backgnd[i,j,k] = max(backgnd[i,j,k],capture[i,j,k])
    print('Frame {} processed.'.format(count))
    success, capture = video.read()
cv.imwrite('background.tiff',backgnd)
print("Process completed successfully. Exiting.")
sys.exit(0)

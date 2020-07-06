###############################################################################
#   Ant tracking code for python 3                                            #
#   Code written by Dawith Lim                                                #
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

import numpy as np
import pims
import trackpy as tp
import cv2 as cv
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

# Create an argument parser object

ap = argparse.ArgumentParser()

# Add arguments needed for the code
# -f filename.ext -> Video file to be analyzed

ap.add_argument("-f", "--file", required=True,
                help="Video file name without the extension")

args = vars(ap.parse_args())

# Save a montage as a video file
stack = pims.open(args["file"]+'/'+'*.tiff')

cv.imwrite('stack.tiff',stack[0])

print(stack[1].shape)
height, width, layers = stack[0].shape
montage = cv.VideoWriter(args["file"]+'montage.avi',
                         cv.VideoWriter_fourcc(*"XVID"),15,(width,height))

for i in range(0,len(stack)):
    montage.write(cv.cvtColor(stack[i].astype('uint8'), cv.COLOR_BGRA2BGR))

montage.release()


###############################################################################
#                                                                             #
#   Ant imaging experiment video cropper for python 3.7.4                     #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.3.1                                                            #
#   First written on: 2020/12/07                                              #
#   Last modified: 2021/12/24                                                 #
#                                                                             #
#   Description:                                                              #
#     Crop ant tracking video using manual adjustments                        #
#                                                                             #
#   Packages used:                                                            #
#                                                                             #
###############################################################################

import argparse
import cv2 as cv
import imutils
import numpy as np
import os
import sys

# Load video
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, required=True,
                help='Input video file without file extension')

args = vars(ap.parse_args())
filepath = os.path.dirname(os.path.realpath(__file__))
vidpath = '../../data/videos/'

try:
    vidstream = cv.VideoCapture('{}{}.mp4'.format(vidpath,args['video']))
except FileNotFoundError:
    print('Cannot find the video file. Check filename.')
    sys.exit(0)

success, frame = vidstream.read()

assert success, 'Invalid imageread; process exiting.'

cropped = frame
run = True
cropleft = True;
cropright = True;
croptop = True;
cropbottom = True;

ll, rr, tt, bb = (0,0,0,0)

while run:

# For each direction (left, right, top, bottom), take user input and crop
# image. positive value entry means cropping the image by that many pixels,
# negative value entry means reseting the cropping process and starting over,
# and zero value entry means proceed to the next step.

    while cropleft:
        currentshape = np.shape(cropped)
        cv.imshow('cropped',cropped)
        cv.waitKey()
        xcrops = int(input('Enter the left-crop value: '))
        assert type(xcrops) == int, 'Crop amount must be numerical'
        if xcrops < 0:
            cropped = frame;
        elif xcrops == 0:
            cropleft = False;
        else:
            ll += xcrops
            cropped = cropped[:,xcrops:currentshape[1]]

    while cropright:
        currentshape = np.shape(cropped)
        cv.imshow('cropped', cropped)
        cv.waitKey()
        xcrops = int(input('Enter the right-crop value: '))
        assert type(xcrops) == int, 'Crop amount must be numerical'
        if xcrops < 0:
            cropped = frame;
        elif xcrops == 0:
            cropright = False;
        else:
            rr += xcrops
            cropped = cropped[:,0:currentshape[1] - xcrops]
    
    while croptop:
        currentshape = np.shape(cropped)
        cv.imshow('cropped',cropped)
        cv.waitKey()
        ycrops = int(input('Enter the top-crop value: '))
        assert type(xcrops) == int, 'Crop amount must be numerical'
        if ycrops < 0:
            cropped = frame;
        elif ycrops == 0:
            croptop = False;
        else:
            tt += ycrops
            cropped = cropped[ycrops:currentshape[0],:]

    while cropbottom:
        currentshape = np.shape(cropped)
        cv.imshow('cropped',cropped)
        cv.waitKey()
        ycrops = int(input('Enter the bottom-crop value: '))
        assert type(xcrops) == int, 'Crop amount must be numerical'
        if ycrops < 0:
            cropped = frame;
        elif ycrops == 0:
            cropbottom = False;
            run = False;
        else:
            bb += ycrops
            cropped = cropped[0:currentshape[0] - ycrops,:]

cropped = frame[ll:-rr, tt:-bb]
height, width, _ = np.shape(cropped)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
api = cv.CAP_ANY
out = cv.VideoWriter(filename = '{}{}cropped.mp4'.format(vidpath,args['video']),
                     apiPreference = api,
                     fourcc = fourcc,
                     fps = 10.0,
                     frameSize = (height, width),
                     isColor = True,
                     )

while success:
    cropped = frame[tt:-bb, ll:-rr]
    out.write(cropped)
    success, frame = vidstream.read()

vidstream.release()
out.release()
cv.destroyAllWindows()
sys.exit(0)

#EOF

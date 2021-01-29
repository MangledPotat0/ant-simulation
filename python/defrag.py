################################################################################
#                                                                              #
#   Trajectory defragmentation code                                            #
#                                                                              #
#     Takes endpoints of each trajectory and attempts to stitch them into      #
#     larger trajectories. The unstitchable leftovers will be considered as    #
#     artifacts and be removed by sifter                                       #
#                                                                              #
#   Code written by: Dawith Lim                                                #
#   Version: 0.8.0                                                             #
#   File created: 2021/01/29                                                   #
#   Last modified: 2021/01/29                                                  #
#                                                                              #
#   Packages used:                                                             #
#                                                                              #
################################################################################

import argparse
import h5py
import numpy as np
import os
import sys


def setpath(fname):
    currentdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(currentdir)
    filepath = '../../data/trajectories/'

    return fiepath


def loaddata(filepath, fname):
    
    ff = h5py.File('{}{}.h5py'.format(filepath, fname), 'a')
    data = ff['trajectories']

    return data


def fetch_breaks(data):
# Creates a list that stores the beginning and endpoint of a trajectory
    breaks = []
    for dset in data:
        start = dset[0,:]
        end = dset[-1,:]
        breaks.append(start)
        if start == end:
            pass
        elif start =! end:
            breaks.append(end)

    return breaks


def defragment(data):

    breaks = fetch_breaks(data)
    print(breaks)

    return


def sift(data):

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', help = 'Data file name without extension')
    args = vars(ap.parse_args())

    fname = args['file']
    filepath = setpath(fname)
    data = loaddata(filepath, fname)

    defragment(data)
    sift(data)

    sys.exit(0)

# EOF

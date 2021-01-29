################################################################################
#                                                                              #
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


def defragment():

    return


def sift():

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', help = 'Data file name without extension')
    args = vars(ap.parse_args())

    fname = args['file']
    filepath = setpath(fname)
    data = loaddata(filepath, fname)

    defragment()
    sift()

    sys.exit(0)

# EOF

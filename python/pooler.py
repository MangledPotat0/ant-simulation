################################################################################
#                                                                              #
#   Ant trajectory pooling script for python3                                  #
#   Code written by Dawith Lim                                                 #
#                                                                              #
#   Version 1.0.3                                                              #
#   First written on 2020/05/29                                                #
#   Last Modified: 2021/01/15                                                  #
#                                                                              #
#   Packages used                                                              #
#   - argparse: Argument parser to handle input parameter                      #
#   - h5py: Python library to handle hierarchical data format                  #
#   - numpy: Array manipulation                                                #
#   - time: To generate timestamp                                              #
#                                                                              #
################################################################################


import argparse as app
import h5py
import numpy as np
import os
import sys
import time

ap = app.ArgumentParser()
ap.add_argument('-f', '--files', help='Files', nargs='+')
args = ap.parse_args()

codepath = os.path.dirname(os.path.realpath(__file__))
os.chdir(codepath)
filepath = "../../data/trajectories/"

timestamp = time.strftime('%Y%m%d%H')

try:
    pooledfile = h5py.File('{}{}pooldata.hdf5'.format(filepath,timestamp), 'w-')
except OSError:
    print("File {}.hdf5 already exists.".format(timestamp))
    sys.exit(0)

for filename in args.files:
    if filename is not None:
        try:
            datafile = h5py.File('{}{}data.hdf5'.format(filepath,filename), 'r')
            ct = 0
            for ant in datafile:
                antdata = datafile[ant]
                pooledfile.create_dataset('{}_ant{}'.format(filename,ct),
                        data=antdata)
                ct += 1
                print('Data file {} processed'.format(filename))

        except OSError:
            print("Data file '{}data.hdf5' cannot be found.".format(filename))

pooledfile.flush()
pooledfile.close()
print('Process exited successfully.')

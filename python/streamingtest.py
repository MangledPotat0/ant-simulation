###############################################################################
#                                                                             #
#   Ant project trajectory stream-linking test code                           #
#                                                                             #
#   This is just to test the file streaming functionality for trackpy         #
#   library to sidestep the subnet mask too large problem when processing     #
#   the code directly. When testing is complete, this code is to be merged    #
#   into the tracker.py code.                                                 #
#                                                                             #
###############################################################################

import argparse
import h5py
import os
import numpy as np
import pandas as pd


# Code from the tutorial

filepath = ''
expid = ''
filename = '{}{}.hdf5'.format(filepath, expid))

with tp.PandasHDFStore(filename) as ff:
    for linked in tp.link_df_iter(ff, 3, neighbor_strategy = 'KDTree'):
        ff.put(linked)


# Custom interface to match the desired structure

def put(self, data):
    if len(data) == 0:
        warnings.warn('Empty data passed')
    frame = data[8]



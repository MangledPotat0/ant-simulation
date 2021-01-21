###############################################################################
#                                                                             #
#   Ant project trajectory stream-linking code for python 3.7.4               #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.1.0                                                            #
#   First written on: 2021/01/11                                              #
#   Last modified: 2021/01/21                                                 #
#                                                                             #
#   Description:                                                              # 
#     This script performs file streaming-based trajectory linking for the    #
#     ant trajectory data. Its primary purpose is to avoid memory being       #
#     overloaded during linking, and it's also separate from tracker.py to    #
#     make sure an issue on the linking side doesn't necessitate a whole re-  #
#     run for the trajectory detection process.                               #
#                                                                             #
#   Packages used:                                                            #
#   - 
###############################################################################

import argparse
import h5py
import os
import numpy as np
import pandas as pd
import trackpy as tp


# Custom interface to match the desired structure
class newstore():

# to do- add argparse to take input argument for these properties
    def __init__(self, mode = 'a', **kwargs):
        filepath = ''#'../../data/trajectories/'
        self.expid = 'test'
        filename = '{}{}.hdf5'.format(filepath, self.expid)
        
        self.t_column = 8
        self.filename = os.path.abspath(filename)
        self.store = h5py.File(self.filename, mode)


    def __enter__(self):
        return self


# Getter functions
    @property
    def t_column(self):
        return self.t_column
    

    @property
    def expid(self):
        return self.expid


    @property
    def filename(self):
        return filename


    def put(self, df):
        if len(df) == 0:
            warnings.warn('Empty DataFrame passed to put().')
            return

        data = self.reformat()

        for antid in df[1]:
            if not str(antid) in self.store.keys():
                dset = self.store.create_dataset(
                                    str(antid),
                                    (0, 4),
                                    dtype = np.float,
                                    maxshape = (None, 4),
                                    chunks = (1, 4))
            else:
                dset = self.store[str(antid)]
            ind = df[1].index(antid)
            entry = np.empty(4)
            entry[:2] = data[df[0]][ind]
            entry[2] = df[0]
            entry[3] = ind
            dset.resize((dset.shape[0] + 1, 4))
            dset[-1,:] = entry[:]
            self.store.flush()


    def reformat(self):
        data = obje.store
        coords = self.store['dump/block0_values'][:,0:2]
        frame = self.store['dump/block1_values'][:,0]
        dump = []
        for n in range(max(frame)):
            dump.append([])
            for m in range(len(frame)):
                if frame[m] == n:
                    dump[n].append(coords[m])
            dump[n] = np.array(dump[n])
        return dump

# Code from the tutorial

obje = newstore()
#with obje.store['dump/block0_values'] as dump:

dump = obje.reformat()
    
for linked in tp.link_iter(
                dump, # Iterable data
                3, # Search distance in float, optionally as tuple of floats
                memory = 2, # Search depth in frames
                predictor = None, # Prediction model function
                adaptive_stop = None, # Float, minimum search range acceptable
                                      # To use when subnet is too large.
                adaptive_step = None, # Step size for reducing search range
                                      # when subnet is too big
                neighbor_strategy = 'KDTree',
                    # KDTree:
                    # BTree:
                link_strategy = 'auto',
                    # recursive
                    # nonrecursive
                    # hybrid
                    # numba
                    # drop
                    # auto
                dist_func = None,
                    # Used only for BTree
                to_eucl = None
                    # Mapping function to transform position array to a
                    # Euclidean space
                           ):
    print(linked)
    obje.put(linked)


print('Process completed successfully. Exiting')
sys.exit(0)

# EOF

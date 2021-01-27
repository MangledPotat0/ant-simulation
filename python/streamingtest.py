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
#   -                                                                         # 
###############################################################################

import argparse
import h5py
import os
import numpy as np
import pandas as pd
import time as tt
import trackpy as tp


# Custom interface to match the desired structure
class newstore():

# to do- add argparse to take input argument for these properties
    def __init__(self, mode = 'a', **kwargs):
        filepath = ''#'../../data/trajectories/'
        self.expid_ = 'test'
        filename = '{}{}.hdf5'.format(filepath, self.expid)
        
        self.t_column_ = 8
        self.filename_ = os.path.abspath(filename)
        self.store = h5py.File(self.filename, mode)


    def __enter__(self):
        return self


# Getter functions
    @property
    def t_column(self):
        return self.t_column_
    

    @property
    def expid(self):
        return self.expid_


    @property
    def filename(self):
        return self.filename_


    def put(self, indices, block):
        if len(indices) == 0:
            warnings.warn('Empty Indices table passed to put().')
            return

        #data = self.reformat()

        for antid in indices[1]:
            if not str(antid) in self.store.keys():
                dset = self.store.create_dataset(
                                    str(antid),
                                    (0, 4),
                                    dtype = np.float,
                                    maxshape = (None, 4),
                                    chunks = (1, 4))
            else:
                dset = self.store[str(antid)]
            ind = indices[1].index(antid)
            entry = np.empty(4)
            entry[:2] = block[indices[0]][ind]
            entry[2] = indices[0]
            entry[3] = ind
            dset.resize((dset.shape[0] + 1, 4))
            dset[-1,:] = entry[:]
            self.store.flush()


    def reformat(self, start=0, end=-1):
        coords = self.store['dump/block0_values'][start:end,0:2]
        frames = self.store['dump/block1_values'][start:end]
        frame = frames[:,0]
        dump = []
        for n in range(max(frame)):
            dump.append(np.array(coords[frames[:,0] == n],
                                 dtype = np.float32))
        return dump


@tp.predict.predictor
def predict(t1, particle):
    velocity = None
    return tp.predict.NearestVelocityPredict()


if __name__ == "__main__":
    obje = newstore()
    chunksize = 18000
    ct = 0
    dump = obje.reformat()
    indexable = True
    loopstart = tt.time()

    while ct < 1:
        #start = ct * chunksize
        try:
            pass
            #end = (ct + 1) * chunksize
            #block = obje.reformat(start, end)
        except IndexError:
            print('bee')
            #print('Index Error; reached end of file ({}/{})'.format(end,
            #                                                        len(dump)))
            #end = -1
            #block = obje.reformat(start, end)
            #indexable - False
        finally:
            for linked in tp.link_iter(
                # Iterable data
                dump,
                # Search distance in float, optionally as tuple of floats
                3,
                # Search depth in frames
                memory = 2,
                # Prediction model function
                #predictor = predict(1, block),
                # Float; minimum search range acceptable to use when subnet
                # mask is too large
                adaptive_stop = None, 
                # Step size for reducing serach range when subnet is too big
                adaptive_step = None,
                # Nearest neighbor finding strategy
                neighbor_strategy = 'KDTree',
                    # KDTree:
                    # BTree:
                # Trajectory linking strategy
                link_strategy = 'numba',
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
                linked = (linked[0] + ct * chunksize, linked[1])
                #print(linked)
                obje.put(linked, dump)
            ct += 1

    print('Process completed successfully. Exiting')
    sys.exit(0)

# EOF

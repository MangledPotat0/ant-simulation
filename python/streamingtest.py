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
import trackpy as tp


# Custom interface to match the desired structure
class newstore():

    def __init__(self, mode='a', **kwargs):
        filepath = ''#'../../data/trajectories/'
        expid = 'test'
        filename = '{}{}.hdf5'.format(filepath, expid)

        self.filename = os.path.abspath(filename)
        self.store = h5py.File(self.filename, mode)


    def __enter__(self):
        return self


    @property
    def t_column(self):
        return 8#self.t_column
    

    #@property
    #def max_Frame(self):
    #    return max(self.frames)


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
    
for linked in tp.link_iter(dump, 3, neighbor_strategy = 'KDTree'):
    print(linked)
    obje.put(linked)
print('meow')

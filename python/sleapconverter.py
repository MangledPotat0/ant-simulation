################################################################################
#                                                                              #
#  SLEAP output conversion code for python 3.7.4                               #
#                                                                              #
#  The code takes the .h5 output from SLEAP run, and save it into a new file   #
#  in .hdf5 file with a slightly altered structure.                            #
#                                                                              #
################################################################################

import argparse as aps
import h5py as hp
import numpy as np
import os
import sys


class converter():

    def __init__(self):
        try:
            super.__init__()
            self.params = super.params
        except TypeError:
            print('Called without a parent class;',
                  'proceeding with default params.')
            ap = aps.ArgumentParser()
            ap.add_argument('-f', '--filename', required = True, help = 'File name')
            args = vars(ap.parse_args())

            codepath = os.path.dirname(os.path.realpath(__file__))
            os.chdir(codepath)
            
            filepath = '../../data/sleap/'

            self.params = {'file' : args['filename'],
                           'path' : filepath}

        return


    def run(self):
        fullpath = '{}{}'.format(self.params['path'], self.params['file'])
        ct = 0
        with hp.File('{}.h5'.format(fullpath), 'r') as src:
            with hp.File('{}.hdf5'.format(fullpath), 'w') as out:
                srctable = src['tracks']
                for traj in srctable:
                    if traj[0,0,0] > 0:
                        out.create_dataset('trajectory_{}'.format(ct), 
                                           data = traj)
                        print('Added trajectory number {}'.format(ct))
                        out.flush()
                        ct += 1
                    else:
                        print('Skipped a NaN trajectory')
        return


if __name__ == '__main__':
    conv = converter()

    conv.run()

    sys.exit(0)

# EOF

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
            ap.add_argument('-v', '--videoname', help = 'Video name')

            args = vars(ap.parse_args())

            if len(args['videoname']) == 0:
                videoname = args['filename']
            else:
                videoname = args['videoname']

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
                    frameID = np.array([], dtype=int)
                    for frame in range(len(traj[0,0,:])):
                        if np.linalg.norm(traj[:,:,frame] > 0):
                            frameID = np.append(frameID, int(frame))
                        else:
                            print('Skipped NaN entry')
                    coords = traj[:,:,frameID]
                    print('Added trajectory number {}'.format(ct))
                    out.create_dataset('trajectory_{}'.format(ct), 
                                       data = coords)
                    out.create_dataset('frames_{}'.format(ct), 
                                       data = frameID)
                    out.flush()
                    ct += 1
        return


if __name__ == '__main__':
    conv = converter()

    conv.run()

    sys.exit(0)

# EOF

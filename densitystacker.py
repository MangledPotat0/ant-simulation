################################################################################
#                                                                              #
#   Ant density data stacker for single trajectories                           #
#                                                                              #
#   Created: 2020/08/17                                                        #
#   Last modified: 2020/08/17                                                  #
#                                                                              #
################################################################################

from matplotlib import pyplot as plt
import argparse as ap
import h5py
import numpy as np
import os

arg = ap.ArgumentParser()
arg.add_argument('-f', '--file', nargs='+', 
        help='.hdf5 data files without file extension')
arg.add_argument('-n', '--number', help='Total number of ants')
arg.add_argument('-s', '--size', help='Binsize')
args = vars(arg.parse_args())

# filepath = '../data/density/'
size = args['size']
intsize = int(1100/int(size))
filenames = args['file']
pile = np.empty((intsize,intsize))
length=999999999999999999999999
for name in filenames:
    filetemp = h5py.File('{}.hdf5'.format(name),'r')
    datatemp = filetemp['{}x{}'.format(size,size)]
    if length > len(datatemp):
        length = len(datatemp)
    else:
        length=length
    print(length)
    pile = pile[:length] + datatemp[:length]

outputfile = h5py.File('{}.hdf5'.format('output'),'w')
outputfile.create_dataset('{}x{}'.format(size,size), data=pile)
outputfile.flush()
outputfile.close()
ct=0
try:
    os.makedirs('../data/density/{}{}'.format('output',size))
except:
    print('filepath exists')

for frame in pile:
    maxval = np.max(0.001*int(args['number']))
    plt.figure(figsize=(5.5,5.5))
    plt.imshow(frame, cmap='Blues',interpolation='nearest',vmin=0,
            vmax=maxval)
    plt.colorbar()
    plt.savefig('../data/density/{}{}/{}{}{}.png'.format(
        'output',size,'output',size,ct),
        bbox_inches='tight')
    plt.close()
    ct += 1

print('Process exited successfully.')

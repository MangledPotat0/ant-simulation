################################################################################
#                                                                              #
#   Ant density data stacker for single trajectories                           #
#   Code written by: Dawith Lim                                                #
#                                                                              #
#   Version 1.2.2                                                              #
#   Created: 2020/08/17                                                        #
#   Last modified: 2021/03/11 (03/14 G)                                        #
#                                                                              #
#   Description:                                                               #
#     This code takes multiple optical density data from multiple experiments  #
#     and combine them into a single lattice.                                  #
#                                                                              #
################################################################################

from matplotlib import animation as ani
from matplotlib import pyplot as plt
import argparse as ap
import h5py
import numpy as np
import os
import time

arg = ap.ArgumentParser()
arg.add_argument('-f', '--file', nargs = '+', 
                 help = '.hdf5 data files without file extension')
arg.add_argument('-n', '--number', help = 'Total number of ants')
arg.add_argument('-b', '--bincount', help = 'Number of bins')
args = vars(arg.parse_args())

filepath = '../../data/density/'
antcount = int(args['number'])
bincount = int(args['bincount'])
filenames = args['file']
outlen = 9000
pile = np.zeros((outlen, bincount, bincount))
out = time.strftime('%Y%m%d%H')

for name in filenames:
    filetemp = h5py.File('{}{}.hdf5'.format(filepath, name), 'r')
    datatemp = filetemp['{}x{}'.format(bincount, bincount)]
    length = len(datatemp)
    skips = 1
    if round(length / 9000) == 2:
        skips = 2
        length = 18000
    elif round(length / 9000) == 3:
        skips = 3
        length = 27000
    #if length == -1:
    #    length = len(datatemp)
    #    pile = np.empty((length, bincount, bincount))
    #else:
    #    pile += datatemp
    #if length > len(datatemp):
    #    length = len(datatemp)
    #else:
    #    length = length
    #print(length)
    try:
        pilea = pile[:length]
        pileb = datatemp[:length:skips]
        pile = pilea + pileb
    except IndexError:
        print('Something happened here')
    except ValueError:
        print(np.shape(pilea), np.shape(pileb))

outputfile = h5py.File('{}{}.hdf5'.format(filepath, out), 'w')
outputfile.create_dataset('{}x{}'.format(bincount, bincount), data = pile)
outputfile.flush()
outputfile.close()
try:
    os.makedirs('../../data/density/{}{}'.format('output', size))
except:
    print('filepath exists')


plotstack = []

try:
    os.makedirs('{}{}{}'.format(filepath,
                                out, 
                                bincount))
except:
    print('Directory already exists.\n')

fig = plt.figure(figsize = (5.5, 5.5))

ims = []

for frame in pile:
    normed = antcount * frame / np.sum(frame)
    ims.append((plt.pcolor(normed,
                           norm = plt.Normalize(0, antcount),
                           cmap = 'Blues'), ))
        

anim = ani.ArtistAnimation(fig, ims)
anim.save('../../data/density/{}{}/{}{}.mp4'.format(
                                out, bincount,
                                out, bincount),
          fps = 10)
plt.close(fig)

print('Process exited successfully.')

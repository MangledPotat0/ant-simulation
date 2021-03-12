###############################################################################
#                                                                             #
#   Density lattice collapser for python 3.7.4                                #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   File created: 2021/03/15 (03/18 G)                                        #
#   Last modified: 2021/03/15 (03/18 G)                                       #
#                                                                             #
#   Description:                                                              #
#    Collapses a 2D lattice into a key-value pair, where key is the linear    #
#    distance from the nearest boundary and value is the density at that      #
#    distance.                                                                #
#                                                                             #
###############################################################################


import argparse as arg
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


class lattice_collapser():

    def __init__(self, dfilename, antcount):
        dfile = h5py.File(dfilename, 'r')
        key = list(dfile.keys())
        self.stack_ = dfile[key[0]]
        self.antcount_ = antcount

        return
    
    
    def stack(self):
        return self.stack_


    def antcount(self):
        return self.antcount_


    def collapse(self, lattice):
        size = len(lattice)
        center = np.array([size + 1, size + 1]) / 2
        output = {}
        for i in range(size):
            for j in range(size):
                xdistance = abs(i - center[0])
                ydistance = abs(j - center[1])
                distance = math.sqrt(xdistance ** 2 + ydistance ** 2)
                try:
                    output[distance] += lattice[i, j]
                except KeyError:
                    output[distance] = lattice[i, j]

        print(output)
        return output

    
    def tintegrate(self, stack):
        shape = np.shape(stack[0])
        lattice = np.zeros(shape)
        for frame in stack:
            lattice = lattice + frame
        
        print(lattice)
        return lattice / shape[0]


    def run(self):
        stack = self.stack()[:,0,...]
        lattice = self.tintegrate(stack)
        kvpair = self.collapse(lattice)
        density = list(kvpair.items())
        density = np.array(density, dtype = float)
        density = antcount * density / np.sum(density)

        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(density[0], density[1])

        fig.savefig('test.png')

        return


if __name__ == '__main__':

    ap = arg.ArgumentParser()
    ap.add_argument('-f', '--file', help = 'input file name')
    ap.add_argument('-n', '--antcount', help = 'number of ants', dtype = int)
    args = vars(ap.parse_args())
    dfile = args['file']
    antcount = args['antcount']
    
    collapser = lattice_collapser(dfile, antcount) 

    collapser.run()

    sys.exit(0)

# EOF

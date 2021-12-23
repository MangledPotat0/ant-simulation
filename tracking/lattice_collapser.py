###############################################################################
#                                                                             #
#   Density lattice collapser for python 3.7.4                                #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   File created: 2021/03/15 (03/18 G)                                        #
#   Last modified: 2021/03/18 (03/18 G)                                       #
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
        center = np.array([size - 1, size - 1]) / 2
        output = {}
        duplicate = {}
        for i in range(size):
            for j in range(size):
                xdistance = abs(i - center[0])
                ydistance = abs(j - center[1])
                distance = math.sqrt(xdistance ** 2 + ydistance ** 2)
                try:
                    output[distance] += lattice[i, j]
                    duplicate[distance] += 1
                except KeyError:
                    output[distance] = lattice[i, j]
                    duplicate[distance] = 1
        
        for key in output.keys():
            output[key] = output[key] / duplicate[key]

        return output


    def duplicates(self, size):
        ct = math.ceil(size / 2)
        sequence = []
        if size % 2 == 0:
            while ct > 1:
                sequence.append(4)
                ct2 = ct - 1
                while ct2 > 0:
                    sequence.append(8)
                    ct2 -= 1
                ct -= 1
            sequence.append(4)
                
        elif size % 2 == 1:
            while ct > 1:
                sequence.append(4)
                ct2 = ct - 2
                while ct2 > 0:
                    sequence.append(8)
                    ct2 -= 1
                sequence.append(4)
                ct -= 1
            sequence.append(1)

        return sequence

    
    def tintegrate(self, stack):
        shape = np.shape(stack[0])
        lattice = np.zeros(shape)
        for frame in stack:
            lattice = lattice + frame
        
        return lattice / np.shape(stack)[0]


    def run(self):
        stack = self.stack()[:,0,...]
        lattice = self.tintegrate(stack)
        kvpair = self.collapse(lattice)
        kvpair = sorted((key, value) for (key, value) in kvpair.items())
        #density = list(kvpair)
        density = np.array(kvpair, dtype = float)
        density = np.transpose(density)

        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(density[0], density[1])

        ax.set_xlabel('Distance from center (dx)')
        ax.set_ylabel('Average ant count')

        fig.savefig('test.png')

        return


if __name__ == '__main__':

    ap = arg.ArgumentParser()
    ap.add_argument('-f', '--file', help = 'input file name')
    ap.add_argument('-n', '--antcount', help = 'number of ants', type = int)
    args = vars(ap.parse_args())
    dfile = args['file']
    antcount = args['antcount']
    
    collapser = lattice_collapser(dfile, antcount) 

    collapser.run()

    sys.exit(0)

# EOF

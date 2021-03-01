###############################################################################
#                                                                             #
#   Lattice ant model simulation for python3.7.4                              #
#   Code written by: Dawith Lim                                               #
#                                                                             #
#   Created: 2021/03/04  (2021/03/01 G)                                       #
#   Last modified: 2021/03/04  (2021/03/01 G)                                 #
#                                                                             #
#   Description:                                                              #
#     This code performs the simulation for age dynamics of a system of       #
#     ants based on simulated annealing. The model used here is a slightly    #
#     modified version of the model from doi: 10.1016/j.jtbi.2011.04.033      #
#     the only difference is that the original model from the paper uses a    #
#     toroidal (and thus infinite) lattice whereas this code uses a bounded   #
#     lattice (i.e. finite).                                                  #
#                                                                             #
###############################################################################


import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time


class lattice_ant_model():

    def __init__():

        with json.load(open('lattice_ant_model_params.json', 'r+')) as param:
            # These parameters are for setting up the lattice and the ants.
            self.latticesize_ = param['lattice_size']
            self.nspecies_ = param['n_species']
            self.mcount_ = params['m_count']
            self.interaction_ = params['interaction_table']
            self.lengthscale_ = params['length_scale']

            # These parameters are for the metropolis algorithm used for MCMC.
            self.threshold_ = params['threshold']
            self.energyscale_ = params['energyscale']
            self.temperature_ = params['temperature']

        # This block checks that the input parameters are consistent with
        # each other and with what is physically sensible.
        errmsg1 = ('The number of ant species ({}) and the number of' +
                  ' entries on the ant count list m_count ({}) do not' +
                  ' match. Check the params file.')
        errmsg2 = errmsg1.format(self.nspecies_, len(self.mcount_))

        errmsg2 = ('The number of ant species ({}) and the dimensions of' +
                  ' the interaction table ({}) do not match. Check the' + 
                  ' params file.')
        errmsg2 = errmsg2.format(self.nspecies_, len(self.interaction_))
        
        errmsg3 = 'Temperature is set to negative value! ({})'
        errmsg3 = errmsg3.format(self.temperature_)
        
        errmsg4 = 'Probability threshold is out of range ({})'
        errmsg4 = errmsg4.format(self.threshold_)

        assert len(self.mcount_) == self.nspecies_, errmsg1
        assert len(self.interaction_) == self.nspecies_, errmsg2
        assert len(self.interaction_[0]) == self.nspecies_, errmsg2
        assert self.temperature_ > 0, errmsg3
        assert (self.threshold_ > 0) & (self.threshold_ <= 1), errmsg4

        return
    
   
    # Getter methods
    def latticesize(self):
        return self.latticesize_

    def nspecies(self):
        return self.nspecies_

    def mcount(self):
        return self.mcount_

    def interaction(self):
        return self.interaction_

    def lengthscale(self):
        return self.lengthscale_

    def threshold(self):
        return self.threshold_

    def energyscale(self):
        return self.energyscale_

    def temperature(self):
        return self.temperature_

    
    def compute_damping(self):
        damp_length = self.latticesize ** 2
        damparray = np.zeros((damp_length, damp_length))
        for i in range(damp_length):
            for f in range(damp_length):
                xdistance = abs(f % self.latticesize - i % self.latticesize) 
                ydistance = abs(math.floor(f / self.latticesize)
                                - math.floor(i / self.latticesize))
                distance = math.sqrt(xdistance ** 2 + ydistance ** 2)
                damparray[i,f] = math.exp(-lengthscale * distance)

        return damparray


    def compute_energy(self):
        energy = np.sum(np.dot(self.interaction, self.damparray))
        return energy


    def compute_dE(self, start, end):
        dE = np.sum(np.dot((damping[:, end] - damping[x,start]), vecn[:,i])
                    + 2jj[l,l] * (1 - damping[end, start]))
        return dE

    def run(self, tt):
        ct = 0
        while ct < tt:
            ct += 1
            anttype = random.randint((1,n))
            flattened = np.flatten(self.lattice)
        return

if __name__ == '__main__':
    print('Process completed successfully. Exiting.')
    sys.exit(0)


# EOF

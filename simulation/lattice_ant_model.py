###############################################################################
#                                                                             #
#   Lattice ant model simulation for python3.7.4                              #
#   Code written by: Dawith Lim                                               #
#                                                                             #
#   Created: 2021/03/04  (2021/03/01 G)                                       #
#   Last modified: 2021/03/06  (2021/03/03 G)                                 #
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
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time


class lattice_ant_model:

    def __init__(self):

        param = json.load(open('lattice_ant_model_params.json', 'r+'))

        # These parameters are for configuring what simulation to set up
        self.boundarycondition_ = param['boundary_condition']
        self.selectionmethod_ = param['ant_selection_method']
        self.wallinteraction_ = param['enable_wall_effect']

        # These parameters are for setting up the lattice and the ants.
        self.tsteps_ = param['timesteps']
        self.latticesize_ = param['lattice_size']
        self.nspecies_ = param['n_species']
        self.mcount_ = np.array(param['m_count'])
        self.interaction_ = np.array(param['interaction_table'])
        self.lengthscale_ = param['length_scale']

        # These parameters are for the metropolis algorithm used for MCMC.
        self.threshold_ = param['threshold']
        self.energyscale_ = param['energy_scale']
        self.temperature_ = param['temperature']

        # This block checks that the input parameters are consistent with
        # each other and with what is physically sensible.
        errmsg0 = 'Runtime parameter invalid (tsteps = {} < 1'.format(
                    self.tsteps)
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

        assert self.tsteps_ > 1, errmsg0
        assert len(self.mcount_) == self.nspecies_, errmsg1
        assert len(self.interaction_) == self.nspecies_, errmsg2
        assert len(self.interaction_[0]) == self.nspecies_, errmsg2
        assert self.temperature_ > 0, errmsg3
        assert (self.threshold_ > 0) & (self.threshold_ <= 1), errmsg4

        # params to enable/disable optional features
        self.tt_ = param['do_time_trial']
        self.makemontage_ = param['make_montage']
        return
    
   
    # Getter methods
    def boundarycondition(self):
        return self.boundarycondition_

    def selectionmethod(self):
        return self.selectionmethod_

    def wallinteraction(self):
        return self.wallinteraction_

    def tsteps(self):
        return self.tsteps_

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

    def tt(self):
        return self.tt_

    def makemontage(self):
        return self.makemontage_


    # Methods to compute stuff
    def compute_damping(self):
        damp_length = self.latticesize() ** 2
        damparray = np.zeros((damp_length, damp_length))
        for i in range(damp_length):
            for f in range(damp_length):
                xdistance = abs((f % self.latticesize()) 
                                 - (i % self.latticesize())) 
                ydistance = abs(math.floor(f / self.latticesize())
                                - math.floor(i / self.latticesize()))
                distance = math.sqrt(xdistance ** 2 + ydistance ** 2)
                damparray[i,f] = math.exp(-self.lengthscale() * distance ** 2)

        return damparray

    
    def lattice_force(self, flattice):
        fllat = np.array([[flat] for flat in flattice])
        fllat_tp = np.transpose(fllat)
        temp = np.matmul(fllat_tp, self.interaction())
        latforce = np.matmul(temp, fllat)
        return latforce


    def compute_energy(self, flattice):
        latforce = self.lattice_force(flattice)
        energy = np.sum(np.dot(latforce, self.compute_damping()))
        return energy


    def compute_dE(self, start, end, flattice, anttype):
        lsize = self.latticesize()
        damping = self.compute_damping()
        origin = start[0] * self.latticesize() + start[1]
        destination = end[0] * self.latticesize() + end[1]
        arg1 = np.array([damping[:, destination] - damping[:, origin]])
        arg2 = np.dot(np.transpose(flattice), self.interaction()[anttype, :])
        dE = 2 * np.dot(arg1[0], arg2)
        dE += (2 * self.interaction()[anttype, anttype] 
                 * (1 - damping[destination, origin]))

        return dE


    def run(self):
        tt = self.tsteps()
        if self.makemontage():
            ims = []
            fig = plt.figure(figsize = (5.5, 5.5))
        ct = 1
        lattice = self.populate_lattice()
        flattice = np.array([lat.flatten() for lat in lattice])
        energy = np.zeros(tt)
        energy[0] = self.compute_energy(flattice)
        while ct < tt:
            assert int(np.sum(flattice)) == np.sum(self.mcount())
            anttype, origin, destination = self.generate_step(flattice)
            valid, dE = self.validate_step(origin, destination,
                                           flattice, anttype)

            if valid:
                lattice[anttype, origin[0], origin[1]] -= 1
                lattice[anttype, destination[0], destination[1]] += 1
                energy[ct] = energy[ct - 1] + dE
                flattice = np.array([lat.flatten() for lat in lattice])
            else:
                energy[ct] = energy[ct - 1]
            ct += 1
            if self.makemontage():
                ims.append((plt.pcolor(lattice[0],
                                      cmap = 'Blues'),) )

        print(lattice)
        if self.makemontage():
            anim = ani.ArtistAnimation(fig, ims)
            anim.save('montage.mp4', fps = 15)
        fig = plt.figure()
        ax = fig.subplots()
        ax.pcolor(lattice[0])
        fig.savefig('lattice.png')
        plt.close(fig)

        return energy

    
    def populate_lattice(self):
        lattice = np.zeros((self.nspecies(), 
                            self.latticesize(),
                            self.latticesize()))
        spp = 0
        for count in self.mcount():
            remaining = count
            while remaining > 0:
                x = random.randint(0, self.latticesize() - 1)
                y = random.randint(0, self.latticesize() - 1)
                lattice[spp, x, y] += 1
                remaining -= 1
            spp += 1

        return lattice


    def generate_step(self, flattice):
        # Pick a random species (out of n) and then a random ant (out of m)
        anttype = random.randint(0, self.nspecies() - 1)
        pick = random.randint(0, self.mcount()[anttype] - 1)
        position = 0
 
        # Iterate through the lattice points to figure out which ant we
        # just picked.
        while pick > flattice[anttype, position]:
            pick -= flattice[anttype, position]
            position += 1

        if flattice[anttype, position] < 1 & position < l:
            position += 1

        origin = np.array([math.floor(position / self.latticesize()),
                           position % self.latticesize()])

        moves = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        done = False
        while not done:
            step = random.randint(0, 3)
            destination = origin + moves[step]
            # Ensures 'destination' is still inside the lattice
            criteria = (destination >= 0) & (destination < self.latticesize())
            boundcheck = np.all(criteria)
            done = boundcheck
        return anttype, origin, destination


    def validate_step(self, origin, destination, flattice, anttype):
        # First check if the move is physically possible
        coords1 = origin[0] * self.latticesize() + origin[1]
        coords2 = destination[0] * self.latticesize() + destination[1]
        if (flattice[0][coords1] < 1 or flattice[0][coords2]) > 13:
            return False, 0

        dE = self.compute_dE(origin, destination, flattice, anttype)
        threshold = self.threshold() * random.random()
        if dE > 0:
            return True, dE
        elif math.exp(dE / self.temperature()) > threshold:
            return True, dE
        else:
            return False, 0


if __name__ == '__main__':
    sim = lattice_ant_model()

    if sim.tt():
        start = time.time()

    #try:
    #    runtime = int(input('Enter number of timesteps: '))
    #except (ValueError, TypeError, AssertionError):
    #    print('Input value for number of timestep is invalid. Terminating.')
    #    sys.exit(0)
    energy = sim.run()

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(np.arange(len(energy)), energy)
    fig.savefig('energy.png')
    plt.close()

    if sim.tt():
        end = time.time()
        print('Total runtime: ', end - start, ' s')
    print('Process completed successfully. Exiting.')
    sys.exit(0)


# EOF

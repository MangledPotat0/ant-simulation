################################################################################
#                                                                              #
#   SLEAP trajectory extraction code                                           #
#   Code written by: Dawith Lim                                                #
#                                                                              #
#   Version 0.9                                                                #
#   Created: 2021/12/21                                                        #
#   Last Modified: 2022/01/04                                                  #
#                                                                              #
#   Description:                                                               #
#     Python version of sleap_trajectory_extraction.nb                         #
#                                                                              #
#   Procedure:                                                                 #
#     1. Load raw position data from SLEAP                                     #
#     2. Run a 'simple' linking based on proximity and direction               #
#     3. Do 'proper' stitching based on probability                            #
#     4. Reformat output and export                                            #
#                                                                              #
################################################################################

import argparse
import h5py
from hungarian_algorithm import algorithm as hungarian
import kde
import math
import numpy as np

#   Input data structure:
#     FILE (.h5)
#       -
#       -
#       -
#       -
#       -
#       -
#       -

#   Output data structure:
#     FILE (.hdf5)
#       > trajectory            (1 x N HDF5 dataset)
#         > frame               (1 x 2 list)
#           > frame number      (int)
#           > position          (1 x 2 list)
#             > x coordinate    (float)
#             > y coordinate    (float)


def prepare(trajectories):
# Convert data format from frame-by-frame structure to trajectory-by-trajectory
# format. Also add orientation to the data

    return

def simplelink(trajectories):
# Perform a conservative simple linking based on proximity and orientation,
# just to have some trajectory fragments to get started.

    return

def oriAnt(position, weights):
# Finding orientation using relative positions of body segments

    head = position[0]
    thorax = position[1]
    abdomen = position[2]
    
# Taking a weighed sum in case simple average isn't good
    direction = weights[0] * (head - thorax) + weights[1] * (thorax - abdomen)

    return math.arctan(direction[0], direction[1])

# If body segments are unavailable, use avg positions over two frames
def oriAnt2(pos1, pos2):
    direction = pos2 - pos1

    return math.arctan(direction[0], direction[1])

def build_graph(trajectories, sources, targets:
# Build a graph mapping between source trajectories and target trajectories
# sources : list of trajectory numbers for sources
# targets: list of trajectory numbers for targets

    graph = {}
    for source in sources:
        for target in targets:
            graph.append([sources, targets, get_prob(trajectories,
                                                     source, target)])

    return np.array(graph)


def get_prob(trajectories, source, target):
# trajectories: all trajectories
# source : trajectory number for source
# target : trajectory number for target
    presource = trajectories[source, -2]
    source = trajectories[source, -1]
    target = trajectories[target, 1]
    linear = linear_prob(presource, source, target)
    angular = angular_prob(source, target)
    
    return linear * angular

def linear_prob(presource, source, target)
    tsteps = target[0] - source[0]
    speed1 = np.linalg.norm(target[1,0] - source[1,0],
                            target[1,1] - source[1,1])
    speed2 = np.linalg.norm(source[1,0] - presource[1,0],
                            source[1,1] - presource[1,1])
    acc = (2 / tsteps**2) * (speed1 - speed2)
    
    prob_value = acceleration_distribution(acc) ** tsteps

    return prob_value

def angular_prob(presource, source, target)
    tsteps = target[0] - source[0]
    aacc = (2 / tsteps**2) * mod(-math.arctan(target[1,0] - source[1,0],
                                              target[1,1] - source[1,1])
                                        - presource[2] + math.pi)
    prob_value = angular_acceleration_distribution(aacc) ** tsteps

    return prob_value


def link(trajectories):
    frames = trajectories[:,0]
    for t in range(max(frames)):
        assign = []
        sources = []
        targets = []
        for n in range(len(trajectories)):
        # Add source candidates
            if trajectories[n,-1,1] == t:
                if len(trajectories[n]) > 1,
                    sources.append(n)
        
            # Add target candidates
            if trajectories[n,1,1] == t + 1:
                targets.append()

    if min([len(sources), len(targets)]) > 0:
        # Perform Linear-Sum Assignment on the candidate pairs
        graph = build_graph(trajectories, sources, targets)
        assign = hungarian(graph)

        # Check for input-output size mismatch
        if max[len(sources), len(targets)]) - len(assign) != 0:
            print(t)
    
    # Join trajectories and remove redundancies
    sourceremove = []
    targetremove = []
        

def outputformat(trajectories):
    
    return

def run():

    ap = argparse.ArgumentParser()

    ap.add_argument('-f', '--file', type = str, required = True,
                    help = 'File name without extension')
    args = vars(ap.parse_args())
    trajectories = prepare(arg['file'])
    trajectories = prune(trajectories)
    trajectories = link(trajectories)

    return



if __name__=="__main__":
    run()
    #do stuff

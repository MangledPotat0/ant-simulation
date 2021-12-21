################################################################################
#                                                                              #
#   SLEAP trajectory extraction code                                           #
#   Code written by: Dawith Lim                                                #
#                                                                              #
#   Version 0.9                                                                #
#   Created: 2021/12/21                                                        #
#   Last Modified: 2021/12/21                                                  #
#                                                                              #
#   Description:                                                               #
#     Python version of sleap_trajectory_extraction.nb                         #
#                                                                              #
################################################################################

import argparse
import h5py
from hungarian_algorithm import algorithm as hungarian
import numpy as np

def prepare():
    
    return

def oriAnt():

    return

def build_graph(trajectories, sources, targets):
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
    linear = pdvdt(presource, source, target)
    angular = pdwdt(source, target)
    
    return linear * angular

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

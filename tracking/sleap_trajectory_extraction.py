################################################################################
#                                                                              #
#   SLEAP trajectory extraction code                                           #
#   Code written by: Dawith Lim                                                #
#                                                                              #
#   Version 0.9                                                                #
#   Created: 2021/12/21                                                        #
#   Last Modified: 2022/01/11                                                  #
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
import kde
import math
import numpy as np
import os
import scipy

#   DATA HIERARCHY

#   INPUT DATA:
#     FILE (.h5)
#       > node_names            (1 x 3 dset)
#         >NODE_NAME            (string)
#       > track_names           (1 x N dset)
#         > TRACK_NAME          (string)
#       > track_occupancy       (T x N dset)
#         > TRACK_OCCUPANCY     (boolean)
#       > tracks                (N x 2 x 3 x T dset)
#         > trajectory          (2 x 3 x T array)
#           > x coordinates     (3 x T array)
#             > NODES positions (1 x T array)
#               > xval          (float)
#           > y coordinates     (3 x T array)
#             > NODES positions (1 x T array)
#               > yval          (float)

#   OUTPUT DATA:
#     FILE (.hdf5)
#       > trajectory            (1 x N HDF5 dataset)
#         > frame               (1 x 2 array)
#           > frame number      (int)
#           > position          (1 x 2 array)
#             > x coordinate    (float)
#             > y coordinate    (float)

cwd = os.path.dirname(os.path.realpath(__file__))
datapath = "../../data/trajectories/"

########## Load and prepare data for postprocessing ##########

# TEST AND DEBUG THIS BLOCK

def prepare(datapath, inputfilename):
# Convert data format and add orientation to the data
    inputfile = h5py.File('{}{}data.h5'.format(datapath, inputfilename))
    trajectories = inputfile['tracks']
    occupancy = inputfile['track_occupancy']
    converted = []
    w = [0.7, 0.3] # Weights for orientation calculation

# Make a new list that contains trajectories
# Each entry of trajectory is [framenumber, coordinates, orientation]
    for trajectory in range(len(trajectories)):
        converted.append([])
        for frame in range(len(occupancy)):
            if occupancy[frame, trajectory]:
                current = trajectories[trajectory,:,:, frame].transpose()
                orientation = find_orientation(current, w)
                if not(np.isnan(orientation)):
                    entry = np.append([[frame, orientation]], current, axis=0)
                    converted[trajectory].append(entry)
        converted[trajectory] = np.array(converted[trajectory])
    return converted

def find_orientation(positions, weights):
# Finding orientation using relative positions of body segments
    
    parts = [part for part in positions if not np.isnan(np.sum(part))]
    if len(parts) == 3:
        head = parts[0]
        thorax = parts[1]
        abdomen = parts[2]
        dir1 = head - thorax
        dir2 = thorax - abdomen
        angle1 = math.atan2(dir1[1], dir1[0])
        angle2 = math.atan2(dir2[1], dir2[0])

# Taking a weighed sum in case simple average isn't good
        return weights[0] * angle1 + weights[1] * angle2

    elif len(parts) == 2:
        direction = parts[0] - parts[1]

        return math.atan2(direction[1], direction[0])

    elif len(parts) == 1:

        return np.nan


########## Cut up all the jumps in the trajectory ##########

# Test and debug this block

def cut_trajectories(trajectories, threshold):
    newtrajectories = []
    for trajectory in trajectories:
        jump = 0
        for t in range(1, len(trajectory)):
# When there is jump, cut and append the pice to the new list
            jumpsize = np.linalg.norm(trajectory[t][:] - trajectory[t-1][:])
            if np.nanmean(jumpsize) > threshold:
                newtrajectories.append(trajectory[jump:t])
                jump = t
# Append the leftover at the end, and also single point trajectories
            if t == len(trajectory):
                newtrajectories.append(trajectories[jump:t + 1])

    return newtrajectories
########## Re-link trajectories using probability-based cost function ##########

# Test and debug this block

def build_graph(trajectories, sources, targets):
# Build a graph mapping between source trajectories and target trajectories
# sources : list of trajectory numbers for sources
# targets: list of trajectory numbers for targets

    graph = []
    for source in sources:
        row = []
        for target in targets:
            row.append(get_prob(trajectories, source, target)) 
        graph.append(row)

    return np.array(graph)


def get_prob(trajectories, source, target):
# trajectories: all trajectories
# source : trajectory number for source
# target : trajectory number for target
    presource = trajectories[source][-2]
    source = trajectories[source][-1]
    target = trajectories[target][0]
    linear = linear_prob(presource, source, target)
    angular = angular_prob(source, target)
    
    return linear * angular

def linear_prob(presource, source, target):
    tsteps = target[0] - source[0]
    speed1 = np.linalg.norm([target[1,0] - source[1,0],
                            target[1,1] - source[1,1]])
    speed2 = np.linalg.norm([source[1,0] - presource[1,0],
                            source[1,1] - presource[1,1]])
    acc = (2 / tsteps**2) * (speed1 - speed2)
    
    prob_value = acceleration_distribution(acc) ** tsteps

    return prob_value

def angular_prob(presource, source, target):
    tsteps = target[0] - source[0]
    aacc = (2 / tsteps**2) * mod(-math.arctan(target[1,0] - source[1,0],
                                              target[1,1] - source[1,1])
                                        - presource[2] + math.pi)
    prob_value = angular_acceleration_distribution(aacc) ** tsteps

    return prob_value

def acceleration_distribution(acc):
    dset = h5py.File('acceleration.hdf5', 'r')
    for hist in dset:
        try:
            arr_index = np.asarray(hist[0] == acc).nonzero()[0][0]
            probability = hist[1,arr_index]
        except IndexError:
            probability = 0

    return probability

def angular_acceleration_distribution(aacc):
    dset = h5py.File('angular_acceleration.hdf5', 'r')
    for hist in dset:
        try:
            arr_index = np.asarray(hist[0] == acc).nonzero()[0][0]
            probability = hist[1,arr_index]
        except IndexError:
            probability = 0

    return probability

def link_trajectories(timerange, trajectories):
    frames = range(timerange)
    for t in frames:
        assign = []
        sources = []
        targets = []
        for n in range(len(trajectories)):
        # Add source candidates
            if trajectories[n][-1,0,0] == t:
                if len(trajectories[n]) > 1:
                    sources.append(n)
        
            # Add target candidates
            if trajectories[n][0,0,0] == t + 1:
                targets.append(n)

        if min([len(sources), len(targets)]) > 0:
            # Perform Linear-Sum Assignment on the candidate pairs
            graph = build_graph(trajectories, sources, targets)
            assign = scipy.optimize.linear_sum_assignment(graph, maximize=True)

            # Check for input-output size mismatch
            mismatch = max[len(sources), len(targets)] - len(assign)
            assert dim == 0, "the output size does not match the input size"
    
        # Join trajectories and remove redundancies
        sourceremove = []
        targetremove = []

        for [s, t] in assign:
            try:
                assignment = [sources[s], targets[t], graph[s, t]]
            except IndexError:
                proceed = False
            if proceed:
                sourceremove.append(sources[s])
                targetremove.append(targets[t])
                trajectories[sources[s]] = [*trajectories[sources[s]], 
                                        *trajectories[targets[t]]]
                del trajectories[target[t]]
        if len(sourceremove) > 0:
            sources = [s for s in sources if s not in sourceremove]
        if len(targetremove) > 0:
            targets = [t for t in targets if t not in targetremove]

    return trajectories


########## Formatting and exporting the output ##########
        
# I don't think reformatting is necessary? Run the code and see what
# the output ends up looking like

def outputformat(trajectories):
    
    return trajectories

def export(trajectories):

    trajectories = outputformat(trajectories)
    dfile = h5py.File('{}_proc.hdf5'.format(args['file']), 'w')
    ct = 0
    for trajectory in trajectories:
        dfile.create_dataset('trajectory{}'.format(ct),
                             data = trajectory)
        ct += 1

    return


########## Main loop for launching processes ##########


if __name__=="__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-f', '--file', type = str, required = True,
                    help = 'File name without extension')
    ap.add_argument('-s', '--skip', type = int, required = True,
                    help = 'Skip trajectory re-stitching?')
    args = vars(ap.parse_args())
    trajectories = prepare(datapath, args['file'])
    
    if not args['skip']:
        threshold = 15.1 # HARDCODED VALUE; FIX THIS LATER
        frames = 600
        trajectories = cut_trajectories(trajectories, threshold)
        print('pre-link length: ',len(trajectories))
        trajectories = link_trajectories(frames, trajectories)
        print('post-link length: ',len(trajectories))

    export(trajectories)


# EOF

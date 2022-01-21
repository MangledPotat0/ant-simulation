################################################################################
#                                                                              #
#   Created: 2022/01/10                                                        #
#   Last Modified: 2022/01/21                                                  #
################################################################################

import argparse
import h5py
import kde
import math
import numpy as np

datapath = '../../data/trajectories/'

## Input data structure:
#   HDF5 FILE
#   >   trajectory#
#       >   [framenumber, orientation]
#       >   [head_x, head_y]
#       >   [thorax_x, thorax_y]
#       >   [abdomen_x, abdomen_y]

def load(filename):
    datafile = h5py.File('{}_proc.hdf5'.format(filename), 'r')
    tracks = []
    for key in datafile.keys():
        tracks.append(datafile[key][:])
    
    return tracks

def get_difference(trajectory):
    before = trajectory[:-1]
    after = trajectory [1:]
    frames = after[1:,0]
    difference = after - before
    # Divide by nsteps in case trajectory skipped frames
    difference = np.array([difference[:,2,0] / difference[:,0,0]],
                          [difference[:,2,1] / difference[:,0,0]])
    difference = difference[:,2]
    # Set the value to actual frames
    difference[:,0] = frames

    return difference

def get_acceleration(frames, velocity):
    before = velocity[:-1]
    after = velocity[1:]
    acceleration = np.linalg.norm(after - before, axis = 1) / frames
    
    return np.array(acceleration)

def get_angular_acceleration(frames, angular_velocity):
    before = angular_velocity[:-1]
    after = angular_velocity[1:]
    angular_acceleration = mod((after - before) / frames, 2 * math.pi)

    return angular_acceleration

if __name__ == '__main__':
    
    # SLEAP output resolution is 0.01
    resolution = 0.001
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type = str, help = 'File name')
    args = vars(ap.parse_args())
    trajectories = load('{}{}'.format(datapath,args['file']))
    print(trajectories)
    frames = []
    velocity = []
    angularvelocity = []
    acceleration = []
    angular_acceleration = []
    for trajectory in trajectories:
        diff = get_difference(trajectory)
        frames.append(getdifference[:,0])
        velocity.append(getdifference[:,1])
        angularvelocity.append(getdifference[:,2])
        acceleration.append(get_acceleration(frames, velocity))
        angular_acceleration.append(
                    get_angular_acceleration(frames, angular_velocity))

    acceleration_distribution = kde.kde(acceleration, resolution)
    angular_acceleraiton_distribution = kde.kde(angular_acceleration,
                                                resolution)

    acceleration_file = h5py.File('acceleration.hdf5', 'w',
                                  dset = acceleration_distribution)
    angular_acceleration_file = h5py.File('angular_acceleration.hdf5', 'w',
                                  dset = angular_acceleration_distribution)


#EOF

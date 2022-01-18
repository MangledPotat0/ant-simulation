################################################################################
################################################################################

import h5py
import kde
import math
import numpy as np

def get_difference(trajectory):
    before = trajectory[:-1]
    after = trajectory [1:]
    frames = after[1:,0]
    difference = after - before
    # Divide by nsteps in case trajectory skipped frames
    difference = difference / difference[:,0]
    # Set the value to actual frames
    difference[:,0] = frames

    return np.array(difference)

def get_acceleration(frames, velocity):
    before = velocity[:-1]
    after = velocity[1:]
    acceleration = np.linalg.norm(after - before, axis = 1) / frames
    
    return np.array(acceleration)

def get_angular_acceleration(frames, angular_velocity):
    before = angular_velocity[:-1]
    after = angular_velocity[1:]
    angular_acceleration = mod((after - before) / frames, 2 * math.pi)

    return np.array(angular_acceleration)

if __name__ == '__main__':
    
    # SLEAP output resolution is 0.01
    resolution = 0.001
    
    frames = []
    velocity = []
    angularvelocity = []
    acceleration = []
    angular_acceleration = []
    for trajectory in trajectories:
        diff = getdifference(trajectory)
        frames.append(getdifference[:,0])
        velocity.append(getdifference[:,1])
        angularvelocity.append(getdifference[:,2])
        acceleration.append(get_acceleration(frames, velocity))
        angular_acceleration.append(
                    get_angular_acceleration(frames, angular_velocity))

    acceleration_distribution = kde.kde(acceleration, resolution)
    angular_acceleraiton_distribution = kde.kde(angular_acceleration,
                                                resolution)

    acceleration_file = h5py.File('Acceleration.hdf5', 'w'
                                  dset = acceleration_distribution)
    angular_acceleration_file = h5py.File('Angular_acceleration.hdf5', 'w',
                                  dset = angular_acceleration_distribution)


#EOF

###############################################################################
#                                                                             #
#   Ant trajectory plotter for python3                                        #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   First written on 2019/11/14                                               #
#   Last modified: 2020/06/22                                                 #
#                                                                             #
#   Packages used                                                             #
#   -   argsparse: Argument parser to handle input parameters                 #
#   -   inspect: To check if the file exists.                                 #
#   -   math: Needed for the floor function in position binning.              #
#   -   matplotlib: pyplot is used to generate all the plots.                 #
#   -   numpy: Mainly needed for convenient array manipulation.               #
#   -   os: Relative file path finding.                                       #
#   -   trackpy: Soft matter tracking package. Used here primarily to call    #
#           the PandasHDFStoreBig function that just makes data handling      #
#           easier.                                                           #
#                                                                             #
###############################################################################

import argparse
import h5py
import inspect
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import trackpy as tp

def angularvelocity_hist(filename, angularvelocity,plotres):
    w = [item for sublist in angularvelocity for item in sublist]
    w = np.array(w)

    plt.figure()
    plt.hist(w,bins=np.linspace(-1,1,plotres), label='Single ant speed')
    plt.xlabel('angular velocity (rad/s)')
    plt.ylabel('frequency')
    plt.savefig(filename+'_angularvelocity.png')
    plt.close()

def distance(position):

    trajectory = np.transpose(position)
    length = len(trajectory)
    distance = np.zeros((length,length))
    for n in range(length):
        for m in range(length - n):
            try:
                distance[n,m] = get_distance(trajectory[m+n]-trajectory[m])
            except:
                meh = 1
    return distance

def init():
    ap = argparse.ArgumentParser()
    print('Initialized')
    return ap

def getangularvelocity(orientation, fps):

    angularvelocity = []
    old = 0

    for th in orientation:
        angularvelocity.append(th - old)
        old = th

    angularvelocity = np.array(angularvelocity[1:])

    return angularvelocity

def get_distance(pair):
    x,y = pair
    output = math.sqrt(x**2 + y**2)
    return output

def getorientation(velocity):
    orientation = [0]
    for x in velocity:
        try:
            angle = math.atan2(x[0],x[1])
            orientation.append(angle)
        except:
            print('Math error')

    orientation = np.array(orientation)

    return orientation

def getposition(dataset,jump,roll):
    pos = dataset[::jump,0:2]
    pos[pos==0] = np.nan
    rat = 0.027692307

    x = rolling_average(pos[:,0],roll)
    y = rolling_average(pos[:,1],roll)
    zeros = np.where(x==0)
    x = np.delete(x,zeros)
    y = np.delete(y,zeros)
    position = [(x-min(x))*rat,(y-min(y))*rat]
    return position

def getvelocity(position):

    velocity = [[0,0]]
    speed = []
    position = np.transpose(position)
    prev = [0,0]
    for positions in position:
        velocity.append([positions[0] - prev[0],positions[1] - prev[1]])
        prev = positions
    velocity.pop(0)
    for vel in velocity:
        speed.append(np.sqrt(vel[0]**2 + vel[1]**2))
    speed = np.array(speed)
    velocity = np.array(velocity)
    return velocity, speed

def openfile(ap):
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(
                                inspect.getfile(inspect.currentframe()))))
    ap.add_argument("-f", "--file", required=True, help='Name for .hdf5 data file')
    filename = vars(ap.parse_args())['file']
    # filepath = '{}/trackpy/'.format(parentdir)
    datafile = h5py.File('{}data.hdf5'.format(filename), 'r')

    print('File ready')
    return filename, datafile

def orientation_hist(filename, orientation,plotres):
    orientation = [item for sublist in orientation for item in sublist]
    orientation = np.array(orientation)

    plt.figure()
    plt.hist(orientation,bins=np.linspace(-math.pi,math.pi,plotres),
            label='Single ant speed')
    plt.xlabel('Orientation (radians from x axis)')
    plt.ylabel('frequency')
    plt.savefig(filename+'_orientation.png')
    plt.close()

    return

def plot_trajectory(filename,position):
    plt.figure(figsize=(5.5,5.5))

    for traj in position:
        pos = np.array(traj)
        plt.plot(pos[0],pos[1])

    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.savefig('{}_trajectory'.format(filename))
    plt.close()

def rolling_average(source, count):
    try:
        source = np.ma.masked_array(source,np.isnan(source))
        output = np.cumsum(source.filled(0))
        output[count:] = output[count:] - output[:-count]
        counts = np.cumsum(~source.mask)
        counts[count:] = counts[count:] - counts[:-count]
        counts[counts==0] = 1
        try:
            with np.errstate(invalid='ignore',divide='ignore'):
                output = output/counts
        except RuntimeWarning:
            print('zero or something')
    except:
        output = source
    return output

def speed_hist(filename, speed, plotres):
    speed = [item for sublist in speed for item in sublist]
    speed = np.array(speed)
    speed = speed[speed<=3]
    plt.figure()
    plt.hist(speed,bins=np.linspace(min(speed),max(speed),plotres),
            label='Single ant speed')
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('frequency (frames)')
    plt.savefig(filename+'_speednew.png')
    plt.close()

    return

def plot_distances(distances,filename):
    avgs = np.empty(0)

    for thing in distances:
        avgs = np.append(avgs, np.average(thing))
        # this is the cause of the error. Consider the example:
        # length of trajectory = l = 10
        # then distances is 10x10 array. consider 6th row (n=5)
        # this is distances between points 5 time-points away
        # starting position (m) goes from 0 to 9
        # distances[5][m] is not zero for all m<5
        # but distances[5][m] is zero for all m>=5
        # thus, np.average is taking an average of 5 non-zero values and 5 zeros
        # thus, the average you get is half the actual average
        # the more you increase n, the more zero values are included in your avg
        # also note, some of the non-zero values may be zero due to ant not moving
        # or returning to same place, thus you can't use trim_zeros
        # instead, take part of list (eg thing[:len(thing)-n]). Thus,
        # avgs = np.append(avgs, np.average(thing[:len(thing)-n]))
    avgs = np.trim_zeros(avgs)
    # remove the above line
    plt.figure()
    plt.plot(avgs)
    plt.xlabel('t (s)')
    plt.ylabel('displacement (cm)')
    plt.savefig('{}_displacement.png'.format(filename))
    plt.close()

def position_hist(filename, position,plotres):
    position = [item for sublist in position for item in np.transpose(sublist)]
    position = np.transpose(position)

    plt.figure(figsize=(5.5,5.5))
    plt.hist2d(position[0],position[1],plotres,label='Single ant position')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.savefig(filename+'_2dhistnew.png')
    plt.close()

def main():
    jump = 5
    fps = 10
    roll = 1
    plotres = 50
    argsparse = init()
    filename, datafile = openfile(argsparse)

    position_pool = []
    speed_pool = []
    orientation_pool = []
    angular_velocity_pool = []
    distance_pool = []

    for setname in datafile:
        dataset = datafile[setname]
        position = getposition(dataset,jump,roll)
        distances = distance(position)

        velocity, speed = getvelocity(position)
        orientation = getorientation(velocity)
        angularvelocity = getangularvelocity(orientation, fps)

        position_pool.append(position)
        distance_pool[:].append(distances[:])
        # this doesn't change distance_pool, only appends to a copy of distance_pool that is not saved
        # right way:
        # distance_pool.append(distances[:])
        speed_pool.append(speed)
        orientation_pool.append(orientation)
        angular_velocity_pool.append(angularvelocity)

    plot_trajectory(filename, position_pool)
    plot_distances(distances,filename)
    # this is only for one trajectory, what about combining all trajectories
    # using distance_pool?
    position_hist(filename,position_pool,plotres)
    speed_hist(filename, speed_pool,plotres)
    orientation_hist(filename,orientation_pool,plotres)
    angularvelocity_hist(filename,angular_velocity_pool,plotres)

    print('Process exited normally.')

if __name__ == '__main__':
    main()

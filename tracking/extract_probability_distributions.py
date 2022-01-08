###############################################################################
#                                                                             #
#   Sample distribution extractor                                             #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.0.0                                                            #
#   First written on 2022/01/06                                               #
#   Last modified: 2022/01/06                                                 #
#                                                                             #
###############################################################################


# Public modules
import argparse
import copy
import h5py
import math
import numpy as np
import os
import trackpy as tp

# Local modules
from distributions import *
import kde


class AntProcessor:

    def __init__(self, argparser):
#  jump = Number of steps (in frames) between calculations. jump = 1 means
#  running calculation at every frame, and jump = 2 means skipping every other
#  frame, etc.
#  future since different plots have different appropriate binspec
#  boundary = if True, plot the boundary only; if False, plot all space.
#  boundmin, boundmax demarcate where the 'boundary' region begins.
#  (i.e. x < boundmin or x > boundmax means it's part of the boundary region)
#  size = actual size of the box in cm, fps = frames per second,
#  roll = number of frames to fold into rolling average
#  density = whether to plot cumulative counts or densities on histogram
#  trajs = number of trajectories
        self.jump = 1
        self.plotres = 40
        self.boundary = False
        self.boundmin = 0
        self.boundmax = 17.5
        self.size = 17.5
        self.fps = 10
        self.roll = 10
        self.density = True
        self.trajs = 0

        self.meancovars = []
        self.partialtraj = np.empty(0, dtype = np.int)

        args = vars(argparser.parse_args())
#  timebins = number of bins along the time direction, for analyses of figures
#  during different time intervals. runtype is choice between 'p' (pool), meaning
#  the file is a collection of trajectories, or 's' (single), meaning the file
#  contains only one trajectory. Running with 'p' for single file should still
#  run, but running with 's' option on multiple trajectories will cause error.
        self.timebins = args['timebins']
        self.runtype = args['runtype']
        
        temppath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(temppath)
        self.filename = args['file']
        self.filepath = "../../data/trajectories/"
        self.datafile = h5py.File(
                    '{}{}data.hdf5'.format(self.filepath, self.filename),
                    'r')
        self.figpath = "../../data/plots/{}/".format(self.filename)

#  Set up new figure export directory if it doesn't already exist
        try:
            os.mkdir(self.figpath)
            print('Target directory not found; creating new directory.')
        except OSError:
            print('Directory {} already exists.'.format(self.figpath))
        
        print('File ready')
        print('Process initialized.')

    def get_acceleration(self, speed):
#  Calculate acceleration as a vector (cm / s^2)
        acceleration = []
        temp = 0

        for value in speed:
            acceleration.append(value - temp)
            temp = value

        acceleration = np.delete(acceleration, 0)

        return acceleration * self.fps

    def get_angularvelocity(self, traj):
#  Calculate angular velocity (radians / s)
        orientation = traj.orientation
        angularvelocity = np.empty(0)
        old = 0

        for th in orientation:
            angularvelocity = np.append(angularvelocity, th - old)
            old = th

        angularvelocity = np.mod(angularvelocity, 2 * np.pi) - np.pi
        angularvelocity = np.array(angularvelocity[1:]) * self.fps

        return angularvelocity

    def get_displacement(self, position):
#  Compute time series of mean displacement starting from each positions along
#  the trajectory
        trajectory = position
        length = len(trajectory)
        maxrange = 1000
        displacement = np.empty((length, maxrange))
        displacement.fill(np.nan)

        for n in range(length):
            terminate = 0
            for m in range(length - n):
                if terminate < maxrange:
                    displacement[n,m] = distance(
                                            trajectory[m+n] - trajectory[m])
                    terminate += 1

        return displacement

    def get_distance_to_center(self, position):
#  Compute distance from each point on the trajectory to the center of
#  the box.
        dfc = np.empty(0)

        for coords in position:
            distance = math.sqrt((coords[0] - self.size / 2) ** 2 
                                 + (coords[1] - self.size / 2) ** 2)
            dfc = np.append(dtb, distance)
        
        return dfc

    def get_orientation(self, velocity):
        orientation = np.empty(0)
        
        for x in velocity:
            try:
                angle = math.atan2(x[0], x[1])
                orientation = np.append(orientation, angle)
# When infinity gets involved, pass error message
            except:
                print('Math error')

        orientation = np.array(orientation)
    
        return orientation

    def get_position(self, dataset):
        pos = dataset[::self.jump, 0:2]
        pos[pos==0] = np.nan

        x = rolling_average(pos[:,0], self.roll)
        y = rolling_average(pos[:,1], self.roll)
        zeros = np.where(x == 0)
        x = np.delete(x, zeros)
        y = np.delete(y, zeros)
        
# Rescale the position to be in cm instead of pixel counts.
        position = np.array(
                    [(x - min(x)) * self.size / (max(x) - min(x)),
                     (y - min(y)) * self.size / (max(y) - min(y))])

        return np.transpose(position)

    def get_velocity(self, position):
    
        velocity = np.empty((0, 2))
        speed = np.empty(0)
        prev = [0, 0]
        for positions in position:
            velocity = np.vstack((velocity, np.array(
                                                [[positions[0] - prev[0],
                                                  positions[1] - prev[1]]])))
            prev = positions
        velocity = velocity * self.fps

        for vel in velocity:
            speed = np.append(speed, np.sqrt(vel[0] ** 2 + vel[1] ** 2))

        velocity = np.delete(velocity, 0, axis = 0)
        speed = np.delete(speed, 0)
        
        return velocity, speed

    def plot_acceleration_hist(self, acc, n):
        #speed = speed[speed<5]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        binheight, binborders, _ = ax.hist(
                        acc, label = 'Data',
                        alpha = 0.8, 
                        density = self.density,
                        bins = np.linspace(-6, 6, self.plotres))
        bincenters = binborders[:-1] + np.diff(binborders) / 2
        plotbins = np.linspace(-6, 6, 10000)
        
        popt, _ = fit(gaussian1D, bincenters, binheight, [1., 0., 1.])
        ax.plot(plotbins, gaussian1D(plotbins, *popt),label = 'Gaussian fit')
        
        popt, _ = fit(lorentz1D, bincenters, binheight, [1., 0., 1.])
        ax.plot(plotbins, lorentz1D(plotbins,* popt),label = 'Lorentz fit')
        
        popt, _ = fit(laplace1D, bincenters, binheight, [0., 1.])
        ax.plot(plotbins, laplace1D(plotbins, *popt),label = 'Laplace fit')
        
        popt, _ = fit(logistic1D, bincenters, binheight, [0., 1.])
        ax.plot(plotbins, logistic1D(plotbins, *popt), label = 'Logistic fit')
        
        ax.set_xlabel('Acceleration (cm/s^2)')
        ax.set_ylabel('frequency (frames)')
        ax.legend()
        
        plt.savefig('{}{}_acc_hist_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n), bbox_inches = 'tight')
        plt.close()

        return

    def plot_angularvelocity_hist(self, angularvelocity, n):
        fig = plt.figure()
        fig1 = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = fig1.add_subplot(111)
        width = 0.5
        
        histbins = np.linspace(-width * np.pi, width * np.pi, 60)
        plotbins = np.linspace(-width * np.pi, width * np.pi, 10000)
        
        binheight, binborders, _ = ax.hist(
                            angularvelocity,
                            bins = histbins,
                            label = 'Data',
                            density = self.density)
        bincenters = binborders[:-1] + np.diff(binborders) / 2

        ax1.hist(angularvelocity, 
                bins = histbins,
                label = 'Data', 
                density = self.density,
                cumulative = True)

def distance(pair):
    x, y = pair
    output = math.sqrt(x ** 2 + y ** 2)
    return output 

def rolling_average(source, count):
    try:
        source = np.ma.masked_array(source, np.isnan(source))
        output = np.cumsum(source.filled(0))
        output[count:] = output[count:] - output[:-count]
        counts = np.cumsum(~source.mask)
        counts[count:] = counts[count:] - counts[:-count]
        counts[counts == 0] = 1
        try:
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
                output = output / counts
        except RuntimeWarning:
            print('Dividing by zero!')
    except:
        output = source
    return output


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file', required = True,
                           help = 'Name for .hdf5 data file')
    argparser.add_argument('-t', '--timebins', required = True,
                           help = 'Number of time bins', type = int)
    argparser.add_argument('-r', '--runtype', required = True,
                           help = 'Run type', type = str)
    ap = AntProcessor(argparser)


    

# EOF

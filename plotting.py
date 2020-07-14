###############################################################################
#                                                                             #
#   Ant trajectory plotter for python3                                        #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.4.4.0.3.2                                                      #
#   First written on 2019/11/14                                               #
#   Last modified: 2020/07/01                                                 #
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
import copy
import h5py
import inspect
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import trackpy as tp

class AntProcessor:
    def __init__(self, argparser):
        self.jump = 1
        self.roll = 4
        self.plotres = 50
        self.boundary = True
        self.boundmin = 1
        self.boundmax = 16.5
        args = vars(argparser.parse_args())
        self.timebins = args['timebins']
        self.runtype = args['runtype']
        temppath = os.path.dirname(os.path.realpath(__file__))
        os.chdir(temppath)
        self.datapath = os.getcwd()
        self.filename = args['file']
        self.filepath = "../data/trajectories/"
        self.datafile = h5py.File('{}{}data.hdf5'.format(
            self.filepath,self.filename),'r')
        self.figpath = "../data/plots/{}/".format(self.filename)
        try:
            os.mkdir(self.figpath)
            print('Target directory not found; creating new directory.')
        except OSError:
            print('Directory {} already exists.'.format(self.figpath))
        print('File ready')
        print('Process initialized.')

    def get_angularvelocity(self,orientation):
    
        angularvelocity = np.empty(0)
        old = 0

        for th in orientation:
            angularvelocity = np.append(angularvelocity,th - old)
            old = th

        angularvelocity = np.array(angularvelocity[1:])

        return angularvelocity

    def get_displacement(self, position):

        trajectory = np.transpose(position)
        length = len(trajectory)
        displacement = np.empty((length,length))
        displacement.fill(np.nan)
        for n in range(length):
            for m in range(length - n):
                try:
                    displacement[n,m] = distance(trajectory[m+n]-trajectory[m])
                except:
                    meh = 1
        return displacement

    def get_orientation(self, velocity):
        orientation = np.empty(0)
        for x in velocity:
            try:
                angle = math.atan2(x[0],x[1])
                orientation = np.append(orientation,angle)
            except:
                print('Math error')

        orientation = np.array(orientation)
    
        return orientation

    def get_position(self, dataset):
        pos = dataset[::self.jump,0:2]
        pos[pos==0] = np.nan
        rat = 0.027692307

        x = rolling_average(pos[:,0],self.roll)
        y = rolling_average(pos[:,1],self.roll)
        zeros = np.where(x==0)
        x = np.delete(x,zeros)
        y = np.delete(y,zeros)
        position = np.array([(x-min(x))*rat,(y-min(y))*rat])
        return np.transpose(position)

    def get_velocity(self, position):
    
        velocity = [[0,0]]
        speed = []
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

    def plot_angularvelocity_hist(self, angularvelocity, n):
        plt.figure()
        plt.hist(angularvelocity,bins=np.linspace(-1,1,self.plotres), label=
                'Single ant speed')
        plt.xlabel('angular velocity (rad/s)')
        plt.ylabel('frequency')
        plt.savefig('{}{}_angularvelocity_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_distances(self, distances,n):
        avgs = np.nanmean(distances, axis=1)
        plt.figure()
        plt.plot(avgs)
        plt.xlabel('t (s)')
        plt.ylabel('displacement (cm)')
        plt.savefig('{}{}_displacement_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_orientation_hist(self, orientation, n):
        plt.figure()
        plt.hist(orientation,bins=np.linspace(-math.pi,math.pi,self.plotres),
                label='Single ant speed')
        plt.xlabel('Orientation (radians)')
        plt.ylabel('frequency')
        plt.savefig('{}{}_orientation_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

        return

    def plot_position_hist(self, position,n):
        position = np.transpose(position)
        plt.figure(figsize=(5.5,5.5))
        plt.hist2d(position[0],position[1],self.plotres,label=
                'Single ant position')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.savefig('{}{}_2dhist_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_speed(self, speed,n):
        speed = np.delete(speed,0) 
        time = np.arange(0,len(speed),1)
        plt.figure()
        plt.plot(time,speed)
        plt.xlabel('Time (frames)')
        plt.ylabel('Speed (m/s)')
        plt.savefig('{}{}_speed_{}-{}.png'.format(self.figpath,self.filename,
                    n,self.timebins))

    def plot_speed_hist(self, speed,n):
        speed = speed[speed<4]
        plt.figure()
        plt.hist(speed,bins=np.linspace(min(speed),max(speed),self.plotres),
                label='Single ant speed')
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('frequency (frames)')
        plt.savefig('{}{}_speed_hist_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n))
        plt.close()

        return
    
    def plot_trajectory(self, position,n):
        plt.figure(figsize=(5.5,5.5))
        position = np.transpose(position)
        plt.plot(position[0],position[1])
        plt.savefig('{}{}_trajectory_{}-{}.png'.format(self.figpath,
                    self.filename,self.timebins,n))
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.close()

    class Trajectory:
        def __init__(self, antproc, data):
            length = len(data)
            shape = np.array([length,2])
            self.position = antproc.get_position(data)
            self.displacement = antproc.get_displacement(self.position)
            [self.velocity, self.speed] = antproc.get_velocity(self.position)
            self.orientation = antproc.get_orientation(self.velocity)
            self.angularvelocity = antproc.get_angularvelocity(self.orientation)
    
        def make_indices(self, data, criteria):
            minthresh,maxthresh = criteria
            truthfunction = ((data[:,0] > minthresh) & (data[:,0] < maxthresh)
                    ) & ((data[:,1] > minthresh) & (data[:,1] < maxthresh))
            self.indices = np.broadcast_to(truthfunction, len(data))

def distance(pair):
    x,y = pair
    output = math.sqrt(x**2 + y**2)
    return output 

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

def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--file', required=True, help=
                'Name for .hdf5 data file')
    argparser.add_argument('-t', '--timebins', required=True, help=
                'Number of time bins', type=int)
    argparser.add_argument('-r', '--runtype', required=True, help='Run type',
                type=str)
    ap = AntProcessor(argparser)
    
    trajectories = np.array([])
    
    ct = 1
    for setname in ap.datafile: 
        dataset = ap.datafile[setname]
        trajectories = np.append(trajectories, ap.Trajectory(ap, 
                                 dataset[1:]))
        trajectories[-1].make_indices(trajectories[-1].position,
                [ap.boundmin,ap.boundmax])
        if ap.runtype == 's':
            for t in range(ap.timebins):
                length=len(trajectories[-1].position)
                binsize=math.floor(length/ap.timebins)
                start = t*binsize
                end = (t+1)*binsize - 1
                indices = np.empty((len(trajectories[-1].position)),dtype=bool)
                indices.fill(False)
                if ap.boundary:
                    trajectories[-1].indices = not trajectories[-1].indices
                indices[start:end] = trajectories[-1].indices[start:end]
                print(indices)
                ap.plot_trajectory(trajectories[-1].position[indices],t+1)
                ap.plot_position_hist(trajectories[-1].position[indices],t+1)
                ap.plot_speed(trajectories[-1].speed[indices],t+1)
                ap.plot_speed_hist(trajectories[-1].speed[indices],t+1)
                ap.plot_orientation_hist(trajectories[-1].orientation[indices[
                                :]],t+1)
                ap.plot_angularvelocity_hist(trajectories[-1].angularvelocity
                            [indices[1:]],t+1)
    if ap.runtype =='p':
        pool = copy.deepcopy(trajectories[-1])
        for t in range(ap.timebins):
            pool.position = np.empty((0,2))
            pool.speed = np.array([])
            pool.orientation = np.array([])
            pool.angularvelocity = np.array([])
            for traj in trajectories:
                traj.make_indices(traj.position,[ap.boundmin,ap.boundmax])
                length=len(traj.position)
                binsize=math.floor(length/ap.timebins)
                start = t*binsize
                end = (t+1)*binsize - 1
                indices = np.empty(length,dtype=bool)
                indices.fill(False)
                if ap.boundary:
                    trajectories[-1].indices = ~trajectories[-1].indices[:]
                indices[start:end] = traj.indices[start:end]
                pool.position = np.vstack((pool.position, 
                        traj.position[indices]))
                pool.speed = np.append(pool.speed, 
                            traj.speed[indices],axis=0)
                pool.orientation = np.append(pool.orientation, 
                            traj.orientation[indices],axis=0)
                pool.angularvelocity = np.append(pool.angularvelocity, 
                        traj.angularvelocity[indices[1:]],axis=0)

            ap.plot_position_hist(pool.position,t)
            ap.plot_speed_hist(pool.speed,t)
            ap.plot_orientation_hist(pool.orientation,t)
            ap.plot_angularvelocity_hist(pool.angularvelocity,t)
                    
                    
        #pooled data plots

    

    print('Process exited normally.')

if __name__ == '__main__':
    main()

# Put in as subclass to antprocessor


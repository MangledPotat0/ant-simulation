###############################################################################
#                                                                             #
#   Ant trajectory plotter for python3                                        #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.4.1.0.3.1                                                      #
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
        self.roll = 1
        self.plotres = 50
        args = vars(argparser.parse_args())
        self.timebins = args["timebins"]
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
    
        angularvelocity = []
        old = 0

        for th in orientation:
            angularvelocity.append(th - old)
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
        orientation = [0]
        for x in velocity:
            try:
                angle = math.atan2(x[0],x[1])
                orientation.append(angle)
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
        position = [(x-min(x))*rat,(y-min(y))*rat]
        return position

    def get_velocity(self, position):
    
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

    def plot_angularvelocity_hist(self, angularvelocity, n):
        w = [item for sublist in angularvelocity for item in sublist]
        w = np.array(w)
    
        plt.figure()
        plt.hist(w,bins=np.linspace(-1,1,self.plotres), label=
                'Single ant speed')
        plt.xlabel('angular velocity (rad/s)')
        plt.ylabel('frequency')
        plt.savefig('{}{}_angularvelocity_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_distances(self, distances,n):
        print(np.shape(distances))
        avgs = np.nanmean(distances, axis=1)
        plt.figure()
        plt.plot(avgs)
        plt.xlabel('t (s)')
        plt.ylabel('displacement (cm)')
        plt.savefig('{}{}_displacement_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_orientation_hist(self, orientation, n):
        orientation = [item for sublist in orientation for item in sublist]
        orientation = np.array(orientation)
    
        plt.figure()
        plt.hist(orientation,bins=np.linspace(-math.pi,math.pi,self.plotres),
                label='Single ant speed')
        plt.xlabel('Orientation (radians from x axis)')
        plt.ylabel('frequency')
        plt.savefig('{}{}_orientation_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

        return

    def plot_position_hist(self, position,n):
        position = [item for sublist in position for item in 
                np.transpose(sublist)]
        position = np.transpose(position)
   
        plt.figure(figsize=(5.5,5.5))
        plt.hist2d(position[0],position[1],self.plotres,label=
                'Single ant position')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.savefig('{}{}_2dhist_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_speed_hist(self, speed,n):
        plt.figure()
        if self.collected:
            for lists in speed:
                thing = [item for sublist in lists for item in sublist]
                thing = np.array(thing) 
                thing = thing[thing<=3]
                plt.hist(thing,bins=np.linspace(min(thing),max(thing),self.plotres),
                        label='Single ant speed', alpha = 0.3)
            plt.savefig('{}{}_speed_{}.png'.format(self.figpath,
                self.filename,self.timebins))
                
        else:
            speed = [item for sublist in speed for item in sublist]
            speed = np.array(speed) 
            speed = speed[speed<=3]
            plt.figure()
            plt.hist(speed,bins=np.linspace(min(speed),max(speed),self.plotres),
                    label='Single ant speed')
            plt.savefig('{}{}_speed_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n))
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('frequency (frames)')
        plt.close()

        return
    
    def plot_trajectory(self, position,n):
        plt.figure(figsize=(5.5,5.5))
        
        if self.collected:
            for traj in position:
                for segment in traj:
                    pos = np.array(segment)
                    plt.plot(pos[0],pos[1])
            plt.savefig('{}{}_trajectory_{}.png'.format(self.figpath,
                        self.filename,self.timebins))
        else:
            for traj in position:
                pos = np.array(traj)
                plt.plot(pos[0],pos[1])
            plt.savefig('{}{}_trajectory_{}-{}.png'.format(self.figpath,
                            self.filename,self.timebins,n))
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.close()

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
    ap = AntProcessor(argparser)
    
    x = []
    v = []
    o = []
    w = []
    d = []
    
    ap.collected = False;
    for t in range(ap.timebins):
        
        position_pool = []
        speed_pool = []
        orientation_pool = []
        angular_velocity_pool = []
        displacement_pool = []

        for setname in ap.datafile:
            dataset = ap.datafile[setname]
            setlength = math.floor(len(dataset)/ap.timebins)
            dataset = dataset[t*setlength:(t+1)*setlength-1]
            position = ap.get_position(dataset)
            displacement = ap.get_displacement(position)
            velocity, speed = ap.get_velocity(position)
            orientation = ap.get_orientation(velocity)
            angularvelocity = ap.get_angularvelocity(orientation)
            position_pool.append(position)
            if len(displacement_pool) != 0:
                for n in range(len(displacement_pool)):
                    displacement_pool[n].append(displacement[n])
                    print(displacement_pool)
            speed_pool.append(speed)
            orientation_pool.append(orientation)
            angular_velocity_pool.append(angularvelocity)
        x.append(position_pool)
        v.append(speed_pool)
        ap.plot_trajectory(position_pool,t)
        ap.plot_distances(displacement,t)
        ap.plot_position_hist(position_pool,t)
        ap.plot_speed_hist(speed_pool,t)
        ap.plot_orientation_hist(orientation_pool,t)
        ap.plot_angularvelocity_hist(angular_velocity_pool,t)
    ap.collected = True
    ap.plot_trajectory(x,t)
    ap.plot_speed_hist(v,1)
    print('Process exited normally.')

if __name__ == '__main__':
    main()

class trajectory:
    def __init__(self, data):
        length = len(data)
        shape = np.array([length,2])
        self.position = np.empty(shape)
        self.displacement = np.empty()
        self.velocity = np.empty(shape)
        self.speed = np.empty(length)
        self.orientation = np.empty(length)
        self.angularvelocity = np.empty(length)

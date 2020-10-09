###############################################################################
#                                                                             #
#   Ant trajectory plotter for python3                                        #
#   Code written by Dawith Lim                                                #
#                                                                             #
#   Version: 1.7.2.3.0.0                                                      #
#   First written on 2019/11/14                                               #
#   Last modified: 2020/09/14                                                 #
#                                                                             #
#   Packages used                                                             #
#   -   argsparse: Argument parser to handle input parameters                 #
#   -   distributions: Local module in the codebase that contains all the     #
#           distribution functions used in the code.                          #
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
from distributions import * # (LOCAL)
import h5py
import inspect
import kde # LOCAL
import math
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import os
from scipy.optimize import curve_fit as fit
from scipy.special import kn as modbessel
import trackpy as tp

class AntProcessor:
    def __init__(self, argparser):
        self.jump = 1
        self.plotres = 40
        self.boundary = False
        self.boundmin = 0
        self.boundmax = 18
        self.size = 18
        self.fps = 10
        self.roll = 2*self.fps
        self.density = True
        self.trajs = 0
        self.meancovars = []
        self.blyat = np.empty(0, dtype=np.int)
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

    def get_acceleration(self, speed):
        acceleration = []
        temp = 0
        for value in speed:
            acceleration.append(value-temp)
            temp = value
        acceleration = np.delete(acceleration,0)
        return acceleration*self.fps

    def get_angularvelocity(self,traj):
        orientation = traj.orientation
        angularvelocity = np.empty(0)
        old = 0

        for th in orientation:
            angularvelocity = np.append(angularvelocity,th - old)
            old = th

        angularvelocity = np.mod(angularvelocity, 2*np.pi)-np.pi
        angularvelocity = np.array(angularvelocity[1:])*self.fps
        return angularvelocity

    def get_displacement(self, position):

        trajectory = position
        length = len(trajectory)
        maxrange = 1000
        displacement = np.empty((length,maxrange))
        displacement.fill(np.nan)
        for n in range(length):
            terminate = 0
            for m in range(length - n):
                if terminate < maxrange:
                    displacement[n,m] = distance(trajectory[m+n]-trajectory[m])
                    terminate += 1

        return displacement

    def get_distance_to_boundary(self, position):
        dtb = np.empty(0)
        for coords in position:
            distance = math.sqrt((coords[0]-self.size/2)**2 + 
                    (coords[1]-self.size/2)**2)
            dtb = np.append(dtb,distance)
        
        return dtb

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

        x = rolling_average(pos[:,0],self.roll)
        y = rolling_average(pos[:,1],self.roll)
        zeros = np.where(x==0)
        x = np.delete(x,zeros)
        y = np.delete(y,zeros)
        position = np.array([(x-min(x))*self.size/(max(x)-min(x)),
            (y-min(y))*self.size/(max(y)-min(y))])
        return np.transpose(position)

    def get_velocity(self, position):
    
        velocity = np.empty((0,2))
        speed = np.empty(0)
        prev = [0,0]
        for positions in position:
            velocity = np.vstack((velocity, np.array([[positions[0] - prev[0],
                positions[1] - prev[1]]])))
            prev = positions
        velocity = velocity * self.fps
        for vel in velocity:
            speed = np.append(speed, np.sqrt(vel[0]**2 + vel[1]**2))
        velocity = np.delete(velocity, 0,axis=0)
        speed = np.delete(speed, 0)
        return velocity, speed

    def plot_acceleration_hist(self, acc, n):
        #speed = speed[speed<5]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        binheight, binborders, _ = ax.hist(acc,label='Data',alpha=0.8, 
                density=self.density, bins=np.linspace(-6,6,self.plotres))
        bincenters = binborders[:-1] + np.diff(binborders)/2
        plotbins = np.linspace(-6,6,10000)
        popt, _ = fit(gaussian1D, bincenters, binheight, [1.,0.,1.])
        ax.plot(plotbins,gaussian1D(plotbins,*popt),label='Gaussian fit')
        popt, _ = fit(lorentz1D, bincenters, binheight, [1.,0.,1.])
        ax.plot(plotbins,lorentz1D(plotbins,*popt),label='Lorentz fit')
        popt, _ = fit(laplace1D, bincenters, binheight, [0.,1.])
        ax.plot(plotbins,laplace1D(plotbins,*popt),label='Laplace fit')
        popt, _ = fit(logistic1D, bincenters, binheight, [0.,1.])
        ax.plot(plotbins,logistic1D(plotbins,*popt),label='Logistic fit')
        ax.set_xlabel('Acceleration (cm/s^2)')
        ax.set_ylabel('frequency (frames)')
        ax.legend()
        plt.savefig('{}{}_acc_hist_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

        return

    def plot_angularvelocity_hist(self, angularvelocity, n):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 2
        histbins = np.linspace(-width*np.pi, width*np.pi,self.plotres)
        plotbins = np.linspace(-width*np.pi, width*np.pi,10000)
        binheight, binborders, _ = ax.hist(angularvelocity, bins= histbins,
                label='Data', density=self.density)
        bincenters = binborders[:-1] + np.diff(binborders)/2

#       Logistic fitting

        #popt, cov = fit(logistic1D, bincenters, binheight, p0=[0.,1.])
        #ax.plot(plotbins, logistic1D(plotbins,*popt), label='Logistic fit')
        #self.paramnames = ['Mean', 'Amplitude']

#       Lorentz fitting

        #popt, cov = fit(lorentz1D, bincenters, binheight, p0=[1.,0.,1.])
        #ax.plot(plotbins, lorentz1D(plotbins,*popt), label='Lorentz fit')
        #self.paramnames = ['Mean', 'Amplitude', 'Gamma']

#       Gaussian fitting

        #popt, cov = fit(gaussian1D, bincenters, binheight, [1.,0.,1.])
        #ax.plot(plotbins, gaussian1D(plotbins,*popt), label='Gaussian fit')
        #self.paramnames = ['Mean', 'standard deviation']

#       Laplace fitting
#       popt, cov = fit(laplace1D, bincenters, binheight, p0=[0.,0.4])
#        ax.plot(plotbins, laplace1D(plotbins,*popt), label='Laplace fit')
#        self.paramnames = ['Mean', 'Scale']

#       von Mises fitting

        #popt, cov = fit(vonMises1D, bincenters, binheight, p0=[0.,5.])
        #ax.plot(plotbins, vonMises1D(plotbins,*popt), label=fitlabel)
        #self.paramvals = popt
        ax.set_xlabel('anglular velocity (rad/s)')
        ax.set_ylabel('frequency')
        ax.legend()
        plt.savefig('{}{}_angularvelocity_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n), bbox_inches='tight')
        plt.close(fig)

        fig1 = plt.figure()
        ax = fig1.subplots()
        ax.plot(bincenters,binheight-np.flip(binheight),label='Left-Right')
        ax.legend()
        fig1.savefig('{}{}_avsymmetry_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n), bbox_inches='tight')
        plt.close(fig1)

    def plot_dfc_hist(self, dfc, n):
        fig = plt.figure()
        ax = fig.subplots(3,1)
        binheight, histbins, _ = ax[0].hist(dfc, self.plotres, 
                density=self.density,label='Data')
        binheight, histbins, _ = ax[1].hist(dfc, self.plotres, 
                density=self.density,label='Data')
        binheight, histbins, _ = ax[2].hist(dfc, self.plotres, 
                density=self.density,label='Data')
        bincenters = histbins[:-1] + np.diff(histbins)/2
        plotbins = np.linspace(min(dfc),max(dfc),10000)
        popt, cov = fit(composite_lorentz_polyo1, bincenters, binheight,
                p0=[.1,0.,9.,1.,1.])
        density = kde.kde(dfc, 1000)
        ax[0].plot(plotbins,composite_lorentz_polyo1(plotbins,*popt), 
                label='Lorentz + linear fit')
        ax[0].plot(density[0],density[1])
        ax[0].set_xlabel('distance from center (cm)')
        ax[0].set_ylabel('frequency')
        ax[0].fill_between(density[0],density[1],y2=0)
        ax[0].legend()
        ax[1].plot(density[0],density[1])
        ax[1].fill_between(density[0],density[1],y2=0)
        ax[1].plot(plotbins,composite_lorentz_polyo1(plotbins,*popt), 
                label='Lorentz + linear fit')
        ax[1].set_xlabel('distance from center (cm)')
        ax[1].set_ylabel('frequency')
        ax[1].set_yscale('log', nonposy='clip')
        #ax[1].set_xscale('log')
        ax[2].plot(density[0],density[1])
        ax[2].fill_between(density[0],density[1],y2=0)
        ax[2].plot(plotbins,composite_lorentz_polyo1(plotbins,*popt), 
                label='Lorentz + linear fit')
        ax[2].set_xlabel('distance from center (cm)')
        ax[2].set_ylabel('frequency')
        ax[2].set_yscale('log', nonposy='clip')
        ax[2].set_xscale('log', nonposx='clip')
        plt.savefig('{}{}_dfc_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n))
        plt.close()

    def plot_displacement(self, displacement,n):
        avgs = np.nanmean(displacement, axis=1)
        plt.figure()
        plt.plot(avgs)
        plt.xlabel('t (s)')
        plt.ylabel('displacement (cm)')
        plt.savefig('{}{}_displacement_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

    def plot_orientation_hist(self, orientation, n):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='polar')
        ax.hist(orientation,bins=np.linspace(-math.pi,math.pi,self.plotres),
                label='Single ant speed',density=self.density)
        temp = np.mod(orientation,2*np.pi)
        temp1 = np.append(temp, temp - 2*np.pi)
        temp2 = np.append(temp1, temp + 2*np.pi)
        density = kde.kde(temp2,3000)
        leng = len(temp)
        ax.plot(density[0,1000:2001],3*density[1,1000:2001],'r',
                label='kernel density estimate')
        ax.fill_between(density[0],density[1],y2=0)
        ax.set_ylabel('frequency')
        plt.savefig('{}{}_orientation_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

        return

    def plot_position_hist(self, position,n):
        position = np.transpose(position)
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        h = ax.hist2d(position[0],position[1],self.plotres,label=
                'Single ant position', norm=LogNorm())
        #plt.colorbar(h[3])
        #plt.scatter(position[0],position[1],alpha=0.005*self.timebins)
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.savefig('{}{}_2dhist_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

    def plot_speed(self, speed,n):
        speed = np.delete(speed,0) 
        time = np.arange(0,len(speed),1)
        plt.figure()
        plt.plot(time,speed)
        plt.xlabel('Time (frames)')
        plt.ylabel('Speed (m/s)')
        plt.savefig('{}{}_speed_{}-{}.png'.format(self.figpath,self.filename,
                    n,self.timebins), bbox_inches='tight')
        plt.close()

    def plot_speed_hist(self, speed,n):
        speed = speed[speed<5]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(speed,bins=np.linspace(min(speed),max(speed),self.plotres),
                label='Single ant speed',density=self.density)
        temp = -speed
        temp = np.append(temp, speed)
        density = kde.kde(temp,1000)
        ax.plot(density[0,500:],2*density[1,500:],'r',
                label='kernel density estimate')
        ax.fill_between(density[0],density[1],y2=0)
        ax.set_xlabel('Speed (cm/s)')
        ax.set_ylabel('frequency (frames)')
        #ax.set_xlim([0,5])
        ax.set_ylim([0,1.2])
        fig.savefig('{}{}_speed_hist_{}-{}.png'.format(self.figpath,
                self.filename,self.timebins,n), bbox_inches='tight')
        plt.close(fig)

        return
    
    def plot_blyat(self, position,n):
        blyat = self.blyat
        blyat1 = np.arange(len(blyat))
        blyat = blyat1[blyat]
        fig = plt.figure(figsize=(5.5,5.5))
        ax = fig.subplots()
        position = np.transpose(position)
        ax.plot(position[0][blyat],position[1][blyat],'.',alpha=0.05)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        fig.savefig('{}{}_trajectory_{}-{}.png'.format(self.figpath,
                    self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

    def plot_trajectory(self, position,n):
        plt.figure(figsize=(5.5,5.5))
        position = np.transpose(position)
        plt.plot(position[0],position[1])
        plt.savefig('{}{}_trajectory_{}-{}.png'.format(self.figpath,
                    self.filename,self.timebins,n), bbox_inches='tight')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.close()

    def plot_velocity(self, velocity, n):
        velocity = np.transpose(velocity)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(velocity[0], velocity[1],alpha=0.005)
        ax.scatter([0],[0], c='r')
        ax.set_aspect(1)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        plt.savefig('{}{}_velocity_{}-{}.png'.format(self.figpath,
            self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

    def plot_orient_vs_dist_from_center(self, data1, data2, n):
        fig = plt.figure()
        ax = fig.add_subplot(111,polar=True)
        ax.scatter(data1,data2, alpha=0.012*self.timebins/self.trajs)
        x = np.linspace(0,2*np.pi,360)
        y = np.full((360),9)
        ax.plot(x,y, 'r')
        #plt.scatter(data1,data2,alpha=0.05)
        ax.set_ylabel('Distance from center (cm)')
        #plt.xlabel('distance from center(cm)')
        ax1 = fig.add_subplot(211)
        ax1.hist2d(data1, data2, 40)
        plt.savefig('{}{}_radialdistance{}{}.png'.format(self.figpath,
                    self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()

    def plot_dfc_vs_speed(self,dfc,speed,n):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dfc,speed, alpha=0.002)
        ax.set_ylabel('Speed (cm/s)')
        ax.set_xlabel('Distance from center(cm)')
        plt.savefig('{}{}_dfcvsspeed{}{}.png'.format(self.figpath,
                    self.filename,self.timebins,n), bbox_inches='tight')
        plt.close()
    
    class Trajectory:
        def __init__(self, antproc, data):
            length = len(data)
            shape = np.array([length,2])
            antproc.trajs += 1
            self.position = antproc.get_position(data)
            #self.displacement = antproc.get_displacement(self.position)
            [self.velocity, self.speed] = antproc.get_velocity(self.position)
            self.orientation = antproc.get_orientation(self.velocity)
            self.angularvelocity = antproc.get_angularvelocity(self)
            self.acceleration = antproc.get_acceleration(self.speed)
            self.dfc = antproc.get_distance_to_boundary(self.position)[2:]
            self.position = self.position[2:]
            self.velocity = self.velocity[1:]
            self.speed = self.speed[1:]
            self.orientation = self.orientation[1:]
    
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
                    trajectories[-1].indices = ~trajectories[-1].indices
                indices[start:end] = trajectories[-1].indices[start:end]
                ap.plot_trajectory(trajectories[-1].position[indices],t+1)
                ap.plot_position_hist(trajectories[-1].position[indices],t+1)
                ap.plot_speed(trajectories[-1].speed[indices[1:]],t+1)
                ap.plot_speed_hist(trajectories[-1].speed[indices[1:]],t+1)
                ap.plot_orientation_hist(trajectories[-1].orientation[indices[
                                1:]],t+1)
                ap.plot_angularvelocity_hist(trajectories[-1].angularvelocity
                            [indices[2:]],t+1)
    
    if ap.runtype =='p':
        pool = copy.deepcopy(trajectories[-1])
        for t in range(ap.timebins):
            pool.position = np.empty((0,2))
            pool.velocity = np.empty((0,2))
            pool.dfc = np.empty(0)
            pool.speed = np.array([])
            pool.orientation = np.array([])
            pool.angularvelocity = np.array([])
            pool.acceleration = np.empty(0)
            for traj in trajectories:
                traj.make_indices(traj.position,[ap.boundmin,ap.boundmax])
                length=len(traj.position)
                binsize=math.floor(length/ap.timebins)
                start = t*binsize
                end = (t+1)*binsize - 1
                indices = np.empty(length,dtype=bool)
                indices.fill(False)
                if ap.boundary:
                    traj.indices = ~traj.indices[:]

                indices[start:end] = traj.indices[start:end]
                pool.position = np.vstack((pool.position, 
                        traj.position[indices]))
                pool.dfc = np.append(pool.dfc, 
                        traj.dfc[indices])
                pool.speed = np.append(pool.speed, 
                        traj.speed[indices],axis=0)
                pool.velocity = np.vstack((pool.velocity,
                        traj.velocity[indices]))
                pool.orientation = np.append(pool.orientation, 
                        traj.orientation[indices],axis=0)
                pool.angularvelocity = np.append(pool.angularvelocity, 
                        traj.angularvelocity[indices],axis=0)
                pool.acceleration = np.append(pool.acceleration, 
                        traj.acceleration[indices],axis=0)

            #ap.plot_position_hist(pool.position,t+1)
            ap.plot_speed_hist(pool.speed,t+1)
            ap.plot_orientation_hist(pool.orientation,t+1)
            ap.plot_angularvelocity_hist(pool.angularvelocity,t+1)
            ap.plot_orient_vs_dist_from_center(pool.orientation,
                    pool.dfc,t+1)
            ap.plot_dfc_hist(pool.dfc,t+1)
            boo = pool.angularvelocity[(0.0<pool.speed) & (pool.speed<0.2)]
            ap.plot_angularvelocity_hist(boo,t+1000)
            ap.plot_acceleration_hist(pool.acceleration,t+1)
            ap.plot_velocity(pool.velocity,t+1)
            ap.blyat = (pool.speed<0.2) #& (
            #        np.abs(pool.angularvelocity < 0.0175))
            ap.plot_blyat(pool.position, 1000)
#        pool.displacement = np.empty((2,18014))
#        for traj in trajectories:
#            pool.displacement = np.append(pool.displacement,traj.displacement,axis=1)
#        ap.plot_displacement(pool.displacement,0)
    
    logfile = open('{}/vislog.txt'.format(ap.figpath),'w')
    lines = ['-----Plot settings-----\n',
            'timebins = {}\n'.format(ap.timebins),
            'plot type = {}\n'.format(ap.runtype),
            'jump = {}\n'.format(ap.jump),
            'roll = {}\n'.format(ap.roll),
            'plotres = {}\n'.format(ap.plotres),
            'boundary = {}\n'.format(ap.boundary),
            'boundmin = {}\n'.format(ap.boundmin),
            'boundmax = {}\n'.format(ap.boundmax),
            'size = {}\n'.format(ap.size),
            'fps = {}\n'.format(ap.fps),
            'density = {}\n'.format(ap.density),
            'source data file = {}'.format(ap.filename),'\n',
            '\n-----Plot fitting parameters-----\n',
            'Angular velocity plot:\n'
            'Distance from center plot:\n'
            ]
    logfile.writelines(lines)
    logfile.close()
    
    #beep = pool.angularvelocity[(pool.angularvelocity < 0.01) & (pool.angularvelocity > -0.0001)]

    print('Process exited normally.')

if __name__ == '__main__':
    main()

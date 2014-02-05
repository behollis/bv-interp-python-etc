# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

#use import numpypy as np #if using pypy interpreter
import numpy as np

import sys, struct
import rpy2.robjects as robjects
import random
import math as pm
import math

import matplotlib.mlab as mlab
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import operator
import gaussian_fit as gf
from scipy.optimize import leastsq
from scipy.stats import *
from peakfinder import *
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import StrVector

from scipy import stats
from scipy import linalg
from scipy import mat
from scipy import misc
from scipy import interpolate

from numpy import linspace,exp

from rpy2.robjects import numpy2ri
numpy2ri.activate()

from netcdf_reader import NetcdfReader

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

 
from skimage import data
from skimage import measure

r = robjects.r

SAMPLES = 100
        
if __name__ == '__main__':
    #dist1
    mean0 = -3;stdev0 = 1.5
    mean1 = 3;stdev1 = 1.0
    dist0 = r.rnorm(SAMPLES, mean = mean0, sd = stdev0) 
    dist1 = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    
    #points0 = []; points1 = []
    #for s0,s1 in dist0, dist1:
    #    points0.append([s0,mean0])
    #    points1.append([s1,mean1])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #subplot(211)
        
    mean0_list = [mean0]*SAMPLES
    mean1_list = [mean1]*SAMPLES
    
    x = linspace(-8,8,100)
    #y1 = mlab.normpdf( x, mean0, stdev0)
    #y2 = mlab.normpdf( x, mean1, stdev1)
    plt.plot(mean0_list, dist0, 'b.')
    plt.plot(mean1_list, dist1, 'g.')
    
    plt.grid(True)
    
    for idx in range(0,len(dist0)):
        plt.plot([ 0, 1], [dist0[idx], dist1[idx]], color='Black', linewidth=0.5, alpha = 0.5)
        plt.plot([ 0 ], [ dist0[idx] ], '.', color='Blue', alpha = 0.2)
        plt.plot([ 1 ], [ dist1[idx] ], '.', color='Green', alpha = 0.2)
       
    
    ax.set_xlim(-0.1, +1.1)
    ax.set_xlabel('alpha')
    ax.set_ylabel('Y')
    plt.show()
    
    ax2 = fig.add_subplot(111)
    
    for a in [0.2, .4, .6, .8]:
        isamp = (1.-a)*np.asarray(dist0) + a*np.asarray(dist1)
        
        imean = np.mean(isamp)
        ivar = np.var(isamp)
        
        interpolant = mlab.normpdf(x, imean, np.sqrt(ivar))
        plt.plot(x, interpolant, '--', color='Black', linewidth=1)
    
    x = linspace(-8,8,100)
    y1 = mlab.normpdf( x, mean0, stdev0)
    y2 = mlab.normpdf( x, mean1, stdev1)
    plt.plot(x, y1, color='Blue', linewidth=2)
    plt.plot(x, y2, color='Green', linewidth=2)
    ax2.set_xlim(-8, +8)
    ax2.set_ylim(0, 0.61)
    
    plt.show()
       
        
    
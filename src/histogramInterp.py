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

SAMPLES = 10000000
        
if __name__ == '__main__':
    
    x = linspace(-8,8,1000)
    
    #dist1
    mean0 = -3;stdev0 = 1.5
    mean1 = 3;stdev1 = 1.0
    dist1 = r.rnorm(SAMPLES, mean = mean0, sd = stdev0) 
    dist2 = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    
    bins = np.r_[-8:+8:70j]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # histogram our data with numpy
    n1, bins1 = np.histogram(dist1, bins = bins, normed = True)

    g0 = mlab.normpdf( x, mean0, stdev0 )
    plt.plot(x, g0, '-', color='blue', linewidth=2)
    
    g1 = mlab.normpdf( x, mean1, stdev1 )
    plt.plot(x, g1, '-', color='green', linewidth=2)
    
    '''
    # get the corners of the rectangles for the histogram
    left = np.array(bins1[:-1])
    right = np.array(bins1[1:])
    bottom = np.zeros(len(left))
    top = bottom + n1
    
    
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T
    
    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)
    
    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=1.0)
    ax.add_patch(patch)
    '''
    
    ############ second histogram ##############
    
    ax = fig.add_subplot(111)
    
    # histogram our data with numpy
    n2, bins2 = np.histogram(dist2, bins = bins, normed = True)
    
    
    # take alpha to be 0.5
    n_interp = 0.5*np.asarray(n1) + 0.5*np.asarray(n2)
    
    n3_array = []
    for idx in range(0,len(bins)-1):
        #val = np.mean(bins[idx], bins[idx+1])
        for count in range(0,int(n_interp[idx]*100)):
            n3_array.append(bins[idx])
    
    # GMM method
    r.library('mixtools')
    mixmdl = r.normalmixEM(robjects.vectors.FloatVector(n3_array),k=2,maxit = 5000, maxrestarts=5000)
    
    mu = []
    sd = []
    lb = []
    for i in mixmdl.iteritems():
        if i[0] == 'mu':
            mu.append(i[1])
        if i[0] == 'sigma':
            sd.append(i[1])
        if i[0] == 'lambda':
            lb.append(i[1])
            
    g1_interp = mlab.normpdf( x, mu[0][0], sd[0][0] )
    g2_interp = mlab.normpdf( x, mu[0][1], sd[0][1] )
    plt.plot(x, lb[0][0]*g1_interp + lb[0][1]*g2_interp, '--', color='black', linewidth=1)
    
    '''
    # get the corners of the rectangles for the histogram
    left = np.array(bins2[:-1])
    right = np.array(bins2[1:])
    bottom = np.zeros(len(left))
    top = bottom + n2
    
    
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T
    
    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)
    
    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='green', edgecolor='gray', alpha=1.0)
    ax.add_patch(patch)
    
    # update the view limits
    #ax.set_xlim(-7, +7)
    #ax.set_ylim(bottom.min(), top.max())
    '''
    
    
    
    ## histogram our data with numpy
    #n3, bins3 = np.histogram(n_interp, bins = bins)
    
    '''
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n_interp
    
    
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T
    
    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)
    
    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='Black', edgecolor='gray', alpha=1.0)
    ax.add_patch(patch)
    '''
    
    # update the view limits
    ax.set_xlim(-8, +8)
    ax.set_ylim(0,0.5)
    
    
    plt.show()
       
        
    
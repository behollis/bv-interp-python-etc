'''
    Author: Brad Hollister.
    Started: 11/11/2012.
    Code performs density matrix interpolation for velocities in ensemble.
'''

import numpy as np

import sys, struct
import rpy2.robjects as robjects
import random
import math as pm

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import operator

from numpy import linspace,exp
from numpy.random import randn
from scipy.interpolate import UnivariateSpline
from scipy import interpolate

r = robjects.r

from fractions import Fraction

#project modules
from spline_cdf_curve_morphing import spread

from scipy import stats
from scipy import linalg
from scipy import mat
from scipy import misc

from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure

def densityMatrix1(u_samples, v_samples):
    ''' takes a list of pure states, in this case possible velocities'''
    dMat = [[0,0],[0,0]]
    
    #SAMPLES = 100
    for idx in range(0,len(u_samples)):
        #print  mat([u_samples[idx], v_samples[idx]])
        #A = ( mat([u_samples[idx], v_samples[idx]]).transpose() * mat([ u_samples[idx], v_samples[idx]] ) )
        #print A
        dMat += ( mat([u_samples[idx], v_samples[idx]]).transpose() * mat( [u_samples[idx], v_samples[idx] ]) ) * 1. / float( len(u_samples) )
        #print dMat
        
    return dMat

import numpy as np
from pylab import plot, show, grid

def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])
 
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    
    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts
    
def lerp(a, b, w):
    return a + (b - a) * w

if __name__ == '__main__':
    '''
    # Plot a unit circle centered at (+2, +3)
    pts = get_ellipse_coords(a=1.0, b=1.0, x=2, y=3,k=1./8)
    ax = plot(pts[:,0], pts[:,1])

    # Set the aspect ratio so it looks like a circle; add a grid as well
    ax[0].get_axes().set_aspect(1)
    grid('on')


    # Ellipse, with major axis length = 4, minor axis = 1, centered at (0,0)
    pts = get_ellipse_coords(a=4.0, b=1.0)
    ax = plot(pts[:,0], pts[:,1])

    # Rotate the above ellipse by 30 degrees and use only 11 points!
    pts = get_ellipse_coords(a=4.0, b=1.0, angle=30,k=1./36)
    ax = plot(pts[:,0], pts[:,1])

    # Use all the options and 721 points:
    pts = get_ellipse_coords(a=2.0, b=0.25, x=-4, y=-2, angle=250,k=2)
    ax = plot(pts[:,0], pts[:,1])

    show()   
    '''
    
    SAMPLES = 600
    
    mean1 = -2; stdev1 = 0.5
    mean2 = +2; stdev2 = 2.0
    mean3 = +4; stdev3 = 1.0
    mean4 = -4; stdev4 = 1.0
    
    gp0_dist_u0 = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    gp0_dist_v0 = r.rnorm(SAMPLES, mean = mean2, sd = stdev2)#r.rnorm(SAMPLES/2, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/2, mean = mean2, sd = stdev2) 
    gp1_dist_u1 = r.rnorm(SAMPLES, mean = mean3, sd = stdev3) 
    gp1_dist_v1 = r.rnorm(SAMPLES, mean = mean4, sd = stdev4)#r.rnorm(SAMPLES/2, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/2, mean = mean2, sd = stdev2) 
    
    gp0_prob_kernel_u0 = stats.gaussian_kde(gp0_dist_u0)
    gp0_prob_kernel_v0 = stats.gaussian_kde(gp0_dist_v0)
    gp1_prob_kernel_u1 = stats.gaussian_kde(gp1_dist_u1)
    gp1_prob_kernel_v1 = stats.gaussian_kde(gp1_dist_v1)
        
    gp0_dMat = densityMatrix1(np.asarray(gp0_dist_u0), np.asarray(gp0_dist_v0)) #densityMatrix(gp0_prob)
    gp1_dMat = densityMatrix1(np.asarray(gp1_dist_u1), np.asarray(gp1_dist_v1)) #densityMatrix(gp1_prob)
        
    evals0, evecs0 = np.linalg.eig(gp0_dMat)
    evals1, evecs1 = np.linalg.eig(gp1_dMat)
    
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
    
    for alpha in interp_dist:
        
        lerpDMat = lerp(gp0_dMat, gp1_dMat, alpha)
        print lerpDMat
        
        evals, evecs = np.linalg.eig(lerpDMat)
        
        print "vectors:" + str(evecs)
        
        evecs_list = np.matrix.tolist(evecs.T)
        
        plt.figure()
        # Plot a unit circle centered at (+2, +3)
        plt.title( str(alpha) )
        pts = get_ellipse_coords(a=np.sqrt(np.inner(np.abs(evecs_list[0][:]),np.abs(evecs_list[0][:]))), b=np.sqrt(np.inner(np.abs(evecs_list[1][:]),np.abs(evecs_list[0][:]))), x=0, y=0)
        plt.plot(pts[:,0], pts[:,1])
        plt.plot([[0.,0.],[evecs_list[0][0], evecs_list[0][1]]],'--k', color='black')
        #plt.plot([[0.,0.],[evecs_list[1][0], evecs_list[1][1]]],'--k', color='blue')
        plt.savefig("./png/" + str(alpha) + "_" + "ellipsoid" + ".png")
         
        # Set the aspect ratio so it looks like a circle; add a grid as well
        #ax[0].get_axes().set_aspect(1)
        #grid('on')
        
        plt.show()  
    
            
        
        
       
    
    
        
     
        
    
    
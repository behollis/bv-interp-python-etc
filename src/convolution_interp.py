'''
    Author: Brad Hollister.
    Started: 11/15/2012.
    Code performs convolution interp.
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

def lerp(a, b, w):
    return (1-w) * a + w * b 
    #return a + (b - a) * w
    
def convolveLerp(a,b,w):
    return np.convolve((1-w) * a, w * b, mode='same') 

if __name__ == '__main__':
    SAMPLES = 600
    
    mean1 = 0; stdev1 = 1.0
    mean2 = +5; stdev2 = 0.5
    
    gp0_dist = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    gp1_dist = r.rnorm(SAMPLES, mean = mean2, sd = stdev2)#r.rnorm(SAMPLES/2, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/2, mean = mean2, sd = stdev2) 
    
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html?highlight=density%20matrix
    
    gp0_prob_kernel = stats.gaussian_kde(gp0_dist)
    gp1_prob_kernel = stats.gaussian_kde(gp1_dist)
    
    x = linspace(-10,10,100)
    
    # get sequences of kernel function
    gp0_seq = gp0_prob_kernel(x)
    gp1_seq = gp1_prob_kernel(x)
    
    fig = plt.figure()
    plt.title("check for kde eval")
    plt.plot(x,gp0_seq,'-')
    plt.plot(x,gp1_seq,'-')
    plt.savefig("./png/" + "kde_of_samples.png")
    plt.show()
    
    ''' 
    fig = plt.figure()
    plt.title("check for arrays")
    gp0_vals = []
    for each in gp0_prob.values():
        gp0_vals.append(each[0])
    gp1_vals = []
    for each in gp1_prob.values():
        gp1_vals.append(each[0])
        
    plt.plot(gp0_vals,gp0_prob.keys(),'.')
    plt.plot(gp1_vals,gp1_prob.keys(),'.')
    plt.show()
    '''
    
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
    
    for alpha in interp_dist:
        
        interp_seq = convolveLerp(gp0_seq, gp1_seq, alpha)
        
        x = linspace(-10,10,100)
         
        fig = plt.figure()
        plt.title(alpha)
        plt.plot(x,interp_seq,'-')
        plt.savefig("./png/" + str(alpha) + "_convolve_lerping.png") 
        plt.show()
        
        
            
        
        
       
    
    
        
     
        
    
    
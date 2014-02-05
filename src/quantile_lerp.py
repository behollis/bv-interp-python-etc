'''
    Author: Brad Hollister.
    Started: 1/8/2013.
    Code performs linear interp of histograms via quantiles.
'''

#use import numpypy as np #if using pypy interpreter
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

from scipy import stats
from scipy import linalg
from scipy import mat
from scipy import misc

from netcdf_reader import NetcdfReader
from peakfinder import *
from mayavi.mlab import *

import csv

r = robjects.r

from fractions import Fraction
'''
http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#plot3d
'''
def spread(start, end, count, mode=1):
    """spread(start, end, count [, mode]) -> generator

    Yield a sequence of evenly-spaced numbers between start and end.

    The range start...end is divided into count evenly-spaced (or as close to
    evenly-spaced as possible) intervals. The end-points of each interval are
    then yielded, optionally including or excluding start and end themselves.
    By default, start is included and end is excluded.

    For example, with start=0, end=2.1 and count=3, the range is divided into
    three intervals:

        (0.0)-----(0.7)-----(1.4)-----(2.1)

    resulting in:

        >>> list(spread(0.0, 2.1, 3))
        [0.0, 0.7, 1.4]

    Optional argument mode controls whether spread() includes the start and
    end values. mode must be an int. Bit zero of mode controls whether start
    is included (on) or excluded (off); bit one does the same for end. Hence:

        0 -> open interval (start and end both excluded)
        1 -> half-open (start included, end excluded)
        2 -> half open (start excluded, end included)
        3 -> closed (start and end both included)

    By default, mode=1 and only start is included in the output.

    (Note: depending on mode, the number of values returned can be count,
    count-1 or count+1.)
    """
    if not isinstance(mode, int):
        raise TypeError('mode must be an int')
    if count != int(count):
        raise ValueError('count must be an integer')
    if count <= 0:
        raise ValueError('count must be positive')
    if mode & 1:
        yield start
    width = Fraction(end-start)
    start = Fraction(start)
    for i in range(1, count):
        yield float(start + i*width/count)
    if mode & 2:
        yield end

def lerp(a, b, w):
    return a + (b - a) * w

def quantileLerp(f1, f2, x1, x2, alpha):
    a = 1.0 - alpha
    b = alpha
    return ( f1(x1) * f2(x2) ) / ( a*f2(x2) + b*f1(x1) )  


if __name__ == '__main__':
    
    FILE_NAME = 'pe_dif_sep2_98.nc' 
    REL_FILE_DIR = '/home/behollis/netcdf/'

    COM =  2
    LON = 53
    LAT = 90
    LEV = 16
    MEM = 600
    
    #realizations file 
    pe_dif_sep2_98_file = REL_FILE_DIR + FILE_NAME
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')
    
    #SAMPLES = 6000
    KNOTS_CONTROL_POINTS = 50
    
    percentiles = list(spread(0, 1.0, KNOTS_CONTROL_POINTS-1, mode=3)) #[0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.99]
    #percentiles_ext_lower = list(spread(0.0, 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, KNOTS_CONTROL_POINTS-1, mode=3))
    #percentiles_ext_upper = list(spread(0.9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999, 1.0, KNOTS_CONTROL_POINTS-1, mode=3))
    #percentiles.extend(percentiles_ext_upper)
    #percentiles.extend(percentiles_ext_lower) 
    percentiles.sort()
    
    KNOTS_CONTROL_POINTS = len(percentiles)
    
    #QUANTILES_TO_LERP = 1000
    #percentiles = list(spread(0.0, 1.0, QUANTILES_TO_LERP, mode=3)) 
    
    #values of cdf at percentiles
    percentiles_uni = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    percentiles_bi = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    
    #redefine with netcdf data...
    gp1 = []
    gp2 = []
    
    
    #c = csv.writer(open("./ocean.csv", "wb"))

    #Now we will apply the method to write a row. Writerow The method takes one argument - this
    #argument must be a list and each list item is equivalent to a column.
    #Here, we try to make an address book:
    for idx in range(0,600):
        gp1.append(vclin[idx][41][12][0][0])
        gp2.append(vclin[idx][40][12][0][0])
        
    #c.writerow(gp1)
    #c.writerow(gp2)
        
    #print gp1
        
    a_uni = np.asarray(gp1)
    a_bi = np.asarray(gp2)
    
    #print a_uni
    
    np.random.shuffle(a_uni)
    np.random.shuffle(a_bi)
    
    #print a_uni
        
    
    gp0_u_kd = stats.gaussian_kde(a_uni)
    gp1_u_kd = stats.gaussian_kde(a_bi)
    
    x = linspace(-3,3,2000)
    
    '''
    _max0, _min0 = peakdetect(gp0_u_kd(x),x,lookahead=2,delta=0)
    _max1, _min1 = peakdetect(gp1_u_kd(x),x,lookahead=2,delta=0)
    
    xm0 = [p[0] for p in _max0]
    ym0 = [p[1] for p in _max0]
    xn0 = [p[0] for p in _min0]
    yn0 = [p[1] for p in _min0]
    
    xm1 = [p[0] for p in _max1]
    ym1 = [p[1] for p in _max1]
    xn1 = [p[0] for p in _min1]
    yn1 = [p[1] for p in _min1]
    
    plt.figure()
    plt.title("kde's @ grid points")
    p1, = plt.plot(x,gp0_u_kd(x),'-', color='red')
    p2, = plt.plot(x,gp1_u_kd(x),'-', color='blue')
   
    #plot peaks
    plt.hold(True)
    plt.plot(xm0, ym0, 'o', color='orange')
    plt.plot(xn0, yn0, 'o', color='black')
  
    #plot peaks
    plt.hold(True)
    plt.plot(xm1, ym1, 'o', color='orange')
    plt.plot(xn1, yn1, 'o', color='black')
   
    plt.legend([p2, p1], ["gp0", "gp1"])
    plt.savefig("../png/" + "gridpoint_kdes.png")
    plt.show()
    '''
    
        
    #find values of percentiles for pdf and use those as control points for cdf spline approx.
    for per in percentiles:
        if (per == 0.0):
            x = r.quantile(robjects.FloatVector(gp1), per, type = 8)[0]
            #need to correct for differences between R's quantile function and scipy's kde
            #solution: add more values above and below quantile 0th / 1.0
            percentiles_uni.append(x-1.0)
            percentiles_uni.append(x-0.75)
            percentiles_uni.append(x-0.5)
            percentiles_uni.append(x-0.25)
         
        percentiles_uni.append(r.quantile(robjects.FloatVector(gp1), per, type = 8)[0])
        
        if (per == 1.0):
            x = r.quantile(robjects.FloatVector(gp1), per, type = 8)[0]
            percentiles_uni.append(x+0.25)
            percentiles_uni.append(x+0.5)
            percentiles_uni.append(x+0.75)
            percentiles_uni.append(x+1.0)
        
        if (per == 0.0):
            x = r.quantile(robjects.FloatVector(gp2), per, type = 8)[0]
            percentiles_bi.append(x-1.0)
            percentiles_bi.append(x-0.75)
            percentiles_bi.append(x-0.5)
            percentiles_bi.append(x-0.25)
        
        percentiles_bi.append(r.quantile(robjects.FloatVector(gp2), per, type = 8)[0])
        
        if (per == 1.0):
            x = r.quantile(robjects.FloatVector(gp2), per, type = 8)[0]
            percentiles_bi.append(x+0.25)
            percentiles_bi.append(x+0.5)
            percentiles_bi.append(x+0.75)
            percentiles_bi.append(x+1.0)
             
    
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
       
    for alpha in interp_dist:
        per_uni_array = np.asarray(percentiles_uni)
        per_bi_array = np.asarray(percentiles_bi)
        
        ensemble_lerp = lerp(np.asarray(gp1), np.asarray(gp2), alpha)
        lerped_quantiles = lerp(per_uni_array, per_bi_array, alpha)
        lerped_prob_values = quantileLerp(gp0_u_kd, gp1_u_kd, per_uni_array, per_bi_array, alpha)
        
        #find peaks...
        _max, _min = peakdetect(lerped_prob_values, lerped_quantiles,lookahead=2, delta=0.001)
        xm = [p[0] for p in _max]
        ym = [p[1] for p in _max]
        xn = [p[0] for p in _min]
        yn = [p[1] for p in _min]
        
       
        
        plt.figure()
            
        plt.title("quantile lerp using pdf, alpha " + str(alpha) )
        p1, = plt.plot(lerped_quantiles,lerped_prob_values,'-',color='blue')
        
        x = linspace(-3,3,1000)
        kde = stats.gaussian_kde(ensemble_lerp)
        p2, = plt.plot(x,kde(x),'-',color='green')
        
        _max2, _min2 = peakdetect(kde(x),x ,lookahead=2, delta=0.001)
        xm2 = [p[0] for p in _max2]
        ym2 = [p[1] for p in _max2]
        xn2 = [p[0] for p in _min2]
        yn2 = [p[1] for p in _min2]
        
        x1,x2,y1,y2 = plt.axis()
        plt.axis((-3,3,0,y2))
        
        #plot peaks
        plt.hold(True)
        plt.plot(xm, ym, '+', color='red')
        plt.plot(xm2, ym2, '+', color='orange')
        plt.savefig("../png/" + str(alpha) + "ensemble_direct_pdf_lerp.png")
        #plt.legend([p2, p1], ["ensemble lerp", "quantile lerp"])
        plt.show()
        
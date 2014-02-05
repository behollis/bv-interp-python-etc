'''
    Author: Brad Hollister.
    Started: 11/5/2012.
    Code performs spline approx of cdf's and curve morphing by control point euclidean interpolation.
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

if __name__ == '__main__':
    
    '''
    i = 10000
    x = np.linspace(0,3.7*pi,i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 *
    np.random.randn(i))
    y *= -1
    
    _max, _min = peakdetect(y, x, 750, 0.30)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    xn = [p[0] for p in _min]
    yn = [p[1] for p in _min]
    
    plot = pylab.plot(x, y)
    pylab.hold(True)
    pylab.plot(xm, ym, 'r+')
    pylab.plot(xn, yn, 'g+')
    
    
    pylab.show()
    '''
    
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
    
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html
    #http://bespokeblog.wordpress.com/2011/07/07/basic-data-plotting-with-matplotlib-part-2-lines-points-formatting/
    #plt.plot(radius, square, marker='o', linestyle='--', color='r')
    
    SAMPLES = 6000
    KNOTS_CONTROL_POINTS = 1000
    
    percentiles = list(spread(0.01, 0.99, KNOTS_CONTROL_POINTS-1, mode=3)) #[0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.99]
    
    percentiles_ext_mid = list(spread(0.4, 0.6, KNOTS_CONTROL_POINTS-1, mode=3)) #[0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.99]
   
    percentiles_ext_lower = list(spread(0.001, 0.01, KNOTS_CONTROL_POINTS-1, mode=3))
    percentiles_ext_lower2 = list(spread(0.0, 0.001, KNOTS_CONTROL_POINTS-1, mode=3))
    percentiles_ext_upper = list(spread(0.99, 0.999, KNOTS_CONTROL_POINTS-1, mode=3))
    percentiles_ext_upper2 = list(spread(0.999, 1.0, KNOTS_CONTROL_POINTS-1, mode=3))
    
    percentiles.extend(percentiles_ext_mid)
    percentiles.extend(percentiles_ext_upper)
    percentiles.extend([1.0, 1.0, 1.0,1.0,0.9995, 0.9995, 0.9995,0.9995])
    percentiles.extend([0., 0., 0., 0.,0.005, 0.005, 0.005, 0.005])
    percentiles.extend(percentiles_ext_lower) 
    percentiles.extend(percentiles_ext_upper2) 
    percentiles.sort()
    
    KNOTS_CONTROL_POINTS = len(percentiles)
    
    
    #values of cdf at percentiles
    percentiles_uni = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    percentiles_bi = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    percentiles_quad = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    
    mean1 = -2; stdev1 = 0.5
    mean2 = +2; stdev2 = 0.5
    mean3 = -4; stdev3 = 1.0
    mean4 = +4; stdev4 = 1.0
    
    unimodal0 = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    bimodal0 = r.rnorm(SAMPLES/2, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/2, mean = mean2, sd = stdev2)
    #quadmodal = r.rnorm(SAMPLES/4, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/4, mean = mean2, sd = stdev2) + r.rnorm(SAMPLES/4, mean = mean3, sd = stdev3) + r.rnorm(SAMPLES/4, mean = mean4, sd = stdev4)
    
    #redefine with netcdf data...
    unimodal1 = []
    bimodal1 = []
    
    for idx in range(0,600):
        unimodal1.append(vclin[idx][41][12][0][0])
        bimodal1.append(vclin[idx][40][12][0][0])
        
    print unimodal1
        
    a_uni = np.asarray(unimodal1)
    a_bi = np.asarray(bimodal1)
    
    print a_uni
    
    np.random.shuffle(a_uni)
    np.random.shuffle(a_bi)
    
    print a_uni
        
    gp0_u_kd = stats.gaussian_kde(a_uni)
    gp1_u_kd = stats.gaussian_kde(a_bi)
    
    x = linspace(-3,3,2000)
    
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
    #plt.plot(unimodal.sort(),'.',color='black')
    #plt.plot(bimodal.sort(),'.', color='purple')
    
    #plot peaks
    plt.hold(True)
    plt.plot(xm0, ym0, 'o', color='orange')
    plt.plot(xn0, yn0, 'o', color='black')
  
    #plot peaks
    plt.hold(True)
    plt.plot(xm1, ym1, 'o', color='orange')
    plt.plot(xn1, yn1, 'o', color='black')
   
    plt.legend([p2, p1], ["gp0", "gp1"])
    plt.savefig("./png/" + "gridpoint_kdes.png")
    plt.show()
    
    unimodal2 = gp0_u_kd.resample(10000)[0]
    bimodal2 = gp1_u_kd.resample(10000)[0]
      
    unimodal1.sort();list(bimodal1).sort()
    unimodal2.sort();list(bimodal2).sort()
    
    #np.random.shuffle(unimodal1);np.random.shuffle(bimodal1)
    #np.random.shuffle(unimodal2);np.random.shuffle(bimodal2)
    
    print unimodal1
    print bimodal1
    print unimodal2
    print bimodal2
        
    #find values of percentiles for pdf and use those as control points for cdf spline approx.
    percentiles_uni2 = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    percentiles_bi2 = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
    for per in percentiles:
        percentiles_uni2.append(r.quantile(robjects.FloatVector(unimodal2), per)[0])
        percentiles_bi2.append(r.quantile(robjects.FloatVector(bimodal2), per)[0])
        #percentiles_quad.append(r.quantile(quadmodal, per)[0])

    print percentiles_uni2
    print percentiles_bi2
    
    for per in percentiles:
        percentiles_uni.append(r.quantile(robjects.FloatVector(unimodal1), per)[0])
        percentiles_bi.append(r.quantile(robjects.FloatVector(bimodal1), per)[0])
        #percentiles_quad.append(r.quantile(quadmodal, per)[0])
        
    print percentiles_uni
    print percentiles_bi
    
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
    
    uni_a2 = np.asarray(unimodal2)
    bi_a2 = np.asarray(bimodal2)
    
    for randomized_run in range(0, 10):
        
        np.random.shuffle(uni_a2)
        np.random.shuffle(bi_a2)
       
        for alpha in interp_dist:
            per_uni_array = np.asarray(percentiles_uni2)
            per_bi_array = np.asarray(percentiles_bi2)
            
            out = lerp( per_uni_array, per_bi_array, alpha )
            
            #np.random.shuffle(unimodal1);np.random.shuffle(bimodal1)
            #np.random.shuffle(unimodal2);np.random.shuffle(bimodal2)
            
            kern_out = lerp(uni_a2, bi_a2, alpha)
            
            k = stats.gaussian_kde(kern_out)
            
            x = linspace(out[0], out[-1], 3000)    
            
            #find peaks...
            _maxk, _mink = peakdetect(k(x), x, lookahead=2, delta=0)
            xmk = [p[0] for p in _maxk]
            ymk = [p[1] for p in _maxk]
            xnk = [p[0] for p in _mink]
            ynk = [p[1] for p in _mink]
            
            percentiles_narray = np.asarray(percentiles)
            
            tck1 = interpolate.splrep(out,percentiles_narray,k=3,s=0.0029,per=0)
            
            y1_out = interpolate.splev(out,tck1,der=0,ext=1)
            #yder1 = interpolate.splev(out,tck1,der=1,ext=1)
            
            y1_out2 = interpolate.splev(x,tck1,der=1,ext=1)
            y1_out3 = interpolate.splev(x,tck1,der=0,ext=1)
            
           
            
            #find peaks...
            _max, _min = peakdetect(y1_out2, x, lookahead=2, delta=0)
            xm = [p[0] for p in _max]
            ym = [p[1] for p in _max]
            xn = [p[0] for p in _min]
            yn = [p[1] for p in _min]
            
           
            plt.figure()
                
            plt.title("pairings run: " + str(randomized_run) + " number of knots " + str(KNOTS_CONTROL_POINTS)  + " alpha " + str(alpha) )
            plt.plot(x,y1_out3,'-',color='blue')
            plt.plot(out,y1_out, '.', color='red')
            plt.axis((-3,3,0,1.0))
            
            
            
            plt.savefig("./png/" + str(alpha) + "_" + str(randomized_run) + "_" + "interp_cdf2" + ".png")
            #plt.show()
            
            #plt.plot(out,yder1, '-', color='green')
            #plt.figure()
            xc = linspace(-3,3,6000)
            plt.figure()
            p1, = plt.plot(xc,k(xc),'-', color='black')
            
            plt.title("pairings run: " + str(randomized_run) + " number of knots " + str(KNOTS_CONTROL_POINTS)  + " alpha " + str(alpha) )
            p2, = plt.plot(x,y1_out2, '-', color='orange')
            x1,x2,y1,y2 = plt.axis()
            plt.axis((-3,3,0,y2))
            plt.legend([p2, p1], ["quantile interp", "ensemble interp"])
            
            #plot peaks
            pylab.hold(True)
            pylab.plot(xm, ym, 'x', color='red')
            #pylab.plot(xn, yn, 'o', color='blue')
            
            pylab.hold(True)
            pylab.plot(xmk, ymk, 'x', color='red')
            #pylab.plot(xnk, ynk, 'o', color='blue')
            plt.savefig("./png/" + str(alpha) + "_" + str(randomized_run) + "_" + "quantile_interp2" + ".png")
            #plt.show()
     
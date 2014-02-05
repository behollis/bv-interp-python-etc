'''
    Author: Brad Hollister.
    Started: 11/25/2012.
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

from scipy import fftpack

from netcdf_reader import NetcdfReader

r = robjects.r

from fractions import Fraction

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
    
    #redefine with netcdf data...
    unimodal1 = []
    bimodal1 = []
    
    for idx in range(0,600):
        unimodal1.append(vclin[idx][41][12][0][0])
        bimodal1.append(vclin[idx][40][12][0][0])
        
    gp0_u_kd = stats.gaussian_kde(unimodal1)
    gp1_u_kd = stats.gaussian_kde(bimodal1)
    
    x = linspace(-3,3,600)
    plt.figure()
    plt.title("kde's @ grid points")
    p1, = plt.plot(x,gp0_u_kd(x),'-', color='red')
    p2, = plt.plot(x,gp1_u_kd(x),'-', color='blue')
    plt.legend([p2, p1], ["gp0", "gp1"])
    plt.savefig("./png/" + "gridpoint_kdes.png")
    plt.show()
    
    samp = 5000
    
    unimodal2 = gp0_u_kd.resample(samp)[0]
    bimodal2 = gp1_u_kd.resample(samp)[0]
    
    print unimodal1
    print bimodal1
    print unimodal2
    print bimodal2
    
    '''
    gp0_freq = fftpack.rfft(unimodal2, n=None, axis=-1, overwrite_x=0)
    gp1_freq = fftpack.rfft(bimodal2, n=None, axis=-1, overwrite_x=0)
    
    samp_d = samp / 6.0
    
    gp0_freq_bins = fftpack.rfftfreq(samp, d=samp_d)
    gp1_freq_bins = fftpack.rfftfreq(samp, d=samp_d)
    
    #plot freq bins
    x = linspace(-3,3,samp)
    plt.figure()
    plt.title("grid point frequencies")
    p1, = plt.plot(gp0_freq_bins,gp0_freq)
    
    plt.legend([p1], ["gp1"])
    plt.savefig("./png/" + "freq_bins.png")
    plt.show()
    '''
    
     
    #gp0
    t = linspace(-3,3,6000)
    #acc = lambda t: 10*scipy.sin(2*pi*2.0*t) + 5*scipy.sin(2*pi*8.0*t) + 2*scipy.random.random(len(t))

    signal = gp0_u_kd(t)
    #signal = np.sin(t)

    plt.subplot(611)
    plt.title('original')
    plt.plot(t,signal,color='red')
    
    FFT = abs(fftpack.fft(signal))
   
    freqs = fftpack.fftfreq(signal.size, t[1]-t[0])
    
    
    plt.subplot(612)
    plt.title('fft')
    plt.plot(freqs,np.log10(FFT),'.',color='red')
    #plt.plot(freqs,FFT,'x')
    #plt.show()
    
    iFFT = fftpack.fft(FFT)
    plt.subplot(613)
    plt.title('inverse fft')
    plt.plot(t,iFFT,'-',color='red')
    #plt.plot(freqs,FFT,'x')
    #plt.show()
    
    #gp1
    t = linspace(-3,3,6000)
    #acc = lambda t: 10*scipy.sin(2*pi*2.0*t) + 5*scipy.sin(2*pi*8.0*t) + 2*scipy.random.random(len(t))

    signal = gp1_u_kd(t)

    plt.subplot(614)
    plt.title('original')
    plt.plot(t,signal,color='blue')
    
    FFT = abs(fftpack.fft(signal))
    freqs = fftpack.fftfreq(signal.size, t[1]-t[0])
    
    plt.subplot(615)
    plt.title('fft')
    plt.plot(freqs,np.log10(FFT),'.',color='blue')
    #plt.plot(freqs,FFT,'x')
    #plt.show()
    
    iFFT = fftpack.ifft(FFT)
    plt.subplot(616)
    plt.title('inverse fft')
    plt.plot(t,iFFT,'-',color='blue')
    #plt.plot(freqs,FFT,'x')
    plt.show()
    
    
    
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
    
'''    
    for alpha in interp_dist:
        
        
        
        per_uni_array = np.asarray(percentiles_uni2)
        per_bi_array = np.asarray(percentiles_bi2)
        
        out = lerp( per_uni_array, per_bi_array, alpha )
        
        kern_out = lerp(np.asarray(unimodal2), np.asarray(bimodal2), alpha)
        
        k = stats.gaussian_kde(kern_out)
        
        
        
        percentiles_narray = np.asarray(percentiles)
        
        tck1 = interpolate.splrep(out,percentiles_narray,k=3,s=0.0005,per=0)
        
        y1_out = interpolate.splev(out,tck1,der=0,ext=1)
        yder1 = interpolate.splev(out,tck1,der=1,ext=1)
        
        x = linspace(out[0], out[-1], 1000)    
        y1_out2 = interpolate.splev(x,tck1,der=1,ext=1)
        y1_out3 = interpolate.splev(x,tck1,der=0,ext=1)
        
        plt.figure()
        plt.title("number of knots " + str(KNOTS_CONTROL_POINTS)  + " alpha " + str(alpha) )
        plt.plot(x,y1_out3,'-',color='blue')
        plt.plot(out,y1_out, '.', color='red')
        plt.axis((-3,3,0,1.0))
        plt.savefig("./png/" + str(alpha) + "_" + "interp_cdf" + ".png")
        plt.show()
        
        #plt.plot(out,yder1, '-', color='green')
        #plt.figure()
        xc = linspace(-3,3,6000)
        plt.figure()
        p1, = plt.plot(xc,k(xc),'-', color='black')
        
        plt.title("number of knots " + str(KNOTS_CONTROL_POINTS)  + " alpha " + str(alpha) )
        p2, = plt.plot(x,y1_out2, '-', color='orange')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((-3,3,0,y2))
        plt.legend([p2, p1], ["quantile interp", "ensemble interp"])
        plt.savefig("./png/" + str(alpha) + "_" + "quantile_interp" + ".png")
        plt.show()
'''
        
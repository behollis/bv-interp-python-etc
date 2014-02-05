
# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 10/31/2012.
    Code performs linear interpolation between mixture of gaussians by linearly interpolating parameters.
'''

'''
i.e. component by component interpolation, where "component" means
corresponding gaussians.  so, starting sog is made up of bunch of
gaussians.  ending sog is made up by another bunch of gaussians.
the total number of gaussians is the union of those 2 sets.  if
one is missing from the other sog, we add it there, but with initial
weight of 0.  the weights vary linearly, while the params vary linearly
too.
'''

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

#import numpy as np
#import matplotlib.pyplot as plt

r = robjects.r

#select between single gaussian fit and GMM model
ASSUME_NORM = False

INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/pics/skl/'

SAMPLES = 5000

#values of cdf at percentiles
percentiles_uni = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')
percentiles_bi = []#np.zeros(shape = KNOTS_CONTROL_POINTS, dtype = float, order = 'F')


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

def norm(sample_distribution):
    tot = 0.0
    for idx in range(len(sample_distribution)):
        tot += sample_distribution[idx]
    
    for idx in range(len(sample_distribution)):
        sample_distribution[idx] /= tot
        
    return sample_distribution

def uniformLerpMix(norm_params1, norm_params2, alpha):
    
    mins1 = np.ndarray(shape = len(norm_params1), dtype = float, order = 'F')
    mins2 = np.ndarray(shape = len(norm_params2), dtype = float, order = 'F')
    LOW_PER = 0.00001 
    #get 0.1 percentiles from gp1 sog
    for idx in range(0,len(norm_params1)):
        mins1[idx] = r.qnorm(LOW_PER,mean=norm_params1[idx][0], sd=norm_params1[idx][1])[0]
    for idx in range(0,len(norm_params2)):
        mins2[idx] = r.qnorm(LOW_PER,mean=norm_params2[idx][0], sd=norm_params2[idx][1])[0]
       
    mins_list = np.concatenate((mins1, mins2))
    smallest_min = mins_list[np.argmin(mins_list)]
    
    maxs1 = np.ndarray(shape = len(norm_params1), dtype = float, order = 'F')
    maxs2 = np.ndarray(shape = len(norm_params2), dtype = float, order = 'F')
    HIGH_PER = 0.99999
    #get 0.1 percentiles from gp2 sog
    for idx in range(0,len(norm_params1)):
        maxs1[idx] = r.qnorm(HIGH_PER,mean=norm_params1[idx][0], sd=norm_params1[idx][1])[0]
    for idx in range(0,len(norm_params2)):
        maxs2[idx] = r.qnorm(HIGH_PER,mean=norm_params2[idx][0], sd=norm_params2[idx][1])[0]
        
    maxs_list = np.concatenate((maxs1, maxs2))   
    largest_max = maxs_list[np.argmax(maxs_list)]
    
    SAMPLES = 100
    #uniform defining params, range and probability of all samples
    uniform_range = np.abs(largest_max - smallest_min)
    uniform_prob = 1.0 / uniform_range
    uniform_dist = [uniform_prob]*SAMPLES
    
    #set plot ranges, etc.
    xmin = smallest_min; xmax = largest_max; ymin=0.0; ymax = 0.1
    x = np.linspace(xmin,xmax,SAMPLES)
    v = [xmin, xmax, ymin, ymax]
    plt.axis(v)
    
    plt.title(str(alpha))
    
    #test to see if this agrees with cases below.
    if alpha == 0.5:
        #we have the uniform distribution between min / max of all end point distributions
        plt.plot(x,uniform_dist, color='black')
        plt.show() 

    total_dist = np.zeros(shape = SAMPLES, dtype = float, order = 'F')
    
    if alpha <= 0.5:
        for idx in range(0,len(norm_params1)):
            
            cur_sog = mlab.normpdf(x,norm_params1[idx][0],norm_params1[idx][1])
            
            cur_weighted_sog = [0.0]*SAMPLES
            for idx in range(len(cur_sog)):
                cur_weighted_sog[idx] = ((0.5 - alpha) / 0.5) * cur_sog[idx]  
            #plt.plot(x,cur_weighted_sog)
            
            cur_weighted_uniform = [uniform_prob]*SAMPLES
            for idx in range(len(cur_sog)):
                cur_weighted_uniform[idx] = (alpha / 0.5) * cur_weighted_uniform[idx]
                
            #plt.plot(x,cur_weighted_uniform)
            
            total_dist = np.add(total_dist, np.add(cur_weighted_uniform, cur_weighted_sog))
            
    elif alpha > 0.5:
        for idx in range(0,len(norm_params2)):
             
            cur_sog = mlab.normpdf(x,norm_params2[idx][0],norm_params2[idx][1])
            
            cur_weighted_sog = [0.0]*SAMPLES
            for idx in range(len(cur_sog)):
                cur_weighted_sog[idx] = (np.abs(alpha - 0.5) / 0.5) * cur_sog[idx]
                    
            #plt.plot(x,cur_weighted_sog)
            
            cur_weighted_uniform = [uniform_prob]*SAMPLES
            for idx in range(len(cur_sog)):
                cur_weighted_uniform[idx] = (1.0 - (np.abs(alpha - 0.5) / 0.5)) * cur_weighted_uniform[idx]
               
            #plt.plot(x,cur_weighted_uniform)
            
            total_dist = np.add(total_dist, np.add(cur_weighted_uniform, cur_weighted_sog))
            
    #plt.plot(x,total_dist, color="black")
    #plt.savefig("./png/" + str(alpha) + "_" + "uniform_method" + ".png")
    #plt.show()
    
    plt.title(str(alpha))    
    plt.plot(x,norm(total_dist), color="orange")  
    plt.savefig(OUTPUT_DATA_DIR + str(alpha) + "_" + "normalized_uniform_method" + ".png")      
    plt.show()
    
def kl_div(A,B,min_x=-5, max_x=5):
    "Calculates the KL divergence D(A||B) between the distributions A and B.\nUsage: div = kl_divergence(A,B)"
    D = .0
    i = min_x
    div = 1000
    incr = math.fabs(min_x - max_x) / div
    for steps in range(0,div,1):
        if A(i) != .0:
            #print A(i)
            D += A(i) * math.log( A(i) / B(i) ) 
            #print math.log( A(i) / B(i) )
        else:
            D+= B(i)
        i += incr
    return D 
      
def lerpMix(norm_params1, norm_params2, alpha, steps, method, num_gs, gp1_mean, gp1_var, gp2_mean, gp2_var):    
    ''' handles equal number of constituent gaussians '''
    
    sorted(norm_params2, key=operator.itemgetter(0), reverse=False)
    sorted(norm_params1, key=operator.itemgetter(0), reverse=False)
    
    ensemble_lerp = lerp(np.asarray(gp1), np.asarray(gp2), alpha)
   
    if steps != 0:  
        incr = alpha / steps
    else:
        incr = alpha
   
    for idx in range(0,steps+1):
        
        if method == 1: #resort to minimize distances in means of pairings
            sorted(norm_params1, key=operator.itemgetter(0), reverse=False)
        elif method == 2: #randominze to minimize errors in mispairings from "real" process
            random.shuffle(norm_params1)
            
        subalpha = float(idx) * incr
        
        inter_means = []; inter_stdevs = []; inter_comp_ratios = []
        
        max_comps = len(norm_params1)
        
        if max_comps < len(norm_params2):
            max_comps = len(norm_params2)
            
        total_dist_kde = None
        
        # interpolate each gaussian
        for idx in range(0,max_comps):
            '''
            # fill in null gaussians
            if idx > len(norm_params1) - 1:
                cur_mean1 = 0.0
                cur_std1 = 2.0
                cur_ratio1 = 0.0
            else:
            '''
            cur_mean1 = norm_params1[idx][0]
            cur_std1 = norm_params1[idx][1]
            cur_ratio1 = norm_params1[idx][2]
            '''    
            if idx > len(norm_params2) - 1:
                cur_mean2 = 0.0
                cur_std2 = 2.0
                cur_ratio2 = 0.0
            else:
            '''
            cur_mean2 = norm_params2[idx][0]
            cur_std2 = norm_params2[idx][1]
            cur_ratio2 = norm_params2[idx][2]
            
            inter_means.append(cur_mean1*(1.0-subalpha) + cur_mean2*subalpha)
            inter_stdevs.append(cur_std1*(1.0-subalpha) + cur_std2*subalpha)
            inter_comp_ratios.append(cur_ratio1*(1.0-subalpha) + cur_ratio2*subalpha)
            
        norm_params1 = []
        for j in range(len(inter_means)):    
            norm_params1.append((inter_means[j], inter_stdevs[j], inter_comp_ratios[j]))
        
        
    #if idx == steps:
        
    #set plot ranges, etc.
    xmin = -8; xmax = 8; ymin=0.0; ymax = 1.0
    
    #plot interp GMM 
    SAMPLES = 600
    total_dist = []
    for idx in range(0,len(inter_means)):
        cur_inter_mean = inter_means[idx];cur_inter_stdev = inter_stdevs[idx];cur_inter_ratio = inter_comp_ratios[idx] 
        total_dist += list(np.asarray(r.rnorm(int(SAMPLES*cur_inter_ratio), mean=cur_inter_mean, sd = cur_inter_stdev)))
    
    #norm(data1)
    x = np.linspace(-20,20,300)
    total_dist_kde = gaussian_kde(total_dist)
    _max_u, _min_u = peakdetect(total_dist_kde(x),x,lookahead=2,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    ym_u = [p[1] for p in _max_u]
   
    #print "peaks " + str(xm_u) 
    
    plt.figure()
    
    #plt.title("GMM / Quantile / Ensemble Lerp: " + str(subalpha))
    p1 = None
    if ASSUME_NORM == False:
        p1, = plt.plot(x,total_dist_kde(x), color='red')  
    else:
        p1, = plt.plot(x,mlab.normpdf(x,inter_means[0],inter_stdevs[0]), color='red')  
    #plt.plot(xm_u,ym_u,'x',color='red') 
    

    #save out plot for current interpolated distribution
    
    per_uni_array = np.asarray(percentiles_uni)
    per_bi_array = np.asarray(percentiles_bi)
    
    
    lerped_quantiles = lerp(per_uni_array, per_bi_array, alpha)
    lerped_prob_values = quantileLerp(gp0_u_kd, gp1_u_kd, per_uni_array, per_bi_array, alpha)
    
    #find peaks...
    _max, _min = peakdetect(lerped_prob_values, lerped_quantiles,lookahead=2, delta=0.001)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    xn = [p[0] for p in _min]
    yn = [p[1] for p in _min]

    
    
    
    #plt.title("quantile lerp using pdf, alpha " + str(alpha) )

    kde = stats.gaussian_kde(ensemble_lerp)
    p3, = plt.plot(x,kde(x),'-',color='green')
    
    QUANTILE_KDE_SAMPLE_NUMBER = 100
    samples_numbers = lerped_prob_values * QUANTILE_KDE_SAMPLE_NUMBER
    samples_lerp = []
    for prob_idx in range(0,len(lerped_prob_values)):
        #if not math.isnan(samples_numbers2[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers[prob_idx])):
            samples_lerp.append(lerped_quantiles[prob_idx])
      
    quantile_interp = interpolate.interp1d(lerped_quantiles,lerped_prob_values)
    quantile_kde = stats.gaussian_kde(samples_lerp)      
    #p4, = plt.plot(x, quantile_kde(x),'-',color='orange')
    #x = linspace(lerped_quantiles.sort()[0],lerped_quantiles[-1], 100)
    
    x = linspace(lerped_quantiles[0],lerped_quantiles[-1],100)
    #p2, = plt.plot(x, quantile_kde(x),'-',color='blue')
    p2, = plt.plot(lerped_quantiles,lerped_prob_values,'-',color='blue')
    
    #_max2, _min2 = peakdetect(kde(x),x ,lookahead=2, delta=0.001)
    _max2, _min2 = peakdetect(quantile_interp(x),x ,lookahead=2, delta=0.001)
    xm2 = [p[0] for p in _max2]
    ym2 = [p[1] for p in _max2]
    xn2 = [p[0] for p in _min2]
    yn2 = [p[1] for p in _min2]
    
    x1,x2,y1,y2 = plt.axis()
    y2 = 0.15
    plt.axis((-20,20,0,y2))
    
    grange = np.linspace(-20,20,300)
    im = gp1_mean*(1.0-alpha) + gp2_mean*alpha
    ivar = gp1_var*(1.0-alpha) + gp2_var*alpha
    p4, = plt.plot(grange,mlab.normpdf(grange,im,np.sqrt(ivar)), color='purple') 
    g_total_dist = list(np.asarray(r.rnorm(SAMPLES, mean=im, sd = np.sqrt(ivar))))
    
    
    #plot peaks
    #plt.hold(True)
    #plt.plot(xm, ym, '+', color='red')
    #plt.plot(xm2, ym2, '+', color='orange')
    #plt.savefig("../png/" + str(alpha) + "ensemble_direct_pdf_lerp.png")
    plt.legend([p1, p2, p3, p4], ["GMM", "Quantile", "Ensemble","Gaussian"])
    
    #if method == 1:
    plt.savefig(OUTPUT_DATA_DIR + str(subalpha) +"_" + str(num_gs) + "_entropy_test_rev3.png")
         
    #plt.show()
    
   
    _min = min(lerped_quantiles); _max = max(lerped_quantiles)
    print "_min: " + str(_min)
    print "_max: " + str(_max)
    gmm_e_entropy = kl_div(total_dist_kde , kde,max_x=_max,min_x=_min ) + kl_div(kde, total_dist_kde,max_x=_max,min_x=_min )
    q_e_entropy = kl_div( quantile_interp , kde,max_x=_max,min_x=_min ) + kl_div(kde, quantile_interp,max_x=_max,min_x=_min)
    g_kde = stats.gaussian_kde(g_total_dist)
    g_e_entropy = kl_div( g_kde , kde,max_x=_max,min_x=_min ) + kl_div(kde, g_kde,max_x=_max,min_x=_min)
    
    

    
    print 'gmm entropy ' + str(gmm_e_entropy)
    print 'quantile entropy ' + str(q_e_entropy)
    
    return gmm_e_entropy, q_e_entropy, g_e_entropy 
        
    #####
    '''
    print len(total_dist);print len(samples_lerp);print len(ensemble_lerp)
    
    total_dist.sort()
    sub_e_lerp = ensemble_lerp[0:len(total_dist)]
    sub_e_lerp.sort()
    sub_samples_lerp = samples_lerp[0:len(ensemble_lerp)]
    sub_samples_lerp.sort()
    ensemble_lerp.sort()
    
    #shift distributions to haveing all positive values to calculate KL distance
    if total_dist[0] < 0:
        offset = math.fabs(total_dist[0])
        for idx in range(0,len(total_dist)):
            total_dist[idx] += offset
    if sub_e_lerp[0] < 0:
        offset = math.fabs(sub_e_lerp[0])
        for idx in range(0,len(sub_e_lerp)):
            sub_e_lerp[idx] += offset
    if sub_samples_lerp[0] < 0:
        offset = math.fabs(sub_samples_lerp[0])
        for idx in range(0,len(sub_samples_lerp)):
            sub_samples_lerp[idx] += offset
            
    '''
       
        
def lerp(a, b, w):
    return a + (b - a) * w

def quantileLerp(f1, f2, x1, x2, alpha):
    a = 1.0 - alpha
    b = alpha
    return ( f1(x1) * f2(x2) ) / ( a*f2(x2) + b*f1(x1) ) 

def gatherRandomSamples():
    ''' this is just a test routine, which will not be used by clients of this module '''
    
    #dist1
    mean0 = 2;stdev0 = 1.0
    mean1 = 5;stdev1 = 1.0
    #mean2 = -4;stdev2 = 1.0
    
    #dist2
    mean3 = 3;stdev3 = 1.5
    mean4 = 6;stdev4 = 1.5
    #mean5 = 1;stdev5 = 1.5
    
    dist1 = r.rnorm(SAMPLES*0.5, mean = mean0, sd = stdev0) + r.rnorm(SAMPLES*0.5, mean = mean1, sd = stdev1)
    dist2 = r.rnorm(0.5*SAMPLES, mean = mean3, sd = stdev3) + r.rnorm(0.5*SAMPLES, mean = mean4, sd = stdev4) #\
    #+ r.rnorm(0.4*SAMPLES, mean = mean5, sd = stdev5)
    #random.shuffle(dist2)
    #random.shuffle(dist1)
   
    return dist1, dist2  
        
if __name__ == '__main__':
    FILE_NAME = 'pe_dif_sep2_98.nc' 

    COM =  2
    LON = 53
    LAT = 90
    LEV = 16
    MEM = 600
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')
    
    #redefine with netcdf data...
    gp1 = []#[[],[]]
    gp2 = []#[[],[]]
    
    gp1_x = np.zeros(shape=(MEM,1))
    gp1_y = np.zeros(shape=(MEM,1))
    gp2_x = np.zeros(shape=(MEM,1))
    gp2_y = np.zeros(shape=(MEM,1))
    
    #gp1, gp2 = gatherRandomSamples()
    
    #gp1 = list(gp1)
    #gp2 = list(gp2)
    
    #gp1.sort()
    #gp2.sort()
    
    #print gp1
    #print gp2  
    
    #lat41,lon12,level0, u comp
    '''
    for idx in range(0,MEM):
        gp1_x[idx] = vclin[idx][86][33][0][0]
        gp1_y[idx] = vclin[idx][86][33][0][1]
        
        gp2_x[idx] = vclin[idx][86][34][0][0]
        gp2_y[idx] = vclin[idx][86][34][0][1]
    '''
        
    #alpha=0.0 to alpha=1.0, increments 0.1, left-to-right, 59LAT 20LON TO 60LAT 20LON U COMP    
    for idx in range(0,MEM):
        gp1.append( vclin[idx][60][20][0][0] )
        #gp1_y[idx] = vclin[idx][86][33][0][1]
        
        gp2.append( vclin[idx][59][20][0][0] )
        #gp2_y[idx] = vclin[idx][86][34][0][1]
        
    
    #SAMPLES = 6000
    KNOTS_CONTROL_POINTS = 100
    
    percentiles = list(spread(0, 1.0, KNOTS_CONTROL_POINTS-1, mode=3)) #[0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.99]
    percentiles.sort()
    
    KNOTS_CONTROL_POINTS = len(percentiles)
      
    a_uni = np.asarray(gp1)
    a_bi = np.asarray(gp2)
    
    #print a_uni
    
    #np.random.shuffle(a_uni)
    #np.random.shuffle(a_bi)
        
    gp0_u_kd = stats.gaussian_kde(a_uni)
    gp1_u_kd = stats.gaussian_kde(a_bi)
    
    #x = linspace(-3,3,2000)
    
    #find values of percentiles for pdf and use those as control points for cdf spline approx.
    for per in percentiles:
        
        percentiles_uni.append(r.quantile(robjects.FloatVector(gp1), per, type = 8)[0])
        percentiles_bi.append(r.quantile(robjects.FloatVector(gp2), per, type = 8)[0])
        if (per == 0.0):
            x = r.quantile(robjects.FloatVector(gp1), per, type = 8)[0]
            #need to correct for differences between R's quantile function and scipy's kde
            #solution: add more values above and below quantile 0th / 1.0
            print x
    
            #percentiles_uni.append(x-.01)
            #percentiles_uni.append(x-0.02)
            #percentiles_uni.append(x-0.2)
            #percentiles_uni.append(x-0.3)
            percentiles_uni.append(x-1.0)
            percentiles_uni.append(x-2.0)
            percentiles_uni.append(x-3.0)
            percentiles_uni.append(x-4.0)
            percentiles_uni.append(x-10.0)
            
         
        if (per == 1.0):
            x = r.quantile(robjects.FloatVector(gp1), per, type = 8)[0]
            
            
            percentiles_uni.append(x+1.0)
            percentiles_uni.append(x+2.0)
            percentiles_uni.append(x+3.0)
            percentiles_uni.append(x+4.0)
            percentiles_uni.append(x+5.0)
            
        if (per == 0.0):
            x = r.quantile(robjects.FloatVector(gp2), per, type = 8)[0]
        
        
            #percentiles_bi.append(x-.01)
            #percentiles_bi.append(x-0.02)
            #percentiles_bi.append(x-0.2)
            #percentiles_bi.append(x-0.3)
            percentiles_bi.append(x-1.0)
            percentiles_bi.append(x-2.0)
            percentiles_bi.append(x-3.0)
            percentiles_bi.append(x-4.0)
            percentiles_bi.append(x-10.0)
            
            
            
        if (per == 1.0):
            x = r.quantile(robjects.FloatVector(gp2), per, type = 8)[0]
            
            percentiles_bi.append(x+1.0)
            percentiles_bi.append(x+2.0)
            percentiles_bi.append(x+3.0)
            percentiles_bi.append(x+4.0)
            percentiles_bi.append(x+5.0)
            
    interp_dist = list(spread(0.0, 1.0, 10, mode=3))
    
    percentiles_uni.sort()
    percentiles_bi.sort()
    
    #norm(data1)
    x = np.linspace(-20,20,300)
   
    k1_samples = gaussian_kde(gp1)
    _max_u1, _min_u1 = peakdetect(k1_samples(x),x,lookahead=2,delta=0)
    
    xm_u1 = [p[0] for p in _max_u1]
    ym_u1 = [p[1] for p in _max_u1]
    
    print "peaks " + str(xm_u1) 
    
    #norm(data2)
    x2 = np.linspace(-20,20,300)
    k2_samples = gaussian_kde(gp2)
    _max_u2, _min_u2 = peakdetect(k2_samples(x2),x2,lookahead=2,delta=0)
    
    xm_u2 = [p[0] for p in _max_u2]
    ym_u2 = [p[1] for p in _max_u2]
   
    print "peaks " + str(xm_u2) 
    
    max_gs = len(xm_u1)
    if len(xm_u2) > max_gs:
        max_gs = len(xm_u2)
        
    #try this...    
    max_gs=4
        
    # GMM method
    r.library('mixtools')
    mixmdl = r.normalmixEM(robjects.vectors.FloatVector(gp1),k=max_gs,maxit = 5000, maxrestarts=5000)
    #fit single gaussian
    m1 = r.mean(robjects.vectors.FloatVector(gp1));var1= r.var(robjects.vectors.FloatVector(gp1))
    mixmdl2 = r.normalmixEM(robjects.vectors.FloatVector(gp2),k=max_gs,maxit = 5000, maxrestarts=5000)
    m2 = r.mean(robjects.vectors.FloatVector(gp2));var2= r.var(robjects.vectors.FloatVector(gp2))
    
    print r.summary(mixmdl)
    print r.summary(mixmdl2)
    
    mu_1 = []
    sd_1 = []
    lb_1 = []
    for i in mixmdl.iteritems():
        if i[0] == 'mu':
            mu_1.append(i[1])
        if i[0] == 'sigma':
            sd_1.append(i[1])
        if i[0] == 'lambda':
            lb_1.append(i[1])
        
    n_params_1 = []     
    if ASSUME_NORM == False:   
        for idx in range(0,len(mu_1[0])):
            n_params_1.append([mu_1[0][idx], sd_1[0][idx], lb_1[0][idx]])
    else:
        n_params_1.append([m1[0],np.sqrt(var1[0]), 1.0])
    
    mu_2 = []
    sd_2 = []
    lb_2 = []
    for i in mixmdl2.iteritems():
        if i[0] == 'mu':
            mu_2.append(i[1])
        if i[0] == 'sigma':
            sd_2.append(i[1])
        if i[0] == 'lambda':
            lb_2.append(i[1])
            
    n_params_2 = []     
    if ASSUME_NORM == False:   
        for idx in range(0,len(mu_2[0])):
            n_params_2.append([mu_2[0][idx], sd_2[0][idx], lb_2[0][idx]])
    else:
        n_params_2.append([m2[0],np.sqrt(var2[0]), 1.0])
        
    #plot GMM versus KDEs
    SAMPLES = 1000
    data1 = []
    for idx in range(0,len(mu_1[0])):
        data1 += list(np.asarray(r.rnorm(SAMPLES*lb_1[0][idx], mean=mu_1[0][idx], sd = sd_1[0][idx])))
        
    k1 = gaussian_kde(data1)
    
    #plot GMM versus KDEs
    data2 = []
    for idx in range(0,len(mu_2[0])):
        #MOG = list(r.rnorm(SAMPLES*lb_2[0][idx], mean=mu_2[0][idx], sd = sd_2[0][idx]))
        #MOG_kde = gaussian_kde(MOG)
        #plt.plot(x,MOG_kde(x), color='red')
        data2 += list(np.asarray(r.rnorm(SAMPLES*lb_2[0][idx], mean=mu_2[0][idx], sd = sd_2[0][idx])))
        
    k2 = gaussian_kde(data2)
      
    gmm_ent = []
    qua_ent = [] 
    gauss_ent = [] 
    for cur_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        g, q, gauss = lerpMix(n_params_1, n_params_2, alpha = cur_alpha, steps = 1, method = 1, num_gs=max_gs, \
                       gp1_mean = np.mean(gp1), gp1_var = np.var(gp1), gp2_mean = np.mean(gp2), gp2_var = np.var(gp2))
        gmm_ent.append(g)
        qua_ent.append(q)
        gauss_ent.append(gauss)
        print cur_alpha 
    #plt.title("quantile lerp using pdf, alpha " + str(alpha) )
    
    
    
    #randomize intermedia sog
    #lerpMix(n_params1, n_params2, alpha = 0.3, steps = 10, method = 2)
    
    #for idx in range(0,21):
    #    cur_alpha = idx * 0.05
    #    uniformLerpMix(n_params1, n_params2, alpha=cur_alpha)
    #    gmm_ent.append(g); qua_ent.append(q)
        
    
    plt.figure()
    alphas = linspace(0,1,11)
    p1, = plt.plot(alphas,gmm_ent,'-',color='red')
    p2, = plt.plot(alphas,qua_ent, '-', color='blue')
    #p3, = plt.plot(alphas,gauss_ent, '-', color='purple')
    
    plt.legend([p1, p2], ["GMM entropy", "Quantile entropy"])
    
    #if method == 1:
    plt.savefig(OUTPUT_DATA_DIR + "entropy_tests_graph_rev3.png")
    
    plt.figure()
    alphas = linspace(0,1,11)
    #p1, = plt.plot(alphas,gmm_ent,'-',color='red')
    #p2, = plt.plot(alphas,qua_ent, '-', color='blue')
    p3, = plt.plot(alphas,gauss_ent, '-', color='purple')
    
    plt.legend([p3], ["Gaussian entropy"])
    
    #if method == 1:
    plt.savefig(OUTPUT_DATA_DIR + "gaussian_entropy_tests_graph_rev3.png")
         
    #plt.show()
    
    print "finished!"
    
    
    
    
    
    




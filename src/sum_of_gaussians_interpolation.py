
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

import matplotlib.mlab as mlab
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

from rpy2.robjects import numpy2ri
numpy2ri.activate()

from netcdf_reader import NetcdfReader
from GMM_Quantile_lerp import *

r = robjects.r

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
    plt.savefig("./png/" + str(alpha) + "_" + "normalized_uniform_method" + ".png")      
    plt.show()
      
def lerpMix(norm_params1, norm_params2, alpha, steps, method):    
    ''' handles equal number of constituent gaussians '''
    
    sorted(norm_params2, key=operator.itemgetter(0), reverse=False)
    sorted(norm_params1, key=operator.itemgetter(0), reverse=False)
   
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
        
        # interpolate each gaussian
        for idx in range(len(norm_params1)):
            cur_mean1 = norm_params1[idx][0];cur_mean2 = norm_params2[idx][0]
            cur_std1 = norm_params1[idx][1];cur_std2 = norm_params2[idx][1];
            cur_ratio1 = norm_params1[idx][2];cur_ratio2 = norm_params2[idx][2];
            
            inter_means.append(cur_mean1*(1.0-subalpha) + cur_mean2*subalpha)
            inter_stdevs.append(cur_std1*(1.0-subalpha) + cur_std2*subalpha)
            inter_comp_ratios.append(cur_ratio1*(1.0-subalpha) + cur_ratio2*subalpha)
            
        norm_params1 = []
        for j in range(len(inter_means)):    
            norm_params1.append((inter_means[j], inter_stdevs[j], inter_comp_ratios[j]))
        
        #set plot ranges, etc.
        xmin = -3; xmax = 3; ymin=0.0; ymax = 1.0
        
        #plot interp GMM 
        SAMPLES = 1000
        total_dist = []
        for idx in range(0,len(inter_means)):
            cur_inter_mean = inter_means[idx];cur_inter_stdev = inter_stdevs[idx];cur_inter_ratio = inter_comp_ratios[idx] 
            total_dist += list(np.asarray(r.rnorm(int(SAMPLES*cur_inter_ratio), mean=cur_inter_mean, sd = cur_inter_stdev)))
        
        #norm(data1)
        x = np.linspace(-3,3,300)
        total_dist_kde = gaussian_kde(total_dist)
        _max_u, _min_u = peakdetect(total_dist_kde(x),x,lookahead=2,delta=0)
        
        xm_u = [p[0] for p in _max_u]
        ym_u = [p[1] for p in _max_u]
       
        #print "peaks " + str(xm_u) 
        
        plt.title("GMM lerping " + str(subalpha))
        plt.plot(x,total_dist_kde(x), color='black')  
        plt.plot(xm_u,ym_u,'x',color='red') 
        
    
        #save out plot for current interpolated distribution
        
        if method == 1:
            plt.savefig("../png/" + str(subalpha) + "_" + "resortMethod" + ".png")
        elif method == 2:
            plt.savefig("../png/" + str(subalpha) + "_" + "randomizeMethod" + ".png") 
        elif method == 0:
             plt.savefig("../png/" + str(subalpha) + "_" + "noStepMethod" + ".png") 
             
        plt.show()
        
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
    
    #redefine with netcdf data...
    gp1 = []
    gp2 = []
    
    for idx in range(0,600):
        gp1.append(vclin[idx][41][12][0][0])
        gp2.append(vclin[idx][40][12][0][0])
    
    r.library('mixtools')
    mixmdl = r.normalmixEM(robjects.vectors.FloatVector(gp1))
    mixmdl2 = r.normalmixEM(robjects.vectors.FloatVector(gp2))
    
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
    for idx in range(0,len(mu_1[0])):
        n_params_1.append([mu_1[0][idx], sd_1[0][idx], lb_1[0][idx]])
    
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
    for idx in range(0,len(mu_2[0])):
        n_params_2.append([mu_2[0][idx], sd_2[0][idx], lb_2[0][idx]])
    
    #plot GMM versus KDEs
    SAMPLES = 1000
    data1 = []
    for idx in range(0,len(mu_1[0])):
        data1 += list(np.asarray(r.rnorm(SAMPLES*lb_1[0][idx], mean=mu_1[0][idx], sd = sd_1[0][idx])))
    
    #norm(data1)
    x = np.linspace(-3,3,300)
    k1 = gaussian_kde(data1)
    k1_samples = gaussian_kde(gp1)
    _max_u, _min_u = peakdetect(k1(x),x,lookahead=2,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    ym_u = [p[1] for p in _max_u]
   
    print "peaks " + str(xm_u) 
    
    
    
    plt.title("GMM in black - gp1")
    plt.plot(x,k1(x), color='black') 
    plt.plot(x,k1_samples(x), color='blue')  
    plt.plot(xm_u,ym_u,'x',color='red') 
    plt.savefig("../png/GP1_GMM versus KDE.png") 
    plt.show()
    
    #plot GMM versus KDEs
    x2 = np.linspace(-3,3,100)
    data2 = []
    for idx in range(0,len(mu_2[0])):
        #MOG = list(r.rnorm(SAMPLES*lb_2[0][idx], mean=mu_2[0][idx], sd = sd_2[0][idx]))
        #MOG_kde = gaussian_kde(MOG)
        #plt.plot(x,MOG_kde(x), color='red')
        data2 += list(np.asarray(r.rnorm(SAMPLES*lb_2[0][idx], mean=mu_2[0][idx], sd = sd_2[0][idx])))
    
    #norm(data2)
    k2 = gaussian_kde(data2)
    k2_samples = gaussian_kde(gp2)
    _max_u, _min_u = peakdetect(k2(x2),x2,lookahead=2,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    ym_u = [p[1] for p in _max_u]
   
    print "peaks " + str(xm_u) 
    
    plt.title("GMM in black - gp2")
    plt.plot(x,k2(x), color='black') 
    plt.plot(x,k2_samples(x), color='blue')  
    plt.plot(xm_u,ym_u,'x',color='red') 
    plt.savefig("../png/GP2_GMM versus KDE.png") 
    plt.show()
    
    #resort intermediate sog
    #for cur_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    '''
    cur_alpha = 0.3
    lerpMix(n_params1, n_params2, alpha = cur_alpha, steps = 10, method = 1)
    lerpMix(n_params1, n_params2, alpha = cur_alpha, steps = 10, method = 2)
    '''
        
    #for cur_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #    lerpMix(n_params1, n_params2, alpha = cur_alpha, steps = 1, method = 1) 
      
    for cur_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        lerpMix(n_params_1, n_params_2, alpha = cur_alpha, steps = 1, method = 1)
        
    #randomize intermedia sog
    #lerpMix(n_params1, n_params2, alpha = 0.3, steps = 10, method = 2)
    
    #for idx in range(0,21):
    #    cur_alpha = idx * 0.05
    #    uniformLerpMix(n_params1, n_params2, alpha=cur_alpha)
    
    
    
    
    
    




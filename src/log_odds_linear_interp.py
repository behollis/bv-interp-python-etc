
# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 10/24/2012.
    Code performs linear interpolation between discrete distributions in logOdds space.
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

r = robjects.r

'''
logOdds interpolation over-view

1. Gather the continuous probability distributions being interpolated
between (here the defined gaussian / non-gaussian ones on grid points)

2. Derive discrete representations of continuous probabilities from #1
(respective histograms from sampling distributions), where each
histogram's bins are represented by (p1, ..., pM) from simplex
definition

3. Transform both discrete distribution representations to logOdds domain

4. Perform linear interpolation on logOdds representations of discrete
distributions

5. Transform interpolation result from logOdds space back to discrete
distribution

6. Perform best fit / density estimation on discrete distribution to
get back to continuous distribution

7. Measure / visualize plot of interpolated distribution as a
continuous function
'''

SAMPLES = 10000
DISCRETE_BINS = 500

def gatherRandomSamples():
    ''' this is just a test routine, which will not be used by clients of this module '''
    
    global SAMPLES
    
    mean0 = -5.0;stdev0 = 0.1
    mean1 = -5;stdev1 = 1.0
    mean2 = 5;stdev2 = 1.0
    
    mean3 = 5.0;stdev3 = 2.0
    
    dist1 = r.rnorm(SAMPLES, mean = mean0, sd = stdev0)
    dist2 = r.rnorm(SAMPLES/2, mean = mean1, sd = stdev1) + r.rnorm(SAMPLES/2, mean = mean2, sd = stdev2)
    random.shuffle(dist2)
   
    return dist1, dist2

def transformToPmf(dist_samples):
    
    #here we treate each bin starting value as a discrete value in the pmf,
    #where each bins frequency is the frequency of the starting x value of the bin
    '''
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html#numpy.histogram
    
    Examples

    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([ 0.25,  0.25,  0.25,  0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([ 0.5,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist*np.diff(bin_edges))
    1.0
    '''
    hist, bin_edges = np.histogram(dist_samples, bins=DISCRETE_BINS, density=True)
    pmf_probabilities = hist*np.diff(bin_edges)
    
    pmf_random_variable_values_labels = bin_edges
    pmf = [pmf_probabilities, pmf_random_variable_values_labels]
    
    # return the pmf's
    return pmf


def plotPmf(hist, alpha=-1):
    ''' visual inspection of pmf '''
    
    # get the corners of the rectangles for the histogram
    left = np.array(hist[1][:-2])
    right = np.array(hist[1][1:-1])
    bottom = np.zeros(len(left-1))
    top = bottom + hist[0][0:len(left-1)]
    nrects = len(left-1)

    # here comes the tricky part -- we have to set up the vertex and path
    # codes arrays using moveto, lineto and closepoly

    # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
    # CLOSEPOLY; the vert for the closepoly is ignored but we still need
    # it to keep the codes aligned with the vertices
    nverts = nrects*(1+3+1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5,0] = left
    verts[0::5,1] = bottom
    verts[1::5,0] = left
    verts[1::5,1] = top
    verts[2::5,0] = right
    verts[2::5,1] = top
    verts[3::5,0] = right
    verts[3::5,1] = bottom

    fig = plt.figure("png")
    ax = fig.add_subplot(111)

    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    
    plt.title(str(alpha))
    plt.savefig(str(alpha) + "_interp_.png")
    plt.show()
    
def transformToLogOdds(pmf):
    ''' [logit(p)]i := log(pi/pm) = log(pi) - log(pm) '''
    pm = pmf[0][len(pmf[0])-1]
    
    log_odds_pmf = np.ndarray(shape = (len(pmf[0]) - 1), dtype = float, order = 'F')
    for i in range(len(pmf[0]) - 1):
        #calculate each pi
        pi = pmf[0][i]
        log_odds_pmf[i] = np.log(pi/pm) 
        if np.isinf(np.log(pi/pm)) == True:
            log_odds_pmf[i] = 0.0
        
    # return logOdds transform and the labels / values for each pi
    return (log_odds_pmf, pmf[1])

def transformFromLogOdds(log_odds_pmf):
    ''' generalized logistics function, transforms back from logOdds '''
    
    EULERS_NUM = 2.718281
    
    Z = 1
    for idx in range(len(log_odds_pmf[0])):
        Z += EULERS_NUM**log_odds_pmf[0][idx]
         
    pmf = np.ndarray(shape = (len(log_odds_pmf[0]) + 1), dtype = float, order = 'F')
    
    pmf[len(log_odds_pmf[0])] = 1.0 / Z
    
    for idx in range(len(log_odds_pmf[0])):
        pmf[idx]= EULERS_NUM**log_odds_pmf[0][idx] / Z

        
    return (pmf, log_odds_pmf[1])
    
if __name__ == '__main__':
    #unit tests for module
    s1, s2 = gatherRandomSamples()
    
    #test...direct linear interpolation on parameters
    mean0 = -3.0;variance0 = 0.5
    mean1 = +3.0;variance1 = 2.0
    
    sigma0 = np.sqrt(variance0)
    x = np.linspace(-10,10,100)
    
    #plt.figure("png")
    
    STEPS = 10
    for idx in range(STEPS):
        alpha = float(idx)/STEPS
        
        inter_mean = mean0*(1.0-alpha) + mean1*alpha
        inter_stdev = np.sqrt(variance0)*(1.0-alpha) + np.sqrt(variance1)*alpha
        
        x = np.linspace(-10,10,100)
        plt.plot(x,mlab.normpdf(x,inter_mean,inter_stdev))
        
        #plt.title(str(alpha) + "_" + str(inter_mean) + "_" + str(inter_stdev))
        #plt.savefig("./png/" + str(inter_mean) + "_" + str(inter_stdev) + "_" + str(alpha) + "_interpDist_along_x_unit.png")
        plt.show()
    
    '''
    pmf1 = transformToPmf(s1)#;print np.sum(pmf1[0]);plotPmf(pmf1)
    pmf2 = transformToPmf(s2)#;print np.sum(pmf2[0]);plotPmf(pmf2)
    
    lo_pmf1 = transformToLogOdds(pmf1)#;plotPmf(lo_pmf1)
    lo_pmf2 = transformToLogOdds(pmf2)#;plotPmf(lo_pmf2)
    
    #pmf_back_1 = transformFromLogOdds(lo_pmf1);print np.sum(pmf_back_1[0]);plotPmf(pmf_back_1)
    #pmf_back_2 = transformFromLogOdds(lo_pmf2);print np.sum(pmf_back_1[0]);plotPmf(pmf_back_2)
    
    #interpolate distributions in logOdds space
    STEPS = 10
    empty_array = np.ndarray(shape = (DISCRETE_BINS-1), dtype = float, order = 'F')
    labels_copy = pmf1[1][:-1]
    lo_pmf_interp = [ empty_array , labels_copy ]

    for idx in range(STEPS):
        alpha = float(idx)/STEPS
        
        #done on each member of pmf array
        for idx in range(DISCRETE_BINS-1):
            lo_pmf_interp[0][idx] = lo_pmf1[0][idx]*(1.0-alpha) + lo_pmf2[0][idx]*alpha
            
        pmf_interp = transformFromLogOdds(lo_pmf_interp)
    
        print np.sum(pmf_interp[0])
    
        #plotPmf(lo_pmf_interp)
        plotPmf(pmf_interp, alpha)
    '''
        
   
    
    
    
    




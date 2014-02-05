"""

http://scikit-image.org/
http://scikit-image.org/docs/dev/auto_examples/plot_contours.html

===============
Contour finding
===============

``skimage.measure.find_contours`` uses a marching squares method to find
constant valued contours in an image.  Array values are linearly interpolated
to provide better precision of the output contours.  Contours which intersect
the image edge are open; all others are closed.

The `marching squares algorithm
<http://www.essi.fr/~lingrand/MarchingCubes/algo.html>`__ is a special case of
the marching cubes algorithm (Lorensen, William and Harvey E. Cline. Marching
Cubes: A High Resolution 3D Surface Construction Algorithm. Computer Graphics
(SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).

"""

#from skimage_contours import data
#from skimage_contours import measure
import netCDF4 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
#import math
import math 
import math as pm
from scipy import stats
from numpy import linspace
from scipy import interpolate

from netcdf_reader import *
import skimage_contours  

from mayavi.mlab import *
import mayavi

import rpy2.robjects as robjects
_R = robjects.r
_R.library('mixtools')

from quantile_lerp import quantileLerp
from prob_particle_advection import lerp
from prob_particle_advection import fitGaussian
from prob_particle_advection import lerpGMMPair
from prob_particle_advection import spread

FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
REL_FILE_DIR = '/home/behollis/netcdf/'
COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600
OUT_DIR = '/home/behollis/thesis/data/out/pics/contour/'

def fitGMM(gp, max_gs=2):
    
    #suppress std out number of iterations using r.invisible()
    mixmdl = _R.invisible(_R.normalmixEM(robjects.vectors.FloatVector(gp), k = max_gs, maxit = 5000, maxrestarts=5))
   
    mu = [];sd = [];lb = []
    for i in mixmdl.iteritems():
        if i[0] == 'mu':
            mu.append(i[1])
        if i[0] == 'sigma':
            sd.append(i[1])
        if i[0] == 'lambda':
            lb.append(i[1])
        
    n_params = []     
    for idx in range(0,len(mu[0])):
        n_params.append([mu[0][idx], sd[0][idx], lb[0][idx]])
        
    return n_params

def interpGMM(ppos=[0.0,0.0], field=None):
    #assume grid points are defined by integer indices
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    gpt0_dist = np.zeros(shape=(600))
    gpt1_dist = np.zeros(shape=(600))
    gpt2_dist = np.zeros(shape=(600))
    gpt3_dist = np.zeros(shape=(600))
    
    for idx in range(0,MEM):
        gpt0_dist[idx] = field[idx][gpt0[0]][gpt0[1]]
        gpt1_dist[idx] = field[idx][gpt1[0]][gpt1[1]]
        gpt2_dist[idx] = field[idx][gpt2[0]][gpt2[1]]
        gpt3_dist[idx] = field[idx][gpt3[0]][gpt3[1]]
    
    #get gmm's
    #NOTE: need to check if dist is guassian-like. if so, don't try to fit more than one gaussian to distribution or you'll get convergence issues with EM alg
    MAX_GMM_COMP = 3
    try:
        gp0_parms = fitGMM(gp=list(gpt0_dist[:]),max_gs=MAX_GMM_COMP)
    except:
        return []
    try:
        gp1_parms = fitGMM(gp=list(gpt1_dist[:]),max_gs=MAX_GMM_COMP)
    except:
        return []
    try:    
        gp2_parms = fitGMM(gp=list(gpt2_dist[:]),max_gs=MAX_GMM_COMP)
    except:
        return []
    try:
        gp3_parms = fitGMM(gp=list(gpt3_dist[:]),max_gs=MAX_GMM_COMP)
    except:
        return []
    
    lerp_scalar_gp0_gp1_params = lerpGMMPair(np.asarray(gp0_parms), np.asarray(gp1_parms), \
                                             alpha = ppos_parts[0][0], steps = 1, \
                                             num_gs = MAX_GMM_COMP )
    lerp_scalar_gp2_gp3_params = lerpGMMPair(np.asarray(gp2_parms), np.asarray(gp3_parms), \
                                             alpha = ppos_parts[0][0], steps = 1, \
                                             num_gs = MAX_GMM_COMP )
    lerp_scalar_params = lerpGMMPair( np.asarray(lerp_scalar_gp0_gp1_params), \
                                      np.asarray(lerp_scalar_gp2_gp3_params), \
                                      alpha = ppos_parts[1][0], steps = 1, \
                                      num_gs = MAX_GMM_COMP )
  
    #return interp GMM 
    SAMPLES = MEM
    total_dist = []
    for idx in range(0,len(lerp_scalar_params)):
        cur_inter_mean = lerp_scalar_params[idx][0];cur_inter_stdev = lerp_scalar_params[idx][1];cur_inter_ratio = lerp_scalar_params[idx][2] 
        total_dist += list(np.asarray(_R.rnorm(int(SAMPLES*cur_inter_ratio), mean=cur_inter_mean, sd = cur_inter_stdev)))

    return total_dist  

def interpEnsemble(ppos=[0.0,0.0],field=None):
    #assume grid points are defined by integer indices
    
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    #print "ensemble alpha x: " + str( ppos_parts[0][0] )
    #print "ensemble alpha y: " + str( ppos_parts[1][0] )
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    gpt0_dist = np.zeros(shape=600)
    gpt1_dist = np.zeros(shape=600)
    gpt2_dist = np.zeros(shape=600)
    gpt3_dist = np.zeros(shape=600)
    
    for idx in range(0,600):
        gpt0_dist[idx] = field[idx][gpt0[0]][gpt0[1]]
        gpt1_dist[idx] = field[idx][gpt1[0]][gpt1[1]]
        gpt2_dist[idx] = field[idx][gpt2[0]][gpt2[1]]
        gpt3_dist[idx] = field[idx][gpt3[0]][gpt3[1]]
        
    #lerp ensemble samples
    lerp_scalar_gp0_gp1 = lerp( np.asarray(gpt0_dist), np.asarray(gpt1_dist), w = ppos_parts[0][0] )
    lerp_scalar_gp2_gp3 = lerp( np.asarray(gpt2_dist), np.asarray(gpt3_dist), w = ppos_parts[0][0] ) 
    lerp_scalar = lerp( np.asarray(lerp_scalar_gp0_gp1), np.asarray(lerp_scalar_gp2_gp3), w = ppos_parts[1][0] )  
    
    return lerp_scalar

def interpGaussian(ppos=[0.0,0.0],field=None):
    #assume grid points are defined by integer indices
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    gpt0_dist = np.zeros(shape=(600))
    gpt1_dist = np.zeros(shape=(600))
    gpt2_dist = np.zeros(shape=(600))
    gpt3_dist = np.zeros(shape=(600))
    
    for idx in range(0,MEM):
        gpt0_dist[idx] = field[idx][gpt0[0]][gpt0[1]]
        gpt1_dist[idx] = field[idx][gpt1[0]][gpt1[1]]
        gpt2_dist[idx] = field[idx][gpt2[0]][gpt2[1]]
        gpt3_dist[idx] = field[idx][gpt3[0]][gpt3[1]]
         
    
    #get gmm's
    #NOTE: need to check if dist is guassian-like. if so, don't try to fit more than one gaussian to distribution or you'll get convergence issues with EM alg
    gp0_parms = fitGaussian(gp=list(gpt0_dist[:]))
    gp1_parms = fitGaussian(gp=list(gpt1_dist[:]))
    gp2_parms = fitGaussian(gp=list(gpt2_dist[:]))
    gp3_parms = fitGaussian(gp=list(gpt3_dist[:]))
    
    lerp_scalar_gp0_gp1_params = lerp(np.asarray(gp0_parms), np.asarray(gp1_parms), w = ppos_parts[0][0] )
    lerp_scalar_gp2_gp3_params = lerp(np.asarray(gp2_parms), np.asarray(gp3_parms), w = ppos_parts[0][0] )
    lerp_scalar_params = lerp( np.asarray(lerp_scalar_gp0_gp1_params), np.asarray(lerp_scalar_gp2_gp3_params), w = ppos_parts[1][0] )
    
    NUM_SAMPLES = 600
    samples = _R.rnorm(NUM_SAMPLES, mean=lerp_scalar_params[0], sd=math.sqrt(lerp_scalar_params[1]))

    return np.asarray(samples)

def bilinearQuantLerp(f1, f2, f3, f4, x1, x2, x3, x4, alpha, beta):
    a0 = 1.0 - alpha
    b0 = alpha
    a1 = 1.0 - beta
    b1 = beta
    
    try:
        f_one = f1(x1)
        f_two = f2(x2)
        f_three = f3(x3)
        f_four = f4(x4)            
        
        f_bar_0 = f_one * f_two / (a0*f_two + b0*f_one) 
        f_bar_1 = f_three * f_four / (a0*f_four + b0*f_three) 
        
        f_bar_01 = f_bar_0 * f_bar_1 / (a1*f_bar_1 + b1*f_bar_0)
    except:
        print 'problem with calculated interpolant y value...'
        f_bar_01 = [] #failed
    
    return f_bar_01



def interpQuantiles(ppos=[0.0,0.0],field=None):
    #assume grid points are defined by integer indices
    
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    #print "quantile alpha x: " + str( ppos_parts[0][0] )
    #print "quantile alpha y: " + str( ppos_parts[1][0] )
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    gpt0_dist = np.zeros(shape=(600))
    gpt1_dist = np.zeros(shape=(600))
    gpt2_dist = np.zeros(shape=(600))
    gpt3_dist = np.zeros(shape=(600))
    
    for idx in range(0,600):
        gpt0_dist[idx] = field[idx][gpt0[0]][gpt0[1]]
        gpt1_dist[idx] = field[idx][gpt1[0]][gpt1[1]]
        gpt2_dist[idx] = field[idx][gpt2[0]][gpt2[1]]
        gpt3_dist[idx] = field[idx][gpt3[0]][gpt3[1]]
      
    QUANTILES = 100
    quantiles = list(spread(0.0, 1.0, QUANTILES-1, mode=3)) 
    quantiles.sort()
    
    #find random variable value of quantiles for pdf 
    q_gpt0_dist = []
    q_gpt1_dist = []
    q_gpt2_dist = []
    q_gpt3_dist = []
    
    for q in quantiles:
        q_gpt0_dist.append(_R.quantile(robjects.FloatVector(gpt0_dist), q)[0])
        q_gpt1_dist.append(_R.quantile(robjects.FloatVector(gpt1_dist), q)[0])
        q_gpt2_dist.append(_R.quantile(robjects.FloatVector(gpt2_dist), q)[0])
        q_gpt3_dist.append(_R.quantile(robjects.FloatVector(gpt3_dist), q)[0])
    
    try:
        gp0kde = stats.gaussian_kde(gpt0_dist)
        gp1kde = stats.gaussian_kde(gpt1_dist)
        gp2kde = stats.gaussian_kde(gpt2_dist)
        gp3kde = stats.gaussian_kde(gpt3_dist)
    except:
        print "kde failed..."
        return []

    alpha_x = ppos_parts[0][0]
    alpha_y = ppos_parts[1][0]
    
    print alpha_x
    print alpha_y

    lerp_scalar_gp0_gp1_values = lerp(np.asarray(q_gpt0_dist), np.asarray(q_gpt1_dist), w = alpha_x )
    lerp_scalar_gp2_gp3_values = lerp(np.asarray(q_gpt2_dist), np.asarray(q_gpt3_dist), w = alpha_x )
    lerp_scalar_values = lerp(np.asarray(lerp_scalar_gp0_gp1_values), np.asarray(lerp_scalar_gp2_gp3_values), w = alpha_y )
    
    #lerp_prob_values = []
    #for idx in range(0,len(q_gpt0_dist)):
    
    #lerp_prob_values0 = quantileLerp(gp0kde, gp1kde,q_gpt0_dist[:], q_gpt1_dist[:], alpha_x)
    #lerp_prob_values2 = quantileLerp(gp2kde, gp3kde,q_gpt2_dist[:], q_gpt3_dist[:], alpha_x)
        
    lerp_scalar_prob = bilinearQuantLerp(gp0kde, gp1kde, \
                          gp2kde, gp3kde, \
                          q_gpt0_dist[:], q_gpt1_dist[:], \
                          q_gpt2_dist[:], q_gpt3_dist[:], \
                          alpha_x, alpha_y)
    
    #print 'lerped prob value: ' + str(lerp_prob_values[idx])
    
    #gp0_gp1_interp = interpolate.interp1d(lerp_scalar_gp0_gp1_values,lerp_scalar_gp0_gp1_prob)
    
    '''
    fx = interpolate.UnivariateSpline(np.asarray(lerp_scalar_values), np.asarray(lerp_prob_values))#, \
                                        #w=None, bbox=[-100,100], k=3, s=None)
          
    #integrate over probabilites
    total_prob = 0      
    for i in range(0,len(lerp_prob_values)-1,1):
        total_prob += np.abs(lerp_prob_values[i+1] - lerp_prob_values[i])
        print total_prob
        
    threshold_integral = 0
    for i in range(0,len(lerp_prob_values)-1,1):
        total_prob += np.abs(lerp_prob_values[i+1] - lerp_prob_values[i])
        print total_prob
    
    
                                        
    print fx.integral(lerp_scalar_values[0],35)
    print 'cumulative density: ' + str(fx.integral(lerp_scalar_values[0],35) / total_prob)
    density = fx.integral(lerp_scalar_values[0],35) / total_prob 
                                        
    #s = linspace(0,35,100)
    '''
    '''
    print fx.integral(-10, 35)
    plt.plot(lerp_scalar_values,lerp_prob_values, linewidth=1.0)
    plt.show()
    '''
    '''
                                        
    return 1.-density
    
    #lerp_scalar_gp2_gp3_prob = quantileLerp( m, n, np.asarray(q_gpt2_dist), np.asarray(q_gpt3_dist), alpha = ppos_parts[0][0] )
    
    '''
    NUM_SAMPLES = 2000
    
    '''
    samples_numbers = lerp_scalar_gp0_gp1_prob * NUM_SAMPLES
    samples_gp0_gp1_lerp = []
    for prob_idx in range(0,len(lerp_scalar_gp0_gp1_prob)):
        #if not math.isnan(samples_numbers[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers[prob_idx])):
            samples_gp0_gp1_lerp.append(lerp_scalar_gp0_gp1_values[prob_idx])
            
    samples_numbers2 = lerp_scalar_gp2_gp3_prob * NUM_SAMPLES
    samples_gp2_gp3_lerp = []
    for prob_idx in range(0,len(lerp_scalar_gp2_gp3_prob)):
        #if not math.isnan(samples_numbers2[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers2[prob_idx])):
            samples_gp2_gp3_lerp.append(lerp_scalar_gp2_gp3_values[prob_idx])
    
    gp0_gp1_interp = interpolate.interp1d(lerp_scalar_gp0_gp1_values,lerp_scalar_gp0_gp1_prob)
    gp2_gp3_interp = interpolate.interp1d(lerp_scalar_gp2_gp3_values,lerp_scalar_gp2_gp3_prob)
    
    #min_0 = min(lerp_scalar_gp0_gp1_values); max_0 = max(lerp_scalar_gp0_gp1_values)
    #min_1 = min(lerp_scalar_gp2_gp3_values); max_1 = max(lerp_scalar_gp2_gp3_values)
    
    #dom_0 = linspace(min_0,max_0,500)
    #dom_1 = linspace(min_1,max_1,500)
    
    #samples_gp0_gp1_lerp = lerp_scalar_gp0_gp1_values#gp0_gp1_interp(dom_0)
    #samples_gp2_gp3_lerp = lerp_scalar_gp2_gp3_values#gp2_gp3_interp(dom_1)
    
    #q_lerp_gpt0_gpt1_dist = [];q_lerp_gpt2_gpt3_dist = []
    
    #quantiles = list(spread(0.1, 0.9, QUANTILES-1, mode=3)) 
    #quantiles.sort()
    '''
    '''
    for q in quantiles:
        #print q
        #print _R.quantile(robjects.FloatVector(samples_gp0_gp1_lerp), q)[0]
        #print _R.quantile(robjects.FloatVector(samples_gp2_gp3_lerp), q)[0]
        q_lerp_gpt0_gpt1_dist.append(_R.quantile(robjects.FloatVector(samples_gp0_gp1_lerp), q)[0])
        q_lerp_gpt2_gpt3_dist.append(_R.quantile(robjects.FloatVector(samples_gp2_gp3_lerp), q)[0])
    '''
    
    '''
    lerp_scalar_prob = quantileLerp( stats.gaussian_kde(samples_gp0_gp1_lerp), \
                                     stats.gaussian_kde(samples_gp2_gp3_lerp), \
                                     np.asarray(q_lerp_gpt0_gpt1_dist), \
                                     np.asarray(q_lerp_gpt2_gpt3_dist), \
                                     alpha = ppos_parts[1][0] )
    '''
    
    #try:
    '''
    lerp_scalar_prob = quantileLerp( interpolate.interp1d( lerp_scalar_gp0_gp1_values, lerp_scalar_gp0_gp1_prob ), \
                                    interpolate.interp1d( lerp_scalar_gp2_gp3_values, lerp_scalar_gp2_gp3_prob ), \
                                    np.asarray(q_lerp_gpt0_gpt1_dist), \
                                    np.asarray(q_lerp_gpt2_gpt3_dist), alpha = ppos_parts[1][0] )
    '''
    #except:
    #    print"failed quantile lerp..."
    #    return []
    
        
    samples_numbers3 = lerp_scalar_prob * NUM_SAMPLES
    samples_scalar_lerp = []
    for prob_idx in range(0,len(lerp_scalar_prob)):
        #if not math.isnan(samples_numbers3[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers3[prob_idx])):
            if not math.isnan(lerp_scalar_values[prob_idx]) and not math.isinf(lerp_scalar_values[prob_idx]):
                samples_scalar_lerp.append(lerp_scalar_values[prob_idx])
            
    return samples_scalar_lerp

def addCentralForecastVar( _in, central_forecast, level_start, level_end):
    #adds central forecast to each realization in ensemble '''
    #curr_level = level
    for curr_level in range(level_start, level_end+1, 1):
        for curr_lon in range(LON):
            for curr_lat in range(LAT): 
                for curr_realization in range(MEM):
                    _in[curr_realization][curr_lat][curr_lon][curr_level] += central_forecast[curr_lat][curr_lon][curr_level]
                    _in[curr_realization][curr_lat][curr_lon][curr_level] += central_forecast[curr_lat][curr_lon][curr_level]
    return _in

def isoContourProb(ensemble,threshold,sample_factor=1,interp='g',start_x=0, end_x=LAT-1, start_y=0, end_y=LON-1, sub_image = False): 
    ''' MAKE SUB-IMAGE SQUARE!!! THIS ALGOR PRODUCES EQUAL SAMPLES IN EACH DIM
    SUCH THAT IF THE START AND END DISTANCES ARE NOT SQUARE THEN WE WILL GET SCALING IN OUTPUT WITH 
    HIGHER SUB-SAMPLING IN SMALLER DIM ''' 
    ''' ALSO, SUB-IMAGES MUST ALWAYS BE ON GRID-POINT BOUNDARIES FROM ORIGINAL FIELD / IMAGE DATA'''
    #output = np.zeros(shape=(ensemble.shape[1]* sample_factor, ensemble.shape[2]* sample_factor))
    # number of grid points in sub_image
    
    if sub_image == False:
        sample_factor = 1
    
    output = None
    if sub_image == True:
        num_grid_pts_y = end_y - start_y + 1
        num_grid_pts_x = end_x - start_x + 1
        
        num_sub_samples_along_grid_cell_y = sample_factor - 1
        num_sub_samples_along_grid_cell_x = sample_factor - 1
        
        num_sub_samples_in_grid_cell = num_sub_samples_along_grid_cell_y * num_sub_samples_along_grid_cell_x 
                                           
        total_grid_cells = (num_grid_pts_y - 1) * (num_grid_pts_x - 1)
        total_sub_samples_within_grid_cells = num_sub_samples_in_grid_cell * total_grid_cells
        total_sub_samples_along_grid_cell_boundaries = (num_sub_samples_along_grid_cell_y + num_sub_samples_along_grid_cell_x) * total_grid_cells \
                                                       + num_sub_samples_along_grid_cell_y * ( end_y - start_y ) \
                                                       + num_sub_samples_along_grid_cell_x * ( end_x - start_x ) 
        total_samples_including_gpts = total_sub_samples_within_grid_cells + num_grid_pts_y * num_grid_pts_x \
                                       +  total_sub_samples_along_grid_cell_boundaries 
    
        output = np.zeros(shape=(math.sqrt(total_samples_including_gpts), math.sqrt(total_samples_including_gpts)))
    else:
        output = np.zeros(shape=(ensemble.shape[1], ensemble.shape[2]))  
    
    
    #num_square_steps = (ensemble.shape[1] - 1) * (ensemble.shape[2] - 1) * sample_factor * sample_factor
    #if sample_factor == 1:
        #make sure we don't sample outside of full data set
    #    num_square_steps = (end_y - start_y - 1) * (end_x - start_x -1) * (sample_factor) * (sample_factor)
    #else:
    num_samples = output.size#(end_y - start_y) * (end_x - start_x) * (sample_factor) * (sample_factor)
        
    
    # Current coords start at 0,0.
    in_coords = [start_y,start_x]
    out_coords = [0,0]
    #not_gp = False
    #func = None
    quant_dens = 0.
    for n in range(0,num_samples,1):
        print "SAMPLE RATE: " + str(sample_factor)
        print "lcp " + str(n) + " of " + str(num_samples)
        print str(in_coords[0]) + ' ' + str(in_coords[1])
        cur_gp_samples = []
        
        #if pm.modf(in_coords[0])[0] == 0. and pm.modf(in_coords[1])[0] == 0.:
        #    for idx in range(0,MEM):
        #        cur_gp_samples.append(ensemble[idx][int(in_coords[0])][int(in_coords[1])])
        #else:
            
        # we need to interpolate distributions
        if interp == 'e':
            #print 'transformed coords for lookup: ' + str(LAT-1-coords[1]) + ' ' + str(LON-1-coords[0])
            cur_gp_samples = interpEnsemble(in_coords,field=ensemble)
        elif interp == 'gmm':
            cur_gp_samples = interpGMM(in_coords,field=ensemble)
        elif interp == 'q':
            #cur_gp_samples = interpQuantiles(in_coords,field=ensemble)
            #func = interpQuantiles(in_coords,field=ensemble)
            #cur_gp_samples.append(1) #just to pass test
            #not_gp = True
            cur_gp_samples = interpQuantiles(in_coords,field=ensemble)
            #cur_gp_samples = [1]
        else: #gaussian case
            cur_gp_samples = interpGaussian(in_coords,field=ensemble)
          
        if len(cur_gp_samples) > 0:# and func != []: 
            #if interp == 'q': #and not_gp:
                #print "quantile integral value..."
                #x = np.abs(func.integral(0, threshold) / func.integral(0,100)) #normalizes
                #x = quant_dens
                #if x > 1.:
                #    x = 1.0
                #elif x < 0.:
                #    x = 0.
                
            #else:  
            cdf = _R.ecdf( robjects.FloatVector(np.asarray(cur_gp_samples)) ) 
            x = cdf(threshold)[0]
            #lcp = 2.*x*(1.-x) #level crossing prob
            lcp = 1. - math.pow(x,4.) - math.pow((1.-x),4.)
            
        else:
            print "samples returned were zero!!!!"
            lcp = 0.
        
        #print coords
        #print lcp
            
        output[out_coords[0]][out_coords[1]] = lcp
        #output[in_coords[0]-start_y][in_coords[1]-start_x] = lcp
        
        # now in advance the coords indices
        if out_coords[0] < output.shape[0] - 1:#in_coords[0] < end_y - 2. / sample_factor:
            in_coords[0] += 1. / sample_factor
            out_coords[0] += 1
        else:
            in_coords[1] += 1. / sample_factor
            in_coords[0] = start_y
            out_coords[1] += 1
            out_coords[0] = 0
       
    return output
    

if __name__ == "__main__": 
    
    lev = 0
    
    #realizations file 
    pe_dif_sep2_98_file = REL_FILE_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = REL_FILE_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #central forecasts reader 
    creader = NetcdfReader(pe_fct_aug25_sep2_file)
    temp = rreader.readVarArray('temp')
    temp8 = creader.readVarArray('temp', 7)
    temp = addCentralForecastVar(temp, temp8, level_start=lev, level_end=lev)
    
    #r = np.zeros(shape=(LON, LAT))
    r_dist = np.zeros(shape=(MEM, LON, LAT))
    for mem in range(0,MEM,1):
        for lat in range(0,LAT):
            for lon in range(0,LON):
                #r_dist[mem][lon][lat] = temp[mem][LAT-1-lat][(LON-1)-lon][lev]
                r_dist[mem][lon][lat] = temp[mem][LAT-1-lat][(LON-1)-lon][lev]
    
    iso_value = 35

    '''
    #write out base image with sampling rate 1 (lcp only at grid points)
    prob_contours = isoContourProb(r_dist, iso_value, sample_factor = 1, start_x=0,end_x=LAT-1, \
                                           start_y = 0, end_y = LON-1, sub_image = False)
    
    fig = plt.figure()
    plt.imshow(prob_contours, interpolation='none', cmap = cm.Greys_r)
    #pb_mayavi = mayavi.mlab.imshow(prob_contours, colormap='Reds', interpolate=True,transparent=True)
    #mayavi.mlab.show()
    
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    
    fig.savefig(OUT_DIR + 'base_rate_1_full_level' + '_' + str(0) \
                 + '_' + str(LAT-1) + '_' + str(0) + '_' + str(LON-1) + '.png', dpi = fig.dpi*2)
    
    '''
    #sub images should be square!! or you will get scaling in shortest dimension with lower res
    #since all output images (except full image sampled only at grid points) are square
    st_x = 2; en_x = 50
    st_y = 2 ; en_y = 50       
    
    '''
    #write out base image with sampling rate 1 (lcp only at grid points)
    prob_contours = isoContourProb(r_dist, iso_value, sample_factor = 1, start_x=st_x,end_x=en_x, \
                                           start_y = st_y, end_y = en_y, sub_image = True)
    
    fig = plt.figure()
    plt.imshow(prob_contours, interpolation='none', cmap = cm.Greys_r)
    #pb_mayavi = mayavi.mlab.imshow(prob_contours, colormap='Reds', interpolate=True,transparent=True)
    #mayavi.mlab.show()
    
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    
    fig.savefig(OUT_DIR + 'base_rate_1' + '_' + str(st_x) \
                 + '_' + str(en_x) + '_' + str(st_y) + '_' + str(en_y) + '.png', dpi = fig.dpi*2)
    '''
    
    interps = ['g']#,'q']#'g','q','gmm']
    
    '''
    for rate in range(2,7,1):
        for interp_type in interps:
            #origin upper left
            prob_contours = isoContourProb(r_dist, iso_value, sample_factor = rate, interp = interp_type, start_x=st_x,end_x=en_x, \
                                           start_y = st_y, end_y = en_y, sub_image = True)
            
            fig = plt.figure()
            plt.imshow(prob_contours, interpolation='none', cmap = cm.Greys_r)
            #pb_mayavi = mayavi.mlab.imshow(prob_contours, colormap='Reds', interpolate=True,transparent=True)
            #mayavi.mlab.show()
            
            plt.axis('image')
            plt.xticks([])
            plt.yticks([])
            
            fig.savefig(OUT_DIR + str(rate) + '_' + str(interp_type) + '_' + str(st_x) \
                         + '_' + str(en_x) + '_' + str(st_y) + '_' + str(en_y) + '.png', dpi = fig.dpi*2)
      
    '''     
    for rate in range(5,6,1):
        for interp_type in interps:
            #origin upper left
            prob_contours = isoContourProb(r_dist, iso_value, sample_factor = rate, \
                                           interp = interp_type, start_x=st_x,end_x=en_x, \
                                           start_y = st_y, end_y = en_y, sub_image = True)
            
            fig = plt.figure()
            plt.imshow(prob_contours, interpolation='none', cmap = cm.Greys_r)
            #pb_mayavi = mayavi.mlab.imshow(prob_contours, colormap='Reds', interpolate=True,transparent=True)
            #mayavi.mlab.show()
            
            plt.axis('image')
            plt.xticks([])
            plt.yticks([])
            
            fig.savefig(OUT_DIR + str(rate) + '_' + str(interp_type) + '_' + str(st_x) \
                         + '_' + str(en_x) + '_' + str(st_y) + '_' + str(en_y) + '.png', dpi = fig.dpi*2)
            
    print "finished!"
                  

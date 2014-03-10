#!/usr/bin/python
import netCDF4 
import sys, struct
import rpy2.robjects as robjects
import random
import math as pm
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import math

#import gaussian_fit
import sum_of_gaussians_interpolation as sog
from netcdf_reader import *
#from spline_cdf_curve_morphing import *
from mayavi.mlab import *
import mayavi
from peakfinder import *
from quantile_lerp import *
import os

import datetime 
dt = 'stampeMicroSec_' + str(datetime.datetime.now().microsecond) + '_' 

import time



q_prev_max_vel_x = 0.0
q_prev_max_vel_y = 0.0
e_prev_max_vel_x = 0.0
e_prev_max_vel_y = 0.0
gmm_prev_max_vel_x = 0.0
gmm_prev_max_vel_y = 0.0

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 


SEED_LEVEL = 0
vclin = []


reused_vel_quantile = 0

DEBUG = False


INPUT_DATA_DIR = '/home/behollis/thesis_data/data/in/ncdf/'
#OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/gpDist/'
  
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '/home/behollis/thesis_data/data/in/ncdf/'
OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/pics/bv_interp/'

#COM =  2
#LON = 9
#LAT = 9
#LEV = 16
#MEM = 600 

EM_MAX_ITR = 5
EM_MAX_RESTARTS = 1000
DEPTH = -2.0
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
NUM_GAUSSIANS = 4#2
MAX_GMM_COMP = 4#NUM_GAUSSIANS 


#part_pos_e = []
#part_pos_e.append([0,0])
#part_pos_e[0][0] = SEED_LAT
#part_pos_e[0][1] = SEED_LON

r = robjects.r

ZERO_ARRAY = np.zeros(shape=(MEM,1))

SAMPLES = 600
vclin_x = np.ndarray(shape=(SAMPLES,LAT,LON))
vclin_y = np.ndarray(shape=(SAMPLES,LAT,LON))

vclin_half = np.ndarray(shape=(SAMPLES,LAT/2 + 1,LON/2 + 1,2))
vclin_half = np.ndarray(shape=(SAMPLES,LAT/2 + 1,LON/2 + 1,2))

g_grid_params_array = []
g_grid_kde_array = []
g_grid_quantile_curves_array = []

QUANTILES = 50
'''
divs = 80
div = complex(divs)
div_real = divs
start = -22#works with +/-5
end = +22
TOL = 0.05 #( 1.0 / QUANTILES ) / 3.0
'''

def lerpBivGMMPair(norm_params1, norm_params2, alpha, steps=1, num_gs=MAX_GMM_COMP):     
    ''' handles equal number of constituent gaussians '''
    # pair based on gaussian contribution to gmm
    sorted(norm_params2, key=operator.itemgetter(2), reverse=False)
    sorted(norm_params1, key=operator.itemgetter(2), reverse=False)
    
    if steps != 0:  
        incr = alpha / steps
    else:
        incr = alpha
        
    interpolant_params = []    
    for idx in range(0,num_gs):
        #get vector between means
        mean_vec = np.asarray(norm_params2[idx][0]) - np.asarray(norm_params1[idx][0])
        dist = np.sqrt(np.dot(mean_vec, mean_vec))
        mean_vec_n = mean_vec / dist
        
        interpolant_mean = np.asarray(norm_params1[idx][0]) + alpha * dist * mean_vec_n
        interpolant_cov = np.matrix(norm_params1[idx][1]) * (1.-alpha) + np.matrix(norm_params2[idx][1]) * alpha
        interpolant_ratio = norm_params1[idx][2] * (1.-alpha) + norm_params2[idx][2] * alpha 
        
        interpolant_params.append([interpolant_mean, interpolant_cov, interpolant_ratio])                                                                 
   
    '''
    for idx in range(0,steps+1):
        #resort to minimize distances in means of pairings
        sorted(norm_params1, key=operator.itemgetter(0), reverse=False)
                
        subalpha = float(idx) * incr
        
        inter_means = []; inter_stdevs = []; inter_comp_ratios = []
        
        max_comps = len(norm_params1)
        
        if max_comps < len(norm_params2):
            max_comps = len(norm_params2)
        
        # interpolate each gaussian
        for idx in range(0,max_comps):
            cur_mean1 = norm_params1[idx][0]
            cur_std1 = norm_params1[idx][1]
            cur_ratio1 = norm_params1[idx][2]
        
            cur_mean2 = norm_params2[idx][0]
            cur_std2 = norm_params2[idx][1]
            cur_ratio2 = norm_params2[idx][2]
            
            inter_means.append(cur_mean1*(1.0-subalpha) + cur_mean2*subalpha)
            inter_stdevs.append(cur_std1*(1.0-subalpha) + cur_std2*subalpha)
            inter_comp_ratios.append(cur_ratio1*(1.0-subalpha) + cur_ratio2*subalpha)
            
        norm_params1 = []
        for j in range(len(inter_means)):    
            norm_params1.append([inter_means[j], inter_stdevs[j], inter_comp_ratios[j]])
    '''
    
    #return interp GMM params
    return interpolant_params

def createGlobalParametersArray(dimx, dimy):
    global g_grid_params_array
    
    for idx in range(0,dimx):
        g_grid_params_array.append([])
        for idy in range(0,dimy):
            g_grid_params_array[idx].append([])

def createGlobalKDEArray(dimx, dimy):
    global g_grid_kde_array

    for idx in range(0,dimx):
        g_grid_kde_array.append([])
        for idy in range(0,dimy):
            g_grid_kde_array[idx].append(None)
            
def createGlobalQuantileArray(dimx, dimy):
    global g_grid_quantile_curves_array

    for idx in range(0,dimx):
        g_grid_quantile_curves_array.append([])
        for idy in range(0,dimy):
            g_grid_quantile_curves_array[idx].append([[],[],[]])

#....................... bivariate interp helper functions

def getCoordParts(ppos=[0.0,0.0]):
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    return ppos_parts 

def getGridPoints(ppos=[0.0,0.0], half=False):
    #assume grid points are defined by integer indices
    
    ppos_parts = getCoordParts(ppos)
    
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
    if half == False:
        gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
        gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
        gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
        gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    else:
        gpt0 = [ppos_parts[0][1] / 2, ppos_parts[1][1] / 2]
        gpt1 = [ppos_parts[0][1] / 2 + 1, ppos_parts[1][1] / 2]
        gpt2 = [ppos_parts[0][1] / 2, ppos_parts[1][1] / 2 + 1]
        gpt3 = [ppos_parts[0][1] / 2 + 1, ppos_parts[1][1] / 2 + 1]
    
    return gpt0, gpt1, gpt2, gpt3

def getVclinSamples(gpt0, gpt1, gpt2, gpt3, half=False):
    gpt0_dist = np.zeros(shape=(2,SAMPLES))
    gpt1_dist = np.zeros(shape=(2,SAMPLES))
    gpt2_dist = np.zeros(shape=(2,SAMPLES))
    gpt3_dist = np.zeros(shape=(2,SAMPLES))
    
    for idx in range(0,MEM):#SAMPLES):
        '''
        gpt0_dist[0][idx] = vclin_x[idx][gpt0[0]][gpt0[1]]
        gpt0_dist[1][idx] = vclin_y[idx][gpt0[0]][gpt0[1]]
        
        gpt1_dist[0][idx] = vclin_x[idx][gpt1[0]][gpt1[1]]
        gpt1_dist[1][idx] = vclin_y[idx][gpt1[0]][gpt1[1]]
        
        gpt2_dist[0][idx] = vclin_x[idx][gpt2[0]][gpt2[1]]
        gpt2_dist[1][idx] = vclin_y[idx][gpt2[0]][gpt2[1]]
        
        gpt3_dist[0][idx] = vclin_x[idx][gpt3[0]][gpt3[1]]
        gpt3_dist[1][idx] = vclin_y[idx][gpt3[0]][gpt3[1]]
        '''
        
        if not half:
            gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
            gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
            gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
            gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
            
            gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
            gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
            
            gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
            gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
        else:
            gpt0_dist[0][idx] = vclin_half[idx][gpt0[0]][gpt0[1]][0]
            gpt0_dist[1][idx] = vclin_half[idx][gpt0[0]][gpt0[1]][1]
         
            gpt1_dist[0][idx] = vclin_half[idx][gpt1[0]][gpt1[1]][0]
            gpt1_dist[1][idx] = vclin_half[idx][gpt1[0]][gpt1[1]][1] 
            
            gpt2_dist[0][idx] = vclin_half[idx][gpt2[0]][gpt2[1]][0]
            gpt2_dist[1][idx] = vclin_half[idx][gpt2[0]][gpt2[1]][1] 
            
            gpt3_dist[0][idx] = vclin_half[idx][gpt3[0]][gpt3[1]][0]
            gpt3_dist[1][idx] = vclin_half[idx][gpt3[0]][gpt3[1]][1] 
            
            
   
    return gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist

'''
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, 
    mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.
     html>`_
    at mathworld.
    """
    Xmu = X - mux
    Ymu = Y - muy
    rho = sigmaxy / (sigmax * sigmay)
    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu 
     / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / denom
'''

def plotKDE(kde,distro, mx = (0,0), my = (0, 0), title = '', co = 'green'):
    
    global DIV, DIVR
    
    FIG = plt.figure()

    #x_flat = np.r_['''np.asarray(distro[0]).min()'''mx[0]:'''np.asarray(distro[0]).max()'''mx[1]:DIV]
    #y_flat = np.r_['''np.asarray(distro[1]).min()'''my[0]:'''np.asarray(distro[1]).max()'''my[1]:DIV]
    x_flat = np.r_[mx[0]:mx[1]:DIV]
    y_flat = np.r_[my[0]:my[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(DIVR, DIVR)
    
    AX = FIG.gca(projection='3d')
    AX.set_xlabel('u')
    AX.set_ylabel('v')
    AX.set_xlim(x_flat.max(), x_flat.min())
    AX.set_ylim(y_flat.min(), y_flat.max())
    AX.set_zlabel('density')
    AX.set_zlim(0, z.max())
    AX.plot_surface(x, y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True, alpha=1.0, color=co)
    plt.savefig(OUTPUT_DATA_DIR + title + ".jpg")
    #plt.show() 
  
def getKDE(x_min_max,y_min_max,kde):
    
    global DIV, DIVR
    
    #x_flat = np.r_[np.asarray(distro[0]).min():np.asarray(distro[0]).max():div]
    #y_flat = np.r_[np.asarray(distro[1]).min():np.asarray(distro[1]).max():div]
    #x,y = np.meshgrid(x_flat,y_flat)
    
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:DIV]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(DIVR, DIVR)
    
    return z
  
def getBivariateGMM(x_min_max,y_min_max,params = [0.0,0.0,0.0]):
    global DIV, DIVR

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:DIV]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    #z = kde(grid_coords.T)
    #z = z.reshape(div_real,div_real)
    
    Z_total = np.zeros(shape=(len(x_flat),len(y_flat)))
    
    for idx in range(0,len(params)):
        cur_inter_mean  =  params[idx][0]
        cur_inter_cov   =  params[idx][1]
        cur_inter_ratio =  params[idx][2] 
        
        print 'interp ratio: ' + str(cur_inter_ratio)
        
        #x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,SAMPLES).T
        
        '''IMPORTANT: x and y axes are swapped between kde and rpy2 vectors!!!! '''
        
        #instead of drawing samples from bv normal, get surface rep via matplot lib
        Z_total += mlab.bivariate_normal(x, y, cur_inter_cov.item((1,1)), \
                                   cur_inter_cov.item((0,0)), \
                                   cur_inter_mean[1], cur_inter_mean[0] ) * cur_inter_ratio
                                   
    return Z_total

def plotDistro(x_min_max,y_min_max,title='', params = [0.0,0.0,0.0],color='b'):
    
    global DIV, DIVR

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:DIV]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    #z = kde(grid_coords.T)
    #z = z.reshape(div_real,div_real)
    
    Z_total = np.zeros(shape=(len(x_flat),len(y_flat)))
    
    for idx in range(0,len(params)):
        cur_inter_mean  =  params[idx][0]
        cur_inter_cov   =  params[idx][1]
        cur_inter_ratio =  params[idx][2] 
        
        print 'interp ratio: ' + str(cur_inter_ratio)
        
        #x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,SAMPLES).T
        
        '''IMPORTANT: x and y axes are swapped between kde and rpy2 vectors!!!! '''
        
        #instead of drawing samples from bv normal, get surface rep via matplot lib
        Z_total += mlab.bivariate_normal(x, y, cur_inter_cov.item((1,1)), \
                                   cur_inter_cov.item((0,0)), \
                                   cur_inter_mean[1], cur_inter_mean[0] ) * cur_inter_ratio
        
    
    fig = plt.figure()
    
    #p3.view_init(elev,azim)
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, Z_total, rstride=2, cstride=2, linewidth=0.1, antialiased=True, alpha=1.0,\
                            color=color)#,,cmap=cm.spectral)
    
    #ax.set_xticks([-4,-2,0,2,4])
    #ax.set_yticks([-4,-2,0,2,4])
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.max(), x_flat.min())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('density')
    ax.set_zlim(0, Z_total.max())
   
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('kde')
    
    
    
    #for angle in range(45, 360, 90 ):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.savefig(OUTPUT_DATA_DIR + str(title) +str(angle)+ ".png")
    plt.savefig(OUTPUT_DATA_DIR + str(title) + ".jpg")
    
    #ax.view_init(90, 90 )
    #plt.draw()
    #plt.savefig(OUTPUT_DATA_DIR + str(title) + "top.png")
         
    #plt.show()    
 
def plotXYZ(x_tot2,y_tot2,z,title=''):
    fig = plt.figure(title)
                        
    ax = fig.gca(projection='3d')
    surf = ax.scatter(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('')
          
    plt.show()    
 
def defineVclin():
    div = 50j
    div_real = 50
    
    SAMPLES = 1000
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    
    #right half
    mean2 = [0,+1]
    cov2 = [[1,0],[0,1]] 
    x2,y2 = np.random.multivariate_normal(mean2,cov2,SAMPLES).T
    
    #bimodal bivariate grid point distribution
    mean3 = [+2,+1]
    cov3 = [[1,0],[0,1]] 
    mean4 = [-2,-1]
    cov4 = [[1.5,0],[0,1.5]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.6*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.4*SAMPLES)).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    
    #left half
    '''
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    
    x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    '''
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
     
    '''    
    kde = stats.kde.gaussian_kde((x1,y1))#gp1_2d_array.T)
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[x1.min():x1.max():div]
    y_flat = np.r_[y1.min():y1.max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)

    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
    
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('kde')
          
    plt.show()       
    '''
    
    #right half
    '''
    mean2 = [0,+1]
    cov2 = [[1,0],[0,1]] 
    
    x2,y2 = np.random.multivariate_normal(mean2,cov2,SAMPLES).T
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()    
    '''
    
    '''
    kde = stats.kde.gaussian_kde((x2,y2))#gp1_2d_array.T)
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[x2.min():x2.max():div]
    y_flat = np.r_[y2.min():y2.max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
     
    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
    
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('kde')
          
    plt.show()   
    '''   
    
    #bimodal bivariate grid point distribution
    '''
    mean3 = [-2,-1]
    cov3 = [[0.5,0],[0,0.5]] 
    mean4 = [+2,+1]
    cov4 = [[1.0,0],[0,1.0]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,0.6*SAMPLES).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,0.4*SAMPLES).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    '''
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
    
    #gp1_2d_array = np.append(x1,y1,axis=1)
    #gp2_2d_array = np.append(x2,y2,axis=1)
       
    ''' 
    kde = stats.kde.gaussian_kde((x_tot,y_tot))#gp1_2d_array.T)

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_tot.min():x_tot.max():div]
    y_flat = np.r_[y_tot.min():y_tot.max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
   
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('kde')
          
    plt.show() 
    '''
    
    #define 10x10 gp's samples
    for x in range(0,10):
        for y in range(0,10):
            for idx in range(0,SAMPLES):
                if x <=4: 
                    vclin_x[idx][x][y] = x1[idx] 
                    vclin_y[idx][x][y] = y1[idx]  
                elif x > 4:
                    vclin_x[idx][x][y] = x2[idx]  
                    vclin_y[idx][x][y] = y2[idx]
                
    #define bimodal bivariate grid point
    for idx in range(0,SAMPLES):
        vclin_x[idx][4][4] = x_tot[idx]
        vclin_y[idx][4][4] = y_tot[idx]        
        
    
    x_tot2 = []
    y_tot2 = []
    for idx in range(0,SAMPLES):
        x_tot2.append(vclin_x[idx][5][2])
        y_tot2.append(vclin_y[idx][5][2])
        
    #kde = stats.kde.gaussian_kde((x_tot2,y_tot2))#gp1_2d_array.T)

    #plotDistro(kde,x_tot2,y_tot2)

    '''
    # Regular grid to evaluate kde upon
    x_flat = np.r_[np.asarray(x_tot2).min():np.asarray(x_tot2).max():div]
    y_flat = np.r_[np.asarray(y_tot2).min():np.asarray(y_tot2).max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
   
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('kde')
          
    plt.show()         
    '''
    
def interpVelFromEnsemble(ppos=[0.0,0.0]):
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos)
    
    gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist = getVclinSamples(gpt0, gpt1, gpt2, gpt3)
    
    #lerp ensemble samples
    lerp_u_gp0_gp1 = lerp( np.asarray( gpt0_dist[0] ), np.asarray( gpt1_dist[0]), w = ppos_parts[0][0] )
    lerp_u_gp2_gp3 = lerp( np.asarray( gpt2_dist[0] ), np.asarray( gpt3_dist[0]), w = ppos_parts[0][0] ) 
    lerp_u = lerp( np.asarray(lerp_u_gp0_gp1), np.asarray(lerp_u_gp2_gp3), w = ppos_parts[1][0] )  
    
    lerp_v_gp0_gp1 = lerp( np.asarray(gpt0_dist[1] ), np.asarray(gpt1_dist[1]), w = ppos_parts[0][0] )
    lerp_v_gp2_gp3 = lerp( np.asarray(gpt2_dist[1] ), np.asarray(gpt3_dist[1]), w = ppos_parts[0][0] ) 
    lerp_v = lerp( np.asarray(lerp_v_gp0_gp1), np.asarray(lerp_v_gp2_gp3), w = ppos_parts[1][0] )  
    
    #x = linspace( lerp_u[0], lerp_u[-1], len(lerp_u) )
    #y = linspace( lerp_v[0], lerp_v[-1], len(lerp_v) )
    
    x = linspace( -50, 50, 600 )
    y = linspace( -50, 50, 600 )
        
    try:
        k = stats.kde.gaussian_kde((lerp_u,lerp_v)) 
        return k

    except:
        print "kde not working"
        return None
  
def interpFromGMM(ppos=[0.0,0.0], ignore_cache = 'False', half=False):
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gp0, gp1, gp2, gp3 = getGridPoints(ppos, half)
    
    gp0_dist, gp1_dist, gp2_dist, gp3_dist = getVclinSamples(gp0, gp1, gp2, gp3, half)
    
    global g_grid_params_array 
    
    i = int(gp0[0])
    j = int(gp0[1])
    params0 = g_grid_params_array[i][j] 
    if len(params0) == 0 or ignore_cache is 'True': 
        gp0_dist_transpose = gp0_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp0_dist_transpose[:,[0, 1]] = gp0_dist_transpose[:,[1, 0]]
        params0 = fitBvGmm(gp0_dist_transpose)
        g_grid_params_array[i][j] = params0

    '''
    total_dist_x = np.zeros(shape=0)
    total_dist_y = np.zeros(shape=0)
    #x = []
    #y = []
    for idx in range(0,len(params0)):
        cur_inter_mean = params0[idx][0]
        cur_inter_cov = params0[idx][1]
        cur_inter_ratio = params0[idx][2] 
        
        x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov,cur_inter_ratio*SAMPLES).T
        
        total_dist_x = np.append(total_dist_x, x)
        total_dist_y = np.append(total_dist_y, y)
        
        
        #total_dist += list(np.asarray(r.mvrnorm( n = int(SAMPLES*cur_inter_ratio), mu = cur_inter_mean, Sigma = cur_inter_cov)))
        
        
    total_dist = np.asarray([list(total_dist_x), list(total_dist_y)]).T                                  
    k = stats.kde.gaussian_kde( (total_dist[:,1], total_dist[:,0]) ) 
    '''
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-4, 4), (-4, 4),title='gp0' )
    
    i = int(gp1[0])
    j = int(gp1[1])
    params1 = g_grid_params_array[i][j] 
    if len(params1) == 0 or ignore_cache is 'True': 
        gp1_dist_transpose = gp1_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp1_dist_transpose[:,[0, 1]] = gp1_dist_transpose[:,[1, 0]]
        params1 = fitBvGmm(gp1_dist_transpose)
        g_grid_params_array[i][j] = params1
        
    i = int(gp2[0])
    j = int(gp2[1])    
    params2 = g_grid_params_array[i][j] 
    if len(params2) == 0 or ignore_cache is 'True':
        gp2_dist_transpose = gp2_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp2_dist_transpose[:,[0, 1]] = gp2_dist_transpose[:,[1, 0]]
        params2 = fitBvGmm(gp2_dist_transpose)
        g_grid_params_array[i][j] = params2
    
    '''
    total_dist_x = np.zeros(shape=0)
    total_dist_y = np.zeros(shape=0)
    #x = []
    #y = []
    for idx in range(0,len(params2)):
        cur_inter_mean = params2[idx][0]
        cur_inter_cov = params2[idx][1]
        cur_inter_ratio = params2[idx][2] 
        
        x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,SAMPLES).T
        
        total_dist_x = np.append(total_dist_x, x)
        total_dist_y = np.append(total_dist_y, y)
        
        
        #total_dist += list(np.asarray(r.mvrnorm( n = int(SAMPLES*cur_inter_ratio), mu = cur_inter_mean, Sigma = cur_inter_cov)))
        
        
    total_dist = np.asarray([list(total_dist_x), list(total_dist_y)]).T                                  
    k = stats.kde.gaussian_kde( (total_dist[:,1], total_dist[:,0]) ) 
    '''
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-4, 4), (-4, 4),title='gp2' )
    
    i = int(gp3[0])
    j = int(gp3[1])
    params3 = g_grid_params_array[i][j] 
    if len(params3) == 0 or ignore_cache is 'True':
        gp3_dist_transpose = gp3_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp3_dist_transpose[:,[0, 1]] = gp3_dist_transpose[:,[1, 0]]
        params3 = fitBvGmm(gp3_dist_transpose)
        g_grid_params_array[i][j] = params3
     
   
    lerp_gp2_gp3_params = lerpBivGMMPair(np.asarray(params2), \
                                         np.asarray(params3), \
                                         alpha = ppos_parts[0][0], \
                                         steps = 1, \
                                         num_gs = MAX_GMM_COMP )
    
    
    
    lerp_gp0_gp1_params = lerpBivGMMPair(np.asarray(params0), \
                                         np.asarray(params1), \
                                         alpha = ppos_parts[0][0], \
                                         steps = 1, \
                                         num_gs = MAX_GMM_COMP )
    
    
    lerp_params = lerpBivGMMPair( np.asarray(lerp_gp0_gp1_params), \
                               np.asarray(lerp_gp2_gp3_params), \
                               alpha = ppos_parts[1][0], \
                               steps = 1, \
                               num_gs = MAX_GMM_COMP )
    
    '''
    total_dist_x = np.zeros(shape=0)
    total_dist_y = np.zeros(shape=0)
    #x = []
    #y = []
    for idx in range(0,len(lerp_params)):
        cur_inter_mean =  lerp_params[idx][0]
        cur_inter_cov =  lerp_params[idx][1]
        cur_inter_ratio =  lerp_params[idx][2] 
        
        x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,SAMPLES).T
        
        total_dist_x = np.append(total_dist_x, x)
        total_dist_y = np.append(total_dist_y, y)
        
        
        #total_dist += list(np.asarray(r.mvrnorm( n = int(SAMPLES*cur_inter_ratio), mu = cur_inter_mean, Sigma = cur_inter_cov)))    
        
    total_dist = np.asarray([list(total_dist_x), list(total_dist_y)]).T                                  
    k = stats.kde.gaussian_kde( (total_dist[:,1], total_dist[:,0]) ) 
    '''
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-3, 3), (-3, 3),title='total lerp' )
    
    pars = True

    return lerp_params

green = '#98fb98'
blue = '#87ceeb'
red = '#f08080'
purple = '#ee82ee'
    
def main():
    #gen_streamlines = str(sys.argv[1])
    
    global NUM_GAUSSIANS, MAX_GMM_COMP, EM_MAX_ITR
      
    NUM_GAUSSIANS = 4
    MAX_GMM_COMP = 4
    EM_MAX_ITR = 30
    
    #r.library('mixtools')
    loadNetCdfData()
    remapGridData()
    
    #createGlobalParametersArray(LAT,LON)
    createGlobalKDEArray(LAT,LON)
    createGlobalQuantileArray(LAT,LON)
    
    #vclin = np.zeros(shape=(10,10,2))
    
    #defineVclin()
    #createGlobalParametersArray(LAT,LON)
    
    #python -m cProfile -o outputfile.profile nameofyour_program alpha
    
    ppos = [0,37.0]
   
    kdes = []
    
    ''' http://www.tayloredmktg.com/rgb/#PA '''
    green = '#98fb98'
    blue = '#87ceeb'
    red = '#f08080'
    purple = '#ee82ee'
      
    for idx in range(0,3):
        ypos = ppos[1] + idx 
        
        print idx
        
        '''
        #find GMM approx
        lerp_params = interpFromGMM([ppos[0],ypos], ignore_cache = 'True', half=True)
        bivGMM = getBivariateGMM((x_min,x_max), (y_min,y_max), params = lerp_params)
        '''
        
        #find quantile approx (include surface interpolant choice)
        '''
        http://docs.python.org/2/library/time.html
        time.clock()
           On Unix, return the current processor time as a 
        floating point number expressed in seconds. The precision, 
        and in fact the very definition of the meaning of 'processor time', 
        depends on that of the C function of the same name, but in any case, 
        this is the function to use for benchmarking Python or timing algorithms.
        '''
        
        qstart = time.clock() #cpu time
        samples_arr, evalfunc = interpFromQuantiles3(ppos=[ppos[0],ypos], ignore_cache = 'True', half=True)
        
        #find KDE benchmark
        distro = getVclinSamplesSingle([ppos[0],ypos])
        kdes.append(distro)
        
        kde = stats.kde.gaussian_kde(distro)
        
        x_min = np.asarray(distro[0]).min()
        x_max = np.asarray(distro[0]).max()
        y_min = np.asarray(distro[1]).min()
        y_max = np.asarray(distro[1]).max()
        
        '''
        for gp in samples_arr:
            if x_min > gp[0].min():
                x_min = gp[0].min()
            if x_max < gp[0].max():
                x_max = gp[0].max()
                
            if y_min > gp[1].min():
                y_min = gp[1].min()
            if y_max < gp[1].max():
                y_max = gp[1].max()
        '''    
        
        mfunc1 = getKDE((x_min,x_max), (y_min,y_max),kde)
        
        
        distro2, interpType, suc = computeDistroFunction(evalfunc[0],evalfunc[1],evalfunc[2], \
                                                         (x_min,x_max), (y_min,y_max))
        qend = time.clock()
        
        qtot = qend - qstart
        
        skl1 = kl_div_2D_M(mfunc1, mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        
        '''
        skl3f = kl_div_2D_M(mfunc1=bivGMM, mfunc2=mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl3b = kl_div_2D_M(mfunc1=mfunc1, mfunc2=bivGMM, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl3 = skl3f + skl3b
        '''
        
        skl4f = kl_div_2D_M(mfunc1=distro2, mfunc2=mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl4b = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl4 = skl4f + skl4b
        
        title1 = dt + str(ppos[0]) + '_' + str(ypos) + '_kde_skl_' + str(skl1) 
        #title3 = str(ppos[0]) + '_' + str(ypos) + '_gmm_' + str(skl3)
        title4 = dt + str(ppos[0]) + '_' + str(ypos) + '_q_skl_'   + str(skl4) + '_elapsed_' + str(qtot)
        
        plotKDE(kde,distro,mx=(x_min,x_max), my=(y_min,y_max), title=title1, co = green)
        
        #plotDistro((x_min,x_max), (y_min,y_max), title3, params = lerp_params,color=red)
        plotXYZSurf((x_min,x_max), (y_min,y_max), distro2, title4, samples_arr, col=blue)
        plotXYZScatter((x_min,x_max), (y_min,y_max), evalfunc[0],evalfunc[1],evalfunc[2], title=dt + str(ppos[0]) + \
                       '_' + str(ypos) + '_Interp_', arr=samples_arr )
        
        if ypos == 38.0:
            '''
            lerp_params_actual = interpFromGMM([ppos[0],ypos], ignore_cache = 'True', half=False)
            bivGMM_actual = getBivariateGMM((x_min,x_max), (y_min,y_max), params = lerp_params_actual)
            
            skl3f_a = kl_div_2D_M(mfunc1=bivGMM_actual, mfunc2=mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl3b_a = kl_div_2D_M(mfunc1=mfunc1, mfunc2=bivGMM_actual, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl3_a = skl3f_a + skl3b_a
            
            title3_a = str(ppos[0]) + '_' + str(ypos) + '_gmm_not_interpolated_' + str(skl3_a)
            plotDistro((x_min,x_max), (y_min,y_max), title3_a, params = lerp_params_actual, color=red)
            '''
            
            #find quantile approx (include surface interpolant choice)
            samples_arr_a, evalfunc_a = interpFromQuantiles3(ppos=[ppos[0],ypos], ignore_cache = 'True', half=False)
            distro2_a, interpType_a, suc = computeDistroFunction(evalfunc_a[0],evalfunc_a[1],evalfunc_a[2], \
                                                             (x_min,x_max), (y_min,y_max))
            
            skl4f_a = kl_div_2D_M(mfunc1=distro2_a, mfunc2=mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl4b_a = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2_a, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl4_a = skl4f_a + skl4b_a
            
            title4_a = dt + str(ppos[0]) + '_' + str(ypos) + '_q_not_interpolated_skl_'   + str(skl4_a)
            plotXYZSurf((x_min,x_max), (y_min,y_max), distro2_a, title4_a, samples_arr_a, col=blue)
            plotXYZScatter((x_min,x_max), (y_min,y_max), evalfunc_a[0],evalfunc_a[1],evalfunc_a[2], title=dt + str(ppos[0]) + \
                           '_' + str(ypos) + '_NOT_Interp_', arr=samples_arr_a )
    
    print 'finished!'
            
    #find single gaussian approx
'''
    idx = 0
    for distro in kdes:
        mu_uv = np.mean(distro, axis=1)
        var_uv = np.var(distro, axis=1)
        cov = np.zeros(shape=(2,2)); cov[0,0] = var_uv[1]; cov[1,1] = var_uv[0]
        mean_params = [[(mu_uv[1], mu_uv[0]), cov, 1.0]]
        bivG = getBivariateGMM((x_min,x_max), (y_min,y_max), params = mean_params)
        
        skl2f = kl_div_2D_M(mfunc1, bivG, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl2b = kl_div_2D_M(bivG, mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        skl2 = skl2f + skl2b
        
        title2 = str(ppos[0]) + '_' + str(ypos) + '_g_'   + str(skl2)
        
        plotDistro((x_min,x_max), (y_min,y_max), title2, params = mean_params,color=purple)
        
        if idx == 1:
            mu_uv0 = np.mean(distro[0], axis=1)
            var_uv0 = np.var(distro[0], axis=1)
            cov0 = np.zeros(shape=(2,2)); cov0[0,0] = var_uv0[1]; cov0[1,1] = var_uv0[0]
            
            mu_uv1 = np.mean(distro[2], axis=1)
            var_uv1 = np.var(distro[2], axis=1)
            cov1 = np.zeros(shape=(2,2)); cov1[0,0] = var_uv1[1]; cov1[1,1] = var_uv1[0]
            
            mu_uv_i = 0.5*mu_uv0 + 0.5*mu_uv1
            var_uv_i = 0.5*var_uv0 + 0.5*var_uv1
            cov_i = np.zeros(shape=(2,2)); cov_i[0,0] = var_uv_i[1]; cov_i[1,1] = var_uv_i[0]
            
            mean_params = [[(mu_uv_i[1], mu_uv_i[0]), cov_i, 1.0]]
            bivG = getBivariateGMM((x_min,x_max), (y_min,y_max), params = mean_params)
            
            skl2f = kl_div_2D_M(mfunc1, bivG, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl2b = kl_div_2D_M(bivG, mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl2 = skl2f + skl2b
            
            title2 = str(ppos[0]) + '_' + str(ypos) + '_g_interpolated_'   + str(skl2)
            
            plotDistro((x_min,x_max), (y_min,y_max), title2, params = mean_params,color=purple)
        
        idx += 1
        
'''
            
             
'''  
        for g in range(2,11):
            NUM_GAUSSIANS = g
            MAX_GMM_COMP = g
            for itr in range(5,21,5):
                EM_MAX_ITR = itr
                title1 = str(ppos[0]) + '_' + str(ypos) + '_gmm_' + str(g) + '_iter_' + str(itr)
                lerp_params = interpFromGMM([ppos[0],ypos], ignore_cache = 'True')#float(float(idx)/10.)])
                plotDistro( (x_min,x_max), (y_min,y_max), title1, params = lerp_params,color='red' )
'''
            

def kl_div_2D_M(mfunc1,mfunc2,min_x=-5, max_x=5, min_y=-5, max_y=5):
    "Calculates the KL divergence D(A||B) between the distributions A and B.\nUsage: div = kl_divergence(A,B)"
    
    global DIV
    
    D = .0

    # Regular grid to evaluate upon
    u_vals = np.r_[min_x:max_x:DIV]
    v_vals = np.r_[min_y:max_y:DIV]
    
    #incr = math.fabs(min_x - max_x) / div
    for u in u_vals:
        for v in v_vals:
            if mfunc2[u,v] != .0 and mfunc1[u,v] != .0:
                D += mfunc1[u,v] * math.log( mfunc1[u,v] / mfunc2[u,v] ) 
    return D 
'''
#this impl can't be correct unless there is coversion between indices and u,v coords for both kde / mfunc
def kl_div_2D(mfunc,kde,min_x=-5, max_x=5, min_y=-5, max_y=5):
    "Calculates the KL divergence D(A||B) between the distributions A and B.\nUsage: div = kl_divergence(A,B)"
    D = .0
    #i = min_x
    div = 10j

    # Regular grid to evaluate kde upon
    u_vals = np.r_[min_x:max_x:div]
    v_vals = np.r_[min_y:max_y:div]
    
    #incr = math.fabs(min_x - max_x) / div
    for u in u_vals:
        for v in v_vals:
            if mfunc[u,v] != .0:
                #print A(i)
                D += mfunc[u,v] * math.log( mfunc[u,v] / kde([u,v])[0] ) 
                #print u
                #print v
                #print mfunc[u,v] * math.log( mfunc[u,v] / kde([u,v])[0] )
            else:
                D +=  kde([u,v])[0]
    return D 
'''
    
########################################################################################################

#from rpy2.robjects.numpy2ri import numpy2ri
#robjects.conversion.py2ri = numpy2ri
def getVclinSamplesSingle(gpt):
    
    global vclin
    
    gpt0_dist = np.zeros(shape=(2,MEM))
   
    for idx in range(0,MEM):
        gpt0_dist[0][idx] = vclin[idx][gpt[0]][gpt[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt[0]][gpt[1]][SEED_LEVEL][1]
   
    return gpt0_dist

#store in a subsampled array that can be accessed
#with existing interpolation code
def remapGridData():
    for idx in range(0,MEM):
        for lat in range(0,LAT):
            for lon in range(0,LON):
                if lat % 2 != 0 or lon % 2 != 0:
                    #print 'removed lat / lon values:'
                    #print lat
                    #print lon
                    continue
                
                mlat = 0; mlon = 0
                
                if lat == 0:
                    mlat = 0
                else:
                    mlat = pm.floor(lat / 2)
                    
                if lon == 0:
                    mlon = 0
                else:
                    mlon = pm.floor(lon / 2)
                    
                #print mlon
                #print mlat
                
                vclin_half[idx][mlat][mlon][0] = vclin[idx][lat][lon][SEED_LEVEL][0]
                vclin_half[idx][mlat][mlon][1] = vclin[idx][lat][lon][SEED_LEVEL][1]
                
                #print vclin_half[idx][mlat][mlon][0]
                #print vclin_half[idx][mlat][mlon][1]
        
def loadNetCdfData():
    global vclin
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #central forecasts reader 
    creader = NetcdfReader(pe_fct_aug25_sep2_file)
    vclin8 = creader.readVarArray('vclin', 7)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')  
    vclin = addCentralForecast(vclin, vclin8, level_start=SEED_LEVEL, level_end=SEED_LEVEL)  
    
        
import rpy2.robjects.numpy2ri as rpyn
#vector=rpyn.ri2numpy(vector_R)

def fitBvGmm(gp, max_gs=NUM_GAUSSIANS):
    #From numpy to rpy2:
    #http://rpy.sourceforge.net/rpy2/doc-2.2/html/numpy.html
    
    '''
    x
        A matrix of size nxp consisting of the data.
    lambda
        Initial value of mixing proportions. Entries should sum to 1. This determines number of components. If NULL, then lambda is random from uniform Dirichlet and number of components is determined by mu.
    mu
        A list of size k consisting of initial values for the p-vector mean parameters. If NULL, then the vectors are generated from a normal distribution with mean and standard deviation according to a binning method done on the data. If both lambda and mu are NULL, then number of components is determined by sigma.
    sigma
        A list of size k consisting of initial values for the pxp variance-covariance matrices. If NULL, then sigma is generated using the data. If lambda, mu, and sigma are NULL, then number of components is determined by k.
    k
        Number of components. Ignored unless lambda, mu, and sigma are all NULL.
    arbmean
        If TRUE, then the component densities are allowed to have different mus. If FALSE, then a scale mixture will be fit.
    arbvar
        If TRUE, then the component densities are allowed to have different sigmas. If FALSE, then a location mixture will be fit.
    epsilon
        The convergence criterion.
    maxit
        The maximum number of iterations.
    verb
    If TRUE, then various updates are printed during each iteration of the algorithm. 
    
    mvnormalmixEM(x, lambda = NULL, mu = NULL, sigma = NULL, k = 2,
              arbmean = TRUE, arbvar = TRUE, epsilon = 1e-08, 
              maxit = 10000, verb = FALSE)
    '''

    matrix = robjects.conversion.py2ri(gp)

    #suppress std out number of iterations using r.invisible()
    #http://www.inside-r.org/packages/cran/mixtools/docs/mvnormalmixEM
    #try:
    mixmdl = r.mvnormalmixEM(matrix, k = max_gs, maxit = EM_MAX_ITR, verb = True)
    #except:
    #    return [[0.]*max_gs,[0.]*max_gs, [0.]*max_gs ]
    
    mu = [];sigma = [];lb = []
    for i in mixmdl.iteritems():
        if i[0] == 'mu':
            mu.append(i[1])
        if i[0] == 'sigma':
            sigma.append(i[1])
        if i[0] == 'lambda':
            lb.append(i[1])
        
    n_params = [] 
    for idx in range(0,len(mu[0])):
        n_params.append([mu[0][idx],sigma[0][idx],lb[0][idx]])

    return n_params 

def kl_div_1D(A,B,min_x=-5, max_x=5):
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


###### Quantile interp code ########
def bilinearBivarQuantLerp(f1, f2, f3, f4, x1, y1, x2, y2, x3, y3, x4, y4, alpha, beta):
    a0 = 1.0 - alpha
    b0 = alpha
    a1 = 1.0 - beta
    b1 = beta
    
    try:
        f_one = f1((x1,y1))
        f_two = f2((x2,y2))
        f_three = f3((x3,y3))
        f_four = f4((x4,y4))            
        
        f_bar_0 = f_one * f_two / (a0*f_two + b0*f_one) 
        f_bar_1 = f_three * f_four / (a0*f_four + b0*f_three) 
        
        f_bar_01 = f_bar_0 * f_bar_1 / (a1*f_bar_1 + b1*f_bar_0)
    except:
        print 'problem with calculated interpolant z value...'
        f_bar_01[0] = -1 #failed
    
    return f_bar_01[0]

DIV = 50j 
DIVR = 50

def findBivariateQuantilesSinglePass(kde,arr):
    
    #cpu time
    qstart = time.clock() 
    
    global QUANTILES, DIV, DIVR
    
    
    u_min = arr.T[:,0].min()
    u_max = arr.T[:,0].max()
    v_min = arr.T[:,1].min()
    v_max = arr.T[:,1].max()
    
    # tolearance of cdf value, not a function of integration grid
    TOL = (1./QUANTILES) / 2. #np.max([((u_max - u_min) / divs), ((v_max - v_min) / divs)])
    
    u_extent = math.fabs( u_min - u_max )
    v_extent = math.fabs( v_min - v_max )
    
    #empirically determined ratio of 200 div per 10 units
    '''
    integ_div_ratio = 200. / 10.
    div_x = u_extent * integ_div_ratio
    div_y = v_extent * integ_div_ratio
    
    #empirically determined TOL should be 0.01 per 10 units
    #base this on largest extent
    TOL_RATIO = 0.01 / 10.
    TOL = 0.01
    if u_extent > v_extent:
        TOL = TOL_RATIO * u_extent
    else:
        TOL = TOL_RATIO * v_extent 
    
    #QUANTILES_RATIO = 150 / 10.
    '''
    
    incr_x = u_extent / DIVR#div_x 
    incr_y = v_extent / DIVR#div_y
    x_div = np.r_[u_min:u_max:DIV]
    y_div = np.r_[v_min:v_max:DIV]
    
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #integrate kde to find bivariate ecdf
    qs = list(spread(0.0, 1.0, QUANTILES-1, mode=3)) 
    qs.sort()
    
    qcurvex = []
    qcurvey = []
    for q in qs:
        qcurvex.append([])
        qcurvey.append([])
    
    for x in x_div:
        cd = 0.0
        
        for y in y_div:
            #print y
            low_bounds = (u_min,v_min)
            high_bounds = (x+incr_x,y+incr_y)
            
            cd = kde.integrate_box(low_bounds, high_bounds, maxpts=None)
            
            for idx, q in enumerate(qs):
                '''
                if ( q >= 0. and q <= 2. ) or ( q >= 98. and q <= 100. ): 
                    #higher tolerance for small quantiles and high quantiles
                    #to capture more data points if possible
                    if cd <= q + TOL+0.005 and cd >= q - TOL+0.005:
                        #print "gathering points for quantile curve #: " + str(idx) + " out of " + str(QUANTILES)
                        qcurvex[idx].append(x)
                        qcurvey[idx].append(y)
                else:
                '''
                if cd <= q + TOL and cd >= q - TOL:
                    #print "gathering points for quantile curve #: " + + str(idx) + " out of " + str(QUANTILES)
                    qcurvex[idx].append(x)
                    qcurvey[idx].append(y)
            
            z_pos.append(cd)
            x_pos.append(x)
            y_pos.append(y)
            
    print 'finished computing quantile curves'
    
    #cpu time
    qend = time.clock()
    qtot = qend - qstart
    print 'cpu time for KDE integration (EDCF): ' + str(qtot) 
            
    return x_pos, y_pos, z_pos, qcurvex, qcurvey

MID_RANGE_QUANTILE_CURVE_POINTS = 50

def lerpBivariate3(gp0, gp1, gp2, gp3, alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3, arr, use_cache=False):
    global g_grid_quantile_curves_array, QUANTILES, MID_RANGE_QUANTILE_CURVE_POINTS

    degree = 3;smoothing = None
    INTERP = 'linear'
    #spline_curve0=[];spline_curve1=[];spline_curve2=[];spline_curve3=[]
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex0 = gp0_qcurve[0]
    qcurvey0 = gp0_qcurve[1]
    spline_curve0 = gp0_qcurve[2]
    if len(gp0_qcurve[0]) == 0 or use_cache == False:
        print 'computing quantile curves gp0...'
        x_pos0, y_pos0, z_pos0, qcurvex0, qcurvey0 = findBivariateQuantilesSinglePass(gp0,arr[0])
        #plotXYZScatterQuants(qcurvex0, qcurvey0, title='qcurve0')
        spline_curve0 = []
        for q in range(0,len(qcurvex0)):
            if len(qcurvex0[q]) > degree: #must be greater than k value
                #spline_curve0.append(interpolate.UnivariateSpline(qcurvex0[q], qcurvey0[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve0.append(interpolate.interp1d(qcurvex0[q], qcurvey0[q], kind=INTERP))
            else:
                spline_curve0.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex0
        g_grid_quantile_curves_array[i][j][1] = qcurvey0
        g_grid_quantile_curves_array[i][j][2] = spline_curve0
        
    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex1 = gp1_qcurve[0]
    qcurvey1 = gp1_qcurve[1]
    spline_curve1 = gp1_qcurve[2]
    if len(gp1_qcurve[0]) == 0 or use_cache == False:
        print 'computing quantile curves gp1...'
        x_pos1, y_pos1, z_pos1, qcurvex1, qcurvey1 = findBivariateQuantilesSinglePass(gp1,arr[1])
        #plotXYZScatterQuants(qcurvex1, qcurvey1, title='qcurve1')
        spline_curve1 = []
        for q in range(0,len(qcurvex1)):
            if len(qcurvex1[q]) > degree:    
                #spline_curve1.append(interpolate.UnivariateSpline(qcurvex1[q], qcurvey1[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve1.append(interpolate.interp1d(qcurvex1[q], qcurvey1[q], kind=INTERP))
            else:
                spline_curve1.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex1
        g_grid_quantile_curves_array[i][j][1] = qcurvey1
        g_grid_quantile_curves_array[i][j][2] = spline_curve1
        
    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex2 = gp2_qcurve[0]
    qcurvey2 = gp2_qcurve[1]
    spline_curve2 = gp2_qcurve[2]
    if len(gp2_qcurve[0]) == 0 or use_cache == False:
        print 'computing quantile curves gp2...'
        x_pos2, y_pos2, z_pos2, qcurvex2, qcurvey2 = findBivariateQuantilesSinglePass(gp2,arr[2])
        #plotXYZScatterQuants(qcurvex2, qcurvey2, title='qcurve2')
        spline_curve2 = []
        for q in range(0,len(qcurvex2)):
            if len(qcurvex2[q]) > degree:
                #spline_curve2.append(interpolate.UnivariateSpline(qcurvex2[q], qcurvey2[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve2.append(interpolate.interp1d(qcurvex2[q], qcurvey2[q], kind=INTERP))
            else:
                spline_curve2.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex2
        g_grid_quantile_curves_array[i][j][1] = qcurvey2
        g_grid_quantile_curves_array[i][j][2] = spline_curve2
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex3 = gp3_qcurve[0]
    qcurvey3 = gp3_qcurve[1]
    spline_curve3 = gp3_qcurve[2]
    if len(gp3_qcurve[0]) == 0 or use_cache == False:
        print 'computing quantile curves gp3...'
        x_pos3, y_pos3, z_pos3, qcurvex3, qcurvey3 = findBivariateQuantilesSinglePass(gp3, arr[3])
        #plotXYZScatterQuants(qcurvex3, qcurvey3, title='qcurve3')
        spline_curve3 = []
        for q in range(0,len(qcurvex3)):
            if len(qcurvex3[q]) > degree:
                #spline_curve3.append(interpolate.UnivariateSpline(qcurvex3[q], qcurvey3[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve3.append(interpolate.interp1d(qcurvex3[q], qcurvey3[q], kind=INTERP))
            else:
                spline_curve3.append([None])
        g_grid_quantile_curves_array[i][j][0] = qcurvex3
        g_grid_quantile_curves_array[i][j][1] = qcurvey3
        g_grid_quantile_curves_array[i][j][2] = spline_curve3
        
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #smaller quantiles have longer quantile curves, so we adjust this number based on quantile below
    num_pts_to_eval_on_curve = MID_RANGE_QUANTILE_CURVE_POINTS
    for iq in range(0,QUANTILES):#, q in enumerate(qcurvex0):
        #if iq <= 0.5*QUANTILES:
        #    num_pts_to_eval_on_curve = 3*MID_RANGE_QUANTILE_CURVE_POINTS
        #else:
        #    num_pts_to_eval_on_curve = MID_RANGE_QUANTILE_CURVE_POINTS
            
        print str(iq) + "th quantile curve being lerped out of " + str(QUANTILES)
        #get an x,y pair for current quantile on each pdf end points
        #limit = min([len(qcurvex0[iq]), len(qcurvex1[iq]), len(qcurvex2[iq]), len(qcurvex3[iq])])
        #for idx in range(0,limit):
        epts0 = [];epts1 = [];epts2=[];epts3=[]
        cur_y0_parametrized_pts=[];cur_y1_parametrized_pts=[];cur_y2_parametrized_pts=[];cur_y3_parametrized_pts=[]
        if spline_curve0[iq] != None and spline_curve1[iq] != None \
            and spline_curve2[iq] != None and spline_curve3[iq] != None and len(qcurvex0[iq]) > degree and \
            len(qcurvex1[iq]) > degree and len(qcurvex2[iq]) > degree and len(qcurvex3[iq]) > degree:
            #if iq < int(0.25*QUANTILES):
            #    num_pts_to_eval_on_curve = int(2*MID_RANGE_QUANTILE_CURVE_POINTS )
            #elif iq > int(0.75*QUANTILES):
            #    num_pts_to_eval_on_curve = int(0.5*MID_RANGE_QUANTILE_CURVE_POINTS )
            print '    evaluating spline...'
            epts0 = linspace(qcurvex0[iq][0], qcurvex0[iq][-1], num_pts_to_eval_on_curve)
            epts1 = linspace(qcurvex1[iq][0], qcurvex1[iq][-1], num_pts_to_eval_on_curve)
            epts2 = linspace(qcurvex2[iq][0], qcurvex2[iq][-1], num_pts_to_eval_on_curve)
            epts3 = linspace(qcurvex3[iq][0], qcurvex3[iq][-1], num_pts_to_eval_on_curve) 
            cur_y0_parametrized_pts = spline_curve0[iq](epts0) 
            cur_y1_parametrized_pts = spline_curve1[iq](epts1)
            cur_y2_parametrized_pts = spline_curve2[iq](epts2)
            cur_y3_parametrized_pts = spline_curve3[iq](epts3)
            print '...finished evaluating spline!'
        else:
            continue
        
        for idx in range(0,num_pts_to_eval_on_curve): #evaluate points along each parameterized quantile curve
            print '    lerping point: ' +str(idx)+ ' out of ' + str(num_pts_to_eval_on_curve)
            cur_x0 = epts0[idx]#qcurvex0[iq][idx]
            cur_y0 = cur_y0_parametrized_pts[idx]#qcurvey0[iq][idx]
            cur_x1 = epts1[idx]#qcurvex1[iq][idx]
            cur_y1 = cur_y1_parametrized_pts[idx]#qcurvey1[iq][idx]
            cur_x2 = epts2[idx]#qcurvex2[iq][idx]
            cur_y2 = cur_y2_parametrized_pts[idx]#qcurvey2[iq][idx]
            cur_x3 = epts3[idx]#qcurvex3[iq][idx]
            cur_y3 = cur_y3_parametrized_pts[idx]#qcurvey3[iq][idx]
            
            dir_vec0 = np.asarray([cur_x1, cur_y1]) - np.asarray([cur_x0, cur_y0])
            dir_vec1 = np.asarray([cur_x3, cur_y3]) - np.asarray([cur_x2, cur_y2])
            
            '''
            dist0 = np.sqrt(np.dot(dir_vec0, dir_vec0))
            dist1 = np.sqrt(np.dot(dir_vec1, dir_vec1))
            
            dir_vec_n0 = None
            dir_vec_n1 = None
            
            if dist0 > 0.:
                dir_vec_n0 = dir_vec0 / dist0
            else:
                dist0 = 0.
                dir_vec_n0 = np.asarray([0.,0.])
            
            if dist1 > 0.:
                dir_vec_n1 = dir_vec1 / dist1
            else:
                dist1 = 0.
                dir_vec_n1 = np.asarray([0.,0.])
            
            interpolant_xy0 = np.asarray([cur_x0,cur_y0]) + alpha_x * dist0 * dir_vec_n0
            interpolant_xy1 = np.asarray([cur_x2,cur_y2]) + alpha_x * dist1 * dir_vec_n1
            '''
            
            interpolant_xy0 = np.asarray([cur_x0,cur_y0]) + alpha_x * dir_vec0
            interpolant_xy1 = np.asarray([cur_x2,cur_y2]) + alpha_x * dir_vec1
            
            dir_vec_bar = np.asarray([interpolant_xy1[0],interpolant_xy1[1]]) - \
                np.asarray([interpolant_xy0[0],interpolant_xy0[1]])
                
            '''
            dist_bar = np.sqrt(np.dot(dir_vec_bar, dir_vec_bar))
            
            dir_vec_bar_n = None
            
            if dist_bar > 0.:
                dir_vec_bar_n = dir_vec_bar / dist_bar
            else:
                dist_bar = 0.
                dir_vec_bar_n = np.asarray([0.,0.])
            
            interpolant_xy_bar = np.asarray([interpolant_xy0[0],interpolant_xy0[1]]) + \
                alpha_y * dist_bar * dir_vec_bar_n
            '''
            
            interpolant_xy_bar = np.asarray([interpolant_xy0[0],interpolant_xy0[1]]) + alpha_y * dir_vec_bar
            
            z = bilinearBivarQuantLerp(gp0, gp1, gp2, gp3, cur_x0, cur_y0, \
                                        cur_x1, cur_y1, cur_x2, cur_y2, cur_x3, cur_y3, \
                                         alpha_x, alpha_y)
            
            if z < 0. or math.isnan(z) or math.isinf(z):
                print 'WARNING: interpolated pdf has NaN, inf or negative result...skipping data point.'
                continue
           
            x_pos.append(interpolant_xy_bar[0])
            y_pos.append(interpolant_xy_bar[1])
            z_pos.append(z)  
            print '        lerping point between quantile curves: ' + str(iq) + ' was successful!'
            
    print 'finished lerping all quantile curve for interpolant distro...'
    return x_pos, y_pos, z_pos

def interpFromQuantiles3(ppos=[0.0,0.0], number=0,sl=0, ignore_cache = 'False', half=False):
    
    print ppos
    
    global g_grid_kde_array
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos, half)
    
    #gpt0_samp=None; gpt1_samp=None; gpt2_samp=None; gpt3_samp=None
    
    #only need to collect samples if we haven't already calculated kde's for grid cell
    #if g_grid_kde_array[int(gpt0[0])][int(gpt0[1])] is None and g_grid_kde_array[int(gpt1[0])][int(gpt1[1])] is None \
    #    and g_grid_kde_array[int(gpt2[0])][int(gpt2[1])] is None and g_grid_kde_array[int(gpt3[0])][int(gpt3[1])] is None:
    gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp = getVclinSamples(gpt0, gpt1, gpt2, gpt3, half)
    
    samples_arr = [gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp]
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_kde = g_grid_kde_array[i][j]
    if gp0_kde is None or ignore_cache == 'True':
        try:
            gp0_kde = stats.kde.gaussian_kde( ( gpt0_samp[0][:], gpt0_samp[1][:] ) )
        except:
            print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
           
        g_grid_kde_array[i][j] = gp0_kde

    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_kde = g_grid_kde_array[i][j]
    if gp1_kde is None or ignore_cache == 'True':
        try:
            gp1_kde = stats.kde.gaussian_kde( ( gpt1_samp[0][:], gpt1_samp[1][:] ) )
        
        except:
            print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
        
        g_grid_kde_array[i][j] = gp1_kde

    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_kde = g_grid_kde_array[i][j]
    if gp2_kde is None or ignore_cache == 'True':
        try:
            gp2_kde = stats.kde.gaussian_kde( ( gpt2_samp[0][:], gpt2_samp[1][:] ) )
        except:
            print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp2_kde
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_kde = g_grid_kde_array[i][j]
    if gp3_kde is None or ignore_cache == 'True':
        try:
            gp3_kde = stats.kde.gaussian_kde( ( gpt3_samp[0][:], gpt3_samp[1][:] ) )
        except:
            print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp3_kde

    #interp dist for x dim  
    alpha_x = 0.
    alpha_y = 0.
    if half == False:
        alpha_x = ppos_parts[0][0]
        alpha_y = ppos_parts[1][0]
    else:
        alpha_x = pm.modf(ppos_parts[0][1] / 2)[0]
        alpha_y = pm.modf(ppos_parts[1][1] / 2)[0]
    
    x3, y3, z3 = lerpBivariate3(gp0_kde, gp1_kde, gp2_kde, gp3_kde,\
                                            alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3, samples_arr, use_cache=half )
    
    
    x_min = np.asarray(gpt0_samp[0]).min()
    x_max = np.asarray(gpt0_samp[0]).max()
    y_min = np.asarray(gpt0_samp[1]).min()
    y_max = np.asarray(gpt0_samp[1]).max()
    
    '''
    for gp in samples_arr:
        if x_min > gp[0].min():
            x_min = gp[0].min()
        if x_max < gp[0].max():
            x_max = gp[0].max()
            
        if y_min > gp[1].min():
            y_min = gp[1].min()
        if y_max < gp[1].max():
            y_max = gp[1].max()
    
    plotKDE(gp0_kde,gpt0_samp, mx=(x_min,x_max), my=(y_min,y_max), title= "gpt0_kde", co = green)
    plotKDE(gp1_kde,gpt1_samp, mx=(x_min,x_max), my=(y_min,y_max), title="gpt1_kde", co = green)
    plotKDE(gp2_kde,gpt2_samp, mx=(x_min,x_max), my=(y_min,y_max), title="gpt2_kde", co = green)
    plotKDE(gp3_kde,gpt3_samp, mx=(x_min,x_max), my=(y_min,y_max), title="gpt3_kde", co = green)
    '''
    #plotXYZScatter(x3, y3, z3, str(number)+'final'+str(sl), arr=samples_arr )
    #plotXYZSurf(distro,title=str(number)+'final'+str(sl),arr=samples_arr )
    
    return samples_arr, [x3,y3,z3]

def computeDistroFunction(x_pos,y_pos,z_pos, mmx, mmy):
    
    global DIV, DIVR
   
    X = np.r_[mmx[0]:mmx[1]:DIV]
    Y = np.r_[mmy[0]:mmy[1]:DIV]
    X,Y = np.meshgrid(X,Y)
    
    pts = np.append(np.asarray(x_pos).reshape(-1,1),np.asarray(y_pos).reshape(-1,1),axis=1)
    
    
    distro_eval = None
    success = False
    interp_type = ''
    
    print "in computeDistroFunction..."
    
    while success is False:
        try:
            distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='linear', fill_value=0.0)
            success = True
            interp_type = 'linear'
        except:
            try:
                distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='cubic', fill_value=0.0)
                success = True
                interp_type = 'cubic'
            except:
                continue
    
    #distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
    #                                   xi=(X,Y), method='nearest', fill_value=0.0)
    
    #distro_eval = interpolate.Rbf(x_pos,y_pos,z_pos)#,epsilon=2)
    
    print "finished distro function comp..."
    print 'success? -> ' + str(success)
    print 'type? -> ' + str(interp_type)
    return distro_eval, interp_type, success

def plotXYZScatter((u_min,u_max),(v_min, v_max), x_pos,y_pos,z_pos, title = '', arr=[]):
    
    '''
    u_min = arr[0].T[:,0].min()
    u_max = arr[0].T[:,0].max()
    v_min = arr[0].T[:,1].min()
    v_max = arr[0].T[:,1].max()
    
    for d in arr:
        u_min_temp = d.T[:,0].min()
        if u_min_temp < u_min:
            u_min = u_min_temp
        
        u_max_temp = d.T[:,0].max()
        if u_max_temp > u_max:
            u_max = u_max_temp
        
        v_min_temp = d.T[:,1].min()
        if v_min_temp < v_min:
            v_min = v_min_temp
        
        v_max_temp = d.T[:,1].max()
        if v_max_temp > v_max:
            v_max = v_max_temp
    '''
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(list(x_pos), list(y_pos), list(z_pos),s=0.05)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(u_max, u_min)
    ax.set_ylim(v_min, v_max)
        
    ax.set_zlabel('density')
    ax.set_zlim(0, np.asarray(z_pos).max())
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "scatter.png")
    #plt.show()
    
def plotXYZSurf(mmx, mmy, surface, title = '', arr=[], col='0.75'):
    
    #plotXYZSurf((x_min,x_max), (y_min,y_max), distro2,'', samples_arr)
    
    global DIV, DIVR
    
    FIG = plt.figure()
    
    x_flat = np.r_[mmx[0]:mmx[1]:DIV]
    y_flat = np.r_[mmy[0]:mmy[1]:DIV]
    X,Y = np.meshgrid(x_flat,y_flat)
    
    AX = FIG.gca(projection='3d')
    AX.set_xlabel('u')
    AX.set_ylabel('v')
    AX.set_xlim(x_flat.max(), x_flat.min())
    AX.set_ylim(y_flat.min(), y_flat.max())
    AX.set_zlabel('density')
    AX.set_zlim(0, surface.max())
    AX.plot_surface(X, Y, surface, rstride=2, cstride=2, alpha=1.0, \
                    linewidth=0.1, antialiased=True, color=col)
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + ".jpg")
    #plt.show() 
    
from skimage import data
from skimage import measure
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.exposure as skie

def findBivariatePeaks(dist_fnt,method='e',streamline=0,step=0, arr=[]):
    
    u_min = arr[0].T[:,0].min()
    u_max = arr[0].T[:,0].max()
    v_min = arr[0].T[:,1].min()
    v_max = arr[0].T[:,1].max()
    for d in arr:
        u_min_temp = d.T[:,0].min()
        if u_min_temp < u_min:
            u_min = u_min_temp
        
        u_max_temp = d.T[:,0].max()
        if u_max_temp > u_max:
            u_max = u_max_temp
        
        v_min_temp = d.T[:,1].min()
        if v_min_temp < v_min:
            v_min = v_min_temp
        
        v_max_temp = d.T[:,1].max()
        if v_max_temp > v_max:
            v_max = v_max_temp
        
    # Regular grid to evaluate kde upon
    x_flat = np.r_[u_min:u_max:DIV]
    y_flat = np.r_[v_min:v_max:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = None
    
    if method == 'q':
        z = dist_fnt
        
        #if using griddata interp
        for idx_x in range(0,dist_fnt.shape[0]):
            for idx_y in range(0,dist_fnt.shape[1]):
                if math.isnan(z[idx_x][idx_y]):
                    z[idx_x][idx_y] = 0.0
                    
    #else:
    #if using Rbf with quantile
    #if method == 'q':
    #    z = dist_fnt(x,y)
    else:    
        z = dist_fnt(grid_coords.T)
                
        z = z.reshape(DIVR, DIVR)
    
    
    #fig = plt.figure()
    #i = plt.imshow(z)#,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower')
    #plt.show()
    
    limg = np.arcsinh(z)
    limg = limg / limg.max()
    low = np.percentile(limg, 0.25)
    high = np.percentile(limg, 99.9)
    opt_img = skie.exposure.rescale_intensity(limg, in_range=(low,high))
    lim = 0.6*high
    
    #http://scikit-image.org/docs/0.5/api/skimage.morphology.html
    #skimage.morphology.is_local_maximum
    #this needn't be too narrow, i.e. we don't want a per pixel resolution
    lm = None
    #if INTERP_METHOD == 'q': 
    #pl = 1
    fp = 9 #smaller odd values are more sensitive to local maxima in image (use roughtly 5 thru 13)
    #while pl != 2 and fp >= 1:
    lm = morph.is_local_maximum(limg,footprint=np.ones((fp, fp)))
    #    x1, y1 = np.where(lm.T == True)
    #    v = limg[(y1, x1)]
    #    pl = countPeaks(lim,limg,x1,y1)
    #    fp -= 2
    #else:
    #    lm = morph.is_local_maximum(limg)
        
    x1, y1 = np.where(lm.T == True)
    v = limg[(y1, x1)]
    
    
    #x2, y2 = x1[v > lim], y1[x > lim]
    
    peaks = [[],[]]
    for idx in range(0,len(x1)):
        if limg[(y1[idx],x1[idx])] > lim: 
            peaks[0].append(x1[idx])
            peaks[1].append(y1[idx])
    
    
    #print peaks
    #print x_flat[peaks[0]]
    #print y_flat[peaks[1]]
    
    num_peaks = len(peaks[0])
    
    peak_distances = []
    max_peak_distance = 0
    
    peak_vels = []
    peak_probs = []
    
    
    
    for idx in range(0,len(peaks[0][:])):
        peak_vels.append( ( x_flat[ peaks[0][idx] ], y_flat[ peaks[1][idx] ] ) )
        print 'peak vels: ' + str(( x_flat[ peaks[0][idx] ], y_flat[ peaks[1][idx] ] ))
        if method is not 'q':
            peak_probs.append( dist_fnt( ( x_flat[ peaks[0][idx] ], y_flat[ peaks[1][idx] ] ) )[0] )
        else:
            peak_probs.append( dist_fnt[x_flat[ int(peaks[0][idx]) ] ][ y_flat[ int(peaks[1][idx]) ] ] )
            
        
    
    if len(peaks[0]) > 1:
        for idx in range(0,len(peaks[0])):
            for idx2 in range(0,len(peaks[1])):
                peak_distances.append(math.sqrt(math.pow(x_flat[peaks[0][idx]]-x_flat[peaks[0][idx2]],2) \
                                       + math.pow(y_flat[peaks[1][idx]] - y_flat[peaks[1][idx2]],2)))
        max_peak_distance = max(peak_distances)  
        print " %%max peak dist%% = " + str(max_peak_distance) 
        
    if streamline == 0 or streamline == 1:
        fig = plt.figure()
        
        img = plt.imshow(np.rot90(opt_img,4),origin='lower',interpolation='bicubic')#,cmap=cm.spectral)
        
        #cb2 = fig2.colorbar(img2, shrink=0.5, aspect=5)
        #cb2.set_label('density estimate')
        ax = fig.add_subplot(111)
        ax.scatter(peaks[0],peaks[1], s=100, facecolor='none', edgecolor = '#009999')
        
        
        plt.savefig(OUTPUT_DATA_DIR + method + '_sl_number_' + str(streamline) +'_step_' + str(step) + '_peaks.png')
        #plt.show()
    
    
    return peak_vels, peak_probs, max_peak_distance

if __name__ == "__main__":  
    main()

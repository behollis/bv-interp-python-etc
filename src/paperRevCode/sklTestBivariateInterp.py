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
cf_vclin = []

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

EM_MAX_ITR = 15
EM_MAX_RESTARTS = 1000
DEPTH = -2.0
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
NUM_GAUSSIANS = 3#2
MAX_GMM_COMP = 3#NUM_GAUSSIANS 

g_cc = np.zeros(shape=(LAT,LON))

g_part_positions_ensemble = [[],[],[]]

#part_pos_e = []
#part_pos_e.append([0,0])
#part_pos_e[0][0] = SEED_LAT
#part_pos_e[0][1] = SEED_LON

r = robjects.r

ZERO_ARRAY = np.zeros(shape=(MEM,1))

SAMPLES = 600
vclin_x = np.ndarray(shape=(SAMPLES,LAT,LON))
vclin_y = np.ndarray(shape=(SAMPLES,LAT,LON))

divs = 100
div = complex(divs)
div_real = divs
start = -3
end = +3
TOL = 0.01


def lerpBivGMMPair(norm_params1, norm_params2, alpha, steps=1, num_gs=3):     
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

#3d np array for storing parameters found for grid points
#check this array first before fitting g comps
g_grid_params_array = []

def createGlobalParametersArray(dimx, dimy):
    global g_grid_params_array
    
    for idx in range(0,dimx):
        g_grid_params_array.append([])
        for idy in range(0,dimy):
            g_grid_params_array[idx].append([])


#....................... bivariate interp helper functions

def getCoordParts(ppos=[0.0,0.0]):
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    return ppos_parts 

def getGridPoints(ppos=[0.0,0.0]):
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
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    return gpt0, gpt1, gpt2, gpt3

def getVclinSamples(gpt0, gpt1, gpt2, gpt3):
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
        
        gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
   
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

def plotKDE(kde,distro,title):
    div = 200j
    div_real = 200
    
    FIG = plt.figure()
    
    x_flat = np.r_[np.asarray(distro[0]).min():np.asarray(distro[0]).max():div]
    y_flat = np.r_[np.asarray(distro[1]).min():np.asarray(distro[1]).max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    AX = FIG.gca(projection='3d')
    AX.set_xlabel('u')
    AX.set_ylabel('v')
    AX.set_xlim(x_flat.max(), x_flat.min())
    AX.set_ylim(y_flat.min(), y_flat.max())
    AX.set_zlabel('density')
    AX.set_zlim(0, z.max())
    AX.plot_surface(x, y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True, alpha=0.2, color='green')
    plt.savefig(OUTPUT_DATA_DIR + title + ".jpg")
    #plt.show() 
  
def getKDE(x_min_max,y_min_max,kde):
    div = 200j
    div_real = 200
    
    #x_flat = np.r_[np.asarray(distro[0]).min():np.asarray(distro[0]).max():div]
    #y_flat = np.r_[np.asarray(distro[1]).min():np.asarray(distro[1]).max():div]
    #x,y = np.meshgrid(x_flat,y_flat)
    
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    return z
  
def getBivariateGMM(x_min_max,y_min_max,params = [0.0,0.0,0.0]):
    div = 200j
    div_real = 200

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
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
    
    div = 200j
    div_real = 200

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
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
    surf = ax.plot_surface(x, y, Z_total, rstride=2, cstride=2, linewidth=0.1, antialiased=True, alpha=0.2, color=color)#,,cmap=cm.spectral)
    
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

def plotSpline(spline,x_min_max,y_min_max,title=''):
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    
    z = spline.reshape(div_real,div_real)
    
    fig = plt.figure(title)
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
   
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('kde')
    
    #plt.savefig(OUTPUT_DATA_DIR + str(title) + "5.png")
          
    plt.show()    
 
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
  
def interpFromGMM(ppos=[0.0,0.0], ignore_cache = 'False'):
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gp0, gp1, gp2, gp3 = getGridPoints(ppos)
    
    gp0_dist, gp1_dist, gp2_dist, gp3_dist = getVclinSamples(gp0, gp1, gp2, gp3)
    
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
    
def main():
    #gen_streamlines = str(sys.argv[1])
    
    global NUM_GAUSSIANS, MAX_GMM_COMP, EM_MAX_ITR
      
    NUM_GAUSSIANS = 2
    MAX_GMM_COMP = 2
    EM_MAX_ITR = 5
    
    r.library('mixtools')
    loadNetCdfData()
    createGlobalParametersArray(LAT,LON)
    
    #vclin = np.zeros(shape=(10,10,2))
    
    #defineVclin()
    #createGlobalParametersArray(LAT,LON)
    
    SEED_LAT = 4 #x dim
    SEED_LON = 4 #y dim
    
    #python -m cProfile -o outputfile.profile nameofyour_program alpha
    
    #alpha = float(sys.argv[1])
    
    if True:#gen_streamlines == 'True':
        #print "generating streamlines"
        
        particle = 0
        #part_pos_e[particle][0] = SEED_LAT; part_pos_e[particle][1] = SEED_LON
        
        #g_part_positions_ensemble[0].append(SEED_LAT)
        #g_part_positions_ensemble[1].append(SEED_LON) 
        #g_part_positions_ensemble[2].append(DEPTH) 
        
        ppos = [44.0,31.0]
       
          
        for idx in range(0,3):#,11):
            ypos = ppos[1] + idx 
            
            print idx
            distro = getVclinSamplesSingle([ppos[0],ypos])
            kde = stats.kde.gaussian_kde(distro)
            
            x_min = np.asarray(distro[0]).min()
            x_max = np.asarray(distro[0]).max()
            y_min = np.asarray(distro[1]).min()
            y_max = np.asarray(distro[1]).max()
            
            mfunc1 = getKDE((x_min,x_max), (y_min,y_max),kde)
            
            #find single gaussian
            mu_uv = np.mean(distro, axis=1)
            var_uv = np.var(distro, axis=1)
            cov = np.zeros(shape=(2,2)); cov[0,0] = var_uv[1]; cov[1,1] = var_uv[0]
            mean_params = [[(mu_uv[1], mu_uv[0]), cov, 1.0]]
            bivG = getBivariateGMM((x_min,x_max), (y_min,y_max), params = mean_params)
            
            lerp_params = interpFromGMM([ppos[0],ypos], ignore_cache = 'True')
            bivGMM = getBivariateGMM((x_min,x_max), (y_min,y_max), params = lerp_params)
            
            skl1 = kl_div_2D_M(mfunc1, mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            
            skl2f = kl_div_2D_M(mfunc1, bivG, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl2b = kl_div_2D_M(bivG, mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl2 = skl2f + skl2b
            
            skl3f = kl_div_2D_M(mfunc1=bivGMM, mfunc2=mfunc1, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl3b = kl_div_2D_M(mfunc1=mfunc1, mfunc2=bivGMM, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
            skl3 = skl3f + skl3b
            
            title1 = str(ppos[0]) + '_' + str(ypos) + '_kde_' + str(skl1)
            title2 = str(ppos[0]) + '_' + str(ypos) + '_g_' + str(skl2)
            title3 = str(ppos[0]) + '_' + str(ypos) + '_gmm_' + str(skl3)
            
            lerp_params = interpFromGMM([ppos[0],ypos], ignore_cache = 'True')
             
            plotKDE(kde,distro, title1)
            plotDistro((x_min,x_max), (y_min,y_max), title2, params = mean_params,color='purple')
            plotDistro((x_min,x_max), (y_min,y_max), title3, params = lerp_params,color='red')
           
            #plotDistro( (x_min,x_max), (y_min,y_max), title3, params = mean_params,color='purple' ) 
          
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
    D = .0
    #i = min_x
    div = 10j

    # Regular grid to evaluate kde upon
    u_vals = np.r_[min_x:max_x:div]
    v_vals = np.r_[min_y:max_y:div]
    
    #incr = math.fabs(min_x - max_x) / div
    for u in u_vals:
        for v in v_vals:
            if mfunc1[u,v] != .0:
                #print A(i)
                D += mfunc1[u,v] * math.log( mfunc1[u,v] / mfunc2[u,v] ) 
                #print u
                #print v
                #print mfunc[u,v] * math.log( mfunc[u,v] / kde([u,v])[0] )
            else:
                D +=  mfunc1[u,v]
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

if __name__ == "__main__":  
    main()

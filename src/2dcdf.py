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


q_prev_max_vel_x = 0.0
q_prev_max_vel_y = 0.0
e_prev_max_vel_x = 0.0
e_prev_max_vel_y = 0.0
gmm_prev_max_vel_x = 0.0
gmm_prev_max_vel_y = 0.0

QUANTILES = 100
TOTAL_STEPS = 25
integration_step_size = 0.1
SEED_LEVEL = 0
vclin = []
cf_vclin = []

reused_vel_quantile = 0

DEBUG = False
  
MODE = 1
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/pics/bv_interp/quant6/'
MODE_DIR1 = 'mode1/'
MODE_DIR2 = 'mode2/'
COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 
MAX_GMM_COMP = 3 
EM_MAX_ITR = 2000
EM_MAX_RESTARTS = 1000
DEPTH = -2.0
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak

g_cc = np.zeros(shape=(LAT,LON))

g_part_positions_ensemble = [[],[],[]]

#part_pos_e = []
#part_pos_e.append([0,0])
#part_pos_e[0][0] = SEED_LAT
#part_pos_e[0][1] = SEED_LON

r = robjects.r

ZERO_ARRAY = np.zeros(shape=(MEM,1))

SAMPLES = 500
vclin_x = np.ndarray(shape=(1000,10,10))
vclin_y = np.ndarray(shape=(1000,10,10))


def defineVclin():
    div = 50j
    div_real = 50
    
    #left half
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    
    x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    
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
    mean2 = [0,+1]
    cov2 = [[1,0],[0,1]] 
    
    x2,y2 = np.random.multivariate_normal(mean2,cov2,SAMPLES).T
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()    
    
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
    mean3 = [-2,+1]
    cov3 = [[0.5,0],[0,0.5]] 
    mean4 = [+2,-1]
    cov4 = [[1.0,0],[0,1.0]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,SAMPLES/2).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,SAMPLES/2).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    
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
        
    kde = stats.kde.gaussian_kde((x_tot2,y_tot2))#gp1_2d_array.T)

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

def testECDF():

    div = 50j
    div_real = 50
    
    
    #left half
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    
    x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
     
      
    kde2 = stats.kde.gaussian_kde((x1,y1))#gp1_2d_array.T)
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[x1.min():x1.max():div]
    y_flat = np.r_[y1.min():y1.max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde2(grid_coords.T)

    z = z.reshape(div_real,div_real)
    
    '''
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
    
    
    #right half
    mean2 = [0,+1]
    cov2 = [[1,0],[0,1]] 
    
    x2,y2 = np.random.multivariate_normal(mean2,cov2,SAMPLES).T
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()    
    
    
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
    mean3 = [-2,+1]
    cov3 = [[0.5,0],[0,0.5]] 
    mean4 = [+2,-1]
    cov4 = [[1.0,0],[0,1.0]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,SAMPLES/2).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,SAMPLES/2).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
    
    #gp1_2d_array = np.append(x1,y1,axis=1)
    #gp2_2d_array = np.append(x2,y2,axis=1)
       
    
    kde = stats.kde.gaussian_kde((x_tot,y_tot))#gp1_2d_array.T)

    '''
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
          
    plt.show() 
    '''

    
    start = -4
    end = +4
    divs = 50
    
    incr = math.fabs( start - end ) / divs 
    x_div = np.r_[start:end:complex(divs)]
    y_div = np.r_[start:end:complex(divs)]
    
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #integrate kde to find bivariate ecdf
    QUANTILES = 20
    quantiles = list(spread(0, 1.0, QUANTILES-1, mode=3)) 
    quantiles.sort()
    
    qcurvex = []
    qcurvey = []
    TOL = 0.005
    q = 0.3
    
    for y in y_div:
        csd = 0.0
        for x in x_div:
            low_bounds = (start,start)
            high_bounds = (x+incr,y+incr)
            
            csd = kde.integrate_box(low_bounds, high_bounds, maxpts=None)
            
            if csd <= q + TOL and csd >= q - TOL:
                qcurvex.append(x)
                qcurvey.append(y)
            
            z_pos.append(csd)
            x_pos.append(x)
            y_pos.append(y)
    
    fig = plt.figure()
    
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    X, Y = np.meshgrid(X, Y)
    z_pos = np.asarray(z_pos).reshape(divs,divs)
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, z_pos, rstride=1, cstride=1, alpha=0.7, linewidth=0.1, antialiased=True,cmap=cm.spectral)
    ax.plot(qcurvex,qcurvey, np.zeros(shape=(len(qcurvex))))
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_div.min(), x_div.max())
    ax.set_ylim(y_div.min(), y_div.max())
        
    ax.set_zlabel('ecdf')
    ax.set_zlim(0, np.asarray(z_pos).max())
    
    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('cumulative density')
          
    plt.show() 
    
if __name__ == "__main__":  
    
    #gen_streamlines = str(sys.argv[1])
    
    vclin = np.zeros(shape=(10,10,2))
    
    testECDF()
    
   

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
OUTPUT_DATA_DIR = '../../data/out/pics/bv_interp/qparm10/'
#OUTPUT_DATA_DIR = '../../data/out/csv/toy4/bvINTERP3/'
MODE_DIR1 = 'mode1/'
MODE_DIR2 = 'mode2/'
COM =  2
LON = 9
LAT = 9
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

SAMPLES = 1000

vclin_x = np.ndarray(shape=(SAMPLES,20,20))
vclin_y = np.ndarray(shape=(SAMPLES,20,20))

QUANTILES = 150
divs = 200
div = complex(divs)
div_real = divs
start = -5#works with +/-5
end = +5
TOL = 0.01 #( 1.0 / QUANTILES ) / 3.0


#3d np array for storing parameters found for grid points
#check this array first before fitting g comps
g_grid_kde_array = []
g_grid_quantile_curves_array = []

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
    
    for idx in range(0,SAMPLES):
        gpt0_dist[0][idx] = vclin_x[idx][gpt0[0]][gpt0[1]]
        gpt0_dist[1][idx] = vclin_y[idx][gpt0[0]][gpt0[1]]
        
        gpt1_dist[0][idx] = vclin_x[idx][gpt1[0]][gpt1[1]]
        gpt1_dist[1][idx] = vclin_y[idx][gpt1[0]][gpt1[1]]
        
        gpt2_dist[0][idx] = vclin_x[idx][gpt2[0]][gpt2[1]]
        gpt2_dist[1][idx] = vclin_y[idx][gpt2[0]][gpt2[1]]
        
        gpt3_dist[0][idx] = vclin_x[idx][gpt3[0]][gpt3[1]]
        gpt3_dist[1][idx] = vclin_y[idx][gpt3[0]][gpt3[1]]
   
    return gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist
 
def plotDistro(kde,x_min_max,y_min_max,title=''):
    
    div = 50j
    div_real = 50
    
    

    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=2, cstride=2, alpha=0.1, linewidth=0, antialiased=True, color='b')#,cmap=cm.spectral)
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
        
    #ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
    
    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
        
    ax.set_zlabel('density')
    #ax.set_zlim(0, np.asarray(z_pos).max())
   
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('kde')
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "20.png")
          
    #plt.show()    
 
def plotXYZ(x_tot2,y_tot2,z,title=''):
    fig = plt.figure(title)
                        
    ax = fig.gca(projection='3d')
    #surf = ax.scatter(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('')
          
    plt.show()    
 
def defineVclin():
    div = 50j
    div_real = 50
    
    #left half
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    
    x1,y1 = np.float64(np.random.multivariate_normal(mean1,cov1,SAMPLES).T)
    
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
    #mean3 = [+3,0]
    #cov3 = [[1,0],[0,1]] 
    #mean4 = [-2,0]
    #cov4 = [[1,0],[0,1]] 
    
    mean3 = [+2,1]
    cov3 = [[1,0],[0,1]] 
    mean4 = [-2,-1]
    cov4 = [[1.5,0],[0,1.5]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.6*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.4*SAMPLES)).T
    
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
    for x in range(0,LAT):
        for y in range(0,LON):
            for idx in range(0,SAMPLES):
                if x <= 4: 
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
from scipy.interpolate import SmoothBivariateSpline

def plotXYZSurf(surface, title = ''):
    incr = math.fabs( start - end ) / divs 
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    X, Y = np.meshgrid(X, Y)
                   
    fig = plt.figure()      
    ax = fig.gca(projection='3d')
 
    ax.plot_surface(X, Y, surface.T, rstride=1, cstride=1, alpha=1.0, linewidth=0.1, antialiased=True)
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
        
    ax.set_zlabel('density')
    
    #ax.set_zlim(0, np.asarray(z_pos).max())
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('density')
          
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "surface5.png")   
      
    #plt.show() 
    
    fig2 = plt.figure()
    plt.imshow(np.rot90(surface.T,+1), extent=(start,end,start,end),interpolation='bicubic')
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "top5.png") 
    #plt.show() 

def plotXYZScatter(x_pos,y_pos,z_pos, title = ''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(list(x_pos), list(y_pos), list(z_pos),s=0.05)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
        
    ax.set_zlabel('density')
    ax.set_zlim(0, np.asarray(z_pos).max())
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "scatter5.png")
    #plt.show()

def plotXYZScatterQuants(quantx, quanty, title = ''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    for idx in range(0,QUANTILES):
        r = np.random.ranf()#;print r
        g = np.random.ranf()#;print g
        b = np.random.ranf()#;print b
        ax.scatter(quantx[idx], quanty[idx], 0.,c=(r,g,b))
        
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    
    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
        
    #ax.set_zlabel('density')
    #ax.set_zlim(0, 1.0)
    #plt.savefig(OUTPUT_DATA_DIR + str(title) + "scatterQuants.png")
    plt.show()
    
    
def plotBivariateECDF(x_pos, y_pos, z_pos, qcurvex, qcurvey):
    
    incr = math.fabs( start - end ) / divs 
    x_div = np.r_[start:end:complex(divs)]
    y_div = np.r_[start:end:complex(divs)]
    
    fig = plt.figure()
    
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    X, Y = np.meshgrid(X, Y)
    z_pos = np.asarray(z_pos).reshape(divs,divs)
                        
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, z_pos, rstride=1, cstride=1, alpha=0.7, linewidth=0.1, antialiased=True,cmap=cm.spectral)
    
    for idx in range(0,len(qcurvex)):
        ax.plot(np.asarray(qcurvex[idx]),np.asarray(qcurvey[idx]), 0.0)
        
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
    
def bilinearBivarQuantLerp(f1, f2, f3, f4, x1, y1, x2, y2, x3, y3, x4, y4, alpha, beta):
    a0 = 1.0 - alpha
    b0 = alpha
    a1 = 1.0 - beta
    b1 = beta
    
    f_one = f1((x1,y1))
    f_two = f2((x2,y2))
    f_three = f3((x3,y3))
    f_four = f4((x4,y4))            
    
    f_bar_0 = f_one * f_two / (a0*f_two + b0*f_one) 
    f_bar_1 = f_three * f_four / (a0*f_four + b0*f_three) 
    
    f_bar_01 = f_bar_0 * f_bar_1 / (a1*f_bar_1 + b1*f_bar_0)
    
    return f_bar_01[0]
    
def findBivariateQuantilesSinglePass(kde):
    
    global QUANTILES
    #divs = 100
    incr = math.fabs( start - end ) / divs 
    x_div = np.r_[start:end:complex(divs)]
    y_div = np.r_[start:end:complex(divs)]
    
    x_pos = []
    y_pos = []
    z_pos = [] 
    
    #integrate kde to find bivariate ecdf
    qs = list(spread(0.0, 1.0, QUANTILES, mode=3)) 
    print 'number of quantile curves to find ' + str(len(qs))
    print qs[0]
    print qs[1]
    print qs[2]
    print qs[-1]
    print TOL
    qs.sort()
    
    qcurvex = []
    qcurvey = []
    for q in qs:
        qcurvex.append([])
        qcurvey.append([])
    
    for x in x_div:
        cd = 0.0
        
        for y in y_div:
            low_bounds = (start,start)
            high_bounds = (x+incr,y+incr)
            
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
                if cd <= q + TOL and cd > q - TOL:
                    #print "gathering points for quantile curve #: " + + str(idx) + " out of " + str(QUANTILES)
                    qcurvex[idx].append(x)
                    qcurvey[idx].append(y)
            
            z_pos.append(cd)
            x_pos.append(x)
            y_pos.append(y)
            
    print 'finished computing quantile curves'
            
    return x_pos, y_pos, z_pos, qcurvex, qcurvey

MID_RANGE_QUANTILE_CURVE_POINTS = 80

def lerpBivariate3(gp0, gp1, gp2, gp3, alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3):
    global g_grid_quantile_curves_array, QUANTILES, MID_RANGE_QUANTILE_CURVE_POINTS

    degree = 3;smoothing = None
    #spline_curve0=[];spline_curve1=[];spline_curve2=[];spline_curve3=[]
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex0 = gp0_qcurve[0]
    qcurvey0 = gp0_qcurve[1]
    spline_curve0 = gp0_qcurve[2]
    if len(gp0_qcurve[0]) == 0:
        print 'computing quantile curves gp0...'
        x_pos0, y_pos0, z_pos0, qcurvex0, qcurvey0 = findBivariateQuantilesSinglePass(gp0)
        #plotXYZScatterQuants(qcurvex0, qcurvey0, title='qcurve0')
        spline_curve0 = []
        for q in range(0,len(qcurvex0)):
            if len(qcurvex0[q]) > degree: #must be greater than k value
                spline_curve0.append(interpolate.UnivariateSpline(qcurvex0[q], qcurvey0[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                #spline_curve0.append(interpolate.interp1d(qcurvex0[q], qcurvey0[q], kind='linear'))
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
    if len(gp1_qcurve[0]) == 0:
        print 'computing quantile curves gp1...'
        x_pos1, y_pos1, z_pos1, qcurvex1, qcurvey1 = findBivariateQuantilesSinglePass(gp1)
        #plotXYZScatterQuants(qcurvex1, qcurvey1, title='qcurve1')
        spline_curve1 = []
        for q in range(0,len(qcurvex1)):
            if len(qcurvex1[q]) > degree:    
                spline_curve1.append(interpolate.UnivariateSpline(qcurvex1[q], qcurvey1[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                #spline_curve1.append(interpolate.interp1d(qcurvex1[q], qcurvey1[q], kind='linear'))
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
    if len(gp2_qcurve[0]) == 0:
        print 'computing quantile curves gp2...'
        x_pos2, y_pos2, z_pos2, qcurvex2, qcurvey2 = findBivariateQuantilesSinglePass(gp2)
        #plotXYZScatterQuants(qcurvex2, qcurvey2, title='qcurve2')
        spline_curve2 = []
        for q in range(0,len(qcurvex2)):
            if len(qcurvex2[q]) > degree:
                spline_curve2.append(interpolate.UnivariateSpline(qcurvex2[q], qcurvey2[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                #spline_curve2.append(interpolate.interp1d(qcurvex2[q], qcurvey2[q], kind='linear'))
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
    if len(gp3_qcurve[0]) == 0:
        print 'computing quantile curves gp3...'
        x_pos3, y_pos3, z_pos3, qcurvex3, qcurvey3 = findBivariateQuantilesSinglePass(gp3)
        #plotXYZScatterQuants(qcurvex3, qcurvey3, title='qcurve3')
        spline_curve3 = []
        for q in range(0,len(qcurvex3)):
            if len(qcurvex3[q]) > degree:
                spline_curve3.append(interpolate.UnivariateSpline(qcurvex3[q], qcurvey3[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                #spline_curve3.append(interpolate.interp1d(qcurvex3[q], qcurvey3[q], kind='linear'))
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
            
            dir_vec_bar = np.asarray([interpolant_xy1[0],interpolant_xy1[1]]) - \
                np.asarray([interpolant_xy0[0],interpolant_xy0[1]])
            dist_bar = np.sqrt(np.dot(dir_vec_bar, dir_vec_bar))
            
            dir_vec_bar_n = None
            
            if dist_bar > 0.:
                dir_vec_bar_n = dir_vec_bar / dist_bar
            else:
                dist_bar = 0.
                dir_vec_bar_n = np.asarray([0.,0.])
            
            interpolant_xy_bar = np.asarray([interpolant_xy0[0],interpolant_xy0[1]]) + alpha_y * dist_bar * dir_vec_bar_n
            
            z = bilinearBivarQuantLerp(gp0, gp1, gp2, gp3, cur_x0, cur_y0, \
                                        cur_x1, cur_y1, cur_x2, cur_y2, cur_x3, cur_y3, \
                                         alpha_x, alpha_y)
            
            if z < 0. or math.isnan(z) or math.isinf(z):
                continue
           
            x_pos.append(interpolant_xy_bar[0])
            y_pos.append(interpolant_xy_bar[1])
            z_pos.append(z)  
            print '        lerping point between quantile curves: ' + str(iq) + ' was successful!'
            
    print 'finished lerping all quantile curve for interpolant distro...'
    return x_pos, y_pos, z_pos
    
def interpFromQuantiles3(ppos=[0.0,0.0], number=0):
    
    print ppos
    
    global g_grid_kde_array
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos)
    
    gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp = getVclinSamples(gpt0, gpt1, gpt2, gpt3)
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_kde = g_grid_kde_array[i][j]
    if gp0_kde is None:
        gp0_kde = stats.kde.gaussian_kde( ( gpt0_samp[0][:], gpt0_samp[1][:] ) )
        g_grid_kde_array[i][j] = gp0_kde

    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_kde = g_grid_kde_array[i][j]
    if gp1_kde is None:
        gp1_kde = stats.kde.gaussian_kde( ( gpt1_samp[0][:], gpt1_samp[1][:] ) )
        g_grid_kde_array[i][j] = gp1_kde

    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_kde = g_grid_kde_array[i][j]
    if gp2_kde is None:
        gp2_kde = stats.kde.gaussian_kde( ( gpt2_samp[0][:], gpt2_samp[1][:] ) )
        g_grid_kde_array[i][j] = gp2_kde
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_kde = g_grid_kde_array[i][j]
    if gp3_kde is None:
        gp3_kde = stats.kde.gaussian_kde( ( gpt3_samp[0][:], gpt3_samp[1][:] ) )
        g_grid_kde_array[i][j] = gp3_kde

    #interp dist for x dim  
    alpha_x = ppos_parts[0][0]
    alpha_y = ppos_parts[1][0]
    
    x3, y3, z3 = lerpBivariate3(gp0_kde, gp1_kde, gp2_kde, gp3_kde,\
                                            alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3)
    
    distro = computeDistroFunction(x3,y3,z3)
    
    plotXYZScatter(x3, y3, z3, title=str(number)+'final' )
    plotXYZSurf(distro,title=str(number)+'final' )
    
    return distro 

def computeDistroFunction(x_pos,y_pos,z_pos):
    print 'computing distribution function...'
    incr = math.fabs( start - end ) / divs 
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    Y, X = np.meshgrid(X, Y)
    pts = np.append(np.asarray(x_pos).reshape(-1,1),np.asarray(y_pos).reshape(-1,1),axis=1)
   
    success = False
    while success is False:
        try:
            distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='linear')#, fill_value=0.0)
            success = True
        except:
            try:
                distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='cubic')#, fill_value=0.0)
                success = True
            except:
                continue
        
        
    
    return distro_eval
        

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
    
def main():
    #gen_streamlines = str(sys.argv[1])
    
    vclin = np.zeros(shape=(10,10,2))
    
    defineVclin()
    
    SEED_LAT = 4 #x dim
    SEED_LON = 4 #y dim
    
    #python -m cProfile -o outputfile.profile nameofyour_program alpha
    
    #alpha = float(sys.argv[1])
    
    if True:#gen_streamlines == 'True':
        
        createGlobalKDEArray(LAT, LON)
        createGlobalQuantileArray(LAT,LON)
        
        print "generating streamlines"
        
        particle = 0
        #part_pos_e[particle][0] = SEED_LAT; part_pos_e[particle][1] = SEED_LON
        
        g_part_positions_ensemble[0].append(SEED_LAT)
        g_part_positions_ensemble[1].append(SEED_LON) 
        g_part_positions_ensemble[2].append(DEPTH) 
        
        ppos = [SEED_LAT,SEED_LON]
        
        '''
        for idx in range(0,20):
            kde = interpVelFromEnsemble([ppos[0],ppos[1]+idx/10.])
            plotDistro( kde, (-4,4), (-4,4), str(idx) + '_bivar_ensem_lerp_' ) 
        '''
         
          
        for idx in range(0,11):
            #kde = interpFromQuantiles([ppos[0],ppos[1]+idx/10.])
            print idx
            kde = interpFromQuantiles3([ppos[0],ppos[1]+float(idx/10.)], idx)
            #plotDistro( kde, (-7,7), (-7,7), str(idx) + '_bivar_quant_lerp_' )
            
            #kde = interpVelFromEnsemble([ppos[0],ppos[1]+idx/10.])
            #plotDistro( kde, (-4,4), (-4,4), str(idx) + '_bivar_ensem_lerp_' ) 
    
        
        
        #kde = interpVelFromEnsemble([ppos[0],ppos[1]+alpha])
        #plotDistro( kde, (-4,4), (-4,4), str(alpha) + '_bivar_ensem_lerp_' ) 
        
        
        
        #kde2 = interpFromQuantiles([ppos[0],ppos[1]+alpha])
        #plotDistro( kde2, (-4,4), (-4,4), str(alpha) + '_bivar_quant_lerp_' )
        
        
        
    else:
        print "reading particles"
        #readParticles()  
        #plotParticles(ts_per_gp)
        
    print "finished!"

if __name__ == "__main__":  
    main()
    
            
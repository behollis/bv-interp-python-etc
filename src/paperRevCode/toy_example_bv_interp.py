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
import time
from cv2 import *

dt = 'stampMicroSec_' + str(datetime.datetime.now().microsecond) + '_' 

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

EM_MAX_ITR = 5
EM_MAX_RESTARTS = 1000
DEPTH = -2.0
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
NUM_GAUSSIANS = 4#2
MAX_GMM_COMP = 4#NUM_GAUSSIANS 

r = robjects.r

ZERO_ARRAY = np.zeros(shape=(MEM,1))

SAMPLES = 2000
QUANTILES = 100
vclin_x = np.ndarray(shape=(SAMPLES,3,3))
vclin_y = np.ndarray(shape=(SAMPLES,3,3))

g_grid_kde_array = []
g_grid_quantile_curves_array = []

def defineVclin():
    div = DIV
    div_real = DIVR
    
    mean1 = [0,-1]
    cov1 = [[1,0],[0,1]] 
    x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    
    #bimodal bivariate grid point distribution
    mean3 = [+2,+1]
    cov3 = [[1,0],[0,1]] 
    mean4 = [-2,-1]
    cov4 = [[1.5,0],[0,1.5]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.6*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.4*SAMPLES)).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    
    for x in range(0,3):
        for y in range(0,3):
            for idx in range(0,SAMPLES):
                if x == 1 and y == 1:
                    continue
                
                vclin_x[idx][x][y] = x1[idx] 
                vclin_y[idx][x][y] = y1[idx]  
                
    #define bimodal bivariate grid point
    for idx in range(0,SAMPLES):
        vclin_x[idx][1][1] = x_tot[idx]
        vclin_y[idx][1][1] = y_tot[idx]        

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
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    return gpt0, gpt1, gpt2, gpt3

def getVclinSamples(gpt0, gpt1, gpt2, gpt3, half=False):
    gpt0_dist = np.zeros(shape=(2,SAMPLES))
    gpt1_dist = np.zeros(shape=(2,SAMPLES))
    gpt2_dist = np.zeros(shape=(2,SAMPLES))
    gpt3_dist = np.zeros(shape=(2,SAMPLES))
    
    for idx in range(0,MEM):#SAMPLES):
        gpt0_dist[0][idx] = vclin_x[idx][gpt0[0]][gpt0[1]]
        gpt0_dist[1][idx] = vclin_y[idx][gpt0[0]][gpt0[1]]
     
        gpt1_dist[0][idx] = vclin_x[idx][gpt1[0]][gpt1[1]]
        gpt1_dist[1][idx] = vclin_y[idx][gpt1[0]][gpt1[1]] 
        
        gpt2_dist[0][idx] = vclin_x[idx][gpt2[0]][gpt2[1]]
        gpt2_dist[1][idx] = vclin_y[idx][gpt2[0]][gpt2[1]] 
        
        gpt3_dist[0][idx] = vclin_x[idx][gpt3[0]][gpt3[1]]
        gpt3_dist[1][idx] = vclin_y[idx][gpt3[0]][gpt3[1]]
    
    return gpt0_dist, gpt1_dist, gpt2_dist, gpt3_dist

def plotKDE(kde,distro, mx = (0,0), my = (0, 0), title = '', co = 'green'):
    
    global DIV, DIVR
    
    FIG = plt.figure()

    x_flat = np.r_[np.asarray(distro[0]).min():np.asarray(distro[0]).max():DIV]
    y_flat = np.r_[np.asarray(distro[1]).min():np.asarray(distro[1]).max():DIV]
    #x_flat = np.r_[mx[0]:mx[1]:DIV]
    #y_flat = np.r_[my[0]:my[1]:DIV]
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
  
def getKDEGriddata(x_min_max,y_min_max,kde):
    
    global DIV, DIVR
    
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:DIV]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:DIV]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(DIVR, DIVR)
    
    return z
  
 
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
        #print "kde not working"
        return None
  

green = '#98fb98'
blue = '#87ceeb'
red = '#f08080'
purple = '#ee82ee'
    
########################################################################################################

#from rpy2.robjects.numpy2ri import numpy2ri
#robjects.conversion.py2ri = numpy2ri
def getVclinSamplesSingle(gpt):
    
    global vclin
    
    gpt0_dist = np.zeros(shape=(2,MEM))
   
    for idx in range(0,MEM):
        gpt0_dist[0][idx] = vclin_x[idx][gpt[0]][gpt[1]][0]
        gpt0_dist[1][idx] = vclin_y[idx][gpt[0]][gpt[1]][1]
   
    return gpt0_dist
        
import rpy2.robjects.numpy2ri as rpyn

###### Quantile interp code ########
def bilinearBivarQuantLerp(f1, f2, f3, f4, x1, y1, x2, y2, x3, y3, x4, y4, alpha, beta):
    qstart = time.clock()
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
        #print 'problem with calculated interpolant z value...'
        f_bar_01[0] = -1 #failed
    
    qend = time.clock()
    totTime = qend-qstart
    
    return f_bar_01[0], totTime

DIV = 200j 
DIVR = 200

def findBivariateQuantilesSinglePass(kde,arr):
    
    #cpu time
    print 'integrating KDE...'
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
                if cd <= q + TOL and cd >= q - TOL:
                    #print "gathering points for quantile curve #: " + + str(idx) + " out of " + str(QUANTILES)
                    qcurvex[idx].append(x)
                    qcurvey[idx].append(y)
            
            z_pos.append(cd)
            x_pos.append(x)
            y_pos.append(y)
            
    #print 'finished computing quantile curves'
    
    #cpu time
    qend = time.clock()
    qtot = qend - qstart
    print 'cpu time for KDE integration (EDCF): ' + str(qtot) 
            
    return x_pos, y_pos, z_pos, qcurvex, qcurvey

MID_RANGE_QUANTILE_CURVE_POINTS = 150

def lerpBivariate3(gp0, gp1, gp2, gp3, alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3, arr, use_cache=True):
    
    totalCpuStart = time.clock()
    
    global g_grid_quantile_curves_array, QUANTILES, MID_RANGE_QUANTILE_CURVE_POINTS

    degree = 3;smoothing = None
    INTERP = 'linear'
    #spline_curve0=[];spline_curve1=[];spline_curve2=[];spline_curve3=[]
    
    qcurveCpu = 0.
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex0 = gp0_qcurve[0]
    qcurvey0 = gp0_qcurve[1]
    spline_curve0 = gp0_qcurve[2]
    if len(gp0_qcurve[0]) == 0 or use_cache is False:
        #print 'computing quantile curves gp0...'
        x_pos0, y_pos0, z_pos0, qcurvex0, qcurvey0 = findBivariateQuantilesSinglePass(gp0,arr[0])
        #plotXYZScatterQuants(qcurvex0, qcurvey0, title='qcurve0')
        spline_curve0 = []
        
        qstart = time.clock() 
        for q in range(0,len(qcurvex0)):
            if len(qcurvex0[q]) > degree: #must be greater than k value
                #spline_curve0.append(interpolate.UnivariateSpline(qcurvex0[q], qcurvey0[q], w=None, k=degree, s=smoothing))#bbox=[-20, 20])
                spline_curve0.append(interpolate.interp1d(qcurvex0[q], qcurvey0[q], kind=INTERP))
            else:
                spline_curve0.append([None])
        qend = time.clock()
        qcurveCpu += qend - qstart
        
        print 'Quantile curve parameterization: ' + str(qcurveCpu)
        
        g_grid_quantile_curves_array[i][j][0] = qcurvex0
        g_grid_quantile_curves_array[i][j][1] = qcurvey0
        g_grid_quantile_curves_array[i][j][2] = spline_curve0
        
    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_qcurve = g_grid_quantile_curves_array[i][j]
    qcurvex1 = gp1_qcurve[0]
    qcurvey1 = gp1_qcurve[1]
    spline_curve1 = gp1_qcurve[2]
    if len(gp1_qcurve[0]) == 0 or use_cache is False:
        #print 'computing quantile curves gp1...'
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
    if len(gp2_qcurve[0]) == 0 or use_cache is False:
        #print 'computing quantile curves gp2...'
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
    if len(gp3_qcurve[0]) == 0 or use_cache is False:
        #print 'computing quantile curves gp3...'
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
    
    pdfEvalsCpu = 0.
    qCurveInterpCpu = 0.
    qstartCurv = time.clock()
    for iq in range(0,QUANTILES):#, q in enumerate(qcurvex0):
        
        #if iq <= 0.5*QUANTILES:
        #    num_pts_to_eval_on_curve = 3*MID_RANGE_QUANTILE_CURVE_POINTS
        #else:
        #    num_pts_to_eval_on_curve = MID_RANGE_QUANTILE_CURVE_POINTS
            
        #print str(iq) + "th quantile curve being lerped out of " + str(QUANTILES)
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
            #print '    evaluating spline...'
            epts0 = linspace(qcurvex0[iq][0], qcurvex0[iq][-1], num_pts_to_eval_on_curve)
            epts1 = linspace(qcurvex1[iq][0], qcurvex1[iq][-1], num_pts_to_eval_on_curve)
            epts2 = linspace(qcurvex2[iq][0], qcurvex2[iq][-1], num_pts_to_eval_on_curve)
            epts3 = linspace(qcurvex3[iq][0], qcurvex3[iq][-1], num_pts_to_eval_on_curve) 
            cur_y0_parametrized_pts = spline_curve0[iq](epts0) 
            cur_y1_parametrized_pts = spline_curve1[iq](epts1)
            cur_y2_parametrized_pts = spline_curve2[iq](epts2)
            cur_y3_parametrized_pts = spline_curve3[iq](epts3)
            #print '...finished evaluating spline!'
        else:
            continue
        
        
        
        for idx in range(0,num_pts_to_eval_on_curve): #evaluate points along each parameterized quantile curve
            #print '    lerping point: ' +str(idx)+ ' out of ' + str(num_pts_to_eval_on_curve)
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
            
            interpolant_xy0 = np.asarray([cur_x0,cur_y0]) + alpha_x * dir_vec0
            interpolant_xy1 = np.asarray([cur_x2,cur_y2]) + alpha_x * dir_vec1
            
            dir_vec_bar = np.asarray([interpolant_xy1[0],interpolant_xy1[1]]) - \
                np.asarray([interpolant_xy0[0],interpolant_xy0[1]])
            
            interpolant_xy_bar = np.asarray([interpolant_xy0[0],interpolant_xy0[1]]) + alpha_y * dir_vec_bar
            
            qend = time.clock()
            qCurveInterpCpu += qend - qstartCurv 
            #print iq
            #print qCurveInterpCpu
            
            z, cpuTime = bilinearBivarQuantLerp(gp0, gp1, gp2, gp3, cur_x0, cur_y0, \
                                        cur_x1, cur_y1, cur_x2, cur_y2, cur_x3, cur_y3, \
                                         alpha_x, alpha_y)
            
            pdfEvalsCpu += cpuTime
            
            qstartCurv = time.clock()
            
            if z < 0. or math.isnan(z) or math.isinf(z):
            #    print 'WARNING: interpolated pdf has NaN, inf or negative result...skipping data point.'
                continue
           
            x_pos.append(interpolant_xy_bar[0])
            y_pos.append(interpolant_xy_bar[1])
            z_pos.append(z)  
            #print '        lerping point between quantile curves: ' + str(iq) + ' was successful!'
            
    #print 'finished lerping all quantile curve for interpolant distro...'
    
   
    totalCpu = time.clock() - totalCpuStart
    
    print 'Quantile curve interpolation: ' + str(qCurveInterpCpu)
    print 'PDF surface pts. eval: ' + str(pdfEvalsCpu)
    print 'Total Cpu for interp: ' + str(totalCpu)
    return x_pos, y_pos, z_pos

def interpFromQuantiles3(ppos=[0.0,0.0], number=0,sl=0, ignore_cache = 'False', half=False):
    
    #print ppos
    
    global g_grid_kde_array
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos, half)
    gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp = getVclinSamples(gpt0, gpt1, gpt2, gpt3, half)
    
    samples_arr = [gpt0_samp, gpt1_samp, gpt2_samp, gpt3_samp]
    
    i = int(gpt0[0])
    j = int(gpt0[1])
    gp0_kde = g_grid_kde_array[i][j]
    if gp0_kde is None or ignore_cache == 'True':
        try:
            gp0_kde = stats.kde.gaussian_kde( ( gpt0_samp[0][:], gpt0_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
           
        g_grid_kde_array[i][j] = gp0_kde

    i = int(gpt1[0])
    j = int(gpt1[1])
    gp1_kde = g_grid_kde_array[i][j]
    if gp1_kde is None or ignore_cache == 'True':
        try:
            gp1_kde = stats.kde.gaussian_kde( ( gpt1_samp[0][:], gpt1_samp[1][:] ) )
        
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
        
        g_grid_kde_array[i][j] = gp1_kde

    i = int(gpt2[0])
    j = int(gpt2[1])
    gp2_kde = g_grid_kde_array[i][j]
    if gp2_kde is None or ignore_cache == 'True':
        try:
            gp2_kde = stats.kde.gaussian_kde( ( gpt2_samp[0][:], gpt2_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp2_kde
        
    i = int(gpt3[0])
    j = int(gpt3[1])
    gp3_kde = g_grid_kde_array[i][j]
    if gp3_kde is None or ignore_cache == 'True':
        try:
            gp3_kde = stats.kde.gaussian_kde( ( gpt3_samp[0][:], gpt3_samp[1][:] ) )
        except:
            #print 'kde failed...trigger stopping condition for streamline'
            return samples_arr, None
            
        g_grid_kde_array[i][j] = gp3_kde

    #interp dist for x dim  
    alpha_x = 0.
    alpha_y = 0.
    if half == False:
        alpha_x = ppos_parts[0][0]
        alpha_y = ppos_parts[1][0]
    else:
        #alpha_x = pm.modf(ppos_parts[0][1] / 2)[0]
        #alpha_y = pm.modf(ppos_parts[1][1] / 2)[0]
        alpha_x = pm.modf(ppos[0] / 2)[0]
        alpha_y = pm.modf(ppos[1] / 2)[0]
    
    x3, y3, z3 = lerpBivariate3(gp0_kde, gp1_kde, gp2_kde, gp3_kde, alpha_x, \
                                alpha_y, gpt0, gpt1, gpt2, gpt3, samples_arr)
    
    '''
    plotKDE(gp0_kde,gpt0_samp, title="gpt0_kde", co = green)
    plotKDE(gp1_kde,gpt1_samp, title="gpt1_kde", co = green)
    plotKDE(gp2_kde,gpt2_samp, title="gpt2_kde", co = green)
    plotKDE(gp3_kde,gpt3_samp, title="gpt3_kde", co = green)
    '''
    
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
    
    #print "in computeDistroFunction..."
    
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

    #print "finished distro function comp..."
    #print 'success? -> ' + str(success)
    #print 'type? -> ' + str(interp_type)
    return distro_eval, interp_type, success

def plotXYZScatter((u_min,u_max),(v_min, v_max), x_pos,y_pos,z_pos, title = '', arr=[]):
    
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
    qstart = time.clock()
    AX.plot_surface(X, Y, surface, rstride=2, cstride=2, alpha=1.0, \
                    linewidth=0.1, antialiased=True, color=col)
    qend = time.clock()
    tot = qend - qstart
    print 'PDF linear surface interpolation: ' + str(tot)
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + ".jpg")
    #plt.show() 
    
def main():
    defineVclin()
    createGlobalKDEArray(3,3)
    createGlobalQuantileArray(3,3)
    
    ppos = [1,0]
   
    ''' http://www.tayloredmktg.com/rgb/#PA '''
    green = '#98fb98'
    blue = '#87ceeb'
    red = '#f08080'
    purple = '#ee82ee'
      
    for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] :
        ypos = ppos[1] + beta
        
        samples_arr, evalfunc = interpFromQuantiles3(ppos=[ppos[0],ypos])
        
        x_min = np.asarray(evalfunc[0]).min()
        x_max = np.asarray(evalfunc[0]).max()
        y_min = np.asarray(evalfunc[1]).min()
        y_max = np.asarray(evalfunc[1]).max()
        
        distro, interpType, suc = computeDistroFunction(evalfunc[0],evalfunc[1],evalfunc[2], \
                                                         (x_min,x_max), (y_min,y_max))
        
        titleQuantInterp = dt + str(ppos[0]) + '_' + str(ypos) + '_toy_'  
        
        #plot interpolants in their range
        plotXYZSurf((x_min,x_max), (y_min,y_max), distro, titleQuantInterp, samples_arr, col=blue)
        plotXYZScatter((x_min,x_max), (y_min,y_max), evalfunc[0],evalfunc[1],evalfunc[2], title=dt + str(ppos[0]) + \
                       '_' + str(ypos) + '_InterpToy_', arr=samples_arr )
    
    print 'finished!'
            
if __name__ == "__main__":  
    main()

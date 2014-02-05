#!/usr/bin/python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 10/7/2012.
    Code shows advection of particles in 2d velocity field with configurable distributions at each grid point.
'''

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

QUANTILES = 150
TOTAL_STEPS = 25
integration_step_size = 0.1
SEED_LAT = 42
SEED_LON = 21
SEED_LEVEL = 0
vclin = []
cf_vclin = []

divs = 300
div = complex(divs)
div_real = divs
start = -5
end = +5
TOL = ( ( 1.0 / QUANTILES ) / 2.0 ) / 2.0

from skimage import data
from skimage import measure
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.exposure as skie

reused_vel_quantile = 0

SAMPLES = 1000
vclin_x = np.ndarray(shape=(SAMPLES,10,10))
vclin_y = np.ndarray(shape=(SAMPLES,10,10))

DEBUG = False
  
MODE = 1
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/toy5/'
MODE_DIR1 = 'mode1/'
MODE_DIR2 = 'mode2/'
COM =  2
LON = 9
LAT = 9
LEV = 16
MEM = 600 
#MAX_GMM_COMP = 3 l
#EM_MAX_ITR = 2000
#EM_MAX_RESTARTS = 1000
DEPTH = -2.0 
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
BIFURCATED = 'n'
INTERP_METHOD = 'e'

MAX_BIFURCATIONS = 2
PEAK_SPLIT_THRESHOLD = 1.0
MIN_NUM_STEPS_BETWEEN_BRANCHING = 0

EM_MAX_ITR = 5
EM_MAX_RESTARTS = 1000
DEPTH = -2.0
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
NUM_GAUSSIANS = 2#4
MAX_GMM_COMP = NUM_GAUSSIANS 

#3d np array for storing parameters found for grid points
#check this array first before fitting g comps
g_grid_params_array = []
#3d np array for storing parameters found for grid points
#check this array first before fitting g comps
g_grid_kde_array = []
g_grid_quantile_curves_array = []


import Tkinter as Tk
rootTk = Tk.Tk()
canvas_1 = None 

'''
def handler():
    #writeParticles();
    rootTk.quit()
'''
#rootTk.protocol("WM_DELETE_WINDOW", handler)
#rootTk.bind("WM_DELETE_WINDOW", handler)
#canvas_1.bind("<Key>", handler)




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

def findBivariateQuantilesSinglePass(kde):
    
    global QUANTILES
    
    incr = math.fabs( start - end ) / divs 
    x_div = np.r_[start:end:complex(divs)]
    y_div = np.r_[start:end:complex(divs)]
    
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
                if cd <= q + TOL and cd >= q - TOL:
                    #print "gathering points for quantile curve #: " + + str(idx) + " out of " + str(QUANTILES)
                    qcurvex[idx].append(x)
                    qcurvey[idx].append(y)
            
            z_pos.append(cd)
            x_pos.append(x)
            y_pos.append(y)
            
    print 'finished computing quantile curves'
            
    return x_pos, y_pos, z_pos, qcurvex, qcurvey
    


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
 

MID_RANGE_QUANTILE_CURVE_POINTS = 100

def lerpBivariate3(gp0, gp1, gp2, gp3, alpha_x, alpha_y, gpt0, gpt1, gpt2, gpt3):
    global g_grid_quantile_curves_array, QUANTILES, MID_RANGE_QUANTILE_CURVE_POINTS

    degree = 5;smoothing = None
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

    
def interpFromQuantiles3(ppos=[0.0,0.0], number=0,sl=0):
    
    print ppos
    
    global g_grid_kde_array
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gpt0, gpt1, gpt2, gpt3 = getGridPoints(ppos)
    
    #gpt0_samp=None; gpt1_samp=None; gpt2_samp=None; gpt3_samp=None
    
    #only need to collect samples if we haven't already calculated kde's for grid cell
    #if g_grid_kde_array[int(gpt0[0])][int(gpt0[1])] is None and g_grid_kde_array[int(gpt1[0])][int(gpt1[1])] is None \
    #    and g_grid_kde_array[int(gpt2[0])][int(gpt2[1])] is None and g_grid_kde_array[int(gpt3[0])][int(gpt3[1])] is None:
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
    
    plotXYZScatter(x3, y3, z3, title=INTERP_METHOD + '-' + str(number)+'final'+str(sl) )
    plotXYZSurf(distro,title=INTERP_METHOD + '-' + str(number)+'final'+str(sl) )
    
    return distro 

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
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "scatter.png")
    #plt.show()

def computeDistroFunction(x_pos,y_pos,z_pos):
    print 'computing distribution function for the QUANTILE INTERP RESULT SAMPLES...'
    incr = math.fabs( start - end ) / divs 
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    X, Y = np.meshgrid(X, Y)
    pts = np.append(np.asarray(x_pos).reshape(-1,1),np.asarray(y_pos).reshape(-1,1),axis=1)
    
    
    distro_eval = None
    success = False
    
    while not success:
        try:
            distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='cubic')#, fill_value=0.0)
            success = True
        except:
            '''
            try:
                distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
                                               xi=(X,Y), method='linear')#, fill_value=0.0)
                success = True
                continue
            '''
            #except:
            continue
    
    #distro_eval = interpolate.griddata(points=pts, values=np.asarray(z_pos), \
    #                                   xi=(X,Y), method='nearest', fill_value=0.0)
    
    #distro_eval = interpolate.Rbf(x_pos,y_pos,z_pos)#,epsilon=2)
    
    print "finished distro function comp..."
    return distro_eval
        

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

class Particle():
    def __init__(self):
        self.part_positions = [[],[],[],[],[],[],[]]
        #self.can_branch = True
        self.steps_since_last_branch = 0 
        self.branching_min = MIN_NUM_STEPS_BETWEEN_BRANCHING
        self.prev_vel = (0.,0.)
        self.parent = None
        self.children = []
        self.is_root = False
        self.age = 0#number of advection steps, used for pruning precedence
    def canBranch(self):
        if self.steps_since_last_branch >= self.branching_min:
            return True
        else:
            return False
    def incrBranchCount(self):
        if self.canBranch():
            ''' we are spawning a new particle / streamline branch...'''
            ''' @returns True if branching successful '''
            self.steps_since_last_branch = 0
            print '    branch allowed'
            return True
        else:
            self.steps_since_last_branch += 1
            print '    branch DISallowed'
            return False
    
g_part_positions = []
#g_part_positions.append(Particle())

#g_cc = np.zeros(shape=(LAT,LON))
#g_crisp_streamlines = []
#g_part_positions = [[],[],[],[],[],[],[]]
#g_part_positions2 = [[],[],[],[],[],[],[]]

'''
g_part_positions_b = [[],[],[],[],[],[],[]]

part_pos_e = [];part_pos_q = [];part_pos_gmm = [];part_pos_g = []
part_pos_e.append([0,0])
part_pos_e[0][0] = SEED_LAT
part_pos_e[0][1] = SEED_LON

part_pos_e_b = [];part_pos_q_b = [];part_pos_gmm_b = [];part_pos_g_b = []
part_pos_e_b.append([0,0])
part_pos_e_b[0][0] = SEED_LAT
part_pos_e_b[0][1] = SEED_LON
'''

ZERO_ARRAY = np.zeros(shape=(MEM,1))

def plotParticles(ts_per_gp=[]):
    #http://docs.enthought.com/mayavi/mayavi/mlab_figures_decorations.html
    f = mayavi.mlab.gcf()
    #cam = f.scene.camera
    #cam.parallel_scale = 10
    f.scene.isometric_view()
     
    grid_verts = np.zeros(shape=(LAT,LON))
    #grid_verts = grid_verts * 10
    grid_lat, grid_lon = np.ogrid[0:LAT,0:LON]
    
    size = 10
    nx, ny, nz = (size,size,1)
    x = np.linspace(0, 9, 1)
    y = np.linspace(0, 9, 1)
    z = np.linspace(0, 1, 1)
    ox, oy, oz = np.meshgrid(x, y, z)
   
    grid_plane = mayavi.mlab.surf(grid_lat, grid_lon, grid_verts, color=(1,1,0),representation='wireframe',line_width=0.3)
    
    g_grid_mean_array_u = np.ndarray(shape=(9,9,1))
    g_grid_mean_array_v = np.ndarray(shape=(9,9,1))
    w = np.zeros(shape=(9,9,1))
    
    for x in range(0,9):
        for y in range(0,9):
            #print np.mean(vclin_y[x][y][:])
            samp_x = []
            samp_y = []
            for sample in range(0,SAMPLES):
                samp_x.append(vclin_x[sample][x][y])
                samp_y.append(vclin_y[sample][x][y])
            g_grid_mean_array_u[x][y][0] = np.mean(samp_x)
            g_grid_mean_array_v[x][y][0] = np.mean(samp_y)
            
    u = g_grid_mean_array_u
    v = g_grid_mean_array_v
    
    mayavi.mlab.quiver3d(u, v, w, mode='2darrow', scale_mode ='vector', color=(1,1,1))
    
    #bimodal distro
    mayavi.mlab.points3d(SEED_LAT,SEED_LON,0,scale_factor = 0.05,color=(0,1.0,0))
    mayavi.mlab.points3d(4,4,0,scale_factor = 0.05,color=(1.0,0,0))
    
    for part in range(0,len(g_part_positions)):
        if len(g_part_positions[part].part_positions[0]) < 2:
            continue
        
        x_list = g_part_positions[part].part_positions[0][:]
        y_list = g_part_positions[part].part_positions[1][:]
        z_list = g_part_positions[part].part_positions[2][:]
        #max_peak_separation_list = g_part_positions[part].part_positions[3][:]
        #print part
        r = np.random.ranf()#;print r
        g = np.random.ranf()#;print g
        b = np.random.ranf()#;print b
        
        mayavi.mlab.plot3d(x_list, y_list, z_list, \
                       #max_peak_separation_list, \
                        name='Ensemble',tube_radius = None,line_width=0.5, color=(r, g, b))#, colormap='Greens')
    
    mayavi.mlab.show() 
                


def writeStreamlinePositions(data,filename):
    #change to 'wb' after initial debug...
    filename = OUTPUT_DATA_DIR + filename
    writer = csv.writer(open(filename + ".csv", 'w'))
    
    #writes velocities with central forecast...
    for curr_comp in range(0,len(data),1):
        #for curr_pos in range(0,len(data[curr_comp][:]),1):
            #print "curr pos " + str(curr_pos)
            #print "curr comp" + str(curr_comp)
            #print data[curr_comp][curr_pos]
        writer.writerow(data[curr_comp][:])

def readStreamlinePositions(data, filename):
    #change to 'wb' after initial debug...
    filename = OUTPUT_DATA_DIR + filename
    #os.path.exists( filename )
    try:
        reader = csv.reader(open(filename + ".csv", 'r'), delimiter=',')
    except:
        print "no file named: " + filename
        return False
    
    idx = 0
    for row in reader:
        #print row
        data[idx] = [float(i) for i in row]
        idx += 1
        
    return True
    
    
def writeParticles(dir = 'f'):
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR) 
   
    if dir == 'f':
        for part in range(0,len(g_part_positions)):
            writeStreamlinePositions(g_part_positions[part].part_positions,\
                                 str(INTERP_METHOD) + '_' +str(part) + '_Toy_lat'+str(SEED_LAT)+'_lon'+\
                                 str(SEED_LON)+'_lev'+str(SEED_LEVEL) + str_integration_values + '_' + str(MAX_BIFURCATIONS)) 
       
    #elif dir == 'b':
    #    writeStreamlinePositions(g_part_positions_b,\
    #                             'Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
    '''   
    else:
        # forward and backward streamlines
        # concatenate particle positions
        e = []
    
        #for each component
        for idx in range(0,len(g_part_positions_b)):
            g_part_positions_b[idx].reverse()
           
            e.append(g_part_positions_b[idx] + g_part_positions[idx])
           
        
        writeStreamlinePositions(e,'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
    ''' 
        
def readParticles():
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR) \
    
    
    global g_part_positions, MAX_BIFURCATIONS
    
    #for line in range(0,MAX_BIFURCATIONS):
    #    g_part_positions.append(Particle())
    
    for part in range(0,MAX_BIFURCATIONS):
        g_part_positions.append(Particle())
        if readStreamlinePositions(g_part_positions[part].part_positions,\
                            str(INTERP_METHOD) + '_' + str(part) + '_Toy_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+ \
                            '_lev'+str(SEED_LEVEL)+str_integration_values + '_' +str(MAX_BIFURCATIONS) ) is False:
            #g_part_positions.pop(part) 
            print "reading false..."
    
    '''
    if BIFURCATED == 'y': 
        readStreamlinePositions(g_part_positions2,\
                            '2Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+\
                            '_lev'+str(SEED_LEVEL)+str_integration_values)
    '''
  

def countPeaks(lim,limg,x1,y1):
    peaks = [[],[]]
    for idx in range(0,len(x1)):
        if limg[(y1[idx],x1[idx])] > lim: 
            peaks[0].append(x1[idx])
            peaks[1].append(y1[idx])
    return len(peaks[0])
    

def findBivariatePeaks(dist_fnt,method='e',streamline=0,step=0):
    # Regular grid to evaluate kde upon
    x_flat = np.r_[start:end:div]
    y_flat = np.r_[start:end:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = None
    
    if method == 'q':
        z = dist_fnt
        
        #if using griddata interp
        for idx_x in range(0,len(x_flat)):
            for idx_y in range(0,len(y_flat)):
                if math.isnan(z[idx_x][idx_y]):
                    z[idx_x][idx_y] = 0.0
                    
    #else:
    #if using Rbf with quantile
    #if method == 'q':
    #    z = dist_fnt(x,y)
    else:    
        z = dist_fnt(grid_coords.T)
                
        z = z.reshape(div_real,div_real)
    
    
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
    fp = 13
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
 
#from bivariate_interp import interpVelFromEnsemble as interpEnsemble
#from bivariate_interp import getCoordParts
#from bivariate_interp import getGridPoints
#from bivariate_interp import getVclinSamples
#from bivariate_interp import defineVclin
#from bivariate_interp import vclin_x, vclin_y

def interpEnsemble(ppos=[0.0,0.0]):
    
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
        #bw_method = 'scott', alternative bandwidth estimator...
        #http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        k = stats.kde.gaussian_kde((lerp_u,lerp_v), bw_method='silverman') 
        return k

    except:
        print "kde not working"
        return None

def defineVclin():
    
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
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.55*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.45*SAMPLES)).T
    
    #left half
    #mean1 = [0,-4]
    #cov1 = [[0.3,0],[0,0.3]] 
    #x1,y1 = np.random.multivariate_normal(mean1,cov1,SAMPLES).T
    
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
    #mean2 = [0,+4]
    #cov2 = [[0.3,0],[0,0.3]] 
    #x2,y2 = np.random.multivariate_normal(mean2,cov2,SAMPLES).T
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
    #mean3 = [+4,0]
    #cov3 = [[0.3,0],[0,0.3]] 
    #mean4 = [-4,0]
    #cov4 = [[0.3,0],[0,0.3]] 
    #x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.5*SAMPLES)).T
    #x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.5*SAMPLES)).T
    
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
        
    '''
    x_tot2 = []
    y_tot2 = []
    for idx in range(0,SAMPLES):
        x_tot2.append(vclin_x[idx][5][2])
        y_tot2.append(vclin_y[idx][5][2])
    '''
        
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
    
def findVecNorm(vec):
    #find vec norm
    vec = np.asarray(vec)
    vec_len = np.sqrt(np.dot(vec, vec))
    if vec_len > 0.: 
        return vec / vec_len
    else:
        return False
       
def findClosestVelDir( vel=(0.,0.), vel_list = []  ):
    cos_angle_min = np.pi#-1 #max possible (180degrees/pi radians) angle or -1 <= cos(theta) <= +1
    found_idx = 0
    
    vel_norm = findVecNorm(vel)
    
    for v_idx in range(0,len(vel_list)):
        cur_vel_norm = findVecNorm(vel_list[v_idx])
        
        if cur_vel_norm is False:
            continue
        try:
            cosangle = np.dot(vel_norm,cur_vel_norm)
            temp_cos_angle = np.pi
            if cosangle == 1.0:
                temp_cos_angle = 0.0
            else:
                temp_cos_angle = math.acos(cosangle) #cos(theta), theta = angle between vecs
            print temp_cos_angle
            #if math.isinf(temp_cos_angle) or math.isnan(temp_cos_angle):
            #    print "arccosine failed..."
            #    exit()
            if temp_cos_angle < cos_angle_min: #denotes a smaller angle between vectors
                cos_angle_min = temp_cos_angle
                print cos_angle_min
                found_idx = v_idx
        except:
            continue
        
    #smooth data if aberrant value appears
    '''
    if cos_angle_min > ( pi / 180.0 ) * 30.:
        print 'FILLING VELOCITY NOISE BY REPEAT!!!'
        return vel, found_idx 
    '''
            
    print vel_list[found_idx]
    print found_idx
    return vel_list[found_idx], found_idx

#from bivariate_interp import interpFromQuantiles3 as interpFromQuantiles

def findAngleVecs(vec1, vec2):
    vec1n = findVecNorm(vec1)
    vec2n = findVecNorm(vec2)
    if vec1n is not False and vec2n is not False:
        try:
            math.acos(np.dot(vec1n,vec2n))
            return math.acos(np.dot(vec1n,vec2n))
        except:
            return False
    else:
        return False

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def ccw(A,B,C):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

#from: http://www.bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def plotXYZSurf(surface, title = ''):
    
    '''
    incr = math.fabs( start - end ) / divs 
    X = np.arange(start, end, incr)
    Y = np.arange(start, end, incr)
    X, Y = np.meshgrid(X, Y)
    
    Z = surface(X,Y)
    '''
      
    '''             
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
    '''
    
    fig2 = plt.figure()
    #if using griddata()
    plt.imshow(surface.T, extent=(start,end,start,end),origin=None,interpolation='bicubic')

    #if using Rbf interp
    #plt.pcolor(X, Y, Z, cmap=cm.jet)
    
    
    plt.savefig(OUTPUT_DATA_DIR + str(title) + "top.png") 
    #plt.show() 

def previewTk():
    for sl in g_part_positions:
        r = np.random.randint(255)#;print r
        g = np.random.randint(255)#;print g
        b = np.random.randint(255)#;print b
        
        rgb = r, g, b
        Tk.Hex = '#%02x%02x%02x' % rgb
        
        scale = 500
        offset = 1500
        
        for idx in range(1,len(sl.part_positions[0])):
            canvas_1.create_line(sl.part_positions[0][idx-1]*scale - offset  \
                                ,1000-sl.part_positions[1][idx-1]*scale + offset \
                                ,sl.part_positions[0][idx]*scale - offset \
                                ,1000-sl.part_positions[1][idx]*scale + offset \
                                ,fill=str(Tk.Hex) \
                                ,width=2)

def advect(step, dir = 'f',method='e'):
    
    global g_part_positins, SEED_LAT,SEED_LON,SEED_LEVEL,TOTAL_STEPS,INTEGRATION_DIR,\
        integration_step_size,gen_streamlines, BIFURCATED, MAX_BIFURCATIONS, PEAK_SPLIT_THRESHOLD, INTERP_METHOD
    
    #global g_part_positions
    
    advection_step_num_streamlines = len(g_part_positions)
    bifurcation_additions = 0 #additional bifurcations added this advect step
    
    list_to_prune = []#contains refs to particles
    
    #if step % 20  == 0:
    print '**step number: ' + str(step) 
    #split_this_step = False
    for particle in range(0, advection_step_num_streamlines):
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        #if dir == 'f':
        
        #find current position of current streamline particle
        
        try:
            cur_posx = g_part_positions[particle].part_positions[0][-1]
            cur_posy = g_part_positions[particle].part_positions[1][-1]
        except:
            print particle
            print g_part_positions[particle].parent
            print g_part_positions[particle].part_positions
            exit()
        
        ppos = [ cur_posx , cur_posy ]
        
        #stop advection of streamline if close to grid boundary
        if cur_posx > LAT - 1.0 or cur_posy > LON - 1.0 or cur_posx < 1.0 or cur_posy < 1.0:
            #print particle
            #print ppos
            #print g_part_positions[particle].part_positions
            continue
        
        #else:
        #ppos = [ part_pos_e_b[particle][0], part_pos_e_b[particle][1] ]
            
        #get peaks
        interpolant_distro = None
        peak_vels = []; peak_probs = []; max_peak_distance = 0.
        if method == 'e':
            interpolant_distro = interpEnsemble(ppos)
            peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(interpolant_distro,method,particle,step)
        elif method == 'gmm':
            interpolant_distro, params_peaks = interpFromGMM(ppos)
            peak_vels, peak_probs, max_peak_distance = findGMMPeaksFromParams(params_peaks)
        elif method == 'q':
            interpolant_distro = interpFromQuantiles3(ppos,number=step,sl=particle)
            peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(interpolant_distro,method,particle,step)
            
        
        
        #pick velocity that has the smallest angle between last advection velocity.
        prev_vel = ( g_part_positions[particle].prev_vel[0], \
                     g_part_positions[particle].prev_vel[1] )
        
        m = max(peak_probs)
        p = peak_probs.index(m)
        vel_hp = peak_vels[p]
        vel_hp2 = (0,0)
        
        closest_vel_peak = vel_hp
            
        num_peaks = len(peak_vels)
        if  num_peaks > 1:
            closest_vel_peak, idx = findClosestVelDir(prev_vel, peak_vels)
            '''
            #m = max(peak_probs)
            #p = peak_probs.index(m)
            #vel_hp = peak_vels[p]
            peak_vels.pop(p) 
            peak_probs.pop(p)
            #find second highest peak
            m2 = max(peak_probs)
            p2 = peak_probs.index(m2)
            vel_hp2 = peak_vels[p2]
            '''
            #remove velocity used for parent streamline from consideration
            peak_vels.pop(idx) 
            peak_probs.pop(idx)
            
            #most probably peak remaining
            m2 = max(peak_probs)
            p2 = peak_probs.index(m2)
            vel_hp2 = peak_vels[p2]
            
            
        #advect current particle along main peak
        g_part_positions[particle].prev_vel = closest_vel_peak
        new_x = ppos[0] + closest_vel_peak[0]*integration_step_size#vel_hp[0]*integration_step_size
        new_y = ppos[1] + closest_vel_peak[1]*integration_step_size#vel_hp[1]*integration_step_size
        
        terminate = False
        part_to_remove = g_part_positions[particle]
        if g_part_positions[particle].is_root == False: #don't check root, we don't want to remove it.
            #terminate = checkIntersectWithAncestors(particle,(new_x,new_y),list_to_prune)
            terminate, part_to_remove = checkIntersectWithAllOtherStreamlines(particle,(new_x,new_y),list_to_prune)
            #terminate = False
        print 'streamline#: ' + str(particle)
        
        if terminate is True:
            list_to_prune.append(part_to_remove)
            print '    terminated due to intersection with other streamline'
        #else:
        #age particle
        g_part_positions[particle].age += 1
        
        g_part_positions[particle].part_positions[0].append(new_x)
        g_part_positions[particle].part_positions[1].append(new_y) 
        g_part_positions[particle].part_positions[2].append(DEPTH) 
        g_part_positions[particle].part_positions[3].append(closest_vel_peak[0])
        g_part_positions[particle].part_positions[4].append(closest_vel_peak[1])
        #g_part_positions[particle].part_positions[3].append(max_peak_distance)
        #g_part_positions[4].append(getSpeed(vel_hp))
            
        #find difference in peak prob and other peak probs
        '''
        max_peak_prob_diff = 0.0
        for idx in range(0,len(peak_probs)):
            diff = np.abs( p - peak_probs.index(idx) )
            if diff > max_peak_prob_diff:
                max_peak_prob_diff = diff 
        ''' 
        
        print '    max peak velocity: ' + str(vel_hp)
        print '    previous parent velocity for streamline: ' + str(prev_vel)
        print '    chosen velocity for this step on the parent streamline: ' + str(closest_vel_peak)
        print '    current position: ' + str(ppos)
        print '    peak velocities: ' + str(peak_vels)
        print '    branch age (number of advection steps): ' + str(g_part_positions[particle].age)
        
        #spawn new particle if necessary
        if num_peaks > 1 and np.abs(max_peak_distance) > PEAK_SPLIT_THRESHOLD \
            and advection_step_num_streamlines + bifurcation_additions <= MAX_BIFURCATIONS: 
            if g_part_positions[-1].incrBranchCount():
                
                branched_x = ppos[0] + vel_hp2[0]*integration_step_size
                branched_y = ppos[1] + vel_hp2[1]*integration_step_size
                
                #copy current positions for newly spawned particle branch point,
                #same as parent current position
                g_part_positions.append(Particle())
                g_part_positions[-1].parent = g_part_positions[-2]#store ref/pointer to parent
                g_part_positions[-1].part_positions[0].append(ppos[0])
                g_part_positions[-1].part_positions[1].append(ppos[1])  
                g_part_positions[-1].part_positions[2].append(DEPTH) 
                g_part_positions[-1].part_positions[3].append( closest_vel_peak[0] )
                g_part_positions[-1].part_positions[4].append( closest_vel_peak[1] )
                
                #g_part_positions[-1].part_positions[3].append(max_peak_distance)
                
                #advect newly spawned particle for current advection step
                g_part_positions[-1].prev_vel = vel_hp2
                g_part_positions[-1].part_positions[0].append( branched_x )
                g_part_positions[-1].part_positions[1].append( branched_y )
                g_part_positions[-1].part_positions[2].append( DEPTH )
                g_part_positions[-1].part_positions[3].append( vel_hp2[0] )
                g_part_positions[-1].part_positions[4].append( vel_hp2[1] )
                
                #terminate = checkIntersectWithAncestors(len(g_part_positions)-1,(branched_x,branched_y),list_to_prune)
                part_to_remove = g_part_positions[-1]
                terminate, part_to_remove = checkIntersectWithAllOtherStreamlines(len(g_part_positions)-1,\
                                                                  (branched_x,branched_y),list_to_prune,new_branch=True)
                #terminate = False
                if terminate is True:
                    print '    **branch not taken due to intersection'
                    list_to_prune.append(part_to_remove)
                else:
                    #record this as a child of the parent
                    g_part_positions[-1].parent.children.append(g_part_positions[-1])
                    #age particle
                    g_part_positions[-1].age += 1
                    bifurcation_additions += 1
                    
    #prune streamlines that cross ancestors
    
    print "this many instersections this advection step: " + str(len(list_to_prune))
    
    previewTk()
    
    for sl in list_to_prune:
        try:
            idx_sl = g_part_positions.index(sl)
            print "PRUNING THIS BRANCH AND THEIR DESCENDENTS IN THIS ADVECTION STEP: " + str(idx_sl) 
            if idx_sl == 0 or idx_sl == 1:
                print "pruning root streamline!!!"
                continue
            #FIND ALL DESCENDENTS OF THESE PRUNED BRANCHES AND REMOVE THEM AS WELL!
            g_part_positions.pop(idx_sl)
            
            killChildStreamlines(sl)
            #pruneDanglingStreamlines()
            
        except:
            print "couldn't find particle in global list, must have been killed as a child..."
            
    previewTk()
            
        
def pruneDanglingStreamlines():
    for sl in g_part_positions:
        if sl.is_root:
            continue
        try:
            g_part_positions.index(sl.parent)
        except:
            print "parent doesn't exist. killing streamline unless it is root..."
            sl_index = g_part_positions.index(sl)
            g_part_positions.pop(sl_index)
            
def killChildStreamlines(sl):
    #kill all the direct descendents and the descendents of the direct descendents
    #recurse thru children lists
    for child in sl.children:
        
        killChildStreamlines(child)#kill youngest decscendents first and work backwards
        
        try:
            idx_child = g_part_positions.index(child)
#            if idx_child == 0:
#                print "pruning root streamline!!!"
            g_part_positions.pop(idx_child)
        except:
            print "child index not found!"
        
                
                
#STREAMLINE_CROSSING_DISTANCE_THRESHOLD = 0.02 
ANGLE_SEP_THRESHOLD = 0.002*20#in radians, 57.3 X radians = degrees, 0.002 = 1 deg
MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK = 50
  

def checkIntersectWithAllOtherStreamlines(idx_of_streamline,new_pos, list_to_prune=[],new_branch=False):
    
    #we only need to check the possible new position of the current streamline
    #against positions from ancestors (which are advected first for this step)
    
    sl_to_check = g_part_positions[idx_of_streamline]
    parent = sl_to_check.parent
    sl_prev_x = sl_to_check.part_positions[0][-2]
    sl_prev_y = sl_to_check.part_positions[1][-2]
    
    #points on line of current streamline
    sl_point_prev = Point(sl_prev_x, sl_prev_y)
    sl_new_point = Point(new_pos[0],new_pos[1])
    
    #walk backwards since local ancestors more likely to intersect
    for particle in range(len(g_part_positions)-1,-1,-1):
        
        # don't check new position of streamline against parent, if this is a newly spawned branch
        if g_part_positions[particle] == parent and new_branch == True: \
            #or (sl_to_check == g_part_positions[particle] and new_branch == True):
            continue
        
        number_of_positions = len(g_part_positions[particle].part_positions[0])
        end_index = -1
        if MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK < number_of_positions:
            end_index = number_of_positions - MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK - 1
            
        for pos in range(number_of_positions-1,end_index,-1):
            
            if pos - end_index < 1: #check to see if we have enough line segments
                break
            
            print pos
            other_pos_x = g_part_positions[particle].part_positions[0][pos]
            other_pos_y = g_part_positions[particle].part_positions[1][pos]
            other_pos_pt = Point(other_pos_x, other_pos_y)
            
            other_pos_x_prev = g_part_positions[particle].part_positions[0][pos-1]
            other_pos_y_prev = g_part_positions[particle].part_positions[1][pos-1]
            other_pos_pt_prev = Point(other_pos_x_prev,other_pos_y_prev)
            
            #check for intersections with self but not for the new position
            if g_part_positions[particle] == sl_to_check:# and \
                #(pos == number_of_positions-1 or pos == number_of_positions-2):
                continue
            
            #dist = math.sqrt(math.pow(new_pos[0]-other_pos_x,2) \
            #                 + math.pow(new_pos[1]-other_pos_y,2))
            
            print 'part: ' + str(pos)
            #part_u = g_part_positions[particle].part_positions[3][pos]
            #part_v = g_part_positions[particle].part_positions[4][pos]
            
            
            
            #print dist
            #print angle
            
            
            cross = intersect(other_pos_pt, other_pos_pt_prev, sl_new_point, sl_point_prev)
            #v1 = (other_pos_pt.x-other_pos_pt_prev.x,other_pos_pt.y-other_pos_pt_prev.y)
            #v2 = (sl_new_point.x-sl_point_prev.x,sl_new_point.y-sl_point_prev.y)
            
            '''
            angle = findAngleVecs(v1, v2)
            
            if angle is False:
                angle = 1.5#approx 90 degrees
            '''
            if cross: #and angle >= ANGLE_SEP_THRESHOLD:
                particle_to_remove = sl_to_check
                #never remove offender if it is older than the streamline that it has intersected with
                #but remove the younger streamline instead
                if sl_to_check.age >= g_part_positions[particle].age:
                    particle_to_remove = g_part_positions[particle]
                return True, particle_to_remove
                  
    return False, None #note that the root streamline can never be pruned / deleted (considered intersecting itself) 
            
def getSpeed(vector):
    return np.abs(np.dot(np.asarray(vector), np.asarray(vector)))  
 
#from bivariate_gmm_interp import createGlobalParametersArray
#from bivariate_gmm_interp import NUM_GAUSSIANS
#from bivariate_gmm_interp import EM_MAX_ITR
#from bivariate_gmm_interp import fitBvGmm

def updateAdvect():
    global canvas_1
    for i_step in range(1, TOTAL_STEPS):
        canvas_1.delete(Tk.ALL)
        if INTERP_METHOD == 'e':
            advect(i_step,dir = 'f',method='e')
        elif INTERP_METHOD == 'gmm':
            advect(i_step,dir = 'f', method='gmm')
        elif INTERP_METHOD == 'q':
            advect(i_step,dir = 'f', method='q')
            
        canvas_1.update()
            
    rootTk.quit()
    
def findGMMPeaksFromParams(params): 
    means = []; weights = []
    for idx in range(0,len(params)):
        means.append( [params[idx][0][1],params[idx][0][0]] )
        #cur_inter_cov =  params[idx][1]
        weights.append( params[idx][2] ) 
    
    peak_distances = []
    mean_idxs_for_paired_peaks_distances = []
    if len(means) > 1:
        for idx in range(0,len(means)):
            for idx2 in range(0,len(means)):
                peak_distances.append(math.sqrt(math.pow(means[idx][0]-means[idx2][0],2) \
                                       + math.pow(means[idx][1] - means[idx2][1],2)))
                mean_idxs_for_paired_peaks_distances.append((idx,idx2))
                
    max_peak_distance = max(peak_distances)
    peak_idx_pair = peak_distances.index(max_peak_distance)
    
    return  means, weights, max_peak_distance
        
def main():
    global SEED_LAT,SEED_LON,SEED_LEVEL,TOTAL_STEPS,INTEGRATION_DIR,\
        integration_step_size,gen_streamlines, BIFURCATED, MAX_BIFURCATIONS, PEAK_SPLIT_THRESHOLD, INTERP_METHOD, \
        MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK, canvas_1
    SEED_LAT = 4 
    SEED_LON = 5 
    gen_streamlines = 'True'
    gen_streamlines = sys.argv[1]
    SEED_LAT = float(sys.argv[2])
    SEED_LON = float(sys.argv[3])
    SEED_LEVEL = int(sys.argv[4])
    integration_step_size = float(sys.argv[5])
    TOTAL_STEPS = int(sys.argv[6])
    INTEGRATION_DIR = str(sys.argv[7]).lower()
    MAX_BIFURCATIONS = int(sys.argv[8])
    PEAK_SPLIT_THRESHOLD = float(sys.argv[9])
    INTERP_METHOD = str(sys.argv[10])
    #MODE = int(sys.argv[8])
    #level = SEED_LEVEL
    
    #vclin = np.zeros(shape=(10,10,2))
    
    rootTk.title('preview for: ' + str(INTERP_METHOD))
    canvas_1 = Tk.Canvas(rootTk,width=1000,height=1000,background='#aaaaaa')
    canvas_1.pack(expand=True, fill=Tk.BOTH)
    
    if gen_streamlines == 'True':
        defineVclin()
        r.library('mixtools')
        
        if INTERP_METHOD == 'gmm':
            createGlobalParametersArray(LAT,LON)
        elif INTERP_METHOD == 'q':
            createGlobalKDEArray(LAT,LON)
            createGlobalQuantileArray(LAT,LON)
        
        print "generating streamlines"
        '''
        particle = 0
        
        part_pos_e[particle][0] = SEED_LAT; part_pos_e[particle][1] = SEED_LON
        #part_pos_e_b[particle][0] = SEED_LAT; part_pos_e_b[particle][1] = SEED_LON
        
        ppos = [ part_pos_e[particle][0], part_pos_e[particle][1] ]
        '''
        
        distro_f = None
        peak_vels = []; peak_probs = []; max_peak_distance = 0.
        if INTERP_METHOD == 'e':
            distro_f = interpEnsemble([SEED_LAT, SEED_LON])
            peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(distro_f,INTERP_METHOD)
        elif INTERP_METHOD == 'gmm':
            distro_f, param_peaks = interpFromGMM([SEED_LAT, SEED_LON])
            peak_vels, peak_probs, max_peak_distance = findGMMPeaksFromParams(param_peaks)
        elif INTERP_METHOD == 'q':
            distro_f = interpFromQuantiles3([SEED_LAT, SEED_LON])
            peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(distro_f,INTERP_METHOD)
           
        
             
        
       
        
        m = max(peak_probs)
        p = peak_probs.index(m)
        vel_hp = peak_vels[p]
    
        print peak_vels
        print peak_probs
        print max_peak_distance
        
        #find two or less peaks in distro defined by kde
        initial_seed_particle = Particle()
        #we allow trunk to branch instantly after one advection step
        #otherwise 'steps_since_last_branch' is defalut 0
        initial_seed_particle.steps_since_last_branch = MIN_NUM_STEPS_BETWEEN_BRANCHING
        initial_seed_particle.prev_vel = vel_hp
        initial_seed_particle.is_root = True
        initial_seed_particle.age += 1
        g_part_positions.append(initial_seed_particle)
        g_part_positions[0].part_positions[0].append(SEED_LAT)
        g_part_positions[0].part_positions[1].append(SEED_LON) 
        g_part_positions[0].part_positions[2].append(DEPTH) 
        g_part_positions[0].part_positions[3].append( vel_hp[0] )
        g_part_positions[0].part_positions[4].append( vel_hp[1] )
        #g_part_positions[0].part_positions[3].append(max_peak_distance)
        #g_part_positions[0].part_positions[4].append(getSpeed(vel_hp))
        #g_part_positions[0].part_positions[5].append(0)
        #g_part_positions[0].part_positions[6].append(0)
        
        
        #if INTEGRATION_DIR == 'f':  
        
        canvas_1.after(1, updateAdvect)
        rootTk.mainloop()
        
        '''
        for i_step in range(1, TOTAL_STEPS):
            if INTERP_METHOD == 'e':
                advect(i_step,dir = 'f',method='e')
            elif INTERP_METHOD == 'gmm':
                advect(i_step,dir = 'f', method='gmm')
            elif INTERP_METHOD == 'q':
                advect(i_step,dir = 'f', method='q')
        '''
        
        writeParticles(dir = 'f')
        '''       
        elif INTEGRATION_DIR == 'b':
            for i_step in range(1, TOTAL_STEPS):
                advectEnsemble(i_step,dir = 'b')
              
                
                writeParticles(dir = 'b')
        else:
            for i_step in range(1, TOTAL_STEPS ):
                advectEnsemble(i_step,dir = 'f')
                
            for i_step in range(1, TOTAL_STEPS + 1):
                advectEnsemble(i_step,dir = 'b')
        '''        
                
        #writeParticles(dir = 'a')
        print 'finished'
    else:
        print "reading particles"
        readParticles()  
        defineVclin()
        plotParticles()
            
 
    
def fitBvGmm(gp, max_gs = NUM_GAUSSIANS):
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

def interpFromGMM(ppos=[0.0,0.0]):
    
    ppos_parts = getCoordParts(ppos)
    
    #find grid cell that contains position to be interpolated
    gp0, gp1, gp2, gp3 = getGridPoints(ppos)
    
    gp0_dist, gp1_dist, gp2_dist, gp3_dist = getVclinSamples(gp0, gp1, gp2, gp3)
    
    global g_grid_params_array 
    
    i = int(gp0[0])
    j = int(gp0[1])
    params0 = g_grid_params_array[i][j] 
    if len(params0) == 0: 
        gp0_dist_transpose = gp0_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp0_dist_transpose[:,[0, 1]] = gp0_dist_transpose[:,[1, 0]]
        params0 = fitBvGmm(gp0_dist_transpose)
        g_grid_params_array[i][j] = params0

    
    total_dist_x = np.zeros(shape=0)
    total_dist_y = np.zeros(shape=0)
    #x = []
    #y = []
    for idx in range(0,len(params0)):
        cur_inter_mean = params0[idx][0]
        cur_inter_cov = params0[idx][1]
        cur_inter_ratio = params0[idx][2] 
        
        x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,SAMPLES).T
        
        total_dist_x = np.append(total_dist_x, x)
        total_dist_y = np.append(total_dist_y, y)
        
        
        #total_dist += list(np.asarray(r.mvrnorm( n = int(SAMPLES*cur_inter_ratio), mu = cur_inter_mean, Sigma = cur_inter_cov)))
        
        
    total_dist = np.asarray([list(total_dist_x), list(total_dist_y)]).T                                  
    k = stats.kde.gaussian_kde( (total_dist[:,1], total_dist[:,0]) ) 
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-4, 4), (-4, 4),title='gp0' )
    
    i = int(gp1[0])
    j = int(gp1[1])
    params1 = g_grid_params_array[i][j] 
    if len(params1) == 0: 
        gp1_dist_transpose = gp1_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp1_dist_transpose[:,[0, 1]] = gp1_dist_transpose[:,[1, 0]]
        params1 = fitBvGmm(gp1_dist_transpose)
        g_grid_params_array[i][j] = params1
        
    i = int(gp2[0])
    j = int(gp2[1])    
    params2 = g_grid_params_array[i][j] 
    if len(params2) == 0:
        gp2_dist_transpose = gp2_dist.T
        #swap columns, i.e. u and v comps after transposing matrix
        gp2_dist_transpose[:,[0, 1]] = gp2_dist_transpose[:,[1, 0]]
        params2 = fitBvGmm(gp2_dist_transpose)
        g_grid_params_array[i][j] = params2
    
    
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
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-4, 4), (-4, 4),title='gp2' )
    
    i = int(gp3[0])
    j = int(gp3[1])
    params3 = g_grid_params_array[i][j] 
    if len(params3) == 0:
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
    
    total_dist_x = np.zeros(shape=0)
    total_dist_y = np.zeros(shape=0)
    #x = []
    #y = []
    for idx in range(0,len(lerp_params)):
        cur_inter_mean =  lerp_params[idx][0]
        cur_inter_cov =  lerp_params[idx][1]
        cur_inter_ratio =  lerp_params[idx][2] 
        
        x,y = np.random.multivariate_normal(cur_inter_mean,cur_inter_cov ,int(SAMPLES*cur_inter_ratio)).T
        
        total_dist_x = np.append(total_dist_x, x)
        total_dist_y = np.append(total_dist_y, y)
        
        
        #total_dist += list(np.asarray(r.mvrnorm( n = int(SAMPLES*cur_inter_ratio), mu = cur_inter_mean, Sigma = cur_inter_cov)))
        
        
    total_dist = np.asarray([list(total_dist_x), list(total_dist_y)]).T                                  
    k = stats.kde.gaussian_kde( (total_dist[:,1], total_dist[:,0]) , bw_method = 'silverman') 
    #plotDistro( stats.kde.gaussian_kde(gp0_dist), (-3, 3), (-3, 3),title='kde' )                                  
    #plotDistro( k, (-3, 3), (-3, 3),title='total lerp' )
    
    pars = True
    
    return k, lerp_params

       
if __name__ == "__main__":  
    main()  
    
'''              
def checkIntersectWithAncestors(idx_of_streamline,new_pos, list_to_prune=[]):
    
    #we only need to check the possible new position of the current streamline
    #against positions from ancestors (which are advected first for this step)
    
    #walk backwards since local ancestors more likely to intersect
#    for particle in range(idx_of_streamline-1,-1,-1):
#    for particle in reversed(g_part_positions[idx_of_streamline].ancestors):
    current_ancestor = g_part_positions[idx_of_streamline].parent
    for gen in range(0,MAX_NUMBER_OF_GENERATIONS_TO_CHECK):
        
        ancestor_index = g_part_positions.index(current_ancestor)#index of ancestor
        if ancestor_index in list_to_prune:#if ancestor is marked for pruning, then this streamline needs to be pruned as well
            return True
        number_of_positions = len(current_ancestor.part_positions[0])
        end_index = -1
        if MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK < number_of_positions:
            end_index = number_of_positions - MAX_NUMBER_OF_POSITIONS_BACK_TO_CHECK - 1
        for pos in range(number_of_positions-1,end_index,-1):
            print pos
            ancestor_pos_x = current_ancestor.part_positions[0][pos]
            ancestor_pos_y = current_ancestor.part_positions[1][pos]
            dist = math.sqrt(math.pow(new_pos[0]-ancestor_pos_x,2) \
                             + math.pow(new_pos[1]-ancestor_pos_y,2))
            print dist
            if dist <= STREAMLINE_CROSSING_DISTANCE_THRESHOLD:
                return True
        #update for next ancestor
        
        #try:    
        #    current_ancestor = current_ancestor.parent
        #except:
        #    break
      
        if current_ancestor is None:
            break
                  
    return False #note that the root streamline can never be pruned / deleted (considered intersecting itself) 
'''

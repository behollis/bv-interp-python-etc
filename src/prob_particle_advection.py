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

QUANTILES = 100
TOTAL_STEPS = 25
integration_step_size = 0.1
SEED_LAT = 42
SEED_LON = 21
SEED_LEVEL = 0
vclin = []
cf_vclin = []

reused_vel_quantile = 0



DEBUG = False
  
MODE = 1
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/'
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
g_crisp_streamlines = []
g_part_positions_ensemble = [[],[],[],[],[],[],[]]
g_part_positions_quantile = [[],[],[],[],[],[],[]]
g_part_positions_gmm = [[],[],[],[],[],[],[]]
g_part_positions_g = [[],[],[],[],[],[],[]]
g_part_positions_ensemble_b = [[],[],[],[],[],[],[]]
g_part_positions_quantile_b = [[],[],[],[],[],[],[]]
g_part_positions_gmm_b = [[],[],[],[],[],[],[]]
g_part_positions_g_b = [[],[],[],[],[],[],[]]

part_pos_e = [];part_pos_q = [];part_pos_gmm = [];part_pos_g = []
part_pos_e.append([0,0])
part_pos_e[0][0] = SEED_LAT
part_pos_e[0][1] = SEED_LON

part_pos_q.append([0,0])
part_pos_q[0][0] = SEED_LAT
part_pos_q[0][1] = SEED_LON

part_pos_gmm.append([0,0])
part_pos_gmm[0][0] = SEED_LAT
part_pos_gmm[0][1] = SEED_LON

part_pos_g.append([0,0])
part_pos_g[0][0] = SEED_LAT
part_pos_g[0][1] = SEED_LON

part_pos_e_b = [];part_pos_q_b = [];part_pos_gmm_b = [];part_pos_g_b = []
part_pos_e_b.append([0,0])
part_pos_e_b[0][0] = SEED_LAT
part_pos_e_b[0][1] = SEED_LON

part_pos_q_b.append([0,0])
part_pos_q_b[0][0] = SEED_LAT
part_pos_q_b[0][1] = SEED_LON

part_pos_gmm_b.append([0,0])
part_pos_gmm_b[0][0] = SEED_LAT
part_pos_gmm_b[0][1] = SEED_LON

part_pos_g_b.append([0,0])
part_pos_g_b[0][0] = SEED_LAT
part_pos_g_b[0][1] = SEED_LON

r = robjects.r

ZERO_ARRAY = np.zeros(shape=(MEM,1))

# from.. http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
#http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method
def rk4(x, v, a, dt):
    """Returns final (position, velocity) tuple after
    time dt has passed.

    x: initial position (number-like object)
    v: initial velocity (number-like object)
    a: acceleration function a(x,v,dt) (must be callable)
    dt: timestep (number)"""
    
    x1 = x
    v1 = v
    a1 = a(x1, v1, 0)

    x2 = x + 0.5*v1*dt
    v2 = v + 0.5*a1*dt
    a2 = a(x2, v2, dt/2.0)

    x3 = x + 0.5*v2*dt
    v3 = v + 0.5*a2*dt
    a3 = a(x3, v3, dt/2.0)

    x4 = x + v3*dt
    v4 = v + a3*dt
    a4 = a(x4, v4, dt)

    xf = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
    vf = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

    return xf, vf
    
def plotParticles(ts_per_gp=[]):
    
    #http://docs.enthought.com/mayavi/mayavi/mlab_figures_decorations.html
    f = mayavi.mlab.gcf()
    #cam = f.scene.camera
    #cam.parallel_scale = 10
    f.scene.isometric_view()
    
     
    grid_verts = np.zeros(shape=(LAT,LON))
    grid_lat, grid_lon = np.ogrid[0:LAT,0:LON]
   
    grid_plane = mayavi.mlab.surf(grid_lat, grid_lon, grid_verts, color=(1,1,0),representation='wireframe',line_width=0.1)
    ks_plane = mayavi.mlab.surf(grid_lat, grid_lon, ts_per_gp, colormap='gray')
    #xmin=0; xmax=LAT; ymin=0; ymax=LON; zmin=-1; zmax=-1
    #ext = [xmin, xmax, ymin, ymax,zmin,zmax]
    
    #cc_plane = mayavi.mlab.imshow(g_cc, colormap='Blues', interpolate=True,transparent=True)
    
    #mayavi.mlab.points3d(63, 21, -5, color=(1,0,0),scale_factor=0.1)
    
    mode1 = np.loadtxt( OUTPUT_DATA_DIR + "crisp/" \
                          + 'mode_members1_'+str(SEED_LAT)+'_lon'+str(SEED_LON)\
                          +'_lev'+str(SEED_LEVEL))
    mode2 = np.loadtxt( OUTPUT_DATA_DIR + "crisp/" \
                          + 'mode_members2_'+str(SEED_LAT)+'_lon'+str(SEED_LON)\
                          +'_lev'+str(SEED_LEVEL))
    
    col1 = (255./255.,228./255.,196./255.)
    col_mode1 = (1.,0.,0.)
    col_mode2 = (0.,1.,0.)
    for idx in range(0,MEM, 1):
        if idx not in mode1 and idx not in mode2:
            mayavi.mlab.plot3d(g_crisp_streamlines[idx][0][:], g_crisp_streamlines[idx][1][:], \
                               g_crisp_streamlines[idx][2][:],tube_radius = None,line_width=0.1, \
                               color=col1, name='Crisp Member '+str(idx+1)) 
    for idx in range(0,MEM,1):
        if idx in mode1:
            mayavi.mlab.plot3d(g_crisp_streamlines[idx][0][:], g_crisp_streamlines[idx][1][:], \
                                   g_crisp_streamlines[idx][2][:],tube_radius = None,line_width=0.1, \
                                   color=col_mode1, name='Crisp Member '+str(idx+1)) 
        elif idx in mode2:
            mayavi.mlab.plot3d(g_crisp_streamlines[idx][0][:], g_crisp_streamlines[idx][1][:], \
                                   g_crisp_streamlines[idx][2][:],tube_radius = None,line_width=0.1, \
                                   color=col_mode2, name='Crisp Member '+str(idx+1)) 
     
    #tubes with peak number
    '''
    mayavi.mlab.plot3d(g_part_positions_ensemble[0][:], g_part_positions_ensemble[1][:], g_part_positions_ensemble[2][:], \
                       g_part_positions_ensemble[5][:], colormap='Greens',name='Ensemble peaks')
    mayavi.mlab.plot3d(g_part_positions_quantile[0][:], g_part_positions_quantile[1][:], g_part_positions_quantile[2][:], \
                       g_part_positions_quantile[5][:], colormap='Blues',name='Quantile peaks')
    mayavi.mlab.plot3d(g_part_positions_gmm[0][:], g_part_positions_gmm[1][:], g_part_positions_gmm[2][:], \
                       g_part_positions_gmm[5][:], colormap='Reds',name='GMM peaks')
    mayavi.mlab.plot3d(g_part_positions_g[0][:], g_part_positions_g[1][:], g_part_positions_g[2][:], \
                       g_part_positions_g[5][:], colormap='Purples',name='g peaks')
    
    #tubes with speed
    mayavi.mlab.plot3d(g_part_positions_ensemble[0][:], g_part_positions_ensemble[1][:], g_part_positions_ensemble[2][:], \
                       g_part_positions_ensemble[3][:], colormap='Greens',name='Ensemble speed')
    mayavi.mlab.plot3d(g_part_positions_quantile[0][:], g_part_positions_quantile[1][:], g_part_positions_quantile[2][:], \
                       g_part_positions_quantile[3][:], colormap='Blues',name='Quantile speed')
    mayavi.mlab.plot3d(g_part_positions_gmm[0][:], g_part_positions_gmm[1][:], g_part_positions_gmm[2][:], \
                       g_part_positions_gmm[3][:], colormap='Reds',name='GMM speed')
    mayavi.mlab.plot3d(g_part_positions_g[0][:], g_part_positions_g[1][:], g_part_positions_g[2][:], \
                       g_part_positions_g[3][:], colormap='Purples',name='g speed')
    
    #tubes with speed
    mayavi.mlab.plot3d(g_part_positions_ensemble[0][:], g_part_positions_ensemble[1][:], g_part_positions_ensemble[2][:], \
                       g_part_positions_ensemble[6][:], colormap='Greens',name='Ensemble peak separation')
    mayavi.mlab.plot3d(g_part_positions_quantile[0][:], g_part_positions_quantile[1][:], g_part_positions_quantile[2][:], \
                       g_part_positions_quantile[6][:], colormap='Blues',name='Quantile peak separation')
    mayavi.mlab.plot3d(g_part_positions_gmm[0][:], g_part_positions_gmm[1][:], g_part_positions_gmm[2][:], \
                       g_part_positions_gmm[6][:], colormap='Reds',name='GMM peak separation')
    mayavi.mlab.plot3d(g_part_positions_g[0][:], g_part_positions_g[1][:], g_part_positions_g[2][:], \
                       g_part_positions_g[6][:], colormap='Purples',name='g speed separation')
    '''

    mayavi.mlab.show() 
    
   
    
def advectGaussian(step, dir = 'f'):
    
    if step % 50 == 0:
        print '********************************' + str(step) + ' out of ' + str(TOTAL_STEPS)
    
    for particle in range(0, len(part_pos_e)):
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        
        if dir == 'f':
            ppos = [ part_pos_g[particle][0], part_pos_g[particle][1] ]
        else:
            ppos = [ part_pos_g_b[particle][0], part_pos_g_b[particle][1] ]
        
        #velx, vely, velz = interpVel(ppos)
        params = interpFromGaussian(ppos)
        
        velx = params[0][0]
        vely = params[1][0]
        var_u = params[0][1]
        var_v = params[1][1]
         
        if dir == 'f':
            part_pos_g[particle][0] += velx*integration_step_size
            part_pos_g[particle][1] += vely*integration_step_size
            
            # enqueue for rendering       
            for part in part_pos_g:     
                g_part_positions_g[0].append(part[0])
                g_part_positions_g[1].append(part[1]) 
                g_part_positions_g[2].append(DEPTH) 
                g_part_positions_g[3].append(np.sqrt(np.square(velx)+np.square(vely)))
                g_part_positions_g[4].append((var_u + var_v) / 2.0)
                g_part_positions_g[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions[5].append(var_v)
                g_part_positions_g[6].append(0.0)
        else:
            part_pos_g_b[particle][0] -= velx*integration_step_size
            part_pos_g_b[particle][1] -= vely*integration_step_size
            
            # enqueue for rendering       
            for part in part_pos_g_b:     
                g_part_positions_g_b[0].append(part[0])
                g_part_positions_g_b[1].append(part[1]) 
                g_part_positions_g_b[2].append(DEPTH) 
                g_part_positions_g_b[3].append(np.sqrt(np.square(velx)+np.square(vely)))
                g_part_positions_g_b[4].append((var_u + var_v) / 2.0)
                g_part_positions_g_b[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions_b[5].append(var_v)
                g_part_positions_g_b[6].append(0.0)

def getMaxPeaks(vel_prob,vel):
    m = max(vel_prob)
    p = vel_prob.index(m)
    max_1 = vel[p]
    
    vel_prob.pop(p)
    m = max(vel_prob)
    p = vel_prob.index(m)
     
    max_2 = vel[p]
    
    meet_threshold = False
    
    if max_2 >= THRESHOLD_PER * max_1:
        meet_threshold = True
    
    return max_1, max_2, meet_threshold
                
def advectGMM(step, dir = 'f'):
    
    if step % 50 == 0:
        print '********************************' +str(step) + ' out of ' + str(TOTAL_STEPS)
    
    for particle in range(0, len(part_pos_e)):
        
        global gmm_prev_max_vel_x 
        global gmm_prev_max_vel_y
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        if dir == 'f':
            ppos = [ part_pos_gmm[particle][0], part_pos_gmm[particle][1] ]
        else:
            ppos = [ part_pos_gmm_b[particle][0], part_pos_gmm_b[particle][1] ]
        
        #get peaks
        #velx, vely, velz = interpVel(ppos)
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v, u_params, v_params = interpFromGMM(ppos)
         
        #find highest prob vel
        velx_hp = gmm_prev_max_vel_x
        vely_hp = gmm_prev_max_vel_y
        
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        
        num_x_peaks = len(velx_prob) #len(u_params)
        num_y_peaks = len(vely_prob) #len(v_params)
        
        # take peaks from largest g comps
        '''
        if len(u_params) > 0:
            temp_max = 0
            max_idx = 0
            for i in range(0,len(u_params)):
                if u_params[i] > temp_max:
                    max_idx = i
            
            
            
            #get max peak mean
            max_x_1 = u_params[max_idx][0]
            u_params[0].pop(max_idx);u_params[1].pop(max_idx);u_params[2].pop(max_idx)
            
            temp_max = 0
            max_idx = 0
            for i in range(0,len(u_params)):
                if u_params[i] > temp_max:
                    max_idx = i
            
            
            #get 2nd peak mean
            max_x_2 = u_params[max_idx][0]
        
        # take peaks from largest g comps
        if len(v_params) > 0:
            temp_max = 0
            max_idx = 0
            for i in range(0,len(v_params)):
                if v_params[i] > temp_max:
                    max_idx = i
            
            
            
            #get max peak mean
            max_y_1 = v_params[max_idx][0]
            v_params[0].pop(max_idx);v_params[1].pop(max_idx);v_params[2].pop(max_idx)
            
            temp_max = 0
            max_idx = 0
            for i in range(0,len(v_params)):
                if v_params[i] > temp_max:
                    max_idx = i
            
            
            #get 2nd peak mean
            max_y_2 = v_params[max_idx][0]
        
        x_diff = math.fabs(max_x_1 - max_x_2)
        y_diff = math.fabs(max_y_1 - max_y_2)
        max_peak_diff = max([x_diff,y_diff])
        
        if MODE == 1:
            velx_hp = max_x_1
        else: #MODE ==2:
            velx_hp = max_x_2
        
        if MODE == 1:
            vely_hp = max_y_1
        else: #MODE ==2:
            vely_hp = max_y_2
        
        '''
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
    
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                 
                velx_hp = velx[p] 
        else:
            print "WARNING: no max velx returned for GMM lerp @ position " + str(ppos) 
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
        else:
            print "WARNING: no max vely returned for GMM lerp @ position " + str(ppos)  
        
        gmm_prev_max_vel_x = velx_hp
        gmm_prev_max_vel_y = vely_hp
        
        #if step % 10 == 0 or step == 1:
        #    print str(step) + " ensemble: pos " + str(ppos) + " u peak: " + str(velx_hp) + " v peak: " + str(vely_hp)
        
        
        if dir == 'f':
            part_pos_gmm[particle][0] += velx_hp*integration_step_size
            part_pos_gmm[particle][1] += vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_gmm:     
                g_part_positions_gmm[0].append(part[0])
                g_part_positions_gmm[1].append(part[1]) 
                g_part_positions_gmm[2].append(DEPTH) 
                g_part_positions_gmm[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_gmm[4].append((var_u + var_v) / 2.0)
                g_part_positions_gmm[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions[5].append(var_v)
                g_part_positions_gmm[6].append(max_peak_diff)
        else:
            part_pos_gmm_b[particle][0] -= velx_hp*integration_step_size
            part_pos_gmm_b[particle][1] -= vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_gmm_b:     
                g_part_positions_gmm_b[0].append(part[0])
                g_part_positions_gmm_b[1].append(part[1]) 
                g_part_positions_gmm_b[2].append(DEPTH) 
                g_part_positions_gmm_b[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_gmm_b[4].append((var_u + var_v) / 2.0)
                g_part_positions_gmm_b[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions_b[5].append(var_v)
                g_part_positions_gmm_b[6].append(max_peak_diff)
            
    
def advectEnsemble(step, dir = 'f'):
    
    if step % 50 == 0:
        print '********************************' + str(step) + ' out of ' + str(TOTAL_STEPS)
    
    for particle in range(0, len(part_pos_e)):
        
        global e_prev_max_vel_x 
        global e_prev_max_vel_y
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        if dir == 'f':
            ppos = [ part_pos_e[particle][0], part_pos_e[particle][1] ]
        else:
            ppos = [ part_pos_e_b[particle][0], part_pos_e_b[particle][1] ]
            
        #get peaks
        #velx, vely, velz = interpVel(ppos)
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v = interpVelFromEnsemble(ppos)
         
        #find highest prob vel
        velx_hp = e_prev_max_vel_x
        vely_hp = e_prev_max_vel_y
        
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        
        num_x_peaks = len(velx_prob)
        num_y_peaks = len(vely_prob)
        
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
            
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if num_y_peaks > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]    
    
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                 
                velx_hp = velx[p] 
        else:
            print "WARNING: no max velx returned for ensemble lerp @ position " + str(ppos) 
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
        else:
            print "WARNING: no max vely returned for ensemble lerp @ position " + str(ppos) 
        
            
        e_prev_max_vel_x = velx_hp
        e_prev_max_vel_y = vely_hp
        
        print "Ensemble u vel: " + str(velx_hp)
        print "Ensemble v vel: " + str(vely_hp)
        
        #if step % 10 == 0 or step == 1:
        #    print str(step) + " ensemble: pos " + str(ppos) + " u peak: " + str(velx_hp) + " v peak: " + str(vely_hp)
        
        if dir == 'f':
            part_pos_e[particle][0] += velx_hp*integration_step_size
            part_pos_e[particle][1] += vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_e:     
                g_part_positions_ensemble[0].append(part[0])
                g_part_positions_ensemble[1].append(part[1]) 
                g_part_positions_ensemble[2].append(DEPTH) 
                g_part_positions_ensemble[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_ensemble[4].append((var_u + var_v) / 2.0)
                g_part_positions_ensemble[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions[5].append(var_v)
                g_part_positions_ensemble[6].append(max_peak_diff)
        else:
            part_pos_e_b[particle][0] -= velx_hp*integration_step_size
            part_pos_e_b[particle][1] -= vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_e_b:     
                g_part_positions_ensemble_b[0].append(part[0])
                g_part_positions_ensemble_b[1].append(part[1]) 
                g_part_positions_ensemble_b[2].append(DEPTH) 
                g_part_positions_ensemble_b[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_ensemble_b[4].append((var_u + var_v) / 2.0)
                g_part_positions_ensemble_b[5].append((len(velx_prob)+len(vely_prob)) / 2.0)#g_part_positions_b[5].append(var_v)
                g_part_positions_ensemble_b[6].append(max_peak_diff)
            
            
def advectQuantile(step, dir = 'f'):
    
    global reused_vel_quantile
    
    if step % 50 == 0:
        print str(step) + ' out of ' + str(TOTAL_STEPS)
    
    for particle in range(0, len(part_pos_q)):
        
        global q_prev_max_vel_x
        global q_prev_max_vel_y
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        if dir == 'f':
            ppos = [ part_pos_q[particle][0], part_pos_q[particle][1] ]
        else:
            ppos = [ part_pos_q_b[particle][0], part_pos_q_b[particle][1] ]
            
        
        #get peaks
        #velx, vely, velz = interpVel(ppos)
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v =  interpFromQuantiles(ppos)
        
        #find highest prob vel
        velx_hp = q_prev_max_vel_x
        vely_hp = q_prev_max_vel_y
        
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        num_x_peaks = len(velx_prob)
        num_y_peaks = len(vely_prob)
        
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
    
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if num_y_peaks > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
    
        '''
    
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                 
                velx_hp = velx[p] 
        else:
            print "WARNING: no max velx returned for quantile lerp @ position " + str(ppos) 
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
        else:
            print "WARNING: no max vely returned for quantile lerp @ position " + str(ppos) 
        
        
        print "Quantile u vel: " + str(velx_hp)
        print "Quantile v vel: " + str(vely_hp)
        
        '''
        if velx_hp <= 0.15 and velx_hp >= -0.15 and velx_hp == q_prev_max_vel_x:
            velx_hp = 0
        
        if vely_hp <= 0.15 and vely_hp >= -0.15 and vely_hp == q_prev_max_vel_y:
            vely_hp = 0    
        '''
            
        q_prev_max_vel_x = velx_hp
        q_prev_max_vel_y = vely_hp
        
        #if step % 10 == 0 or step == 1:
        #    print str(step) + " quantile: pos " + str(ppos) + " u peak: " + str(velx_hp) + " v peak: " + str(vely_hp)
        if dir == 'f':
            part_pos_q[particle][0] += velx_hp*integration_step_size
            part_pos_q[particle][1] += vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_q:     
                g_part_positions_quantile[0].append(part[0])
                g_part_positions_quantile[1].append(part[1]) 
                g_part_positions_quantile[2].append(DEPTH) 
                g_part_positions_quantile[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_quantile[4].append((var_u + var_v) / 2.0)
                g_part_positions_quantile[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
                g_part_positions_quantile[6].append(max_peak_diff)
        else:
            part_pos_q_b[particle][0] -= velx_hp*integration_step_size
            part_pos_q_b[particle][1] -= vely_hp*integration_step_size
        
            # enqueue for rendering       
            for part in part_pos_q_b:     
                g_part_positions_quantile_b[0].append(part[0])
                g_part_positions_quantile_b[1].append(part[1]) 
                g_part_positions_quantile_b[2].append(DEPTH) 
                g_part_positions_quantile_b[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
                g_part_positions_quantile_b[4].append((var_u + var_v) / 2.0)
                g_part_positions_quantile_b[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
                g_part_positions_quantile_b[6].append(max_peak_diff)
            
        
def interpVelFromEnsemble(ppos=[0.0,0.0]):
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
    
    gpt0_dist = np.zeros(shape=(2,600))
    gpt1_dist = np.zeros(shape=(2,600))
    gpt2_dist = np.zeros(shape=(2,600))
    gpt3_dist = np.zeros(shape=(2,600))
    
    '''
    if DEBUG is True:
        print "ensemble interp"
        print "gp0";print gpt0[0]; print gpt0[1]
        print "gp1";print gpt1[0]; print gpt1[1]
        print "gp2";print gpt2[0]; print gpt2[1]
        print "gp3";print gpt3[0]; print gpt3[1]
    '''
    
    for idx in range(0,600):
        gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
        
    #SAMP = 2000

    
       
    #lerp ensemble samples
    lerp_u_gp0_gp1 = lerp( np.asarray(gpt0_dist[0] ), np.asarray(gpt1_dist[0]), w = ppos_parts[0][0] )
    lerp_u_gp2_gp3 = lerp( np.asarray(gpt2_dist[0] ), np.asarray(gpt3_dist[0]), w = ppos_parts[0][0] ) 
    lerp_u = lerp( np.asarray(lerp_u_gp0_gp1), np.asarray(lerp_u_gp2_gp3), w = ppos_parts[1][0] )  
    
    lerp_v_gp0_gp1 = lerp( np.asarray(gpt0_dist[1] ), np.asarray(gpt1_dist[1]), w = ppos_parts[0][0] )
    lerp_v_gp2_gp3 = lerp( np.asarray(gpt2_dist[1] ), np.asarray(gpt3_dist[1]), w = ppos_parts[0][0] ) 
    lerp_v = lerp( np.asarray(lerp_v_gp0_gp1), np.asarray(lerp_v_gp2_gp3), w = ppos_parts[1][0] )  
    
    #x = linspace( lerp_u[0], lerp_u[-1], len(lerp_u) )
    #y = linspace( lerp_v[0], lerp_v[-1], len(lerp_v) )
    
    x = linspace( -50, 50, 600 )
    y = linspace( -50, 50, 600 )
        
    #find peaks...
    try:
        k = [ stats.gaussian_kde(lerp_u), stats.gaussian_kde(lerp_v) ]
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    var0 = np.std(k[0](x), axis=None, dtype=None, out=None, ddof=0)
    var1 = np.std(k[1](y), axis=None, dtype=None, out=None, ddof=0)

    _max_u, _min_u = peakdetect(k[0](x),x,lookahead=2,delta=0)
    _max_v, _min_v = peakdetect(k[1](y),y,lookahead=2,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    xm_v = [p[0] for p in _max_v]
    ym_u = [p[1] for p in _max_u]
    ym_v = [p[1] for p in _max_v]
    
    '''
    #plot interpolated kde's
    plt.figure()
    plt.title("ensemble")
    p1, = plt.plot(x,k[0](x),'-', color='red')
    p2, = plt.plot(y,k[1](y),'-', color='blue')
    plt.legend([p2, p1], ["v", "u"])
    
    #plot peaks
    plt.hold(True)
    plt.plot(xm_u, ym_u, 'x', color='black')
    plt.plot(xm_v, ym_v, 'x', color='black')
    plt.savefig('../png/e_'+str(ppos)+'.png')
    '''
    
    
    return (xm_u, ym_u, xm_v, ym_v, 0.0, var0, var1) 
            
            
def interpFromQuantiles(ppos=[0.0,0.0]):
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
    
    gpt0_dist = np.zeros(shape=(2,600))
    gpt1_dist = np.zeros(shape=(2,600))
    gpt2_dist = np.zeros(shape=(2,600))
    gpt3_dist = np.zeros(shape=(2,600))
    
    '''
    if DEBUG is True:
        print "quantile interp"
        print "gp0";print gpt0[0]; print gpt0[1]
        print "gp1";print gpt1[0]; print gpt1[1]
        print "gp2";print gpt2[0]; print gpt2[1]
        print "gp3";print gpt3[0]; print gpt3[1]
    '''
    
    for idx in range(0,600):
        gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
    
    quantiles = list(spread(0, 1.0, QUANTILES-1, mode=3)) 
    quantiles.sort()
    
    #find random variable value of quantiles for pdf 
    q_gpt0_dist_u = [];q_gpt0_dist_v = []
    q_gpt1_dist_u = [];q_gpt1_dist_v = []
    q_gpt2_dist_u = [];q_gpt2_dist_v = []
    q_gpt3_dist_u = [];q_gpt3_dist_v = []
    
    for q in quantiles:
        q_gpt0_dist_u.append(r.quantile(robjects.FloatVector(gpt0_dist[0]), q)[0])
        q_gpt0_dist_v.append(r.quantile(robjects.FloatVector(gpt0_dist[1]), q)[0])
        
        q_gpt1_dist_u.append(r.quantile(robjects.FloatVector(gpt1_dist[0]), q)[0])
        q_gpt1_dist_v.append(r.quantile(robjects.FloatVector(gpt1_dist[1]), q)[0])
        
        q_gpt2_dist_u.append(r.quantile(robjects.FloatVector(gpt2_dist[0]), q)[0])
        q_gpt2_dist_v.append(r.quantile(robjects.FloatVector(gpt2_dist[1]), q)[0])
        
        q_gpt3_dist_u.append(r.quantile(robjects.FloatVector(gpt3_dist[0]), q)[0])
        q_gpt3_dist_v.append(r.quantile(robjects.FloatVector(gpt3_dist[1]), q)[0])
       
    #create np arrays 
    #q_gpt0_dist_u_array = np.asarray(q_gpt0_dist_u);q_gpt0_dist_v_array = np.asarray(q_gpt0_dist_v)
    #q_gpt1_dist_u_array = np.asarray(q_gpt1_dist_u);q_gpt1_dist_v_array = np.asarray(q_gpt1_dist_v)
    #q_gpt2_dist_u_array = np.asarray(q_gpt2_dist_u);q_gpt2_dist_v_array = np.asarray(q_gpt2_dist_v)
    #q_gpt3_dist_u_array = np.asarray(q_gpt3_dist_u);q_gpt3_dist_v_array = np.asarray(q_gpt3_dist_v)

    #lerp quantiles
    
    #find peaks...
    '''
    if len(gpt0_dist[0]) < 5 or len(gpt1_dist[0]) < 5 or len(gpt2_dist[0]) < 5 or len(gpt3_dist[0]) < 5 or \
        len(gpt0_dist[1]) < 5 or len(gpt1_dist[1]) < 5 or len(gpt2_dist[1]) < 5 or len(gpt3_dist[1]) < 5:
        print "return in quantile interp @" + str(ppos)
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    if np.array_equal(gpt0_dist[0], ZERO_ARRAY) or np.array_equal(gpt1_dist[0], ZERO_ARRAY):
        print "return in quantile interp @" + str(ppos)
        return ([], [], [], [], 0.0, 0.0, 0.0)
    '''
    
    try:
        k = stats.gaussian_kde(gpt0_dist[0]); l = stats.gaussian_kde(gpt1_dist[0])
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    lerp_u_gp0_gp1_prob = quantileLerp( k, l, np.asarray(q_gpt0_dist_u), np.asarray(q_gpt1_dist_u), alpha = ppos_parts[0][0] )
    lerp_u_gp0_gp1_values = lerp(np.asarray(q_gpt0_dist_u), np.asarray(q_gpt1_dist_u), w = ppos_parts[0][0] )
    
    try:
        lerp_u_gp2_gp3_prob = quantileLerp( stats.gaussian_kde(gpt2_dist[0]), stats.gaussian_kde(gpt3_dist[0]), np.asarray(q_gpt2_dist_u), np.asarray(q_gpt3_dist_u), alpha = ppos_parts[0][0] )
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    lerp_u_gp2_gp3_values = lerp(np.asarray(q_gpt2_dist_u), np.asarray(q_gpt3_dist_u), w = ppos_parts[0][0] )
    
    '''
    plt.figure()
    plt.title("gpt0_dist, alpha: " + str(ppos_parts[0][0]))
    x = linspace( -5, 2, 600 )
    plt.plot(lerp_u_gp0_gp1_values,lerp_u_gp0_gp1_prob,'-', color='black')
    plt.show()
    '''
    
    NUM_SAMPLES = 1000
    
    samples_numbers = lerp_u_gp0_gp1_prob * NUM_SAMPLES
    samples_gp0_gp1_lerp = []
    for prob_idx in range(0,len(lerp_u_gp0_gp1_prob)):
        #if not math.isnan(samples_numbers[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers[prob_idx])):
            samples_gp0_gp1_lerp.append(lerp_u_gp0_gp1_values[prob_idx])
            
    samples_numbers2 = lerp_u_gp2_gp3_prob * NUM_SAMPLES
    samples_gp2_gp3_lerp = []
    for prob_idx in range(0,len(lerp_u_gp2_gp3_prob)):
        #if not math.isnan(samples_numbers2[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers2[prob_idx])):
            samples_gp2_gp3_lerp.append(lerp_u_gp2_gp3_values[prob_idx])
    
    '''
    plt.figure()
    plt.title("gpt0_dist resampled, alpha: " + str(ppos_parts[0][0]))
    x = linspace( -5, 2, 600 )
    plt.plot(x,stats.gaussian_kde(samples_gp0_gp1_lerp)(x),'-', color='black')
    plt.show()
    '''
    
    '''
    plt.figure()
    plt.title("lerp_u_gp0_gp1, alpha: " + str(ppos_parts[0][0]))
    x = linspace( -10, 10, 600 )
    plt.plot(x,stats.gaussian_kde(lerp_u_gp0_gp1)(x),'-', color='black')
    plt.show()
    plt.figure()
    plt.title("lerp_u_gp2_gp3")
    x = linspace( -10, 10, 600 )
    plt.plot(x,stats.gaussian_kde(lerp_u_gp2_gp3)(x),'-', color='black')
    plt.show()
    '''
    
    q_lerp_gpt0_gpt1_dist_u = [];q_lerp_gpt2_gpt3_dist_u = []
    for q in quantiles:
        q_lerp_gpt0_gpt1_dist_u.append(r.quantile(robjects.FloatVector(samples_gp0_gp1_lerp), q)[0])
        q_lerp_gpt2_gpt3_dist_u.append(r.quantile(robjects.FloatVector(samples_gp2_gp3_lerp), q)[0])
    
    try:
        lerp_u_prob = quantileLerp( stats.gaussian_kde(samples_gp0_gp1_lerp), stats.gaussian_kde(samples_gp2_gp3_lerp), np.asarray(q_lerp_gpt0_gpt1_dist_u), np.asarray(q_lerp_gpt2_gpt3_dist_u), alpha = ppos_parts[1][0] )
        lerp_u_prob_2 = quantileLerp( interpolate.interp1d( lerp_u_gp0_gp1_values, lerp_u_gp0_gp1_prob ), \
                                    interpolate.interp1d( lerp_u_gp2_gp3_values, lerp_u_gp2_gp3_prob ), \
                                    np.asarray(q_lerp_gpt0_gpt1_dist_u), \
                                    np.asarray(q_lerp_gpt2_gpt3_dist_u), alpha = ppos_parts[1][0] )
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
        
    lerp_u_values = lerp(np.asarray(q_lerp_gpt0_gpt1_dist_u), np.asarray(q_lerp_gpt2_gpt3_dist_u), w = ppos_parts[1][0] )
    
    samples_numbers3 = lerp_u_prob * NUM_SAMPLES
    samples_u_lerp = []
    for prob_idx in range(0,len(lerp_u_prob)):
        #if not math.isnan(samples_numbers3[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers3[prob_idx])):
            if not math.isnan(lerp_u_values[prob_idx]) and not math.isinf(lerp_u_values[prob_idx]):
                samples_u_lerp.append(lerp_u_values[prob_idx])
    
    '''
    plt.figure()
    x = linspace( -10, 10, 600 )
    plt.plot(x,stats.gaussian_kde(samples_lerp_u)(x),'-', color='black')
    plt.show()
    '''
     
    '''            
    if np.array_equal(gpt0_dist[1], ZERO_ARRAY) or np.array_equal(gpt1_dist[1], ZERO_ARRAY):
        print "return in quantile interp @" + str(ppos)
        return ([], [], [], [], 0.0, 0.0, 0.0)
    '''
           
    try:            
        k = stats.gaussian_kde(gpt0_dist[1]); l = stats.gaussian_kde(gpt1_dist[1])
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    lerp_v_gp0_gp1_prob = quantileLerp( k, l, np.asarray(q_gpt0_dist_v), np.asarray(q_gpt1_dist_v), alpha = ppos_parts[0][0] )
    lerp_v_gp0_gp1_values = lerp(np.asarray(q_gpt0_dist_v), np.asarray(q_gpt1_dist_v), w = ppos_parts[0][0] )
    
    try:
        lerp_v_gp2_gp3_prob = quantileLerp( stats.gaussian_kde(gpt2_dist[1]), stats.gaussian_kde(gpt3_dist[1]), \
                                             np.asarray(q_gpt2_dist_v), np.asarray(q_gpt3_dist_v), alpha = ppos_parts[0][0] ) 
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
        
    lerp_v_gp2_gp3_values = lerp(np.asarray(q_gpt2_dist_v), np.asarray(q_gpt3_dist_v), w = ppos_parts[0][0] )
    
    samples_numbers4 = lerp_v_gp0_gp1_prob * NUM_SAMPLES
    samples_gp0_gp1_lerp_v = []
    for prob_idx in range(0,len(lerp_v_gp0_gp1_prob)):
        #if not math.isnan(samples_numbers4[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers4[prob_idx])):
            samples_gp0_gp1_lerp_v.append(lerp_v_gp0_gp1_values[prob_idx])
            
    samples_numbers5 = lerp_v_gp2_gp3_prob * NUM_SAMPLES
    samples_gp2_gp3_lerp_v = []
    for prob_idx in range(0,len(lerp_v_gp2_gp3_prob)):
        #if not math.isnan(samples_numbers5[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers5[prob_idx])):
            samples_gp2_gp3_lerp_v.append(lerp_v_gp2_gp3_values[prob_idx])
   
    #samples_gp2_gp3_lerp_v = 
    #samples_gp2_gp3_lerp_v = 
   
    q_lerp_gpt0_gpt1_dist_v = [];q_lerp_gpt2_gpt3_dist_v = []
    for q in quantiles:
        q_lerp_gpt0_gpt1_dist_v.append(r.quantile(robjects.FloatVector(samples_gp0_gp1_lerp_v), q)[0])
        q_lerp_gpt2_gpt3_dist_v.append(r.quantile(robjects.FloatVector(samples_gp2_gp3_lerp_v), q)[0])
    
    try:
        lerp_v_prob = quantileLerp( stats.gaussian_kde(samples_gp0_gp1_lerp_v), stats.gaussian_kde(samples_gp2_gp3_lerp_v), \
                                     np.asarray(q_lerp_gpt0_gpt1_dist_v), np.asarray(q_lerp_gpt2_gpt3_dist_v), alpha = ppos_parts[1][0] ) 
        lerp_v_prob_2 = quantileLerp( interpolate.interp1d( lerp_v_gp0_gp1_values, lerp_v_gp0_gp1_prob ), \
                                    interpolate.interp1d( lerp_v_gp2_gp3_values, lerp_v_gp2_gp3_prob ), \
                                    np.asarray(q_lerp_gpt0_gpt1_dist_v), \
                                    np.asarray(q_lerp_gpt2_gpt3_dist_v), alpha = ppos_parts[1][0] )
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
         
    lerp_v_values = lerp(np.asarray(q_lerp_gpt0_gpt1_dist_v), np.asarray(q_lerp_gpt2_gpt3_dist_v), w = ppos_parts[1][0] )
    
    samples_numbers6 = lerp_v_prob * NUM_SAMPLES
    samples_v_lerp = []
    for prob_idx in range(0,len(lerp_v_prob)):
        #if not math.isnan(samples_numbers6[prob_idx]):
        #    continue
        for num in range(0,int(samples_numbers6[prob_idx])):
            if not math.isnan(lerp_v_values[prob_idx]) and not math.isinf(lerp_v_values[prob_idx]): 
                samples_v_lerp.append(lerp_v_values[prob_idx])
    
    x = linspace( -20, 20, 1000 )
    y = linspace( -20, 20, 1000 )
        
    #find peaks...
    '''
    if len(samples_u_lerp) < 20 or len(samples_v_lerp) < 20:
        print "return in quantile interp @" + str(ppos)
        return ([], [], [], [], 0.0, 0.0, 0.0)
    '''
    
    quantile_interp_u = interpolate.interp1d(lerp_u_values,lerp_u_prob_2)
    quantile_interp_v = interpolate.interp1d(lerp_v_values,lerp_v_prob_2)
    
    try:
        k = [ stats.gaussian_kde(samples_u_lerp), stats.gaussian_kde(samples_v_lerp) ]
        k2 = [ quantile_interp_u, quantile_interp_v ]
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0)
    
    #var0 = np.std(k[0](x), axis=None, dtype=None, out=None, ddof=0)
    #var1 = np.std(k[1](y), axis=None, dtype=None, out=None, ddof=0)
    
    x = linspace( min(lerp_u_values), max(lerp_u_values), 1000 )
    y = linspace( min(lerp_v_values), max(lerp_v_values), 1000 )
    var0 = np.std(k2[0](x), axis=None, dtype=None, out=None, ddof=0)
    var1 = np.std(k2[1](y), axis=None, dtype=None, out=None, ddof=0)

    #_max_u, _min_u = peakdetect(k[0](x),x,lookahead=5,delta=0)
    #_max_v, _min_v = peakdetect(k[1](y),y,lookahead=5,delta=0)
    _max_u, _min_u = peakdetect(k2[0](x),x,lookahead=5,delta=0)
    _max_v, _min_v = peakdetect(k2[1](y),y,lookahead=5,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    xm_v = [p[0] for p in _max_v]
    ym_u = [p[1] for p in _max_u]
    ym_v = [p[1] for p in _max_v]
    
    '''
    #plot interpolated kde's
    #if len(_max_u) == 0 or len(_max_v) == 0:
    plt.figure()
    plt.title("quantile")
    p1, = plt.plot(x,k[0](x),'-', color='red')
    p2, = plt.plot(y,k[1](y),'-', color='blue')
    plt.legend([p2, p1], ["v", "u"])
    
    #plot peaks
    plt.hold(True)
    plt.plot(xm_u, ym_u, 'x', color='black')
    plt.plot(xm_v, ym_v, 'x', color='black')
    plt.savefig('../png/q_'+str(ppos)+'.png')
    '''

    return (xm_u, ym_u, xm_v, ym_v, 0.0, var0, var1) 


def fitGaussian(gp=[0.,0.]):
    #fit single gaussian
    m = r.mean(robjects.vectors.FloatVector(gp));var= r.var(robjects.vectors.FloatVector(gp))
    return [m[0],var[0]]
    
def fitGMM(gp, max_gs=2):
    
    #suppress std out number of iterations using r.invisible()
    try:
        mixmdl = r.invisible(r.normalmixEM(robjects.vectors.FloatVector(gp), k = max_gs, maxit = EM_MAX_ITR, maxrestarts=EM_MAX_RESTARTS))
    except:
        return [[0.]*max_gs,[0.]*max_gs, [0.]*max_gs ]
    
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

def lerpGMMPair(norm_params1, norm_params2, alpha, steps=1, num_gs=3):     
    ''' handles equal number of constituent gaussians '''
    sorted(norm_params2, key=operator.itemgetter(0), reverse=False)
    sorted(norm_params1, key=operator.itemgetter(0), reverse=False)
   
    if steps != 0:  
        incr = alpha / steps
    else:
        incr = alpha
   
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
    
    #return interp GMM params
    return norm_params1

def interpFromGMM(ppos=[0.0,0.0]):
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
    
    gpt0_dist = np.zeros(shape=(2,600))
    gpt1_dist = np.zeros(shape=(2,600))
    gpt2_dist = np.zeros(shape=(2,600))
    gpt3_dist = np.zeros(shape=(2,600))
    
    '''
    if DEBUG is True:
        print "gp0";print gpt0[0]; print gpt0[1]
        print "gp1";print gpt1[0]; print gpt1[1]
        print "gp2";print gpt2[0]; print gpt2[1]
        print "gp3";print gpt3[0]; print gpt3[1]
    '''
    
    for idx in range(0,MEM):
        gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
    
    #check for "bad" distributions
    if len(gpt0_dist[0]) < 5 or len(gpt1_dist[0]) < 5 or len(gpt2_dist[0]) < 5 or len(gpt3_dist[0]) < 5 or \
        len(gpt0_dist[1]) < 5 or len(gpt1_dist[1]) < 5 or len(gpt2_dist[1]) < 5 or len(gpt3_dist[1]) < 5:
        print "return in GMM interp @" + str(ppos)
        return ([], [], [], [], 0.0, 0.0, 0.0, [], [])
    
    #get gmm's
    #NOTE: need to check if dist is guassian-like. if so, don't try to fit more than one gaussian to distribution or you'll get convergence issues with EM alg
    gp0_parms_u = fitGMM(gp=list(gpt0_dist[0][:]),max_gs=MAX_GMM_COMP);gp0_parms_v = fitGMM(list(gpt0_dist[1][:]),max_gs=MAX_GMM_COMP)
    gp1_parms_u = fitGMM(gp=list(gpt1_dist[0][:]),max_gs=MAX_GMM_COMP);gp1_parms_v = fitGMM(list(gpt1_dist[1][:]),max_gs=MAX_GMM_COMP)
    gp2_parms_u = fitGMM(gp=list(gpt2_dist[0][:]),max_gs=MAX_GMM_COMP);gp2_parms_v = fitGMM(list(gpt2_dist[1][:]),max_gs=MAX_GMM_COMP)
    gp3_parms_u = fitGMM(gp=list(gpt3_dist[0][:]),max_gs=MAX_GMM_COMP);gp3_parms_v = fitGMM(list(gpt3_dist[1][:]),max_gs=MAX_GMM_COMP)
    
    lerp_u_gp0_gp1_params = lerpGMMPair(np.asarray(gp0_parms_u), np.asarray(gp1_parms_u), alpha = ppos_parts[0][0], steps = 1, num_gs = MAX_GMM_COMP )
    lerp_u_gp2_gp3_params = lerpGMMPair(np.asarray(gp2_parms_u), np.asarray(gp3_parms_u), alpha = ppos_parts[0][0], steps = 1, num_gs = MAX_GMM_COMP )
    lerp_u_params = lerpGMMPair( np.asarray(lerp_u_gp0_gp1_params), np.asarray(lerp_u_gp2_gp3_params), alpha = ppos_parts[1][0], steps = 1, num_gs = MAX_GMM_COMP )
    
    lerp_v_gp0_gp1_params = lerpGMMPair(np.asarray(gp0_parms_v), np.asarray(gp1_parms_v), alpha = ppos_parts[0][0], steps = 1, num_gs = MAX_GMM_COMP )
    lerp_v_gp2_gp3_params = lerpGMMPair(np.asarray(gp2_parms_v), np.asarray(gp3_parms_v), alpha = ppos_parts[0][0], steps = 1, num_gs = MAX_GMM_COMP )
    lerp_v_params = lerpGMMPair( np.asarray(lerp_v_gp0_gp1_params), np.asarray(lerp_v_gp2_gp3_params), alpha = ppos_parts[1][0], steps = 1, num_gs = MAX_GMM_COMP )
    
    x = linspace( -50, 50, MEM )
    y = linspace( -50, 50, MEM )
    
    #return interp GMM 
    SAMPLES = MEM
    total_dist_u = []
    for idx in range(0,len(lerp_u_params)):
        cur_inter_mean = lerp_u_params[idx][0];cur_inter_stdev = lerp_u_params[idx][1];cur_inter_ratio = lerp_u_params[idx][2] 
        total_dist_u += list(np.asarray(r.rnorm(int(SAMPLES*cur_inter_ratio), mean=cur_inter_mean, sd = cur_inter_stdev)))
    total_dist_v = []
    for idx in range(0,len(lerp_v_params)):
        cur_inter_mean = lerp_v_params[idx][0];cur_inter_stdev = lerp_v_params[idx][1];cur_inter_ratio = lerp_v_params[idx][2] 
        total_dist_v += list(np.asarray(r.rnorm(int(SAMPLES*cur_inter_ratio), mean=cur_inter_mean, sd = cur_inter_stdev)))
        
    try:
        k = [ stats.gaussian_kde(total_dist_u), stats.gaussian_kde(total_dist_v) ]
    except:
        return ([], [], [], [], 0.0, 0.0, 0.0, [], [])
    
    var0 = np.std(k[0](x), axis=None, dtype=None, out=None, ddof=0)
    var1 = np.std(k[1](y), axis=None, dtype=None, out=None, ddof=0)

    _max_u, _min_u = peakdetect(k[0](x),x,lookahead=2,delta=0)
    _max_v, _min_v = peakdetect(k[1](y),y,lookahead=2,delta=0)
    
    xm_u = [p[0] for p in _max_u]
    xm_v = [p[0] for p in _max_v]
    ym_u = [p[1] for p in _max_u]
    ym_v = [p[1] for p in _max_v]

    return (xm_u, ym_u, xm_v, ym_v, 0.0, var0, var1,lerp_u_params,lerp_v_params)  

def interpFromGaussian(ppos=[0.0,0.0]):
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
    
    gpt0_dist = np.zeros(shape=(2,600))
    gpt1_dist = np.zeros(shape=(2,600))
    gpt2_dist = np.zeros(shape=(2,600))
    gpt3_dist = np.zeros(shape=(2,600))
    
    '''
    if DEBUG is True:
        print "gp0";print gpt0[0]; print gpt0[1]
        print "gp1";print gpt1[0]; print gpt1[1]
        print "gp2";print gpt2[0]; print gpt2[1]
        print "gp3";print gpt3[0]; print gpt3[1]
    '''
    
    for idx in range(0,MEM):
        gpt0_dist[0][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_dist[1][idx] = vclin[idx][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_dist[0][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_dist[1][idx] = vclin[idx][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_dist[0][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_dist[1][idx] = vclin[idx][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_dist[0][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_dist[1][idx] = vclin[idx][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
    
    #get gmm's
    #NOTE: need to check if dist is guassian-like. if so, don't try to fit more than one gaussian to distribution or you'll get convergence issues with EM alg
    gp0_parms_u = fitGaussian(gp=list(gpt0_dist[0][:]));gp0_parms_v = fitGaussian(list(gpt0_dist[1][:]))
    gp1_parms_u = fitGaussian(gp=list(gpt1_dist[0][:]));gp1_parms_v = fitGaussian(list(gpt1_dist[1][:]))
    gp2_parms_u = fitGaussian(gp=list(gpt2_dist[0][:]));gp2_parms_v = fitGaussian(list(gpt2_dist[1][:]))
    gp3_parms_u = fitGaussian(gp=list(gpt3_dist[0][:]));gp3_parms_v = fitGaussian(list(gpt3_dist[1][:]))
    
    lerp_u_gp0_gp1_params = lerp(np.asarray(gp0_parms_u), np.asarray(gp1_parms_u), w = ppos_parts[0][0] )
    lerp_u_gp2_gp3_params = lerp(np.asarray(gp2_parms_u), np.asarray(gp3_parms_u), w = ppos_parts[0][0] )
    lerp_u_params = lerp( np.asarray(lerp_u_gp0_gp1_params), np.asarray(lerp_u_gp2_gp3_params), w = ppos_parts[1][0] )
    
    lerp_v_gp0_gp1_params = lerp(np.asarray(gp0_parms_v), np.asarray(gp1_parms_v), w = ppos_parts[0][0] )
    lerp_v_gp2_gp3_params = lerp(np.asarray(gp2_parms_v), np.asarray(gp3_parms_v), w = ppos_parts[0][0] )
    lerp_v_params = lerp( np.asarray(lerp_v_gp0_gp1_params), np.asarray(lerp_v_gp2_gp3_params), w = ppos_parts[1][0] )

    return [lerp_u_params, lerp_v_params]
                    
def KSTestForLevel(vclin, curr_level=0):
    
    array_of_ts_per_gp = np.zeros(shape=(LAT,LON))
    max_ts = 0
    
    curr_gp = np.zeros(shape=(2, MEM), dtype = float, order = 'F')
    for curr_lon in range(LON):
        for curr_lat in range(LAT): 
            for curr_realization in range(MEM):
                
                curr_gp[0][curr_realization] = vclin[curr_realization][curr_lat][curr_lon][curr_level][0]
                curr_gp[1][curr_realization] = vclin[curr_realization][curr_lat][curr_lon][curr_level][1] 
                
                #print "lon " + str(curr_lon)
                #print "lat " + str(curr_lat)
                #print 'mem ' + str(curr_realization)
                #print "vclin values: " + str(curr_gp[0][curr_realization])
                #curr_gp[curr_realization][2] = vclin[curr_realization][curr_lat][curr_lon][curr_level][2]
                
            x = linspace(-15, +15, 1000)  
            
            u_pass = False;v_pass = False
            for idx in range(0,MEM):
                if not u_pass and curr_gp[0][idx] != 0.:
                    u_pass = True
                    gp_u_kd = stats.gaussian_kde(curr_gp[0][:])
                    
                if not v_pass and curr_gp[1][idx] != 0.:
                    v_pass = True
                    gp_v_kd = stats.gaussian_kde(curr_gp[1][:])
                          
            mu = np.mean(curr_gp[0][:],axis=0)
            sigma = np.var(curr_gp[0][:],axis=0)
            ts = 1.
            if not math.isinf(sigma) and not math.isnan(sigma):
                normed_data = (curr_gp[0][:]-mu)/sigma
                var_std_norm = np.var(normed_data,axis=0) #equals one for std normal dist with mean = 0
                ts, p_val = stats.kstest(normed_data,'norm')
                
                k2, pvalue = stats.normaltest(curr_gp[0][:], axis=0)
                zscore, pvalue_s = stats.skewtest(curr_gp[0][:], axis=0)
                vals, counts = stats.mode(curr_gp[0][:], axis=0)
                
                '''
                if u_pass and p_val == 0.0 and var_std_norm <= 1.0 :#and pvalue > 0.01 and pvalue_s > 0.01:
                        
                          
                    plt.figure()
                    plt.title( str(ts) + "_"  + str(curr_level) + "_" + str(curr_lon) + "_" + str(curr_lat) + "_u")
                    plt.hist(curr_gp[0][:],normed=1,alpha=.3,color='purple')
                    
                    plt.plot(x,gp_u_kd(x),'-',color='red')    
                    #file = "./png/" + str(ts) + "_" + str(depth) + "_"  + str(curr_level) + "_" + str(curr_lon) + "_" + str(curr_lat) + "_u" + ".png"
                    #plt.savefig(file)
                    #sendFile(file)
                    
                    plt.show()  
                ''' 
                
                
            mu2 = np.mean(curr_gp[1][:],axis=0)
            sigma2 = np.var(curr_gp[1][:],axis=0)
            ts2 = 1.
            if not math.isinf(sigma2) and not math.isnan(sigma2):
                normed_data = (curr_gp[1][:]-mu2)/sigma2
                var_std_norm = np.var(normed_data,axis=0) #equals one for std normal dist with mean = 0
                ts, p_val = stats.kstest(normed_data,'norm')
                
                k2, pvalue = stats.normaltest(curr_gp[0][:], axis=0)
                zscore, pvalue_s = stats.skewtest(curr_gp[0][:], axis=0)
                vals, counts = stats.mode(curr_gp[0][:], axis=0)
                
                '''
                if v_pass and p_val == 0.0 and var_std_norm <= 1.0 :#and pvalue > 0.01 and pvalue_s > 0.01:
                            
                    plt.figure()
                    plt.title( str(ts) + "_" + str(curr_level) + "_" + str(curr_lon) + "_" + str(curr_lat) + "_v")
                    plt.hist(curr_gp[0][:],normed=1,alpha=.3,color='purple')
                
                    
                    plt.plot(x,gp_u_kd(x),'-',color='red')    
                    #file = "./png/" + str(ts) + "_" + str(depth) + "_"  + str(curr_level) + "_" + str(curr_lon) + "_" + str(curr_lat) + "_u" + ".png"
                    #plt.savefig(file)
                    #sendFile(file)
                    
                    plt.show()  
                ''' 
                 
            avg_u_v_ts = (ts + ts2) / 2.0
            
            #print ts
            #print ts2
            #print avg_u_v_ts
            
            array_of_ts_per_gp[curr_lat][curr_lon] = avg_u_v_ts
                
            if avg_u_v_ts > max_ts:
                max_ts = avg_u_v_ts
                
    return array_of_ts_per_gp, max_ts
                        

#from http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    
    # 

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)
            
#http://www.unidata.ucar.edu/software/netcdf/examples/programs/  
def writeNetCDF(array):
    # the output array to write will be nx x ny
    nx = LAT; ny = LON
    # open a new netCDF file for writing.
    ncfile = netCDF4.Dataset('ks_test_level_0.nc','w') 
    # create the output data.
    data_out = array#arange(nx*ny) # 1d array
    data_out.shape = array.shape # reshape to 2d array
    # create the x and y dimensions.
    ncfile.createDimension('lat',nx)
    ncfile.createDimension('lon',ny)
    # create the variable (4 byte integer in this case)
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.
    data = ncfile.createVariable('ks_test_stat',np.dtype('float64').char,('lat','lon'))
    # write data to variable.
    data[:] = data_out
    # close the file.
    ncfile.close()

#http://www.unidata.ucar.edu/software/netcdf/examples/programs/  
def writeUVelocityNetCDF(array):
    # the output array to write will be nx x ny
    nx = LAT; ny = LON
    # open a new netCDF file for writing.
    ncfile = netCDF4.Dataset('vclin_level_0.nc','w') 
    # create the output data.
    data_out = array#arange(nx*ny) # 1d array
    data_out.shape = array.shape # reshape to 2d array
    # create the x and y dimensions.
    ncfile.createDimension('lat',nx)
    ncfile.createDimension('lon',ny)
    # create the variable (4 byte integer in this case)
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.
    data = ncfile.createVariable('u',np.dtype('float64').char,('lat','lon'))
    #data2 = ncfile.createVariable('v',np.dtype('float64').char,('lat','lon'))
    # write data to variable.
    data[:] = data_out
    # close the file.
    ncfile.close()
  
def readVelNetCDF(file):
    # open a the netCDF file for reading.
    ncfile = netCDF4.Dataset(file,'r') 
    # read the data in variable named 'data'.
    data = ncfile.variables['ks_test_stat'][:]
    #nx,ny = data.shape
    # check the data.
    #data_check = arange(nx*ny) # 1d array
    #data_check.shape = (nx,ny) # reshape to 2d array

    # close the file.
    ncfile.close()
    
    return data  

'''
def writeVelocityToCSBinary(data,filename):
    writer = csv.writer(open(filename + ".csv", 'w'))
    
    #writes velocities with central forecast...
    for curr_level in range(LEV):
        for curr_lon in range(LON):
            for curr_lat in range(LAT): 
                for curr_realization in range(MEM):
                    writer.writerow(data[curr_realization][curr_lat][curr_lon][curr_level][0])
                    writer.writerow(data[curr_realization][curr_lat][curr_lon][curr_level][1]) 
   
def readVelocityToCSBinary(filename):
    reader = csv.reader(open(filename + ".csv", 'w'))
    
    #writes velocities with central forecast...
    for curr_level in range(LEV):
        for curr_lon in range(LON):
            for curr_lat in range(LAT): 
                for curr_realization in range(MEM):
                    writer.writerow(data[curr_realization][curr_lat][curr_lon][curr_level][0])
                    writer.writerow(data[curr_realization][curr_lat][curr_lon][curr_level][1]) 
'''

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
    reader = csv.reader(open(filename + ".csv", 'r'), delimiter=',')
    idx = 0
    for row in reader:
        #print row
        data[idx] = [float(i) for i in row]
        idx += 1
    
def readCellCounts(data,filename):
    #read cell counts
    filename = OUTPUT_DATA_DIR + filename
    reader = csv.reader(open(filename + ".csv", 'r'), delimiter=',')
    
    lat = 0;lon = 0
    for row in reader:
        print g_cc[lat][lon]
        g_cc[lat][lon] = len(row)-1
        if lon < LON - 1:
            lon += 1
        else:
            lon = 0
            lat += 1
        if lat >= LAT:
            break
 
def writeParticles(dir = 'f'):
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR)
    mode_dir = ''
    if MODE == 1:
        mode_dir = MODE_DIR1
    elif MODE == 2:
        mode_dir = MODE_DIR2
    
    if dir == 'f':
        writeStreamlinePositions(g_part_positions_ensemble,mode_dir+'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
        writeStreamlinePositions(g_part_positions_quantile,mode_dir+'q_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(g_part_positions_gmm,mode_dir+'gmm_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(g_part_positions_g,mode_dir+'g_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
    elif dir == 'b':
        writeStreamlinePositions(g_part_positions_ensemble_b,mode_dir+'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
        writeStreamlinePositions(g_part_positions_quantile_b,mode_dir+'q_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(g_part_positions_gmm_b,mode_dir+'gmm_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(g_part_positions_g_b,mode_dir+'g_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
    else:
        # forward and backward streamlines
        # concatenate particle positions
        e = [];q = [];gmm = [];g = []
    
        #for each component
        for idx in range(0,len(g_part_positions_ensemble_b)):
            g_part_positions_ensemble_b[idx].reverse()
            g_part_positions_quantile_b[idx].reverse()
            g_part_positions_gmm_b[idx].reverse()
            g_part_positions_g_b[idx].reverse()
            e.append(g_part_positions_ensemble_b[idx] + g_part_positions_ensemble[idx])
            q.append(g_part_positions_quantile_b[idx] + g_part_positions_quantile[idx])
            gmm.append(g_part_positions_gmm_b[idx] + g_part_positions_gmm[idx])
            g.append(g_part_positions_g_b[idx] + g_part_positions_g[idx])
        
        writeStreamlinePositions(e,mode_dir+'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
        writeStreamlinePositions(q,mode_dir+'q_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(gmm,mode_dir+'gmm_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        writeStreamlinePositions(g,mode_dir+'g_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
        
def readParticles():
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR)
    
    mode_dir = ''
    if MODE == 1:
        mode_dir = MODE_DIR1
    elif MODE == 2:
        mode_dir = MODE_DIR2
    '''
    readStreamlinePositions(g_part_positions_ensemble,mode_dir+'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
    readStreamlinePositions(g_part_positions_quantile,mode_dir+'q_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
    readStreamlinePositions(g_part_positions_gmm,mode_dir+'gmm_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)
    readStreamlinePositions(g_part_positions_g,mode_dir+'g_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values)  
    '''
    
    #read crisp sphaghetti plots
    
    #crisp_lat45.0_lon26.0_lev0_mem277_ss0.01_ts100_dir_a.csv
    for idx in range(0,MEM):
        curr_member_sl = [[],[],[]]
        readStreamlinePositions(curr_member_sl,'crisp/crisp_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+'_mem'+str(idx)+str_integration_values) 
        g_crisp_streamlines.append(curr_member_sl)
    
    #read cell counts
    filename = 'crisp/cellcounts_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values
    readCellCounts(g_cc, filename)
        
#def getCmdLineArgs():
#    for arg in sys.argv[1:]:
#        print arg           
  
def readNetCDF(file):
    # open a the netCDF file for reading.
    ncfile = netCDF4.Dataset(file,'r') 
    # read the data in variable named 'data'.
    data = ncfile.variables['ks_test_stat'][:]
    #nx,ny = data.shape
    # check the data.
    #data_check = arange(nx*ny) # 1d array
    #data_check.shape = (nx,ny) # reshape to 2d array

    # close the file.
    ncfile.close()
    
    return data
                
if __name__ == "__main__":  
    
    SEED_LAT = float(sys.argv[2])
    SEED_LON = float(sys.argv[3])
    SEED_LEVEL = int(sys.argv[4])
    integration_step_size = float(sys.argv[5])
    TOTAL_STEPS = int(sys.argv[6])
    INTEGRATION_DIR = str(sys.argv[7]).lower()
    MODE = int(sys.argv[8])
    level = SEED_LEVEL
    
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
    
    #vclin = readVelocityFromCSV(filename) 
    
    #writeVelocityToCSVBinary(vclin)
    
    ts_per_gp = readNetCDF(INPUT_DATA_DIR +'ks_test_level_0.nc')
    #vclin = readNetCDF('vclin_level_0.nc')
    
    gen_streamlines = 'True'
    gen_streamlines = sys.argv[1]
    
    if gen_streamlines == 'True':
        r.library('mixtools')
        print "generating streamlines"
        particle = 0
        
        part_pos_q[particle][0] = SEED_LAT; part_pos_q[particle][1] = SEED_LON
        part_pos_gmm[particle][0] = SEED_LAT; part_pos_gmm[particle][1] = SEED_LON
        part_pos_g[particle][0] = SEED_LAT; part_pos_g[particle][1] = SEED_LON
        part_pos_e[particle][0] = SEED_LAT; part_pos_e[particle][1] = SEED_LON
        
        part_pos_q_b[particle][0] = SEED_LAT; part_pos_q_b[particle][1] = SEED_LON
        part_pos_gmm_b[particle][0] = SEED_LAT; part_pos_gmm_b[particle][1] = SEED_LON
        part_pos_g_b[particle][0] = SEED_LAT; part_pos_g_b[particle][1] = SEED_LON
        part_pos_e_b[particle][0] = SEED_LAT; part_pos_e_b[particle][1] = SEED_LON
        
        ppos = [ part_pos_q[particle][0], part_pos_q[particle][1] ]
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v = interpFromQuantiles(ppos)
         
        #find highest prob vel
        velx_hp = e_prev_max_vel_x
        vely_hp = e_prev_max_vel_y
        
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        num_x_peaks = len(velx_prob)
        num_y_peaks = len(vely_prob)
        
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
            
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if num_y_peaks > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
            
        '''
         
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                velx_hp = velx[p] 
            
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
        
        q_prev_max_vel_x = velx_hp
        q_prev_max_vel_y = vely_hp
        
        g_part_positions_quantile[0].append(SEED_LAT)
        g_part_positions_quantile[1].append(SEED_LON) 
        g_part_positions_quantile[2].append(DEPTH) 
        g_part_positions_quantile[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
        g_part_positions_quantile[4].append((var_u + var_v) / 2.0)
        g_part_positions_quantile[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
        g_part_positions_quantile[6].append(max_peak_diff)
        
        #get peaks for ensemble
        #velx, vely, velz = interpVel(ppos)
        ppos = [ part_pos_e[particle][0], part_pos_e[particle][1] ]
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v = interpVelFromEnsemble(ppos)
         
        #find highest prob vel
        velx_hp = e_prev_max_vel_x
        vely_hp = e_prev_max_vel_y
        
        '''
        if len(velx_prob) > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if len(vely_prob) > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
        
        '''
        
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        num_x_peaks = len(velx_prob)
        num_y_peaks = len(vely_prob)
        
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
        
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if num_y_peaks > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
        
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                 
                velx_hp = velx[p] 
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
        
        e_prev_max_vel_x = velx_hp
        e_prev_max_vel_y = vely_hp
        
        g_part_positions_ensemble[0].append(SEED_LAT)
        g_part_positions_ensemble[1].append(SEED_LON) 
        g_part_positions_ensemble[2].append(DEPTH) 
        g_part_positions_ensemble[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
        g_part_positions_ensemble[4].append((var_u + var_v) / 2.0)
        g_part_positions_ensemble[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
        g_part_positions_ensemble[6].append(max_peak_diff)
        
        #get peaks for gmm
        #velx, vely, velz = interpVel(ppos)
        ppos = [ part_pos_gmm[particle][0], part_pos_gmm[particle][1] ]
        velx, velx_prob, vely, vely_prob, velz, var_u, var_v, u_params, v_params = interpFromGMM(ppos)
         
        #find highest prob vel
        velx_hp = gmm_prev_max_vel_x
        vely_hp = gmm_prev_max_vel_y
        
        '''
        if len(velx_prob) > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if len(vely_prob) > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
        '''
        
        #find difference in peaks
        '''
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        num_x_peaks = len(u_params)#len(velx_prob)
        num_y_peaks = len(v_params)#len(vely_prob)
        
        # take peaks from largest g comps
    
        if len(u_params) > 0:
            temp_max = 0
            max_idx = 0
            for i in range(0,len(u_params)):
                if u_params[i] > temp_max:
                    max_idx = i
            
            
            
            #get max peak mean
            max_x_1 = u_params[max_idx][0]
            u_params[0].pop(max_idx);u_params[1].pop(max_idx);u_params[2].pop(max_idx)
            
            temp_max = 0
            max_idx = 0
            for i in range(0,len(u_params)):
                if u_params[i] > temp_max:
                    max_idx = i
            
            
            #get 2nd peak mean
            max_x_2 = u_params[max_idx][0]
        
        # take peaks from largest g comps
        if len(v_params) > 0:
            temp_max = 0
            max_idx = 0
            for i in range(0,len(v_params)):
                if v_params[i] > temp_max:
                    max_idx = i
            
            
            
            #get max peak mean
            max_y_1 = v_params[max_idx][0]
            v_params[0].pop(max_idx);v_params[1].pop(max_idx);v_params[2].pop(max_idx)
            
            temp_max = 0
            max_idx = 0
            for i in range(0,len(v_params)):
                if v_params[i] > temp_max:
                    max_idx = i
            
            
            #get 2nd peak mean
            max_y_2 = v_params[max_idx][0]
        
        
        
        x_diff = math.fabs(max_x_1 - max_x_2)
        y_diff = math.fabs(max_y_1 - max_y_2)
        max_peak_diff = max([x_diff,y_diff])
        
        if MODE == 1:
            velx_hp = max_x_1
        else: #MODE ==2:
            velx_hp = max_x_2
        
        if MODE == 1:
            vely_hp = max_y_1
        else: #MODE ==2:
            vely_hp = max_y_2
    
        '''
    
        #find difference in peaks
        max_x_1 = 0.0
        max_x_2 = 0.0
        max_y_1 = 0.0
        max_y_2 = 0.0
        x_diff = 0.0;y_diff = 0.0
        max_peak_diff = 0.0
        
        num_x_peaks = len(velx_prob)
        num_y_peaks = len(vely_prob)
        
        if num_x_peaks > 1:
            velx_prob_copy = velx_prob[:]
            velx_copy = velx[:]
            max_x_1, max_x_2, sig = getMaxPeaks(velx_prob_copy,velx_copy)
            if sig == True:
                x_diff = pm.fabs(max_x_1 - max_x_2)
       
        if num_y_peaks > 1:
            vely_prob_copy = vely_prob[:]
            vely_copy = vely[:]
            max_y_1, max_y_2, sig = getMaxPeaks(vely_prob_copy,vely_copy)
            if sig == True:
                y_diff = pm.fabs(max_y_1 - max_y_2)
            
        if x_diff > y_diff:
            max_peak_diff = x_diff
        else:
            max_peak_diff = y_diff
    
    
        '''
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            velx_hp = velx[p]
        
        if num_y_peaks > 0:
            m1 = max(vely_prob)
            p1 = vely_prob.index(m1)
            vely_hp = vely[p1]
    
        '''
    
        if num_x_peaks > 0:
            m = max(velx_prob)
            p = velx_prob.index(m)
            
            if MODE == 1 or num_x_peaks == 1:
                velx_hp = velx[p]
            elif MODE == 2 and num_x_peaks > 1:
                velx_prob.pop(p)
                m = max(velx_prob)
                p = velx_prob.index(m)
                 
                velx_hp = velx[p] 
                
        if num_y_peaks > 0:
            m = max(vely_prob)
            p = vely_prob.index(m)
            
            if MODE == 1 or num_y_peaks == 1:
                vely_hp = vely[p]
            elif MODE == 2 and num_y_peaks > 1:
                vely_prob.pop(p)
                m = max(vely_prob)
                p = vely_prob.index(m)
                 
                vely_hp = vely[p]
    
        gmm_prev_max_vel_x = velx_hp
        gmm_prev_max_vel_y = vely_hp
        
        g_part_positions_gmm[0].append(SEED_LAT)
        g_part_positions_gmm[1].append(SEED_LON) 
        g_part_positions_gmm[2].append(DEPTH) 
        g_part_positions_gmm[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
        g_part_positions_gmm[4].append((var_u + var_v) / 2.0)
        g_part_positions_gmm[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
        g_part_positions_gmm[6].append(max_peak_diff)
        
        #get peaks for gaussian
        ppos = [ part_pos_g[particle][0], part_pos_g[particle][1] ]
        params =  interpFromGaussian(ppos)
        
        velx = params[0][0]
        vely = params[1][0]
        var_u = params[0][1]
        var_v = params[1][1]
        
        g_part_positions_g[0].append(SEED_LAT)
        g_part_positions_g[1].append(SEED_LON) 
        g_part_positions_g[2].append(DEPTH) 
        g_part_positions_g[3].append(np.sqrt(np.square(velx_hp)+np.square(vely_hp)))
        g_part_positions_g[4].append((var_u + var_v) / 2.0)
        g_part_positions_g[5].append((len(velx_prob)+len(vely_prob)) / 2.0)
        g_part_positions_g[6].append(0.0)
            
        if INTEGRATION_DIR == 'f':    
            for i_step in range(1, TOTAL_STEPS):
                advectEnsemble(i_step,dir = 'f')
                advectQuantile(i_step, dir = 'f')
                advectGMM(i_step, dir = 'f')
                advectGaussian(i_step, dir = 'f')
                
                writeParticles(dir = 'f')
                
        elif INTEGRATION_DIR == 'b':
            for i_step in range(1, TOTAL_STEPS):
                advectEnsemble(i_step,dir = 'b')
                advectQuantile(i_step, dir = 'b')
                advectGMM(i_step, dir = 'b')
                advectGaussian(i_step, dir = 'b')
                
                writeParticles(dir = 'b')
        else:
            for i_step in range(1, TOTAL_STEPS ):
                advectEnsemble(i_step,dir = 'f')
                advectQuantile(i_step, dir = 'f')
                advectGMM(i_step, dir = 'f')
                advectGaussian(i_step, dir = 'f')
            for i_step in range(1, TOTAL_STEPS + 1):
                advectEnsemble(i_step,dir = 'b')
                advectQuantile(i_step, dir = 'b')
                advectGMM(i_step, dir = 'b')
                advectGaussian(i_step, dir = 'b')
                
            writeParticles(dir = 'a')
        
        print "reused vel for quantile lerp: " + str(reused_vel_quantile)    
        print "finished!"
        
        
    else:
        print "reading particles"
        readParticles()  
        plotParticles(ts_per_gp)
    
    

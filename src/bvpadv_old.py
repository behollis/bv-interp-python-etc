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

SAMPLES = 500
#vclin_x = np.ndarray(shape=(SAMPLES,10,10))
#vclin_y = np.ndarray(shape=(SAMPLES,10,10))

DEBUG = False
  
MODE = 1
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/old_test/'
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
BIFURCATED = 'n'

MAX_BIFURCATIONS = 2
PEAK_SPLIT_THRESHOLD = 1.0

class Particle():
    def __init__(self):
        self.part_positions = [[],[],[],[],[],[],[]]
    
g_part_positions_ensemble = []
#g_part_positions_ensemble.append(Particle())

g_cc = np.zeros(shape=(LAT,LON))
g_crisp_streamlines = []
#g_part_positions_ensemble = [[],[],[],[],[],[],[]]
#g_part_positions_ensemble2 = [[],[],[],[],[],[],[]]

g_part_positions_ensemble_b = [[],[],[],[],[],[],[]]

part_pos_e = [];part_pos_q = [];part_pos_gmm = [];part_pos_g = []
part_pos_e.append([0,0])
part_pos_e[0][0] = SEED_LAT
part_pos_e[0][1] = SEED_LON

part_pos_e_b = [];part_pos_q_b = [];part_pos_gmm_b = [];part_pos_g_b = []
part_pos_e_b.append([0,0])
part_pos_e_b[0][0] = SEED_LAT
part_pos_e_b[0][1] = SEED_LON

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
   
    grid_plane = mayavi.mlab.surf(grid_lat, grid_lon, grid_verts, color=(1,1,0),representation='wireframe',line_width=0.3)
    
    #bimodal distro
    mayavi.mlab.points3d(SEED_LAT,SEED_LON,0,scale_factor = 0.05,color=(0,1.0,0))
    mayavi.mlab.points3d(4,4,0,scale_factor = 0.05,color=(1.0,0,0))
    
    for part in range(0,len(g_part_positions_ensemble)):
        if len(g_part_positions_ensemble[part].part_positions[0]) < 2:
            continue
        
        x_list = g_part_positions_ensemble[part].part_positions[0][:]
        y_list = g_part_positions_ensemble[part].part_positions[1][:]
        z_list = g_part_positions_ensemble[part].part_positions[2][:]
        #max_peak_separation_list = g_part_positions_ensemble[part].part_positions[3][:]
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
        for part in range(0,len(g_part_positions_ensemble)):
            writeStreamlinePositions(g_part_positions_ensemble[part].part_positions,\
                                 str(part) + '_Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
       
    #elif dir == 'b':
    #    writeStreamlinePositions(g_part_positions_ensemble_b,\
    #                             'Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
    '''   
    else:
        # forward and backward streamlines
        # concatenate particle positions
        e = []
    
        #for each component
        for idx in range(0,len(g_part_positions_ensemble_b)):
            g_part_positions_ensemble_b[idx].reverse()
           
            e.append(g_part_positions_ensemble_b[idx] + g_part_positions_ensemble[idx])
           
        
        writeStreamlinePositions(e,'e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values) 
    ''' 
        
def readParticles():
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR) \
    
    
    global g_part_positions_ensemble, MAX_BIFURCATIONS
    
    #for line in range(0,MAX_BIFURCATIONS):
    #    g_part_positions_ensemble.append(Particle())
    
    for part in range(0,MAX_BIFURCATIONS):
        g_part_positions_ensemble.append(Particle())
        if readStreamlinePositions(g_part_positions_ensemble[part].part_positions,\
                            str(part) + '_Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+\
                            '_lev'+str(SEED_LEVEL)+str_integration_values) is False:
            #g_part_positions_ensemble.pop(part) 
            print "reading false..."
    
    '''
    if BIFURCATED == 'y': 
        readStreamlinePositions(g_part_positions_ensemble2,\
                            '2Toy_e_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+\
                            '_lev'+str(SEED_LEVEL)+str_integration_values)
    '''
  
divs = 300
div = complex(divs)
div_real = divs
start = -30
end = +30
TOL = 0.01

from skimage import data
from skimage import measure
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.exposure as skie


def findBivariatePeaks(kde):
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[start:end:div]
    y_flat = np.r_[start:end:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    z = z.reshape(div_real,div_real)
    
    #fig = plt.figure()
    #i = plt.imshow(z)#,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower')
    #plt.show()
    
    limg = np.arcsinh(z)
    limg = limg / limg.max()
    low = np.percentile(limg, 0.25)
    high = np.percentile(limg, 99.9)
    opt_img = skie.exposure.rescale_intensity(limg, in_range=(low,high))
    
    lm = morph.is_local_maximum(limg)
    x1, y1 = np.where(lm.T == True)
    v = limg[(y1, x1)]
    lim = 0.6*high
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
        peak_probs.append( kde( ( x_flat[ peaks[0][idx] ], y_flat[ peaks[1][idx] ] ) )[0] )
    
    if len(peaks[0]) > 1:
        for idx in range(0,len(peaks[0])):
            for idx2 in range(0,len(peaks[1])):
                peak_distances.append(math.sqrt(math.pow(x_flat[peaks[0][idx]]-x_flat[peaks[0][idx2]],2) \
                                       + math.pow(y_flat[peaks[1][idx]] - y_flat[peaks[1][idx2]],2)))
        max_peak_distance = max(peak_distances)  
        print " %%max peak dist%% = " + str(max_peak_distance) 
    
    return peak_vels, peak_probs, max_peak_distance
 
from bivariate_interp import interpVelFromEnsemble as interpEnsemble
#from bivariate_interp import defineVclin
from bivariate_interp import vclin_x, vclin_y



def defineVclin():
    
    #left half
    mean1 = [0,-10]
    cov1 = [[0.1,0],[0,0.1]] 
    
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
    mean2 = [0,+10]
    cov2 = [[0.1,0],[0,0.1]] 
    
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
    mean3 = [+15,0]
    cov3 = [[0.1,0],[0,0.1]] 
    mean4 = [-15,0]
    cov4 = [[0.1,0],[0,0.1]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.5*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.5*SAMPLES)).T
    
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
bifurcated = False

PEAK_SPLIT_THRESHOLD = 1.5

def advectEnsemble(step, dir = 'f'):
    
    #global bifurcated
    #global part_pos_e
    
    global g_part_positions_ensemble
    
    advection_step_num_streamlines = len(g_part_positions_ensemble)
    
    #if step % 20  == 0:
    print '**step number: ' + str(step) 
    #split_this_step = False
    for particle in range(0, advection_step_num_streamlines):
        
        # get modal velocities @ position, if more than one modal position
        # spawn a new particle for each mode idx over one
        #if dir == 'f':
        
        #find current position of current streamline particle
        cur_posx = g_part_positions_ensemble[particle].part_positions[0][-1]
        cur_posy = g_part_positions_ensemble[particle].part_positions[1][-1]
        
        ppos = [ cur_posx , cur_posy ]
        
        #stop advection of streamline if close to grid boundary
        if cur_posx > LAT - 1.0 or cur_posy > LON - 1.0 or cur_posx < 1.0 or cur_posy < 1.0:
            #print particle
            #print ppos
            #print g_part_positions_ensemble[particle].part_positions
            continue
        
        #else:
        #ppos = [ part_pos_e_b[particle][0], part_pos_e_b[particle][1] ]
            
        #get peaks
        kde = interpEnsemble(ppos)
        peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(kde)
        
        m = max(peak_probs)
        p = peak_probs.index(m)
        vel_hp = peak_vels[p]
        vel_hp2 = (0,0)
        num_peaks = len(peak_vels)
        if  num_peaks > 1:
            #m = max(peak_probs)
            #p = peak_probs.index(m)
            #vel_hp = peak_vels[p]
            peak_vels.pop(p) 
            peak_probs.pop(p)
            #find second highest peak
            m2 = max(peak_probs)
            p2 = peak_probs.index(m2)
            vel_hp2 = peak_vels[p2]
            
        #advect current particle along main peak
        new_x = ppos[0] + vel_hp[0]*integration_step_size
        new_y = ppos[1] + vel_hp[1]*integration_step_size
        g_part_positions_ensemble[particle].part_positions[0].append(new_x)
        g_part_positions_ensemble[particle].part_positions[1].append(new_y) 
        g_part_positions_ensemble[particle].part_positions[2].append(DEPTH) 
        #g_part_positions_ensemble[particle].part_positions[3].append(max_peak_distance)
        #g_part_positions_ensemble[4].append(getSpeed(vel_hp))
        
        #find difference in peak prob and other peak probs
        '''
        max_peak_prob_diff = 0.0
        for idx in range(0,len(peak_probs)):
            diff = np.abs( p - peak_probs.index(idx) )
            if diff > max_peak_prob_diff:
                max_peak_prob_diff = diff 
        ''' 
        
        print 'streamline#: ' + str(particle)
        print '    max peak velocity: ' + str(vel_hp)
        print '    current position: ' + str(ppos)
        print '    peak velocities: ' + str(peak_vels)
        
        #spawn new particle if necessary
        if num_peaks > 1 and np.abs(max_peak_distance) > PEAK_SPLIT_THRESHOLD \
            and advection_step_num_streamlines < MAX_BIFURCATIONS: 
            #copy current positions for newly spawned particle branch point, same as parent current position
            g_part_positions_ensemble.append(Particle())
            g_part_positions_ensemble[-1].part_positions[0].append(ppos[0])
            g_part_positions_ensemble[-1].part_positions[1].append(ppos[1])  
            g_part_positions_ensemble[-1].part_positions[2].append(DEPTH) 
            #g_part_positions_ensemble[-1].part_positions[3].append(max_peak_distance)
            
            #part_pos_e[particle+1][0] += vel_hp2[0]*integration_step_size
            #part_pos_e[particle+1][1] += vel_hp2[1]*integration_step_size
            #split_this_step = True
            #bifurcated = True
            
            #advect newly spawned particle for current advection step
            branched_x = ppos[0] + vel_hp2[0]*integration_step_size
            branched_y = ppos[1] + vel_hp2[1]*integration_step_size
            g_part_positions_ensemble[-1].part_positions[0].append( branched_x )
            g_part_positions_ensemble[-1].part_positions[1].append( branched_y )
            g_part_positions_ensemble[-1].part_positions[2].append( DEPTH )
            
def getSpeed(vector):
    return np.abs(np.dot(np.asarray(vector), np.asarray(vector)))  
        
def main():
    global SEED_LAT,SEED_LON,SEED_LEVEL,TOTAL_STEPS,INTEGRATION_DIR,\
        integration_step_size,gen_streamlines, BIFURCATED, MAX_BIFURCATIONS
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
    #MODE = int(sys.argv[8])
    #level = SEED_LEVEL
    
    #vclin = np.zeros(shape=(10,10,2))
    
    
    
    if gen_streamlines == 'True':
        defineVclin()
        
        print "generating streamlines"
        '''
        particle = 0
        
        part_pos_e[particle][0] = SEED_LAT; part_pos_e[particle][1] = SEED_LON
        #part_pos_e_b[particle][0] = SEED_LAT; part_pos_e_b[particle][1] = SEED_LON
        
        ppos = [ part_pos_e[particle][0], part_pos_e[particle][1] ]
        '''
        
        kde = interpEnsemble([SEED_LAT, SEED_LON])
        peak_vels, peak_probs, max_peak_distance = findBivariatePeaks(kde)
        m = max(peak_probs)
        p = peak_probs.index(m)
        vel_hp = peak_vels[p]
    
        #print peak_vels
        #print peak_probs
        #print max_peak_distance
        
        #find two or less peaks in distro defined by kde
         
        g_part_positions_ensemble.append(Particle())
        g_part_positions_ensemble[0].part_positions[0].append(SEED_LAT)
        g_part_positions_ensemble[0].part_positions[1].append(SEED_LON) 
        g_part_positions_ensemble[0].part_positions[2].append(DEPTH) 
        #g_part_positions_ensemble[0].part_positions[3].append(max_peak_distance)
        #g_part_positions_ensemble[0].part_positions[4].append(getSpeed(vel_hp))
        #g_part_positions_ensemble[0].part_positions[5].append(0)
        #g_part_positions_ensemble[0].part_positions[6].append(0)
        
        
        #if INTEGRATION_DIR == 'f':  
        
        for i_step in range(1, TOTAL_STEPS):
            advectEnsemble(i_step,dir = 'f')
        
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
        plotParticles()
            
        
if __name__ == "__main__":  
    main() 
    

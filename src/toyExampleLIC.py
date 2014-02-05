#!/usr/bin/python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 5/20/2013.
    Code saves off an image of the LIC of the mean vector field of paper toy example.
'''

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
from pylab import *

import sum_of_gaussians_interpolation as sog
from netcdf_reader import *
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

integration_step_size = 0.1
SEED_LAT = 42
SEED_LON = 21
SEED_LEVEL = 0
vclin = []
cf_vclin = []

SAMPLES = 1000
size = 600
vclin_x = np.ndarray(shape=(size,size,SAMPLES))
vclin_y = np.ndarray(shape=(size,size,SAMPLES))

nx, ny = (size,size)
x = np.linspace(0, size, nx)
y = np.linspace(0, size, ny)
xv, yv = np.meshgrid(x, y)

#X, Y = np.mgrid[0:9:10j, 0:9:10j]

INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/toy5/'
DEPTH = -2.0 
INTEGRATION_DIR = 'b'
THRESHOLD_PER = 0.9 #percentage that second greatest peak needs to be of the max peak
BIFURCATED = 'n'
INTERP_METHOD = 'e'
DEPTH = -2.0
INTEGRATION_DIR = 'b'

g_grid_mean_array_u = np.ndarray(shape=(size,size))
g_grid_mean_array_v = np.ndarray(shape=(size,size))

import lic_internal

texture = np.random.rand(size,size).astype(np.float32)

plt.bone()
frame=0

vectors = np.zeros((size,size,2),dtype=np.float32)
#for (x,y) in vortices:
#    rsq = (xs-x)**2+(ys-y)**2
#    vectors[...,0] +=  (ys-y)/rsq
#    vectors[...,1] += -(xs-x)/rsq

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
    mean3 = [+10,0]
    cov3 = [[1,0],[0,1]] 
    mean4 = [-10,0]
    cov4 = [[1.5,0],[0,1.5]] 
    
    x3,y3 = np.random.multivariate_normal(mean3,cov3,int(0.55*SAMPLES)).T
    x4,y4 = np.random.multivariate_normal(mean4,cov4,int(0.45*SAMPLES)).T
    
    x_tot = np.append(x3,x4)
    y_tot = np.append(y3,y4)
    
    #define 10x10 gp's samples
    for x in range(0,size):
        for y in range(0,size):
            for idx in range(0,SAMPLES):
                if x <= int(size / 4.): 
                    vclin_x[y][x][idx] = x1[idx] 
                    vclin_y[y][x][idx] = y1[idx]  
                elif x > int(size/4.):
                    vclin_x[y][x][idx] = x2[idx]  
                    vclin_y[y][x][idx] = y2[idx]
                elif x > int(size/4.) and x < int(size/4.):
                    vclin_x[y][x][idx] = x_tot[idx] 
                    vclin_y[y][x][idx] = y_tot[idx]  
                    
                
    #define bimodal bivariate grid point
    '''
    for idx in range(0,SAMPLES):
        vclin_x[int(size / 2.)][int(size / 2.)][idx] = x_tot[idx]
        vclin_y[int(size / 2.)][int(size / 2.)][idx] = y_tot[idx]   
    '''     
        
        
if __name__ == '__main__':#def main():
    defineVclin()
    
    '''
    xs = np.linspace(-1,1,size).astype(np.float32)[None,:]
    ys = np.linspace(-1,1,size).astype(np.float32)[:,None]

    
    vortex_spacing = 0.5
    extra_factor = 2.

    a = np.array([1,0])*vortex_spacing
    b = np.array([np.cos(np.pi/3),np.sin(np.pi/3)])*vortex_spacing
    rnv = int(2*extra_factor/vortex_spacing)
    vortices = [n*a+m*b for n in range(-rnv,rnv) for m in range(-rnv,rnv)]
    vortices = [(x,y) for (x,y) in vortices if -extra_factor<x<extra_factor and -extra_factor<y<extra_factor]
    
    for (x,y) in vortices:
        rsq = (xs-x)**2+(ys-y)**2
        vectors[...,0] +=  (ys-y)/rsq
        vectors[...,1] += -(xs-x)/rsq
    '''
    
    for x in range(0,size):
        for y in range(0,size):
            #mean_vector = (np.mean(vclin_y[x][y][:]), np.mean(vclin_x[x][y][:]))
            #g_grid_mean_array_u[x][y] = np.mean(vclin_x[x][y])
            #g_grid_mean_array_v[x][y] = np.mean(vclin_y[x][y])
            vectors[x][y] = ( np.mean(vclin_x[x][y]), np.mean(vclin_y[x][y]) )
            #print g_grid_mean_array_u[x][y]
            
    u = g_grid_mean_array_u#.flatten()
    v = g_grid_mean_array_v#.flatten()
    #fig = plt.figure()
    '''
    plt.streamplot(xv, yv, u, v, density=5, linewidth=None, \
               color='black', cmap=None, norm=None, arrowsize=1, arrowstyle='-|>', transform=None)
    #ax = fig.add_subplot(1,1,1)
    #ax.yaxis.grid(color=None)#, linestyle='dashed')
    #ax.xaxis.grid(color=None)#, linestyle='dashed')
    plt.show()
    '''
    
    kernellen=31
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)

    print 'going into lic_internal'
    image = lic_internal.line_integral_convolution(vectors, texture, kernel)
    
    #plt.imshow(image)
    #plt.show()

    dpi = 100
    plt.clf()
    plt.axis('off')
    plt.figimage(image)
    plt.gcf().set_size_inches((size/float(dpi),size/float(dpi)))
    plt.show()
            
#if __name__ == '__main__':
#    main()
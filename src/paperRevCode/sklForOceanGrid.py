#!/usr/bin/python

PROJ_PATH = '/home/behollis/thesis_code/code/interpRevProj/src/'
import sys
sys.path.append(PROJ_PATH)
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
import sum_of_gaussians_interpolation as sog
from netcdf_reader import *
from mayavi.mlab import *
import mayavi
from peakfinder import *
from quantile_lerp import *
import os
import datetime 
import time
from sklTestBivariateInterp import *

OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/pics/bv_interp/'

def main():
    
    #cpu time
    qstart = time.clock() 
    
    loadNetCdfData()
    remapGridData()
    
    createGlobalKDEArray(LAT,LON)
    createGlobalQuantileArray(LAT,LON)
    
    startlat = 0; endlat = LAT
    startlon = 0; endlon = LON
    
    oslat = startlat
    oslon = startlon
    
    BLOCK_SIZE = 1
    
    #5x5 blocks
    sklgrid = np.zeros(shape=(BLOCK_SIZE,BLOCK_SIZE))
    
    gclock_total = time.clock()
    gclock = time.clock()
    
    for oslat in range(18, LAT-BLOCK_SIZE, BLOCK_SIZE):
        for oslon in range(0, LON-BLOCK_SIZE, BLOCK_SIZE):
            #calculate skl for block 
            for ilat in range(oslat,BLOCK_SIZE+oslat,1):
                for ilon in range(oslon,BLOCK_SIZE+oslon,1):
                    
                    qcurr = time.clock()
                    print 'time elapsed: ' + str(qcurr - qstart)
                   
                    #find KDE benchmark
                    distro = getVclinSamplesSingle([ilat,ilon])
                    
                    skl = 0
                    kde = None
                    
                    try:
                        kde = stats.kde.gaussian_kde(distro)
                         
                        x_min = np.asarray(distro[0]).min()
                        x_max = np.asarray(distro[0]).max()
                        y_min = np.asarray(distro[1]).min()
                        y_max = np.asarray(distro[1]).max()
                        
                        mfunc1 = getKDE((x_min,x_max), (y_min,y_max),kde)
                        
                        if ilat % 2 == 0. and ilon % 2 == 0.:
                            #find quantile approx (include surface interpolant choice)
                            
                            samples_arr_a, evalfunc_a = interpFromQuantiles3(ppos=[ilat,ilon], \
                                                                             ignore_cache = 'False', half=False)
                            
                            if evalfunc_a == None:
                                print 'lat: ' + str(ilat)
                                print 'lon: ' + str(ilon)
                                print 'interpFromQuantile Failed...writing zero.'
                                continue
                            
                            distro2_a, interpType_a, success = computeDistroFunction(evalfunc_a[0],evalfunc_a[1], \
                                                                                     evalfunc_a[2], \
                                                                                     (x_min,x_max), (y_min,y_max))
                            
                            if not success:
                                print 'lat: ' + str(ilat)
                                print 'lon: ' + str(ilon)
                                print 'computeDistro Failed...writing zero.'
                                continue
                            
                            skl4f_a = kl_div_2D_M(mfunc1=distro2_a, mfunc2=mfunc1, min_x=x_min, max_x=x_max, \
                                                  min_y=y_min, max_y=y_max)
                            skl4b_a = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2_a, min_x=x_min, max_x=x_max, \
                                                  min_y=y_min, max_y=y_max)
                            skl = skl4f_a + skl4b_a
                            
                            title1 = dt + str(oslat) + '_' + str(oslon) + '_kde_' 
                            title4 = dt + str(oslat) + '_' + str(oslon) + '_q_skl_NOTINTERPOLATED_'   + str(skl) 
                            plotKDE(kde,distro, title1, co = green)
                            plotXYZSurf((x_min,x_max), (y_min,y_max), distro2_a, title4, samples_arr_a, col=blue)
                            
                        else:
                            
                            samples_arr, evalfunc = interpFromQuantiles3(ppos=[ilat,ilon], \
                                                                         ignore_cache = 'True', half=True)
                            
                            if evalfunc == None:
                                print 'lat: ' + str(ilat)
                                print 'lon: ' + str(ilon)
                                print 'interpFromQuantile Failed...writing zero.'
                                continue
                            
                            distro2, interpType, success = computeDistroFunction(evalfunc[0],evalfunc[1],evalfunc[2], \
                                                                        (x_min,x_max), (y_min,y_max))
                            
                            if not success:
                                print 'lat: ' + str(ilat)
                                print 'lon: ' + str(ilon)
                                print 'computeDistro Failed...writing zero.'
                                continue
                            
                            skl4f = kl_div_2D_M(mfunc1=distro2, mfunc2=mfunc1, min_x=x_min, max_x=x_max, \
                                                min_y=y_min, max_y=y_max)
                            skl4b = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2, min_x=x_min, max_x=x_max, \
                                                min_y=y_min, max_y=y_max)
                            skl = skl4f + skl4b
                            
                            title1 = dt + str(oslat) + '_' + str(oslon) + '_kde_' 
                            title4 = dt + str(oslat) + '_' + str(oslon) + '_q_skl_interpolated_'   + str(skl) 
                            plotKDE(kde,distro, title1, co = green)
                            plotXYZSurf((x_min,x_max), (y_min,y_max), distro2, title4, samples_arr, col=blue)
                            
                    except:
                        print 'lat: ' + str(ilat)
                        print 'lon: ' + str(ilon)
                        print 'Gathering samples for grid point failed...writing zero.'
                        
                        
                    print 'lat: ' + str(ilat)
                    print 'lon: ' + str(ilon)
                    print '***skl: ' + str(skl)
                    
                    sklgrid[ilat-oslat,ilon-oslon] = skl
             
            ''' Finished calculating skl for current block. Now write to disk. '''        
            #print 'lat: ' + str(ilat)
            #print 'lon: ' + str(ilon)
            #print 'skl: ' + str(skl)
                    
            #cpu time
            qend = time.clock()
            qtot = qend - gclock
            
            #write current block to disk
            print 'writing current block'
            np.save(OUTPUT_DATA_DIR + str(oslat) + '_' + str(oslon) + '_blocksize_' + str(BLOCK_SIZE) + \
                    '_cputime_' + str(qtot), sklgrid)
            
            #reset 
            gclock = time.clock()
            sklgrid = np.zeros(shape=(BLOCK_SIZE,BLOCK_SIZE))
    
    #cpu time
    qend = time.clock()
    qtot = qend - gclock_total
    print 'COMPLETE!'
    print 'TOTAL TIME: ' + str(qtot)
            
if __name__ == "__main__":  
    main()

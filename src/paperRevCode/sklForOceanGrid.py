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



def main():
    
    #cpu time
    qstart = time.clock() 
    
    loadNetCdfData()
    remapGridData()
    
    createGlobalKDEArray(LAT,LON)
    createGlobalQuantileArray(LAT,LON)
    
    startlat = 0; endlat = LAT
    startlon = 0; endlon = LON
    
    sklgrid = np.zeros(shape=(endlat-startlat+1,endlon-startlon+1))
      
    for ilat in range(startlat,endlat,1):
        for ilon in range(startlon,endlon,1):
        
            #find KDE benchmark
            distro = getVclinSamplesSingle([ilat,ilon])
            
            skl = 0.
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
                    
                    samples_arr_a, evalfunc_a = interpFromQuantiles3(ppos=[ilat,ilon], ignore_cache = 'True', half=False)
                    
                    if evalfunc_a == None:
                        continue
                    
                    distro2_a, interpType_a, success = computeDistroFunction(evalfunc_a[0],evalfunc_a[1], \
                                                                             evalfunc_a[2], (x_min,x_max), (y_min,y_max))
                    
                    if not success:
                        continue
                    
                    skl4f_a = kl_div_2D_M(mfunc1=distro2_a, mfunc2=mfunc1, min_x=x_min, max_x=x_max, \
                                          min_y=y_min, max_y=y_max)
                    skl4b_a = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2_a, min_x=x_min, max_x=x_max, \
                                          min_y=y_min, max_y=y_max)
                    skl = skl4f_a + skl4b_a
                else:
                    
                    samples_arr, evalfunc = interpFromQuantiles3(ppos=[ilat,ilon], ignore_cache = 'True', half=True)
                    
                    if evalfunc_a == None:
                        continue
                    
                    distro2, interpType = computeDistroFunction(evalfunc[0],evalfunc[1],evalfunc[2], \
                                                                (x_min,x_max), (y_min,y_max))
                    
                    if not success:
                        continue
                    
                    skl4f = kl_div_2D_M(mfunc1=distro2, mfunc2=mfunc1, min_x=x_min, max_x=x_max, \
                                        min_y=y_min, max_y=y_max)
                    skl4b = kl_div_2D_M(mfunc1=mfunc1, mfunc2=distro2, min_x=x_min, max_x=x_max, \
                                        min_y=y_min, max_y=y_max)
                    skl = skl4f + skl4b
                    
            except:
                print 'lat: ' + str(ilat)
                print 'lon: ' + str(ilon)
                print 'EXCEPTION!'
                
            sklgrid[ilat-startlat,ilon-startlon] = skl
            print 'lat: ' + str(ilat)
            print 'lon: ' + str(ilon)
            print 'skl: ' + str(skl)
                    
    #cpu time
    qend = time.clock()
    qtot = qend - qstart
    
    fig = plt.figure()
    #plt.imshow(sklgrid)
    plt.show()
    plt.savefig(OUTPUT_DATA_DIR + str(startlat) + '_' + str(endlat) + '_skl_cputime_' + str(qtot) + '.jpg')
            
if __name__ == "__main__":  
    main()

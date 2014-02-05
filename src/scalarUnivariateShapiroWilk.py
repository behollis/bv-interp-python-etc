
#finds peaks in bivariate distributions...

#use import numpypy as np #if using pypy interpreter
import numpy as np

import sys, struct
import rpy2.robjects as robjects
import random
import math as pm
import math

import matplotlib.mlab as mlab
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import operator
import gaussian_fit as gf
from scipy.optimize import leastsq
from scipy.stats import *
from peakfinder import *
#from rpy2.robjects.vectors import FloatVector
#from rpy2.robjects.vectors import StrVector
from netcdf_reader import *

from scipy import stats
from scipy import linalg
from scipy import mat
from scipy import misc
from scipy import interpolate
import time


from numpy import linspace,exp

from rpy2.robjects import numpy2ri
numpy2ri.activate()

from netcdf_reader import NetcdfReader

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

 
from skimage import data
from skimage import measure
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.exposure as skie
#import pyfits

INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/bv_peaks/'


#import numpy as np
#import matplotlib.pyplot as plt

r = robjects.r

#http://stackoverflow.com/questions/13317141/error-trying-to-use-a-r-library-with-rpy2
#from rpy import * #need for mvnormtest library

#select between single gaussian fit and GMM model
ASSUME_NORM = False

INPUT_DATA_DIR = '../../data/in/ncdf/'
#OUTPUT_DATA_DIR = '../../data/out/pics/bv_peaks2/'
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'

from fractions import Fraction
import sys, socket, os

import code, traceback, signal
#import rpy2.robjects.numpy2ri

from rpy2.robjects.packages import importr
mvnormtest = importr('mvnormtest')
mixtools = importr('mixtools')
rstats = importr('stats')

def plotDistro(kde,x_min_max,y_min_max,title=''):
    div = 400j
    div_real = 400
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[x_min_max[0]:x_min_max[1]:div]
    y_flat = np.r_[y_min_max[0]:y_min_max[1]:div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    #positions = np.vstack([x_flat.ravel(), y_flat.ravel()])
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    z = z.reshape(div_real,div_real)
    
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=5, cstride=5, alpha=1.0, linewidth=0.1, antialiased=True)
    #fig.colorbar(surf)
    plt.show()    
    
    #cax = ax
    plt.imshow(z, extent=[x_min_max[1], x_min_max[0], y_min_max[1], y_min_max[0]])
    #cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
    #cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
  
    #ax.set_xlabel('u')
    #ax.set_ylabel('v')
    
    #cbar = fig.colorbar(cax, ticks=[0, z.max()], orientation = 'horizontal')
    plt.colorbar()
    plt.show()    
    
    #ax.set_xlim(x_flat.min(), x_flat.max())
    #ax.set_ylim(y_flat.min(), y_flat.max())
    
    '''    
    #ax.set_zlabel('kde')
    ax.set_zlim(0, z.max())
        
    ax.set_zlabel('density')
    #ax.set_zlim(0, np.asarray(z_pos).max())
   
    #cb = fig.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label('kde')
    '''
          
    #plt.savefig(OUTPUT_DATA_DIR + str(title) + ".png")
    #plt.show()    
    
def fitBvGmm(gp, max_gs = 2):
    matrix = robjects.conversion.py2ri(gp)
    
    mixmdl = r.mvnormalmixEM(matrix, k = max_gs, maxit = 5, verb = True)
    
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

def findHistogramPeaks(gp1_2d_array,xmin,xman,ymin,ymax):
    divs_per_unit_ratio = 20
    bins = [ int(np.abs(xmin-xmax) * divs_per_unit_ratio), int(np.abs(ymin-ymax) * divs_per_unit_ratio) ]            
    z, xedges, yedges = np.histogram2d(gp1_2d_array.T[1][:], gp1_2d_array.T[0][:], bins=bins)#, range, normed, weights)
    

    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    #>>> import matplotlib.pyplot as plt
    #plt.imshow(H, extent=extent, interpolation='nearest')
  
    #plt.colorbar()
    #plt.show()
    
    m = z.max()
    p = z.index(m)
    peak_one = z[p[0],p[1]] 
    
    return 

def findBvPeaks(gp1_2d_array):
    #bandwidth = 'scott'
    kde = stats.kde.gaussian_kde( ( gp1_2d_array.T[0][:], gp1_2d_array.T[1][:] ))#, bw_method = bandwidth )
    
    div = 128j
    
    # Regular grid to evaluate kde upon
    
    x_flat = np.r_[gp1_2d_array[:,0].min():gp1_2d_array[:,0].max():div]
    y_flat = np.r_[gp1_2d_array[:,1].min():gp1_2d_array[:,1].max():div]
    x,y = np.meshgrid(x_flat,y_flat)
    
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    z = kde(grid_coords.T)
    
    div_real = 128
    
    z = z.reshape(div_real,div_real)
    
    #plotDistro(kde,(u_min,u_max),(v_min,v_max),title='')
    
   
    
    limg = np.arcsinh(z)
    limg = limg / limg.max()
    low = np.percentile(limg, 0.25)
    high = np.percentile(limg, 99.9)
    opt_img = skie.exposure.rescale_intensity(limg, in_range=(low,high))
    
    lm = morph.is_local_maximum(limg)
    x1, y1 = np.where(lm.T == True)
    v = limg[(y1, x1)]
    lim = 0.6*high
    
    fp = 13 #smaller odd values are more sensitive to local maxima in image (use roughtly 5 thru 13)
    lm = morph.is_local_maximum(limg,footprint=np.ones((fp, fp)))
    x1, y1 = np.where(lm.T == True)
    v = limg[(y1, x1)]
    
    peaks = [[],[]]
    for idx in range(0,len(x1)):
        if limg[(y1[idx],x1[idx])] > lim: 
            peaks[0].append(x1[idx])
            peaks[1].append(y1[idx])
    
    fig = plt.figure()
    
    img = plt.imshow(opt_img,origin='lower',interpolation='bicubic')#,cmap=cm.spectral)
    ax = fig.add_subplot(111)
    ax.scatter(peaks[0],peaks[1], s=100, facecolor='none', edgecolor = '#009999')

    #plt.savefig(OUTPUT_DATA_DIR + '_peaks.png')
    plt.colorbar()
    plt.show()

    return peaks


'''
class BandwidthCallable:
    def __init__(self,k=1):
        self.k = k
    def __call__(self, dist):
        sum_dist = 0.
        for samp_i in range(dist.shape[1]):
            for samp_j in range(dist.shape[1]):
                sum_dist += math.sqrt(math.pow(samp_i[0] - samp_j[0], 2) \
                              + math.pow(samp_i[1] - samp_i[1], 2)) 
        return sum_dist / ( dist.shape[1] * self.k )
'''   

def calcBandwidth(dist, method='scott', k=1, dim=2):
    bw = 0.
    num_samples = dist.shape[0]
    
    if method == 'distance':
        for samp_i in dist:
            for samp_j in dist:
                #print bw
                bw += math.sqrt(math.pow(samp_i[0] - samp_j[0], 2) \
                             + math.pow(samp_i[1] - samp_i[1], 2)) 
        bw /= num_samples * k
    elif method == 'scott':                     
        bw = num_samples**(-1./(dim+4))
    else:#silverman's rule
        bw = ( num_samples * (dim + 2) / 4.)**(-1. / (dim + 4))
                         
    return bw  

if __name__ == '__main__':
    FILE_NAME = 'pe_dif_sep2_98.nc' 
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    read_type = str(sys.argv[1])
   
    COM =  2
    LON = 53
    LAT = 90
    LEV = 16
    MEM = 600
    
    SEED_LEVEL = 0
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
   
    '''
    landmask = rreader.readVarArray('landv')
   
    np.savetxt(OUTPUT_DATA_DIR + 'landv.txt', landmask)
    arr = np.loadtxt(OUTPUT_DATA_DIR + 'landv.txt')
    plt.figure()    
    plt.imshow(np.rot90(arr,2),cmap=cm.spectral)
    plt.show() 
    '''
    
    #deviations from central forecast for all 600 realizations
    temp = rreader.readVarArray('temp')
    
    #central forecasts reader 
    creader = NetcdfReader(pe_fct_aug25_sep2_file)
    temp8 = creader.readVarArray('temp', 7)
    
    #deviations from central forecast for all 600 realizations
    temp = rreader.readVarArray('temp')  
    temp = addCentralForecastScalar(temp, temp8, level_start=0, level_end=LEV)  
    
    
    #gp1_y = np.zeros(shape=(MEM,1))
    
    grid = np.ones(shape=(LON,LAT)) #assumes gaussianity / 1 peak
    peaks = np.ndarray(shape=(LON,LAT))
    
    bw_types = ['distance','scott','silverman']
    
    gp1_x = np.zeros(shape=(MEM,1))
    
    #lat41,lon12,level0, u comp
    for lev in range(0,1,1):
        for lat in range(0,LAT,1):
            for lon in range(0,LON,1):
                
                print 'Evaluating: ' + str(lat) + ' ' + str(lon) + ' ' + str(lev) + '... '
                
                #gp1_x = []
                for idx in range(0,MEM):
                    #gp1_x.append(temp[idx][lat][lon][lev])
                    gp1_x[idx] = temp[idx][lat][lon][lev]
                    #print gp1_x[idx]
                      
                #xmin = min(gp1_x)
                #xmax = max(gp1_x)
                xmin = gp1_x.min()
                xmax = gp1_x.max()
                print xmin
                print xmax
                
                #try:
                #w, pval = stats.shapiro( gp1_x )#mvnormtest.mshapiro_test(matrix)
                
                try:
                    result = rstats.shapiro_test(gp1_x)
                    
                    for item in result.iteritems():
                        if item[0] == 'p.value':
                            grid[lon][lat] = item[1][0]
                            print 'pvalue = ' + str(grid[lon][lat])
                    
                    #print '***pval:  ' + str(pval)
                    #grid[lon][lat] = temp[0][lat][lon][lev]#pval
                    
                    
                    if grid[lon][lat] > 0.8 or grid[lon][lat] < 0.2:
                        try:
                            kde = stats.gaussian_kde(gp1_x[:])
                            x = linspace(xmin,xmax,100)
                            plt.figure()
                            plt.plot(x,kde)
                            plt.show()
                        except:
                            print 'kde failed...'
                            pass
                except:
                    grid[lon][lat] = 0
                    
                #except:
                #    print "normal test failed..."
                #    grid[lon][lat] = 0
            
               
            
    #if read_type == 'n':    
    #    filename = 'lev0_bv_nongaussian_univariate.txt'
                   
    #np.savetxt(OUTPUT_DATA_DIR + filename,grid.T)
    #arr = np.loadtxt(OUTPUT_DATA_DIR + filename)
    
               
    plt.figure()    
    plt.imshow(np.rot90(grid,2),cmap=cm.gist_gray)
    plt.show()
    
     
    #plt.figure()
    #plt.imshow(grid)
    #plt.show()
                   
                
print "finished!!"
    

 
 

            
            
        
    

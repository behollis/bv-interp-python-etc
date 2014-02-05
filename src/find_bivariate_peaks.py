
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
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import StrVector
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
import pyfits

INPUT_DATA_DIR = '../../data/in/ncdf/'


#import numpy as np
#import matplotlib.pyplot as plt

r = robjects.r

#select between single gaussian fit and GMM model
ASSUME_NORM = False

INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/pics/bv_peaks/'
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'

from fractions import Fraction
import sys, socket, os

import code, traceback, signal



def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal recieved : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def listen():
    #http://stackoverflow.com/questions/132058/showing-the-stack-trace-from-a-running-python-application
    signal.signal(signal.SIGUSR1, debug)  # Register handler

def sendFile(file):    
    print "sending file"
    
    HOST = 'riverdance.soe.ucsc.edu'
    CPORT = 9091
    MPORT = 9090
    FILE = file#sys.argv[1]
    
    cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cs.connect((HOST, CPORT))
    cs.send("SEND " + FILE)
    cs.close()
    
    ms = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ms.connect((HOST, MPORT))
    
    f = open(OUTPUT_DATA_DIR + FILE, "rb")
    data = f.read()
    f.close()
    
    ms.send(data)
    ms.close()
    
    os.remove(OUTPUT_DATA_DIR + file)


if __name__ == '__main__':
    FILE_NAME = 'pe_dif_sep2_98.nc' 
    
    listen()
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 

    COM =  2
    LON = 53
    LAT = 90
    LEV = 16
    MEM = 600
    
    SEED_LEVEL = 1
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')
    
    #central forecasts reader 
    creader = NetcdfReader(pe_fct_aug25_sep2_file)
    vclin8 = creader.readVarArray('vclin', 7)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')  
    vclin = addCentralForecast(vclin, vclin8, level_start=0, level_end=LEV)  
    
    gp1_x = np.zeros(shape=(MEM,1))
    gp1_y = np.zeros(shape=(MEM,1))
    
    #lat41,lon12,level0, u comp
    for lev in range(0,MEM,1):
        for lat in range(0,LAT,1):
            for lon in range(0,LON,1):
                
                print 'Evaluating: ' + str(lat) + ' ' + str(lon) + ' ' + str(lev) + '... '
            
                for idx in range(0,MEM):
                    gp1_x[idx] = vclin[idx][lat][lon][lev][0]
                    gp1_y[idx] = vclin[idx][lat][lon][lev][1]
                  
                xmin = gp1_x.min()
                xmax = gp1_x.max()
                ymin = gp1_y.min()
                ymax = gp1_y.max()
            
                # for our current gp, count the contours to find peaks
                # near the main peak...
                gp1_2d_array = np.append(gp1_x,gp1_y,axis=1)
                
                try:
                    '''
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    print X
                    print Y
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    print positions
                    values = np.vstack([gp1_x, gp1_y])
                    print values
                    kernel = stats.gaussian_kde(values)
                    print kernel
                    Z = np.reshape(kernel.evaluate(positions).T, X.shape)
                    '''
                    
                    kde = stats.kde.gaussian_kde(gp1_2d_array.T)
                    
                    div = 128j
                
                    # Regular grid to evaluate kde upon
                    x_flat = np.r_[gp1_2d_array[:,0].min():gp1_2d_array[:,0].max():div]
                    y_flat = np.r_[gp1_2d_array[:,1].min():gp1_2d_array[:,1].max():div]
                    x,y = np.meshgrid(x_flat,y_flat)
                    
                    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
                    z = kde(grid_coords.T)
                    
                    div_real = 128
                    
                    z = z.reshape(div_real,div_real)
                    
                    #contours : list of (n,2)-ndarrays
                    #Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates along the contour.
                    #http://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=find_contours#skimage.measure.find_contours
                    
                    '''    
                    #imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
                    
                    LOWER_PROB_PEAK_LIMIT = 0.80*z.max()
                    
                    contours = measure.find_contours(np.rot90(z,4),LOWER_PROB_PEAK_LIMIT)
                    
                    contour_centroids = []
                    for idx in range(0,len(contours)):
                        contour_centroids.append(np.average(contours[idx], axis=0))
                                
                    peak_distances = []
                    for idx in range(0,len(contour_centroids)):
                        for idx2 in range(0,len(contour_centroids)):
                            peak_distances.append(math.sqrt(math.pow(contour_centroids[idx][0]-contour_centroids[idx2][0],2) \
                                                   + math.pow(contour_centroids[idx][1] - contour_centroids[idx2][1],2)))
                    '''
                    
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
                
                    
                    print peaks
                    
                    peak_distances = []
                    for idx in range(0,len(peaks)):
                        for idx2 in range(0,len(peaks)):
                            peak_distances.append(math.sqrt(math.pow(peaks[idx][0]-peaks[idx2][0],2) \
                                                   + math.pow(peaks[idx][1] - peaks[idx2][1],2)))
                    
                    max_peak_distance = max(peak_distances)  
                    print "max peak dist = " + str(max_peak_distance) 
                
                    #note distances are measured on a 128 unit by 128 unit grid
                    if len(peaks) >= 2:# and max_peak_distance >= div_real*0.3:
                        
                        print "lat: " + str(lat)
                        print "lon: " + str(lon)
                        print "lev: " + str(lev)
                        #print "contour count: " + str(len(contours))
                        
                        
                        fig = plt.figure()
                        
                        ax = fig.gca(projection='3d')
                        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7, linewidth=0, antialiased=True,cmap=cm.spectral)
                        cset = ax.contour(x, y, z, zdir='y', offset=y_flat.max(),antialiased=True,colors='r')
                        cset = ax.contour(x, y, z, zdir='x', offset=x_flat.min(),antialiased=True, colors='b')
                        #cset = ax.contour(X, Y, Z, zdir='y', offset=xmax,antialiased=True,colors='r')
                        #cset = ax.contour(X, Y, Z, zdir='x', offset=xmin,antialiased=True, colors='b')
                        
                        
                        ax.set_xlabel('u')
                        ax.set_ylabel('v')
                        
                        '''
                        if x_flat.min() <= y_flat.min() and x_flat.max() >= y_flat.max() :
                            ax.set_xlim(x_flat.min(), x_flat.max())
                            ax.set_ylim(x_flat.min(), x_flat.max())
                        elif x_flat.min() <= y_flat.min() and x_flat.max() < y_flat.max() :
                            ax.set_xlim(x_flat.min(), y_flat.max())
                            ax.set_ylim(x_flat.min(), y_flat.max())
                        elif x_flat.min() > y_flat.min() and x_flat.max() < y_flat.max() :
                            ax.set_xlim(y_flat.min(), y_flat.max())
                            ax.set_ylim(y_flat.min(), y_flat.max())
                        else: #x_flat.min() > y_flat.min() and x_flat.max() >= y_flat.max() 
                            ax.set_xlim(y_flat.min(), x_flat.max())
                            ax.set_ylim(y_flat.min(), x_flat.max())
                        '''
                        
                        ax.set_xlim(x_flat.min(), x_flat.max())
                        ax.set_ylim(y_flat.min(), y_flat.max())
                            
                        ax.set_zlabel('density')
                        ax.set_zlim(0, z.max())
                        
                        ax.zaxis.set_major_locator(LinearLocator(10))
                        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                        
                        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
                        cb.set_label('density')
                              
                        #plt.show()            
                        plt.savefig(OUTPUT_DATA_DIR + "3dplot_" + str(len(peaks[0])) + '_' + str(lat) \
                                                          + '_' +  str(lon) + '_' +  str(lev) + ".png")
                        
                        
                        sendFile("3dplot_" + str(len(peaks[0])) + '_' + str(lat) + '_' +  str(lon) + '_' +  str(lev) + ".png")
                        
                        #plt.show()
                        
                        # let's no bloat disk i/o
                        #time.sleep(0.5)
                        
                        fig2 = plt.figure()
                        
                        ax = fig2.add_subplot(111)
                        
                        img2 = plt.imshow(np.rot90(opt_img,4),cmap=cm.spectral)
                        
                        
                        
                        
                        #for n, contour in enumerate(contours):
                        #    plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='white')
                            
                        cb2 = fig2.colorbar(img2, shrink=0.5, aspect=5)
                        cb2.set_label('density estimate')
                        
                        ax.scatter(peaks[0],peaks[1], s=100, facecolor='none', edgecolor = '#009999')
                        
                        #plt.savefig(OUTPUT_DATA_DIR + "contours_" + str(len(contours)) + '_' + str(lat) \
                        #                            + '_' +  str(lon) + '_' +  str(lev) + ".png")
                        
                        #plt.show()
                        plt.savefig(OUTPUT_DATA_DIR + "local_max_" + str(len(peaks[0])) + '_' + str(lat) \
                                                    + '_' +  str(lon) + '_' +  str(lev) + ".png")
                        sendFile("local_max_" + str(len(peaks[0])) + '_' + str(lat) + '_' +  str(lon) + '_' +  str(lev) + ".png")
                        
                        # let's no bloat disk i/o
                        #time.sleep(10)
                        
                except:
                    continue
                    #print "kde failed for gp: " + str(LAT) + " " + str(LON) + " " + str(LEV)
                
    print "finished!!"
 
 
 

            
            
        
    

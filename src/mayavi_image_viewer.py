
#!/usr/bin/python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

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

from skimage import data
from skimage import measure
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.exposure as skie

INPUT_DIR = '../../data/out/pics/contour/'

def displayImage(g,e,q,gmm):
    
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(4)
    #f, (ax1, ax2, ax3) = plt.subplots(1, 2, sharey=True)
    #axarr[0].plot(x, y)
    #axarr[1].scatter(x, y)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    pic0 = axarr[0].imshow(g,cmap=cm.gist_heat)
    pic1 = axarr[1].imshow(e,cmap=cm.gist_heat)
    pic2 = axarr[2].imshow(q,cmap=cm.gist_heat)
    pic3 = axarr[3].imshow(gmm,cmap=cm.gist_heat)
    
    plt.setp([a.get_xticklines() for a in axarr], visible=False)
    plt.setp([a.get_yticklines() for a in axarr], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr], visible=False)

    
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    #plt.setp([a.get_frame() for a in ax1], visible=False)
    
    
    #plt.axis('off') 
    cbar_ax = f.add_axes([0.65, 0.35, 0.02, 0.4])
    f.colorbar(pic0,cax=cbar_ax)
    plt.show()
    #plt.savefig(INPUT_DIR + '5_e_2_50_2_50_cropped_gist_heat.png',bbox_inches='tight')
    
    #obj = mayavi.mlab.imshow(grid, interpolate=False, colormap='gist_heat')   
    #mayavi.mlab.show() 
    print 'complete!'
    
def loadImage():
    
    g = plt.imread(INPUT_DIR + '5_g_2_50_2_50.png')
    e = plt.imread(INPUT_DIR + '5_e_2_50_2_50.png')
    q = plt.imread(INPUT_DIR + '5_q_2_50_2_50.png')
    gmm = plt.imread(INPUT_DIR + '5_gmm_2_50_2_50.png')
    
    width = q.shape[0]
    height = q.shape[1] 
    
    g_out = np.zeros(shape=(width,height))
    e_out = np.zeros(shape=(width,height))
    q_out = np.zeros(shape=(width,height))
    gmm_out = np.zeros(shape=(width,height))
    
    for w in range(0,width):
        for h in range(0,height):
            g_out[w][h] = g[w][h][2]
            e_out[w][h] = e[w][h][2] #take just a single channel
            q_out[w][h] = q[w][h][2] #take just a single channel
            gmm_out[w][h] = gmm[w][h][2] #take just a single channel
            
    return g_out, e_out,q_out,gmm_out

def main():
    g, e, q, gmm = loadImage()
    displayImage(g,e,q,gmm)
            
if __name__ == "__main__":  
    main()  
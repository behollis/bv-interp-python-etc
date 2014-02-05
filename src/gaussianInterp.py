# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

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

from scipy import stats
from scipy import linalg
from scipy import mat
from scipy import misc
from scipy import interpolate

from numpy import linspace,exp

from rpy2.robjects import numpy2ri
numpy2ri.activate()

from netcdf_reader import NetcdfReader

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

 
from skimage import data
from skimage import measure

r = robjects.r

SAMPLES = 500
        
if __name__ == '__main__':
    #dist1
    mean0 = -3;stdev0 = 1.5
    mean1 = 3;stdev1 = 1.0
    dist1 = r.rnorm(SAMPLES, mean = mean0, sd = stdev0) 
    dist2 = r.rnorm(SAMPLES, mean = mean1, sd = stdev1) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x = linspace(-8,7,100)
    y1 = mlab.normpdf( x, mean0, stdev0)
    y2 = mlab.normpdf( x, mean1, stdev1)
    plt.plot(x, y1, color='Blue', linewidth=2)
    plt.plot(x, y2, color='Green', linewidth=2)
    
    for a in [0.2, .4, .6, .8]:
        ia = (1.-a)*mean0 + a*mean1
        isigma = (1.-a)*stdev0 + a*stdev1
        interpolant = mlab.normpdf( x, ia, isigma)
        plt.plot(x, interpolant, '--', color='Black', linewidth=1)
    
    #ax.set_xlim(-8, +8)
    ax.set_ylim(0.0,0.5)
    
    plt.show()
       
        
    
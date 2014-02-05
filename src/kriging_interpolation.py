#!/usr/bin/python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 10/30/2012.
    Code shows advection of particles in 2d velocity field with configurable distributions at each grid point.
'''

import numpy as np
import sys, struct
import rpy2.robjects as robjects
import random
import math 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# for 1-D interpolation
# 1. define normal distributions, X1 and X2
# 2. define covariance function, such as the squared exponential with 
# 3. 


if __name__ == "__main__":
    pass
#import os
#import re

PROJ_PATH = '/home/behollis/thesis_code/code/interpRevProj/src/'
import sys
sys.path.append(PROJ_PATH)
from netcdf_reader import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from mayavi.mlab import *
import netCDF4 

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 

OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/pics/bv_interp/'

INPUT_DATA_DIR = '/home/behollis/thesis_data/data/in/ncdf/'
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'

def loadNetCDFMask():
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    mask = rreader.readVarArray('landv')
    np.save(OUTPUT_DATA_DIR + "landmask.npy",mask)
    
    im = plt.matshow(mask.T, origin='lower')
    plt.show()
    
def main():
    
    #loadNetCDFMask()
    
    startlat = 0; endlat = LAT
    startlon = 0; endlon = LON
    
    oslat = startlat
    oslon = startlon
    
    BLOCK_SIZE_X = 1
    BLOCK_SIZE_Y = 1
    
    OFFSET = 0
    
    skigrid = None
    cols = []
        
    for oslat in range(0, 35, BLOCK_SIZE_X):#LAT-BLOCK_SIZE, BLOCK_SIZE):
        cols.append([])
        ccols = cols[int(oslat / BLOCK_SIZE_X) - (OFFSET / BLOCK_SIZE_X)]
        for oslon in range(0, LON-BLOCK_SIZE_Y, BLOCK_SIZE_Y):
            block_file = OUTPUT_DATA_DIR + str(oslat) + '_' + str(oslon) + '*.npy'
            print block_file
            files = glob.glob(block_file)
            print files
            
            cblock = None
            if len(files) != 0:
                cblock = np.load(files[0])
                
            print cblock
            
            ccols.append(cblock)    
        print ccols
        
    skigrid = None 
    fullcol = []
    #concatenate blocks
    for cidx in range(0,len(cols),1):
        skigrid = np.asarray(cols[cidx][0])
        for ridx in range(1,len(cols[cidx]),1):
            skigrid = np.concatenate((skigrid,cols[cidx][ridx]),axis=0)
        fullcol.append(skigrid)
        
    fullgrid = np.asarray(fullcol[0])
    
    for idx in range(1,len(fullcol), 1):
        fullgrid = np.concatenate((fullgrid,fullcol[idx]), axis=1)
    
    #draw grid
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(fullgrid, origin='lower', interpolation='none', \
                         extent=(0,fullgrid.shape[1], 0, fullgrid.shape[0]))
    imgplot.set_cmap('spectral')
    plt.colorbar()
    ax.grid(which='both', axis='both', linestyle='-', color='white')
    plt.show()
    '''
    im = plt.matshow(fullgrid, origin='lower',cmap=plt.cm.spectral)
    plt.colorbar()
    plt.show()
    
    #s = imshow(fullgrid, colormap='gist_earth')
    print 'finished!'
        
if __name__ == "__main__":  
    main()
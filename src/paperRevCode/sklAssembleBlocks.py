#import os
#import re
import glob
import numpy as np
import matplotlib.pyplot as plt

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 

OUTPUT_DATA_DIR = '/home/behollis/thesis_data/data/outRev/pics/bv_interp/'
    
def main():
    
    startlat = 0; endlat = LAT
    startlon = 0; endlon = LON
    
    oslat = startlat
    oslon = startlon
    
    BLOCK_SIZE_X = 2
    BLOCK_SIZE_Y = 2
    
    skigrid = None
    cols = []
        
    for oslat in range(0, 6, 2):#BLOCK_SIZE):#LAT-BLOCK_SIZE, BLOCK_SIZE):
        cols.append([])
        ccols = cols[int(oslat / BLOCK_SIZE_X)]
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
    for col in fullcol:
        fullgrid = np.concatenate((fullgrid,col), axis=1)
    
    #draw grid
    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(fullgrid, origin='lower', interpolation='none', \
                         extent=(0,fullgrid.shape[1], 0, fullgrid.shape[0]))
    imgplot.set_cmap('spectral')
    plt.colorbar()
    ax.grid(which='major', axis='both', linestyle='-', color='white')
    plt.show()
        
if __name__ == "__main__":  
    main()
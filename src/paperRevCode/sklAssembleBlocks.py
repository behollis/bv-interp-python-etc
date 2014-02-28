#import os
#import re
import glob
import numpy as np
import matplotlib as plt

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
    
    BLOCK_SIZE = 2
    
    skigrid = None
    cols = []
        
    for oslat in range(0, LAT-BLOCK_SIZE*2, BLOCK_SIZE):
        for oslon in range(0, LON-BLOCK_SIZE*2, BLOCK_SIZE):
            block_file = OUTPUT_DATA_DIR + str(oslat) + '_' + str(oslon) + '*.npy'
            print block_file
            files = glob.glob(block_file)
            print files
            
            cblock = None
            if len(files) != 0:
                cblock = np.load(files[0])
                
            print cblock
            
            '''
            #concatenate blocks
            if cblock != None:
                np.concatenate((skigrid,cblock),axis=1)
            '''
                
            cols.append(cblock)    
        print cols
        skigrid = None
        
    print cols
        
    
            
            
            
            
            
            
if __name__ == "__main__":  
    main()
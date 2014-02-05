#!/usr/bin/python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

'''
    Author: Brad Hollister.
    Started: 02.07.2013
    Code advects streamlines for each ensemble member and increments the number of times a streamline crosses a given cell.
'''

import netCDF4 
import sys, struct
import math as pm
import numpy as np
import pylab as p
import math
import csv

from netcdf_reader import *

TOTAL_STEPS = 25
integration_step_size = 0.1
SEED_LAT = 42
SEED_LON = 21
SEED_LEVEL = 0
vclin = []
cf_vclin = []
  
FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
INPUT_DATA_DIR = '../../data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/'
COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600 
DEPTH = -2.0
INTEGRATION_DIR = 'b'
CELL_DENSITY = 0 # number of quad-divisions of regular grid for cell counting 
g_cc = [[['none'] for x in xrange(LON)] for x in xrange(LAT)]

#streamline positions (forward/backward from seed point) and cell counts
g_sl_f = [[],[],[]]
g_sl_b = [[],[],[]]

# from.. http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
#http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method
def rk4(x, v, a, dt):
    """Returns final (position, velocity) tuple after
    time dt has passed.

    x: initial position (number-like object)
    v: initial velocity (number-like object)
    a: acceleration function a(x,v,dt) (must be callable)
    dt: timestep (number)"""
    x1 = x
    v1 = v
    a1 = a(x1, v1, 0)

    x2 = x + 0.5*v1*dt
    v2 = v + 0.5*a1*dt
    a2 = a(x2, v2, dt/2.0)

    x3 = x + 0.5*v2*dt
    v3 = v + 0.5*a2*dt
    a3 = a(x3, v3, dt/2.0)

    x4 = x + v3*dt
    v4 = v + a3*dt
    a4 = a(x4, v4, dt)

    xf = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
    vf = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

    return xf, vf
                     
#from http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    
    # 

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)
           
def advect(step, member = 0):
    curr_pos_f = [[],[]]
    curr_pos_f[0] = g_sl_f[0][-1]
    curr_pos_f[1] = g_sl_f[1][-1]
    #curr_pos_f[2] = g_sl_f[2][-1]
    
    curr_pos_b = [[],[]]
    curr_pos_b[0] = g_sl_b[0][-1]
    curr_pos_b[1] = g_sl_b[1][-1]
    #curr_pos_b[2] = g_sl_b[2][-1]
    
    #Euler's method, forward integration
    velx_f, vely_f = interpInCell(curr_pos_f,member)
    
    
    #check which modes members belong to...
    '''
    if step == 1:
        print member
        print velx_f
        print vely_f
    '''
    
    if step == 1 and velx_f > 0.8 and velx_f < 1.2 and vely_f > -3.2 and vely_f < -2.8:
        global mode_members1
        mode_members1 = np.append(mode_members1,int(member))
        #print "mode #1 (1,-3), member: " + str(member)
    
    if step == 1 and velx_f > 2.8 and velx_f < 3.2 and vely_f > -3.53 and vely_f < -3.13:
        global mode_members2
        mode_members2 = np.append(mode_members2,int(member))
        #print "mode #2 (3,-3.33), member: " + str(member)

    curr_pos_f[0] += velx_f*integration_step_size
    curr_pos_f[1] += vely_f*integration_step_size
    g_sl_f[0].append(curr_pos_f[0])
    g_sl_f[1].append(curr_pos_f[1])
    g_sl_f[2].append(DEPTH)
    
    #Euler's method, backward integration
    velx_b, vely_b = interpInCell(curr_pos_b,member)
    curr_pos_b[0] -= velx_b*integration_step_size
    curr_pos_b[1] -= vely_b*integration_step_size
    g_sl_b[0].append(curr_pos_b[0])
    g_sl_b[1].append(curr_pos_b[1])
    g_sl_b[2].append(DEPTH)
    
def countCell(vert,member):
    # lat / lon is the gpt0 coord
    found_cell = False
    for cell in g_cc[ int(vert[0]) ][ int(vert[1]) ]:
        if cell == member:
            found_cell = True
            break
    if found_cell is False:
        g_cc[ int(vert[0]) ][ int(vert[1]) ].append(member)
        #print 'incrementing cell: lat->' + str(vert[0]) + ' lon->' + str(vert[1])
        #print 'current count: ' + str(len(g_cc[int(vert[0])][int(vert[1])])-1)
        
 
def interpInCell(ppos=[0.0,0.0],member = 0):
    #decompose fract / whole from particle position
    ppos_parts = [[0.0,0.0],[0.0,0.0]] #[fract,whole] for each x,y comp
    ppos_parts[0][0] = pm.modf(ppos[0])[0];ppos_parts[0][1] = pm.modf(ppos[0])[1]
    ppos_parts[1][0] = pm.modf(ppos[1])[0];ppos_parts[1][1] = pm.modf(ppos[1])[1]
    
    #print "ensemble alpha x: " + str( ppos_parts[0][0] )
    #print "ensemble alpha y: " + str( ppos_parts[1][0] )
    
    # grid point numbers:
    #
    # (2)---(3)
    # |      |
    # |      |
    # (0)---(1)
    
    #find four corner grid point indices, numbered from gpt0 = (bottom, left) TO gpt3 = (top, right)
    #calculated from whole parts 
    gpt0 = [ppos_parts[0][1], ppos_parts[1][1]]
    gpt1 = [ppos_parts[0][1] + 1, ppos_parts[1][1]]
    gpt2 = [ppos_parts[0][1], ppos_parts[1][1] + 1]
    gpt3 = [ppos_parts[0][1] + 1, ppos_parts[1][1] + 1]
    
    countCell(vert=gpt0, member = member)
    
    #vclin[curr_realization][curr_lat][curr_lon][curr_level][0] 
    #vclin[curr_realization][curr_lat][curr_lon][curr_level][1] 
    
    try:
        gpt0_velx = vclin[member][gpt0[0]][gpt0[1]][SEED_LEVEL][0]
        gpt0_vely = vclin[member][gpt0[0]][gpt0[1]][SEED_LEVEL][1]
         
        gpt1_velx = vclin[member][gpt1[0]][gpt1[1]][SEED_LEVEL][0]
        gpt1_vely = vclin[member][gpt1[0]][gpt1[1]][SEED_LEVEL][1] 
        
        gpt2_velx = vclin[member][gpt2[0]][gpt2[1]][SEED_LEVEL][0]
        gpt2_vely = vclin[member][gpt2[0]][gpt2[1]][SEED_LEVEL][1] 
        
        gpt3_velx = vclin[member][gpt3[0]][gpt3[1]][SEED_LEVEL][0]
        gpt3_vely = vclin[member][gpt3[0]][gpt3[1]][SEED_LEVEL][1] 
    except:
        return 0.0, 0.0
    
    lerp_velx = bilinear_interpolation(ppos[0], ppos[1], [(gpt0[0],gpt0[1],gpt0_velx),(gpt1[0],gpt1[1],gpt1_velx),\
                                                          (gpt2[0],gpt2[1],gpt2_velx),(gpt3[0],gpt3[1],gpt3_velx)])
    lerp_vely = bilinear_interpolation(ppos[0], ppos[1], [(gpt0[0],gpt0[1],gpt0_vely),(gpt1[0],gpt1[1],gpt1_vely),\
                                                          (gpt2[0],gpt2[1],gpt2_vely),(gpt3[0],gpt3[1],gpt3_vely)])
    
    return lerp_velx, lerp_vely
            
def readVelNetCDF(file):
    # open a the netCDF file for reading.
    ncfile = netCDF4.Dataset(file,'r') 
    # read the data in variable named 'data'.
    data = ncfile.variables['ks_test_stat'][:]
    #nx,ny = data.shape
    # check the data.
    #data_check = arange(nx*ny) # 1d array
    #data_check.shape = (nx,ny) # reshape to 2d array

    # close the file.
    ncfile.close()
    
    return data  

def writeStreamlinePositions(data,filename):
    #change to 'wb' after initial debug...
    filename = OUTPUT_DATA_DIR + filename
    writer = csv.writer(open(filename + ".csv", 'w'))
    
    #writes velocities with central forecast...
    for curr_comp in range(0,len(data),1):
        writer.writerow(data[curr_comp][:])
 
def writeParticles(member = 0):
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR)
    
    g_sl_b[0].reverse(); g_sl_b[0].pop();g_sl_b[1].reverse(); g_sl_b[1].pop();g_sl_b[2].reverse(); g_sl_b[2].pop()
    sl = [ g_sl_b[0] + g_sl_f[0], g_sl_b[1] + g_sl_f[1], g_sl_b[2] + g_sl_f[2] ]
        
    writeStreamlinePositions(sl,'crisp/crisp_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+'_mem'+str(member)+str_integration_values) 

def writeCellCounts():
    #change to 'wb' after initial debug...
    str_integration_values = '_ss' + str(integration_step_size) + '_ts' + str(TOTAL_STEPS) + '_dir_' + str(INTEGRATION_DIR)
    filename = OUTPUT_DATA_DIR + "crisp/" + 'cellcounts_lat'+str(SEED_LAT)+'_lon'+str(SEED_LON)+'_lev'+str(SEED_LEVEL)+str_integration_values
    writer = csv.writer(open(filename + ".csv", 'w'))
    
    #writes velocities with central forecast...
    for lat in range(0,LAT,1):
        for lon in range(0,LON,1):
            #print g_cc[lat][lon]
            writer.writerow(g_cc[lat][lon])
                
if __name__ == "__main__":  
    
    SEED_LAT = float(sys.argv[2])
    SEED_LON = float(sys.argv[3])
    SEED_LEVEL = int(sys.argv[4])
    integration_step_size = float(sys.argv[5])
    TOTAL_STEPS = int(sys.argv[6])
    INTEGRATION_DIR = str(sys.argv[7]).lower()
    #CELL_DENSITY = int(sys.argv[8])
    level = SEED_LEVEL
    
    #realizations file 
    pe_dif_sep2_98_file = INPUT_DATA_DIR + FILE_NAME
    pe_fct_aug25_sep2_file = INPUT_DATA_DIR + FILE_NAME_CENTRAL_FORECAST 
    
    #realizations reader 
    rreader = NetcdfReader(pe_dif_sep2_98_file)
    
    #central forecasts reader 
    creader = NetcdfReader(pe_fct_aug25_sep2_file)
    vclin8 = creader.readVarArray('vclin', 7)
    
    #deviations from central forecast for all 600 realizations
    vclin = rreader.readVarArray('vclin')  
    vclin = addCentralForecast(vclin, vclin8, level_start=SEED_LEVEL, level_end=SEED_LEVEL)  
    
    gen_streamlines = 'True'
    gen_streamlines = sys.argv[1]
   
    print "generating streamlines and cell counts"
    
    g_sl_f[0].append(SEED_LAT);g_sl_f[1].append(SEED_LON);g_sl_f[2].append(DEPTH)
    g_sl_b[0].append(SEED_LAT);g_sl_b[1].append(SEED_LON);g_sl_b[2].append(DEPTH)
    
    
    global mode_members1
    global mode_members2
    mode_members1 = np.ndarray(shape=0)
    mode_members2 = np.ndarray(shape=0)
    
    for mem in range(0,MEM):
        for i_step in range(1, TOTAL_STEPS):
            #print mem
            advect(i_step, member = mem)
        writeParticles(member = mem)
        g_sl_f = [[],[],[]];g_sl_b = [[],[],[]]
        g_sl_f[0].append(SEED_LAT);g_sl_f[1].append(SEED_LON);g_sl_f[2].append(DEPTH)
        g_sl_b[0].append(SEED_LAT);g_sl_b[1].append(SEED_LON);g_sl_b[2].append(DEPTH)
        
    writeCellCounts()
    np.savetxt( fname = OUTPUT_DATA_DIR + "crisp/" \
                          + 'mode_members1_'+str(SEED_LAT)+'_lon'+str(SEED_LON)\
                          +'_lev'+str(SEED_LEVEL), X = mode_members1)
    np.savetxt( fname = OUTPUT_DATA_DIR + "crisp/" \
                          + 'mode_members2_'+str(SEED_LAT)+'_lon'+str(SEED_LON)\
                          +'_lev'+str(SEED_LEVEL), X = mode_members2)
        
    print "finished!"
    
    
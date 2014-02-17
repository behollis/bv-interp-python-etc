''' Author: Brad Eric Hollister 
    Module: NetCDFReader
    Usage: Object to read velocity data from netcdf format
'''

from netCDF4 import Dataset as cdfdata
from numpy import arange # array module from http://numpy.scipy.org
from numpy import mgrid
from numpy.testing import assert_array_equal, assert_array_almost_equal

FILE_NAME = 'pe_dif_sep2_98.nc' 
FILE_NAME_CENTRAL_FORECAST = 'pe_fct_aug25_sep2.nc'
#OUTPUT_FILE_NAME = "..//..//mass_bay.vts"
#OUTPUT_FILE_NAME_LEGACY_VTK = "..//..//mass_bay.vtk"
INPUT_DATA_DIR = '/home/behollis/thesis_data/data/in/ncdf/'
OUTPUT_DATA_DIR = '../../data/out/csv/'
VCLIN_LAYER_FILE_BASENAME = "_vclin_outlev.vti"
#holds lat, lon and meter length depth info for grid indices...lat,lon and outlev
#float vgrid3_in[LAT][LON][LEV][AXIS3];

COM =  2
LON = 53
LAT = 90
LEV = 16
MEM = 600
AXIS3 = 3

VCLIN_NAME = "vclin"
#CHL_NAME = "CHL"


def addCentralForecast( vclin_in, vclin_central_forecast, level_start, level_end):
    #adds central forecast to each realization in ensemble '''
    #curr_level = level
    for curr_level in range(level_start, level_end+1, 1):
        for curr_lon in range(LON):
            for curr_lat in range(LAT): 
                for curr_realization in range(MEM):
    #                    print str(curr_level) + ' ' + str(curr_lon) + ' ' + str(curr_lat) + ' ' + str(curr_realization)
                    vclin_in[curr_realization][curr_lat][curr_lon][curr_level][0] += vclin_central_forecast[curr_lat][curr_lon][curr_level][0] 
                    vclin_in[curr_realization][curr_lat][curr_lon][curr_level][1] += vclin_central_forecast[curr_lat][curr_lon][curr_level][1]
        return vclin_in
                  
#              vclin_in[curr_realization][curr_lat][curr_lon][curr_level][2] 
#                  += vclin_central_forecast[0][curr_lat][curr_lon][curr_level][2];
                  
#               chl_in[curr_realization][curr_lat][curr_lon][curr_level] 
#                  += chl_central_forecast[0][curr_lat][curr_lon][curr_level];
                  
#               temp_in[curr_realization][curr_lat][curr_lon][curr_level] += temp_central_forecast[0][curr_lat][curr_lon][curr_level]
                  
#               salt_in[curr_realization][curr_lat][curr_lon][curr_level] += salt_central_forecast[0][curr_lat][curr_lon][curr_level]


def addCentralForecastScalar( vclin_in, vclin_central_forecast, level_start, level_end):
    #adds central forecast to each realization in ensemble '''
    #curr_level = level
    for curr_level in range(level_start, level_end+1, 1):
        for curr_lon in range(LON):
            for curr_lat in range(LAT): 
                for curr_realization in range(MEM):
    #                    print str(curr_level) + ' ' + str(curr_lon) + ' ' + str(curr_lat) + ' ' + str(curr_realization)
                    vclin_in[curr_realization][curr_lat][curr_lon][curr_level] \
                        += vclin_central_forecast[curr_lat][curr_lon][curr_level] 
                    
        return vclin_in
                  
#              vclin_in[curr_realization][curr_lat][curr_lon][curr_level][2] 
#                  += vclin_central_forecast[0][curr_lat][curr_lon][curr_level][2];
                  
#               chl_in[curr_realization][curr_lat][curr_lon][curr_level] 
#                  += chl_central_forecast[0][curr_lat][curr_lon][curr_level];
                  
#               temp_in[curr_realization][curr_lat][curr_lon][curr_level] += temp_central_forecast[0][curr_lat][curr_lon][curr_level]
                  
#               salt_in[curr_realization][curr_lat][curr_lon][curr_level] += salt_central_forecast[0][curr_lat][curr_lon][curr_level]


class NetcdfReader:
    
    def __init__(self, file_name):
        self._file = file_name
        self._dataset = cdfdata(self._file)
        ''' self.vgrid3: depth values at regular grid indices '''
 #       self._field  = None#vgrid3 = None #mgrid[0:LAT,0:LON,0:LEV,0:AXIS3] #vgrid3_in[LAT][LON][LEV][AXIS3];
    
 #   def getDataset(self):
 #           return self._dataset
        
    def readVarArray(self, netcdf_var, day=-1):
#        self.vgrid3 = netcdf_dataset.readVariable
        # read the data in variable named 'data'.
        field = None
        if day == -1:
            field = self._dataset.variables[netcdf_var][:]
        else:
            field = self._dataset.variables[netcdf_var][day]
            
 #       print field
 #       print field.shape
        return field
        self.netcdf_dataset.close()
#        nx,ny,nz = self.vgrid3.shape
        # check the data.
#        data_check = arange(nx*ny*nz) # 1d array
#        data_check.shape = (nx,ny,nz) # reshape to 2d array
#        try:
#            assert_array_equal(self.vgrid3, data_check)
#            print '*** SUCCESS reading example file simple_xy.nc'
#        except:
#            print '*** FAILURE reading example file simple_xy.nc'

if __name__ == "__main__":
    file = INPUT_DATA_DIR + FILE_NAME
    reader = NetcdfReader(file)
    reader.readVarArray('vgrid3')
    
    
    
    
    

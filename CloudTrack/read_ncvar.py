from matplotlib.pyplot import figure
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as pl

def retrieve_variable(variable_name,netcdf_path): #open corresponding netcdf file and read data
    all_data=xr.open_dataset(netcdf_path)
    var=all_data[variable_name].values
    if variable_name=='bc':
        variable=np.zeros((len(var[:,0,0]),len(var[0,:,0]),len(var[0,0,:]),3))
        variable[:,:,:,0] = var
    else:
        variable=var
    x=all_data['x'].values
    y=all_data['y'].values
    t=all_data['t'].values
    return variable, x, y, t; 


nc_path='/data/rico_data/'
nc_filename='cloudtrack_variables_1024.nc'
tracked_param = 'ql'; # ql --> cloud path; qr --> rain path; bc --> cloud core path; couvreux --> thermal path
ipath=0;ibase=2;iup=1;
[variable,x,y,t] = retrieve_variable(tracked_param,nc_path+nc_filename);
qr_threshold = 0.01; ql_threshold = 0.01; couv_threshold = 0.05; bc_threshold = 1;
ql_path=variable[:,:,:,ipath]
pl.figure()
pl.contour(x,y,ql_path[:,:,10])

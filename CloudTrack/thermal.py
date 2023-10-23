import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as pl

nx = 256
ny = 256
nz = 200

fid = open('couvreux.0003600','rb')
couv_scalar = np.fromfile(fid)
couv_scalar=np.reshape(couv_scalar,[nz,nx,ny])


def identify_thermal_presence(couv_scalar):
    nz = len(couv_scalar[:,1,1])
    nx = len(couv_scalar[1,:,1])
    ny = len(couv_scalar[1,1,:])
    thermal_presence = np.zeros((nz,nx,ny))
    couv_slab_mean = np.zeros(nz)
    for k in range(nz):
        couv_slab_mean = np.mean(couv_scalar[k,:,:])
        couv_std = np.std(couv_scalar[k,:,:])
        for i in range(nx):
            for j in range(ny):
                thermal_presence[k,i,j] = ((couv_scalar[k,i,j] - couv_slab_mean) / couv_std) > 1
    return thermal_presence


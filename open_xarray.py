import xarray as xr
from netCDF4 import Dataset

def open_xarray(filepath):
    ds=xr.open_dataset(filepath,decode_times=True)
    rootgroup = Dataset(filepath,"r")
    for group in rootgroup.groups:
        ds=xr.merge([ds, xr.open_dataset(filepath,group=group,decode_times=True)])
    return ds
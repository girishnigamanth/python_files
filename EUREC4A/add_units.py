import matplotlib.pyplot as pl
import xarray as xr
import numpy as np


MHH_2d=xr.open_dataset('/fs/ess/PFS0220/eurec4a/microhh_cloud_pdfs/eurec4a_Feb2_512_200m_12kmtop_nudged6hr_jan31st_2d.LWP.LWP_>=_0.01.connected_regions.pdf.nc',decode_times=False)

MHH_2d['time'].attrs['units'] = 's since 2020-01-31 00:00:00'
MHH_2d['time'].attrs['calendar'] = 'proleptic_gregorian'
MHH_2d['time'].attrs['axis'] = 'T'
#write out to netcdf file
MHH_2d.to_netcdf('/fs/ess/PFS0220/eurec4a/microhh_results/eurec4a_Feb2_512_200m_12kmtop_nudged6hr_jan31st_2d.LWP.LWP_>=_0.01.connected_regions.pdf_mod.nc')
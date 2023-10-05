import statistics
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as pl
import math
import scipy as sc
from scipy import stats
import os, glob
import matplotlib.animation as manimation
from IPython.display import HTML
from matplotlib import rc
from matplotlib import cm
import sys
import seaborn as sns

nc_save_path='/data/rico_cloudtrack/cloudtrack_outputnc/'
nc_output_filename='savefile_1024_84k_06.nc'

############################## indices of saved cell variables ########################################
icellid=6;ipath=3;iup=4;ibase=5;
iparent=7;ichildren=8;
ii=0;ij=1;ik=2;
rel_id=0;rel_elements=1;rel_relations=2;

############################## read netcdf file and open variable function ############################
def retrieve_variable(variable_name,netcdf_path): 
    all_data=xr.open_dataset(netcdf_path)
    variable=all_data[variable_name].values
    return variable; 
############################# set relation function ###################################################
def relation_array(ncells,cells,relation_index):
    cell_relations = np.zeros((ncells,3000))
    cell_relations[:,rel_id],cell_relations[:,rel_elements] = np.unique(cells[icellid,:],return_counts=True)
    for i in range(0,ncells):
        tot=0;
        if i>0:
            tot = int(np.sum(cell_relations[0:i-1,rel_elements])); 
        x = cells[relation_index,tot:tot+int(cell_relations[i,rel_elements])]
        y = x[x>0]
        cell_relations[i,rel_relations] = len(y)
        cell_relations[i,3:3+len(y)] = y-1
    cell_relations = cell_relations.astype(np.int32)
    return cell_relations;

################################### get variable arrays###############################

rain = retrieve_variable('rain',nc_save_path+nc_output_filename)
cloud = retrieve_variable('cloud',nc_save_path+nc_output_filename)
thermal = retrieve_variable('thermal',nc_save_path+nc_output_filename)

#################### declare x,y and t sizes and total cells in each array############

nx = int(max(cloud[ii,:])+1); ny = int(max(cloud[ij,:])+1); nt = int(max(cloud[ik,:])+1);
ncloud = int(max(cloud[icellid,:])+1); nrain = int(max(rain[icellid,:])+1); 
nthermal = int(max(thermal[icellid,:])+1);
print('Setting Relations')

################### retrieve relation arrays#########################################
cloud_children = relation_array(ncloud,cloud,ichildren)
cloud_parent = relation_array(ncloud,cloud,iparent)
rain_parent = relation_array(nrain,rain,iparent)
thermal_children = relation_array(nthermal,thermal,ichildren)

# 0 --> cloud ids; 1 --> cloud elements in time step; 2 --> nchildren; 3,4... --> children ids 

####################### statistics over entire cell and relation plots #######################
def dostatistics_overcell():
    global rain_total;
    cloud_total = np.zeros((ncloud)); cloud_depth = np.zeros((ncloud)); cloud_volume = np.zeros((ncloud)); 
    cloud_average = np.zeros((ncloud));
    rain_total = np.zeros((ncloud)); therm_total = np.zeros((ncloud))
    rain_instant = []
    print('starting statistics')
    for i in range(0,ncloud):
        if (i%1000) == 0:
            print('Calculating statistics for',i)
        if cloud_children[i,rel_elements]>nmincells and cloud_children[i,rel_relations]>0:
            tot = np.sum(cloud_children[0:i-1,rel_elements]); 
            x=np.sum(cloud[ipath,tot:tot+int(cloud_children[i,rel_elements])]); 
            y=np.sum(cloud[iup,tot:tot+int(cloud_children[i,rel_elements])]-cloud[ibase,tot:tot+int(cloud_children[i,rel_elements])]);
            cloud_total[i] = cloud_total[i] + x
            cloud_depth[i] = cloud_depth[i] + y/cloud_children[i,rel_elements]
            cloud_average[i] = cloud_total[i] / y;
            for nchild in range(0,(cloud_children[i,rel_relations])):
                j = int(cloud_children[i,3+nchild]); 
                tot = np.sum(rain_parent[0:j-1,rel_elements]); #find index of starting of cell
                x=np.sum(rain[3,tot:tot+int(rain_parent[j,rel_elements])]); #sum elements in cell 
                rain_total[i] = rain_total[i] + x
            for nparent in range(0,(cloud_parent[i,rel_relations])):
                j = int(cloud_parent[i,3+nparent]); 
                tot = np.sum(thermal_children[0:j-1,rel_elements]); #find index of starting of cell
                x=np.sum(thermal[ipath,tot:tot+int(thermal_children[j,rel_elements])]); #sum elements in cell 
                therm_total[i] = therm_total[i] + x
    rain_to_cloud = rain_total/cloud_total
    
    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(rain_total,cloud_depth)
    pl.xlabel('Integral of Rain Water (qr) over the Cell');
    pl.ylabel('Average Cloud Depth (m)');
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    ax.scatter(rain_total,therm_total)
    pl.xlabel('Integral of Rain Water (qr) over the Cell ');
    pl.ylabel('Integral of Thermal Scalar over the Cell ');
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(cloud_depth,rain_to_cloud)
    pl.xlabel('Average Cloud Depth (m)');
    pl.ylabel('Ratio of Rain Water to Condensed Water Integral');
    ax.set_yscale('log')
    ax.set_xscale('log')

########################## instantaneous stats of one cell ######################################
def dostatistics_singlecell(max_cell):
    global cloud_base_atmax, cloud_top_atmax, rain_atmax, rain_base_atmax, rain_sorted;
    global therm_atmax, cloud_nelements,cloud_atmax, rain_ratio;
    cloud_base_atmax = np.zeros((nt)); cloud_top_atmax = np.zeros((nt)); rain_atmax = np.zeros((nt)); 
    rain_base_atmax = np.zeros((nt)); therm_atmax = np.zeros((nt));
    cloud_nelements = np.zeros((nt)); cloud_atmax = np.zeros((nt));
    rain_sorted = np.sort(rain_total)[::-1]
    max_id = np.nonzero(rain_total==rain_sorted[max_cell])[0][0];
    #max_id = int(max_id_tuple[0])
    tot_c = np.sum(cloud_children[0:max_id-1,1]);  # starting index of given cell 
    x=cloud[2,tot_c:tot_c+cloud_children[max_id,1]];
    rain_child = []; rain_base = []
    rain_time = []
    for nchild in range(0,cloud_children[max_id,rel_relations]):
        j=(int(cloud_children[max_id,3+nchild]))
        tot_r = np.sum(rain_parent[0:j-1,1]);
        rain_child = np.concatenate([rain_child,rain[ipath,tot_r:tot_r+int(rain_parent[j,1])]])
        rain_base = np.concatenate([rain_base,rain[ibase,tot_r:tot_r+int(rain_parent[j,1])]])
        rain_time = np.concatenate([rain_time,rain[ik,tot_r:tot_r+int(rain_parent[j,1])]])
    therm_parent = [];therm_time=[]
    for nparent in range(0,cloud_parent[max_id,rel_relations]):
        j=(int(cloud_parent[max_id,3+nparent]))
        tot_r = np.sum(thermal_children[0:j-1,rel_elements]);
        therm_parent = np.concatenate([therm_parent,thermal[ipath,tot_r:tot_r+int(thermal_children[j,rel_elements])]])
        therm_time = np.concatenate([therm_time,thermal[ik,tot_r:tot_r+int(thermal_children[j,rel_elements])]])

    for i in range(0,nt):
        z = (x==i)
        if np.sum(z)>0:
            y = cloud[5,tot_c:tot_c+cloud_children[max_id,1]];
            cloud_base_atmax[i] = np.sum(y[z])/np.sum(z);
            yy = cloud[4,tot_c:tot_c+cloud_children[max_id,1]];
            cloud_top_atmax[i] = np.sum(yy[z])/np.sum(z);
            cloud_nelements[i] = np.sum(yy[z]-y[z]);
            y = cloud[ipath,tot_c:tot_c+cloud_children[max_id,1]];
            cloud_atmax[i] = np.sum(y[z]);
            rain_atmax[i] = np.sum(rain_child[rain_time==i]);
            therm_atmax[i] = np.sum(therm_parent[therm_time==i]);
            rain_base_atmax[i] = np.sum(rain_base[rain_time==i])/len(rain_base[rain_time==i]);
            #cloud_nelements[i] = np.sum(z);
    rain_ratio = rain_atmax/cloud_atmax
    rain_atmax = rain_atmax*2000/max(rain_atmax)
    
    pl.figure()
    fig, ax1=pl.subplots(4)
    ax1[0].plot(cloud_base_atmax[cloud_base_atmax>0],'r-')
    ax1[0].plot(cloud_top_atmax[cloud_base_atmax>0],'r-')
    ax1[0].plot(rain_atmax[cloud_base_atmax>0],'b-')
    pl.ylabel('Height (m)')
    ax2 = ax1[0].twinx()
    ax2.plot(cloud_nelements[cloud_base_atmax>0],'g--')
    pl.xlabel('Time (in Minutes)')
    pl.ylabel('Integral of Cloud Elements')
    #pl.show()

    ax1[1].plot(cloud_base_atmax[cloud_base_atmax>0],'r-')
    ax1[1].plot(cloud_top_atmax[cloud_base_atmax>0],'r-')
    ax1[1].plot(rain_atmax[cloud_base_atmax>0],'b-')
    pl.ylabel('Height (m)')
    ax2 = ax1[1].twinx()
    ax2.plot(cloud_atmax[cloud_base_atmax>0],'y--')
    pl.xlabel('Time (in Minutes)')
    pl.ylabel('Integral of Condensed Water')
    #pl.show()

    #pl.figure()
    #fig, ax1=pl.subplots()
    ax1[2].plot(cloud_base_atmax[cloud_base_atmax>0],'r-')
    ax1[2].plot(cloud_top_atmax[cloud_base_atmax>0],'r-')
    ax1[2].plot(rain_atmax[cloud_base_atmax>0],'b-')
    pl.ylabel('Height (m)')
    ax2 = ax1[2].twinx()
    ax2.plot(rain_ratio[cloud_base_atmax>0],'k--')
    pl.xlabel('Time (in Minutes)')
    pl.ylabel('Rain Water to Condensed Water ration (qr/ql)')
    #pl.show()

    #pl.figure()
    #fig, ax1=pl.subplots()
    ax1[3].plot(cloud_base_atmax[cloud_base_atmax>0],'r-')
    ax1[3].plot(cloud_top_atmax[cloud_base_atmax>0],'r-')
    ax1[3].plot(rain_atmax[cloud_base_atmax>0],'b-')
    pl.ylabel('Height (m)')
    ax2 = ax1[3].twinx()
    ax2.plot(therm_atmax[cloud_base_atmax>0],'m--')
    pl.xlabel('Time (in Minutes)')
    pl.ylabel('Integral of Thermal Scalar')
    pl.show()
############################## instantaneous stats over several cells ############################
def dostatistics_instantaneous():
    ncells=500;
    global cloud_instant,cloud_nelements_instant,rain_instant,therm_instant, rain_sorted;
    cloud_nelements_instant = np.zeros((nt*ncells)); cloud_instant = np.zeros((nt*ncells));
    rain_instant = np.zeros((nt*ncells)); therm_instant = np.zeros((nt*ncells));
    rain_sorted = np.sort(rain_total)[::-1]
    count=0;flag=0;
    if rain_cutoff>0:
        ncells=np.nonzero(rain_sorted==min(rain_sorted[rain_sorted>rain_cutoff]))[0][0];
    print(ncells);
    for iii in range(0,ncells):
        cell_id = np.nonzero(rain_total==rain_sorted[iii])[0][0];
        tot_c = np.sum(cloud_children[0:cell_id-1,1]);  # starting index of given cell 
        x=cloud[ik,tot_c:tot_c+cloud_children[cell_id,1]];
        rainpath_all = []; raintime_all = [];
        thermpath_all = [];thermtime_all=[]
        for nchild in range(0,cloud_children[cell_id,rel_relations]):
            j=(int(cloud_children[cell_id,3+nchild]))
            tot_r = np.sum(rain_parent[0:j-1,1]);
            rainpath_all = np.concatenate([rainpath_all,rain[ipath,tot_r:tot_r+int(rain_parent[j,rel_elements])]])
            raintime_all = np.concatenate([raintime_all,rain[ik,tot_r:tot_r+int(rain_parent[j,rel_elements])]])
        for nparent in range(0,cloud_parent[cell_id,rel_relations]):
            j=(int(cloud_parent[cell_id,3+nparent]))
            tot_r = np.sum(thermal_children[0:j-1,rel_elements]);
            thermpath_all = np.concatenate([thermpath_all,thermal[ipath,tot_r:tot_r+int(thermal_children[j,rel_elements])]])
            thermtime_all = np.concatenate([thermtime_all,thermal[ik,tot_r:tot_r+int(thermal_children[j,rel_elements])]])

        for i in range(0,nt):
            z = (x==i); t_z = (thermtime_all==i);
            if np.sum(z)>0 and np.sum(t_z)>0:
                y = cloud[ibase,tot_c:tot_c+cloud_children[cell_id,rel_elements]];
                yy = cloud[iup,tot_c:tot_c+cloud_children[cell_id,rel_elements]];
                cloud_nelements_instant[count] = np.sum(yy[z]-y[z]);
                y = cloud[ipath,tot_c:tot_c+cloud_children[cell_id,1]];
                cloud_instant[count] = np.sum(y[z]);
                rain_instant[count] = np.sum(rainpath_all[raintime_all==i]);
                therm_instant[count] = np.sum(thermpath_all[t_z]);
                count = count+1
    raintocloud = rain_instant/cloud_instant
    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(cloud_nelements_instant[rain_instant>rain_lim],cloud_instant[rain_instant>rain_lim])
    pl.xlabel('Integral of Cloud Volume at t');
    pl.ylabel('Integral of Condensate Water (ql) over the Cell at t');
    #ax.set_yscale('log')
    #ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(rain_instant[rain_instant>rain_lim],cloud_instant[rain_instant>rain_lim])
    pl.xlabel('Integral of Rainwater (qr) over the Cell at t');
    pl.ylabel('Integral of Condensate Water (ql) over the Cell at t');
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(raintocloud[rain_instant>rain_lim],cloud_instant[rain_instant>rain_lim])
    pl.xlabel('Rain to Cloud Water ratio at t');
    pl.ylabel('Integral of Condensate Water (ql) over the Cell at t');
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(cloud_instant[rain_instant>rain_lim],therm_instant[rain_instant>rain_lim])
    pl.xlabel('Integral of Condensate Water (ql) over the Cell at t');
    pl.ylabel('Integral of Thermal Scalar over the Cell at t');
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig = pl.figure()
    ax = pl.gca()
    pl.scatter(cloud_nelements_instant[rain_instant>rain_lim],therm_instant[rain_instant>rain_lim])
    pl.xlabel('Integral of Cloud Volume at t');
    pl.ylabel('Integral of Thermal Scalar over the Cell at t');
    ax.set_yscale('log')
    ax.set_xscale('log')

#    fig = pl.figure()
#    ax = pl.gca()
#    pl.scatter(rain_instant[rain_instant>rain_lim],therm_instant[rain_instant>rain_lim])
#    pl.xlabel('Integral of Rain Water (qr) at t');
#    pl.ylabel('Integral of Thermal Scalar over the Cell at t');
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#############################################################################################    

nmincells=1e3
dostatistics_overcell()
maximum_cell=0; 
rain_lim=0; rain_cutoff=1e5;
dostatistics_singlecell(maximum_cell)
#dostatistics_instantaneous()



from cv2 import illuminationChange
import statistics
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as pl
import math
import scipy.io as sc
import netCDF4 as nc
import os, glob
import matplotlib.animation as manimation
from IPython.display import HTML
from matplotlib import rc
from matplotlib import cm
import sys
import pickle

def make_netcdffile_fromcross(time_start, time_end, time_gap, x_spacing, path, output_name, del_files): #output netcdf file with variable across time steps from dump files
    float_type = 'f4'
    float_type1 = 'i2'
    time_ind_total = int((time_end-time_start)/time_gap) + 1
    time = np.linspace(time_start,time_end,time_ind_total)
    main_variable=['qr','ql','couvreux','bc']
    attribute=['path','up','base']
    cross_section='.xy.'
    filename = path+main_variable[0] + attribute[0] + cross_section + str(format(int(time[0]),'07d'))
    fid=open(filename,'rb')
    x_length = int(np.sqrt(len(np.fromfile(fid))))
    x=np.linspace(x_spacing,x_length*x_spacing,x_length)
    y=np.linspace(x_spacing,x_length*x_spacing,x_length)
    nvar=3
    fid.close()
    if os.path.exists(path+output_name):
        os.remove(path+output_name)
    nc_file = nc.Dataset(path+output_name, mode="w", datamodel="NETCDF4", clobber=False)
    nc_file.createDimension("x", x_length)
    nc_file.createDimension("y", x_length)
    nc_file.createDimension("t", time_ind_total)
    nc_file.createDimension("nvar",nvar)
    nc_x = nc_file.createVariable("x", float_type, ("x"))
    nc_y = nc_file.createVariable("y", float_type, ("y"))
    nc_t = nc_file.createVariable("t", float_type, ("t"))
    nc_nvar = nc_file.createVariable("nvar", int, ("nvar"))
    nc_multiplying_factor = nc_file.createVariable("multiplying_factor", float_type1, ("1"))

    for j in range(0,len(main_variable)-1):
        var_name = main_variable[j]
        locals()[var_name] = np.zeros((x_length,x_length,time_ind_total,nvar))
        locals()['nc_'+var_name] = nc_file.createVariable(var_name, float_type, ("x","y","t","nvar"))  
        for k in range(0,len(attribute)):                  
            for i in range(0,time_ind_total):
                filename = path+main_variable[j] + attribute[k] + cross_section + str(format(int(time[i]),'07d'))
                fid=open(filename,'rb')
                var = np.fromfile(fid)
                if k==0:
                    var = np.around(var * multiplying_factor);
                locals()[var_name][:,:,i,k]=np.reshape(var,[x_length,x_length])
                fid.close()
        locals()[var_name]=locals()[var_name].astype(np.int16)
        locals()['nc_'+var_name][:] = locals()[var_name][:]
    
    var_name = main_variable[3]
    locals()[var_name] = np.zeros((x_length,x_length,time_ind_total,nvar+1))
    locals()[var_name][:,:,:,0:3] = locals()[main_variable[np.where(main_variable=='ql')]]
    locals()['nc_'+var_name] = nc_file.createVariable(var_name, float_type1, ("x","y","t","nvar"))

    for i in range(0,time_ind_total):
        filename = path+main_variable[3] +attribute[0]+ cross_section + str(format(int(time[i]),'07d'))
        fid=open(filename,'rb')
        var = np.fromfile(fid)
        var = np.around(var * multiplying_factor);
        locals()[var_name][:,:,i,4]=np.reshape(var,[x_length,x_length])
        fid.close()
    locals()[var_name]=locals()[var_name].astype(np.int16)
    locals()['nc_'+var_name][:] = locals()[var_name][:]




    nc_x [:] = x[:]
    nc_y [:] = y[:]
    nc_t [:] = time[:]
    nc_nvar [:]=nvar 
    nc_multiplying_factor = multiplying_factor
    nc_file.close()
    if del_files:
        #files = ["u", "v", "w", "thl", "time", "qr", "ql", "couv", "grid", "nr","qt","therm","bc"]
        files = ["path","up","base.xy"]
        for file in os.listdir(path):
            for i in range(0,len(files)):
                if file.find(files[i])!=-1: 
                    os.remove(path+file)
                #rm u* v* w* thl* time* qr* ql* couv* grid* nr* qt* therm* 
##########################################################################################
def converttoarray_cell(cells):
    total=0;
    for i in range(0,len(cells)):
        total = total + cells[i].nelements
    cell_array = np.zeros((9,total))
    total=0;
    for i in range(0,len(cells)):
        for n in range(0,cells[i].nelements):
            cell_array[0:3,total] = [cells[i].location[0][n],cells[i].location[1][n],cells[i].location[2][n]];
            cell_array[3:6,total] = [cells[i].value[0][n],cells[i].value[1][n],cells[i].value[2][n]];
            cell_array[6,total] = cells[i].id
            total = total+1
        if cells[i].nparents>0:
            for npar in range(0,cells[i].nparents):
                cell_array[7,total-cells[i].nelements+npar-1] = cells[i].parents[npar].id + 1
        if cells[i].nchildren>0:
            for nchild in range(0,cells[i].nchildren):
                cell_array[8,total-cells[i].nelements+nchild-1] = cells[i].children[nchild].id + 1

    return cell_array;
###########################################################################################
def writeout_netcdf(cloud,core,rain,thermal,properties):

    if os.path.exists(nc_save_path+nc_output_filename):
        os.remove(nc_save_path+nc_output_filename)
    
    nc_file = nc.Dataset(nc_save_path+nc_output_filename,mode="w", datamodel="NETCDF4", clobber=False)
    float_type = 'f4'

    nc_file.createDimension("properties",len(properties))
    nc_properties = nc_file.createVariable("properties",str,("properties"))
    
    if len(cloud)>0:
        cloud_length = np.linspace(0,len(cloud[1,:])-1,len(cloud[1,:]))
        nc_file.createDimension("cloud_length",len(cloud_length))
        nc_cloud_length = nc_file.createVariable("cloud_length",float_type,("cloud_length"))
        nc_cloud = nc_file.createVariable("cloud",float_type,("properties","cloud_length"))
        nc_cloud_length[:] = cloud_length[:]
        nc_cloud [:] = cloud [:]
    
    if len(core)>0:
        core_length = np.linspace(0,len(core[1,:])-1,len(core[1,:]))
        nc_file.createDimension("core_length",len(core_length))
        nc_core_length = nc_file.createVariable("core_length",float_type,("core_length"))
        nc_core = nc_file.createVariable("core",float_type,("properties","core_length"))
        nc_core_length[:] = core_length[:]
        nc_core [:] = core [:]

    if len(rain)>0:    
        rain_length = np.linspace(0,len(rain[1,:])-1,len(rain[1,:]))
        nc_file.createDimension("rain_length",len(rain_length))
        nc_rain_length = nc_file.createVariable("rain_length",float_type,("rain_length"))
        nc_rain = nc_file.createVariable("rain",float_type,("properties","rain_length"))
        nc_rain_length[:] = rain_length[:]
        nc_rain [:] = rain [:]
    
    if len(thermal)>0:
        thermal_length = np.linspace(0,len(thermal[1,:])-1,len(thermal[1,:]))
        nc_file.createDimension("thermal_length",len(thermal_length))
        nc_thermal_length = nc_file.createVariable("thermal_length",float_type,("thermal_length"))
        nc_thermal = nc_file.createVariable("thermal",float_type,("properties","thermal_length"))
        nc_thermal_length[:] = thermal_length[:]
        nc_thermal [:] = thermal [:]
    
    nc_properties = properties
    
    nc_file.close()
        
###########################################################################################
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
##########################################################################################
def find_boolean(variable, threshold_criteria, i_tstart,iscore): #variable is f(i, j, t): --> outputs boolean -1 (unsatisfied) 0 (satisfied) 
    boolean = np.zeros(( len(variable[:,0,0,0]), len(variable[0,:,0,0]), len(variable[0,0,:,0])))
    boolean = -1
    if iscore:
        ql_path = variable[:,:,:,0]
        core_path = variable[:,:,:,3]
        boolean = np.where(variable[:,:,:,0] >ql_threshold and variable[:,:,:,3]>bc_threshold \
            and variable[:,:,:,ibase]<(0.5*max(variable[:,:,:,ibase])+0.5*max(variable[:,:,:,itop])),0,-1)
    else:
        boolean = np.where(variable[:,:,:,ipath]>threshold_criteria,0,-1)

    if i_tstart>0:
        boolean[:,:,0:i_tstart-1] = -1
    return boolean;
##########################################################################################
class cell:
    def __init__(self, id):
        self.id = id
        self.value = [[],[],[]]
        self.location = [[],[],[]]
        self.nelements = 0
        self.nelements_local = 0
        self.nchildren = 0
        self.nparents = 0
        self.nsiblings = 0
        self.nsplitters = 0
        self.parents = []
        self.children = []
        self.splitters = []

    def add_elements(self, i, j, k, var_values):
        self.location[0].append(i)
        self.location[1].append(j)
        self.location[2].append(k)
        self.value[0].append(var_values[0])
        self.value[1].append(var_values[1])
        self.value[2].append(var_values[2])
        self.nelements = self.nelements + 1
        self.nelements_local = self.nelements_local + 1
    def __del__(self):
        return
########################################################################################
def identify_elements_in_cell(i,j,k,new_cell):  #input the ijk at which boolean is satisfied along with boolean and new cell created 

    global oldcell,oi,oj,ok,booli;
    new_cell.add_elements(i,j,k,cell_variable[i,j,k,:])
    booli[i,j,k] = -2

    ii=i-1; jj=j; kk=k; #look west
    if ii<0:
       ii = nx-1
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==1:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)  

    ii=i+1; jj=j; kk=k;  #look east
    if ii>nx-1:
       ii = 0
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==2:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)  

    ii=i; jj=j+1; kk=k;  #look north
    if jj>ny-1:
        jj = 0
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==3:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)  

    ii=i; jj=j-1; kk=k;  #look south
    if jj<0:
       jj = ny-1
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==4:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)  

    ii=i; jj=j; kk=min(k+1,nt-1);  #look forward
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==5:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)     

    ii=i; jj=j; kk=max(k-1,0);  #look backward
    if (booli[ii,jj,kk] == 0) and \
      ((cell_variable[i,j,k,ibase] <= cell_variable[ii,jj,kk,iup])\
      and (cell_variable[i,j,k,iup] >= cell_variable[ii,jj,kk,ibase])) :
        if new_cell.nelements_local>nmaxelems and recursion_point==6:
            oldcell = True;
            oi.append(ii);oj.append(jj);ok.append(kk);
        else:
            identify_elements_in_cell(ii,jj,kk,new_cell)  

#################################################################################################
def create_new_cell(variable,bool):                                # input the boolean and the variable, output is the cells tracked (i,j,t) based on boolean 
    cell_number = 0;
    global booli,cell_variable,nx,ny,nt,oldcell,flag,oi,oj,ok;
    nx = len(variable[:,0,0,0])
    ny = len(variable[0,:,0,0])
    nt = len(variable[0,0,:,0])
    booli=bool;cell_variable = variable;
    test=0; 
    variable_cells = []
    for k in range(0,nt):
        for j in range(0,ny):
            for i in range(0,nx):
                if booli[i,j,k]==0: 
                    oldcell=False;oi=[];oj=[];ok=[];
                    new_cell=cell(cell_number) 
                    identify_elements_in_cell(i,j,k,new_cell)
                    ii=oi;jj=oj;kk=ok;
                    while oldcell:
                        oldcell=False;oi=[];oj=[];ok=[];
                        test=test+1
                        new_cell.nelements_local = 0;
                        for n in range(0,len(ii)):
                            if new_cell.nelements_local<nmaxelems:
                                identify_elements_in_cell(ii[n],jj[n],kk[n],new_cell)
                            else:
                                oi.append(ii[n]);oj.append(jj[n]);ok.append(kk[n]);
                                oldcell=True;
                        ii=oi;jj=oj;kk=ok;              
                    if new_cell.nelements>=nminelems:
                        variable_cells.append(new_cell)
                        variable_cells[cell_number].id = cell_number
                        cell_number = cell_number + 1
                        if variable_cells[cell_number-1].nelements>print_nelements:
                            print("Identifying elements of Cell:",variable_cells[cell_number-1].id)
                    else:
                        del new_cell;
                    
    print("Number of While Recursions:",test)
    return variable_cells, cell_number;
##################################################################################################
def insert_cell(cells,new_cell,variable_number):
    new_cell.id - variable_number
    cells.append(new_cell)
    for n_cells in range(0,len(cells)):
        if cells[n_cells].id >= variable_number:
            cells[n_cells].id = cells[n_cells].id + 1;
##################################################################################################
def fill_parent_array(cells,nx,ny,nt):                                     #fills parent_array variable with id numbers of cells corresponding to each grid point 
    parent_array=np.zeros((nx,ny,nt))                             # fills base and up with values of base and up at those grid points
    base=np.zeros((nx,ny,nt))
    up=np.zeros((nx,ny,nt))
    for c in range(0,len(cells)):
        for n in range(0,cells[c].nelements):
            parent_array[cells[c].location[0][n],cells[c].location[1][n],cells[c].location[2][n]]\
                        = cells[c].id + 1
            base[cells[c].location[0][n],cells[c].location[1][n],cells[c].location[2][n]]\
                        = cells[c].value[ibase][n]
            up[cells[c].location[0][n],cells[c].location[1][n],cells[c].location[2][n]]\
                        = cells[c].value[iup][n]
    return parent_array,base,up;
#################################################################################################
def find_parent(cells,parent_cells,parent_array,base,up):
    lnewparent=False
    for n in range(0,len(cells)):
        for nn in range(0,cells[n].nelements):
            i = cells[n].location[0][nn]
            j = cells[n].location[1][nn]
            t = cells[n].location[2][nn]
            if parent_array[i,j,t] > 0:
                if base[i,j,t] <= cells[n].value[iup][cells[n].nelements-1] and up[i,j,t] >= cells[n].value[ibase][cells[n].nelements-1]:
                    lnewparent = True
                    for np in range(0,cells[n].nparents):
                        if cells[n].nparents>0 and (cells[n].parents[np].id == parent_array[i,j,t]-1):
                            lnewparent=False
                    if lnewparent:
                        cells[n].parents.append(parent_cells[int(parent_array[i,j,t]-1)])
                        parent_cells[int(parent_array[i,j,t]-1)].children.append(cells[n])                       
                        cells[n].nparents = cells[n].nparents + 1
                        parent_cells[int(parent_array[i,j,t]-1)].nchildren = \
                        parent_cells[int(parent_array[i,j,t]-1)].nchildren + 1
    return cells,parent_cells;
#################################################################################################
def splitcell(cells, parent_cells, parent_array):
    
    global booli;
    for n_cells in range(0,len(cells)):
        list_var = np.zeros((4,cells[n_cells].nelements)); # contains indices i*j*t*n_splits for each element
        endlist = np.zeros((4,cells[n_cells].nelements));  # contains for all elements consequtively
        newlist = np.zeros((4,cells[n_cells].nelements));  #
        old_cell=cells[n_cells];
        nlist = 0;                                         # number of splits for each element
        nnewlist = 0; 
        newlist=0; 
        nendlist = 0;
        cbstep = 300;
        for n_elems in range(0,cells[n_cells].nelements):
            i=cells[n_cells].location[0][n_elems]; j=cells[n_cells].location[1][n_elems]; t=cells[n_cells].location[2][n_elems];
            if parent_array[i,j,t]!=0:
                lnewparent=True
                for n_spl in range(0,cells[n_cells].nsplitters):
                    if cells[n_cells].splitters[n_spl].parents[0].id==parent_array[i,j,t]:
                        lnewparent=False
                        break;
                if (lnewparent and (cells[n_cells].nsplitters>=1)):
                    if (parent_cells[parent_array[i,j,t]].nelements < minparentelems):
                        lnewparent = False
                    elif cells[n_cells].splitters[0].parents[0].nelements < minparentelems:
                        cells[n_cells].nsplitters=0;
                        parent_cells[parent_array[i,j,t]].nchildren = 0;
                if lnewparent:
                    cells[n_cells].splitters[cells[n_cells].nsplitters-1].parents.append(parent_cells[parent_array[i,j,t]])
                    parent_cells[parent_array[i,j,t]].nchildren = parent_cells[parent_array[i,j,t]].nchildren + 1;
                    parent_cells[parent_array[i,j,t]].children[parent_cells[parent_array[i,j,t]].nchildren].parents.append(cells[n_cells])
        if old_cell.nsplitters >= 2:
            nr = np.zeros((old_cell.nesplitters))
            for n_old in range(0,old_cell.nelements):
                i=cells[n_cells].location[0][n_old]; j=cells[n_cells].location[1][n_old]; t=cells[n_old].location[2][n_old];
                if parent_array[i,j,t]!=0:
                    for n_spl in range(0,old_cell.nsplitters):
                        if cells[n_cells].splitters[n_spl].parents[0].id==parent_array[i,j,t]:
                            nlist=nlist+1;
                            booli[i,j,t] = n_spl;
                            list_var[0,nlist] = i;
                            list_var[1,nlist] = j;
                            list_var[2,nlist] = t;
                            list_var[3,nlist] = n_spl;
                            nr[n_spl] = nr[n_spl] + 1                #indicator for how many ijts per splitter 
                            break
                continue
            endlist[:,nendlist:nendlist+nlist] = list_var[:,0:nlist]
            nendlist = nendlist + nlist 
            max_iteration = 1000/x_spacing
            for iteration in range(0,max_iteration):
                if nlist == 0:
                    exit
                nnewlist = 0;
                for n_nlist in range(0,nlist):
                    i=list_var[0,n_nlist]; j=list_var[1,n_nlist]; t=list_var[2,n_nlist];

                    ii=i-1; 
                    if ii<0:
                        ii = nx-1
                    if booli[ii,j,t]==-2 and (cell_variable[ii,j,t,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[ii,j,t] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1

                    ii=i+1; 
                    if ii>nx-1:
                        ii = 0
                    if booli[ii,j,t]==-2 and (cell_variable[ii,j,t,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[ii,j,t] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1    

                    jj=j-1; 
                    if jj<0:
                        jj = ny-1
                    if booli[i,jj,t]==-2 and (cell_variable[i,jj,t,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[i,jj,t] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1 

                    jj=j+1; 
                    if jj>ny-1:
                        jj = 0
                    if booli[i,jj,t]==-2 and (cell_variable[i,jj,t,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[i,jj,t] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1 

                    tt=min(t+1,nt-1);
                    if booli[i,j,tt]==-2 and (cell_variable[i,j,tt,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[i,j,tt] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1 

                    tt=max(t-1,0);
                    if booli[i,j,tt]==-2 and (cell_variable[i,j,tt,ibase] < cell_variable[i,j,t,ibase] + cbstep):
                        booli[i,j,tt] = -1;
                        nnewlist = nnewlist + 1
                        newlist[0,nnewlist] = ii; newlist[1,nnewlist] = j; newlist[2,nnewlist] = t;
                        newlist[3,nnewlist] = list_var[3,n_nlist]
                        nr[list_var[3,n_nlist]] = nr[list_var[3,n_nlist]] + 1  
                nlist = nnewlist
                list_var[:,0:nnewlist-1] = newlist[:,0:nnewlist-1]
                endlist[:,nendlist:nendlist+nlist-1] = list_var[:,0:nlist-1]
                nendlist = nendlist + nlist
            if nendlist < old_cell.nelements:
                lpassive = False
                for n_old in range(0,old_cell.nelements):
                    i=cells[n_cells].location[0][n_old]; j=cells[n_cells].location[1][n_old]; t=cells[n_old].location[2][n_old];
                    if booli[i,j,t]== -2:
                        booli[i,j,t]= -1
                        nendlist = nendlist + 1
                        nr[old_cell.nsplitters] = nr[old_cell.nsplitters] + 1
                        endlist[0,nendlist] = i; endlist[1,nendlist] = j; endlist[2,nendlist] = t;
                        endlist[3,nendlist] = old_cell.nsplitters
            else:
                lpassive = False
            newcells = np.zeros((old_cell.nsplitters)) 
            for n_old in range(0,old_cell.nsplitters):
                new_cell = cell(old_cell.id + n_old + 1)
                a=0;

#################################################################################################
def run_tracking(tracked_param,param_threshold):
    global nx,ny,nt,x,y,t;
    [tracked_variable,x,y,t] = retrieve_variable(tracked_param,nc_store_path+nc_filename);
    nx = len(tracked_variable[:,0,0,0]); ny = len(tracked_variable[0,:,0,0]); 
    nt = len(tracked_variable[0,0,:,0]);

    if tracked_param=='bc':
        bool = find_boolean(tracked_variable,param_threshold,t_start,True)
    else:
        bool = find_boolean(tracked_variable,param_threshold,t_start,False)
    
    [cells,cell_number] = create_new_cell(tracked_variable,bool);
    print("Number of Cells in ",tracked_param,":",len(cells))
    return cells,cell_number;
##################################################################################################
def make_animation_cells(x,y,all_cells,nt,minimum_elements,p_c):
    fig, ax = pl.subplots()
    cell_plot=np.zeros((len(x),len(y),nt));
    cell_parent_plot=np.zeros((len(x),len(y),nt));
    cell_children_plot=np.zeros((len(x),len(y),nt));
    count=0
    cells=[]
    for i in range(0,len(all_cells)):
        if all_cells[i].nelements>minimum_elements:
            cells.append(all_cells[i])
            count=count+1
    rand_series=np.random.randint(count, size=(count))
    for i in range (0,len(cells)):     
        for n in range(0,cells[i].nelements):      
            cell_plot[cells[i].location[0][n],cells[i].location[1][n],cells[i].location[2][n]] = rand_series[i]
    if p_c=='p' or p_c=='c_p':
        for i in range (0,len(cells)):
            if cells[i].nparents>0:
                for npar in range(0,cells[i].nparents):
                    for n in range(0,cells[i].parents[npar].nelements):      
                        cell_parent_plot[cells[i].parents[npar].location[0][n],cells[i].parents[npar].location[1][n],(cells[i].parents[npar].location[2][n])] = rand_series[i]
    if p_c=='c'or p_c=='c_p':
        for i in range (0,len(cells)):
            if cells[i].nchildren>0:
                for nchild in range(0,cells[i].nchildren):
                    for n in range(0,cells[i].children[nchild].nelements):      
                        cell_children_plot[cells[i].children[nchild].location[0][n],cells[i].children[nchild].location[1][n],(cells[i].children[nchild].location[2][n])] = rand_series[i]
    if count>nmaxcells:
        levels=np.linspace(0,count,(count+1)/math.ceil(count/nmaxcells))
    else:
        levels=np.linspace(0,count,count+1)
    def animate(i):
        ax.clear()
        if p_c=='p':
            ax.contourf(x,y,cell_parent_plot[:,:,i],levels=levels,cmap=cm.tab20b)
            ax.contour(x,y,cell_plot[:,:,i],levels=levels,cmap=cm.tab20b,linestyles='solid',linewidth=1.5)
        elif p_c=='c':
            ax.contourf(x,y,cell_plot[:,:,i],levels=levels,cmap=cm.tab20b)
            ax.contour(x,y,cell_children_plot[:,:,i],levels=levels,cmap=cm.tab20b,linestyles='solid')
        elif p_c=='':
            ax.contour(x,y,cell_plot[:,:,i],levels=levels,cmap=cm.flag)
        elif p_c=='c_p':
            ax.contourf(x,y,cell_plot[:,:,i],levels=levels,cmap=cm.tab20b)
            ax.contour(x,y,cell_children_plot[:,:,i],levels=levels,cmap=cm.tab20b,linestyles='solid')
            ax.contour(x,y,cell_parent_plot[:,:,i],levels=levels,cmap=cm.tab20b,linestyles='dashed')   
        return ax;
    ani = manimation.FuncAnimation(fig,animate,frames=nt,repeat=True)
    pl.show()
    HTML(ani.to_jshtml())
    pl.rcParams["animation.html"] = "jshtml"
    return ani
##################################################################################################
def make_animation_contour(x,y,variable,nt):
    fig, ax = pl.subplots()
    def animate(i):
        ax.clear()
        CS = ax.contour(x,y,variable[:,:,i])
        ax.clabel(CS, inline=True, fontsize=10)
        return ax;
        
    ani = manimation.FuncAnimation(fig,animate,frames=nt,repeat=False)
    pl.show()
    HTML(ani.to_jshtml())
    pl.rcParams["animation.html"] = "jshtml"
    return ani
#################################################################################################
def main_fun():
    global rain,thermal,core,cloud;

    [rain,rain_ncells]= run_tracking('qr',qr_threshold)
    
    [cloud,cloud_ncells] = run_tracking('ql',ql_threshold)
    
    print('Finding parent for Rain')
    [parent_array,base,up] = fill_parent_array(cloud,nx,ny,nt)
    [rain,cloud] = find_parent(rain,cloud,parent_array,base,up)
    
    print('Tracking thermals')
    [thermal,thermal_ncells] = run_tracking('couvreux',couv_threshold)
    
    print('Finding parent for Cloud')
    del parent_array,base,up;
    [parent_array,base,up] = fill_parent_array(thermal,nx,ny,nt)
    [cloud,thermal] = find_parent(cloud,thermal,parent_array,base,up);

    #[core,core_ncells,x,y,t] = run_tracking('bc',bc_threshold)


    array_description = ['x','y','t','path','top','base','cell_id','parent id','children id']
    rain_array = converttoarray_cell(rain);
    cloud_array = converttoarray_cell(cloud);
    thermal_array = converttoarray_cell(thermal)
    #core_array = converttoarray_cell(core)
    del rain,cloud,thermal;


    print('Writing arrays to netcdf')
    writeout_netcdf(cloud_array,[],rain_array,thermal_array,array_description)
    del rain_array,cloud_array,thermal_array;

    #pl.figure()
    #plot1=[]
    #for i in range(0,len(cloud)):
    #    plot1.append(cloud[i].nelements)
    #pl.plot(plot1)
    
    
    
    

    #with open(nc_save_path+nc_output_filename,'wb') as outp:
    #    pickle.dump(cloud,outp,pickle.HIGHEST_PROTOCOL)
        #pickle.dump(core,outp,pickle.HIGHEST_PROTOCOL)
    #    pickle.dump(rain,outp,pickle.HIGHEST_PROTOCOL)
    #    pickle.dump(thermal,outp,pickle.HIGHEST_PROTOCOL)

    #global ani1,ani2,ani3;
    #ani1 = make_animation_cells(x,y,cloud,nt,5e5,'c')
    #cloud_bool = np.where(cloud_path[:,:,:,0]>ql_threshold,1,0)
    #ani2 = make_animation_contour(x,y,cloud_bool,nt)
    #ani3 = make_animation_cells(x,y,cloud,nt,500,'c')
    
###############################################################################################

nc_store_path='/data/rico_cloudtrack/cloudtrack_inputnc/'
nc_filename='cloudtrack_variables_1.nc'
nc_save_path='/data/rico_cloudtrack/cloudtrack_outputnc/'
nc_output_filename='savefile_1.nc'
ipath=0; iup=1; ibase=2
multiplying_factor=1;
qr_threshold = 0.005* multiplying_factor; ql_threshold = 0.08 * multiplying_factor;
couv_threshold = 0.05 * multiplying_factor; bc_threshold = 1 * multiplying_factor;
nminelems=5;nmaxelems=3000;nmaxcells=1500;minparentelems=25;
x_spacing=25;recursion_point=2;

t_start = 0; t_plot=7;print_nelements = 2e6;
sys.setrecursionlimit(20000)

#make_netcdffile_fromcross(24060,84000,60,x_spacing,nc_store_path,nc_filename,True)  
main_fun()
#f = r"/data/rico_cloudtrack/cloudtrack_animations/animation_1024_84k.gif" 
#writergif = manimation.PillowWriter(fps=5) 
#ani1.save(f, writer=writergif)

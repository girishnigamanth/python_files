import matplotlib.pyplot as pl
import xarray as xr
import numpy as np
import seaborn as sns

model='MHH'
hr_plot=54


pl.rcParams['xtick.labelsize'] = 16
pl.rcParams['ytick.labelsize'] = 16
pl.rcParams['axes.labelsize'] = 16
#levs=np.concatenate((np.linspace(0,0.01,20),np.linspace(0.011,1,1)))


def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)
print(powspace(0,1.861e-3,2,10))

if model=='MHH':
    MHH_qlbase=xr.open_dataset('/fs/ess/PFS0220/eurec4a/microhh_results/2D_outputs/Feb_2nd/ql_base.xy.nc',decode_times=False)
    MHH_qltop=xr.open_dataset('/fs/ess/PFS0220/eurec4a/microhh_results/2D_outputs/Feb_2nd/ql_top.xy.nc',decode_times=False)
    MHH_qlbase['ql_top'] = MHH_qlbase['ql_base'].copy()
    MHH_qlbase['ql_top'].values=MHH_qltop['ql_top'].values

    MHH_qlbase_stacked=MHH_qlbase.stack(xy=('x','y'))
    MHH_qlbase_stacked=MHH_qlbase_stacked.where(MHH_qlbase_stacked['ql_base']>0,np.nan)

    
    df=MHH_qlbase_stacked.isel(time=slice(hr_plot*12,(hr_plot+28)*12,24)).to_dataframe()

    g=sns.displot(df,x="ql_base",y="ql_top",kind='kde',fill=False,levels=10,cbar=False,col='time',col_wrap=4)
    pl.xlim(500,2500)
    pl.ylim(500,3000)
    g.set_axis_labels('Cloud Base $(m)$', 'Cloud Top $(m)$')
    g.set_titles('')
    pl.tight_layout()
    pl.show()
    pl.savefig('/users/PFS0220/graghuna/Flower_MIP_Paper/Plots/Cloud_Top_base_Feb2nd_MHH_54hr.jpg',dpi=300, bbox_inches='tight')
elif model=='MONC':
    MONC=xr.open_dataset('/fs/ess/PFS0220/eurec4a/MONC_results/d20200202_diagnostic_2d.nc',decode_times=False)
    MONC_stacked=MONC[['cltop','clbas']].stack(xy=('x','y'))
    MONC_stacked=MONC_stacked.where(MONC_stacked['clbas']>0,np.nan)
    df=MONC_stacked.isel(time=slice(hr_plot*2,(hr_plot+28)*2,4)).to_dataframe()
    g=sns.displot(df,x="clbas",y="cltop",kind='kde',fill=False,levels=10,cbar=False,col='time',col_wrap=7)
    pl.xlim(500,2500)
    pl.ylim(500,3000)
    g.set_axis_labels('Cloud Base $(m)$', 'Cloud Top $(m)$')
    g.set_titles('')
    pl.tight_layout()
    pl.show()
    pl.savefig('/users/PFS0220/graghuna/Flower_MIP_Paper/Plots/Cloud_Top_base_Feb2nd_MONC.jpg',dpi=300, bbox_inches='tight')
elif model=='SAM_UW':
    SAM=xr.open_dataset('/fs/ess/PFS0220/eurec4a/SAM_UW_results/Feb2/EUREC4A_2Feb_1024sqx151_150m_M2005_RRTM_Lagtraj_NC400_km_1024.2Dbin_1.nc',decode_times=False)
    SAM_stacked=SAM[['ZC','LWP']].stack(xy=('x','y'))
    SAM_stacked=SAM_stacked.where(SAM_stacked['ZC']>0,np.nan)
    SAM_stacked['ZC']=SAM_stacked['ZC']*1000
    df=SAM_stacked.isel(time=slice(hr_plot*12,(hr_plot+28)*12,24)).to_dataframe()
    g=sns.displot(df,x="LWP",y="ZC",kind='kde',fill=False,levels=10,cbar=False,col='time',col_wrap=7)
    #pl.xlim(500,2500)
    pl.ylim(500,3000)
    g.set_axis_labels('Mixed-Layer Height $(m)$', 'Cloud Top $(m)$')
    g.set_titles('')
    pl.tight_layout()
    pl.show()
    pl.savefig('/users/PFS0220/graghuna/Flower_MIP_Paper/Plots/Cloud_Top_LWP_Feb2nd_SAM_UW.jpg',dpi=300, bbox_inches='tight')
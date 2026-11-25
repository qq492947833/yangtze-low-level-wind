"""Temporal-scale wind prediction errors at the 25 fixed Yangtze grid points."""
from pathlib import Path
import os

import numpy as np
from netCDF4 import Dataset, num2date

DATA = Path(os.environ.get("YANGTZE_DATA_DIR","data"))
RESULTS = Path(os.environ.get("YANGTZE_RESULTS_DIR","results"))
OUT = RESULTS / "task2_diagnostics" / "mechanism_v4_20261028" / "vmd"
SC = Path(os.environ.get("YANGTZE_WORK_DIR","work")) / "mechanisms" / "vmd"
LI = np.array([0,12,24,36,48])
LJ = np.array([0,18,36,54,72])
HOUR = 3600000000000
SUFFIX = {"A":"","B":"_residual","C":"_vmd","D":"_vmd_residual"}
RAW_NAME = {"u":"U-component-of-wind","v":"V-component-of-wind","w":"Vertical-velocity"}
SCALES=['total','bias_squared','2_to_8h','8_to_24h','24_to_168h','168_to_720h','over_720h']
STAGES=['stage1','final'];COMP=['u','v','omega'];GROUPS=['A','B','C','D'];BLOCKS=[72,168]
SEED=20261028;REPS=1000

def times(ds):
 v=ds.variables['time'];v.set_auto_mask(False);a=np.asarray(v[:]);units=v.units;cal=getattr(v,'calendar','standard')
 vals=np.asarray(num2date(a,units,calendar=cal,only_use_cftime_datetimes=False,only_use_python_datetimes=True),dtype='datetime64[us]').astype('datetime64[ns]').astype(np.int64)
 rounded=np.rint(vals/HOUR).astype(np.int64)*HOUR
 if np.max(np.abs(vals-rounded))>1000:raise ValueError('time encoding off hourly grid by >1000ns')
 if not np.all(np.diff(rounded)>0):raise ValueError('nonunique or decreasing times')
 return rounded


def forecast_path(group, kind, lead, comp):
    return RESULTS / f"da_transunet_{kind}_{comp}_{lead}_24year_changjiang{SUFFIX[group]}.nc"


def read_pair(group,lead,comp):
    arrays = {}
    for kind, key in (("testy","truth"),("predicty","pred")):
        with Dataset(forecast_path(group,kind,lead,comp)) as ds:
            ds.set_auto_mask(False)
            t = times(ds)
            v = ds[f"{kind}_{comp}"]
            if v.dimensions != ("time","level","latitude","longitude"):
                raise ValueError(f"Unexpected dimensions for {kind}_{comp}")
            np.testing.assert_array_equal(ds["level"][:],[850,925,1000])
            for name,reference in (("latitude",MAP["latitude"]),("longitude",MAP["longitude"])):
                np.testing.assert_allclose(ds[name][:],reference,rtol=0,atol=1e-6)
            if "time_ns" in arrays:
                np.testing.assert_array_equal(t,arrays["time_ns"])
            arrays["time_ns"] = t
            field = np.empty((len(t),3,5,5),np.float64)
            for start in range(0,len(t),512):
                stop = min(start+512,len(t))
                field[start:stop] = np.asarray(v[start:stop,:,LI,LJ],np.float64) * (-10. if comp=="w" else 1.)
            arrays[key] = field
    return arrays


def common_bounds():
    begins, ends = [], []
    for group in GROUPS:
        for lead in range(1,7):
            for comp in "uvw":
                with Dataset(forecast_path(group,"testy",lead,comp)) as ds:
                    ts = times(ds)
                    begins.append(ts[0])
                    ends.append(ts[-1])
    return max(begins),min(ends)


def raw_reference(comp,target):
    by_level = []
    for level in (850,925,1000):
        with Dataset(DATA / f"{RAW_NAME[comp]}-{level}hpa.nc") as ds:
            ds.set_auto_mask(False)
            src_times = times(ds)
            idx = np.searchsorted(src_times,target)
            np.testing.assert_array_equal(src_times[idx],target)
            lat = np.asarray(ds["latitude"][:])
            lon = np.asarray(ds["longitude"][:])
            def coordinate_indices(axis,points):
                indices = []
                for p in points:
                    matches = np.flatnonzero(np.isclose(axis,p,rtol=0,atol=1e-6))
                    if len(matches)!=1:
                        raise ValueError(f"Cannot match ERA5 coordinate {p}")
                    indices.append(matches[0])
                return np.asarray(indices)
            iy = coordinate_indices(lat, MAP["latitude"][LI])
            ix = coordinate_indices(lon, MAP["longitude"][LJ])
            field = np.empty((len(idx),5,5),np.float64)
            var = ds[comp]
            for start in range(0,len(idx),512):
                stop = min(start+512,len(idx))
                field[start:stop] = var[idx[start:stop],iy,ix]
            by_level.append(field)
    return np.stack(by_level,axis=1)

def runs(mask,t):
 ids=np.flatnonzero(mask)
 if not len(ids):return []
 cuts=np.flatnonzero((np.diff(ids)!=1)|(np.diff(t[ids])!=HOUR))+1
 return np.split(ids,cuts)

def freq_masks(n):
 f=np.fft.rfftfreq(n,1.0)
 return [f>=1/8,(f>=1/24)&(f<1/8),(f>=1/168)&(f<1/24),(f>=1/720)&(f<1/168),(f>0)&(f<1/720)]

def decompose(errors, own_errors, valid, t):
 # errors: time, configuration, level, fixed point. All four groups share validity.
 n,ncfg,nlev,npoint=errors.shape
 hourly=np.zeros((n,ncfg,nlev,7),np.float64);count=np.zeros((n,nlev),np.int32)
 own_sum=np.zeros((ncfg,nlev,7),np.float64)
 segment_count=np.zeros((nlev,npoint),np.int32);min_segment=np.full((nlev,npoint),n,np.int64)
 maxclosure=0.0
 for lev in range(nlev):
  for point in range(npoint):
   for ids in runs(valid[:,lev,point],t):
    z=errors[ids,:,lev,point];zo=own_errors[ids,:,lev,point];m=len(ids)
    segment_count[lev,point]+=1;min_segment[lev,point]=min(min_segment[lev,point],m);count[ids,lev]+=1
    zz=np.mean(z,axis=0);zoz=np.mean(zo,axis=0)
    f=np.fft.rfft(z-zz,axis=0);fo=np.fft.rfft(zo-zoz,axis=0)
    w=np.full(len(f),2.0);w[0]=1.0
    if m%2==0:w[-1]=1.0
    h=np.zeros((m,ncfg,7));h[:,:,0]=z*z;h[:,:,1]=zz[None,:]**2
    own_sum[:,lev,0]+=np.sum(zo*zo,axis=0);own_sum[:,lev,1]+=m*zoz**2
    for bi,mask in enumerate(freq_masks(m)):
     filtered=np.fft.irfft(f*mask[:,None],n=m,axis=0)
     h[:,:,bi+2]=filtered**2
     own_sum[:,lev,bi+2]+=np.sum(w[:,None]*np.abs(fo)**2*mask[:,None],axis=0)/m
    closure=np.max(np.abs(np.mean(h[:,:,0],axis=0)-np.mean(h[:,:,1:].sum(axis=2),axis=0)))
    maxclosure=max(maxclosure,float(closure));hourly[ids,:,lev,:]+=h
 den=count.sum(axis=0)
 means=hourly.sum(axis=0)/den[None,:,None]
 own=own_sum/den[None,:,None]
 return hourly,count,means,own,segment_count,min_segment,maxclosure

def bootstrap(hourly,count,t):
 # paired nonoverlapping time blocks, shared replicate weights for all contrasts.
 # Only total and periods <=168h: no formal interval for bias or >168h scales.
 ci=np.full((2,3,7,2,2),np.nan)
 delta=hourly[:,[0,1]]-hourly[:,[2,3]]
 allowed=[0,2,3,4]
 for bi,b in enumerate(BLOCKS):
  blocks=[]
  for seg in runs(np.ones(len(t),bool),t):
   for start in range(0,len(seg),b):blocks.append(seg[start:start+b])
  sums=np.stack([delta[idx].sum(axis=0) for idx in blocks]);ns=np.stack([count[idx].sum(axis=0) for idx in blocks])
  rng=np.random.default_rng(SEED+b)
  draws=np.empty((REPS,2,3,7),np.float64)
  # bounded batches avoid a large sampled-time array.
  for r0 in range(0,REPS,50):
   k=min(50,REPS-r0);w=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=k)
   a=np.tensordot(w,sums,axes=(1,0));den=w@ns
   draws[r0:r0+k]=a/den[:,None,:,None]
  q=np.percentile(draws,[2.5,97.5],axis=0).transpose(1,2,3,0)
  for scale in allowed:ci[:,:,scale,bi,:]=q[:,:,scale,:]
 return ci

def compute_case(lead,comp):
 dest=SC/f'h{lead}_{comp}.npz'
 data={g:read_pair(g,lead,comp) for g in GROUPS}
 t=data['A']['time_ns'];t=t[(t>=START)&(t<=END)]
 for g in 'BCD':t=np.intersect1d(t,data[g]['time_ns'],assume_unique=True)
 y=np.stack([data[g]['truth'][np.searchsorted(data[g]['time_ns'],t)].reshape(len(t),3,25) for g in GROUPS],axis=1)
 p=np.stack([data[g]['pred'][np.searchsorted(data[g]['time_ns'],t)].reshape(len(t),3,25) for g in GROUPS],axis=1)
 raw=raw_reference(comp,t).reshape(len(t),3,25)
 valid=np.all(np.isfinite(y)&np.isfinite(p),axis=1)&np.isfinite(raw)
 dif=np.max(np.where(valid[:,None],np.abs(y-raw[:,None]),0),axis=(0,2,3))
 errors=p-raw[:,None];own_errors=p-y
 hourly,count,mse,own,segments,minseg,closure=decompose(errors,own_errors,valid,t)
 ci=bootstrap(hourly,count,t)
 np.testing.assert_allclose(mse[:,:,0],mse[:,:,1:].sum(axis=2),rtol=2e-12,atol=2e-12)
 np.testing.assert_allclose(own[:,:,0],own[:,:,1:].sum(axis=2),rtol=2e-12,atol=2e-12)
 gain=mse[[0,1]]-mse[[2,3]]
 own_gain=own[[0,1]]-own[[2,3]]
 correction=mse[[0,2]]-mse[[1,3]]
 with dest.with_name(dest.name+'.tmp').open('wb') as f:np.savez_compressed(f,time_ns=t,mse=mse,own_mse=own,gain=gain,own_gain=own_gain,correction_gain=correction,ci=ci,hourly_band_sse=hourly.astype(np.float32),hourly_count=count,segment_count=segments,min_segment_length=minseg,truth_max_abs_difference=dif,common_count=count.sum(axis=0),max_parseval_closure=closure)
 os.replace(dest.with_name(dest.name+'.tmp'),dest)

def write_nc():
 file=OUT/'VMD_mechanism_timescale.nc';tmp=file.with_name(file.name+'.tmp')
 shape=(6,4,3,3,7);mse=np.full(shape,np.nan);own=mse.copy();gain=np.full((6,2,3,3,7),np.nan);own_gain=gain.copy();correction=gain.copy();ci=np.full((6,2,3,3,7,2,2),np.nan);counts=np.zeros((6,3,3),np.int64);truthdiff=np.full((6,3,4),np.nan);closure=np.full((6,3),np.nan);availability=np.zeros((6,3),np.int8)
 for lead in range(1,7):
  for comp_i,comp in enumerate(['u','v','w']):
   npz=SC/f'h{lead}_{comp}.npz'
   with np.load(npz,allow_pickle=False) as z:
    mse[lead-1,:,comp_i]=z['mse'];own[lead-1,:,comp_i]=z['own_mse'];gain[lead-1,:,comp_i]=z['gain'];own_gain[lead-1,:,comp_i]=z['own_gain'];correction[lead-1,:,comp_i]=z['correction_gain'];ci[lead-1,:,comp_i]=z['ci'];counts[lead-1,comp_i]=z['common_count'];truthdiff[lead-1,comp_i]=z['truth_max_abs_difference'];closure[lead-1,comp_i]=z['max_parseval_closure'];availability[lead-1,comp_i]=1
 with Dataset(tmp,'w',format='NETCDF4') as ds:
  for n,size in [('lead',6),('configuration',4),('stage',2),('vmd_configuration',2),('component',3),('level',3),('scale',7),('block_hours',2),('ci_bound',2),('sample_latitude',5),('sample_longitude',5)]:ds.createDimension(n,size)
  def var(name,dims,values,attrs=None):
   a=np.asarray(values)
   if a.dtype.kind in 'OUS':v=ds.createVariable(name,str,dims);v[:]=np.asarray(values,dtype=object)
   else:v=ds.createVariable(name,'i8' if a.dtype.kind in 'iu' else 'f8',dims,zlib=True,complevel=4);v[:]=a
   if attrs:v.setncatts(attrs)
   return v
  var('lead',('lead',),range(1,7),{'units':'hour'});var('configuration',('configuration',),['noVMD_stage1','noVMD_final','VMD_stage1','VMD_final']);var('stage',('stage',),STAGES);var('vmd_configuration',('vmd_configuration',),['noVMD','VMD']);var('component',('component',),COMP);var('level',('level',),[850,925,1000],{'units':'hPa'});var('scale',('scale',),SCALES);var('block_hours',('block_hours',),BLOCKS,{'units':'hour'});var('ci_bound',('ci_bound',),['p025','p975']);var('sample_latitude',('sample_latitude',),np.asarray(MAP['latitude'])[LI],{'units':'degrees_north'});var('sample_longitude',('sample_longitude',),np.asarray(MAP['longitude'])[LJ],{'units':'degrees_east'});var('component_squared_error_units',('component',),['m2 s-2','m2 s-2','Pa2 s-2'])
  md=('lead','configuration','component','level','scale');gd=('lead','stage','component','level','scale')
  attrs={'aggregation':'pooled equal-time and equal-point MSE over the fixed25 spatial sample, per pressure level','reference':'original decoded ERA5 on common hourly targets; exact coordinates; no saved model testy substitution','scale_partition':'total = bias_squared + disjoint five spectral bands; exact within each contiguous finite segment; no Hann taper; zero-padding absent'}
  var('error_mse',md,mse,attrs);var('own_reference_error_mse',md,own,{'reference':'each configuration corresponding saved truth; same support as error_mse'})
  var('vmd_mse_reduction',gd,gain,{'sign':'noVMD minus VMD; positive improvement, negative deterioration'});var('own_reference_vmd_mse_reduction',gd,own_gain)
  var('residual_correction_mse_reduction',('lead','vmd_configuration','component','level','scale'),correction,{'sign':'stage1 minus final; positive reduction'})
  var('stage_improvement_sign_agreement',('lead','component','level','scale'),np.sign(gain[:,0])*np.sign(gain[:,1]),{'meaning':'+1 same direction; -1 opposite; 0 one estimate zero; NaN unavailable'})
  var('vmd_mse_reduction_ci',gd+('block_hours','ci_bound'),ci,{'interval':'95% percentile of 1000 paired nonoverlapping time-block resamples of already filtered error curves','conditioning':'fixed fitted models, reference, sample and FFT filters; no refitting, no seed replication; multiplicity not corrected'})
  var('ci_availability_reason',('scale',),['available_time_block','not_reported_DC_bias_requires_distinct_estimand','available_conditional_fixed_filter','available_conditional_fixed_filter','available_conditional_fixed_filter_block_may_be_shorter_than_band','not_reported_block_shorter_than_low_frequency_period','not_reported_block_shorter_than_low_frequency_period'])
  var('valid_sample_time_point_count',('lead','component','level'),counts);var('reference_max_absolute_difference_vs_raw',('lead','component','configuration'),truthdiff);var('max_parseval_closure',('lead','component'),closure);var('case_available',('lead','component'),availability)
  ds.setncatts({"time_alignment":"common native time span, followed by four-configuration timestamp intersection",
               "sample_scope":"fixed 5 x 5 grid points at three pressure levels",
               "spectral_estimator":"real DFT of demeaned errors within contiguous valid series, segment bias treated separately",
               "status":"COMPLETE"})
 os.replace(tmp,file)


if __name__=="__main__":
 OUT.mkdir(parents=True,exist_ok=True)
 SC.mkdir(parents=True,exist_ok=True)
 with Dataset(forecast_path("A","testy",1,"u")) as ds:
  MAP = {key:np.asarray(ds[key][:]) for key in ("latitude","longitude")}
 START,END = common_bounds()
 for lead in range(1,7):
  for comp in "uvw":
   compute_case(lead,comp)
 write_nc()

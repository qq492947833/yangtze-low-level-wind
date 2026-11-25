"""Shared accessors and temporal block sampling for the gradient mechanism analysis."""
from pathlib import Path
import os
WORK = Path(os.environ.get("YANGTZE_WORK_DIR","work")).resolve()
ROOT = WORK / "mechanisms" / "gradient"
RAW = Path(os.environ.get("YANGTZE_DATA_DIR","data")).resolve()
RESULT = Path(os.environ.get("YANGTZE_RESULTS_DIR","results")).resolve()
LEGACY = WORK / "gradient"
for p in (ROOT, ROOT/"chunks", ROOT/"receipts"):
    p.mkdir(parents=True, exist_ok=True)
import json, time, hashlib, tempfile, platform
import numpy as np
import xarray as xr
import psutil
from scipy.ndimage import uniform_filter, minimum_filter
COMPONENTS=['u','v','w']; LEVELS=np.array([850.,925.,1000.]); RADIUS=6371000.
SEED=20261025
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(1024*1024),b''):h.update(a)
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.partial');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8');os.replace(t,path)
def atomic_npz(path,**kw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.partial.npz');np.savez_compressed(t,**kw)
    with np.load(t,allow_pickle=False) as z: assert set(z.files)==set(kw)
    os.replace(t,path)
def publish_nc(ds,path):
    path=Path(path);tmp=path.with_suffix('.partial.nc')
    enc={k:{'zlib':True,'complevel':4,'shuffle':True} for k,v in ds.data_vars.items() if v.dtype.kind not in 'OUS'}
    ds.to_netcdf(tmp,engine='netcdf4',encoding=enc)
    with xr.open_dataset(tmp) as check: assert dict(check.sizes)==dict(ds.sizes)
    os.replace(tmp,path)
    return {'path':str(path),'size':path.stat().st_size,'sha256':sha(path)}
def memory_guard():
    v=psutil.virtual_memory(); rss=psutil.Process().memory_info().rss
    return {'time':now(),'rss':rss,'available':v.available,'total':v.total,'cpu_percent':psutil.cpu_percent(),'io':psutil.Process().io_counters()._asdict()}
def identity(paths):
    return [{'path':str(p),'size':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in paths]
def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def normalized_times(a):
    t=np.asarray(a).astype('datetime64[ns]');assert not np.isnat(t).any()
    n=t.astype('int64');unit=3600*10**9; nearest=np.rint(n.astype(np.float64)/unit).astype('int64')*unit
    if np.max(np.abs(n-nearest),initial=0)>1000: raise ValueError('Time is not an hour within strict 1000 ns codec tolerance')
    assert np.all(np.diff(nearest)>0),'Duplicate or unsorted time'
    return nearest.astype('datetime64[ns]')
class Field:
    def __init__(self,path,var=None):
        self.path=Path(path);self.ds=xr.open_dataset(path)
        if var is None:var=next(k for k,v in self.ds.data_vars.items() if {'time','latitude','longitude'}.issubset(v.dims))
        self.da=self.ds[var];self.times=normalized_times(self.ds.time.values)
        self.lat=self.ds.latitude.values;self.lon=self.ds.longitude.values
        self.level=self.ds.level.values if 'level' in self.ds.coords else None
    def read(self,times):
        i=np.searchsorted(self.times,times);assert (i<len(self.times)).all() and np.array_equal(self.times[i],times)
        a=self.da.isel(time=i)
        order=('time','level','latitude','longitude') if self.level is not None else ('time','latitude','longitude')
        return np.asarray(a.transpose(*order).values,dtype=np.float64)
    def close(self):self.ds.close()
def secant(a,x,axis):
    a=np.moveaxis(np.asarray(a,dtype=np.float64),axis,-1);x=np.asarray(x,dtype=np.float64)
    assert len(x)>1 and (np.all(np.diff(x)>0) or np.all(np.diff(x)<0))
    y=np.empty_like(a);y[...,0]=(a[...,1]-a[...,0])/(x[1]-x[0]);y[...,-1]=(a[...,-1]-a[...,-2])/(x[-1]-x[-2])
    y[...,1:-1]=(a[...,2:]-a[...,:-2])/(x[2:]-x[:-2]);return np.moveaxis(y,-1,axis)
def smooth(a):return uniform_filter(a,size=(1,)*(a.ndim-2)+(3,3),mode='nearest')
def partition_blocks(times,hours):
    """Nonoverlapping complete hourly blocks; never bridges a real gap. Last short block retained."""
    cut=np.r_[0,np.flatnonzero(np.diff(times)!=np.timedelta64(1,'h'))+1,len(times)]
    return [(a,min(a+hours,e)) for s,e in zip(cut[:-1],cut[1:]) for a in range(s,e,hours)]
def resampling_weights(times,hours,nrep=1000):
    blocks=partition_blocks(times,hours); rng=np.random.default_rng(SEED+hours)
    weights=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=nrep).astype(np.float64)
    return blocks,weights
def freeze_contract(name,body):
    p=ROOT/f'{name}_contract.json';body=dict(body,implementation=sha(__file__),schema=1);f=fingerprint(body)
    if p.exists():
        old=json.loads(p.read_text(encoding='utf8'));assert old['fingerprint']==f,'Source/config drift: do not reuse checkpoints'
    else:atomic_json(p,{'fingerprint':f,'contract':body,'created':now()})
    return f

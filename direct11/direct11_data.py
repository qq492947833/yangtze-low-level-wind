"""Prepare raw ERA5 arrays and per-lead training scalers for the public Direct11 runs."""
from pathlib import Path
import os
from contextlib import ExitStack
import numpy as np
import netCDF4 as nc
from direct11_common import ROOT, read, atomic, identity, assert_identity, sha, stamp, lock
RAW=Path(os.environ.get("YANGTZE_DATA_DIR","./data")).resolve()
SURFACE=[("Surface-pressure.nc","sp"),("2m-temperature.nc","t2m"),("2m-dewpoint-temperature.nc","d2m")]
WINDS=[(f"{name}-{level}hpa.nc",var) for name,var in [("U-component-of-wind","u"),("V-component-of-wind","v"),("Vertical-velocity","w")] for level in [850,925,1000]]
CACHE=ROOT/"cache"; PROV=ROOT/"provenance"

def raw_times(ds):
    v=ds.variables["time"]
    t=nc.num2date(v[:],v.units,getattr(v,"calendar","standard"),only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    return np.asarray(t,dtype="datetime64[ns]").astype("int64")

def training_stats(array, start, stop):
    """Population moments over each distinct raw training hour, independently per cell/channel."""
    n=0; mean=None; ss=None
    for lo in range(start,stop,64):
        a=np.asarray(array[lo:min(lo+64,stop)],np.float64)
        if not np.isfinite(a).all(): raise ValueError("Nonfinite values in input training hours")
        m=a.mean(0); c=len(a); s=np.sum((a-m)**2,axis=0)
        if n==0: mean=m; ss=s
        else:
            delta=m-mean
            ss+=s+delta**2*(n*c/(n+c)); mean+=delta*(c/(n+c))
        n+=c
    std=np.sqrt(ss/n)
    if not np.all(std>0): raise ValueError("Undefined training scale")
    return mean.astype(np.float32),std.astype(np.float32)

def prepare_cache():
    CACHE.mkdir(parents=True,exist_ok=True); PROV.mkdir(parents=True,exist_ok=True)
    with lock(CACHE/"prepare.lock"), ExitStack() as stack:
        paths=[RAW/name for name,var in SURFACE+WINDS]
        manifest=PROV/"DATA_MANIFEST.json"
        if manifest.exists():
            old=read(manifest)
            if [str(p) for p in paths]!=[r["path"] for r in old["sources"]]: raise ValueError("Use a fresh work directory for different inputs")
            for r in old["sources"]+old["products"]: assert_identity(r)
            return
        datasets=[stack.enter_context(nc.Dataset(p)) for p in paths]
        times=raw_times(datasets[0]); lat=np.asarray(datasets[0]["latitude"][:]); lon=np.asarray(datasets[0]["longitude"][:])
        if (len(lat),len(lon))!=(49,73) or len(times)<64 or not np.all(np.diff(times)==3600*10**9):raise ValueError("Expected consecutive hourly ERA5 on a 49 x 73 grid")
        for ds,(_,var) in zip(datasets,SURFACE+WINDS):
            if ds[var].dimensions!=("time","latitude","longitude") or not np.array_equal(times,raw_times(ds)) or not np.array_equal(ds["latitude"][:],lat) or not np.array_equal(ds["longitude"][:],lon):raise ValueError("Misaligned input: "+var)
        n=len(times)-6
        x=np.lib.format.open_memmap(CACHE/"surface_float32.npy","w+",np.float32,(n+6,49,73,3))
        y=np.lib.format.open_memmap(CACHE/"wind_storage_float32.npy","w+",np.float32,(n,49,73,9))
        for c,(ds,(_,var)) in enumerate(zip(datasets,SURFACE+WINDS)):
            a=x if c<3 else y; offset=0 if c<3 else 6; j=c if c<3 else c-3
            for lo in range(0,len(a),128):
                hi=min(len(a),lo+128); v=np.ma.asarray(ds[var][lo+offset:hi+offset],dtype=np.float64)
                if np.ma.getmaskarray(v).any() or not np.isfinite(v.data).all(): raise ValueError("Missing ERA5 values: "+var)
                a[lo:hi,:,:,j]=(-0.1*v.data if c>=9 else v.data).astype(np.float32)
            a.flush()
        z=dict(raw_time_ns=times[6:],full_surface_time_ns=times,latitude=lat,longitude=lon,level=np.array([850,925,1000],dtype=np.int32))
        products=[CACHE/"surface_float32.npy",CACHE/"wind_storage_float32.npy"]
        for h in range(1,7):
            # Original Direct6 windows end at aligned row 5 or later. Its test cut used
            # the unwindowed paired target length; extending the window must not move it.
            issue=np.arange(5,n-h,dtype=np.int64); cut=int((1-0.2)*(n-h))
            train_end=int((1-0.1/(1-0.2))*len(issue[:cut]))
            parts={"train":issue[:train_end],"valid":issue[train_end:cut],"test":issue[cut:]}
            for split,ii in parts.items():
                z[f"h{h}_{split}_issue_idx"]=ii; z[f"h{h}_{split}_target_idx"]=ii+h
            train=parts["train"]; vx_mean,vx_std=training_stats(x,train[0]+6-10,train[-1]+6+1)
            vy_mean,vy_std=training_stats(y,train[0]+h,train[-1]+h+1)
            norm=PROV/f"train_normalization_lead{h}.npz"
            np.savez(norm,vx_mean=vx_mean,vx_std=vx_std,vy_mean=vy_mean,vy_std=vy_std);products.append(norm)
        z["common_test_target_idx"]=z["h1_test_target_idx"]
        for h in range(2,7): z["common_test_target_idx"]=np.intersect1d(z["common_test_target_idx"],z[f"h{h}_test_target_idx"])
        np.savez(PROV/"direct11_indices.npz",**z);products.append(PROV/"direct11_indices.npz")
        atomic(manifest,dict(sources=[identity(p) for p in paths],products=[identity(p) for p in products],
                              normalization="training input hours and training target hours, per lead and grid cell",utc=stamp()))

class Data:
    def __init__(self,lead):
        self.lead=int(lead)
        if not 1<=self.lead<=6:raise ValueError("lead must be 1 to 6")
        info=read(PROV/"DATA_MANIFEST.json")
        for r in info["sources"]+info["products"]:assert_identity(r)
        with np.load(PROV/"direct11_indices.npz",allow_pickle=False) as z:self.z=dict(z)
        with np.load(PROV/f"train_normalization_lead{lead}.npz",allow_pickle=False) as z:self.norm=dict(z)
        self.x=np.load(CACHE/"surface_float32.npy",mmap_mode="r");self.y=np.load(CACHE/"wind_storage_float32.npy",mmap_mode="r")
        self.splits={s:self.z[f"h{lead}_{s}_issue_idx"] for s in ["train","valid","test"]}

    def batch(self,split,start,stop):return self.for_issues(self.splits[split][start:stop])
    def for_issues(self,issues):
        ii=np.asarray(issues,np.int64);tt=ii[:,None]+6+np.arange(-10,1)
        x=((self.x[tt]-self.norm['vx_mean'])/self.norm['vx_std']).transpose(0,1,4,2,3)
        y=((self.y[ii+self.lead]-self.norm['vy_mean'])/self.norm['vy_std']).transpose(0,3,1,2)
        return x,y
    def decode_predictions(self,prediction):
        """Match _all: float64 inverse scaling, one float32 stored-field cast, physical omega once."""
        q=np.asarray(prediction,np.float32).transpose(0,2,3,1).astype(np.float64)
        q=q*self.norm['vy_std'].astype(np.float64)+self.norm['vy_mean'].astype(np.float64)
        q=q.astype(np.float32).astype(np.float64);q[...,6:9]*=-10.0
        return q.transpose(0,3,1,2)
    def physical_truth(self,issues):
        ti=np.asarray(issues,np.int64)+self.lead+6;out=np.empty((len(ti),9,49,73),np.float64)
        assert np.all(np.diff(ti)==1) if len(ti)>1 else True
        for c,(name,var) in enumerate(WINDS):
            with nc.Dataset(RAW/name) as d:
                v=np.ma.asarray(d.variables[var][int(ti[0]):int(ti[-1])+1]);assert not np.ma.getmaskarray(v).any();out[:,c]=v
        assert np.isfinite(out).all();return out

if __name__=="__main__":
    prepare_cache()

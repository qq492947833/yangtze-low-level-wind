"""Read-only canonical ERA5 wind reference; no model normalization or stored-w scaling.
Coordinates are exact labels, time integer-normalized within existing <=1000ns codec tolerance.
Raw variables lack unit attrs: units below follow the verified original ERA5/label convention.
"""
from pathlib import Path
import numpy as np
import xarray as xr
from common import RAW
LEVELS=np.array([850.,925.,1000.],dtype=np.float64)
LABELS={'u':'U-component-of-wind','v':'V-component-of-wind','w':'Vertical-velocity'}
UNITS={'u':'m s-1','v':'m s-1','w':'Pa s-1'}

def normalized_times(values):
 t=np.asarray(values).astype('datetime64[ns]')
 if t.ndim!=1 or np.isnat(t).any():raise ValueError('Invalid time axis')
 n=t.astype(np.int64); hour=3600*10**9; nearest=((n+hour//2)//hour)*hour
 if np.max(np.abs(n-nearest),initial=0)>1000:raise ValueError('Non-hour beyond 1000ns codec tolerance')
 if np.any(np.diff(nearest)<=0):raise ValueError('Unsorted/duplicate times')
 return nearest.astype('datetime64[ns]')

def identity(path):
 p=Path(path);s=p.stat();return {'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}

class RawWindReference:
 def __init__(self,root=RAW):
  self.root=Path(root);self.datasets={};self.source_records=[];self.level=LEVELS.copy()
  self.lat=np.arange(36.,23.99,-.25);self.lon=np.arange(108.,126.01,.25)
  self.latitude=self.lat;self.longitude=self.lon;self.time=None
  try:
   for c,label in LABELS.items():
    for lev in LEVELS.astype(int):
     p=self.root/f'{label}-{lev}hpa.nc';before=identity(p);d=xr.open_dataset(p)
     self.datasets[c,int(lev)]=d
     if d[c].dims!=('time','latitude','longitude'):raise ValueError(f'Unexpected raw dimensions {p}: {d[c].dims}')
     if not np.array_equal(d.latitude.values,self.lat) or not np.array_equal(d.longitude.values,self.lon):raise ValueError(f'Coordinate mismatch {p}')
     times=normalized_times(d.time.values)
     if self.time is None:self.time=times
     elif not np.array_equal(self.time,times):raise ValueError(f'Raw time mismatch {p}')
     self.source_records.append({**before,'component':c,'level_hPa':int(lev),'decoded_dtype':str(d[c].dtype),'raw_variable_attrs':dict(d[c].attrs),'physical_units_by_verified_source_convention':UNITS[c],'units_present_in_raw_variable':bool(d[c].attrs.get('units')),'history':d.attrs.get('history','')})
  except BaseException:
   self.close();raise
  self.times=self.time
 def read(self,times,component):
  if component not in LABELS:raise ValueError('component must be u/v/w')
  target=normalized_times(times);idx=np.searchsorted(self.time,target)
  if np.any(idx>=len(self.time)) or not np.array_equal(self.time[idx],target):raise ValueError('Missing exact raw target hour')
  out=[]
  # Slice contiguous chunks to avoid backend fancy-index surprises and unnecessary decompression.
  index=slice(int(idx[0]),int(idx[-1])+1) if len(idx)>0 and np.array_equal(idx,np.arange(idx[0],idx[-1]+1)) else idx
  for lev in LEVELS.astype(int):
   a=self.datasets[component,int(lev)][component].isel(time=index).transpose('time','latitude','longitude')
   out.append(np.asarray(a.values,dtype=np.float64))
  return np.stack(out,axis=1)
 def check_source_identity(self):
  for rec in self.source_records:
   if identity(rec['path'])!={k:rec[k] for k in ('path','size','mtime_ns')}:raise RuntimeError('Raw source identity changed')
  return True
 def close(self):
  for d in self.datasets.values():d.close()
  self.datasets.clear()
 def __enter__(self):return self
 def __exit__(self,*exc):self.close()

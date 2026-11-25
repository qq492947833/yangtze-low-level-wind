"""Align ZWD and ERA5, construct matched causal histories and fit training scalers."""
from pathlib import Path
import os, json, time, hashlib, datetime
import numpy as np
import netCDF4 as nc
import psutil
from paths import ROOT, OLD, ZWD, RUN

BLOCK = 256
def identity(p):
 p=Path(p);s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def atomic(p,o):
 p=Path(p);t=p.with_name(p.name+'.tmp');b=json.dumps(o,ensure_ascii=False,indent=2,default=str).encode('utf8')
 with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(t,p);assert p.read_bytes()==b
def status(stage,**kw):
 atomic(ROOT/'prepare_status.json',dict(time=stamp(),run_id=RUN['run_id'],pid=os.getpid(),create_time=psutil.Process().create_time(),stage=stage,rss=psutil.Process().memory_info().rss,physical_available=psutil.virtual_memory().available,**kw))
def guard():
 v=psutil.virtual_memory()
 if v.available/v.total<.15:raise MemoryError('Physical available memory below 15%; exit after committed block')
 if psutil.Process().memory_info().rss>20*1024**3:raise MemoryError('Task3 preparation process exceeded 20GiB budget')
def cache_inputs():
 receipt=json.loads((OLD/'completion.json').read_text(encoding='utf8'));assert receipt['status']=='PASS'
 for x in receipt['sources']+receipt['arrays']:assert identity(x['path'])==x,(x['path'],'source identity changed')
 co=dict(np.load(OLD/'coordinates.npz'));raw=np.load(OLD/'raw_predictors.npy',mmap_mode='r');wind=np.load(OLD/'raw_wind.npy',mmap_mode='r')
 assert raw.shape[0]==3 and wind.shape[0]==9 and raw.shape[-2:]==(49,73)
 assert raw.dtype==np.float64 and wind.dtype==np.float32
 rh=co['times'].astype('datetime64[h]').astype(np.int64);assert np.all(np.diff(rh)==1)
 return receipt,co,raw,wind,rh

def context():
 receipt,co,raw,wind,rh=cache_inputs()
 with nc.Dataset(ZWD) as d:
  zt=np.asarray(d['time'][:],np.int64);assert d['time'].units=='hours since 1970-01-01 00:00:00';assert np.all(np.diff(zt)==1)
  last=min(zt[-1],rh[-1]);first=max(zt[0],rh[0]); keep=(zt>=first)&(zt<=last); zi=np.where(keep)[0]; t=zt[zi]
  lat=np.asarray(d['latitude'][:]);lon=np.asarray(d['longitude'][:]);li=np.array([np.where(lat==x)[0][0] for x in co['latitude']]);lj=np.array([np.where(lon==x)[0][0] for x in co['longitude']]);assert np.all(np.diff(li)==1) and np.all(np.diff(lj)==1)
  av=np.asarray(d['zwd_available'][zi[0]:zi[-1]+1],np.int8); assert set(np.unique(av)).issubset({0,1})
 sig=dict(run_id=RUN['run_id'],zwd=identity(ZWD),source_receipt_sha256=sha(OLD/'completion.json'),source_identities=receipt['sources'],array_identities=receipt['arrays'],first_hour=int(t[0]),last_hour=int(t[-1]),n=len(t),lat_indices=li.tolist(),lon_indices=lj.tolist(),code_protocol='task3_prepare_v1')
 p=ROOT/'cache'/'identity.json'
 if p.exists():assert json.loads(p.read_text('utf8'))==sig,'Prepared cache identity mismatch; refuse resume'
 else:atomic(p,sig)
 return co,raw,wind,rh,t,zi,av,li,lj

def crop(ctx,max_blocks=None):
 co,raw,wind,rh,t,zi,av,li,lj=ctx;n=len(t);dest=ROOT/'cache'/'zwd_region.npy';prog=ROOT/'cache'/'zwd_blocks.json'
 if prog.exists():
  s=json.loads(prog.read_text('utf8'));assert s['n']==n and s['block']==BLOCK
  z=np.load(dest,mmap_mode='r+');assert z.shape==(n,49,73) and z.dtype==np.float32
 else:
  if dest.exists():raise RuntimeError('Unregistered ZWD cache exists; no silent overwrite')
  z=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(n,49,73));z[:]=np.nan;z.flush();s=dict(n=n,block=BLOCK,next_index=0,blocks=[],finite_rows=0)
  atomic(prog,s)
 start=time.perf_counter();initial=s['next_index'];steps=0
 with nc.Dataset(ZWD) as d:
  v=d['ZWD'];v.set_var_chunk_cache(16*1024**2,1009,.75)
  while s['next_index']<n:
   guard();a=s['next_index'];b=min(n,a+BLOCK);tic=time.perf_counter()
   chunk=np.asarray(np.ma.filled(v[zi[a]:zi[b-1]+1,li[0]:li[-1]+1,lj[0]:lj[-1]+1],np.nan),dtype=np.float32)
   assert chunk.shape==(b-a,49,73)
   finite=np.isfinite(chunk).all((1,2)); missing=~np.isfinite(chunk).any((1,2))
   if np.any((av[a:b]==0)&~missing):raise ValueError('zwd_available=0 yet regional finite values exist; audit source')
   z[a:b]=chunk;z.flush();digest=hashlib.sha256(chunk.tobytes()).hexdigest()
   np.testing.assert_array_equal(z[a:b],chunk)
   s['blocks'].append(dict(start=a,stop=b,sha256=digest,finite_rows=int(finite.sum()),all_missing_rows=int(missing.sum()),seconds=time.perf_counter()-tic))
   s['next_index']=b;s['finite_rows']+=int(finite.sum());atomic(prog,s)
   elapsed=time.perf_counter()-start;rate=(b-initial)/elapsed
   status('CROPPING_ZWD',completed_hours=b,total_hours=n,hours_per_second=rate,eta_seconds=(n-b)/rate,local_nonfinite_rows=b-s['finite_rows'])
   steps+=1
   if max_blocks and steps>=max_blocks:return False
 atomic(ROOT/'cache'/'ZWD_CROP_COMPLETION.json',dict(status='PASS',count=n,finite_rows=s['finite_rows'],cache=identity(dest),blocks_receipt_sha256=sha(prog),time=stamp()))
 return True

def audit_era(ctx):
 co,raw,wind,rh,t,zi,av,li,lj=ctx;out=ROOT/'cache'/'era5_finite.npz';p=ROOT/'cache'/'ERA5_FINITE_COMPLETION.json'
 if p.exists():
  q=json.loads(p.read_text('utf8'));assert sha(out)==q['sha256'];return dict(np.load(out))
 inputs=np.zeros(len(t),bool);targets=np.zeros(len(t),bool);start=time.perf_counter()
 for a in range(0,len(t),BLOCK):
  guard();b=min(len(t),a+BLOCK);ra=t[a]-rh[0];rb=ra+b-a
  inputs[a:b]=np.isfinite(raw[:,ra:rb]).all((0,2,3));targets[a:b]=np.isfinite(wind[:,ra:rb]).all((0,2,3))
  if a%(BLOCK*16)==0:status('ERA5_FINITE_SCAN',completed=b,total=len(t),elapsed=time.perf_counter()-start)
 with (out.with_suffix('.tmp')).open('wb') as f:np.savez(f,inputs=inputs,targets=targets)
 os.replace(out.with_suffix('.tmp'),out);atomic(p,dict(status='PASS',sha256=sha(out),input_bad=int((~inputs).sum()),target_bad=int((~targets).sum())))
 return dict(inputs=inputs,targets=targets)

def build_manifest(ctx,finite):
 co,raw,wind,rh,t,zi,av,li,lj=ctx;n=len(t);z=np.load(ROOT/'cache'/'zwd_region.npy',mmap_mode='r');zok=np.zeros(n,bool)
 for a in range(0,n,BLOCK):zok[a:a+BLOCK]=np.isfinite(z[a:a+BLOCK]).all((1,2))
 zgood=(av==1)&zok;reason=np.zeros((n,6),np.uint16);allids=np.arange(n)
 for h in range(1,7):
  reason[~zgood,h-1]|=1;reason[~finite['targets'],h-1]|=8
  for lag in range(11):
   ix=allids-h-lag;inside=ix>=0;reason[~inside,h-1]|=16;ix=ix.clip(0,n-1)
   reason[inside&~zgood[ix],h-1]|=2;reason[inside&~finite['inputs'][ix],h-1]|=4
 common=np.all(reason==0,axis=1);ids=np.where(common)[0];N=len(ids);assert N>1000
 train=int(.7*N);validstop=int(.8*N);split=np.full(n,-1,np.int8);split[ids[:train]]=0;split[ids[train:validstop]]=1;split[ids[validstop:]]=2
 p=ROOT/'matched_sample_manifest.nc';tmp=p.with_suffix('.part.nc')
 if p.exists():
  with nc.Dataset(p) as d:np.testing.assert_array_equal(d['target_time'][:],t);np.testing.assert_array_equal(d['retained'][:],common);np.testing.assert_array_equal(d['split'][:],split)
 else:
  with nc.Dataset(tmp,'w') as d:
   for name,size in [('candidate',n),('sample',N),('lead',6),('lagtime',11),('latitude',49),('longitude',73)]:d.createDimension(name,size)
   def put(name,typ,dims,data,**attrs):
    v=d.createVariable(name,typ,dims,zlib=True,complevel=9,shuffle=True);v[:]=data
    for k,val in attrs.items():v.setncattr(k,val)
    return v
   put('target_time','i8',('candidate',),t,units='hours since 1970-01-01 00:00:00',calendar='standard')
   put('sample_target_time','i8',('sample',),t[ids],units='hours since 1970-01-01 00:00:00')
   put('sample_candidate_index','i8',('sample',),ids);put('retained','i1',('candidate',),common);put('split','i1',('candidate',),split,definition='-1 excluded,0 train,1 validation,2 test')
   put('deletion_reason','u2',('candidate','lead'),reason,bit_definition='1 target ZWD unavailable/nonfinite;2 historical ZWD unavailable/nonfinite;4 historical original input nonfinite;8 target wind nonfinite;16 outside available history')
   put('lead_time','i1',('lead',),np.arange(1,7),units='hour');put('lagtime','i1',('lagtime',),np.arange(-10,1),units='hour relative to init')
   put('init_time','i8',('sample','lead'),t[ids,None]-np.arange(1,7)[None,:],units='hours since 1970-01-01 00:00:00')
   hist=d.createVariable('history_time','i8',('sample','lead','lagtime'),zlib=True,complevel=9,shuffle=True)
   for a in range(0,N,4096):hist[a:a+4096]=t[ids[a:a+4096],None,None]-np.arange(1,7)[None,:,None]+np.arange(-10,1)[None,None,:]
   hist.units='hours since 1970-01-01 00:00:00';put('zwd_available','i1',('candidate',),av);put('zwd_region_all_finite','i1',('candidate',),zok)
   put('era5_input_all_finite','i1',('candidate',),finite['inputs']);put('era5_target_all_finite','i1',('candidate',),finite['targets'])
   put('latitude','f8',('latitude',),co['latitude']);put('longitude','f8',('longitude',),co['longitude'])
   d.run_id=RUN['run_id'];d.split_rule='train=[0,floor(.7N)), validation=[floor(.7N),floor(.8N)), test=[floor(.8N),N), chronological common targets';d.history_rule='Each lead full 11-hour causal input dependency; target ZWD is only availability filter';d.source_identity_sha256=sha(ROOT/'cache'/'identity.json');d.no_interpolation_or_fill='true'
  os.replace(tmp,p)
 np.savez(ROOT/'cache'/'manifest_indices.npz',target_hours=t,retained_indices=ids,split=split,train_count=train,valid_stop=validstop,raw_offset=int(t[0]-rh[0]))
 atomic(ROOT/'manifest_summary.json',dict(status='PASS',run_id=RUN['run_id'],manifest_sha256=sha(p),candidate_count=n,common_count=N,train=train,valid=validstop-train,test=N-validstop,first=str(t[ids[0]].astype('datetime64[h]')),last=str(t[ids[-1]].astype('datetime64[h]')),train_last=str(t[ids[train-1]].astype('datetime64[h]')),test_first=str(t[ids[validstop]].astype('datetime64[h]')),zwd_available_missing=int((av==0).sum()),zwd_partial_nonfinite=int(((av==1)&~zok).sum()),era5_input_bad=int((~finite['inputs']).sum()),era5_target_bad=int((~finite['targets']).sum())))
 return ids,train,validstop

def moments(source,idx,axis=0):
 shape=source.shape[1:] if axis==0 else (source.shape[0],)+source.shape[2:];count=0;mean=np.zeros(shape,np.float64);M2=np.zeros_like(mean)
 for a in range(0,len(idx),128):
  guard();v=np.asarray(source[idx[a:a+128]] if axis==0 else source[:,idx[a:a+128]].transpose(1,0,2,3),np.float64);assert np.isfinite(v).all()
  k=len(v);mu=v.mean(0);dm=mu-mean;M2+=((v-mu)**2).sum(0)+dm*dm*count*k/(count+k);mean+=dm*k/(count+k);count+=k
 std=np.sqrt(M2/count);zero=std==0;scale=np.where(zero,1.,std)
 return mean,std,scale,zero,count

def normalize(ctx,ids,ntrain):
 co,raw,wind,rh,t,zi,av,li,lj=ctx;p=ROOT/'normalization_statistics.nc'
 if p.exists():
  with nc.Dataset(p) as d:assert d.manifest_sha256==sha(ROOT/'matched_sample_manifest.nc')
  return
 mask=np.zeros(len(t),bool)
 for delay in range(1,17):mask[ids[:ntrain]-delay]=True
 ix=np.where(mask)[0];rix=ix+(t[0]-rh[0]);yix=ids[:ntrain]+(t[0]-rh[0]);z=np.load(ROOT/'cache'/'zwd_region.npy',mmap_mode='r')
 status('TRAIN_ONLY_STATISTICS',input_unique_hours=len(ix),target_count=len(yix))
 xm= moments(raw,rix,axis=1);zm=moments(z,ix);ym=moments(wind,yix,axis=1)
 tmp=p.with_suffix('.part.nc')
 with nc.Dataset(tmp,'w') as d:
  for name,size in [('x_feature',3),('y_feature',9),('latitude',49),('longitude',73),('input_fit_time',len(ix)),('target_fit_time',len(yix))]:d.createDimension(name,size)
  for prefix,data,dims in [('x',xm,('x_feature','latitude','longitude')),('zwd',zm,('latitude','longitude')),('y',ym,('y_feature','latitude','longitude'))]:
   for name,a in zip(('mean','std','scale','zero_std'),data[:4]):
    v=d.createVariable(prefix+'_'+name,'i1' if name=='zero_std' else 'f8',dims,zlib=True,complevel=9,shuffle=True);v[:]=a
   d.setncattr(prefix+'_fit_count',int(data[4]))
  d.createVariable('input_fit_time','i8',('input_fit_time',))[:]=t[ix];d.createVariable('target_fit_time','i8',('target_fit_time',))[:]=t[ids[:ntrain]]
  d.manifest_sha256=sha(ROOT/'matched_sample_manifest.nc');d.scope='training targets only; X/ZWD unique causal history timestamps across all six lead training samples; no validation or test label statistics';d.std_definition='population std ddof=0 float64 merged moments';d.zero_std_handling='scale=1 and zero_std flag, no division by zero';d.y_storage='u/v m s-1; w=-0.1*omega; omega Pa s-1';d.zwd_units='native ZWDX_v100 units; source missing units attribute; verification pending';d.x_channel_order='sp,t2m,d2m';d.y_channel_order='u850,u925,u1000,v850,v925,v1000,w850,w925,w1000'
 os.replace(tmp,p)
 atomic(ROOT/'NORMALIZATION_COMPLETION.json',dict(status='PASS',sha256=sha(p),input_unique_hours=len(ix),target_count=len(yix),input_fit_last=int(t[ix[-1]]),target_fit_last=int(t[ids[ntrain-1]]),time=stamp()))

if __name__ == "__main__":
    (ROOT / "cache").mkdir(parents=True, exist_ok=True)
    ctx = context()
    crop(ctx)
    finite = audit_era(ctx)
    ids, train_count, _ = build_manifest(ctx, finite)
    normalize(ctx, ids, train_count)

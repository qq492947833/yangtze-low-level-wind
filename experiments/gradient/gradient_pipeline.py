"""Gradient-augmented two-stage DA-TransUNet training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc, os, json, hashlib, random, time, shutil

from collections import OrderedDict
import numpy as np
import psutil
import netCDF4
import torch
from baseline_io import open_data_nc, clear_model_memory, ArrayWindowProvider, WindowSelection

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("YANGTZE_WORK_DIR", "work")).resolve() / "gradient"
MODEL = Path(os.environ.get("YANGTZE_MODEL_DIR", "models")).resolve()
RESULT = Path(os.environ.get("YANGTZE_RESULTS_DIR", "results")).resolve()
DATA = Path(os.environ.get("YANGTZE_DATA_DIR", "data")).resolve()
DEVICE = os.environ.get("YANGTZE_DEVICE", "cuda:0")
for directory in (ROOT, MODEL, RESULT):
    directory.mkdir(parents=True, exist_ok=True)
TIMESTEP, DOWNNUM, COVNUM = 6, 4, 2
LEVELS = np.asarray([850,925,1000], dtype=np.float32)
COMPONENTS = {'u':slice(0,3),'v':slice(3,6),'w':slice(6,9)}
CHANNELS = ['sp','t2m','d2m','sp_dx','sp_dy','sp_dt','t2m_dx','t2m_dy','t2m_dt','d2m_dx','d2m_dy','d2m_dt']
COMMON = dict(test_size=.2,valid_size=.1,k_fold=None,task_mode='regression',
 if_bottom_attention_only='yes',if_resnet='yes',layer_time=1,baselayer=16,
 cov_kernelsize=3,cov_strides=1,pool_method='maxpooling',pool_kernel_size=2,pool_strides=2,
 ifnormalization='no',normalization_method=None,da_attention_order='space_first',
 trans_num=1,num_heads=1,dim_feedforward=128,trans_dropout_rate=0.,activate='tanh',
 nlp_activate='relu',if_last_act='no',if_print_model='yes',loss_function='default',
 optimizer='Adam',metrics='Pearsonr',learning_rate=.0001,epochs=5000,batch_size=500,
 if_early_stopping=100,ifheatmap='no',ifweight='no',ifmute='no',device=DEVICE)
SOURCES = [('sp','Surface-pressure.nc'),('t2m','2m-temperature.nc'),('d2m','2m-dewpoint-temperature.nc')]+[(c,f'{label}-{lev}hpa.nc') for c,label in [('u','U-component-of-wind'),('v','V-component-of-wind'),('w','Vertical-velocity')] for lev in (850,925,1000)]

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));f.flush();os.fsync(f.fileno())
    for attempt in range(4):
        try: os.replace(temp,path);break
        except PermissionError:
            if attempt==3: raise
            time.sleep(.25*(attempt+1))
    assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(obj,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def artifact(path):
    p=Path(path);return {'path':str(p),'size':p.stat().st_size,'sha256':sha(p)}
def identity(path):
    p=Path(path);s=p.stat();return {'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def event(kind,**kw):
    with (ROOT/'pipeline_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':now(),'event':kind,**kw},ensure_ascii=False,default=str)+'\n');f.flush()
    print(kind,kw,flush=True)
def exact_hours(values):
    t=np.asarray(values).astype('datetime64[ns]');assert t.ndim==1 and not np.isnat(t).any()
    ns=t.astype(np.int64);unit=3600*10**9;rounded=((ns+unit//2)//unit)*unit
    assert np.max(np.abs(ns-rounded))<=1000, 'Non-hour time beyond confirmed ns encoding tolerance'
    t=rounded.astype('datetime64[ns]');assert np.all(np.diff(t)==np.timedelta64(1,'h'))
    return t
def nc_path(h,kind,c,phase):
    suffix={'test':'_gradient','all':'_all_gradient','final':'_residual_gradient'}[phase]
    return RESULT/f'da_transunet_{kind}_{c}_{h}_24year_changjiang{suffix}.nc'
def final_stem(h,stage):return MODEL/f"DA_TransUNet_{h}_24year_changjiang{'_residual' if stage==2 else ''}_gradient"
def hour_root(h):
    assert isinstance(h,int) and 1<=h<=6
    p=ROOT/f'hour_{h}';p.mkdir(exist_ok=True);return p

def resources():
    available = psutil.virtual_memory().available
    return dict(physical_available=available, commit_available=available,
                process_rss=psutil.Process().memory_info().rss, time=now())

def verify_device():
    requested = torch.device(DEVICE)
    if requested.type == "cuda" and requested.index is not None:
        torch.cuda.set_device(requested)
    return {"device": DEVICE, "torch": str(torch.__version__)}

def seed25():
    random.seed(25);np.random.seed(25);torch.manual_seed(25);torch.cuda.manual_seed_all(25)

def source_cache():
    dest=ROOT/'source_cache';dest.mkdir(exist_ok=True);receipt=dest/'completion.json'
    before=[identity(DATA/name) for _,name in SOURCES]
    if receipt.exists():
        old=load(receipt);assert old['status']=='PASS' and old['sources']==before
        for x in old['arrays']:assert identity(x['path'])==x
        coords=np.load(dest/'coordinates.npz');return np.load(dest/'raw_predictors.npy',mmap_mode='r'),np.load(dest/'raw_wind.npy',mmap_mode='r'),dict(coords)
    paths=[dest/'raw_predictors.npy',dest/'raw_wind.npy']
    if any(p.exists() for p in paths):raise RuntimeError('Incomplete source cache requires explicit owned-cache repair; no silent reuse')
    arrays=None;times=lat=lon=None
    for i,(var,name) in enumerate(SOURCES):
        event('READ_SOURCE',index=i,path=str(DATA/name))
        field,lo,la,_,*_,ti=open_data_nc('one',str(DATA/name),var,'yes','time','2000-01-01-06','2023-12-31-23','yes','longitude','yes','latitude',24.,36.,108.,126.,.25,.25,'no','no',None,None,changeresolution=1,timespace=1,ifchange_west_east='no',ifinterpolate='no')
        tm=exact_hours(ti)
        if arrays is None:
            times,lat,lon=tm,np.asarray(la),np.asarray(lo);n,H,W=field.shape
            arrays=[np.lib.format.open_memmap(paths[0],mode='w+',dtype='float64',shape=(3,n,H,W)),np.lib.format.open_memmap(paths[1],mode='w+',dtype='float32',shape=(9,n,H,W))]
        np.testing.assert_array_equal(tm,times);np.testing.assert_array_equal(la,lat);np.testing.assert_array_equal(lo,lon)
        if var=='w':field=field*-0.1  # exact reference storage convention, once
        target=arrays[0][i] if i<3 else arrays[1][i-3]
        for a in range(0,len(field),512):
            b=min(a+512,len(field));assert np.isfinite(field[a:b]).all();target[a:b]=field[a:b]
        del field,target;gc.collect()
    for a in arrays:a.flush()
    np.savez_compressed(dest/'coordinates.npz',times=times,latitude=lat,longitude=lon,level=LEVELS)
    assert before==[identity(DATA/name) for _,name in SOURCES]
    save(receipt,{'status':'PASS','created':now(),'sources':before,'arrays':[identity(p) for p in paths],
       'dtype':'physical surface predictors float64 as decoded source; wind float32 as exact original assignment','omega_storage':'w=-0.1*omega exactly once','first':str(times[0]),'last':str(times[-1]),'count':len(times)})
    del arrays;gc.collect()
    return source_cache()

def split_info(h,coords):
    times=coords['times'];n=len(times)-h;nall=n-5;test_start=int(.8*n);train_stop=int((1-.1/.8)*test_start)
    all_times=times[h+5:];test_times=all_times[test_start:]
    assert len(all_times)==nall and 0<train_stop<test_start<nall
    rtest=test_start-5;rtrain=train_stop-5;rts=1-(rtest+.5)/nall;rvs=(1-rts)*(1-(rtrain+.5)/rtest)
    assert int((1-rts)*nall)==rtest and int((1-rvs/(1-rts))*rtest)==rtrain
    np.testing.assert_array_equal(all_times[5:][rtest:],test_times)
    np.testing.assert_array_equal(all_times[5:][rtrain:rtest],all_times[train_stop:test_start])
    info={'hour':h,'n_pairs':n,'n_all':nall,'train_stop':train_stop,'test_start':test_start,'n_test':len(test_times),
     'stage2_train_stop':rtrain,'stage2_test_start':rtest,'stage2_test_size':rts,'stage2_valid_size':rvs,
     'baseline_split_on_pre_window_length':True,'gradient_extra_removed_targets':0,'stage2_removed_training_targets':5,
     'all_first':str(all_times[0]),'all_last':str(all_times[-1]),'test_first':str(test_times[0]),'test_last':str(test_times[-1])}
    hr=hour_root(h);alignment=dict(info)
    if (hr/'time_alignment.json').exists():
        previous=load(hr/'time_alignment.json');previous.pop('validated',None);assert previous==alignment
    else:save(hr/'time_alignment.json',{**alignment,'validated':now()})
    indices=dict(raw_times=times,all_times=all_times,test_times=test_times,
      stage1_valid_times=all_times[train_stop:test_start],stage2_valid_times=all_times[5:][rtrain:rtest],
      issue_times=all_times-np.timedelta64(h,'h'),stage2_target_times=all_times[5:])
    if (hr/'time_indices.npz').exists():
        with np.load(hr/'time_indices.npz') as existing:
            assert set(existing.files)==set(indices)
            for key,value in indices.items():np.testing.assert_array_equal(existing[key],value)
    else:np.savez_compressed(hr/'time_indices.npz',**indices)
    return info,all_times,test_times

def normalize_stage1(h,rawx,rawy,train_stop):
    # Same channel-major float32 assignment, transpose, reduction and arithmetic as reference.
    vx=np.array(rawx[:,:-h],dtype=np.float32,copy=True).transpose(1,2,3,0)
    vy=np.array(rawy[:,h:],dtype=np.float32,copy=True).transpose(1,2,3,0)
    physical=np.asarray(rawx[:,:-h]).transpose(1,0,2,3)
    norm_path=hour_root(h)/'stage1_normalization.npz'
    if norm_path.exists():
        with np.load(norm_path) as old:
            assert int(old['forecast_hours'])==h and str(old['scope'])=='unique training input hours and aligned training labels, ddof=0'
            xm,xs,ym,ys=[old[k] for k in ('vx_mean','vx_std','vy_mean','vy_std')]
    else:
        train_x = vx[:train_stop + TIMESTEP - 1]
        train_y = vy[TIMESTEP - 1:train_stop + TIMESTEP - 1]
        ym=np.nanmean(train_y,axis=0);ys=np.nanstd(train_y,axis=0);xm=np.nanmean(train_x,axis=0);xs=np.nanstd(train_x,axis=0)
    for v in (xm,xs,ym,ys):assert np.isfinite(v).all()
    assert (xs>0).all() and (ys>0).all()
    vx=(vx-xm)/xs;vy=(vy-ym)/ys
    if not norm_path.exists():
        np.savez_compressed(norm_path,vx_mean=xm,vx_std=xs,vy_mean=ym,vy_std=ys,
          input_names=np.array(CHANNELS[:3]),output_names=np.array([f'{c}{lev}' for c in COMPONENTS for lev in LEVELS.astype(int)]),
          forecast_hours=h,scope='unique training input hours and aligned training labels, ddof=0',zero_variance_count=0)
    return physical,vx,vy,xm,xs,ym,ys

class RAMWindowCache:
    """Evictable contiguous slabs; next-allocation admission uses observed dynamic headroom."""
    def __init__(self,provider,stage_root,block=500):
        self.provider=provider;self.shape=provider.shape;self.block=block;self.cache=OrderedDict();self.root=stage_root
        self.history=[];self.bytes_per_window=int(np.prod(self.shape[1:]))*4;self.loaded_bytes=0;self.last_pressure_check=0
    def _margin(self):
        r=resources();self.history.append((time.monotonic(),r));self.history=[x for x in self.history if time.monotonic()-x[0]<600]
        # Float32 batch construction plus float64 gradient intermediates and writer buffers.
        work=8*self.block*self.bytes_per_window
        fluctuations=max((x[1]['physical_available']+x[1]['process_rss'] for x in self.history),default=0)-min((x[1]['physical_available']+x[1]['process_rss'] for x in self.history),default=0)
        return r,max(work,fluctuations)
    def _get(self,k):
        if k in self.cache:self.cache.move_to_end(k);return self.cache[k]
        a=k*self.block;b=min(a+self.block,self.shape[0]);need=(b-a)*self.bytes_per_window
        r,margin=self._margin()
        while self.cache and min(r['physical_available'],r['commit_available'])<need+margin:
            _,old=self.cache.popitem(last=False);self.loaded_bytes-=old.nbytes;del old;gc.collect();r=resources()
        if min(r['physical_available'],r['commit_available'])<need+margin:
            # Same GPU batch, smaller temporary CPU construction. Never admit a batch blind.
            minimum=2*need+16*self.bytes_per_window
            started=time.monotonic()
            while min(r['physical_available'],r['commit_available'])<minimum:
                event('WAITING_RESOURCE',phase='mandatory_window_batch',required=minimum,resources=r)
                if time.monotonic()-started>=600:
                    raise MemoryError('Equivalent 16-window CPU construction still cannot fit the mandatory unchanged GPU batch after 10 min')
                time.sleep(15);r=resources()
            result=np.empty((b-a,)+self.shape[1:],dtype=np.float32)
            for start in range(a,b,16):result[start-a:min(start+16,b)-a]=self.provider.take(np.arange(start,min(start+16,b)))
        else:result=np.asarray(self.provider.take(np.arange(a,b)),dtype=np.float32)
        if min(r['physical_available'],r['commit_available'])>=need+margin:
            self.cache[k]=result;self.loaded_bytes+=result.nbytes
        return result
    def take(self,indices):
        if time.monotonic()-self.last_pressure_check>60:
            r,margin=self._margin()
            while self.cache and min(r['physical_available'],r['commit_available'])<margin:
                _,old=self.cache.popitem(last=False);self.loaded_bytes-=old.nbytes;del old;gc.collect();r=resources()
            self.last_pressure_check=time.monotonic()
        ids=np.asarray(indices,dtype=np.int64);out=np.empty((len(ids),)+self.shape[1:],dtype=np.float32)
        for k in np.unique(ids//self.block):
            mask=ids//self.block==k;out[mask]=self._get(int(k))[ids[mask]-k*self.block]
        return out
    def warm(self):
        start=time.monotonic();filled=0
        for k in range((self.shape[0]+self.block-1)//self.block):
            r,margin=self._margin();need=min(self.block,self.shape[0]-k*self.block)*self.bytes_per_window
            if min(r['physical_available'],r['commit_available'])<need+margin:break
            self._get(k);filled+=1
            if k%25==0:event('WINDOW_CACHE_BUILD',blocks=filled,bytes=self.loaded_bytes,stage=str(self.root))
        save(self.root/'cache_strategy.json',{'created':now(),'mode':'evictable RAM slabs, on-demand equivalent fallback','cached_bytes':self.loaded_bytes,
          'full_window_bytes':self.shape[0]*self.bytes_per_window,'warm_seconds':time.monotonic()-start,'block_windows':self.block,
          'training_batch_size_unchanged':500,'dynamic_margin':'max(8 CPU construction batches, observed external physical fluctuation over 600s)',
          'fixed_system_used_percentage':None,'fixed_512GB_cap':None,'current':resources()})

def verify_nc(path,var,times,coords,expected_reader=None):
    with netCDF4.Dataset(str(path)) as ds:
        f=ds[var];assert f.dtype==np.dtype('float32') and f.shape==(len(times),3,len(coords['latitude']),len(coords['longitude']))
        filters=f.filters();assert filters['zlib'] and filters['complevel']==9 and filters['shuffle']
        got=np.asarray(netCDF4.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False),dtype='datetime64[ns]')
        np.testing.assert_array_equal(got,times)
        for k in ('latitude','longitude'):np.testing.assert_allclose(ds[k][:],coords[k],rtol=0,atol=1e-5)
        np.testing.assert_array_equal(ds['level'][:],LEVELS)
        for a in range(0,len(times),512):
            b=min(a+512,len(times));v=np.asarray(f[a:b]);assert np.isfinite(v).all()
            if expected_reader is not None:np.testing.assert_array_equal(v,np.asarray(expected_reader(a,b),dtype=np.float32))

def write_field(h,kind,c,phase,times,coords,reader,metadata):
    dest=nc_path(h,kind,c,phase);record=hour_root(h)/'file_receipts'/(dest.name+'.json');record.parent.mkdir(exist_ok=True)
    if dest.exists():
        if not record.exists():raise FileExistsError(f'Unowned existing destination: {dest}')
        old=load(record);assert old['sha256']==sha(dest) and old['status'] in ('PASS','VALIDATED_TEMP')
        if old['status']=='VALIDATED_TEMP':
            verify_nc(dest,f'{kind}_{c}',times,coords,reader)
            save(record,{'status':'PASS',**artifact(dest),'created':now(),'temporary_closed_and_verified':True,'recovered_after_publish':True})
        return artifact(dest)
    temp=dest.with_name('.'+dest.name+'.gradient_task.tmp')
    if temp.exists():
        if record.exists() and load(record).get('status')=='VALIDATED_TEMP':
            old=load(record);assert old['temporary']==str(temp) and old['sha256']==sha(temp)
            verify_nc(temp,f'{kind}_{c}',times,coords,reader)
            os.rename(temp,dest)
            save(record,{'status':'PASS',**artifact(dest),'created':now(),'temporary_closed_and_verified':True,'recovered_validated_temp':True})
            return artifact(dest)
        raise FileExistsError(f'Incomplete task output requires scoped repair: {temp}')
    assert shutil.disk_usage(RESULT).free > len(times)*3*len(coords['latitude'])*len(coords['longitude'])*4*1.1
    with netCDF4.Dataset(str(temp),'w',format='NETCDF4') as ds:
        for name,n in zip(('time','level','latitude','longitude'),(len(times),3,len(coords['latitude']),len(coords['longitude']))):ds.createDimension(name,n)
        t=ds.createVariable('time','i8',('time',),fill_value=False);t.units='seconds since 1970-01-01 00:00:00';t.calendar='standard';t[:]=times.astype('datetime64[s]').astype(np.int64)
        for name,val,unit in [('level',LEVELS,'hPa'),('latitude',coords['latitude'],'degrees_north'),('longitude',coords['longitude'],'degrees_east')]:
            v=ds.createVariable(name,'f4',(name,),fill_value=False);v[:]=val;v.units=unit
        f=ds.createVariable(f'{kind}_{c}','f4',('time','level','latitude','longitude'),zlib=True,complevel=9,shuffle=True,fill_value=False,chunksizes=(min(128,len(times)),1,len(coords['latitude']),len(coords['longitude'])))
        f.units='m s-1' if c!='w' else 'scaled Pa s-1'
        if c=='w':f.vertical_velocity_note='w = -0.1 * ERA5 omega; multiply by -10 once only for physical omega evaluation'
        ds.setncatts({'Conventions':'CF-1.8','forecast_hours':h,'history_steps':6,'gradient_stage1_only':1,'PINN':0,'VMD':0,**metadata})
        for a in range(0,len(times),128):
            b=min(a+128,len(times));v=np.asarray(reader(a,b));assert np.isfinite(v).all() and np.max(np.abs(v))<=np.finfo(np.float32).max
            f[a:b]=v.astype(np.float32)
            if a%4096==0:event('NC_COMPRESSION_WRITE',file=dest.name,completed=b,total=len(times))
    verify_nc(temp,f'{kind}_{c}',times,coords,reader)
    digest=sha(temp)
    # Preserve an ownership/provenance receipt before publication; complete only after rename.
    save(record,{'status':'VALIDATED_TEMP','temporary':str(temp),'sha256':digest,'destination':str(dest)})
    if dest.exists():raise FileExistsError(dest)
    os.rename(temp,dest)
    assert sha(dest)==digest
    save(record,{'status':'PASS',**artifact(dest),'created':now(),'temporary_closed_and_verified':True})
    return artifact(dest)

def publish_models(h,stage,temp_stem):
    entries=[]
    for suffix in ('.pth','_opt.pth'):
        src=Path(str(temp_stem)+suffix);assert src.is_file()
        obj=torch.load(src,map_location=DEVICE,weights_only=False);assert isinstance(obj,dict) and len(obj)>0;del obj
        dst=Path(str(final_stem(h,stage))+suffix)
        if dst.exists():
            if sha(dst)!=sha(src):raise FileExistsError(f'Existing model differs: {dst}')
        else:
            tmp=dst.with_name('.'+dst.name+'.gradient_task.tmp')
            with src.open('rb') as fin,tmp.open('xb') as fout:shutil.copyfileobj(fin,fout,8*1024**2);fout.flush();os.fsync(fout.fileno())
            assert sha(tmp)==sha(src)
            if dst.exists():raise FileExistsError(dst)
            os.rename(tmp,dst)
        entries.append(artifact(dst))
    return entries

def acceptance_existing(h,stage):
    p=hour_root(h)/f'stage{stage}_acceptance.json'
    if not p.exists():return False
    obj=load(p);assert obj['status']=='PASS' and obj['hour']==h and obj['stage']==stage
    for x in obj['artifacts']:
        q=Path(x['path']);assert q.stat().st_size==x['size'] and sha(q)==x['sha256']
    return True

def make_runtime(h,stage,training):
    from training_runtime import GradientTrainingRuntime
    stage_root=hour_root(h)/f'stage{stage}';stage_root.mkdir(exist_ok=True)
    hr=hour_root(h)
    files=[CODE/n for n in ('gradient_pipeline.py','gradient_features.py','baseline_io.py')]
    files += [ROOT/'source_cache'/'completion.json',ROOT/'source_cache'/'coordinates.npz']
    files += [hr/n for n in ('stage1_normalization.npz','gradient_normalization.npz','gradient_normalization.json','gradient_definition.json','time_indices.npz','time_alignment.json')]
    if stage==2:files += [hr/'stage1_acceptance.json',hr/'stage2_normalization.npz']
    bindings={str(p):sha(p) for p in files}
    return GradientTrainingRuntime(run_root=ROOT,hour=h,stage=stage,
        config={'forecast_hours':h,'stage':stage,'seed':25,'training':training,'channel_count':12,'device':DEVICE,
          'data_statistics_time_and_implementation_sha256':bindings},
        adapter_path=CODE/'network_adapter.py',allow_resume=True)

def run_stage1(h):
    if acceptance_existing(h,1):event('REUSE_ACCEPTED_STAGE',hour=h,stage=1);return
    began=time.monotonic();device=verify_device();hr=hour_root(h);sr=hr/'stage1';sr.mkdir(exist_ok=True)
    rawx,rawy,coords=source_cache();info,all_times,test_times=split_info(h,coords)
    physical,vx,vy,xm,xs,ym,ys=normalize_stage1(h,rawx,rawy,info['train_stop']);del rawx,rawy;gc.collect()
    from gradient_features import GradientFeatureBuilder
    from training_runtime import GradientTrainingRuntime
    import network_adapter as net
    builder=GradientFeatureBuilder(physical,vx.transpose(0,3,1,2),coords['latitude'],coords['longitude'])
    if (hr/'gradient_normalization.npz').exists():builder.load_stats(hr/'gradient_normalization.npz')
    else:
        stats=builder.fit_stats(np.arange(info['train_stop']),batch_size=128,
            progress=lambda completed,total:event('GRADIENT_STATISTICS',hour=h,completed=completed,total=total))
        stats.save(hr/'gradient_normalization.npz')
    gradient_definition={'scope':'training windows; sample and six-history axes share one statistic per grid/channel',
        'channel_order':CHANNELS,'physical_source_dtype':'float64 decoded source before baseline float32 assignment or standardization',
        'earth_radius_m':6371000,'time_interval_s':3600,'x_positive':'east','y_positive':'north','latitude_direction':'36 to 24 descending',
        'spatial_difference':'actual radian coordinates; central interior, first-order one-sided domain edges; no wrap/no crop',
        'temporal_difference':'each six-frame window only; central interior, one-sided earliest/latest; no seventh frame',
        'units':['Pa','K','K','Pa m-1','Pa m-1','Pa s-1','K m-1','K m-1','K s-1','K m-1','K m-1','K s-1'],
        'unit_source':'project physical variable definitions; original source files do not carry units attributes',
        'window_count':info['train_stop'],'first_issue':str(all_times[0]-np.timedelta64(h,'h')),'last_issue':str(all_times[info['train_stop']-1]-np.timedelta64(h,'h'))}
    if (hr/'gradient_definition.json').exists():assert load(hr/'gradient_definition.json')==gradient_definition
    else:save(hr/'gradient_definition.json',gradient_definition)
    class Provider:
        shape=(info['n_all'],6,12,len(coords['latitude']),len(coords['longitude']))
        def take(self,indices):return builder.get_windows(indices)
    provider=RAMWindowCache(Provider(),sr);provider.warm()
    runtime=make_runtime(h,1,COMMON);net.GRADIENT_RUNTIME=runtime;net.WindowSelection=WindowSelection;seed25()
    save(sr/'run_configuration.json',{'hour':h,'stage':1,'device':device,'seed':25,'training':COMMON,'input_channels':CHANNELS,
      'definition':'raw SP/T2M/D2M plus nine physical spatial/window-internal temporal derivatives; no VMD/PINN','time_split':info})
    temp_stem=sr/'trained_best'
    outputs=net.Auto_DA_TransUnet_pytorch(vy,vx,6,4,2,**COMMON,if_best_mode='no',modelpath=None,ifrandom_split='no',ifsave='yes',savepath=str(temp_stem),window_provider=provider)
    model,prediction,truth=outputs[:3];assert prediction.shape==truth.shape==(info['n_test'],9,len(coords['latitude']),len(coords['longitude']))
    for a in range(0,info['n_test'],512):
        np.testing.assert_array_equal(truth[a:a+512],vy[5+info['test_start']+a:5+info['test_start']+a+512].transpose(0,3,1,2))
    records=publish_models(h,1,temp_stem);best_sha=sha(Path(str(final_stem(h,1))+'.pth'))
    meta={'stage':'gradient_stage1_test','source_model_sha256':best_sha,'source_model':str(final_stem(h,1))+'.pth','normalization':'training input hours, labels and gradient windows per grid/channel'}
    for c,ch in COMPONENTS.items():
        mean=ym[:,:,ch].transpose(2,0,1);std=ys[:,:,ch].transpose(2,0,1)
        for kind,values in [('testy',truth),('predicty',prediction)]:
            def reader(a,b,values=values,ch=ch,mean=mean,std=std):return np.asarray(values[a:b,ch],dtype=np.float64)*std+mean
            records.append(write_field(h,kind,c,'test',test_times,coords,reader,meta))
    # Same best checkpoint, exact reference all_test mode, fixed training batch size.
    model.load_state_dict(torch.load(str(final_stem(h,1))+'.pth',map_location=DEVICE,weights_only=False));model.eval()
    all_path=sr/'all_prediction_normalized.npy';all_receipt=sr/'all_inference_progress.json'
    expected_all={'path':str(all_path),'model_sha256':best_sha,'count':info['n_all'],
      'normalization_sha256':sha(hr/'stage1_normalization.npz'),'time_index_sha256':sha(hr/'time_indices.npz')}
    if all_path.exists():
        assert all_receipt.exists(),'Unowned all inference scratch'
        state=load(all_receipt);assert all(state[k]==v for k,v in expected_all.items())
        allpred=np.load(all_path,mmap_mode='r+');all_start=int(state['completed'])
        assert allpred.shape==(info['n_all'],9,len(coords['latitude']),len(coords['longitude'])) and allpred.dtype==np.float64
        assert 0<=all_start<=info['n_all']
    else:
        allpred=np.lib.format.open_memmap(all_path,mode='w+',dtype='float64',shape=(info['n_all'],9,len(coords['latitude']),len(coords['longitude'])))
        all_start=0;save(all_receipt,{**expected_all,'completed':0,'status':'RUNNING'})
    with torch.no_grad():
        for a in range(all_start,info['n_all'],500):
            b=min(a+500,info['n_all']);t=torch.tensor(provider.take(np.arange(a,b)),dtype=torch.float32,device=DEVICE)
            value=model(t);assert torch.isfinite(value).all();allpred[a:b]=value.detach().cpu().numpy();del t,value
            if a%5000==0 or b==info['n_all']:
                allpred.flush();save(all_receipt,{**expected_all,'completed':b,'status':'COMPLETE' if b==info['n_all'] else 'RUNNING'})
                event('STAGE1_ALL_INFERENCE',hour=h,completed=b,total=info['n_all'])
    allpred.flush()
    meta={**meta,'stage':'gradient_stage1_all','prediction_mode':'eval/no_grad, as reference all_test; test mode remains original training mode','all_definition':'all targets with six complete historical frames, no synthetic first five targets'}
    for c,ch in COMPONENTS.items():
        mean=ym[:,:,ch].transpose(2,0,1);std=ys[:,:,ch].transpose(2,0,1)
        for kind in ('testy','predicty'):
            def reader(a,b,kind=kind,ch=ch,mean=mean,std=std):
                source=vy[5+a:5+b,:,:,ch].transpose(0,3,1,2) if kind=='testy' else allpred[a:b,ch]
                return np.asarray(source,dtype=np.float64)*std+mean
            records.append(write_field(h,kind,c,'all',all_times,coords,reader,meta))
    for c in COMPONENTS:
        with netCDF4.Dataset(str(nc_path(h,'testy',c,'test'))) as ds1,netCDF4.Dataset(str(nc_path(h,'testy',c,'all'))) as ds2:
            for a in range(0,info['n_test'],512):np.testing.assert_array_equal(ds1['testy_'+c][a:a+512],ds2['testy_'+c][info['test_start']+a:info['test_start']+a+512])
    records.extend(artifact(hr/x) for x in ('stage1_normalization.npz','gradient_normalization.npz','gradient_normalization.json','gradient_definition.json','time_indices.npz','time_alignment.json'))
    records.extend(artifact(sr/x) for x in ('training_finished.json','training_identity.json','complete_epoch_state.json','epoch_metrics.csv','run_configuration.json'))
    training_evidence=load(sr/'training_finished.json')
    save(hr/'stage1_acceptance.json',{'status':'PASS','hour':h,'stage':1,'completed':now(),'elapsed_seconds':time.monotonic()-began,'artifacts':records,
      'checkpoint_source':{'weights':'best validation restored by original EarlyStopping','sha256':best_sha,'optimizer':'terminal training epoch, original behavior','actual_training_evidence':training_evidence},
      'test_and_all_same_best_checkpoint':True,'all_count':info['n_all'],'test_count':info['n_test'],'nc_count':12,'source_inputs_unchanged':True})
    clear_model_memory(model);outputs=model=prediction=truth=allpred=provider=builder=physical=vx=vy=None;gc.collect();clear_model_memory()
    event('STAGE_COMPLETE',hour=h,stage=1,acceptance=str(hr/'stage1_acceptance.json'))

def run_stage2(h):
    if acceptance_existing(h,2):event('REUSE_ACCEPTED_STAGE',hour=h,stage=2);return
    assert acceptance_existing(h,1),'Stage1 not fully accepted'
    began=time.monotonic();device=verify_device();hr=hour_root(h);sr=hr/'stage2';sr.mkdir(exist_ok=True)
    rawx,rawy,coords=source_cache();del rawy;info,all_times,test_times=split_info(h,coords)
    norm=np.load(hr/'stage1_normalization.npz');xm=norm['vx_mean'];xs=norm['vx_std']
    n,H,W=info['n_all'],len(coords['latitude']),len(coords['longitude'])
    inputs=np.empty((n,H,W,12),dtype=np.float32);labels=np.empty((n,H,W,9),dtype=np.float32)
    # Original standardized predictors reconstructed once with the saved original statistics.
    for a in range(0,n,512):
        b=min(a+512,n);inputs[a:b,:,:,:3]=(np.asarray(rawx[:,5+a:5+b],dtype=np.float32).transpose(1,2,3,0)-xm)/xs
    del rawx;gc.collect()
    for c,ch in COMPONENTS.items():
        with netCDF4.Dataset(str(nc_path(h,'testy',c,'all'))) as dt,netCDF4.Dataset(str(nc_path(h,'predicty',c,'all'))) as dp:
            for a in range(0,n,512):
                b=min(a+512,n);truth=np.asarray(dt['testy_'+c][a:b],dtype=np.float32);forecast=np.asarray(dp['predicty_'+c][a:b],dtype=np.float32)
                error=truth-forecast;assert np.isfinite(error).all() and np.isfinite(forecast).all()
                inputs[a:b,:,:,3+ch.start:3+ch.stop]=forecast.transpose(0,2,3,1);labels[a:b,:,:,ch]=error.transpose(0,2,3,1)
    prediction_inputs=inputs[:,:,:,3:]
    norm_path=hr/'stage2_normalization.npz'
    if norm_path.exists():
        with np.load(norm_path) as old:
            assert int(old['forecast_hours'])==h and str(old['scope'])=='training prediction input hours and residual labels'
            pm,ps_raw,rm,rs_raw=[old[k] for k in ('prediction_input_mean','prediction_input_std_raw','residual_mean','residual_std_raw')]
    else:
        stop=info['stage2_train_stop']+TIMESTEP-1
        pm=np.nanmean(prediction_inputs[:stop],axis=0);ps_raw=np.nanstd(prediction_inputs[:stop],axis=0)
        rm=np.nanmean(labels[TIMESTEP-1:stop],axis=0);rs_raw=np.nanstd(labels[TIMESTEP-1:stop],axis=0)
    for v in (pm,ps_raw,rm,rs_raw):assert np.isfinite(v).all()
    ps=np.where(ps_raw<=1e-12,np.float32(1),ps_raw);rs=np.where(rs_raw<=1e-12,np.float32(1),rs_raw)
    for a in range(0,n,512):
        b=min(a+512,n);prediction_inputs[a:b]=(prediction_inputs[a:b]-pm)/ps;labels[a:b]=(labels[a:b]-rm)/rs
    if not norm_path.exists():
        np.savez_compressed(norm_path,prediction_input_mean=pm,prediction_input_std=ps,prediction_input_std_raw=ps_raw,
          residual_mean=rm,residual_std=rs,residual_std_raw=rs_raw,vx_mean=xm,vx_std=xs,
          input_channels=np.array(['sp','t2m','d2m']+[f'gradient_stage1_pred_{c}{lev}' for c in COMPONENTS for lev in LEVELS.astype(int)]),
          scope='training prediction input hours and residual labels',forecast_hours=h,zero_variance_threshold=1e-12)
    provider=RAMWindowCache(ArrayWindowProvider(inputs,6),sr);provider.warm()
    import network_adapter as net
    kw={**COMMON,'test_size':info['stage2_test_size'],'valid_size':info['stage2_valid_size']}
    runtime=make_runtime(h,2,kw);net.GRADIENT_RUNTIME=runtime;net.WindowSelection=WindowSelection;seed25()
    save(sr/'run_configuration.json',{'hour':h,'stage':2,'device':device,'seed':25,'training':kw,'input_channels':12,
      'direct_gradient_channels':0,'direct_true_wind_input':False,'first_stage_fixed':True,'effective_history_steps':11,'time_split':info})
    temp_stem=sr/'trained_best'
    outputs=net.Auto_DA_TransUnet_pytorch(labels,inputs,6,4,2,**kw,if_best_mode='no',modelpath=None,ifrandom_split='no',ifsave='yes',savepath=str(temp_stem),window_provider=provider)
    model,prediction,residual_truth=outputs[:3];assert prediction.shape==residual_truth.shape==(info['n_test'],9,H,W)
    for a in range(0,info['n_test'],512):
        np.testing.assert_array_equal(residual_truth[a:a+512],labels[info['test_start']+a:info['test_start']+a+512].transpose(0,3,1,2))
    records=publish_models(h,2,temp_stem);best_sha=sha(Path(str(final_stem(h,2))+'.pth'))
    for c,ch in COMPONENTS.items():
        with netCDF4.Dataset(str(nc_path(h,'testy',c,'all'))) as dt,netCDF4.Dataset(str(nc_path(h,'predicty',c,'all'))) as dp:
            for kind in ('testy','predicty'):
                def reader(a,b,kind=kind,ch=ch,c=c):
                    sl=slice(info['test_start']+a,info['test_start']+b)
                    if kind=='testy':return np.asarray(dt['testy_'+c][sl],dtype=np.float32)
                    residual=np.asarray(prediction[a:b,ch],dtype=np.float64)*rs[:,:,ch].transpose(2,0,1)+rm[:,:,ch].transpose(2,0,1)
                    return np.asarray(dp['predicty_'+c][sl],dtype=np.float64)+residual
                meta={'stage':'gradient_residual_corrected_test','source_model':str(final_stem(h,2))+'.pth','source_model_sha256':best_sha,
                  'first_stage_model_sha256':sha(Path(str(final_stem(h,1))+'.pth')),'prediction_definition':'gradient stage1 prediction + inverse-standardized residual prediction',
                  'truth_definition':'original wind truth, not residual label','stage2_direct_gradient_channels':0,'effective_history_steps':11}
                records.append(write_field(h,kind,c,'final',test_times,coords,reader,meta))
    records.append(artifact(hr/'stage2_normalization.npz'))
    records.extend(artifact(sr/x) for x in ('training_finished.json','training_identity.json','complete_epoch_state.json','epoch_metrics.csv','run_configuration.json'))
    training_evidence=load(sr/'training_finished.json')
    save(hr/'stage2_acceptance.json',{'status':'PASS','hour':h,'stage':2,'completed':now(),'elapsed_seconds':time.monotonic()-began,'artifacts':records,
      'checkpoint_source':{'weights':'best validation restored by original EarlyStopping','sha256':best_sha,'optimizer':'terminal training epoch, original behavior','actual_training_evidence':training_evidence},
      'test_count':info['n_test'],'nc_count':6,'no_stage2_all_prediction':True,'final_prediction_is_complete_wind':True})
    clear_model_memory(model);outputs=model=prediction=residual_truth=provider=inputs=labels=None;gc.collect();clear_model_memory()
    event('STAGE_COMPLETE',hour=h,stage=2,acceptance=str(hr/'stage2_acceptance.json'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead', type=int, choices=range(1,7), default=1)
    parser.add_argument('--stage', choices=('cache','1','2','both'), default='both')
    args = parser.parse_args()
    if args.stage == 'cache':
        source_cache()
    if args.stage in ('1','both'):
        run_stage1(args.lead)
    if args.stage in ('2','both'):
        run_stage2(args.lead)

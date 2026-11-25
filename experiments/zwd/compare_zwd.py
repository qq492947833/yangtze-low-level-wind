"""Paired common-test metrics and 72-hour block uncertainty."""
from pathlib import Path
import os
from paths import ROOT, RUN
WORK = ROOT / "comparison"
WORK.mkdir(parents=True, exist_ok=True)
import numpy as np,netCDF4 as nc,json,hashlib,time,datetime,argparse,contextlib,psutil,gc
METRICS=['R','RMSE','MAE','RMSPE'];FEATURES=[c+str(p) for c in ['u','v','omega'] for p in [850,925,1000]]
REPS=1000;BLOCK=72;SEED=25

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,o):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2,default=str),encoding='utf-8');os.replace(t,p)
def status(phase,**kw):
 pr=psutil.Process();atomic(WORK/'status.json',dict(phase=phase,pid=pr.pid,create_time=pr.create_time(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rss=pr.memory_info().rss,**kw))
def guard():
 v=psutil.virtual_memory()
 if v.available/v.total<.15 or psutil.Process().memory_info().rss>20*1024**3:raise MemoryError('Comparison resource guard before next committed block')
 if psutil.disk_usage(ROOT).free<30*1024**3:raise OSError('Insufficient free space in the comparison output directory')
def blocks(times,length=BLOCK):
 times=np.asarray(times,dtype='int64');assert np.all(np.diff(times)>0)
 edges=np.r_[0,np.where(np.diff(times)!=1)[0]+1,len(times)];out=[]
 for segment,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
  for s in range(a,b,length):out.append((s,min(s+length,b),segment))
 return np.asarray(out,dtype='int64')
def sufficient(y,p,mask):
 n=mask.sum(0,dtype='float64');yy=np.where(mask,y,0);pp=np.where(mask,p,0);e=np.where(mask,p-y,0)
 return np.stack([n,yy.sum(0),pp.sum(0),(yy*yy).sum(0),(pp*pp).sum(0),(yy*pp).sum(0),(e*e).sum(0),np.abs(e).sum(0)])
def metric(s,rng):
 n,sy,sp,syy,spp,syp,sse,sae=np.moveaxis(s,-2,0)
 with np.errstate(divide='ignore',invalid='ignore'):
  vy=syy-sy*sy/n;vp=spp-sp*sp/n;cov=syp-sy*sp/n
  R=np.where((n>=3)&(vy>0)&(vp>0),np.clip(cov/np.sqrt(vy*vp),-1,1),np.nan)
  rmse=np.where(n>=2,np.sqrt(np.maximum(sse/n,0)),np.nan);mae=np.where(n>=2,sae/n,np.nan)
  rmspe=np.where((n>=2)&(rng>0),rmse/rng,np.nan)
 return np.stack([R,rmse,mae,rmspe],axis=-2)
def weights(bs,seed=SEED):
 rng=np.random.default_rng(seed);w=np.zeros((REPS,len(bs)),dtype='float64')
 for seg in np.unique(bs[:,2]):
  idx=np.where(bs[:,2]==seg)[0];draw=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=REPS);w[:,idx]=draw
 return w

def paths(h):return [ROOT/'checkpoints'/arm/f'lead_{h}'/'stage2'/'stage2_predictions.nc' for arm in ['A','B']]
def audit_sources():
 rec=[]
 for h in range(1,7):
  for arm in ['A','B']:
   for st in [1,2]:
    d=ROOT/'checkpoints'/arm/f'lead_{h}'/f'stage{st}';p=d/'completion.json';r=json.loads(p.read_text());assert r['status']=='PASS' and r['run_id']==RUN['run_id'] and (r['lead'],r['arm'],r['stage'])==(h,arm,st)
    for a in r['artifacts']:
     f=Path(a['path']);assert f.is_file() and f.stat().st_size==a['size'] and isinstance(a.get('sha256'),str) and sha(f)==a['sha256'],('Completed source artifact changed',str(f))
    rec.append({'path':str(p),'sha256':sha(p),'artifacts':r['artifacts']})
 with nc.Dataset(ROOT/'matched_sample_manifest.nc') as d:
  ts=np.asarray(d['target_time'][:]);keep=np.asarray(d['split'][:])==2;common=ts[keep];lat=np.asarray(d['latitude'][:]);lon=np.asarray(d['longitude'][:])
 signature={'manifest_sha256':sha(ROOT/'matched_sample_manifest.nc'),'source_completion_receipts':rec,'common_test_count':len(common),'time_first':int(common[0]),'time_last':int(common[-1]),'block_hours':BLOCK,'replicates':REPS,'seed':SEED,'method':'paired nonoverlapping time-block bootstrap, stratified by contiguous segment; terminal short blocks kept; fixed full common-test range','code_sha256':sha(__file__)}
 p=WORK/'contract.json'
 if p.exists():assert json.loads(p.read_text())==signature,'Comparison identity changed; cannot mix caches'
 else:atomic(p,signature)
 return common,lat,lon,signature

def lead_stats(h,times):
 d=WORK/f'lead_{h}';d.mkdir(exist_ok=True);bs=blocks(times);p=d/'block_stats.npy';ptr=d/'progress.json';q=d/'range.npy';rmin=d/'min.npy';rmax=d/'max.npy'
 if ptr.exists():
  state0=json.loads(ptr.read_text());assert state0['contract_sha256']==sha(WORK/'contract.json')
  assert isinstance(state0.get('next'),int) and 0<=state0['next']<=len(bs),'Invalid committed comparison progress'
  assert len(state0.get('block_sha256',[]))==state0['next'],'Committed comparison hash count mismatch'
  np.testing.assert_array_equal(np.load(d/'blocks.npy'),bs)
  mm=np.load(p,mmap_mode='r');assert mm.shape==(len(bs),4,8,9,49,73) and mm.dtype==np.float64;del mm
  # Verify only at comparison resume, never in the 60-second monitor.
  # Reopen each committed block to keep verification RAM bounded.
  for j,expected in enumerate(state0['block_sha256']):
   mm=np.load(p,mmap_mode='r');block=np.array(mm[j],copy=True);del mm
   assert hashlib.sha256(block.tobytes()).hexdigest()==expected,('Committed comparison block changed',j)
   del block
 else:
  if p.exists():raise RuntimeError('Unregistered block stats refuse overwrite')
  x=np.lib.format.open_memmap(p,mode='w+',dtype='float64',shape=(len(bs),4,8,9,49,73));x[:]=np.nan;x.flush();del x
  np.save(rmin,np.full((9,49,73),np.inf));np.save(rmax,np.full((9,49,73),-np.inf));state0={'next':0,'contract_sha256':sha(WORK/'contract.json'),'block_sha256':[]};atomic(ptr,state0);np.save(d/'blocks.npy',bs)
 with contextlib.ExitStack() as stack:
  ds=[stack.enter_context(nc.Dataset(f)) for f in paths(h)]
  for arm,z in zip(['A','B'],ds):
   assert z.complete==1 and z.arm==arm and z.lead_time==h and z.stage==2
   np.testing.assert_array_equal(z['target_time'][:],times);np.testing.assert_array_equal(z['init_time'][:],times-h)
   assert z.channel_order=='u850,u925,u1000,v850,v925,v1000,w850,w925,w1000'
  np.testing.assert_array_equal(ds[0]['latitude'][:],ds[1]['latitude'][:]);np.testing.assert_array_equal(ds[0]['longitude'][:],ds[1]['longitude'][:])
  # Float64 physical-unit conversion is applied exactly once to raw stored w.
  fac=np.ones((1,9,1,1));fac[:,6:]=-10
  for j in range(state0['next'],len(bs)):
   guard();a,b,_=bs[j];y0=np.asarray(ds[0]['testy'][a:b],np.float64);y1=np.asarray(ds[1]['testy'][a:b],np.float64);np.testing.assert_array_equal(y0,y1)
   y=y0*fac;pred=[np.asarray(ds[arm][name][a:b],np.float64)*fac for arm in [0,1] for name in ['stage1_prediction','final_prediction']]
   mask=np.isfinite(y)
   for v in pred:mask &= np.isfinite(v)
   # Committed output integrity requires no hidden NaN stage regions.
   assert mask.all(),'Accepted predictions contain missing/nonfinite fields'
   ss=np.stack([sufficient(y,v,mask) for v in pred]);mm=np.load(p,mmap_mode='r+');mm[j]=ss;mm.flush();np.testing.assert_array_equal(mm[j],ss);del mm
   lo=np.load(rmin);hi=np.load(rmax);lo=np.minimum(lo,y.min(0));hi=np.maximum(hi,y.max(0));np.save(rmin,lo);np.save(rmax,hi)
   state0['next']=j+1;state0['block_sha256'].append(hashlib.sha256(ss.tobytes()).hexdigest());atomic(ptr,state0);status('ACCUMULATING_PAIRED_METRICS',lead=h,block=j+1,total_blocks=len(bs))
 np.save(q,np.load(rmax)-np.load(rmin))
 # Validate selected spatial points directly against streaming totals.
 ss=np.load(p,mmap_mode='r');rng=np.load(q);checks=[]
 with nc.Dataset(paths(h)[0]) as a,nc.Dataset(paths(h)[1]) as b:
  for c,la,lo in [(0,0,0),(4,24,36),(8,48,72)]:
   fac1=-10 if c>=6 else 1.;y=np.asarray(a['testy'][:,c,la,lo],np.float64)*fac1
   for k,(z,name) in enumerate([(a,'stage1_prediction'),(a,'final_prediction'),(b,'stage1_prediction'),(b,'final_prediction')]):
    pp=np.asarray(z[name][:,c,la,lo],np.float64)*fac1;direct=metric(sufficient(y[:,None],pp[:,None],np.ones((len(y),1),bool)),np.array([np.ptp(y)]))[:,0]
    cached=metric(np.sum(ss[:,k,:,c,la,lo],axis=0)[:,None],np.array([rng[c,la,lo]]))[:,0]
    np.testing.assert_allclose(cached,direct,rtol=2e-10,atol=2e-11,equal_nan=True);checks.append({'point':[c,la,lo],'configuration':k,'max_abs':float(np.nanmax(abs(cached-direct)))})
 del ss;atomic(d/'direct_check.json',{'status':'PASS','checks':checks});return d

def aggregate_bootstrap(h,d):
 out=d/'metrics_and_bootstrap.npz';receipt=d/'metrics_completion.json'
 if receipt.exists():assert sha(out)==json.loads(receipt.read_text())['sha256'];return
 ss=np.load(d/'block_stats.npy',mmap_mode='r');bs=np.load(d/'blocks.npy');ra=np.load(d/'range.npy');W=weights(bs);N=49*73
 grid=np.full((4,4,9,N),np.nan);validcount=np.zeros((4,9,N),np.int64);boot_sum=np.zeros((REPS,4,4,9));boot_count=np.zeros_like(boot_sum)
 for c in range(9):
  for start in range(0,N,64):
   guard();end=min(N,start+64)
   # stats reshape selection is a bounded copy, not the full 2+GiB mapped file.
   tile=np.asarray(ss[:,:,:,c].reshape(len(bs),4,8,N)[...,start:end]);total=tile.sum(0);rr=ra[c].reshape(-1)[start:end]
   grid[:,:,c,start:end]=metric(total,rr);validcount[:,c,start:end]=total[:,0,:]
   rebuilt=(W @ tile.reshape(len(bs),-1)).reshape(REPS,4,8,end-start);bm=metric(rebuilt,rr)
   boot_sum[:,:,:,c]+=np.nansum(bm,axis=-1);boot_count[:,:,:,c]+=np.isfinite(bm).sum(-1)
  status('PAIRED_BLOCK_BOOTSTRAP',lead=h,feature=c+1,total_features=9,replicates=REPS)
 with np.errstate(invalid='ignore',divide='ignore'):boot_mean=np.where(boot_count>0,boot_sum/boot_count,np.nan)
 means=np.nanmean(grid,axis=-1);diff=np.stack([means[2]-means[0],means[3]-means[1]])
 bootdiff=np.stack([boot_mean[:,2]-boot_mean[:,0],boot_mean[:,3]-boot_mean[:,1]],axis=1)
 # Positive improvement: R augmented - baseline; error metrics baseline - augmented.
 diff[:,1:]*=-1;bootdiff[:,:,1:]*=-1
 ci=np.nanquantile(bootdiff,[.025,.975],axis=0)
 tmp=out.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez(f,grid=grid.reshape(4,4,9,49,73),count=validcount.reshape(4,9,49,73),reference_range=ra,spatial_means=means,difference=diff,bootstrap_means=boot_mean,bootstrap_differences=bootdiff,ci=ci,blocks=bs)
 os.replace(tmp,out);del ss;gc.collect();atomic(receipt,{'status':'PASS','sha256':sha(out),'replicates':REPS,'block_hours':BLOCK})

def write_metrics(times,lat,lon):
 p=ROOT/'zwd_matched_comparison_metrics.nc';tmp=p.with_suffix('.part.nc')
 with nc.Dataset(tmp,'w') as d:
  for n,s in [('lead_time',6),('configuration',4),('stage',2),('metric',4),('y_feature',9),('latitude',49),('longitude',73),('replicate',REPS),('bound',2)]:d.createDimension(n,s)
  for n,vals,typ in [('lead_time',np.arange(1,7),'i1'),('latitude',lat,'f8'),('longitude',lon,'f8'),('bound',[.025,.975],'f8')]:d.createVariable(n,typ,(n,))[:]=vals
  d.createVariable('pressure','i4',('y_feature',))[:]=[850,925,1000]*3
  specs={'grid_metrics':('lead_time','configuration','metric','y_feature','latitude','longitude'),'valid_count':('lead_time','configuration','y_feature','latitude','longitude'),'reference_range':('lead_time','y_feature','latitude','longitude'),'spatial_means':('lead_time','configuration','metric','y_feature'),'improvement':('lead_time','stage','metric','y_feature'),'paired_bootstrap_difference':('lead_time','replicate','stage','metric','y_feature'),'confidence_interval':('lead_time','bound','stage','metric','y_feature')}
  for n,dims in specs.items():d.createVariable(n,'i8' if n=='valid_count' else 'f8',dims,zlib=True,complevel=9,shuffle=True,fill_value=-1 if n=='valid_count' else np.nan)
  mapping={'grid_metrics':'grid','valid_count':'count','reference_range':'reference_range','spatial_means':'spatial_means','improvement':'difference','paired_bootstrap_difference':'bootstrap_differences','confidence_interval':'ci'}
  for h in range(1,7):
   with np.load(WORK/f'lead_{h}/metrics_and_bootstrap.npz') as z:
    for dst,src in mapping.items():d[dst][h-1]=z[src]
  d.run_id=RUN['run_id'];d.configuration_order='A_stage1,A_final,B_stage1,B_final';d.stage_order='stage1,final';d.metric_order=','.join(METRICS);d.y_feature_order=','.join(FEATURES);d.units='u/v RMSE MAE: m s-1; omega RMSE MAE: Pa s-1; R and RMSPE dimensionless';d.rmspe_definition='RMSE/common-test valid truth temporal range, per grid point; range fixed in bootstrap';d.delta_definition='R: B-A; RMSE/MAE/RMSPE: A-B; positive improvement';d.bootstrap_definition='72-hour non-overlapping blocks within each contiguous time segment; terminal short block retained; resample original block count independently in each segment; reconstruct moments then grid metrics then unweighted spatial means; seed25; 1000 replicates';d.uncertainty_scope='Fixed trained paired-seed models; test-period sampling sensitivity, not multiple-training-seed stability';d.common_test_count=len(times);d.source_contract_sha256=sha(WORK/'contract.json');d.complete=1
 with nc.Dataset(tmp) as d:assert d.complete==1 and d['grid_metrics'].shape==(6,4,4,9,49,73)
 os.replace(tmp,p);return p

def aggregate_predictions(arm,times,lat,lon):
 p=ROOT/('zwd_matched_baseline_predictions.nc' if arm=='A' else 'zwd_matched_augmented_predictions.nc');tmp=p.with_suffix('.part.nc');commit=WORK/f'{arm}_aggregate_progress.json';key=sha(WORK/'contract.json');names=['testy','stage1_prediction','residual_prediction','final_prediction']
 if p.exists():
  with nc.Dataset(p) as d:assert d.complete==1 and d.source_contract_sha256==key
  return p
 if tmp.exists():
  z=json.loads(commit.read_text());assert z['contract']==key;d=nc.Dataset(tmp,'r+')
 else:
  d=nc.Dataset(tmp,'w')
  for n,s in [('lead_time',6),('target_time',len(times)),('y_feature',9),('latitude',49),('longitude',73)]:d.createDimension(n,s)
  d.createVariable('lead_time','i1',('lead_time',))[:]=np.arange(1,7);d['lead_time'].units='hour'
  d.createVariable('target_time','i8',('target_time',))[:]=times;d['target_time'].units='hours since 1970-01-01 00:00:00'
  d.createVariable('init_time','i8',('lead_time','target_time'))[:]=times[None,:]-np.arange(1,7)[:,None];d['init_time'].units=d['target_time'].units
  d.createVariable('latitude','f8',('latitude',))[:]=lat;d.createVariable('longitude','f8',('longitude',))[:]=lon;d.createVariable('pressure','i4',('y_feature',))[:]=[850,925,1000]*3
  for n in names:d.createVariable(n,'f4',('lead_time','target_time','y_feature','latitude','longitude'),zlib=True,complevel=9,shuffle=True,fill_value=np.nan,chunksizes=(1,16,9,49,73))
  d.createVariable('lead_complete','i1',('lead_time',),fill_value=0)[:]=0;d.source_contract_sha256=key;d.run_id=RUN['run_id'];d.arm=arm;d.channel_order='u850,u925,u1000,v850,v925,v1000,w850,w925,w1000';d.units_by_feature='u/v m s-1; stored w=-0.1*omega, omega Pa s-1';d.final_prediction_definition='stage1 + inverse-standardized residual, exactly once';d.complete=0;d.sync();z={'contract':key,'next_lead':1};atomic(commit,z)
 try:
  for h in range(z['next_lead'],7):
   with nc.Dataset(paths(h)[0 if arm=='A' else 1]) as s:
    for a in range(0,len(times),72):
     guard();b=min(a+72,len(times))
     for n in names:d[n][h-1,a:b]=s[n][a:b]
    d.sync()
    # Re-read every written field before per-lead commit, preserving float32 exactly.
    for a in range(0,len(times),72):
     for n in names:np.testing.assert_array_equal(d[n][h-1,a:a+72],s[n][a:a+72])
   d['lead_complete'][h-1]=1;d.sync();z['next_lead']=h+1;atomic(commit,z);status('AGGREGATING_REQUIRED_PREDICTIONS',arm=arm,lead=h,total_leads=6)
  assert np.all(d['lead_complete'][:]==1);d.complete=1;d.sync()
 finally:d.close()
 os.replace(tmp,p);return p

def main():
    times,lat,lon,contract=audit_sources()
    for h in range(1,7):aggregate_bootstrap(h,lead_stats(h,times))
    outputs=[write_metrics(times,lat,lon)]+[aggregate_predictions(a,times,lat,lon) for a in ['A','B']]
    # CSV preserves units by reporting each variable-pressure cell separately.
    import csv
    csvp=ROOT/'zwd_matched_spatial_summary.csv'
    with csvp.open('w',newline='',encoding='utf-8-sig') as f:
     w=csv.writer(f);w.writerow(['lead','stage','feature','metric','improvement','ci025','ci975','baseline','augmented'])
     for h in range(1,7):
      with np.load(WORK/f'lead_{h}/metrics_and_bootstrap.npz') as z:
       for st in range(2):
        for mi,m in enumerate(METRICS):
         for c,feature in enumerate(FEATURES):w.writerow([h,['stage1','final'][st],feature,m,z['difference'][st,mi,c],z['ci'][0,st,mi,c],z['ci'][1,st,mi,c],z['spatial_means'][st,mi,c],z['spatial_means'][st+2,mi,c]])
    outputs.append(csvp)
    atomic(ROOT/'COMPARISON_COMPLETION.json',{'status':'PASS','run_id':RUN['run_id'],'common_test_count':len(times),'first_hour':int(times[0]),'last_hour':int(times[-1]),'artifacts':[{'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in outputs],'no_figures_created':True,'single_training_seed_limitation':True});status('COMPLETE')

if __name__ == '__main__':
    main()

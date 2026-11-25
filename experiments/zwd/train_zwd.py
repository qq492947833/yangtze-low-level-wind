"""Train one stage of a matched ERA5 (A) or ERA5 + ZWD (B) experiment."""
from pathlib import Path
import os, json, time, hashlib, random, datetime, argparse, traceback, gc
from paths import ROOT, OLD, CODE, DEVICE, RUN

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def atomic(p,o):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');b=json.dumps(o,ensure_ascii=False,indent=2,default=str).encode('utf8')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def state(dir,phase,**kw):
 import psutil
 p=psutil.Process();row=dict(time=stamp(),pid=p.pid,create_time=p.create_time(),phase=phase,rss=p.memory_info().rss,ram_available=psutil.virtual_memory().available);row.update(kw);atomic(dir/'status.json',row)
def save_torch(path,obj):
 import torch
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def clear_forward_tensors(model,initial_keys):
 import torch
 count=0
 for module in model.modules():
  for key,value in list(module.__dict__.items()):
   if torch.is_tensor(value) and key not in initial_keys[id(module)]:delattr(module,key);count+=1
 return count

def assert_same_state(a,b):
 import torch
 if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a.cpu(),b.detach().cpu())
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:assert_same_state(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):assert_same_state(x,y)
 else:assert a==b

def save_immutable_torch(path,obj):
 import torch
 if path.exists():assert_same_state(torch.load(path,map_location='cpu',weights_only=False),obj)
 else:save_torch(path,obj)

def setseed(seed=25):
 import numpy as np,torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():
 import numpy as np,torch
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
 import numpy as np,torch
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['torch_cuda']])
def signature():
 return {p.name:sha(p) for p in [CODE/'train_zwd.py',CODE/'model_definition.py',ROOT/'matched_sample_manifest.nc',ROOT/'normalization_statistics.nc',ROOT/'cache'/'identity.json']}
def global_science():
 protocol=dict(version='zwd_paired_v1',seed=25,batch=500,epochs=5000,patience=100,optimizer='Adam',learning_rate=.0001,loss='MSE',history=11,model_window=6,split='floor(.7N)/floor(.8N)',normalization='training-only unique causal inputs and common training targets; group-specific train residual/prediction',science_identity=signature())
 return hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()

def paired_init(arm,lead,stage,dir):
 import torch
 from model_definition import build_model
 channels=(3 if arm=='A' else 4) if stage==1 else 12
 setseed();m=build_model(channels,'cpu');changes=[]
 if arm=='B':
  p=ROOT/'checkpoints'/'A'/f'lead_{lead}'/f'stage{stage}'/'initial_state.pt'
  a=torch.load(p,map_location='cpu',weights_only=True);b=m.state_dict()
  for k in b:
   if b[k].shape==a[k].shape:b[k].copy_(a[k])
   elif b[k].ndim>=2 and a[k].shape[1]==3 and b[k].shape[1]==4 and a[k].shape[:1]+a[k].shape[2:]==b[k].shape[:1]+b[k].shape[2:]:
    b[k][:,:3].copy_(a[k]);changes.append(dict(name=k,A_shape=list(a[k].shape),B_shape=list(b[k].shape),shared_original_channels_equal=True,new_channel='original seeded initializer from B constructor, seed25'))
   else:raise ValueError(('Unexpected paired shape difference',k,a[k].shape,b[k].shape))
  m.load_state_dict(b)
 p=dir/'initial_state.pt'
 if p.exists():
  prior=torch.load(p,map_location='cpu',weights_only=True)
  for k,v in m.state_dict().items():assert torch.equal(v,prior[k]),('Initial state mismatch',k)
 else:save_torch(p,cpu_state(m))
 atomic(dir/'initialization.json',dict(seed=25,old_baseline_stage1_seed='not recorded; 25 is fixed for this new paired experiment',pairing='all shape-identical tensors copied from A initial state; shared first three inputs copied in expanded tensors',changes=changes,sha256=sha(p)))
 setseed();return m

class Data:
 def __init__(self,arm,lead,stage):
  import numpy as np,netCDF4
  self.np=np;self.arm=arm;self.lead=lead;self.stage=stage
  provenance=json.loads((ROOT/'cache'/'identity.json').read_text('utf8'))
  for rec in provenance['source_identities']+provenance['array_identities']:
   p=Path(rec['path']);st=p.stat();assert st.st_size==rec['size'] and st.st_mtime_ns==rec['mtime_ns'],('Data identity changed',str(p))
  m=dict(np.load(ROOT/'cache'/'manifest_indices.npz'));self.times=m['target_hours'];self.ids=m['retained_indices'];self.train=int(m['train_count']);self.valid=int(m['valid_stop']);self.offset=int(m['raw_offset'])
  self.coords=dict(np.load(OLD/'coordinates.npz'))
  with netCDF4.Dataset(ROOT/'normalization_statistics.nc') as d:self.norm={k:np.asarray(v[:]) for k,v in d.variables.items() if k.endswith(('_mean','_scale'))}
  self.aux=None;self.pnorm=None;self.rnorm=None
  if stage==2:
   p=ROOT/'checkpoints'/arm/f'lead_{lead}'/'stage1';q=json.loads((p/'auxiliary_completion.json').read_text('utf8'));assert q['status']=='PASS'
   assert q['identity']['arm']==arm and q['identity']['lead']==lead and q['identity']['manifest_sha256']==sha(ROOT/'matched_sample_manifest.nc')
   assert sha(p/'auxiliary_predictions.npy')==q['predictions_sha256'] and sha(p/'auxiliary_issue_indices.npy')==q['issues_sha256']
   self.auxdir=p;self.issue=np.load(p/'auxiliary_issue_indices.npy');self.auxmap=np.full(len(self.times),-1,np.int64);self.auxmap[self.issue]=np.arange(len(self.issue))
   self.fit_residual()
 def raw(self,ids,wind=False):
  # Reopen and unmap every bounded selection; avoid a whole-run mapped RSS accumulation.
  np=self.np;p=OLD/('raw_wind.npy' if wind else 'raw_predictors.npy');mm=np.load(p,mmap_mode='r')
  x=np.array(mm[:,ids+self.offset].transpose(1,0,2,3),copy=True);del mm;return x
 def zwd(self,ids):
  np=self.np;mm=np.load(ROOT/'cache'/'zwd_region.npy',mmap_mode='r');x=np.array(mm[ids],copy=True);del mm;return x
 def pred(self,ids):
  np=self.np;q=self.auxmap[ids];assert (q>=0).all(),'Missing auxiliary initialization prediction'
  mm=np.load(self.auxdir/'auxiliary_predictions.npy',mmap_mode='r');x=np.array(mm[q],copy=True);del mm;return x
 def targets(self,sample_indices):
  np=self.np;ii=self.ids[sample_indices];y=self.raw(ii,True).astype(np.float64)
  if self.stage==1:y=(y-self.norm['y_mean'])/self.norm['y_scale']
  else:y=(y-self.pred(ii-self.lead)-self.rnorm['mean'])/self.rnorm['scale']
  assert np.isfinite(y).all();return y.astype(np.float32)
 def windows_for_issues(self,issues):
  np=self.np;C=(3 if self.arm=='A' else 4) if self.stage==1 else 12;out=np.empty((len(issues),6,C,49,73),np.float32)
  for lag in range(6):
   q=issues-5+lag; x=self.raw(q).astype(np.float64);out[:,lag,:3]=((x-self.norm['x_mean'])/self.norm['x_scale']).astype(np.float32)
   if self.stage==1 and self.arm=='B':out[:,lag,3]=((self.zwd(q).astype(np.float64)-self.norm['zwd_mean'])/self.norm['zwd_scale']).astype(np.float32)
   if self.stage==2:out[:,lag,3:]=((self.pred(q).astype(np.float64)-self.pnorm['mean'])/self.pnorm['scale']).astype(np.float32)
  assert np.isfinite(out).all();return out
 def batch(self,sample_indices):return self.windows_for_issues(self.ids[sample_indices]-self.lead),self.targets(sample_indices)
 def fit_residual(self):
  np=self.np;dir=ROOT/'checkpoints'/self.arm/f'lead_{self.lead}'/'stage2';dir.mkdir(parents=True,exist_ok=True);p=dir/'residual_normalization.npz'
  if p.exists() and (dir/'residual_normalization.json').exists():
   r=json.loads((dir/'residual_normalization.json').read_text('utf8'));assert r['sha256']==sha(p) and r['manifest']==sha(ROOT/'matched_sample_manifest.nc') and r['stage1_aux']==sha(self.auxdir/'auxiliary_completion.json')
   z=dict(np.load(p));self.pnorm={'mean':z['prediction_mean'],'scale':z['prediction_scale']};self.rnorm={'mean':z['residual_mean'],'scale':z['residual_scale']};return
  trainissue=self.ids[:self.train]-self.lead;idx=np.unique((trainissue[:,None]+np.arange(-5,1)).reshape(-1))
  def fit(indices,fn):
   n=0;mean=np.zeros((9,49,73),np.float64);ss=np.zeros_like(mean)
   for a in range(0,len(indices),64):
    x=np.asarray(fn(indices[a:a+64]),np.float64);assert np.isfinite(x).all();k=len(x);u=x.mean(0);delta=u-mean;ss+=((x-u)**2).sum(0)+delta**2*n*k/(n+k);mean+=delta*k/(n+k);n+=k
   std=np.sqrt(ss/n);return mean,std,np.where(std==0,1,std),n
  pm,ps,pz,pn=fit(idx,self.pred);rm,rs,rz,rn=fit(trainissue,lambda q:self.raw(q+self.lead,True).astype(np.float64)-self.pred(q).astype(np.float64))
  tmp=p.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,prediction_mean=pm,prediction_std=ps,prediction_scale=pz,residual_mean=rm,residual_std=rs,residual_scale=rz,prediction_fit_issue_indices=idx,residual_fit_issue_indices=trainissue)
  os.replace(tmp,p);atomic(dir/'residual_normalization.json',dict(status='PASS',sha256=sha(p),manifest=sha(ROOT/'matched_sample_manifest.nc'),stage1_aux=sha(self.auxdir/'auxiliary_completion.json'),scope='only original training target samples and their six initialization records; no validation/test labels or statistics',prediction_fit_count=pn,residual_fit_count=rn))
  self.pnorm={'mean':pm,'scale':pz};self.rnorm={'mean':rm,'scale':rz}

def pearson_metric(y,x):
 import torch
 ym=torch.nanmean(y,dim=0,keepdim=True);xm=torch.nanmean(x,dim=0,keepdim=True)
 r=torch.nansum((y-ym)*(x-xm),dim=0,keepdim=True)/(torch.sqrt(torch.nansum((y-ym)**2,dim=0,keepdim=True))*torch.sqrt(torch.nansum((x-xm)**2,dim=0,keepdim=True)))
 return torch.mean((1-r)**1.5)

def checkpoint(dir,model,opt,epoch,best,bestepoch,bestloss,counter,fp,stagefp,terminal=False):
 import torch
 old=json.loads((dir/'complete_epoch.json').read_text('utf8')) if (dir/'complete_epoch.json').exists() else None
 slot='b' if old and Path(old['checkpoint']).name=='complete_epoch_a.pt' else 'a';p=dir/f'complete_epoch_{slot}.pt'
 payload=dict(full_state=True,science_fingerprint=fp,stage_fingerprint=stagefp,model=cpu_state(model),optimizer=opt.state_dict(),scheduler=None,scaler=None,rng=rng_state(),epoch=epoch,best=best,best_epoch=bestepoch,best_loss=bestloss,counter=counter,terminal=terminal,sample_order='fixed chronological',validation_mode='train with no_grad as original; no mode change',batch=500)
 save_torch(p,payload);atomic(dir/'complete_epoch.json',dict(full_state=True,science_fingerprint=fp,stage_fingerprint=stagefp,checkpoint=str(p),sha256=sha(p),epoch=epoch,best_epoch=bestepoch,early_stop_counter=counter,terminal=terminal,time=stamp()))

def infer(model,data,dir):
 import numpy as np,torch,netCDF4 as nc
 model.eval();stage=data.stage;h=data.lead
 if stage==1:
  issues=np.unique((data.ids[:,None]-h+np.arange(-5,1)).reshape(-1));dest=dir/'auxiliary_predictions.npy';ptr=dir/'inference_progress.json'
  issuepath=dir/'auxiliary_issue_indices.npy'
  if issuepath.exists():np.testing.assert_array_equal(np.load(issuepath),issues)
  else:
   with issuepath.with_suffix('.tmp').open('wb') as f:np.save(f,issues);f.flush();os.fsync(f.fileno())
   os.replace(issuepath.with_suffix('.tmp'),issuepath)
 else:issues=data.ids[data.valid:]-h;dest=dir/'residual_predictions.npy';ptr=dir/'inference_progress.json'
 ident=dict(best_sha256=sha(dir/'best_model.pth'),normalization_sha256=sha(ROOT/'normalization_statistics.nc'),manifest_sha256=sha(ROOT/'matched_sample_manifest.nc'),stage=stage,lead=h,arm=data.arm)
 if ptr.exists():s=json.loads(ptr.read_text('utf8'));assert s['identity']==ident
 else:
  initptr=dir/'inference_initialization.json';init=dict(identity=ident,shape=[len(issues),9,49,73],dtype='float32',destination=str(dest))
  if initptr.exists():assert json.loads(initptr.read_text('utf8'))==init
  elif dest.exists():raise RuntimeError('Prediction data lacks both progress and initialization identity')
  else:atomic(initptr,init)
  if dest.exists():
   # No committed inference block exists before progress publication; only completed NaN initialization is reusable.
   for a in range(0,len(issues),128):
    z=np.load(dest,mmap_mode='r');assert list(z.shape)==init['shape'] and z.dtype==np.float32 and np.isnan(z[a:a+128]).all();del z
  else:
   initializing=dest.with_suffix('.initializing.npy');z=np.lib.format.open_memmap(initializing,mode='w+',dtype=np.float32,shape=tuple(init['shape']));del z
   for a in range(0,len(issues),128):
    z=np.load(initializing,mmap_mode='r+');z[a:a+128]=np.nan;z.flush();del z
   os.replace(initializing,dest)
  s={'identity':ident,'next':0,'count':len(issues),'blocks':[]};atomic(ptr,s)
 assert s['count']==len(issues) and sum(q['stop']-q['start'] for q in s['blocks'])==s['next']
 z=np.load(dest,mmap_mode='r');assert z.shape==(len(issues),9,49,73) and z.dtype==np.float32;del z
 for q in s['blocks']:
  z=np.load(dest,mmap_mode='r');block=np.array(z[q['start']:q['stop']]);del z;assert hashlib.sha256(block.tobytes()).hexdigest()==q['sha256'];del block
 start=time.perf_counter();base=s['next']
 with torch.no_grad():
  for a in range(s['next'],len(issues),128):
   b=min(len(issues),a+128);x=data.windows_for_issues(issues[a:b]);pred=model(torch.as_tensor(x,device=DEVICE)).cpu().numpy().astype(np.float64)
   pred=pred* (data.norm['y_scale'] if stage==1 else data.rnorm['scale'])+(data.norm['y_mean'] if stage==1 else data.rnorm['mean']);assert np.isfinite(pred).all()
   z=np.load(dest,mmap_mode='r+');z[a:b]=pred.astype(np.float32);z.flush();np.testing.assert_array_equal(z[a:b],pred.astype(np.float32));del z
   s['blocks'].append(dict(start=a,stop=b,sha256=hashlib.sha256(pred.astype(np.float32).tobytes()).hexdigest()));s['next']=b;atomic(ptr,s);elapsed=time.perf_counter()-start;rate=(b-base)/elapsed;state(dir,'INFERENCE',completed=b,total=len(issues),samples_per_second=rate,eta_seconds=(len(issues)-b)/rate)
 if stage==1:
  atomic(dir/'auxiliary_completion.json',dict(status='PASS',identity=ident,rows=len(issues),predictions_sha256=sha(dest),issues_sha256=sha(dir/'auxiliary_issue_indices.npy'),no_grad_model_eval=True))
 # Save immutable stage-specific test fields; final aggregate is a separate postprocessing step.
 path=dir/('stage1_predictions.nc' if stage==1 else 'stage2_predictions.nc');tmp=path.with_suffix('.part.nc');testidx=data.ids[data.valid:];testissue=testidx-h
 if stage==1:loc=np.searchsorted(issues,testissue);assert np.array_equal(issues[loc],testissue)
 names=['testy','stage1_prediction']+(['residual_prediction','final_prediction'] if stage==2 else [])
 def verify_prediction_nc(candidate):
  checked=0
  with nc.Dataset(candidate) as d:
   d.set_auto_mask(False)
   assert d.complete==1 and d.arm==data.arm and int(d.lead_time)==h and int(d.stage)==stage
   assert d.manifest_sha256==ident['manifest_sha256'] and d.best_model_sha256==ident['best_sha256']
   np.testing.assert_array_equal(d['target_time'][:],data.times[testidx]);np.testing.assert_array_equal(d['init_time'][:],data.times[testissue])
   np.testing.assert_array_equal(d['latitude'][:],data.coords['latitude']);np.testing.assert_array_equal(d['longitude'][:],data.coords['longitude']);np.testing.assert_array_equal(d['pressure'][:],[850,925,1000]*3)
   for name in names:assert d[name].dtype==np.dtype('float32') and d[name].filters()['zlib'] and d[name].filters()['complevel']==9 and d[name].filters()['shuffle']
   for a in range(0,len(testidx),128):
    b=min(len(testidx),a+128);expected={'testy':data.raw(testidx[a:b],True)}
    z=np.load(dest,mmap_mode='r');p=np.array(z[loc[a:b]] if stage==1 else z[a:b]);del z
    if stage==1:expected['stage1_prediction']=p
    else:
     base=data.pred(testissue[a:b]);expected.update(stage1_prediction=base,residual_prediction=p,final_prediction=(base.astype(np.float64)+p.astype(np.float64)).astype(np.float32))
    for name,value in expected.items():
     actual=np.asarray(d[name][a:b]);assert np.isfinite(actual).all();np.testing.assert_array_equal(actual,value);checked+=actual.size
  return checked
 checked_values=None
 if not path.exists():
  with nc.Dataset(tmp,'w') as d:
   for name,size in [('target_time',len(testidx)),('y_feature',9),('latitude',49),('longitude',73)]:d.createDimension(name,size)
   d.createVariable('target_time','i8',('target_time',))[:]=data.times[testidx];d['target_time'].units='hours since 1970-01-01 00:00:00'
   d.createVariable('init_time','i8',('target_time',))[:]=data.times[testissue];d['init_time'].units='hours since 1970-01-01 00:00:00'
   d.createVariable('latitude','f8',('latitude',))[:]=data.coords['latitude'];d.createVariable('longitude','f8',('longitude',))[:]=data.coords['longitude'];d.createVariable('pressure','i4',('y_feature',))[:]=[850,925,1000]*3
   names=['testy','stage1_prediction']+(['residual_prediction','final_prediction'] if stage==2 else [])
   for name in names:d.createVariable(name,'f4',('target_time','y_feature','latitude','longitude'),zlib=True,complevel=9,shuffle=True,fill_value=np.nan,chunksizes=(16,9,49,73))
   for a in range(0,len(testidx),128):
    b=min(len(testidx),a+128);truth=data.raw(testidx[a:b],True);d['testy'][a:b]=truth
    z=np.load(dest,mmap_mode='r');pred=np.array(z[loc[a:b]] if stage==1 else z[a:b]);del z
    if stage==1:d['stage1_prediction'][a:b]=pred
    else:
     basepred=data.pred(testissue[a:b]);d['stage1_prediction'][a:b]=basepred;d['residual_prediction'][a:b]=pred;d['final_prediction'][a:b]=(basepred.astype(np.float64)+pred.astype(np.float64)).astype(np.float32)
   d.arm=data.arm;d.lead_time=h;d.stage=stage;d.units_by_feature='u/v m s-1; stored w=-0.1*omega, omega Pa s-1';d.channel_order='u850,u925,u1000,v850,v925,v1000,w850,w925,w1000';d.manifest_sha256=ident['manifest_sha256'];d.best_model_sha256=ident['best_sha256'];d.final_prediction_definition='stage1 plus inverse-standardized residual exactly once' if stage==2 else 'direct model physical prediction';d.complete=1
  checked_values=verify_prediction_nc(tmp)
  os.replace(tmp,path)
 with nc.Dataset(path) as d:
  d.set_auto_mask(False)
  assert d.complete==1 and d.arm==data.arm and int(d.lead_time)==h and int(d.stage)==stage
  assert d.manifest_sha256==ident['manifest_sha256'] and d.best_model_sha256==ident['best_sha256']
  np.testing.assert_array_equal(d['target_time'][:],data.times[testidx])
 receipt=dir/'prediction_receipt.json'
 if receipt.exists():
  q=json.loads(receipt.read_text('utf8'));assert q['identity']==ident and q['sha256']==sha(path)
 else:
  # Revalidate from scientific source if publication completed but its receipt did not.
  if checked_values is None:checked_values=verify_prediction_nc(path)
  atomic(receipt,dict(status='PASS',identity=ident,path=str(path),sha256=sha(path),exact_source_values_checked=checked_values,truth_prediction_and_final_sum_exact=True))
 return path

def train(args):
 import numpy as np,torch,psutil
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 if DEVICE.startswith('cuda'):torch.cuda.set_device(torch.device(DEVICE))
 assert str(torch.get_default_dtype())=='torch.float32' and not torch.is_autocast_enabled() and torch.backends.cuda.matmul.allow_tf32==False and torch.backends.cudnn.allow_tf32==True and not torch.backends.cudnn.benchmark
 dir=ROOT/'checkpoints'/args.arm/f'lead_{args.lead}'/f'stage{args.stage}';dir.mkdir(parents=True,exist_ok=True)
 if (dir/'completion.json').exists():
  c=json.loads((dir/'completion.json').read_text('utf8'));assert c['status']=='PASS' and c['science_fingerprint']==global_science()
  assert c['run_id']==RUN['run_id'] and c['arm']==args.arm and c['lead']==args.lead and c['stage']==args.stage
  for x in c['artifacts']:assert sha(x['path'])==x['sha256']
  return
 config=dict(run_id=RUN['run_id'],arm=args.arm,lead=args.lead,stage=args.stage,seed=25,batch_size=500,epochs=5000,early_stopping=100,optimizer='Adam',learning_rate=.0001,loss='MSE',precision='float32 no AMP; original TF32 defaults',science_identity=signature(),validation_behavior='original train mode under no_grad; batch loss arithmetic mean unweighted by last batch',sample_order='fixed chronological',norm='train-only',device=DEVICE)
 if args.stage==2:config['stage1_auxiliary_completion_sha256']=sha(ROOT/'checkpoints'/args.arm/f'lead_{args.lead}'/'stage1'/'auxiliary_completion.json')
 fp=global_science();config['science_fingerprint']=fp;stagefp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['stage_fingerprint']=stagefp
 configpath=dir/'run_configuration.json'
 if configpath.exists():assert json.loads(configpath.read_text('utf8'))==config,'Existing stage configuration changed; resume refused without altering saved state'
 else:atomic(configpath,config)
 state(dir,'LOADING_DATA',science_fingerprint=fp);data=Data(args.arm,args.lead,args.stage);m=paired_init(args.arm,args.lead,args.stage,dir).to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.0001);best=None;be=0;bl=float('inf');counter=0;epoch=0;terminal=False
 pointer=dir/'complete_epoch.json'
 if pointer.exists():
  assert args.resume,'Existing stage requires --resume'
  r=json.loads(pointer.read_text('utf8'));assert r['full_state'] and r['science_fingerprint']==fp and r['stage_fingerprint']==stagefp and sha(r['checkpoint'])==r['sha256']
  payload=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
  required={'full_state','science_fingerprint','stage_fingerprint','model','optimizer','scheduler','scaler','rng','epoch','best','best_epoch','best_loss','counter','terminal'};assert required.issubset(payload) and payload['full_state'] and payload['science_fingerprint']==fp and payload['stage_fingerprint']==stagefp
  m.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);restore_rng(payload['rng']);epoch=payload['epoch'];best={k:v.cpu() for k,v in payload['best'].items()} if payload['best'] else None;be=payload['best_epoch'];bl=payload['best_loss'];counter=payload['counter'];terminal=payload['terminal'];del payload
 else:checkpoint(dir,m,opt,0,best,be,bl,counter,fp,stagefp)
 initial_direct_tensor_keys={id(module):{k for k,v in module.__dict__.items() if torch.is_tensor(v)} for module in m.modules()}
 m.train();trainidx=np.arange(data.train);valididx=np.arange(data.train,data.valid)
 for ep in range(epoch+1,5001):
  if terminal:break
  start=time.perf_counter();tl=[];tm=[];vl=[];vm=[];wait=0.;stepsecs=0.
  for istrain,indices,losses,metrics in [(True,trainidx,tl,tm),(False,valididx,vl,vm)]:
   with torch.set_grad_enabled(istrain):
    for first in range(0,len(indices),500):
     tick=time.perf_counter();x,y=data.batch(indices[first:first+500]);wait+=time.perf_counter()-tick;tick=time.perf_counter();xt=torch.tensor(x,dtype=torch.float32,device=DEVICE);yt=torch.tensor(y,dtype=torch.float32,device=DEVICE);pred=m(xt);loss=torch.nn.functional.mse_loss(pred,yt);metric=pearson_metric(pred,yt)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite model loss')
     losses.append(loss.item());metrics.append(metric.item())
     if istrain:opt.zero_grad();loss.backward(retain_graph=True);opt.step()
     stepsecs+=time.perf_counter()-tick;del x,y,xt,yt,pred,loss,metric
  v=float(np.mean(vl));improved=v<=bl # original EarlyStopping accepts equal loss with delta=0
  if improved:bl=v;be=ep;counter=0;best=cpu_state(m)
  else:counter+=1
  terminal=counter>=100 or ep==5000;checkpoint(dir,m,opt,ep,best,be,bl,counter,fp,stagefp,terminal)
  elapsed=time.perf_counter()-start;row=dict(time=stamp(),epoch=ep,train_loss=float(np.mean(tl)),valid_loss=v,train_metric=float(np.nanmean(tm)),valid_metric=float(np.nanmean(vm)),best_epoch=be,best_validation_loss=bl,early_stop_counter=counter,epoch_seconds=elapsed,data_wait_seconds=wait,step_seconds=stepsecs,samples_per_second=(len(trainidx)+len(valididx))/elapsed,memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved())
  with (dir/'epoch_metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n');f.flush()
  state(dir,'TRAINING',**row);print(json.dumps(row),flush=True)
  if terminal:break
 assert best is not None;m.load_state_dict(best)
 for k,v in m.state_dict().items():assert torch.equal(v.detach().cpu(),best[k]),k
 save_immutable_torch(dir/'best_model.pth',cpu_state(m));save_immutable_torch(dir/'terminal_optimizer.pth',opt.state_dict());atomic(dir/'training_finished.json',dict(status='PASS',run_id=RUN['run_id'],lead=args.lead,arm=args.arm,stage=args.stage,completed_epochs=json.loads((dir/'complete_epoch.json').read_text('utf8'))['epoch'],best_epoch=be,best_validation_loss=bl,actual_best_weights_restored_and_equal=True,time=stamp()))
 # Model.forward stores intermediate tensors in __dict__ via original exec implementation.
 # Remove only non-registered direct tensor attributes after terminal state has been committed.
 clear_forward_tensors(m,initial_direct_tensor_keys)
 gc.collect();torch.cuda.empty_cache()
 path=infer(m,data,dir)
 paths=[dir/'best_model.pth',dir/'terminal_optimizer.pth',path,dir/'initial_state.pt',dir/'initialization.json',dir/'run_configuration.json',dir/'training_finished.json',dir/'prediction_receipt.json']
 if args.stage==1:paths += [dir/'auxiliary_predictions.npy',dir/'auxiliary_issue_indices.npy',dir/'auxiliary_completion.json']
 else:paths += [dir/'residual_normalization.npz',dir/'residual_normalization.json']
 artifacts=[dict(path=str(p),sha256=sha(p),size=p.stat().st_size) for p in paths]
 atomic(dir/'completion.json',dict(status='PASS',run_id=RUN['run_id'],lead=args.lead,arm=args.arm,stage=args.stage,science_fingerprint=fp,stage_fingerprint=stagefp,artifacts=artifacts,best_epoch=be,best_validation_loss=bl,best_restored_verified=True,time=stamp()));state(dir,'COMPLETE',science_fingerprint=fp)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['train']);p.add_argument('--lead',type=int,choices=range(1,7),required=True);p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--stage',type=int,choices=[1,2],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();d=ROOT/'checkpoints'/a.arm/f'lead_{a.lead}'/f'stage{a.stage}';d.mkdir(parents=True,exist_ok=True)
 try:train(a)
 except Exception as e:
  txt=traceback.format_exc();oom=isinstance(e,MemoryError) or 'out of memory' in str(e).lower();state(d,'ERROR',error_type='OOM' if oom else type(e).__name__,error=str(e),traceback=txt,lossless_recovery_pointer=str(d/'complete_epoch.json'));raise
if __name__=='__main__':main()

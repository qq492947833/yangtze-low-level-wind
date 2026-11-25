"""Direct11 one-lead worker: frozen chronological training with transactional resume."""
from direct11_common import *
import time
import argparse, gc, math, random, time, traceback
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from direct11_model import build_model,release_forward_cache
from direct11_data import Data

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([a.cpu() for a in r['cuda']])
def apply_precision(policy):
    assert policy['dtype']=='float32' and policy['autocast'] is False
    torch.backends.cuda.matmul.allow_tf32=policy['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=policy['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark=policy['cudnn_benchmark']
    torch.backends.cudnn.deterministic=policy['cudnn_deterministic']
def inputs(x,y,device):
    # Preserve the notebook's channels-last frame storage, not only logical N,T,C,H,W.
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
def prefetched(data,split,start,stop):
    """One lookahead logical batch; no RNG, no shuffle, no change to update groups."""
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='direct11_prefetch') as pool:
        lo=start
        pending=pool.submit(data.batch,split,lo,min(lo+500,stop)) if lo<stop else None
        while lo<stop:
            hi=min(lo+500,stop);x,y=pending.result()
            pending=pool.submit(data.batch,split,hi,min(hi+500,stop)) if hi<stop else None
            yield lo,hi,x,y
            lo=hi
def logical_train(model,opt,x_np,y_np,device,micro,directory):
    # Transfer each microbatch independently: a lower microbatch also lowers input residency.
    n=len(x_np);r=rng_state();original_micro=micro;ooms=[]
    while True:
        opt.zero_grad(set_to_none=True);total=0.0
        try:
            for start in range(0,n,micro):
                stop=min(n,start+micro);x,y=inputs(x_np[start:stop],y_np[start:stop],device)
                p=model(x);loss=torch.nn.functional.mse_loss(p,y)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training MSE')
                (loss*((stop-start)/n)).backward();total+=float(loss.detach())*((stop-start)/n)
                del x,y,p,loss;release_forward_cache(model)
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise FloatingPointError('Non-finite gradient')
            break
        except torch.cuda.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True);release_forward_cache(model)
            x=y=p=loss=None
            gc.collect();torch.cuda.empty_cache();restore_rng(r)
            ooms.append(dict(time=stamp(),error=repr(e),microbatch=micro,next_microbatch=max(1,micro//2),effective_batch=n,optimizer_updated=False))
            append(directory/'oom_recovery.jsonl',ooms[-1])
            if micro<=1 or len(ooms)>=10:raise
            micro=max(1,micro//2)
    # An optimizer OOM may partially mutate moments; propagate so the supervisor reloads a complete checkpoint.
    opt.step();opt.zero_grad(set_to_none=True)
    return total,micro
def logical_validation(model,x_np,y_np,device,micro,directory):
    n=len(x_np)
    while True:
        try:
            total=0.0
            with torch.no_grad():
                for start in range(0,n,micro):
                    stop=min(n,start+micro);x,y=inputs(x_np[start:stop],y_np[start:stop],device)
                    p=model(x);loss=torch.nn.functional.mse_loss(p,y)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite validation MSE')
                    total+=float(loss)*((stop-start)/n);del x,y,p,loss;release_forward_cache(model)
            return total,micro
        except torch.cuda.OutOfMemoryError as e:
            x=y=p=loss=None;release_forward_cache(model);gc.collect();torch.cuda.empty_cache()
            append(directory/'oom_recovery.jsonl',dict(time=stamp(),phase='validation',error=repr(e),microbatch=micro,next_microbatch=max(1,micro//2)))
            if micro<=1:raise
            micro=max(1,micro//2)
def payload(s,model,opt):
    return dict(s,full_state=True,model=cpu_clone(model.state_dict()),optimizer=cpu_clone(opt.state_dict()),rng=cpu_clone(rng_state()),scheduler=None,scaler=None)
def strip_payload(s):return {k:v for k,v in s.items() if k not in ('model','optimizer','rng','scheduler','scaler','full_state')}
def worker_identity(lead,physical_gpu,device,science_sha):
    return dict(lead=lead,torch_version=torch.__version__,logical_device=device,physical_gpu=physical_gpu,
                gpu_name=torch.cuda.get_device_name(0),frozen_science_sha=science_sha,started_utc=stamp())

def train(lead,physical_gpu):
    cfg,science=frozen();apply_precision(science['precision']);fingerprint=cfg['frozen_science_sha'];d=ROOT/'instances'/f'lead_{lead}';d.mkdir(parents=True,exist_ok=True)
    t=science['training'];assert (t['effective_batch'],t['learning_rate'],t['max_epochs'],t['patience'],t['optimizer'])==(500,0.0001,5000,100,'Adam')
    receipt=d/'TRAINING_COMPLETION.json'
    if receipt.exists():
        r=read(receipt);assert r['status']=='PASS_FROZEN_TRAINING_SELECTED' and r['frozen_science_sha']==fingerprint;assert_identity(r['selected_model'],True);return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for Direct11 training")
    device='cuda:0';torch.set_num_threads(int(os.environ.get('DIRECT11_CPU_THREADS','2')))
    micro=int(os.environ.get('DIRECT11_MICROBATCH','128'))
    if not 1<=micro<=500:raise ValueError("DIRECT11_MICROBATCH must be in [1,500]")
    seed=int(science['seeds'][str(lead)]['training_seed'])
    with lock(d/'training_state_writer.lock'):
        identity_record=worker_identity(lead,physical_gpu,device,fingerprint)
        atomic(d/'worker_identity.json',identity_record)
        seed_all(seed);model=build_model(11,3,device);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
        data=Data(lead);restored=load_checkpoint(d,fingerprint,lead)
        if restored is None:
            initial=save_torch(d/'initial_model.pt',model.state_dict())
            s=dict(lead=lead,frozen_science_sha=fingerprint,epoch=1,phase='train',cursor=0,global_step=0,trained_samples=0,best_loss=None,best_epoch=0,best_state=None,bad_epochs=0,train_loss_sum=0.0,train_batches=0,val_loss_sum=0.0,val_batches=0,history=[],started_utc=stamp(),microbatch=micro,effective_batch=500,seed=seed,initial_model=initial,gpu_history=[identity_record],elapsed_training_seconds=0.0)
            checkpoint(d,payload(s,model,optimizer))
        else:
            model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer']);restore_rng(restored['rng'])
            s=strip_payload(restored);s['microbatch']=min(micro,s['microbatch']);s['gpu_history'].append(identity_record)
            atomic(d/'last_restore.json',dict(status='PASS_MODEL_ADAM_RNG_CURSOR_EARLYSTOP_RESTORED',utc=stamp(),epoch=s['epoch'],phase=s['phase'],cursor=s['cursor'],global_step=s['global_step'],science=fingerprint))
        started=time.monotonic();last_tick=started;start_samples=s['trained_samples'];checkpoint_every=10
        def publish():
            nonlocal last_tick
            now=time.monotonic();s['elapsed_training_seconds']+=now-last_tick;last_tick=now
            rec=checkpoint(d,payload(s,model,optimizer));atomic(d/'status.json',dict(utc=stamp(),status='TRAINING' if s['phase']=='train' else s['phase'].upper(),lead=lead,physical_gpu=physical_gpu,pid=os.getpid(),epoch=s['epoch'],phase=s['phase'],cursor=s['cursor'],global_step=s['global_step'],trained_samples=s['trained_samples'],microbatch=s['microbatch'],effective_batch=500,best_epoch=s['best_epoch'],best_loss=s['best_loss'],bad_epochs=s['bad_epochs'],reliable_checkpoint=rec,samples_per_second_since_process_start=(s['trained_samples']-start_samples)/max(now-started,1e-9),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        while s['phase']!='complete':
            if s['phase']=='train':
                n=len(data.splits['train'])
                for lo,hi,x,y in prefetched(data,'train',s['cursor'],n):
                    assert lo==s['cursor']
                    loss,s['microbatch']=logical_train(model,optimizer,x,y,device,s['microbatch'],d)
                    del x,y;s['cursor']=hi;s['global_step']+=1;s['trained_samples']+=hi-lo;s['train_loss_sum']+=loss;s['train_batches']+=1
                    if s['global_step']%checkpoint_every==0:publish()
                s.update(phase='validation',cursor=0,val_loss_sum=0.0,val_batches=0);publish()
            if s['phase']=='validation':
                n=len(data.splits['valid'])
                # Original train mode + no_grad, each complete logical batch is one equal-weight validation observation.
                model.train()
                for lo,hi,x,y in prefetched(data,'valid',s['cursor'],n):
                    assert lo==s['cursor']
                    loss,s['microbatch']=logical_validation(model,x,y,device,s['microbatch'],d)
                    del x,y;s['cursor']=hi;s['val_loss_sum']+=loss;s['val_batches']+=1
                    if s['val_batches']%10==0:publish()
                val=s['val_loss_sum']/s['val_batches'];train_loss=s['train_loss_sum']/s['train_batches']
                if s['best_loss'] is None or val<=s['best_loss']:
                    s['best_loss']=val;s['best_epoch']=s['epoch'];s['best_state']=cpu_clone(model.state_dict());s['bad_epochs']=0
                else:s['bad_epochs']+=1
                row=dict(epoch=s['epoch'],train_mse_equal_logical_batches=train_loss,validation_mse_equal_logical_batches=val,best_epoch=s['best_epoch'],best_loss=s['best_loss'],bad_epochs=s['bad_epochs'],global_step=s['global_step'],microbatch=s['microbatch'],utc=stamp())
                s['history'].append(row);atomic(d/'training_history.json',s['history']);append(d/'epoch_log.jsonl',row);print(json.dumps(dict(lead=lead,**row)),flush=True)
                if s['bad_epochs']>=100 or s['epoch']>=5000:s.update(phase='complete',cursor=0)
                else:s.update(epoch=s['epoch']+1,phase='train',cursor=0,train_loss_sum=0.0,train_batches=0,val_loss_sum=0.0,val_batches=0)
                publish()
        assert s['best_state'] is not None
        best=save_torch(d/'best_validation_model.pt',s['best_state'])
        stopped=s['bad_epochs']>=100
        if stopped:model.load_state_dict(s['best_state'])
        selected=save_torch(d/'selected_model.pt',model.state_dict())
        selected_epoch=s['best_epoch'] if stopped else s['epoch']
        selected_val=s['best_loss'] if stopped else s['history'][-1]['validation_mse_equal_logical_batches']
        trained=dict(status='PASS_FROZEN_TRAINING_SELECTED',lead=lead,frozen_science_sha=fingerprint,task_id=cfg['task_id'],training_seed=seed,historical_seed_matching=science['seeds'][str(lead)]['historical_match'],selected_model=selected,selected_epoch=selected_epoch,selected_validation_mse=selected_val,stop_reason='early_stopping_best_validation' if stopped else 'epoch_cap_original_last_weight_rule',best_model=best,best_epoch=s['best_epoch'],best_validation_mse=s['best_loss'],completed_epochs=s['epoch'],global_step=s['global_step'],effective_batch=500,final_microbatch=s['microbatch'],training_started_utc=s['started_utc'],training_finished_utc=stamp(),elapsed_training_seconds=s['elapsed_training_seconds'],gpu_history=s['gpu_history'],early_stop_counter=s['bad_epochs'],sample_counts={k:len(v) for k,v in data.splits.items()},training_only=True,test_used_for_selection=False,model_parameter_count=sum(p.numel() for p in model.parameters()),initial_model=s['initial_model'],checkpoint=read(d/'reliable_checkpoint.json'),data_manifest_sha=sha(ROOT/'provenance/DATA_MANIFEST.json'))
        atomic(receipt,trained);print(json.dumps(dict(status=trained['status'],lead=lead,best_epoch=s['best_epoch'])),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["train","export"])
    p.add_argument("--lead",type=int,required=True,choices=range(1,7))
    p.add_argument("--device",choices=["cuda:0","cpu"],default="cuda:0",help="Output device for export; training uses cuda:0")
    a=p.parse_args()
    if a.command=="train":train(a.lead,os.getenv("CUDA_VISIBLE_DEVICES","0"))
    else:
        from direct11_export import export
        export(a.lead,a.device,os.getenv("CUDA_VISIBLE_DEVICES","0") if a.device!="cpu" else None)

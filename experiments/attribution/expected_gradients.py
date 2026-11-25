"""Bounded exact-output EG on one explicitly selected GPU. No production I/O."""
from pathlib import Path
import sys,time,os,hashlib
import numpy as np
import torch
from draws import frozen_draws

def configure_gpu(device):
 torch.cuda.set_device(device)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.set_float32_matmul_precision("highest")

def gpu_expected_gradients(model,x,backgrounds,indices,seed,nsamples,microbatch=2,mode='serial',progress=None,prefix_observer=None,resume_acc=None,resume_offset=0):
 assert x.device.type==backgrounds.device.type=='cpu' and x.dtype==backgrounds.dtype==torch.float32
 assert x.shape[0]==1 and x.shape[1]==6 and x.shape[2] in (3,12) and tuple(x.shape[3:])==(49,73) and tuple(backgrounds.shape[1:])==tuple(x.shape[1:]) and len(backgrounds)==100
 assert 1<=len(indices)<=32 and len(set(map(tuple,indices)))==len(indices)
 assert microbatch in (2,4,8,16,32,64,128) and mode in ('serial','batched_vjp')
 assert not any(m.training for m in model.modules()) and next(model.parameters()).device.type=='cuda'
 assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled()
 assert nsamples in (128,2048,4096,8192,16384,32768,65536,131072,262144) # 128 is isolated batch validation only; engine requires >=2048
 assert 0<=resume_offset<=nsamples and resume_offset%microbatch==0
 assert (resume_acc is None)==(resume_offset==0)
 draws=frozen_draws(seed,nsamples,len(backgrounds));device=next(model.parameters()).device
 shape=(len(indices),)+tuple(x.shape[1:])
 if resume_acc is None:acc=torch.zeros(shape,dtype=torch.float64,device=device)
 else:
  assert np.asarray(resume_acc).shape==shape and np.asarray(resume_acc).dtype==np.float64 and np.isfinite(resume_acc).all()
  acc=torch.from_numpy(np.asarray(resume_acc).copy()).to(device)
 eye=torch.eye(len(indices),device=device,dtype=torch.float32) if mode=='batched_vjp' else None
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();forward=backward=0
 path_seconds=0.0;events={k:[] for k in ('h2d','forward','backward_and_reduce','d2h')}
 def mark(stage):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();events[stage].append((a,b));return b
 try:
  for start in range(resume_offset,nsamples,microbatch):
   stop=min(start+microbatch,nsamples);tick=time.perf_counter();points=torch.empty((stop-start,)+tuple(x.shape[1:]),dtype=torch.float32);delta=torch.empty_like(points)
   # The installed SHAP interpolation is retained on CPU bit for bit. Only these
   # bounded paths and their float32 differences are transferred to the GPU.
   for local,k in enumerate(range(start,stop)):
    rind=int(draws['background_index'][k]);t=float(draws['alpha'][k]);xx=x[0]
    points[local]=t*xx+(1-t)*backgrounds[rind];delta[local]=xx-backgrounds[rind]
   path_seconds+=time.perf_counter()-tick
   end_event=mark('h2d');points=points.to(device).requires_grad_(True);delta=delta.to(device,dtype=torch.float64);end_event.record()
   end_event=mark('forward')
   model.zero_grad(set_to_none=True);outputs=model(points);forward+=1
   values=torch.stack([outputs[:,c,la,lo] for c,la,lo in indices],0).sum(1)
   end_event.record();end_event=mark('backward_and_reduce')
   if mode=='batched_vjp':
    grad=torch.autograd.grad(values,points,grad_outputs=eye,is_grads_batched=True)[0];backward+=1
    for row in range(stop-start):acc+=grad[:,row].double()*delta[row]
    del grad
   else:
    for j in range(len(indices)):
     grad=torch.autograd.grad(values[j],points,retain_graph=j+1<len(indices))[0];backward+=1
     for row in range(stop-start):acc[j]+=grad[row].double()*delta[row]
     del grad
   end_event.record();model.cleanup();del outputs,points,delta,values
   if prefix_observer is not None and (stop%512==0 or stop==nsamples):
    saved_acc=acc.detach().cpu().numpy().copy();prefix_observer(stop,saved_acc/stop,saved_acc)
   if progress is not None:progress(stop,nsamples)
  end_event=mark('d2h');phi=(acc/nsamples).cpu().numpy();end_event.record();torch.cuda.synchronize();elapsed=time.perf_counter()-started
  stage_seconds={k:sum(a.elapsed_time(b) for a,b in pairs)/1000 for k,pairs in events.items()}
 finally:model.cleanup()
 return phi,dict(mode=mode,microbatch=microbatch,output_width=len(indices),nsamples=nsamples,resumed_draws=resume_offset,new_draws=nsamples-resume_offset,background_count=100,effective_rseed=draws['effective_rseed'],forward_calls=forward,backward_calls=backward,seconds_synchronized=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),global_free_bytes=torch.cuda.mem_get_info()[0],cpu_path_preparation_seconds=path_seconds,cuda_stage_seconds=stage_seconds,timing_method='CUDA events recorded asynchronously; one final synchronization',signed_average_before_abs=True,accumulation='float64, original draw order',tf32=False,amp=False,device=str(device))

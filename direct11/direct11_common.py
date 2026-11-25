"""Runtime paths and checkpoint operations for the public Direct11 implementation."""
from pathlib import Path
import os, hashlib, json, sys, contextlib, datetime, uuid
from filelock import FileLock

ROOT = Path(os.getenv("YANGTZE_WORK_DIR", "./work")).resolve() / "direct11"
FINAL = Path(os.getenv("YANGTZE_RESULTS_DIR", "./results")).resolve() / "direct11h_single_stage"
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path), timeout=0)

def frozen():
    here = Path(__file__).resolve().parent
    science = dict(precision=dict(dtype="float32", autocast=False, matmul_allow_tf32=False,
                       cudnn_allow_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False),
                   training=dict(effective_batch=500,learning_rate=0.0001,max_epochs=5000,patience=100,optimizer="Adam"),
                   seeds={str(h):dict(training_seed=25,historical_match="unknown") for h in range(1,7)})
    fingerprint = hashlib.sha256(json.dumps([science, {p.name: sha(p) for p in sorted(here.glob("*.py"))},
                                   sha(ROOT/"provenance/DATA_MANIFEST.json")],sort_keys=True).encode()).hexdigest()
    return dict(task_id="direct11-release", frozen_science_sha=fingerprint), science

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def identity(path,hash_file=False):
    p=Path(path);s=p.stat();v=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    if hash_file:v['sha256']=sha(p)
    return v
def assert_identity(rec,hash_file=False):
    p=Path(rec['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(rec['size'],rec['mtime_ns']),str(p)
    if hash_file:assert sha(p)==rec['sha256'],str(p)
def atomic(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.pending')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    assert read(tmp)==obj;os.replace(tmp,p);assert read(p)==obj
def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def cpu_clone(value):
    import torch,numpy as np,copy
    if torch.is_tensor(value):return value.detach().to('cpu').clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)
def same(a,b):
    import torch,numpy as np
    if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)
def tensor_finite(value):
    import torch
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():tensor_finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:tensor_finite(v)
def save_torch(path,value):
    """Independent CPU snapshot, fsync, complete exact readback and atomic rename."""
    import torch
    p=Path(path);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.pending');snapshot=cpu_clone(value)
    tensor_finite(snapshot)
    with tmp.open('xb') as f:torch.save(snapshot,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(tmp,map_location='cpu',weights_only=False);same(snapshot,loaded);digest=sha(tmp)
    os.replace(tmp,p);return dict(**identity(p),sha256=digest)
def checkpoint(directory,payload):
    """Three slots never overwrite either current or previous reliable pointer."""
    d=Path(directory);pointer=d/'reliable_checkpoint.json';old=read(pointer) if pointer.exists() else None
    reserved={r['path'] for r in ([old,old.get('previous')] if old else []) if r}
    paths=[d/f'full_state_{i}.pt' for i in range(3)];p=next(p for p in paths if str(p) not in reserved)
    rec=save_torch(p,payload)
    previous={k:v for k,v in old.items() if k!='previous'} if old else None
    rec.update(status='PASS_FULL_STATE_READBACK',full_state=True,committed_utc=stamp(),lead=payload['lead'],epoch=payload['epoch'],phase=payload['phase'],cursor=payload['cursor'],global_step=payload['global_step'],frozen_science_sha=payload['frozen_science_sha'],previous=previous)
    atomic(pointer,rec);return rec
def load_checkpoint(directory,science_sha,lead):
    import torch
    p=Path(directory)/'reliable_checkpoint.json'
    if not p.exists():return None
    rec=read(p);failures=[]
    for r in [rec,rec.get('previous')]:
        if not r:continue
        try:
            assert_identity(r,True);v=torch.load(r['path'],map_location='cpu',weights_only=False)
            assert v['frozen_science_sha']==science_sha and v['lead']==lead and v['full_state']
            tensor_finite(v)
            if r is not rec:
                # Re-anchor to the verified fallback before any subsequent slot selection.
                # Otherwise the next save could retain the corrupt current as its previous.
                recovered=dict(r,previous=None,recovered_from_previous_at_utc=stamp())
                atomic(Path(directory)/('recovery_'+uuid.uuid4().hex+'.json'),dict(failed_current=rec,restored=r,failures=failures,utc=stamp()))
                atomic(p,recovered)
            return v
        except Exception as e:failures.append(repr(e))
    raise RuntimeError('No reliable checkpoint: '+repr(failures))

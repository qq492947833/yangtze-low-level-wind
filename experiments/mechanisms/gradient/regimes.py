from common import *
def build():
    dest=ROOT/'regime_inputs.nc'
    specs=[('Surface-pressure.nc','sp'),('2m-temperature.nc','t2m'),('2m-dewpoint-temperature.nc','d2m')]
    fields=[Field(RAW/n,v) for n,v in specs]
    raw=np.load(LEGACY/'hour_1'/'time_indices.npz')['raw_times']
    train=None
    for h in range(1,7):
        z=np.load(LEGACY/f'hour_{h}'/'time_indices.npz');a=json.loads((LEGACY/f'hour_{h}'/'time_alignment.json').read_text());issues=z['issue_times'][:a['train_stop']]
        train=issues if train is None else np.intersect1d(train,issues)
    assert len(train)>0
    inputs=[f.path for f in fields]+[LEGACY/f'hour_{h}'/'time_indices.npz' for h in range(1,7)]+[LEGACY/f'hour_{h}'/'time_alignment.json' for h in range(1,7)]
    contract=freeze_contract('regime_inputs',{'inputs':identity(inputs),'worker_sha':sha(__file__),'train_init_first':str(train[0]),'train_init_last':str(train[-1]),'train_init_count':len(train),'definition':'For each variable, mean spatial derivative magnitude over six raw historical frames and grid; temporal mean absolute derivative using only within-window six frames; divide each by fixed scalar train-input standard deviation then mean three variables. Tertiles from common legal training initializations only.','quantiles':[1/3,2/3],'history':6,'dt_seconds':3600})
    if dest.exists():
        with xr.open_dataset(dest) as d:assert d.attrs['contract']==contract
        return
    cache=ROOT/'chunks'/'features';cache.mkdir(exist_ok=True)
    train_raw=raw[(raw>=train[0]-np.timedelta64(5,'h'))&(raw<=train[-1])]
    norm=cache/'train_scalar_norm.npz'
    if norm.exists():
        z=np.load(norm);assert z['contract'].item()==contract;std=z['std']
    else:
        sums=np.zeros(3);squares=np.zeros(3);count=np.zeros(3)
        for lo in range(0,len(train_raw),168):
            memory_guard();ts=train_raw[lo:lo+168]
            for k,f in enumerate(fields):
                a=f.read(ts);assert np.isfinite(a).all();sums[k]+=a.sum();squares[k]+=(a*a).sum();count[k]+=a.size
        std=np.sqrt(squares/count-(sums/count)**2);assert (std>0).all();atomic_npz(norm,contract=np.array(contract),mean=sums/count,std=std,count=count)
    out=np.full((len(raw),2,3),np.nan)
    start=time.monotonic()
    for lo in range(5,len(raw),168):
        hi=min(lo+168,len(raw));p=cache/f'{lo:06d}_{hi:06d}.npz'
        if p.exists():
            with np.load(p) as z:assert z['contract'].item()==contract;out[lo:hi]=z['intensity']
            continue
        result=np.zeros((hi-lo,2,3));memory_guard()
        for k,f in enumerate(fields):
            a=f.read(raw[lo-5:hi]);assert np.isfinite(a).all()
            dx=secant(a,np.deg2rad(f.lon),-1)/(RADIUS*np.cos(np.deg2rad(f.lat))[None,:,None]);dy=secant(a,RADIUS*np.deg2rad(f.lat),-2)
            spatial=np.sqrt(dx*dx+dy*dy).mean((-2,-1))
            for j in range(hi-lo):
                w=a[j:j+6];dt=secant(w,np.arange(6)*3600.,0)
                result[j,0,k]=spatial[j:j+6].mean()/std[k]
                result[j,1,k]=np.abs(dt).mean()/std[k]
        atomic_npz(p,contract=np.array(contract),intensity=result);out[lo:hi]=result
        atomic_json(ROOT/'regime_inputs_status.json',{'status':'RUNNING','hours':hi,'total':len(raw),'elapsed':time.monotonic()-start,'memory':memory_guard()})
    intensity=out.mean(-1);trainix=np.searchsorted(raw,train);thresholds=np.quantile(intensity[trainix],[1/3,2/3],axis=0).T
    regime=np.full(intensity.shape,-1,np.int8)
    for j in range(2):
        good=np.isfinite(intensity[:,j]);regime[good,j]=np.searchsorted(thresholds[j],intensity[good,j],side='right')
    ds=xr.Dataset({'variable_intensity':(('init_time','regime_axis','x_feature'),out),'intensity':(('init_time','regime_axis'),intensity),'regime':(('init_time','regime_axis'),regime),'threshold':(('regime_axis','quantile'),thresholds),'train_input_scalar_std':('x_feature',std),'is_threshold_fit_initialization':('init_time',np.isin(raw,train).astype(np.uint8))},coords={'init_time':raw,'regime_axis':['spatial_gradient','time_change'],'x_feature':['sp','t2m','d2m'],'quantile':[1/3,2/3]},attrs={'contract':contract,'threshold_fit':'Common six-lead historical training input initialization population only. Physical differences before scalar standardization.','spatial_intensity_units':'m-1','temporal_intensity_units':'s-1','historical_window':'init-5h through init, no future or outside-window temporal difference','regime_definition':'0 low,1 middle,2 high using two predeclared training tertiles; -1 invalid','not_independent_observation':'Gradient channels are deterministic transforms of original inputs; conditional association is not causation.'})
    art=publish_nc(ds,dest);atomic_json(ROOT/'receipts'/'regime_inputs_completion.json',{'status':'PASS','contract':contract,'artifact':art,'training_initializations':len(train),'thresholds':thresholds.tolist()})
    for f in fields:f.close()
if __name__=='__main__':build()

"""Exact per-grid, own-reference conditional metrics and paired block bootstrap.

Only one lead/component/stage pair is read at a time. Base 24-hour moment blocks
are disk backed, allowing 72/24/168-hour whole-block draws without rereading NC.
All background computation is CPU; no training or model calls occur here.
"""
from common import *
from raw_wind import RawWindReference
import argparse,gc
METRIC=['R','RMSE','MAE','RMSPE']
CONDITION=['all','spatial_low','spatial_middle','spatial_high','temporal_low','temporal_middle','temporal_high']
SCALE=['original','box3_smooth','box3_remainder']
STAT=['n','truth_sum','prediction_sum','truth_square_sum','prediction_square_sum','cross_sum','squared_error_sum','absolute_error_sum']
def metric(s,ran):
    n=s[...,0];ok=n>=3
    def div(x):return np.divide(x,n,out=np.full_like(n,np.nan),where=ok)
    y=div(s[...,1]);p=div(s[...,2]);vy=div(s[...,3])-y*y;vp=div(s[...,4])-p*p;cov=div(s[...,5])-y*p
    rmse=np.sqrt(np.maximum(div(s[...,6]),0));mae=div(s[...,7])
    rr=np.divide(cov,np.sqrt(np.maximum(vy,0)*np.maximum(vp,0)),out=np.full_like(n,np.nan),where=ok&(vy>0)&(vp>0));rr=np.clip(rr,-1,1)
    rmspe=np.divide(rmse,ran,out=np.full_like(n,np.nan),where=ran>0)
    return np.stack([rr,rmse,mae,rmspe],-1)
def moments(y,p,mask):
    a=np.stack([np.ones_like(y),y,p,y*y,p*p,y*p,(p-y)**2,np.abs(p-y)],-1)
    return np.where(mask[...,None],a,0).sum(axis=0)
def expand_weights(times,base,hours,nrep=1000):
    coarse,w=resampling_weights(times,hours,nrep)
    index=np.empty(len(base),int)
    for bi,(a,b) in enumerate(base):
        ci=next(i for i,(c,d) in enumerate(coarse) if c<=a<d)
        assert b<=coarse[ci][1],'Base block crosses resampling block'
        index[bi]=ci
    return w[:,index]
def init_cache(path,shape):
    if path.exists():
        a=np.lib.format.open_memmap(path,mode='r+');assert a.shape==shape;return a
    need=int(np.prod(shape))*8
    if psutil.disk_usage(str(ROOT)).free < need+40*1024**3:raise OSError('Insufficient disk space for declared diagnostic cache plus reserve')
    a=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=shape)
    for i in range(shape[0]):
        a[i]=np.nan
        if i%24==23:
            a.flush();a._mmap.close();a=np.lib.format.open_memmap(path,mode='r+')
    a.flush();return a
def one(h,component,stage,truth_policy):
    tag=f'h{h}_{component}_s{stage}'
    job=ROOT/'chunks'/'gradient'/tag;job.mkdir(parents=True,exist_ok=True)
    done=job/'COMPLETION.json'
    suffixes=['','_gradient'] if stage==1 else ['_residual','_residual_gradient']
    fs=[[Field(RESULT/f'da_transunet_{k}_{component}_{h}_24year_changjiang{s}.nc') for k in ('testy','predicty')] for s in suffixes]
    ts = np.load(ROOT / "target_time.npy")
    reference = RawWindReference(RAW)
    lat,lon=fs[0][0].lat,fs[0][0].lon;ng=3*len(lat)*len(lon)
    for ff in sum(fs,[]):assert np.array_equal(ff.lat,lat) and np.array_equal(ff.lon,lon) and np.array_equal(ff.level,LEVELS)
    with xr.open_dataset(ROOT/'regime_inputs.nc') as d:
        origin=ts-np.timedelta64(h,'h');rid=d.regime.sel(init_time=origin).values;threshold=d.threshold.values;regimehash=sha(ROOT/'regime_inputs.nc')
    assert (rid>=0).all()
    base=partition_blocks(ts,24)
    inputs=identity([ff.path for ff in sum(fs,[])])
    body={'inputs':inputs,'worker_sha':sha(__file__),'common_sha':hashlib.sha256(ts.tobytes()).hexdigest(),'regime_sha':regimehash,'hour':h,'component':component,'stage':stage,'truth_policy':truth_policy,'baseblock':24,'bootstrap':[72,24,168,1000],'smooth':'predeclared3x3_equal_box_nearest','range':'fixed own raw full common test finite-support truth range for every scale and every draw','stat_dtype':'float64','seed':SEED}
    body['raw_files'] = [{k:r[k] for k in ('path','size','mtime_ns')} for r in reference.source_records]
    fp=fingerprint(body);cp=job/'contract.json'
    if cp.exists():assert json.loads(cp.read_text())['fingerprint']==fp,'Source or config changed'
    else:atomic_json(cp,{'fingerprint':fp,'contract':body})
    if done.exists():
        saved=json.loads(done.read_text())
        if saved['fingerprint']!=fp: raise ValueError('Case source or reference mode changed')
        for f in sum(fs,[]): f.close()
        reference.close()
        return
    shape=(len(base),2,len(CONDITION),len(SCALE),ng,8)
    cache=init_cache(job/'block_statistics.npy',shape)
    progress=job/'committed_blocks.json'
    state=json.loads(progress.read_text()) if progress.exists() else {'fingerprint':fp,'blocks':{}}
    assert state['fingerprint']==fp
    ymin=np.full((2,ng),np.inf);ymax=-ymin.copy();audit=[]
    start=time.monotonic();peak=0
    for bi,(lo,hi) in enumerate(base):
        rfile=job/f'range_{bi:04d}.npz'
        if str(bi) in state['blocks']:
            record=state['blocks'][str(bi)]
            assert hashlib.sha256(np.ascontiguousarray(cache[bi]).tobytes()).hexdigest()==record['statistics_sha256'],'Committed statistic block corrupted'
            assert sha(rfile)==record['range_sha256'],'Committed fixed range block corrupted'
            with np.load(rfile) as z:ymin=np.minimum(ymin,z['min']);ymax=np.maximum(ymax,z['max']);audit.extend(json.loads(z['audit'].item()))
            if bi%24==23:cache._mmap.close();cache=np.lib.format.open_memmap(job/'block_statistics.npy',mode='r+')
            continue
        mem=memory_guard();peak=max(peak,mem['rss']);tt=ts[lo:hi]
        data=np.stack([[ff.read(tt) for ff in row] for row in fs])
        if component=='w':data*=-10
        canonical = reference.read(tt, component)
        if not np.isfinite(canonical).all() or not np.isfinite(data).all():
            raise ValueError("Expected complete data over the common study grid")
        if truth_policy == "raw":
            data[:,0] = canonical
        group_eps = np.max(np.abs(data[:,0]-canonical[None]),axis=(1,3,4))
        common=np.isfinite(data).all((0,1));smooth_valid=minimum_filter(common.astype(np.uint8),size=(1,1,3,3),mode='nearest').astype(bool)
        mn=np.full((2,ng),np.inf);mx=-mn.copy()
        for group in range(2):
            y=data[group,0];mn[group]=np.where(common,y,np.inf).min(0).ravel();mx[group]=np.where(common,y,-np.inf).max(0).ravel()
        ymin=np.minimum(ymin,mn);ymax=np.maximum(ymax,mx)
        delta=np.abs(data[1,0]-data[0,0]);bad=np.isfinite(delta)&(delta!=0)
        ad=[{'eps':group_eps.tolist(),'first':str(tt[0]),'last':str(tt[-1]),'different_values':int(bad.sum()),'different_times':[str(x) for x in tt[bad.reshape(hi-lo,-1).any(1)]],'max_absolute':float(delta[np.isfinite(delta)].max(initial=0))}]
        audit.extend(ad)
        cond=np.stack([np.ones(hi-lo,bool)]+[rid[lo:hi,ax]==k for ax in range(2) for k in range(3)],1)
        for group in range(2):
            yy,pp=data[group];sy,sp=smooth(yy),smooth(pp)
            for scale,(y,p) in enumerate([(yy,pp),(sy,sp),(yy-sy,pp-sp)]):
                good=common if scale==0 else smooth_valid
                for ci in range(len(CONDITION)):
                    ma=good&cond[:,ci,None,None,None]
                    cache[bi,group,ci,scale]=moments(y,p,ma).reshape(ng,8)
        cache.flush();digest=hashlib.sha256(np.ascontiguousarray(cache[bi]).tobytes()).hexdigest()
        atomic_npz(rfile,min=mn,max=mx,audit=np.array(json.dumps(ad)))
        state['blocks'][str(bi)]={'statistics_sha256':digest,'range_sha256':sha(rfile)};atomic_json(progress,state)
        atomic_json(ROOT/'gradient_status.json',{'status':'BUILDING_MOMENTS','job':tag,'completed_hours':hi,'total_hours':len(ts),'elapsed_seconds':time.monotonic()-start,'samples_per_second':hi/max(time.monotonic()-start,.001),'memory':mem,'cache_bytes':cache.nbytes,'source_reference_difference_values':sum(x['different_values'] for x in audit)})
        if bi%24==0:print(f'{tag} moments {hi}/{len(ts)}',flush=True)
        if bi%24==23:cache._mmap.close();cache=np.lib.format.open_memmap(job/'block_statistics.npy',mode='r+')
    ran=ymax-ymin
    estimate=np.full((2,7,3,ng,4),np.nan);count=np.zeros((2,7,3,ng))
    bs_sums=np.zeros((3,1000,2,7,3,3,4));bs_counts=np.zeros_like(bs_sums)
    block_weights=[expand_weights(ts,base,b) for b in (72,24,168)]
    nspace=len(lat)*len(lon);tiles=job/'tiles';tiles.mkdir(exist_ok=True)
    tilesize=int(os.environ.get('YANGTZE_GRID_TILE','128'))
    for lo in range(0,ng,tilesize):
        hi=min(lo+tilesize,ng);pt=tiles/f'{lo:05d}_{hi:05d}.npz'
        if pt.exists():
            with np.load(pt) as z:
                assert z['fingerprint'].item()==fp;estimate[...,lo:hi,:]=z['estimate'];count[...,lo:hi]=z['count'];bs_sums+=z['bootstrap_sums'];bs_counts+=z['bootstrap_counts']
            continue
        memory_guard();a=np.asarray(cache[...,lo:hi,:]).copy();total=a.sum(0);rg=ran[:,None,None,lo:hi]
        met=metric(total,rg);commonmetric=np.isfinite(met).all(0,keepdims=True);met=np.where(commonmetric,met,np.nan)
        sums=np.zeros_like(bs_sums);counts=np.zeros_like(bs_counts)
        for bi,w in enumerate(block_weights):
            draws=(w@a.reshape(len(base),-1)).reshape((1000,)+a.shape[1:]);m=metric(draws,rg[None]);m=np.where(np.isfinite(m).all(1,keepdims=True),m,np.nan)
            for lev in range(3):
                ix=(np.arange(lo,hi)//nspace)==lev
                if ix.any():sums[bi,...,lev,:]=np.nansum(m[...,ix,:],axis=-2);counts[bi,...,lev,:]=np.isfinite(m[...,ix,:]).sum(axis=-2)
            del draws,m
        atomic_npz(pt,fingerprint=np.array(fp),estimate=met,count=total[...,0],bootstrap_sums=sums,bootstrap_counts=counts)
        estimate[...,lo:hi,:]=met;count[...,lo:hi]=total[...,0];bs_sums+=sums;bs_counts+=counts
        atomic_json(ROOT/'gradient_status.json',{'status':'BOOTSTRAPPING','job':tag,'grid_done':hi,'grid_total':ng,'elapsed_seconds':time.monotonic()-start,'memory':memory_guard(),'interval_statistic':'Exact reconstruction per-grid then equal spatial mean; no block-R averaging'})
        del a,total;gc.collect()
        cache._mmap.close();cache=np.lib.format.open_memmap(job/'block_statistics.npy',mode='r+')
    drawmean=np.divide(bs_sums,bs_counts,out=np.full_like(bs_sums,np.nan),where=bs_counts>0)
    gain=drawmean[:,:,1]-drawmean[:,:,0];gain[...,1:]*=-1
    grid=estimate.reshape(2,7,3,3,len(lat),len(lon),4);cnt=count.reshape(2,7,3,3,len(lat),len(lon))
    mean=np.nanmean(grid,axis=(-3,-2));delta=mean[1]-mean[0];delta[...,1:]*=-1
    dims=('group','condition','scale','level','latitude','longitude','metric')
    ds=xr.Dataset({'grid_metric':(dims,grid),'valid_time_count':(dims[:-1],cnt),'full_common_truth_range':(('group','level','latitude','longitude'),ran.reshape(2,3,len(lat),len(lon))),'spatial_mean':(('group','condition','scale','level','metric'),mean),'gain':(('condition','scale','level','metric'),delta),'paired_gain_ci':(('block_hours','bound','condition','scale','level','metric'),np.nanquantile(gain,[.025,.975],axis=1).swapaxes(0,1)),'bootstrap_gain':(('block_hours','replicate','condition','scale','level','metric'),gain),'regime':(('target_time','regime_axis'),rid),'regime_threshold':(('regime_axis','quantile'),threshold)},coords={'group':['baseline','gradient'],'condition':CONDITION,'scale':SCALE,'level':LEVELS,'latitude':lat,'longitude':lon,'metric':METRIC,'target_time':ts,'init_time':('target_time',ts-np.timedelta64(h,'h')),'regime_axis':['spatial_gradient','time_change'],'quantile':[1/3,2/3],'block_hours':[72,24,168],'bound':['lower','upper'],'replicate':np.arange(1000)},attrs={'status':'COMPLETE','forecast_hours':h,'stage':stage,'component':component,'units':'Pa s-1' if component=='w' else 'm s-1','fingerprint':fp,'own_truth':'Each group uses own prediction and own testy; paired common finite spatial/time support. No reference replacement.','rmspe':'RMSE / own original-field common-test truth range, fixed across conditions, transformed scales and resamples; not percent.','error_scale':'Original error = box3 smooth error + remainder. R also reported for corresponding transformed truth/prediction. Small-scale remainder is not automatically noise.','bootstrap':'Paired nonoverlapping72h blocks primary,24/168h sensitivity. Draw whole temporal blocks first, then apply precomputed conditional masks inside blocks; no blocks跨gap;1000replicates. Reconstruct per-grid R/RMSE/MAE, then equal spatial mean; fixed original-field range.','gain_positive':'R gradient-baseline; all errors baseline-gradient','training_randomness':'Intervals describe test-period resampling of single fixed trained models, not multi-seed reproducibility.','truth_audit':json.dumps(audit),'mechanism_attribution_status':'reference_difference_unresolved' if any(x['different_values'] for x in audit) else 'identical_reference_verified','smooth':'Predeclared3x3box,nearest-edge; all stencil members must be finite.'})
    eps = np.max(np.asarray([row["eps"] for row in audit]),axis=0)
    ds["truth_reference_max_abs"] = (("group","level"), eps)
    ds.attrs["truth_policy"] = truth_policy
    ds.attrs["own_truth"] = "Group-specific saved reference" if truth_policy=="native" else "Common raw ERA5"
    ds.attrs.pop("truth_audit",None)
    assert identity([ff.path for ff in sum(fs,[])])==inputs,'Source identity changed during diagnostic'
    art=publish_nc(ds,job/'result.nc')
    atomic_json(done,{'status':'PASS','truth_policy':truth_policy,'fingerprint':fp,'artifact':art,'hours':len(ts),'grid_points_per_level':nspace,'elapsed_seconds':time.monotonic()-start,'observed_peak_rss':peak,'source_identity_unchanged':identity([ff.path for ff in sum(fs,[])])==inputs,'reference_different_values':sum(x['different_values'] for x in audit)})
    for f in sum(fs,[]):f.close()
    reference.close()
    del cache

# Unified ERA5 comparisons named in the source selection for the mechanism figure.
RAW_REFERENCE_CASES = frozenset((1, h, "u") for h in (1, 2, 3, 4, 5))


def reference_for_case(stage, lead, component):
    return "raw" if (stage, lead, component) in RAW_REFERENCE_CASES else "native"


def main():
    parser = argparse.ArgumentParser(description="Score gradient-augmentation mechanisms on the common test hours.")
    parser.add_argument("--all", action="store_true", help="Prepare regimes and target times, score the 36 published cases, and assemble the figure input.")
    parser.add_argument("--lead", type=int, choices=range(1,7))
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("--stage", type=int, choices=(1,2))
    parser.add_argument("--reference", choices=("native","raw"), help="Override the reference for one individual case.")
    args = parser.parse_args()
    if args.all:
        if any(x is not None for x in (args.lead, args.component, args.stage, args.reference)):
            parser.error("--all selects all cases and their published reference fields")
        from regimes import build
        from target_times import main as align_times
        from assemble import main as assemble
        build()
        align_times()
        for stage in (1,2):
            for lead in range(1,7):
                for component in COMPONENTS:
                    one(lead, component, stage, reference_for_case(stage,lead,component))
        assemble(RESULT / "task2_diagnostics/mechanism_v4_20261028/gradient/gradient_mechanism_v4.nc")
        return
    if any(x is None for x in (args.lead,args.component,args.stage)):
        parser.error("Provide --all or specify --lead, --component and --stage")
    policy = args.reference or reference_for_case(args.stage,args.lead,args.component)
    one(args.lead,args.component,args.stage,policy)


if __name__ == "__main__":
    main()

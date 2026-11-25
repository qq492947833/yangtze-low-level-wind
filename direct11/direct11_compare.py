"""Matched Direct6, Direct11, TwoStage metrics on the six-lead native target intersection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from contextlib import ExitStack
from pathlib import Path

from direct11_common import ROOT, FINAL, atomic, identity, lock, read, sha, stamp
import netCDF4 as nc
import numpy as np

MODELS = ("Direct6", "Direct11", "TwoStage")
METRICS = ("R", "RMSE", "MAE", "RMSPE")
COMPONENTS = ("u", "v", "omega")
LEVELS = (850, 925, 1000)
PAIRS = ((1, 0), (2, 0), (2, 1))
SIGNS = np.array([1., -1., -1., -1.])
WORK = ROOT / "comparison"
BLOCK = 72
NREP = 1000
SEED_BASE = 20261025
HOUR_NS = 3600 * 10**9
# Packed columns n, sum(y), sum(y**2), then 5 per prediction:
# sum(p), sum(p**2), sum(y*p), sum(error**2), sum(abs(error)).
# All means/covariances use an origin held fixed across blocks.
N_SHARED = 3
N_MODEL = 5


def need(condition, message):
    if not condition:
        raise ValueError(message)


def block_design(times, hours=BLOCK, nrep=NREP):
    """Matches task2_common.partition_blocks/resampling_weights without its import-time writes."""
    times = np.asarray(times, dtype="datetime64[ns]")
    need(times.ndim == 1 and len(times) > 0 and np.all(times[1:] > times[:-1]), "Bad target order")
    edges = np.r_[0, np.flatnonzero(np.diff(times) != np.timedelta64(1, "h")) + 1, len(times)]
    blocks = np.asarray([(i, min(i + hours, end)) for start, end in zip(edges[:-1], edges[1:])
                         for i in range(start, end, hours)], dtype=np.int64)
    rng = np.random.default_rng(SEED_BASE + hours)
    weights = rng.multinomial(len(blocks), np.full(len(blocks), 1. / len(blocks)), size=nrep)
    return blocks, weights


def block_stats(truth, predictions, truth_origin, prediction_origins):
    """All models and leads share each point-time mask. First axis of an input field is time."""
    y = np.asarray(truth, dtype=np.float64)
    good = np.isfinite(y)
    for p in predictions:
        need(p.shape == y.shape, "Prediction shape")
        good &= np.isfinite(p)
    need(len(predictions) == len(prediction_origins), "Origins mismatch")
    z = np.where(good, y - truth_origin, 0.)
    stat = np.empty((N_SHARED + N_MODEL * len(predictions), *y.shape[1:]), dtype=np.float64)
    stat[0] = np.sum(good, axis=0)
    stat[1] = z.sum(axis=0)
    stat[2] = np.einsum("i...,i...->...", z, z)
    for m, (p, origin) in enumerate(zip(predictions, prediction_origins)):
        a = np.where(good, p - origin, 0.)
        e = np.where(good, p - y, 0.)
        dst = stat[N_SHARED + N_MODEL*m : N_SHARED + N_MODEL*(m+1)]
        dst[0] = a.sum(axis=0)
        dst[1] = np.einsum("i...,i...->...", a, a)
        dst[2] = np.einsum("i...,i...->...", z, a)
        dst[3] = np.einsum("i...,i...->...", e, e)
        dst[4] = np.abs(e).sum(axis=0)
    lo = np.min(np.where(good, y, np.inf), axis=0)
    hi = np.max(np.where(good, y, -np.inf), axis=0)
    return stat, lo, hi


def scores(stat, span):
    """Input [replicate,stat,cell], output [replicate,model,metric,cell]."""
    n, sy, syy = stat[:, 0], stat[:, 1], stat[:, 2]
    count = (stat.shape[1] - N_SHARED) // N_MODEL
    result = np.full((len(n), count, len(METRICS), n.shape[-1]), np.nan)
    ok = n >= 3
    ndiv = np.where(ok, n, 1.)
    vary = syy - sy * sy / ndiv
    for m in range(count):
        base = N_SHARED + N_MODEL * m
        sp, spp, cross, se, ae = np.moveaxis(stat[:, base:base+N_MODEL], 1, 0)
        varp = spp - sp * sp / ndiv
        den = np.sqrt(np.maximum(vary, 0.)) * np.sqrt(np.maximum(varp, 0.))
        # Reject numerical zero when a nonzero constant has accumulated roundoff.
        valid_r = ok & (vary > 32*np.finfo(float).eps*syy) & (varp > 32*np.finfo(float).eps*spp) & (den > 0)
        np.divide(cross - sy * sp / ndiv, den, out=result[:, m, 0], where=valid_r)
        finite = np.isfinite(result[:, m, 0])
        need(np.all(np.abs(result[:, m, 0][finite]) <= 1. + 1e-8), "Pearson bound exceeded")
        np.clip(result[:, m, 0], -1., 1., out=result[:, m, 0])
        np.sqrt(np.maximum(se, 0.) / ndiv, out=result[:, m, 1], where=ok)
        np.divide(ae, ndiv, out=result[:, m, 2], where=ok)
        np.divide(result[:, m, 1], span, out=result[:, m, 3], where=ok & (span > 0))
    return result


def gains(a, model_axis, metric_axis):
    """Keep axis order, replacing model with candidate/control pair."""
    out = np.stack([np.take(a, cand, axis=model_axis) - np.take(a, ctrl, axis=model_axis)
                    for cand, ctrl in PAIRS], axis=model_axis)
    signshape = [1] * a.ndim
    signshape[metric_axis] = len(METRICS)
    return out * SIGNS.reshape(signshape)


def summaries(stacked, span, weights, leads, tile=32):
    """Summarize a single component-level over space for all leads and models jointly.

    stacked holds one [stat,cell] view per chronological time block. Replicate zero is the
    complete test sample. Each other replicate reconstructs its own grid-time metrics.
    """
    ncell = span.size
    methods = leads * len(MODELS)
    sums = np.zeros((1+len(weights), leads, len(MODELS), len(METRICS)))
    counts = np.zeros((1+len(weights), len(METRICS)), dtype=np.int64)
    grid = np.full((leads, len(MODELS), len(METRICS), ncell), np.nan)
    mask = np.zeros((len(METRICS), ncell), dtype=bool)
    weights = np.asarray(weights, dtype=np.float64, order="C")
    for start in range(0, ncell, tile):
        stop = min(start+tile, ncell)
        packed = np.stack([s[:, start:stop] for s in stacked])
        actual = packed.sum(axis=0, dtype=np.float64)
        packed2d = packed.reshape(len(stacked), -1)
        # Small replica batches bound peak working set at production size.
        for lo in range(0, len(weights)+1, 40):
            hi = min(lo+40, len(weights)+1)
            selected = weights[max(0,lo-1):hi-1] @ packed2d
            stats = selected.reshape(-1, actual.shape[0], stop-start)
            if lo == 0:
                stats = np.concatenate((actual[None], stats), axis=0)
            field = scores(stats, span[start:stop])
            need(field.shape[1] == methods, "Unexpected model/lead axes")
            common = np.all(np.isfinite(field), axis=1)
            shaped = field.reshape(len(stats), leads, len(MODELS), len(METRICS), stop-start)
            sums[lo:hi] += np.where(common[:,None,None], shaped, 0.).sum(axis=-1)
            counts[lo:hi] += common.sum(axis=-1)
            if lo == 0:
                grid[...,start:stop] = shaped[0]
                mask[:,start:stop] = common[0]
    means = np.full_like(sums, np.nan)
    np.divide(sums, counts[:,None,None,:], out=means, where=counts[:,None,None,:] > 0)
    return grid, mask, means, counts


def ns_time(var):
    a = np.ma.asarray(var[:])
    need(not np.ma.getmaskarray(a).any(), "Missing timestamp")
    unit = str(var.units)
    if unit.startswith("seconds since 1970-01-01") and np.issubdtype(a.dtype, np.integer):
        return a.data.astype(np.int64) * 10**9
    vals = nc.num2date(a.data, unit, calendar=getattr(var, "calendar", "standard"),
                       only_use_cftime_datetimes=False)
    return np.array([str(v) for v in vals], dtype="datetime64[ns]").astype(np.int64)


def check_file(rec):
    p = Path(rec["path"])
    st = p.stat()
    need(st.st_size == rec.get("size",rec.get("size_bytes")) and st.st_mtime_ns == rec["mtime_ns"],
         "Source changed: " + str(p))
    return p


def source_contract():
    from direct11_common import frozen
    from direct11_data import Data
    cfg, science = frozen()
    science_sha = cfg["frozen_science_sha"]
    manifest = read(ROOT/"provenance/DATA_MANIFEST.json")
    for rec in manifest["sources"]+manifest["products"]:
        check_file(rec)
    indexfile = ROOT/"provenance/direct11_indices.npz"
    with np.load(indexfile, allow_pickle=False) as z:
        axes = {k:z[k] for k in ("latitude","longitude","level","raw_time_ns","common_test_target_idx")}
        axes["test"] = [z[f"h{h}_test_target_idx"] for h in range(1,7)]
        axes["issue"] = [z[f"h{h}_test_issue_idx"] for h in range(1,7)]
    computed = axes["test"][0]
    for t in axes["test"][1:]:
        computed = np.intersect1d(computed, t)
    common = axes["common_test_target_idx"]
    np.testing.assert_array_equal(computed, common)
    need(np.all(np.diff(common)==1), "Common native target index is discontinuous")
    np.testing.assert_array_equal(axes["level"], LEVELS)
    results = FINAL.parent
    records, paths = [], []
    for h in range(1,7):
        life = ROOT/"instances"/f"lead_{h}"
        trainfile = life/"TRAINING_COMPLETION.json"
        donefile = life/"COMPLETION.json"
        train, done = read(trainfile), read(donefile)
        need(train["status"]=="PASS_FROZEN_TRAINING_SELECTED" and done["status"]=="PASS_FINAL_NC_READBACK"
             and train["lead"]==done["lead"]==h and train["frozen_science_sha"]==done["frozen_science_sha"]==science_sha,
             "Final model/NC acceptance missing")
        selected=check_file(train["selected_model"])
        need(sha(selected)==train["selected_model"]["sha256"],"Changed selected model")
        need(done["training_receipt_sha256"]==sha(trainfile) and done["selected_model"]["sha256"]==train["selected_model"]["sha256"],"Export did not bind current selected training receipt")
        expected = {FINAL/f"Direct11_{h}hour_{kind}.nc" for kind in ("predicty","testy")}
        finals = {check_file(r) for r in done["artifacts"]}
        need(expected==finals and all(len(r["sha256"])==64 for r in done["artifacts"]), "Final file identities")
        chosen = {}
        source_ids = []
        for comp in "uvw":
            for role,suffix in (("stage1_aux","all"),("twostage","residual")):
                path=results/f"da_transunet_predicty_{comp}_{h}_24year_changjiang_{suffix}.nc"
                if not path.is_file(): raise FileNotFoundError(path)
                chosen[role,comp]=path; source_ids.append(identity(path))
        chosen["direct11"] = FINAL/f"Direct11_{h}hour_predicty.nc"
        chosen["truth"] = FINAL/f"Direct11_{h}hour_testy.nc"
        records.append(dict(lead=h,training_receipt=identity(trainfile,True),
                            completion_receipt=identity(donefile,True),selected_model=train["selected_model"],
                            selected_epoch=train["selected_epoch"],stop_reason=train["stop_reason"],
                            final_artifacts=done["artifacts"],historical_prediction_inputs=source_ids))
        paths.append(chosen)
    contract=dict(schema=1,code=identity(__file__,True),
                  data_manifest=identity(ROOT/"provenance/DATA_MANIFEST.json",True),
                  science_sha256=science_sha,records=records,block_hours=BLOCK,replicates=NREP,seed=SEED_BASE+BLOCK,
                  signed_pairs=[f"{MODELS[c]}_vs_{MODELS[b]}" for c,b in PAIRS],
                  methods="All 18 predictions and raw truth jointly finite per target and cell; common metric-specific finite spatial mask across all models and leads; full common truth range fixed for point estimate and all resamples; no mixed-component averaging.")
    return axes, paths, contract, Data(1)


def values(var, start, stop):
    return np.ma.asarray(var[start:stop],dtype=np.float64).filled(np.nan)


def sources(stack, axes, paths):
    """Open raw-source-linked predictions; validate all coordinates before evaluating a field."""
    times = axes["raw_time_ns"]
    common = times[axes["common_test_target_idx"]]
    entries = []
    for h, paths_h in enumerate(paths,1):
        native = times[axes["test"][h-1]]
        bsha = None
        training=read(ROOT/"instances"/f"lead_{h}"/"TRAINING_COMPLETION.json")
        windows = {}
        for name, path in paths_h.items():
            d=stack.enter_context(nc.Dataset(path,'r'))
            targets=ns_time(d["time"])
            if isinstance(name, tuple) and name[0]=="stage1_aux":
                need(np.all(np.diff(targets)==HOUR_NS), "Historical all time discontinuity")
            else:
                np.testing.assert_array_equal(targets,native)
            start=int(np.searchsorted(targets,common[0]))
            np.testing.assert_array_equal(targets[start:start+len(common)],common)
            for key in ("level", "latitude","longitude"):
                np.testing.assert_array_equal(d[key][:],axes[key])
            if isinstance(name,tuple):
                role,comp=name
                source_sha=str(d.source_model_sha256)
                if bsha is None: bsha=source_sha
                need(bool(source_sha) and source_sha==bsha and d.stage==("baseline_all" if role=="stage1_aux" else "residual_corrected_test"),"Historical model mismatch")
                var = d["predicty_"+comp]
                variables=[var]
            else:
                need(d.selected_model_sha256==training["selected_model"]["sha256"] and float(d.downstream_w_scale)==1., "Export/selected model or omega convention mismatch")
                np.testing.assert_array_equal(ns_time(d["forecast_reference_time"]),times[axes["issue"][h-1]])
                lead_value = getattr(d,"lead_hours",None)
                if lead_value is None:
                    need("forecast_period" in d.variables, "Missing forecast period")
                    v=d["forecast_period"];need(str(v.units).lower() in ("hour","hours","h"), "Forecast period units")
                    lead_value=int(np.asarray(v[...]).item())
                need(int(lead_value)==h,"Wrong lead")
                variables=[d[x] for x in "uvw"]
                need(all(str(v.units) == unit for v,unit in zip(variables,["m s-1","m s-1","Pa s-1"])),"Direct11 physical units")
            need(all(v.dimensions==("time","level","latitude","longitude") for v in variables),"Unexpected wind axes")
            windows[name]=(variables,start)
        entries.append(windows)
    return entries


def load_chunk(entries, data, common_idx, lo, hi):
    truth = data.physical_truth(common_idx[lo:hi]-1)
    predictions=[]
    for e in entries:
        for role in ("stage1_aux","direct11","twostage"):
            if role=="direct11":
                var,origin=e[role]
                pred=np.concatenate([values(v,lo+origin,hi+origin) for v in var],axis=1)
                var,origin=e["truth"]
                for i,v in enumerate(var):
                    a=values(v,lo+origin,hi+origin)
                    np.testing.assert_array_equal(a,truth[:,i*3:i*3+3])
            else:
                fields=[]
                for c in "uvw":
                    (v,),origin=e[role,c]
                    vals=values(v,lo+origin,hi+origin)
                    if c=="w":vals *= -10.
                    fields.append(vals)
                pred=np.concatenate(fields,axis=1)
            predictions.append(pred)
    return truth,predictions


def save_arrays(path, **arrays):
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".pending")
    with tmp.open("xb") as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    with np.load(tmp,allow_pickle=False) as z:
        need(set(arrays)==set(z.files),"Array inventory")
        for name,a in arrays.items():np.testing.assert_array_equal(a,z[name])
    os.replace(tmp,path)
    atomic(path.with_suffix(".json"),identity(path,True))


def accepted(path):
    rec=path.with_suffix(".json")
    if not rec.exists():return False
    meta=read(rec);check_file(meta)
    need(Path(meta["path"])==path and sha(path)==meta["sha256"], "Comparison cache hash mismatch")
    return True


def collect(entries,data,axes,blocks):
    common=axes["common_test_target_idx"]
    y0,p0=load_chunk(entries,data,common,0,1)
    origin=np.nan_to_num(y0[0],nan=0.,posinf=0.,neginf=0.)
    origins=[np.nan_to_num(p[0],nan=0.,posinf=0.,neginf=0.) for p in p0]
    for num,(start,stop) in enumerate(blocks):
        out=WORK/"blocks"/f"block_{num:04d}.npz"
        if accepted(out):continue
        stats=None
        sub_samples=[]
        for lo in range(int(start),int(stop),12):
            truth,pred=load_chunk(entries,data,common,lo,min(lo+12,int(stop)))
            positions=sample_locations(truth.shape[-2:]); ci,yi,xi=map(np.asarray,zip(*positions))
            sub_samples.append(np.stack([x[:,ci,yi,xi] for x in [truth,*pred]],axis=1))
            part,lower,upper=block_stats(truth,pred,origin,origins)
            if stats is None:
                stats=part; ymin=lower; ymax=upper
            else:
                stats+=part;np.minimum(ymin,lower,out=ymin);np.maximum(ymax,upper,out=ymax)
        # Each 2-D member is separately loadable from the uncompressed archive.
        arrays={f"c{i}":stats[:,i].reshape(len(stats),-1) for i in range(9)}
        save_arrays(out,**arrays,minimum=ymin,maximum=ymax,samples=np.concatenate(sub_samples,axis=0))
        print(f"committed sufficient statistics {num+1}/{len(blocks)}",flush=True)


def reduce_channels(blocks,weights):
    source=[WORK/"blocks"/f"block_{i:04d}.npz" for i in range(len(blocks))]
    need(all(accepted(path) for path in source),"Incomplete block cache")
    for c in range(9):
        out=WORK/"channels"/f"channel_{c}.npz"
        if accepted(out):continue
        # One channel takes about 1.6 GiB of sufficient statistics, avoiding a full dataset RAM load.
        stacked=[];lower=None;upper=None
        for p in source:
            with np.load(p,allow_pickle=False) as z:
                stacked.append(z[f"c{c}"])
                a,b=z["minimum"][c].reshape(-1),z["maximum"][c].reshape(-1)
            lower=a if lower is None else np.minimum(lower,a)
            upper=b if upper is None else np.maximum(upper,b)
        n=sum((a[0] for a in stacked)).astype(np.int64)
        span=np.full(n.shape,np.nan)
        np.subtract(upper,lower,out=span,where=n>=3)
        grid,mask,means,counts=summaries(stacked,span,weights,leads=6)
        save_arrays(out,grid=grid,mask=mask,means=means,counts=counts,span=span,n=n)
        print(f"committed grid and paired replicates {c+1}/9",flush=True)


def sample_locations(shape):
    rows,cols=shape
    return [(ch,j,k) for ch in range(9)
            for j,k in [(0,0),(rows//2,cols//2),(rows-1,cols-1)]]


def direct_score(y,p,common,span):
    """Direct two-pass unweighted temporal formulas independent of the block statistics."""
    y=y[common];p=p[common]
    out=np.full(4,np.nan)
    if len(y)<3:return out
    err=p-y
    out[1]=np.sqrt(np.mean(err*err));out[2]=np.mean(np.abs(err))
    if span>0:out[3]=out[1]/span
    if np.ptp(y)>0 and np.ptp(p)>0:
        out[0]=np.corrcoef(y,p)[0,1]
    return out


def independent_checks(blocks,weights,axes):
    channels=[WORK/"channels"/f"channel_{c}.npz" for c in range(9)]
    samples=sample_locations((len(axes["latitude"]),len(axes["longitude"])))
    for p in channels:need(accepted(p),"Missing source grid summary")
    records=[];sample_statistics=[];all_values=[]
    for bi in range(len(blocks)):
        path=WORK/"blocks"/f"block_{bi:04d}.npz"
        with np.load(path,allow_pickle=False) as z:
            all_values.append(z["samples"])
            cols=[]
            for c in range(9):
                a=z[f"c{c}"];width=len(axes["longitude"])
                for ch,y,x in samples:
                    if ch==c:cols.append(a[:,y*width+x])
            sample_statistics.append(np.stack(cols,axis=-1))
    values=np.concatenate(all_values,axis=0)
    stats=np.asarray(sample_statistics)
    selected=[0,1,len(weights)//2,len(weights)-1]
    locs=[np.arange(a,b) for a,b in blocks]
    max_gain_error=0.;max_metric_error=0.
    for sid,(c,lat,lon) in enumerate(samples):
        ix=lat*len(axes["longitude"])+lon
        truth=values[:,0,sid];pred=values[:,1:,sid]
        valid=np.isfinite(truth)&np.isfinite(pred).all(1)
        span=float(np.ptp(truth[valid])) if valid.sum()>=3 else np.nan
        with np.load(channels[c],allow_pickle=False) as z:
            grid=z["grid"][...,ix].reshape(18,4)
            np.testing.assert_equal(z["span"][ix],span)
            need(int(z["n"][ix])==np.count_nonzero(valid),"Independent sample temporal support")
        tests=[(-1,np.arange(len(truth)),grid)]
        for rep in selected:
            rows=np.concatenate([np.tile(lo,int(n)) for lo,n in zip(locs,weights[rep])])
            sampled_stat=weights[rep].astype("f8")@stats[:,:,sid]
            rebuilt=scores(sampled_stat[None,:,None],np.asarray([span]))[0,:,:,0]
            tests.append((rep,rows,rebuilt))
        for rep,rows,computed in tests:
            direct=np.stack([direct_score(truth[rows],pred[rows,m],valid[rows],span) for m in range(18)])
            np.testing.assert_allclose(computed,direct,rtol=1e-8,atol=2e-11,equal_nan=True)
            delta=np.abs(computed-direct)
            if np.isfinite(delta).any():max_metric_error=max(max_metric_error,float(np.nanmax(delta)))
            target=direct.reshape(6,3,4)
            actual=gains(computed.reshape(6,3,4),1,2)
            for pair,(a,b) in enumerate(PAIRS):
                reference=np.empty((6,4))
                reference[:,0]=target[:,a,0]-target[:,b,0]
                reference[:,1:]=target[:,b,1:]-target[:,a,1:]
                np.testing.assert_allclose(actual[:,pair],reference,rtol=1e-8,atol=5e-11,equal_nan=True)
                err=np.abs(actual[:,pair]-reference)
                if np.isfinite(err).any():max_gain_error=max(max_gain_error,float(np.nanmax(err)))
        records.append(dict(channel=c,ilat=lat,ilon=lon,n_valid=int(valid.sum())))
    receipt=dict(status="PASS_INDEPENDENT_FULL_SERIES_AND_EXPANDED_BOOTSTRAP_POINT_CHECKS",
                 locations=records,lead_model_combinations=18,paired_gains_checked=3,
                 source_rows=len(values),bootstrap_replica_indices=selected,
                 maximum_metric_absolute_difference=max_metric_error,maximum_gain_absolute_difference=max_gain_error)
    atomic(WORK/"INDEPENDENT_SAMPLE_CHECKS.json",receipt)
    return receipt


def quantiles(replicates):
    """Each cell uses its finite replicates; the exported n_boot_valid exposes any exclusions."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="All-NaN slice encountered")
        return np.nanquantile(replicates,[.025,.975],axis=0)


def grouped(a):
    """Last axes component,level. Add optional equal-level mean; caller also averages leads."""
    return np.concatenate((a,np.mean(a,axis=-1,keepdims=True)),axis=-1)


def csv_records(means):
    # [rep including point estimate, lead, model, metric, component, level].
    values=grouped(means)
    values=np.concatenate((values,values.mean(axis=1,keepdims=True)),axis=1)
    difference=gains(values,model_axis=2,metric_axis=3)
    ci=quantiles(difference[1:])
    count=np.isfinite(difference[1:]).sum(axis=0)
    rows=[]
    for h,label in enumerate([1,2,3,4,5,6,"equal_six_lead_mean"]):
        for k,metric in enumerate(METRICS):
            for c,component in enumerate(COMPONENTS):
                unit=("Pa s-1" if component=="omega" else "m s-1") if metric in ("RMSE","MAE") else "1"
                for j,level in enumerate([*LEVELS,"equal_three_level_mean"]):
                    base=dict(lead=label,component=component,level=level,metric=metric,unit=unit)
                    for m,model in enumerate(MODELS):
                        rows.append(dict(**base,kind="model",candidate=model,control="",value=values[0,h,m,k,c,j],
                                         ci_lower="",ci_upper="",n_boot_valid=""))
                    for pair,(candidate,control) in enumerate(PAIRS):
                        rows.append(dict(**base,kind="paired_gain",candidate=MODELS[candidate],control=MODELS[control],
                                         value=difference[0,h,pair,k,c,j],ci_lower=ci[0,h,pair,k,c,j],
                                         ci_upper=ci[1,h,pair,k,c,j],n_boot_valid=int(count[h,pair,k,c,j])))
    return rows


def write_results(axes,blocks,weights,contract_sha):
    channels=[WORK/"channels"/f"channel_{c}.npz" for c in range(9)]
    need(all(accepted(p) for p in channels),"Incomplete grid summaries")
    lat,lon=axes["latitude"],axes["longitude"]
    times=axes["raw_time_ns"][axes["common_test_target_idx"]]
    means=np.empty((NREP+1,6,3,4,3,3))
    nc_path=WORK/"Direct11_comparison_metrics.nc"
    csv_path=WORK/"Direct11_comparison_summary.csv"
    build=WORK/(nc_path.name+"."+uuid.uuid4().hex+".pending")
    with nc.Dataset(build,"w",format="NETCDF4") as d:
        dimensions=dict(lead=6,model=3,metric=4,component=3,level=3,latitude=len(lat),longitude=len(lon),
                        pair=3,quantile=2,replicate=NREP,block=len(blocks),target_time=len(times))
        for key,n in dimensions.items():d.createDimension(key,n)
        def var(name,typecode,dims,values=None):
            obj=d.createVariable(name,typecode,tuple(dims.split()),zlib=True,complevel=2)
            if values is not None:obj[:]=values
            return obj
        for key,labels in (("model",MODELS),("metric",METRICS),("component",COMPONENTS),("pair",[f"{MODELS[c]}_vs_{MODELS[b]}" for c,b in PAIRS])):
            v=d.createVariable(key,str,(key,))
            for i,value in enumerate(labels):v[i]=value
        var("lead","i4","lead",np.arange(1,7)).units="hours"
        var("level","f4","level",LEVELS).units="hPa"
        var("latitude","f8","latitude",lat).units="degrees_north"
        var("longitude","f8","longitude",lon).units="degrees_east"
        var("quantile","f8","quantile",[.025,.975])
        var("target_time","i8","target_time",times//10**9).units="seconds since 1970-01-01 00:00:00"
        var("block_start_row","i8","block",blocks[:,0])
        var("block_stop_row_exclusive","i8","block",blocks[:,1])
        var("bootstrap_block_multiplicities","i4","replicate block",weights)
        grid=var("grid_metric","f8","lead model metric component level latitude longitude")
        diff=var("grid_paired_gain","f8","lead pair metric component level latitude longitude")
        mask=var("shared_grid_mask","u1","metric component level latitude longitude")
        span=var("truth_temporal_range","f8","component level latitude longitude")
        n=var("valid_time_count","i4","component level latitude longitude")
        spatialcount=var("spatial_grid_count","i4","metric component level")
        repcount=var("bootstrap_spatial_grid_count","i4","replicate metric component level")
        for ch,p in enumerate(channels):
            c,j=divmod(ch,3)
            with np.load(p,allow_pickle=False) as z:
                g=np.where(z["mask"][None,None],z["grid"],np.nan)
                grid[:,:,:,c,j]=g.reshape(6,3,4,len(lat),len(lon))
                diff[:,:,:,c,j]=gains(g,1,2).reshape(6,3,4,len(lat),len(lon))
                mask[:,c,j]=z["mask"].reshape(4,len(lat),len(lon))
                span[c,j]=z["span"].reshape(len(lat),len(lon))
                n[c,j]=z["n"].reshape(len(lat),len(lon))
                means[:,:,:,:,c,j]=z["means"]
                spatialcount[:,c,j]=z["counts"][0]
                repcount[:,:,c,j]=z["counts"][1:]
        alldiff=gains(means,2,3)
        var("spatial_mean","f8","lead model metric component level",means[0])
        var("paired_gain","f8","lead pair metric component level",alldiff[0])
        var("bootstrap_paired_gain","f8","replicate lead pair metric component level",alldiff[1:])
        var("paired_gain_ci","f8","quantile lead pair metric component level",quantiles(alldiff[1:]))
        var("paired_gain_n_boot_valid","i4","lead pair metric component level",np.isfinite(alldiff[1:]).sum(0))
        componentgain=alldiff.mean(axis=-1)
        var("equal_level_paired_gain","f8","lead pair metric component",componentgain[0])
        var("equal_level_gain_ci","f8","quantile lead pair metric component",quantiles(componentgain[1:]))
        all_lead=componentgain.mean(axis=1)
        var("equal_lead_level_gain","f8","pair metric component",all_lead[0])
        var("equal_lead_level_gain_ci","f8","quantile pair metric component",quantiles(all_lead[1:]))
        d.bootstrap_seed=SEED_BASE+BLOCK
        d.bootstrap_definition=f"{NREP} paired multinomial draws of nonoverlapping {BLOCK}-hour target blocks including short terminal blocks, fixed full-common truth ranges, same draws for every lead/model/cell."
        d.support_definition="Intersection of all six original native target axes; joint 18-prediction finite-mask; common spatial valid mask across leads and models per metric."
        d.metric_definition="Time-wise per-cell Pearson R, root mean squared error, mean absolute error, and RMSE/full common truth range (dimensionless). Minimum three valid times."
        d.units_key="u/v errors: m s-1; omega errors: Pa s-1; R and RMSPE: 1. Equal weight per valid grid cell and level, no mixing physical units."
        d.truth_source="Independent decoded original ERA5 pressure-level u, v, omega (float64). Direct11 delivered native truth verified exactly on all compared rows."
        d.gain_direction="Positive favors candidate: candidate minus control for R; control minus candidate for errors."
        d.uncertainty_scope="Percentile 95% interval across test time blocks conditional on these fixed models; new training seeds are not resampled."
        d.comparison_contract_sha256=contract_sha
    with nc.Dataset(build) as d:
        np.testing.assert_array_equal(np.asarray(d["target_time"][:])*10**9,times)
        np.testing.assert_array_equal(d["bootstrap_block_multiplicities"][:],weights)
        # Read every serialized metric field; independently compare all stored grid means and gains.
        numpy_fields={name:np.ma.asarray(v[:]).filled(np.nan) if np.issubdtype(v.dtype,np.floating) else np.asarray(v[:])
                      for name,v in d.variables.items() if name not in ("model","metric","component","pair")}
        gg=numpy_fields["grid_metric"]
        cnt=numpy_fields["spatial_grid_count"]
        recovered=np.full_like(means[0],np.nan)
        np.divide(np.nansum(gg,axis=(-1,-2)),cnt[None,None],out=recovered,where=cnt[None,None]>0)
        np.testing.assert_allclose(recovered,means[0],rtol=1e-11,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(numpy_fields["grid_paired_gain"],gains(gg,1,2),rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(numpy_fields["paired_gain_ci"],quantiles(numpy_fields["bootstrap_paired_gain"]),rtol=0,atol=0,equal_nan=True)
    os.replace(build,nc_path)
    records=csv_records(means)
    tmp=csv_path.with_name(csv_path.name+"."+uuid.uuid4().hex+".pending")
    with tmp.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records);f.flush();os.fsync(f.fileno())
    with tmp.open(encoding="utf-8",newline="") as f:
        stored=list(csv.DictReader(f))
    need(len(stored)==len(records),"CSV row count")
    for row,orig in zip(stored,records):
        need(all(row[k]==str(v) for k,v in orig.items()),"CSV readback")
    os.replace(tmp,csv_path)
    return nc_path,csv_path


def publish(source):
    import shutil
    dest=FINAL/source.name
    digest=sha(source)
    if not dest.exists():
        pending=dest.with_name(dest.name+".comparison.pending")
        if pending.exists():
            # Only resume a completed prior copy. A truncated copy is replaced via opening this
            # exact task-owned temporary path after binding its final target and expected digest.
            ready=sha(pending)==digest
        else:
            ready=False
        if not ready:
            with source.open("rb") as src,pending.open("wb") as dst:
                shutil.copyfileobj(src,dst,8*1024**2);dst.flush();os.fsync(dst.fileno())
        need(sha(pending)==digest,"Pending copy mismatch")
        os.rename(pending,dest)  # Windows no-replace publication
    need(sha(dest)==digest,"Existing final output hash mismatch")
    if dest.suffix==".nc":
        with nc.Dataset(dest) as d:
            need(len(d.dimensions["lead"])==6 and len(d.dimensions["replicate"])==NREP,"Final NC structure")
            for v in d.variables.values():v[:]
    else:
        with dest.open(encoding="utf-8",newline="") as f:
            need(len(list(csv.DictReader(f)))==7*4*3*4*6,"Final CSV structure")
    return identity(dest,True)


def run():
    with lock(WORK/"compare.lock"):
        axes,paths,contract,data=source_contract()
        p=WORK/"contract.json"
        if p.exists():need(read(p)==contract,"Changed inputs cannot reuse comparison cache")
        else:atomic(p,contract)
        completion=WORK/"COMPARISON_COMPLETION.json"
        if completion.exists():
            receipt=read(completion)
            need(receipt["status"]=="PASS_FINAL_COMPARISON_READBACK" and receipt["contract"]["sha256"]==sha(p),"Final comparison receipt mismatch")
            for rec in receipt["outputs"]:
                need(sha(check_file(rec))==rec["sha256"],"Changed final comparison output")
            return
        for sub in ("blocks","channels"):(WORK/sub).mkdir(exist_ok=True)
        timestamps=axes["raw_time_ns"][axes["common_test_target_idx"]].astype("datetime64[ns]")
        blocks,weights=block_design(timestamps)
        with ExitStack() as st:
            entries=sources(st,axes,paths)
            collect(entries,data,axes,blocks)
        reduce_channels(blocks,weights)
        independent_checks(blocks,weights,axes)
        # Recheck source identities and model hashes before exposing outputs.
        need(source_contract()[2]==contract,"Comparison inputs changed during evaluation")
        staged=write_results(axes,blocks,weights,sha(p))
        published=[publish(q) for q in staged]
        atomic(WORK/"COMPARISON_COMPLETION.json",dict(status="PASS_FINAL_COMPARISON_READBACK",utc=stamp(),
               frozen_science_sha=contract["science_sha256"],
               contract=identity(p,True),sample_checks=identity(WORK/"INDEPENDENT_SAMPLE_CHECKS.json",True),common_target_count=len(timestamps),block_count=len(blocks),outputs=published))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["run"])
    parser.parse_args()
    run()

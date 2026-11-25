"""Paired PINN continuity-error contributions on the support used in the mechanism figures."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path

import numpy as np
import xarray as xr

RADIUS = 6_371_000.0
SEED = 20261025
SEEDS = (42, 2026)
EXPERIMENTS = ("no_pinn", "no_shear")
LEVELS = np.array([850., 925., 1000.])

def normalized_times(a):
    t=np.asarray(a).astype('datetime64[ns]');assert not np.isnat(t).any()
    n=t.astype('int64');unit=3600*10**9; nearest=np.rint(n.astype(np.float64)/unit).astype('int64')*unit
    if np.max(np.abs(n-nearest),initial=0)>1000: raise ValueError('Time is not an hour within strict 1000 ns codec tolerance')
    assert np.all(np.diff(nearest)>0),'Duplicate or unsorted time'
    return nearest.astype('datetime64[ns]')

class Field:
    def __init__(self,path,var=None):
        self.path=Path(path);self.ds=xr.open_dataset(path)
        if var is None:var=next(k for k,v in self.ds.data_vars.items() if {'time','latitude','longitude'}.issubset(v.dims))
        self.da=self.ds[var];self.times=normalized_times(self.ds.time.values)
        self.lat=self.ds.latitude.values;self.lon=self.ds.longitude.values
        self.level=self.ds.level.values if 'level' in self.ds.coords else None
    def read(self,times):
        i=np.searchsorted(self.times,times);assert (i<len(self.times)).all() and np.array_equal(self.times[i],times)
        a=self.da.isel(time=i)
        order=('time','level','latitude','longitude') if self.level is not None else ('time','latitude','longitude')
        return np.asarray(a.transpose(*order).values,dtype=np.float64)
    def close(self):self.ds.close()

def secant(a,x,axis):
    a=np.moveaxis(np.asarray(a,dtype=np.float64),axis,-1);x=np.asarray(x,dtype=np.float64)
    assert len(x)>1 and (np.all(np.diff(x)>0) or np.all(np.diff(x)<0))
    y=np.empty_like(a);y[...,0]=(a[...,1]-a[...,0])/(x[1]-x[0]);y[...,-1]=(a[...,-1]-a[...,-2])/(x[-1]-x[-2])
    y[...,1:-1]=(a[...,2:]-a[...,:-2])/(x[2:]-x[:-2]);return np.moveaxis(y,-1,axis)

def physics(a,lat,lon,p):
    u,v,w=a
    h=secant(u,np.deg2rad(lon),-1)/(RADIUS*np.cos(np.deg2rad(lat))[None,None,:,None])+secant(v,RADIUS*np.deg2rad(lat),-2)
    z=secant(w,p,1);return h,z,h+z

def stencil_valid(mask):
    """Require center and every horizontal/vertical neighbor used by training stencils."""
    out=mask.copy()
    for ax in (1,2,3):
        n=mask.shape[ax];l=np.maximum(np.arange(n)-1,0);r=np.minimum(np.arange(n)+1,n-1)
        out &= np.take(mask,l,axis=ax)&np.take(mask,r,axis=ax)
    return out

def partition_blocks(times,hours):
    """Nonoverlapping complete hourly blocks; never bridges a real gap. Last short block retained."""
    cut=np.r_[0,np.flatnonzero(np.diff(times)!=np.timedelta64(1,'h'))+1,len(times)]
    return [(a,min(a+hours,e)) for s,e in zip(cut[:-1],cut[1:]) for a in range(s,e,hours)]

def resampling_weights(times,hours,nrep=1000):
    blocks=partition_blocks(times,hours); rng=np.random.default_rng(SEED+hours)
    weights=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=nrep).astype(np.float64)
    return blocks,weights

def read_field(f,t):
 idx=np.searchsorted(f.times,t);assert np.all(idx<len(f.times)) and np.array_equal(f.times[idx],t)
 ix=slice(int(idx[0]),int(idx[-1])+1) if np.array_equal(idx,np.arange(idx[0],idx[-1]+1)) else idx
 a=f.da.isel(time=ix)
 dims=('time','level','latitude','longitude') if 'level' in a.dims else ('time','latitude','longitude')
 return np.asarray(a.transpose(*dims).values,dtype=np.float64)

def hourly_terms(fields, pressure, target, lat, lon):
    """Pool the four quadratic terms on the identical stencil for all four forecasts."""
    p = LEVELS * 100
    terrain = read_field(pressure, target)[:, None] >= p[None, :, None, None]
    arrays = [np.stack([read_field(f, target) for f in run]) for run in fields]
    shared = np.ones(arrays[0].shape[1:], dtype=bool)
    excluded = np.zeros(len(target), dtype=bool)
    for a in arrays:
        shared &= np.isfinite(a).all(axis=0)
        for co in range(3):
            ref = arrays[0][co]
            difference = np.abs(a[co] - ref)
            material = np.isfinite(difference) & (difference > 1e-7 + 1e-6 * np.abs(ref))
            excluded |= material.reshape(len(target), -1).any(axis=1)
    shared &= ~excluded[:, None, None, None]
    mask = stencil_valid(terrain) & stencil_valid(shared)
    stats = np.empty((len(target), len(SEEDS), len(EXPERIMENTS), len(LEVELS), 5))
    for run,a in enumerate(arrays):
        a[2] *= -10
        a[5] *= -10
        H,Z,C = physics(a[:3], lat, lon, p)
        h,z,c = physics(a[3:], lat, lon, p)
        eh,ez,ec = h-H,z-Z,c-C
        row = stats[:, run//len(EXPERIMENTS), run%len(EXPERIMENTS)]
        for k,v in enumerate([np.ones_like(H), eh*eh, ez*ez, eh*ez, ec*ec]):
            row[..., k] = np.where(mask, v, 0).sum(axis=(-2,-1))
    return stats, excluded


def decomposition(stats):
    n = stats[..., :1]
    terms = np.divide(stats[..., 1:], n, out=np.full_like(stats[..., 1:], np.nan), where=n>0)
    terms[..., 2] *= 2
    return terms


def run(results, surface_pressure, output, chunk_hours):
    with ExitStack() as stack:
        def open_field(path, variable=None):
            field = Field(path, variable)
            stack.callback(field.close)
            return field
        fields = [
            [open_field(results / f"da_transunet_{kind}_{co}_1_24year_changjiang_pinn_followup_v1_seed{seed}_{exp}.nc")
             for kind in ("testy", "predicty") for co in ("u", "v", "w")]
            for seed in SEEDS for exp in EXPERIMENTS]
        pressure = open_field(surface_pressure, "sp")
        reference = fields[0][0]
        t = reference.times
        for run_fields in fields:
            for f in run_fields:
                if not (np.array_equal(f.lat, reference.lat) and np.array_equal(f.lon, reference.lon)
                        and np.array_equal(f.level, LEVELS)):
                    raise ValueError(f"Forecast grid does not match the paired grid: {f.path}")
                t = np.intersect1d(t,f.times)
        if not (np.array_equal(pressure.lat,reference.lat) and np.array_equal(pressure.lon,reference.lon)):
            raise ValueError("Surface pressure and forecast grids differ")
        t = np.intersect1d(t, pressure.times)
        if not len(t):
            raise ValueError("No common hourly targets")
        stats = np.empty((len(t),len(SEEDS),len(EXPERIMENTS),len(LEVELS),5))
        excluded = np.zeros(len(t), dtype=bool)
        for lo in range(0,len(t),chunk_hours):
            sl = slice(lo, min(lo+chunk_hours,len(t)))
            stats[sl], excluded[sl] = hourly_terms(fields, pressure, t[sl], reference.lat, reference.lon)

    t,stats = t[~excluded],stats[~excluded]
    if not len(t):
        raise ValueError("No common target remains after the paired truth comparison")
    blocks,weights = resampling_weights(t,72,1000)
    bs = np.stack([stats[a:b].sum(axis=0) for a,b in blocks])
    estimate = decomposition(bs.sum(axis=0))
    draw = decomposition((weights @ bs.reshape(len(blocks),-1)).reshape((len(weights),)+bs.shape[1:]))
    delta = estimate[:,1]-estimate[:,0]
    ci = np.nanquantile(draw[:,:,1]-draw[:,:,0],[.025,.975],axis=0)
    ds = xr.Dataset(coords={"seed":list(SEEDS), "level":LEVELS, "bound":["lower","upper"],
                            "error_term":["E_eH2","E_eZ2","2E_eHeZ","E_eC2"],
                            "forecast_time":1, "stage":"stage1_main_forecast","condition":"all",
                            "region":"all","support":"above_ground_full_stencil","operator":"training"})
    dims = ("seed","level","error_term")
    ds["delta_continuity_error_contribution"] = (dims,delta,{"units":"s-2"})
    ds["delta_continuity_error_contribution_CI"] = (("bound",)+dims,ci,{"units":"s-2"})
    ds.attrs.update(bootstrap="1000 paired shared 72h nonoverlapping whole-time-block multinomial draws; noncontiguous hours split blocks; terminal short blocks retained",
                    delta="PINN(no_shear) minus NoPINN", error_identity="E[eC²] = E[eH²] + E[eZ²] + 2 E[eH eZ]",
                    omega_scaling="Stored w = -0.1 omega; multiply by -10 before computing derivatives")
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp = output.with_suffix(".partial.nc")
    ds.to_netcdf(tmp,engine="netcdf4")
    tmp.replace(output)


if __name__ == "__main__":
    results = Path(os.environ.get("YANGTZE_RESULTS_DIR", "results"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=results,help="Directory containing the paired first-stage wind files.")
    parser.add_argument("--surface-pressure",type=Path,default=Path(os.environ.get("YANGTZE_DATA_DIR","data"))/"Surface-pressure.nc")
    parser.add_argument("--chunk-hours",type=int,default=64)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.chunk_hours<1:
        parser.error("--chunk-hours must be positive")
    output = args.output or args.results/"task2_diagnostics"/"mechanism_v4_20261028"/"pinn"/"pinn_paired_seed_mechanism_final.nc"
    run(args.results, args.surface_pressure, output, args.chunk_hours)

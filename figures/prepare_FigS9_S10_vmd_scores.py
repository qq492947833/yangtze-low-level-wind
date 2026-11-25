from contextlib import ExitStack
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from figure_paths import RESULTS

GROUPS = ['A', 'B', 'C', 'D']
SUFFIX = {'A': '', 'B': '_residual', 'C': '_vmd', 'D': '_vmd_residual'}
LEVELS = [850, 925, 1000]
HOUR = 3_600_000_000_000
BLOCK = 168
def decode_time(var):
    raw=np.asarray(var[:]);units=str(var.units);cal=str(getattr(var,'calendar','standard'))
    assert cal in ('standard','gregorian','proleptic_gregorian')
    assert raw.ndim==1 and np.isfinite(raw).all(),'NaT/nonfinite time'
    unit,origin=units.split(' since ',1)
    scale={'days':86400000000000,'day':86400000000000,'hours':HOUR,'hour':HOUR,'minutes':60000000000,'minute':60000000000,'seconds':1000000000,'second':1000000000}[unit.lower()]
    orig=pd.Timestamp(origin).value
    with localcontext() as ctx:
        ctx.prec=45
        ns=np.fromiter((orig+int((Decimal.from_float(float(x))*Decimal(scale)).to_integral_value()) for x in raw),dtype=np.int64,count=len(raw))
    nearest=((ns+HOUR//2)//HOUR)*HOUR;off=ns-nearest
    correction=(abs(off)<=1000)&(off!=0)
    used=np.where(correction,nearest,ns)
    assert np.unique(used).size==len(used),'Duplicate times after documented encoding normalization'
    assert np.all(np.diff(used)>0),'Non-increasing time order; requires explicit investigation'
    return ns,used,correction,off,{'units':units,'calendar':cal,'max_offset_ns':int(abs(off).max(initial=0)), 'correction_count':int(correction.sum()),'off_hour_beyond_tolerance':int((abs(off)>1000).sum()),'gap_count':int((np.diff(used)!=HOUR).sum())}
class Stats:
    def __init__(self,shape):
        self.a={k:np.zeros(shape,dtype='f8') for k in ('n','my','mp','vy','vp','cov','sse')}
        self.a.update(lo=np.full(shape,np.inf),hi=np.full(shape,-np.inf),plo=np.full(shape,np.inf),phi=np.full(shape,-np.inf))
    def update(self,index,y,p,valid):
        a=self.a;n=valid.sum(axis=0).astype('f8');safe=np.maximum(n,1)
        by=np.sum(np.where(valid,y,0),axis=0,dtype='f8')/safe;bp=np.sum(np.where(valid,p,0),axis=0,dtype='f8')/safe
        dy=np.where(valid,y-by,0);dp=np.where(valid,p-bp,0)
        old=a['n'][index];total=old+n;ratio=np.divide(n,total,out=np.zeros_like(n),where=total>0);cross=old*ratio
        oy=by-a['my'][index];op=bp-a['mp'][index]
        a['vy'][index]+=np.sum(dy*dy,axis=0,dtype='f8')+oy*oy*cross
        a['vp'][index]+=np.sum(dp*dp,axis=0,dtype='f8')+op*op*cross
        a['cov'][index]+=np.sum(dy*dp,axis=0,dtype='f8')+oy*op*cross
        a['my'][index]+=oy*ratio;a['mp'][index]+=op*ratio;a['n'][index]=total
        e=np.where(valid,p-y,0);a['sse'][index]+=np.sum(e*e,axis=0,dtype='f8')
        a['lo'][index]=np.minimum(a['lo'][index],np.min(np.where(valid,y,np.inf),axis=0))
        a['hi'][index]=np.maximum(a['hi'][index],np.max(np.where(valid,y,-np.inf),axis=0))
        a['plo'][index]=np.minimum(a['plo'][index],np.min(np.where(valid,p,np.inf),axis=0))
        a['phi'][index]=np.maximum(a['phi'][index],np.max(np.where(valid,p,-np.inf),axis=0))
    def metrics(self):
        a=self.a;n=a['n'];rr=a['hi']-a['lo']
        r=np.full(n.shape,np.nan);rm=np.full(n.shape,np.nan);pe=np.full(n.shape,np.nan)
        truth_vary=(a['vy']>0)&(rr>0);pred_vary=(a['vp']>0)&(a['phi']>a['plo'])
        rok=(n>=3)&truth_vary&pred_vary
        np.divide(a['cov'],np.sqrt(a['vy']*a['vp']),out=r,where=rok);r=np.clip(r,-1,1)
        np.sqrt(np.divide(a['sse'],n,out=np.full(n.shape,np.nan),where=n>=3),out=rm)
        np.divide(rm,rr,out=pe,where=(n>=3)&(rr>0)&np.isfinite(rr))
        reason=(n<3).astype('u1')+2*(~truth_vary).astype('u1')+4*(~pred_vary).astype('u1')+8*(~np.isfinite(rr)|(rr<=0)).astype('u1')
        return {'R':r,'RMSE':rm,'RMSPE':pe,'n_valid':n.astype('i4'),'testy_min':np.where(n>0,a['lo'],np.nan),'testy_max':np.where(n>0,a['hi'],np.nan),'testy_range':np.where(n>0,rr,np.nan),'invalid_flags':reason}


def align_files():
    rows = {}
    common = None
    coords = None
    for component in 'uvw':
        for group in GROUPS:
            for lead in range(1, 7):
                for kind in ('testy', 'predicty'):
                    variable = f'{kind}_{component}'
                    path = RESULTS / f'da_transunet_{variable}_{lead}_24year_changjiang{SUFFIX[group]}.nc'
                    with Dataset(path, 'r') as ds:
                        if ds[variable].dimensions != ('time', 'level', 'latitude', 'longitude'):
                            raise ValueError(f'Unexpected dimension order in {path}')
                        _, times, _, _, _ = decode_time(ds['time'])
                        lat = np.asarray(ds['latitude'][:])
                        lon = np.asarray(ds['longitude'][:])
                        lev = np.array(ds['level'][:], dtype=float)
                        unit = str(getattr(ds['level'], 'units', '')).strip().lower()
                        if unit in ('pa', 'pascal', 'pascals'):
                            lev /= 100
                        elif unit not in ('', 'hpa', 'hectopascal', 'hectopascals', 'millibar', 'mbar'):
                            raise ValueError(f'Unrecognized pressure coordinate units in {path}')
                        if sorted(lev.tolist()) != LEVELS:
                            raise ValueError(f'Expected the three study pressure levels in {path}')
                        if coords is None:
                            coords = lat, lon
                        elif not (np.array_equal(lat, coords[0]) and np.array_equal(lon, coords[1])):
                            raise ValueError(f'Mismatched spatial coordinates in {path}')
                        order = [int(np.flatnonzero(lev == level)[0]) for level in LEVELS]
                        rows[component, group, lead, kind] = (path, variable, times, order)
                        common = times if common is None else np.intersect1d(common, times, assume_unique=True)
    expected = pd.date_range('2019-03-15 02:00', '2023-12-31 23:00', freq='h').to_numpy(dtype='datetime64[ns]').view('i8')
    if not np.array_equal(common, expected):
        raise ValueError('The shared timestamps must match the 42,070-hour evaluation window.')
    return common, coords, rows


def component_scores(component, common, coords, rows):
    configs = [(group, lead) for group in GROUPS for lead in range(1, 7)]
    lat, lon = coords
    state = Stats((len(configs), len(LEVELS), len(lat), len(lon)))
    with ExitStack() as stack:
        handles = []
        for group, lead in configs:
            for kind in ('testy', 'predicty'):
                path, variable, times, order = rows[component, group, lead, kind]
                ds = stack.enter_context(Dataset(path, 'r'))
                ds.set_auto_mask(False)
                var = ds[variable]
                var.set_var_chunk_cache(size=4*1024*1024, nelems=1009, preemption=.5)
                fills = [float(getattr(var, attr)) for attr in ('_FillValue', 'missing_value') if hasattr(var, attr)]
                handles.append((var, np.searchsorted(times, common), order, fills))
        for start in range(0, len(common), BLOCK):
            arrays, common_mask = [], None
            for variable, index, order, fills in handles:
                selected = index[start:start + BLOCK]
                key = slice(int(selected[0]), int(selected[-1]) + 1) if np.all(np.diff(selected) == 1) else selected
                array = np.asarray(variable[key, :, :, :], dtype='f8')[:, order]
                for fill in fills:
                    array[array == fill] = np.nan
                if component == 'w':
                    array *= -10.
                finite = np.isfinite(array)
                common_mask = finite if common_mask is None else common_mask & finite
                arrays.append(array)
            for i in range(len(configs)):
                state.update(i, arrays[2*i], arrays[2*i + 1], common_mask)
    return state.metrics()


def stage_gains():
    common, coords, rows = align_files()
    gains = []
    for component in 'uvw':
        metrics = component_scores(component, common, coords, rows)
        means = {}
        for name in ('R', 'RMSPE', 'RMSE'):
            data = metrics[name]
            for li,level in enumerate(LEVELS):
                support = np.isfinite(data[:, li]).all(axis=0)
                for gi, group in enumerate(GROUPS):
                    for lead in range(1, 7):
                        field = data[gi*6 + lead-1, li]
                        means[group, lead, level, name] = float(np.mean(field[support])) if support.any() else np.nan
        for stage, no, yes in [('stage1', 'A', 'C'), ('final_residual', 'B', 'D')]:
            for lead in range(1, 7):
                for level in LEVELS:
                    without = {k: means[no, lead, level, k] for k in ('R', 'RMSE', 'RMSPE')}
                    with_vmd = {k: means[yes, lead, level, k] for k in ('R', 'RMSE', 'RMSPE')}
                    row = {'stage': stage, 'lead_hour': lead, 'component': 'omega' if component == 'w' else component,
                           'level_hpa': level, 'without_vmd': no, 'with_vmd': yes, 'evaluation_scope': 'common_time',
                           'common_target_count': len(common), 'delta_R': with_vmd['R'] - without['R'],
                           'delta_RMSPE': without['RMSPE'] - with_vmd['RMSPE']}
                    for k in without:
                        row[f'mean_{k}_without'], row[f'mean_{k}_with'] = without[k], with_vmd[k]
                    gains.append(row)
    pd.DataFrame(gains).to_csv(RESULTS / 'VMD_common_time_spatial_gains.csv', index=False)


if __name__ == '__main__':
    stage_gains()

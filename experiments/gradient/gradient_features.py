"""Causal six-frame spatial and temporal derivatives for surface fields."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np

EARTH_RADIUS_M = 6_371_000.0
HISTORY = 6
TIME_STEP_SECONDS = 3600.0
VARIABLES = ('sp', 't2m', 'd2m')
CHANNEL_ORDER = VARIABLES + tuple(f'{v}_{d}' for v in VARIABLES for d in ('dx','dy','dt'))
GRADIENT_ORDER = CHANNEL_ORDER[3:]
SCHEMA_VERSION = 1


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


@dataclass(frozen=True)
class GradientStats:
    """Population mean/std [9,H,W]; one set shared by all six history slots."""
    mean: np.ndarray
    std: np.ndarray
    zero_std_mask: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    count: int
    metadata: dict

    def __post_init__(self):
        shape = self.mean.shape
        if len(shape) != 3 or shape[0] != 9:
            raise ValueError('Statistics must have shape [9,H,W]')
        for name in ('std','zero_std_mask','minimum','maximum'):
            if getattr(self, name).shape != shape:
                raise ValueError(f'Statistics {name} has a mismatching shape')
        if self.count <= 0:
            raise ValueError('Statistics require a positive pooled observation count')
        if any(not np.isfinite(getattr(self, k)).all() for k in ('mean','std','minimum','maximum')):
            raise ValueError('Statistics must all be finite')
        if np.any(self.std <= 0) or np.any(self.maximum < self.minimum):
            raise ValueError('Invalid standard deviation or extrema')
        if np.any(self.std[self.zero_std_mask.astype(bool)] != 1):
            raise ValueError('Exactly zero standard deviations must be recorded and replaced with 1')

    def save(self, npz_path, json_path=None):
        """Atomically publish NPZ then its hash-bound JSON receipt; both are required."""
        path = Path(npz_path)
        if path.suffix.lower() != '.npz':
            raise ValueError('Statistics path must end in .npz')
        jp = Path(json_path) if json_path is not None else path.with_suffix('.json')
        if path.resolve() == jp.resolve():
            raise ValueError('NPZ and JSON paths must differ')
        path.parent.mkdir(parents=True, exist_ok=True); jp.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with tmp.open('wb') as f:
                np.savez_compressed(f, mean=self.mean, std=self.std,
                    zero_std_mask=self.zero_std_mask, minimum=self.minimum, maximum=self.maximum,
                    count=np.asarray(self.count,dtype='i8'), schema_version=np.asarray(SCHEMA_VERSION,dtype='i4'),
                    metadata_json=np.asarray(json.dumps(self.metadata,sort_keys=True,allow_nan=False)))
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists(): tmp.unlink()
        receipt = {'status':'COMPLETE','schema_version':SCHEMA_VERSION,'npz_filename':path.name,
                   'npz_sha256':_sha(path),'shape':list(self.mean.shape),'count':self.count,
                   'zero_std_count':int(self.zero_std_mask.sum()),'metadata':self.metadata,
                   'per_gradient_summary':[
                       {'channel':GRADIENT_ORDER[i], 'physical_minimum':float(self.minimum[i].min()),
                        'physical_maximum':float(self.maximum[i].max()),
                        'mean_minimum':float(self.mean[i].min()),'mean_maximum':float(self.mean[i].max()),
                        'effective_std_minimum':float(self.std[i].min()),'effective_std_maximum':float(self.std[i].max()),
                        'zero_std_grid_count':int(self.zero_std_mask[i].sum())} for i in range(9)]}
        _atomic_json(jp, receipt)
        return {'npz':str(path),'json':str(jp),'npz_sha256':receipt['npz_sha256']}

    @classmethod
    def load(cls, npz_path, json_path=None):
        path = Path(npz_path)
        jp = Path(json_path) if json_path is not None else path.with_suffix('.json')
        receipt = json.loads(jp.read_text(encoding='utf-8'))
        if receipt.get('status') != 'COMPLETE' or receipt.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Incomplete or unsupported gradient statistics receipt')
        if receipt['npz_sha256'] != _sha(path):
            raise ValueError('Gradient statistics NPZ hash does not match receipt')
        with np.load(path, allow_pickle=False) as z:
            if int(z['schema_version']) != SCHEMA_VERSION:
                raise ValueError('Unsupported NPZ schema')
            metadata = json.loads(str(z['metadata_json'].item()))
            if metadata != receipt['metadata']:
                raise ValueError('NPZ/JSON statistics metadata mismatch')
            stats = cls(*(np.array(z[k],copy=True) for k in ('mean','std','zero_std_mask','minimum','maximum')),
                        count=int(z['count']),metadata=metadata)
        if stats.count != receipt['count'] or list(stats.mean.shape) != receipt['shape']:
            raise ValueError('NPZ/JSON statistics dimensions or count mismatch')
        return stats


class GradientFeatureBuilder:
    """Lazy CPU provider; get_windows(start_indices) returns [B,6,12,H,W].

    physical/standardized may be numpy arrays or read-only memmaps. Optional
    spatial_gradients is a compact [T,3,2,H,W] array/memmap, with physical dx,dy
    in that order. It must be made from these same physical inputs/coordinates;
    cache provenance and publication are caller responsibilities. No dt cache
    is accepted: dt is always computed inside each six-frame window.
    """
    def __init__(self, physical, standardized, latitude_deg, longitude_deg,
                 stats=None, spatial_gradients=None, output_dtype=np.float32,
                 physical_units=('Pa','K','K')):
        self.physical, self.standardized = physical, standardized
        self.shape = tuple(physical.shape)
        if len(self.shape) != 4 or self.shape[1] != 3 or self.shape[0] < HISTORY:
            raise ValueError('physical must have shape [T>=6,3,H,W]')
        if tuple(standardized.shape) != self.shape:
            raise ValueError('standardized must have exactly the physical array shape')
        if any(not np.issubdtype(np.dtype(x.dtype),np.number) for x in (physical,standardized)):
            raise TypeError('Input arrays must have numeric dtypes')
        self.latitude_deg = self._coordinate(latitude_deg,self.shape[2],'latitude')
        self.longitude_deg = self._coordinate(longitude_deg,self.shape[3],'longitude')
        if np.any(np.abs(self.latitude_deg) >= 90):
            raise ValueError('Latitude must lie strictly between -90 and 90 degrees for finite dx')
        self.lat_rad, self.lon_rad = np.deg2rad(self.latitude_deg), np.deg2rad(self.longitude_deg)
        self._dx_distance_factor = EARTH_RADIUS_M*np.cos(self.lat_rad)[None,None,:,None]
        coord_hash=hashlib.sha256()
        for coord in (self.latitude_deg,self.longitude_deg):
            coord_hash.update(np.asarray(coord.shape,dtype='<i8').tobytes())
            coord_hash.update(np.asarray(coord,dtype='<f8').tobytes())
        self.coordinate_sha256=coord_hash.hexdigest()
        self.output_dtype=np.dtype(output_dtype)
        if self.output_dtype not in (np.dtype('f4'),np.dtype('f8')):
            raise ValueError('output_dtype must be float32 or float64')
        self.physical_units=tuple(physical_units)
        if len(self.physical_units)!=3 or not all(isinstance(x,str) and x for x in self.physical_units):
            raise ValueError('Supply three explicit physical units in sp/t2m/d2m order')
        self.spatial_gradients=spatial_gradients
        if spatial_gradients is not None and tuple(spatial_gradients.shape)!=(self.shape[0],3,2,*self.shape[2:]):
            raise ValueError('spatial_gradients must be compact [T,3,2,H,W] dx/dy')
        self.stats=None
        if stats is not None:self.set_stats(stats)

    @staticmethod
    def _coordinate(values,size,name):
        x=np.asarray(values,dtype='f8')
        if x.ndim!=1 or len(x)!=size or len(x)<2 or not np.isfinite(x).all():
            raise ValueError(f'{name} must be a finite 1D degree coordinate matching a dimension of length >=2')
        d=np.diff(x)
        if not (np.all(d>0) or np.all(d<0)):
            raise ValueError(f'{name} must be strictly monotonic; descending coordinates are supported')
        return x.copy()

    def _indices(self,indices,windows=True):
        a=np.asarray(indices)
        if a.ndim!=1 or a.size==0 or not np.issubdtype(a.dtype,np.integer):
            raise ValueError('Indices must be a nonempty one-dimensional integer sequence')
        a=a.astype('i8',copy=False)
        highest=self.shape[0]-HISTORY if windows else self.shape[0]-1
        if np.any(a<0) or np.any(a>highest):
            raise IndexError(f'Indices outside inclusive range [0,{highest}]')
        return a

    @staticmethod
    def _finite(array,label):
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {label}; no implicit filling or masking is permitted')
        return array

    def _spatial_from_frames(self,frames):
        # np.gradient uses signed actual radian coordinates; no wrapping/cropping.
        dy,dx=np.gradient(frames,self.lat_rad,self.lon_rad,axis=(2,3),edge_order=1)
        dx/=self._dx_distance_factor
        dy/=EARTH_RADIUS_M
        return np.stack((dx,dy),axis=2)

    def compute_spatial_gradients(self,frame_indices):
        """Return physical [N,3,2,H,W] float64 dx/dy for caller-chunked caching."""
        idx=self._indices(frame_indices,windows=False)
        frames=self._finite(np.asarray(self.physical[idx],dtype='f8'),'physical frames')
        return self._finite(self._spatial_from_frames(frames),'spatial gradients')

    def get_raw_gradients(self,indices):
        """Unnormalized physical [B,6,9,H,W] float64, bounded to requested frames."""
        starts=self._indices(indices)
        time_indices=starts[:,None]+np.arange(HISTORY,dtype='i8')[None,:]
        unique,inverse=np.unique(time_indices,return_inverse=True)
        frames=self._finite(np.asarray(self.physical[unique],dtype='f8'),'physical frames')
        windows=frames[inverse].reshape(len(starts),HISTORY,3,*self.shape[2:])
        dt=np.gradient(windows,TIME_STEP_SECONDS,axis=1,edge_order=1)
        if self.spatial_gradients is None:
            compact=self._spatial_from_frames(frames)
        else:
            compact=np.asarray(self.spatial_gradients[unique],dtype='f8')
        self._finite(compact,'spatial gradients')
        spatial=compact[inverse].reshape(len(starts),HISTORY,3,2,*self.shape[2:])
        result=np.empty((len(starts),HISTORY,9,*self.shape[2:]),dtype='f8')
        for variable in range(3):
            result[:,:,variable*3]=spatial[:,:,variable,0]
            result[:,:,variable*3+1]=spatial[:,:,variable,1]
            result[:,:,variable*3+2]=dt[:,:,variable]
        return self._finite(result,'window gradients')

    def fit_stats(self,all_window_start_indices,batch_size=8,progress=None):
        """Pool all caller-supplied available samples x six history slots.

        Fit only the training starts supplied by the caller.
        Population variance (ddof=0), per channel/grid; float64 centered block
        moments merged by Chan identities. No complete expanded-window array is
        retained. A callback, if supplied, receives (processed_windows,total).
        """
        starts=self._indices(all_window_start_indices)
        if not isinstance(batch_size,(int,np.integer)) or isinstance(batch_size,bool) or batch_size<=0:
            raise ValueError('batch_size must be a positive integer')
        if np.unique(starts).size!=starts.size:
            raise ValueError('All available window starts must be supplied once each, without duplicates')
        shape=(9,*self.shape[2:])
        mean=np.zeros(shape,dtype='f8');m2=np.zeros(shape,dtype='f8')
        lo=np.full(shape,np.inf);hi=np.full(shape,-np.inf);count=0
        for start in range(0,len(starts),int(batch_size)):
            block=self.get_raw_gradients(starts[start:start+int(batch_size)]).reshape(-1,*shape)
            block_n=block.shape[0]
            bm=block.mean(axis=0,dtype='f8');centered=block-bm
            bm2=np.einsum('nchw,nchw->chw',centered,centered,dtype=np.float64,optimize=True)
            delta=bm-mean;total=count+block_n
            m2+=bm2+delta*delta*(count*block_n/total)
            mean+=delta*(block_n/total)
            count=total;lo=np.minimum(lo,block.min(axis=0));hi=np.maximum(hi,block.max(axis=0))
            if progress is not None:progress(min(start+int(batch_size),len(starts)),len(starts))
        variance=np.maximum(m2/count,0.0);std=np.sqrt(variance)
        # Exact observed constancy avoids false tiny variance from rounded block means.
        constant=(hi==lo);zero=constant|(std==0)
        mean[constant]=lo[constant];std[zero]=1.0
        gradient_units=[f'{unit} {denominator}-1' for unit in self.physical_units for denominator in ('m','m','s')]
        metadata={'schema_version':SCHEMA_VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
                  'fit_scope':'training_windows',
                  'caller_responsibility':'Pass the training window starts for this horizon.',
                  'sample_axis':'provided_window_start_index','history_axis':'six positions pooled with equal weight; shared per-grid statistics',
                  'window_count':len(starts),'pooled_sample_history_count':count,'history':HISTORY,
                  'index_min':int(starts.min()),'index_max':int(starts.max()),
                  'indices_sha256':hashlib.sha256(np.asarray(starts,dtype='<i8').tobytes()).hexdigest(),
                  'physical_array_shape':list(self.shape),'coordinate_sha256':self.coordinate_sha256,
                  'latitude_degrees':self.latitude_deg.tolist(),'longitude_degrees':self.longitude_deg.tolist(),
                  'physical_units':list(self.physical_units),'gradient_units':gradient_units,
                  'channel_order':list(CHANNEL_ORDER),'statistics_channel_order':list(GRADIENT_ORDER),
                  'accumulator_dtype':'float64','variance_ddof':0,'algorithm':'centered block moments with Chan merge',
                  'earth_radius_m':EARTH_RADIUS_M,'time_step_seconds':TIME_STEP_SECONDS,
                  'spatial_stencil':'np.gradient(actual signed radian coordinates, edge_order=1); dx/(R*cos(latitude)), dy/R; east/north positive; no crop or wrap',
                  'temporal_stencil':'six-frame-window-only np.gradient(dt=3600, edge_order=1): central interior and one-sided first/last; never seventh frame',
                  'zero_std_policy':'observed minimum==maximum or exact computed zero std is replaced by 1; constants use exact observed mean; mask is stored',
                  'spatial_gradient_source':'supplied compact physical dx/dy cache' if self.spatial_gradients is not None else 'computed from physical frames',
                  'output_dtype':self.output_dtype.name,'raw_standardization':'original three input channels already standardized by caller; unchanged here'}
        stats=GradientStats(mean,std,zero,lo,hi,count,metadata)
        self.set_stats(stats)
        return stats

    def set_stats(self,stats):
        if not isinstance(stats,GradientStats):raise TypeError('stats must be GradientStats')
        if stats.metadata.get('fit_scope') != 'training_windows':raise ValueError('Expected statistics fitted to training windows')
        if stats.mean.shape!=(9,*self.shape[2:]):raise ValueError('Statistics grid shape differs')
        if stats.metadata.get('physical_array_shape')!=list(self.shape):
            raise ValueError('Statistics physical array shape/horizon length differs')
        if stats.metadata.get('coordinate_sha256')!=self.coordinate_sha256:
            raise ValueError('Statistics coordinates or their ordering differ')
        if stats.metadata.get('physical_units')!=list(self.physical_units):
            raise ValueError('Statistics physical units differ')
        if stats.metadata.get('statistics_channel_order')!=list(GRADIENT_ORDER) or stats.metadata.get('history')!=HISTORY:
            raise ValueError('Statistics channel/history contract differs')
        self.stats=stats
        return stats

    def save_stats(self,npz_path,json_path=None):
        if self.stats is None:raise RuntimeError('fit_stats or load_stats must be called first')
        return self.stats.save(npz_path,json_path)

    def load_stats(self,npz_path,json_path=None):
        return self.set_stats(GradientStats.load(npz_path,json_path))

    def get_windows(self,indices):
        """Return model input [batch,6,12,H,W]; default dtype is float32."""
        if self.stats is None:raise RuntimeError('Fixed statistics are required before get_windows')
        starts=self._indices(indices)
        raw_gradients=self.get_raw_gradients(starts)
        time_indices=starts[:,None]+np.arange(HISTORY,dtype='i8')[None,:]
        unique,inverse=np.unique(time_indices,return_inverse=True)
        original=self._finite(np.asarray(self.standardized[unique]),'already-standardized frames')
        windows=original[inverse].reshape(len(starts),HISTORY,3,*self.shape[2:])
        output=np.empty((len(starts),HISTORY,12,*self.shape[2:]),dtype=self.output_dtype)
        output[:,:,:3]=windows
        raw_gradients-=self.stats.mean;raw_gradients/=self.stats.std
        output[:,:,3:]=raw_gradients
        return self._finite(output,'model inputs after dtype conversion')

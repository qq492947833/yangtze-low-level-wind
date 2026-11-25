"""Deterministic read-only raw inputs for the original Yangtze native splits.

No fit, normalization, imputation, clipping of values, or neural predictions live
here. point_id = native latitude index * 73 + native longitude index.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import numpy as np

import os
CACHE = Path(os.environ.get("YANGTZE_WORK_DIR","work")).resolve() / "gradient" / "source_cache"
rawsurface = np.load(CACHE / "raw_predictors.npy", mmap_mode="r")
rawwind = np.load(CACHE / "raw_wind.npy", mmap_mode="r")
N_TIME, N_LAT, N_LON = rawsurface.shape[1:]
N_POINTS = N_LAT*N_LON
N_FEATURES = 11*3*9
SURFACE_CHANNELS = ('sp', 't2m', 'd2m')
TARGET_CHANNELS = ('u850','u925','u1000','v850','v925','v1000','omega850','omega925','omega1000')
LAGS = np.arange(-10, 1, dtype=np.int64)
_DY = np.repeat(np.arange(-1, 2, dtype=np.int64), 3)
_DX = np.tile(np.arange(-1, 2, dtype=np.int64), 3)

@lru_cache(maxsize=1)
def load_indices():
    """Return decoded, immutable index arrays; see h{lead}_{split}_issue_idx."""
    with np.load(CACHE / "coordinates.npz", allow_pickle=False) as z:
        result = {key: z[key] for key in ("latitude","longitude","level")}
        result["raw_time_ns"] = z["times"].astype("datetime64[ns]").astype(np.int64)
    for h in range(1,7):
        n = N_TIME - h
        test_start = int(.8*n)
        train_stop = int((1-.1/.8)*test_start)
        # The second six-hour window drops five early first-stage outputs.
        bounds = [0,train_stop-5,test_start-5,n-10]
        for split, start, stop in zip(("train","valid","test"),bounds,bounds[1:]):
            issue = np.arange(10+start,10+stop,dtype=np.int64)
            result[f"h{h}_{split}_issue_idx"] = issue
            result[f"h{h}_{split}_target_idx"] = issue+h
    for value in result.values():
        value.setflags(write=False)
    assert len(result['raw_time_ns']) == N_TIME
    assert np.all(np.diff(result['raw_time_ns']) == 3600*10**9)
    return result

def coordinates():
    """Immutable raw_time_ns, lats, lons, levels in original cache order."""
    z = load_indices()
    return dict(raw_time_ns=z['raw_time_ns'], lats=z['latitude'],
                lons=z['longitude'], levels=z['level'],
                time_ns=z['raw_time_ns'], latitude=z['latitude'],
                longitude=z['longitude'], level=z['level'])

def _vector(values, name, lower, upper):
    v = np.asarray(values)
    assert v.ndim == 1 and v.dtype.kind in 'iu', f'{name} must be a 1-D integer array'
    assert np.all(v >= lower) and np.all(v < upper), f'{name} outside [{lower}, {upper})'
    return v.astype(np.int64, copy=False)

def _lead(lead):
    assert isinstance(lead, (int, np.integer)) and not isinstance(lead, (bool, np.bool_))
    assert 1 <= int(lead) <= 6, 'lead must be in 1..6'
    return int(lead)

def _require_finite(a, name):
    valid = np.isfinite(a)
    if not valid.all():
        first = tuple(int(x) for x in np.argwhere(~valid)[0])
        raise ValueError(f'{name} has nonfinite input at block index {first}; investigate source, no imputation applied')

def feature_block(issue_indices, point_ids):
    """Return float64 [P,B,297], ordered lag, surface, dy/dx row-major.

    dy/dx are ARRAY-index offsets: latitude descends in native storage. Neighbor
    coordinates are clipped to the edge, which duplicates the nearest edge cell.
    Values are not clipped. For every row, the latest source time equals issue.
    """
    ii = _vector(issue_indices, 'issue_indices', 10, N_TIME)
    pp = _vector(point_ids, 'point_ids', 0, N_POINTS)
    yy, xx = np.divmod(pp, N_LON)
    ys = np.clip(yy[:,None]+_DY[None,:], 0, N_LAT-1)
    xs = np.clip(xx[:,None]+_DX[None,:], 0, N_LON-1)
    tt = ii[:,None]+LAGS[None,:]
    assert tt.size == 0 or (tt.min() >= 0 and tt.max() < N_TIME)
    values = rawsurface[
        np.arange(3, dtype=np.int64)[None,None,None,:,None],
        tt[None,:,:,None,None],
        ys[:,None,None,None,:],
        xs[:,None,None,None,:],
    ]
    result = np.asarray(values, dtype=np.float64).reshape(len(pp), len(ii), N_FEATURES)
    _require_finite(result, 'surface features')
    return result

def target_block(issue_indices, point_ids, lead):
    """Return float64 [P,B,9] target at issue+lead; final 3 are Pa s-1 omega.

    Existing cache stored w = -0.1 * ERA5 omega. Convert the float32 cache values
    to float64 first and multiply by -10 once, matching Figure 7 arithmetic.
    """
    h = _lead(lead)
    ii = _vector(issue_indices, 'issue_indices', 0, N_TIME-h)
    pp = _vector(point_ids, 'point_ids', 0, N_POINTS)
    yy, xx = np.divmod(pp, N_LON)
    ti = ii+h
    assert ti.size == 0 or (ti.min() >= 0 and ti.max() < N_TIME)
    result = np.asarray(rawwind[
        np.arange(9, dtype=np.int64)[None,None,:],
        ti[None,:,None], yy[:,None,None], xx[:,None,None],
    ], dtype=np.float64)
    result[:,:,6:9] *= -10.0
    _require_finite(result, 'wind targets')
    return result

def native_test(lead):
    """Return dict(issue, target, time); time is valid datetime64[ns]."""
    h = _lead(lead)
    z = load_indices()
    ii, ti = z[f'h{h}_test_issue_idx'], z[f'h{h}_test_target_idx']
    assert np.array_equal(ti, ii+h)
    tt = z['raw_time_ns'][ti].astype('datetime64[ns]')
    tt.setflags(write=False)
    ns=tt.astype(np.int64)
    ns.setflags(write=False)
    return dict(issue=ii, target=ti, time=tt, issues=ii, test_time_ns=ns)

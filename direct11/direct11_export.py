"""Export one frozen Direct11 model and independently read back both physical fields."""
from pathlib import Path
import gc
import hashlib
import json
import os
import shutil

import direct11_common as common
import netCDF4 as nc
import numpy as np
import torch

from direct11_data import Data
from direct11_model import build_model, release_forward_cache


FIELDS = [('u', 'eastward_wind', 'm s-1'), ('v', 'northward_wind', 'm s-1'),
          ('w', 'lagrangian_tendency_of_air_pressure', 'Pa s-1')]
DIMS = ('time', 'level', 'latitude', 'longitude')
TIME_UNITS = 'seconds since 1970-01-01 00:00:00'
TOL = {'rtol': 5e-5, 'atol': 5e-5}
BLOCK = 128


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ranges(issues, block=BLOCK):
    # Raw ERA5 reader expects a contiguous run. The historical test set has gaps.
    edges = np.r_[0, np.flatnonzero(np.diff(issues) != 1) + 1, len(issues)]
    for start, stop in zip(edges[:-1], edges[1:]):
        for lo in range(int(start), int(stop), block):
            yield lo, min(lo + block, int(stop))


def coords(data, issues, lead):
    z = data.z
    reference = np.asarray(z['raw_time_ns'], dtype=np.int64)[issues]
    target = np.asarray(z['raw_time_ns'], dtype=np.int64)[issues + lead]
    require(len(issues) and np.all(np.diff(issues) > 0), 'test issue coordinates must be unique and increasing')
    require(np.all(target - reference == lead * 3600 * 10**9), 'invalid target minus reference time')
    require(np.all(reference % 10**9 == 0), 'timestamp exceeds lossless seconds precision')
    np.testing.assert_array_equal(z['level'], [850, 925, 1000])
    return dict(time=target // 10**9, forecast_reference_time=reference // 10**9,
                latitude=np.asarray(z['latitude']), longitude=np.asarray(z['longitude']),
                level=np.asarray(z['level']))


def load_model(best, device):
    obj = torch.load(best['path'], map_location='cpu', weights_only=False)
    require(isinstance(obj,dict) and obj and all(torch.is_tensor(v) for v in obj.values()),
            'best model must be the frozen bare state_dict')
    model = build_model(11, 3, 'cpu')
    model.load_state_dict(obj, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model


def prediction(data, model, device, lo, hi):
    x, _ = data.batch('test', lo, hi)
    with torch.inference_mode():
        y = model(torch.from_numpy(x).to(device=device, dtype=torch.float32))
        result = data.decode_predictions(y.detach().to('cpu', dtype=torch.float32).numpy())
        release_forward_cache(model)
    require(result.dtype == np.float64 and np.isfinite(result).all(), 'nonfinite decoded prediction')
    return result


def check_shard(path, row):
    require(path.stat().st_size == row['bytes'] and common.sha(path) == row['sha256'],
            'committed inference shard changed at ' + str(path))


def inference_shards(directory, key, data, model, device, issues, batch):
    pointer = directory / 'progress.json'
    state = common.read(pointer) if pointer.is_file() else {'source_key': key, 'chunks': []}
    require(state['source_key'] == key, 'existing inference blocks belong to different sources')
    plan = list(ranges(issues))
    require(len(state['chunks']) <= len(plan), 'invalid inference pointer')
    for i, (lo, hi) in enumerate(plan):
        path = directory / ('prediction_%06d_%06d.npy' % (lo, hi))
        if i < len(state['chunks']):
            row = state['chunks'][i]
            require((row['lo'], row['hi']) == (lo, hi), 'inference chunk positions changed')
            check_shard(path, row)
            continue
        values = np.concatenate([prediction(data, model, device, j, min(j + batch, hi))
                                 for j in range(lo, hi, batch)], axis=0)
        expect = (hi-lo, 9, len(data.z['latitude']), len(data.z['longitude']))
        require(values.shape == expect, 'unexpected inference output dimensions')
        tmp = path.with_suffix('.pending')
        with tmp.open('wb') as f:
            np.save(f, values, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        np.testing.assert_array_equal(np.load(tmp, allow_pickle=False), values)
        digest = common.sha(tmp)
        os.replace(tmp, path)
        state['chunks'].append(dict(lo=lo, hi=hi, bytes=path.stat().st_size, sha256=digest, batch=int(batch), device=device,
                                    physical_gpu=os.environ.get('DIRECT11_PHYSICAL_GPU','cpu')))
        common.atomic(pointer, state)
    return state['chunks']


def shard(directory, lo, hi):
    return np.load(directory / ('prediction_%06d_%06d.npy' % (lo, hi)), allow_pickle=False)


def create_nc(path, coord, lead, attrs, kind):
    ds = nc.Dataset(path, 'w', format='NETCDF4')
    try:
        for name in DIMS:
            ds.createDimension(name, len(coord[name]))
        for name, vector in coord.items():
            dims = ('time',) if name == 'forecast_reference_time' else (name,)
            var = ds.createVariable(name, vector.dtype, dims, fill_value=False)
            var[:] = vector
            if name in ('time', 'forecast_reference_time'):
                var.units, var.calendar, var.standard_name = TIME_UNITS, 'proleptic_gregorian', name
            elif name in ('latitude', 'longitude'):
                var.standard_name, var.units = name, ('degrees_north' if name == 'latitude' else 'degrees_east')
            else:
                var.standard_name, var.units, var.positive, var.axis = 'air_pressure', 'hPa', 'down', 'Z'
        period = ds.createVariable('forecast_period', 'i4', (), fill_value=False)
        period.assignValue(lead)
        period.standard_name, period.units = 'forecast_period', 'hours'
        for field, std, unit in FIELDS:
            v = ds.createVariable(field, 'f8', DIMS, zlib=True, complevel=1, shuffle=True, fill_value=False,
                                  chunksizes=(min(4,len(coord['time'])), 1, len(coord['latitude']), len(coord['longitude'])))
            v.units, v.standard_name = unit, std
            v.coordinates = 'forecast_reference_time forecast_period'
        ds.setncatts(dict(attrs, lead_hours=lead, downstream_w_scale=1.0, output_role=kind,
                          Conventions='CF-1.8', floating_point_encoding='float64; lossless deflate with shuffle'))
        ds.variables['w'].downstream_w_scale = 1.0
        return ds
    except BaseException:
        ds.close()
        raise


def write_staged(directory, path, data, coord, chunks, issues, lead, attrs, kind):
    pending = path.with_suffix('.building.nc')
    with create_nc(pending, coord, lead, attrs, kind) as ds:
        for row in chunks:
            lo, hi = row['lo'], row['hi']
            values = (shard(directory, lo, hi) if kind == 'prediction'
                      else data.physical_truth(issues[lo:hi]))
            for i, (field, _, _) in enumerate(FIELDS):
                ds[field][lo:hi] = values[:, i*3:i*3+3]
        ds.sync()
    with pending.open('r+b') as f:
        os.fsync(f.fileno())
    os.replace(pending, path)


def validate_staged(paths, data, coord, directory, issues, chunks, lead, attrs, selected, device, batch_limit=None):
    samples = sorted({0, len(issues)//2, len(issues)-1})
    sample_rows = []
    replay = load_model(selected, device)
    if batch_limit is None:
        batch_limit = 16 if device == 'cpu' else int(os.environ.get('DIRECT11_MICROBATCH','128'))
    with nc.Dataset(paths['prediction']) as pred, nc.Dataset(paths['truth']) as truth:
        for ds, kind in ((pred, 'prediction'), (truth, 'truth')):
            ds.set_auto_mask(False)
            require(tuple(ds.dimensions) == DIMS and set(ds.variables) == set(coord) | {'forecast_period', 'u', 'v', 'w'},
                    'unexpected NetCDF schema')
            for name, vector in coord.items():
                np.testing.assert_array_equal(ds[name][:], vector)
                expected_dims=('time',) if name=='forecast_reference_time' else (name,)
                require(ds[name].dimensions==expected_dims, 'coordinate dimensions changed')
            require(all(ds[k].units==TIME_UNITS and ds[k].calendar=='proleptic_gregorian'
                        for k in ('time','forecast_reference_time')), 'wrong time coordinates')
            require(ds['latitude'].units=='degrees_north' and ds['longitude'].units=='degrees_east'
                    and ds['level'].units=='hPa', 'wrong spatial coordinate units')
            require(ds.output_role == kind and ds.lead_hours == int(ds['forecast_period'][...]) == lead,
                    'forecast metadata changed')
            require(ds['forecast_period'].units == 'hours' and ds['forecast_period'].dimensions==()
                    and ds.downstream_w_scale == ds['w'].downstream_w_scale == 1, 'units changed')
            for k,v in attrs.items():
                require(ds.getncattr(k)==v, 'source metadata changed: '+k)
            for field, std, unit in FIELDS:
                v=ds[field]
                require(v.dimensions == DIMS and v.dtype == np.dtype('float64') and v.units == unit
                        and v.standard_name == std and v.filters()['zlib'] and 'least_significant_digit' not in v.ncattrs(),
                        'NetCDF field contract mismatch')
        for row in chunks:
            lo, hi = row['lo'], row['hi']
            source_truth = data.physical_truth(issues[lo:hi])
            source_pred = shard(directory, lo, hi)
            for i, (field, _, _) in enumerate(FIELDS):
                values = pred[field][lo:hi]
                require(np.isfinite(values).all(), 'nonfinite prediction on disk')
                np.testing.assert_array_equal(values, source_pred[:, i*3:i*3+3])
                disk_truth = truth[field][lo:hi]
                require(np.isfinite(disk_truth).all(), 'nonfinite truth on disk')
                np.testing.assert_array_equal(disk_truth, source_truth[:, i*3:i*3+3])
        for i in samples:
            row=next(r for r in chunks if r['lo']<=i<r['hi'])
            # Match the original batch shape/position while using an independently
            # loaded selected model and a fresh read of the actual input data.
            lo=row['lo'] + ((i-row['lo'])//row['batch'])*row['batch']
            hi=min(lo+row['batch'],row['hi'])
            original_range=[lo,hi]
            if hi-lo>batch_limit:
                lo += ((i-lo)//batch_limit)*batch_limit
                hi=min(hi,lo+batch_limit)
            replayed = prediction(data, replay, device, lo, hi)[i-lo]
            stored = np.concatenate([pred[f][i] for f, _, _ in FIELDS], axis=0)
            delta = np.abs(replayed-stored)
            np.testing.assert_allclose(replayed, stored, **TOL)
            sample_rows.append(dict(position=i, original_inference_device=row['device'],
                                    original_physical_gpu=row['physical_gpu'],replay_device=device,
                                    original_batch_range=original_range,replay_batch_range=[lo,hi],issue_index=int(issues[i]), target_seconds=int(coord['time'][i]),
                                    exact=bool(np.array_equal(replayed, stored)), max_abs_by_channel=np.max(delta,axis=(1,2)).tolist()))
    return dict(full_readback_samples=len(issues), raw_truth_exact=True, committed_prediction_exact=True,
                sample_replay=sample_rows, sample_tolerance_physical_units=TOL, prediction_dtype='float64',
                independent_saved_model_reload=True)


def publish(staged, target, digest):
    if target.exists():
        require(target.is_file() and target.stat().st_size==staged.stat().st_size and common.sha(target)==digest,
                'existing final file differs; publication stopped at '+str(target))
        return
    # Pending belongs to this exact content in the same final filesystem.
    pending = target.with_name('.' + target.name + '.' + digest + '.pending')
    if not pending.exists() or pending.stat().st_size != staged.stat().st_size or common.sha(pending) != digest:
        with staged.open('rb') as source, pending.open('wb') as destination:
            shutil.copyfileobj(source, destination, length=8*1024*1024)
            destination.flush()
            os.fsync(destination.fileno())
    require(common.sha(pending) == digest, 'cross-volume copy differed')
    try:
        if os.name == 'nt':
            os.rename(pending, target)  # Windows fails if the destination exists.
        else:
            os.link(pending, target)
            pending.unlink()
    except FileExistsError:
        require(common.sha(target)==digest, 'publication collision at '+str(target))
    require(target.stat().st_size==staged.stat().st_size and common.sha(target)==digest, 'published identity mismatch')


def export(lead, device, physical_gpu=None):
    lead = int(lead)
    require(lead in range(1,7) and device in ('cpu','cuda:0'), 'invalid lead or device')
    cfg, science = common.frozen()
    from direct11_worker import apply_precision
    precision = science['precision']
    apply_precision(precision)
    if device == 'cuda:0':
        batch = int(os.environ.get('DIRECT11_MICROBATCH','128'))
    else:
        require(physical_gpu is None, 'CPU export received a GPU argument')
        torch.set_num_threads(min(4, max(1, int(os.getenv('DIRECT11_CPU_THREADS','2')))))
        batch = 16
    require(batch>0, 'invalid inference batch')
    common.FINAL.mkdir(parents=True,exist_ok=True)
    instance = common.ROOT/'instances'/('lead_'+str(lead))
    done = instance/'COMPLETION.json'
    name = 'Direct11_'+str(lead)+'hour_'
    destinations = {'prediction':common.FINAL/(name+'predicty.nc'), 'truth':common.FINAL/(name+'testy.nc')}
    with common.lock(instance/'export.lock'):
        training = instance/'TRAINING_COMPLETION.json'
        trained = common.read(training)
        require(trained['lead']==lead and trained['status']=='PASS_FROZEN_TRAINING_SELECTED' and
                trained['frozen_science_sha']==cfg['frozen_science_sha'], 'training receipt mismatch')
        selected = trained['selected_model']
        best = trained['best_model']
        common.assert_identity(selected,True)
        if done.exists():
            result=common.read(done)
            require(result['lead']==lead and result['status']=='PASS_FINAL_NC_READBACK'
                    and result['frozen_science_sha']==cfg['frozen_science_sha']
                    and result['training_receipt_sha256']==common.sha(training)
                    and [Path(a['path']) for a in result['artifacts']]==list(destinations.values()),
                    'existing accepted receipt belongs to a different job')
            for artifact in result['artifacts']:
                common.assert_identity(artifact,True)
            return result
        data = Data(lead)
        issues = np.asarray(data.splits['test'],np.int64)
        coord = coords(data,issues,lead)
        attrs=dict(frozen_science_sha=cfg['frozen_science_sha'], best_model_sha256=best['sha256'],
                   selected_model_sha256=selected['sha256'], selected_epoch=int(trained['selected_epoch']),
                   selected_validation_mse=float(trained['selected_validation_mse']),stop_reason=trained['stop_reason'],
                   best_epoch=int(trained['best_epoch']), training_receipt_sha256=common.sha(training),
                   data_manifest_sha256=common.sha(common.ROOT/'provenance/DATA_MANIFEST.json'),
                   inference_precision=json.dumps(precision,sort_keys=True), inference_mode='eval, FP32, no autocast',
                   input_layout='historical channels-last frame storage, logical N,T,C,H,W',
                   task_id=cfg['task_id'], exporter_sha256=common.sha(Path(__file__)),
                   truth_source='original decoded ERA5 pressure-level U, V and vertical velocity',
                   physical_omega_conversion='prediction: inverse training-label normalization, float32 stored field, x -10 once; truth: raw ERA5 omega',
                   test_membership='original historical Stage 1 test target times')
        key=hashlib.sha256(json.dumps(attrs,sort_keys=True).encode()).hexdigest()
        directory=instance/'export_work'
        directory.mkdir(exist_ok=True)
        model=load_model(selected,device)
        chunks=inference_shards(directory,key,data,model,device,issues,batch)
        del model
        gc.collect()
        if device=='cuda:0':
            torch.cuda.empty_cache()
        staged={kind:directory / dst.name for kind,dst in destinations.items()}
        for kind,path in staged.items():
            if not path.exists():
                write_staged(directory,path,data,coord,chunks,issues,lead,attrs,kind)
        audit=validate_staged(staged,data,coord,directory,issues,chunks,lead,attrs,selected,device,batch)
        common.assert_identity(selected,True)
        require(common.sha(training)==attrs['training_receipt_sha256'], 'training receipt changed during export')
        digests={kind:common.sha(path) for kind,path in staged.items()}
        # Precheck both final names before publishing either member of a pair.
        for kind,target in destinations.items():
            if target.exists():
                require(common.sha(target)==digests[kind], 'existing formal output differs: '+str(target))
        artifacts=[]
        for kind,target in destinations.items():
            publish(staged[kind],target,digests[kind])
            item=common.identity(target)
            item.update(sha256=digests[kind],role=kind)
            artifacts.append(item)
        result=dict(status='PASS_FINAL_NC_READBACK',lead=lead,frozen_science_sha=cfg['frozen_science_sha'],
                    training_receipt_sha256=attrs['training_receipt_sha256'],best_model_sha256=best['sha256'],
                    selected_model=selected,selected_model_sha256=selected['sha256'],selected_epoch=int(trained['selected_epoch']),
                    selected_validation_mse=float(trained['selected_validation_mse']),stop_reason=trained['stop_reason'],
                    best_epoch=int(trained['best_epoch']),artifacts=artifacts,verification=audit,
                    export_device=device,physical_gpu=physical_gpu,inference_batch=batch,utc=common.stamp())
        common.atomic(done,result)
        return result

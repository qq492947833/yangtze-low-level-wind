"""Score the five 1 h PINN configurations on common times and component-wise finite samples."""


def main():
    from pathlib import Path
    import json
    import os
    import uuid
    import numpy as np
    import pandas as pd
    import xarray as xr

    try:
        from IPython.display import display
    except ImportError:
        def display(value):
            print(value)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    DATA_DIR = Path(os.environ.get('YANGTZE_RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results'))
    OUTPUT_NC = DATA_DIR / 'da_transunet_1hour_pinn_comparison_metrics.nc'
    TIME_CHUNK = 128
    TIME_START = None
    TIME_END = None
    TRUTH_RTOL = 1.e-6
    TRUTH_ATOL = 1.e-8
    COMPONENTS = ['u', 'v', 'w']

    EXPERIMENTS = [
        dict(id='no_pinn', label='No PINN', suffix=''),
        dict(id='full_pinn', label='Full PINN', suffix='_pinn'),
        dict(id='no_continuity', label='w/o Continuity', suffix='_pinn_no_continuity'),
        dict(id='no_vorticity', label='w/o Vorticity', suffix='_pinn_no_vorticity'),
        dict(id='no_shear', label='w/o Shear', suffix='_pinn_no_shear'),
    ]

    FILE_OVERRIDES = {}

    if TIME_CHUNK < 1:
        raise ValueError('TIME_CHUNK 必须 >= 1。')
    experiment_ids = [item['id'] for item in EXPERIMENTS]
    if len(experiment_ids) < 2 or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError('至少需要两个实验，且每个 id 必须唯一。')

    input_paths = {}
    missing_paths = []
    for item in EXPERIMENTS:
        eid = item['id']
        input_paths[eid] = {}
        for component in COMPONENTS:
            input_paths[eid][component] = {}
            for kind in ['testy', 'predicty']:
                default_path = DATA_DIR / f'da_transunet_{kind}_{component}_1_24year_changjiang{item["suffix"]}.nc'
                path = Path(FILE_OVERRIDES.get((eid, kind, component), default_path))
                input_paths[eid][component][kind] = path
                if not path.is_file():
                    missing_paths.append(str(path))
                if path.resolve() == OUTPUT_NC.resolve():
                    raise ValueError('指标输出路径不能与输入文件相同。')
    if missing_paths:
        raise FileNotFoundError('以下 NC 未找到；请在上方核对 suffix 或 FILE_OVERRIDES：\n' + '\n'.join(missing_paths))

    def new_metric_state(grid_shape):
        state = {key: np.zeros(grid_shape, dtype=np.float64)
                 for key in ['mean_y', 'mean_p', 'm2_y', 'm2_p', 'cross', 'sse']}
        state['n'] = np.zeros(grid_shape, dtype=np.int64)
        state['n_test'] = np.zeros(grid_shape, dtype=np.int64)
        for key in ['test_min', 'pair_y_min', 'pair_p_min']:
            state[key] = np.full(grid_shape, np.inf, dtype=np.float64)
        for key in ['test_max', 'pair_y_max', 'pair_p_max']:
            state[key] = np.full(grid_shape, -np.inf, dtype=np.float64)
        return state

    def update_metric_state(state, truth, prediction, common_mask):
        y = np.asarray(truth, dtype=np.float64)
        p = np.asarray(prediction, dtype=np.float64)
        if y.shape != p.shape or y.shape != common_mask.shape or y.shape[1:] != state['n'].shape:
            raise ValueError('truth、prediction、common_mask 的形状不一致。')
        finite_y = np.isfinite(y)
        valid = np.asarray(common_mask, dtype=bool) & finite_y & np.isfinite(p)
        nb = valid.sum(axis=0)
        safe_y, safe_p = np.where(valid, y, 0.), np.where(valid, p, 0.)
        mean_y = np.divide(safe_y.sum(axis=0), nb, out=np.zeros_like(state['mean_y']), where=nb>0)
        mean_p = np.divide(safe_p.sum(axis=0), nb, out=np.zeros_like(state['mean_p']), where=nb>0)
        dy = np.where(valid, safe_y-mean_y, 0.)
        dp = np.where(valid, safe_p-mean_p, 0.)
        na = state['n']
        n = na+nb
        weight = np.divide(nb, n, out=np.zeros_like(mean_y), where=n>0)
        cross_weight = na*weight
        delta_y, delta_p = mean_y-state['mean_y'], mean_p-state['mean_p']
        state['m2_y'] += (dy*dy).sum(axis=0)+delta_y**2*cross_weight
        state['m2_p'] += (dp*dp).sum(axis=0)+delta_p**2*cross_weight
        state['cross'] += (dy*dp).sum(axis=0)+delta_y*delta_p*cross_weight
        state['mean_y'] += delta_y*weight
        state['mean_p'] += delta_p*weight
        state['sse'] += ((safe_p-safe_y)**2).sum(axis=0)
        state['n'] = n
        state['n_test'] += finite_y.sum(axis=0)
        for prefix, values, mask in [('test', y, finite_y), ('pair_y', y, valid), ('pair_p', p, valid)]:
            state[prefix+'_min'] = np.minimum(state[prefix+'_min'], np.min(np.where(mask, values, np.inf), axis=0))
            state[prefix+'_max'] = np.maximum(state[prefix+'_max'], np.max(np.where(mask, values, -np.inf), axis=0))

    def finish_metric_state(state):
        n = state['n']
        rmse = np.sqrt(np.divide(state['sse'], n, out=np.full(n.shape, np.nan), where=n>0))
        test_min = np.where(state['n_test']>0, state['test_min'], np.nan)
        test_max = np.where(state['n_test']>0, state['test_max'], np.nan)
        test_range = test_max-test_min
        rmspe = np.divide(rmse, test_range, out=np.full(n.shape, np.nan), where=test_range>0)
        denom = np.sqrt(np.maximum(state['m2_y'], 0.))*np.sqrt(np.maximum(state['m2_p'], 0.))
        correlation_valid = ((n>=2) & (denom>0) &
                             (state['pair_y_max']>state['pair_y_min']) &
                             (state['pair_p_max']>state['pair_p_min']))
        pearson_r = np.divide(state['cross'], denom, out=np.full(n.shape, np.nan), where=correlation_valid)
        pearson_r = np.clip(pearson_r, -1., 1.)
        return dict(pearson_r=pearson_r, rmspe=rmspe, rmse=rmse, test_min=test_min,
                    test_max=test_max, test_range=test_range, n_valid=n.astype(np.int32),
                    n_test_finite=state['n_test'].astype(np.int32))

    def summarize_grid_metric(values):
        common_grid = np.isfinite(values).all(dim='experiment')
        valid_values = values.where(common_grid)
        by_level = valid_values.mean(dim=('latitude', 'longitude'), skipna=True)
        overall = valid_values.mean(dim=('level', 'latitude', 'longitude'), skipna=True)
        return by_level, overall, common_grid

    _open_datasets = []
    input_data = {}
    experiment_ids = [item['id'] for item in EXPERIMENTS]
    reference_id = experiment_ids[0]
    spatial_dims = ('level', 'latitude', 'longitude')
    all_dims = ('time',) + spatial_dims
    OUTPUT_NC = Path(OUTPUT_NC)
    alignment_rows = []
    common_time = None
    reference = None

    try:
        for eid in experiment_ids:
            input_data[eid] = {}
            for component in COMPONENTS:
                input_data[eid][component] = {}
                for kind in ['testy', 'predicty']:
                    path = Path(input_paths[eid][component][kind])
                    if path.resolve() == OUTPUT_NC.resolve():
                        raise ValueError('指标输出路径不能与输入文件相同。')
                    ds = xr.open_dataset(path, engine='netcdf4', cache=False, chunks=None)
                    _open_datasets.append(ds)
                    da = ds[f'{kind}_{component}']
                    aliases = {'valid_time': 'time', 'pressure_level': 'level',
                               'lat': 'latitude', 'lon': 'longitude'}
                    da = da.rename({a: b for a, b in aliases.items()
                                    if a in da.dims and b not in da.dims})
                    if set(da.dims) != set(all_dims):
                        raise ValueError(f'{path.name}: 维度必须为 {all_dims}。')
                    da = da.transpose(*all_dims)
                    for dim in all_dims:
                        if dim not in da.coords or da[dim].dims != (dim,):
                            raise ValueError(f'{path.name}: 缺少一维 {dim} 坐标。')
                        if not da.get_index(dim).is_unique:
                            raise ValueError(f'{path.name}: {dim} 存在重复，无法唯一配对。')
                    if not np.issubdtype(da.time.dtype, np.datetime64):
                        raise ValueError(f'{path.name}: time 未正确解码为日期。')
                    times = pd.DatetimeIndex(da.time.values)
                    if times.hasnans:
                        raise ValueError(f'{path.name}: time 中存在 NaT。')
                    selected = times
                    if TIME_START is not None:
                        selected = selected[selected >= pd.Timestamp(TIME_START)]
                    if TIME_END is not None:
                        selected = selected[selected <= pd.Timestamp(TIME_END)]
                    common_time = (selected if common_time is None else
                                   common_time.intersection(selected))
                    if reference is None:
                        reference = da
                    for dim in spatial_dims:
                        source, target = da.get_index(dim), reference.get_index(dim)
                        positions = source.get_indexer(target)
                        if len(source) != len(target) or np.any(positions < 0):
                            raise ValueError(f'{path.name}: {dim} 网格不同，不能只靠时间对齐比较。')
                        if not source.equals(target):
                            da = da.isel({dim: positions})
                    input_data[eid][component][kind] = da
                    alignment_rows.append(dict(experiment=eid, component=component,
                                               kind=kind, original_count=len(times),
                                               selected_count=len(selected)))

        common_time = common_time.sort_values()
        if len(common_time) < 2:
            raise ValueError('所有文件的共同有效时次不足 2 个，请核对 time 坐标及选定时段。')
        test_times = common_time.to_numpy(dtype='datetime64[ns]')
        n_time = len(test_times)
        for eid in experiment_ids:
            for component in COMPONENTS:
                for kind in ['testy', 'predicty']:
                    input_data[eid][component][kind] = input_data[eid][component][kind].sel(time=test_times)
        alignment_table = pd.DataFrame(alignment_rows)
        alignment_table['aligned_count'] = n_time
        alignment_table['excluded_from_selected'] = alignment_table['selected_count'] - n_time
        print(f'共同测试时次：{n_time}；{test_times[0]} 至 {test_times[-1]}')
        display(alignment_table)

        metric_coordinates = {dim: reference[dim].values.copy() for dim in spatial_dims}
        grid_shape = tuple(len(metric_coordinates[dim]) for dim in spatial_dims)
        shape = (len(EXPERIMENTS), len(COMPONENTS)) + grid_shape
        float_metrics = ['pearson_r', 'rmspe', 'rmse', 'test_min', 'test_max', 'test_range']
        count_metrics = ['n_valid', 'n_test_finite', 'n_invalid_prediction']
        metric_arrays = {name: np.full(shape, np.nan) for name in float_metrics}
        metric_arrays.update({name: np.zeros(shape, dtype=np.int32) for name in count_metrics})
        diagnostic_shape = (len(EXPERIMENTS), len(COMPONENTS))
        truth_mismatch_count = np.zeros(diagnostic_shape, dtype=np.int64)
        truth_max_abs_diff = np.full(diagnostic_shape, np.nan)

        for ic, component in enumerate(COMPONENTS):
            states = [new_metric_state(grid_shape) for _ in EXPERIMENTS]
            for start in tqdm(range(0, n_time, TIME_CHUNK), desc=f'计算 {component}', unit='块'):
                stop = min(start + TIME_CHUNK, n_time)
                truths, predictions = [], []
                for eid in experiment_ids:
                    group = input_data[eid][component]
                    truths.append(np.asarray(group['testy'].isel(time=slice(start, stop)).values, dtype=np.float64))
                    predictions.append(np.asarray(group['predicty'].isel(time=slice(start, stop)).values, dtype=np.float64))

                common_mask = np.ones(truths[0].shape, dtype=bool)
                for truth, prediction in zip(truths, predictions):
                    common_mask &= np.isfinite(truth) & np.isfinite(prediction)
                for ie, (truth, prediction, state) in enumerate(zip(truths, predictions, states)):
                    metric_arrays['n_invalid_prediction'][ie, ic] += (~np.isfinite(prediction)).sum(axis=0).astype(np.int32)
                    close = np.isclose(truth, truths[0], rtol=TRUTH_RTOL, atol=TRUTH_ATOL, equal_nan=True)
                    truth_mismatch_count[ie, ic] += np.count_nonzero(~close)
                    both_finite = np.isfinite(truth) & np.isfinite(truths[0])
                    if both_finite.any():
                        max_diff = np.max(np.abs(truth[both_finite] - truths[0][both_finite]))
                        truth_max_abs_diff[ie, ic] = np.fmax(truth_max_abs_diff[ie, ic], max_diff)
                    update_metric_state(state, truth, prediction, common_mask)
            for ie, state in enumerate(states):
                for name, value in finish_metric_state(state).items():
                    metric_arrays[name][ie, ic] = value
            if not np.any(metric_arrays['n_valid'][:, ic] > 0):
                raise ValueError(f'{component}: 对齐后没有共同有效数据。')
            print(f'{component}: 指标计算完成。')
    finally:
        for _ds in _open_datasets:
            _ds.close()

    dims = ('experiment', 'component') + spatial_dims
    coordinates = dict(experiment=experiment_ids, component=COMPONENTS, **metric_coordinates)
    metrics_ds = xr.Dataset({name: (dims, value) for name, value in metric_arrays.items()}, coords=coordinates)
    metrics_ds = metrics_ds.assign_coords(
        experiment_label=('experiment', [item['label'] for item in EXPERIMENTS]),
        evaluation_time=('evaluation_time', test_times))
    metrics_ds['truth_mismatch_count'] = (('experiment', 'component'), truth_mismatch_count)
    metrics_ds['truth_max_abs_diff'] = (('experiment', 'component'), truth_max_abs_diff)
    metrics_ds['truth_mismatch_count'].attrs.update(
        long_name='number of target values differing from the first experiment after time alignment',
        rtol=float(TRUTH_RTOL), atol=float(TRUTH_ATOL), equal_nan='yes')
    metrics_ds['truth_max_abs_diff'].attrs['long_name'] = 'maximum absolute target difference where both targets are finite'
    metrics_ds['latitude'].attrs.update(units='degrees_north', standard_name='latitude')
    metrics_ds['longitude'].attrs.update(units='degrees_east', standard_name='longitude')
    metrics_ds['level'].attrs.update(units='hPa', long_name='pressure level')
    metrics_ds['pearson_r'].attrs.update(units='1', long_name='temporal Pearson correlation at each grid point')
    metrics_ds['rmspe'].attrs.update(
        units='1', long_name='range-normalized RMSE (user-defined RMSPE)',
        formula='sqrt(mean((prediction-truth)^2)) / (max(truth)-min(truth))',
        multiplied_by_100='no',
        denominator='all finite truth values of this experiment over the aligned test times, per component/level/grid point')
    for name in ['rmse', 'test_min', 'test_max', 'test_range']:
        metrics_ds[name].attrs['units_note'] = 'native component units in source NC; no extra conversion'
    for name in count_metrics:
        metrics_ds[name].attrs['units'] = '1'

    for metric in ['pearson_r', 'rmspe']:
        by_level, overall, common_grid = summarize_grid_metric(metrics_ds[metric])
        metrics_ds[metric + '_mean_by_level'] = by_level
        metrics_ds[metric + '_mean_all_levels'] = overall
        metrics_ds[metric + '_common_grid'] = common_grid.astype(np.int8)
        metrics_ds[metric + '_common_grid_count_by_level'] = common_grid.sum(dim=('latitude', 'longitude')).astype(np.int32)

    has_truth_difference = bool(np.any(truth_mismatch_count > 0))
    metrics_ds.attrs.update(
        title='DA-TransUNet PINN comparison on common test timestamps',
        experiment_count=len(EXPERIMENTS), selected_time_count=n_time,
        selected_time_start=str(test_times[0]), selected_time_end=str(test_times[-1]),
        time_alignment='sorted exact timestamp intersection across all experiments, components, truth and prediction files',
        truth_policy='each experiment is evaluated against its own aligned testy',
        truth_diagnostic_reference=reference_id,
        truth_consistency='differences_detected' if has_truth_difference else 'passed_within_tolerance',
        time_sample_policy='common finite truth/prediction mask across all experiments at each component/level/grid point',
        undefined_policy='Pearson r is NaN for fewer than 2 pairs or a constant paired series; RMSPE is NaN for zero truth range',
        spatial_average='unweighted mean over common finite grid cells for each metric',
        source_paths=json.dumps({eid: {comp: {kind: str(path) for kind, path in paths.items()}
            for comp, paths in group.items()} for eid, group in input_paths.items()}, ensure_ascii=False),
        time_alignment_counts=alignment_table.to_json(orient='records', force_ascii=False))

    OUTPUT_NC.parent.mkdir(parents=True, exist_ok=True)
    temporary_nc = OUTPUT_NC.with_name(OUTPUT_NC.stem + '.' + uuid.uuid4().hex + '.tmp.nc')
    encoding = {name: dict(zlib=True, complevel=9, shuffle=True) for name in metrics_ds.data_vars}
    try:
        metrics_ds.to_netcdf(temporary_nc, engine='netcdf4', format='NETCDF4', encoding=encoding)
        os.replace(temporary_nc, OUTPUT_NC)
    finally:
        if temporary_nc.exists():
            temporary_nc.unlink()

    print('已保存：', OUTPUT_NC)
    if has_truth_difference:
        print('提醒：时间对齐后，部分实验的 testy 仍与参考实验不同。')
        print('已使用各组自己的 testy 计算；请结合下表检查差异大小。')
        print('若差异不是舍入误差，应核对真值来源、反标准化及时间标签，再解释实验增益。')
    display(metrics_ds[['truth_mismatch_count', 'truth_max_abs_diff']].to_dataframe())
    print('总体均值：')
    display(metrics_ds[['pearson_r_mean_all_levels', 'rmspe_mean_all_levels']].to_dataframe())
    print('分层均值：')
    display(metrics_ds[['pearson_r_mean_by_level', 'rmspe_mean_by_level']].to_dataframe())


if __name__ == "__main__":
    main()

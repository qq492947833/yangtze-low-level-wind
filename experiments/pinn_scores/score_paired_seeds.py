"""Compute grid scores and paired changes for No PINN and no-shear PINN at seeds 42 and 2026."""


def main():
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd

    RESULT_DIR = Path(os.environ.get('YANGTZE_RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results'))
    INDEX_PATH = RESULT_DIR / 'da_transunet_1hour_pinn_seed_repeats_v1_index.json'
    OUTPUT_DIR = RESULT_DIR
    OUTPUT_PREFIX = 'da_transunet_1hour_pinn_seed_metrics_v1'
    CHUNK_SIZE = 128
    TIME_ROUND_TOLERANCE_NS = 1_000_000  # 只容许 <=1 ms 的整点时间解码舍入
    TRUTH_RTOL, TRUTH_ATOL = 1e-6, 1e-7  # 仅用于审计，不用别组 truth 替换本组 truth

    COMPONENTS = ['u', 'v', 'w']
    EXPECTED_SEEDS = [42, 2026]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import json
    import warnings
    from pathlib import PureWindowsPath

    def read_json(path):
        with Path(path).open('r', encoding='utf-8') as handle:
            return json.load(handle)

    def resolve_saved_path(value, parent):
        path = Path(value)
        candidates = [path] if path.is_absolute() else [Path(parent) / path]
        candidates.append(RESULT_DIR / PureWindowsPath(str(value)).name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f'找不到配对种子训练输出：{value}。请核对 RESULT_DIR 和文件是否保存完整。')

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f'Missing paired-seed index: {INDEX_PATH}')
    index = read_json(INDEX_PATH)
    if index.get('suite_id') != 'seed_repeats':
        raise ValueError('Expected the seed_repeats index from pinn_paired_seeds.ipynb.')
    runs = []
    for item in index['runs']:
        manifest_path = resolve_saved_path(item, INDEX_PATH.parent)
        run = read_json(manifest_path)
        if run.get('status') != 'completed':
            raise RuntimeError(f"{run.get('run_id')} 尚未完整保存，请先完成配对种子训练。")
        run['resolved_paths'] = {
            f'{kind}_{component}': resolve_saved_path(run['paths'][f'{kind}_{component}'], manifest_path.parent)
            for kind in ['testy', 'predicty'] for component in COMPONENTS
        }
        runs.append(run)
    expected = {(s, e) for s in EXPECTED_SEEDS for e in ['no_pinn', 'no_shear']}
    observed = [(int(r['seed']), r['experiment']) for r in runs]
    if len(runs) != 4 or set(observed) != expected or len({r['run_id'] for r in runs}) != 4:
        raise ValueError(f'需要两个种子各自的 No PINN / w/o shear，共四组；实际：{observed}')
    runs.sort(key=lambda r: (int(r['seed']), r['experiment'] != 'no_pinn'))
    for run in runs:
        alpha = float(run['global_scale'])
        if alpha != (0.0 if run['experiment'] == 'no_pinn' else 1.0):
            raise ValueError(f"{run['run_id']} 的整体倍率与配对种子设计不同。")
        if float(run['lambda_shear']) != 0:
            raise ValueError('这里仅比较 No PINN 和 w/o shear。')
    fingerprints = {r.get('dataset_fingerprint') for r in runs}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError('配对种子训练的数据指纹缺失或不一致，不能视为相同数据的配对实验。')
    for seed in EXPECTED_SEEDS:
        group = [r for r in runs if int(r['seed']) == seed]
        hashes = {r.get('init_state_sha256') for r in group}
        if None in hashes or len(hashes) != 1:
            raise ValueError(f'seed={seed} 的两种方法初始模型参数不一致。')
        if len({r.get('testy_standardized_sha256') for r in group}) > 1:
            warnings.warn(f'seed={seed} 的目标记录指纹不同；指标仍使用各自的 testy，并输出真值差异审计。')

    RUN_IDS = [r['run_id'] for r in runs]
    try:
        from IPython.display import display
    except ImportError:
        display = print
    display(pd.DataFrame([{
        'run_id': r['run_id'], 'seed': r['seed'], 'experiment': r['experiment'],
        'best_epoch': r.get('best_epoch'),
        'validation_data_loss': r.get('selected_validation_data_loss', np.nan),
        'testy_u': str(r['resolved_paths']['testy_u']),
        'predicty_u': str(r['resolved_paths']['predicty_u']),
    } for r in runs]))

    import os
    import uuid
    import xarray as xr

    def canonical_array(dataset, name):
        if name not in dataset.data_vars:
            raise KeyError(f'NC 中缺少 {name}；实际变量为 {list(dataset.data_vars)}。')
        array = dataset[name]
        rename = {}
        for old, new in [('lat', 'latitude'), ('lon', 'longitude'),
                         ('pressure_level', 'level'), ('valid_time', 'time')]:
            if old in array.dims and new not in array.dims:
                rename[old] = new
        array = array.rename(rename)
        dims = ('time', 'level', 'latitude', 'longitude')
        if set(array.dims) != set(dims):
            raise ValueError(f'{name} 的维度为 {array.dims}，预期 {dims}。')
        array = array.transpose(*dims)
        raw_times = array.time.values
        if not np.issubdtype(raw_times.dtype, np.datetime64):
            raise TypeError(f'{name} 时间未解码为 datetime64；本 notebook 需要 01 保存的 standard 日历。')
        times = pd.DatetimeIndex(raw_times.astype('datetime64[ns]'))
        if times.hasnans or len(times) == 0:
            raise ValueError(f'{name} 的时间为空或包含 NaT。')
        rounded = times.round('h')
        error_ns = np.abs(times.values.astype('datetime64[ns]').astype('int64') -
                          rounded.values.astype('datetime64[ns]').astype('int64'))
        if error_ns.max() > TIME_ROUND_TOLERANCE_NS:
            raise ValueError(f'{name} 偏离整点超过允许的舍入误差，不能自动对齐。')
        if rounded.has_duplicates or not rounded.is_monotonic_increasing:
            raise ValueError(f'{name} 时间重复或倒序，请核对文件。')
        for dim in dims[1:]:
            if array[dim].size == 0 or len(np.unique(array[dim])) != array[dim].size:
                raise ValueError(f'{name} 的 {dim} 为空或包含重复坐标。')
        return array.assign_coords(time=rounded.values), int(np.count_nonzero(error_ns))

    def require_spatial_coordinates(reference, candidate, label):
        for dim in ['level', 'latitude', 'longitude']:
            if not np.array_equal(reference[dim].values, candidate[dim].values):
                raise ValueError(f'{label} 的 {dim} 与其他文件不一致；不自动插值或按位置拼接。')

    class GridAccumulator:
        """逐格点在线合并中心矩，避免用 E[x²]-E[x]² 计算相关时的消减误差。"""
        def __init__(self, shape):
            self.n = np.zeros(shape, dtype=np.int64)
            for name in ['mean_y', 'mean_p', 'm2_y', 'm2_p', 'cross', 'sse']:
                setattr(self, name, np.zeros(shape, dtype=np.float64))
            self.minimum = np.full(shape, np.inf)
            self.maximum = np.full(shape, -np.inf)
            self.pred_minimum = np.full(shape, np.inf)
            self.pred_maximum = np.full(shape, -np.inf)

        def update(self, prediction, truth, valid):
            n = valid.sum(axis=0, dtype=np.int64)
            denominator = np.maximum(n, 1)
            y = np.where(valid, truth, 0.0)
            p = np.where(valid, prediction, 0.0)
            mean_y = y.sum(axis=0, dtype=np.float64) / denominator
            mean_p = p.sum(axis=0, dtype=np.float64) / denominator
            cy = np.where(valid, y - mean_y, 0.0)
            cp = np.where(valid, p - mean_p, 0.0)
            total = self.n + n
            dy, dp = mean_y - self.mean_y, mean_p - self.mean_p
            weight = self.n.astype(np.float64) * n / np.maximum(total, 1)
            self.m2_y += np.sum(cy * cy, axis=0) + dy * dy * weight
            self.m2_p += np.sum(cp * cp, axis=0) + dp * dp * weight
            self.cross += np.sum(cy * cp, axis=0) + dy * dp * weight
            self.mean_y += dy * n / np.maximum(total, 1)
            self.mean_p += dp * n / np.maximum(total, 1)
            self.n = total
            self.sse += np.sum(np.where(valid, p - y, 0.0) ** 2, axis=0, dtype=np.float64)
            self.minimum = np.minimum(self.minimum, np.min(np.where(valid, truth, np.inf), axis=0))
            self.maximum = np.maximum(self.maximum, np.max(np.where(valid, truth, -np.inf), axis=0))
            self.pred_minimum = np.minimum(self.pred_minimum, np.min(np.where(valid, prediction, np.inf), axis=0))
            self.pred_maximum = np.maximum(self.pred_maximum, np.max(np.where(valid, prediction, -np.inf), axis=0))

        def finish(self):
            value_range = self.maximum - self.minimum
            with np.errstate(divide='ignore', invalid='ignore'):
                rmse = np.sqrt(self.sse / self.n)
                denominator = np.sqrt(np.maximum(self.m2_y, 0) * np.maximum(self.m2_p, 0))
                pearson_r = np.clip(self.cross / denominator, -1, 1)
                pearson_r = np.where((self.n >= 2) & (denominator > 0) & (value_range > 0) &
                                     (self.pred_maximum > self.pred_minimum), pearson_r, np.nan)
                rmse = np.where(self.n > 0, rmse, np.nan)
                rmspe = np.where((self.n >= 2) & (value_range > 0), rmse / value_range, np.nan)
            return dict(pearson_r=pearson_r, rmse=rmse, rmspe=rmspe,
                        valid_count=self.n, truth_range=np.where(self.n > 0, value_range, np.nan))

    def save_metrics_nc(dataset, path):
        path = Path(path)
        temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.tmp.nc')
        try:
            import netCDF4
            engine = 'netcdf4'
        except ImportError:
            try:
                import h5netcdf
                engine = 'h5netcdf'
            except ImportError as exc:
                raise ImportError('保存指标 NC 需要 netCDF4 或 h5netcdf，请在当前内核安装其中一个。') from exc
        encoding = {name: {'zlib': True, 'complevel': 4} for name in dataset.data_vars}
        try:
            dataset.to_netcdf(temporary, engine=engine, encoding=encoding)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    from contextlib import ExitStack

    if CHUNK_SIZE < 1:
        raise ValueError('CHUNK_SIZE 必须 >= 1。')
    alignment_rows, truth_rows = [], []
    with ExitStack() as opened:
        arrays = {}
        reference = None
        common_times = None
        for run in runs:
            rid = run['run_id']
            arrays[rid] = {}
            for component in COMPONENTS:
                arrays[rid][component] = {}
                for kind in ['testy', 'predicty']:
                    name = f'{kind}_{component}'
                    ds = opened.enter_context(xr.open_dataset(run['resolved_paths'][name], cache=False))
                    array, rounded_count = canonical_array(ds, name)
                    if reference is None:
                        reference = array
                    require_spatial_coordinates(reference, array, f'{rid}/{name}')
                    arrays[rid][component][kind] = array
                    this_times = array.time.values.astype('datetime64[ns]')
                    common_times = this_times if common_times is None else np.intersect1d(common_times, this_times)
                    alignment_rows.append(dict(run_id=rid, variable=name, original_times=len(this_times),
                                               rounded_times=rounded_count))
        if common_times is None or len(common_times) < 2:
            raise ValueError('四组预测和真实值没有至少两个共同测试时次。')
        levels = np.asarray(reference.level.values)
        latitude = np.asarray(reference.latitude.values)
        longitude = np.asarray(reference.longitude.values)
        spatial_shape = (len(levels), len(latitude), len(longitude))
        full_shape = (len(runs), len(COMPONENTS)) + spatial_shape
        metric_values = {name: np.full(full_shape, np.nan) for name in ['pearson_r', 'rmse', 'rmspe', 'truth_range']}
        sample_counts = np.zeros(full_shape, dtype=np.int64)
        for row in alignment_rows:
            row.update(common_times=len(common_times), dropped_times=row['original_times'] - len(common_times),
                       first_common_time=str(common_times[0]), last_common_time=str(common_times[-1]))
        print('共同测试时次数：', len(common_times), '；', common_times[0], '至', common_times[-1])

        for ci, component in enumerate(COMPONENTS):
            print(f'正在计算 {component}：四组自己的 testy/predicty，chunk={CHUNK_SIZE}。')
            aligned = [{kind: arrays[run['run_id']][component][kind].sel(time=common_times)
                        for kind in ['testy', 'predicty']} for run in runs]
            accumulators = [GridAccumulator(spatial_shape) for run in runs]
            audits = [dict(run_id=run['run_id'], reference_run=RUN_IDS[0], component=component,
                           compared_values=0, different_values=0, different_times=0,
                           max_absolute_difference=0.0, own_nonfinite_values=0,
                           common_excluded_samples=0) for run in runs]
            for start in range(0, len(common_times), CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, len(common_times))
                truth = [np.asarray(pair['testy'].isel(time=slice(start, stop)).values, dtype=np.float64)
                         for pair in aligned]
                prediction = [np.asarray(pair['predicty'].isel(time=slice(start, stop)).values, dtype=np.float64)
                              for pair in aligned]
                valid = np.ones(truth[0].shape, dtype=bool)
                for i in range(len(runs)):
                    own_valid = np.isfinite(truth[i]) & np.isfinite(prediction[i])
                    audits[i]['own_nonfinite_values'] += int(own_valid.size - own_valid.sum())
                    valid &= own_valid
                excluded = int(valid.size - valid.sum())
                for i, accumulator in enumerate(accumulators):
                    accumulator.update(prediction[i], truth[i], valid)
                    audits[i]['common_excluded_samples'] += excluded
                    compare = np.isfinite(truth[i]) & np.isfinite(truth[0])
                    difference = np.zeros(truth[i].shape, dtype=np.float64)
                    np.subtract(truth[i], truth[0], out=difference, where=compare)
                    difference = np.abs(difference)
                    different = compare & (difference > TRUTH_ATOL + TRUTH_RTOL * np.abs(truth[0]))
                    audits[i]['compared_values'] += int(compare.sum())
                    audits[i]['different_values'] += int(different.sum())
                    audits[i]['different_times'] += int(different.reshape(stop-start, -1).any(axis=1).sum())
                    audits[i]['max_absolute_difference'] = max(audits[i]['max_absolute_difference'], float(difference.max()))
                del truth, prediction, valid
            for ri, accumulator in enumerate(accumulators):
                result = accumulator.finish()
                for name in metric_values:
                    metric_values[name][ri, ci] = result[name]
                sample_counts[ri, ci] = result['valid_count']
            truth_rows.extend(audits)

    dims = ('run', 'component', 'level', 'latitude', 'longitude')
    metrics_ds = xr.Dataset(
        {**{name: (dims, values) for name, values in metric_values.items()},
         'valid_count': (dims, sample_counts)},
        coords={'run': RUN_IDS, 'component': COMPONENTS, 'level': levels,
                'latitude': latitude, 'longitude': longitude,
                'seed': ('run', [int(run['seed']) for run in runs]),
                'experiment': ('run', [run['experiment'] for run in runs])}
    )
    metric_masks = {}
    for name in ['pearson_r', 'rmse', 'rmspe']:
        metric_masks[name] = np.isfinite(metric_values[name]).all(axis=0)
        metrics_ds[f'common_valid_mask_{name}'] = (
            ('component', 'level', 'latitude', 'longitude'), metric_masks[name].astype(np.int8))
    metrics_ds.attrs.update(
        description='01 paired-seed test metrics; each prediction is paired only with its own truth.',
        aggregation='Temporal grid-point metrics; spatial/layer summaries are unweighted means on metric-specific common valid masks.',
        finite_samples='At each component, level and grid cell, use times with finite truth and prediction in all four runs.',
        rmspe_definition='RMSE / (maximum own test truth - minimum own test truth), on the evaluated common finite times; ratio, not percent.',
        w_representation='Saved w = -0.1 * ERA5 omega. No additional inverse scaling is applied here.',
        component_rmse_units='u and v: m s-1; w: units of the saved scaled pressure-velocity variable.',
        common_time_count=int(len(common_times)), common_time_start=str(common_times[0]),
        common_time_end=str(common_times[-1]),
        time_round_tolerance_ns=int(TIME_ROUND_TOLERANCE_NS),
        dataset_fingerprint=next(iter(fingerprints)),
        undefined_metrics='Undefined Pearson R or RMSPE are NaN, not replaced by zero.',
    )
    metrics_ds['pearson_r'].attrs.update(long_name='Temporal Pearson correlation coefficient', units='1')
    metrics_ds['rmse'].attrs.update(long_name='Temporal root mean squared error', units='component dependent; see global attributes')
    metrics_ds['rmspe'].attrs.update(long_name='Range-normalized RMSE', units='1')
    metrics_ds['level'].attrs['units'] = 'hPa'
    metrics_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_grid_metrics.nc'
    save_metrics_nc(metrics_ds, metrics_path)

    summary_rows = []
    for ri, run in enumerate(runs):
        for ci, component in enumerate(COMPONENTS):
            for scope, li in [('all_levels', None)] + [('level', i) for i in range(len(levels))]:
                row = dict(run_id=run['run_id'], seed=int(run['seed']), experiment=run['experiment'],
                           component=component, scope=scope, level_hpa=np.nan if li is None else float(levels[li]))
                for name in ['pearson_r', 'rmse', 'rmspe']:
                    values = metric_values[name][ri, ci]
                    mask = metric_masks[name][ci]
                    if li is not None:
                        values, mask = values[li], mask[li]
                    row[name] = float(values[mask].mean()) if mask.any() else np.nan
                    row[f'{name}_grid_count'] = int(mask.sum())
                values = metric_values['rmse'][ri, ci]
                mask = metric_masks['rmse'][ci]
                counts = sample_counts[ri, ci]
                if li is not None:
                    values, mask, counts = values[li], mask[li], counts[li]
                count = counts[mask].sum()
                row['pooled_rmse'] = float(np.sqrt(np.sum(values[mask]**2 * counts[mask]) / count)) if count else np.nan
                row['common_time_count'] = int(len(common_times))
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_summary.csv', index=False, encoding='utf-8-sig')
    alignment = pd.DataFrame(alignment_rows)
    truth_audit = pd.DataFrame(truth_rows)
    alignment.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_time_alignment.csv', index=False, encoding='utf-8-sig')
    truth_audit.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_truth_audit.csv', index=False, encoding='utf-8-sig')
    if alignment.dropped_times.max() > 0:
        print('部分时次已通过共同时间交集排除，详见 time_alignment.csv。')
    if truth_audit.different_values.sum() > 0:
        print('存在跨实验真实值差异，详见 truth_audit.csv；每组指标始终使用该组自己的真实值。')
    print('逐格点指标已保存：', metrics_path)
    display(summary.loc[summary.scope == 'all_levels',
                        ['seed', 'experiment', 'component', 'pearson_r', 'rmse', 'rmspe']])

    summary = pd.read_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_summary.csv')
    paired_rows = []
    scope_levels = [('all_levels', None)] + [('level', v) for v in sorted(summary.level_hpa.dropna().unique())]
    for seed in EXPECTED_SEEDS:
        for component in COMPONENTS:
            for scope, level in scope_levels:
                subset = summary[(summary.seed == seed) & (summary.component == component) & (summary.scope == scope)]
                if level is not None:
                    subset = subset[subset.level_hpa == level]
                if len(subset) != 2 or set(subset.experiment) != {'no_pinn', 'no_shear'}:
                    raise ValueError(f'{seed}/{component}/{scope}/{level} 未形成完整配对。')
                baseline = subset[subset.experiment == 'no_pinn'].iloc[0]
                pinn = subset[subset.experiment == 'no_shear'].iloc[0]
                row = dict(seed=seed, component=component, scope=scope,
                           level_hpa=np.nan if level is None else level)
                for name in ['pearson_r', 'rmse', 'rmspe']:
                    base, physical = float(baseline[name]), float(pinn[name])
                    row[f'no_pinn_{name}'] = base
                    row[f'no_shear_{name}'] = physical
                    row[f'delta_{name}'] = physical - base
                    row[f'gain_{name}'] = physical - base if name == 'pearson_r' else base - physical
                    if name != 'pearson_r':
                        row[f'{name}_reduction_percent'] = 100 * (base - physical) / base if base > 0 else np.nan
                paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_paired_changes.csv', index=False, encoding='utf-8-sig')

    replication_rows = []
    for component in COMPONENTS:
        for scope, level in scope_levels:
            subset = paired[(paired.component == component) & (paired.scope == scope)]
            if level is not None:
                subset = subset[subset.level_hpa == level]
            for name in ['pearson_r', 'rmse', 'rmspe']:
                values = subset[f'gain_{name}'].to_numpy(dtype=float)
                finite = np.isfinite(values)
                tolerance = 1e-12 * np.maximum(1., np.maximum(np.abs(subset[f'no_pinn_{name}'].to_numpy()),
                                                             np.abs(subset[f'no_shear_{name}'].to_numpy())))
                improved = int(np.sum(finite & (values > tolerance)))
                worsened = int(np.sum(finite & (values < -tolerance)))
                n = int(finite.sum())
                direction = ('undefined' if n == 0 else 'incomplete' if n < 2 else
                             'both_improved' if improved == 2 else 'both_worsened' if worsened == 2 else
                             'both_unchanged' if improved + worsened == 0 else 'mixed_or_one_unchanged')
                replication_rows.append(dict(component=component, scope=scope,
                    level_hpa=np.nan if level is None else level, metric=name,
                    valid_seed_count=n, improved_seed_count=improved, worsened_seed_count=worsened,
                    mean_gain=float(values[finite].mean()) if n else np.nan,
                    minimum_gain=float(values[finite].min()) if n else np.nan,
                    maximum_gain=float(values[finite].max()) if n else np.nan,
                    direction=direction))
    replication = pd.DataFrame(replication_rows)
    replication.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_seed_consistency.csv', index=False, encoding='utf-8-sig')
    print('配对差值 delta = w/o shear - No PINN；R 的 delta > 0 更好，RMSE/RMSPE 的 delta < 0 更好。')
    display(paired.loc[paired.scope == 'all_levels', ['seed', 'component', 'delta_pearson_r',
                      'delta_rmse', 'rmse_reduction_percent', 'delta_rmspe', 'rmspe_reduction_percent']])
    print('下表 gain > 0 均表示改善；两个种子的方向一致性仅作描述，不是显著性检验。')
    display(replication.loc[replication.scope == 'all_levels', ['component', 'metric', 'mean_gain',
                            'improved_seed_count', 'worsened_seed_count', 'direction']])


if __name__ == "__main__":
    main()

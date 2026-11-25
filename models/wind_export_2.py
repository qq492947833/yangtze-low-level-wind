def create_nc(
    times, lat, lon, levels, v,
    timename, latname, lonname, v_name,
    iflevel, levelname, iftime,
    iflat, iflon, newfilename,
    time0_ref_date=None
):
    import os
    import gc
    import numpy as np
    import netCDF4
    from datetime import datetime

    gc.collect()

    times_np = np.asarray(times, dtype='datetime64[ns]')
    lat_np = np.asarray(lat, dtype=np.float32).reshape(-1)
    lon_np = np.asarray(lon, dtype=np.float32).reshape(-1)
    levels_np = np.asarray(levels, dtype=np.float32).reshape(-1)
    value = np.asarray(v)

    if iftime != 'no':
        if times_np.ndim != 1 or times_np.size == 0:
            raise ValueError('times 必须是一维且不能为空。')
        if np.isnat(times_np).any():
            raise ValueError('times 中包含 NaT。')

    expected_shape = []

    if iftime != 'no':
        expected_shape.append(times_np.size)
    if iflevel != 'no':
        expected_shape.append(levels_np.size)
    if iflat != 'no':
        expected_shape.append(lat_np.size)
    if iflon != 'no':
        expected_shape.append(lon_np.size)

    expected_shape = tuple(expected_shape)

    if value.shape != expected_shape:
        raise ValueError(
            f'{v_name} 的形状为 {value.shape}，'
            f'但预期形状为 {expected_shape}。'
        )

    output_path = os.fspath(newfilename)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except PermissionError as exc:
            raise PermissionError(
                f'无法覆盖文件：{output_path}。'
                '请关闭正在读取该 NC 文件的软件。'
            ) from exc

    try:
        with netCDF4.Dataset(
            output_path,
            'w',
            format='NETCDF4'
        ) as newnc:

            if iftime != 'no':
                newnc.createDimension(timename, times_np.size)

            if iflat != 'no':
                newnc.createDimension(latname, lat_np.size)

            if iflon != 'no':
                newnc.createDimension(lonname, lon_np.size)

            if iflevel != 'no':
                newnc.createDimension(levelname, levels_np.size)

            if iflat != 'no':
                lat_var = newnc.createVariable(
                    latname,
                    np.float32,
                    (latname,)
                )
                lat_var[:] = lat_np
                lat_var.units = 'degrees_north'
                lat_var.standard_name = 'latitude'
                lat_var.long_name = 'latitude'

            if iflon != 'no':
                lon_var = newnc.createVariable(
                    lonname,
                    np.float32,
                    (lonname,)
                )
                lon_var[:] = lon_np
                lon_var.units = 'degrees_east'
                lon_var.standard_name = 'longitude'
                lon_var.long_name = 'longitude'

            if iflevel != 'no':
                level_var = newnc.createVariable(
                    levelname,
                    np.float32,
                    (levelname,)
                )
                level_var[:] = levels_np

            if iftime != 'no':
                time_var = newnc.createVariable(
                    timename,
                    np.float64,
                    (timename,)
                )

                if time0_ref_date is None:
                    ref_seconds = (
                        int(times_np[0].astype('int64')) / 1e9
                    )
                    time0_ref = datetime.utcfromtimestamp(
                        ref_seconds
                    )
                elif isinstance(time0_ref_date, str):
                    time0_ref = datetime.fromisoformat(
                        time0_ref_date.replace(' ', 'T')
                    )
                else:
                    time0_ref = time0_ref_date

                units_str = (
                    f'days since '
                    f'{time0_ref.strftime("%Y-%m-%d %H:%M:%S")}'
                )

                datetime_objects = [
                    datetime.utcfromtimestamp(
                        int(t.astype('int64')) / 1e9
                    )
                    for t in times_np
                ]

                time_var[:] = netCDF4.date2num(
                    datetime_objects,
                    units=units_str,
                    calendar='standard'
                )

                time_var.units = units_str
                time_var.calendar = 'standard'

            v_dims_ordered = []

            if iftime != 'no':
                v_dims_ordered.append(timename)

            if iflevel != 'no':
                v_dims_ordered.append(levelname)

            if iflat != 'no':
                v_dims_ordered.append(latname)

            if iflon != 'no':
                v_dims_ordered.append(lonname)

            variable_kwargs = {}

            if iftime != 'no' and len(v_dims_ordered) == 4:
                variable_kwargs['chunksizes'] = (
                    min(256, times_np.size),
                    levels_np.size,
                    lat_np.size,
                    lon_np.size
                )

            vs = newnc.createVariable(
                v_name,
                np.float32,
                tuple(v_dims_ordered),
                **variable_kwargs
            )

            float32_max = np.finfo(np.float32).max

            if iftime != 'no':

                for start in range(0, value.shape[0], 256):

                    end = min(start + 256, value.shape[0])

                    chunk = np.asarray(
                        value[start:end],
                        dtype=np.float64
                    )

                    invalid = (
                        (~np.isfinite(chunk))
                        | (np.abs(chunk) > float32_max)
                    )

                    if np.any(invalid):
                        print(
                            f'警告：{v_name} 的第 '
                            f'{start}:{end} 块中有 '
                            f'{int(np.count_nonzero(invalid))} '
                            '个非有限或超出 float32 范围的值，已置为 0。'
                        )
                        chunk[invalid] = 0.0

                    chunk = np.ascontiguousarray(
                        chunk,
                        dtype=np.float32
                    )

                    vs[start:end] = chunk

            else:

                chunk = np.asarray(
                    value,
                    dtype=np.float64
                )

                invalid = (
                    (~np.isfinite(chunk))
                    | (np.abs(chunk) > float32_max)
                )

                if np.any(invalid):
                    chunk[invalid] = 0.0

                vs[:] = np.ascontiguousarray(
                    chunk,
                    dtype=np.float32
                )

            newnc.sync()

    except Exception:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        raise

from figure_paths import DATA, RESULTS, WORK, FIGURES, setting

import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


DATA_DIR = DATA
RESULTS.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 24.0, 36.0
LON_MIN, LON_MAX = 108.0, 126.0

START_TIME = "2001-01-01 00:00:00"
END_TIME = "2023-12-31 23:59:59"

LEVELS = [850, 925, 1000]

CHUNK_HOURS = 720

FILES = {
    ("u", 850): os.path.join(
        DATA_DIR,
        "U-component-of-wind-850hpa.nc"
    ),
    ("u", 925): os.path.join(
        DATA_DIR,
        "U-component-of-wind-925hpa.nc"
    ),
    ("u", 1000): os.path.join(
        DATA_DIR,
        "U-component-of-wind-1000hpa.nc"
    ),

    ("v", 850): os.path.join(
        DATA_DIR,
        "V-component-of-wind-850hpa.nc"
    ),
    ("v", 925): os.path.join(
        DATA_DIR,
        "V-component-of-wind-925hpa.nc"
    ),
    ("v", 1000): os.path.join(
        DATA_DIR,
        "V-component-of-wind-1000hpa.nc"
    ),
}

OUT_CSV = os.path.join(
    RESULTS,
    "Yangtze_low_level_wind_speed_statistics_2001_2023.csv"
)

OUT_CLIM_NC = os.path.join(
    RESULTS,
    "Yangtze_low_level_wind_speed_climatology_2001_2023.nc"
)


BIN_MIN = 0.0
BIN_MAX = 200.0
BIN_STEP = 0.01

BIN_EDGES = np.arange(
    BIN_MIN,
    BIN_MAX + BIN_STEP,
    BIN_STEP,
    dtype=np.float64,
)


def find_coord_name(ds, candidates):
    """自动识别时间、纬度或经度坐标名称。"""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name

    raise ValueError(
        f"无法从以下候选名称中识别坐标：{candidates}\n"
        f"dims={list(ds.dims)}\n"
        f"coords={list(ds.coords)}"
    )


def find_main_variable(ds, time_name, lat_name, lon_name):
    """自动识别包含time、lat、lon维度的主要变量。"""
    for var_name in ds.data_vars:
        dims = set(ds[var_name].dims)

        if (
            time_name in dims
            and lat_name in dims
            and lon_name in dims
        ):
            return var_name

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(
        "无法自动识别主要变量。\n"
        f"当前数据变量：{list(ds.data_vars)}"
    )


def normalize_longitude(ds, lon_name):
    """若经度为0–360，则转换为−180–180。"""
    lon = ds[lon_name]

    if float(lon.max()) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_name: new_lon})
        ds = ds.sortby(lon_name)

    return ds


def open_selected_dataarray(path, level_hpa):
    """
    打开单个NetCDF文件，自动识别变量和坐标，
    并截取研究时段及研究区域。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    ds = xr.open_dataset(
        path,
        cache=False,
        decode_times=True,
    )

    time_name = find_coord_name(
        ds,
        ["time", "valid_time", "datetime"]
    )

    lat_name = find_coord_name(
        ds,
        ["latitude", "lat"]
    )

    lon_name = find_coord_name(
        ds,
        ["longitude", "lon"]
    )

    ds = normalize_longitude(ds, lon_name)

    var_name = find_main_variable(
        ds,
        time_name,
        lat_name,
        lon_name,
    )

    da = ds[var_name]

    for dim_name in list(da.dims):
        if dim_name in [time_name, lat_name, lon_name]:
            continue

        dim_lower = dim_name.lower()

        if dim_lower in [
            "pressure_level",
            "level",
            "isobaricinhpa",
            "isobaric",
        ]:
            da = da.sel(
                {dim_name: level_hpa},
                method="nearest",
            )

        elif da.sizes[dim_name] == 1:
            da = da.isel(
                {dim_name: 0},
                drop=True,
            )

        elif dim_lower == "expver":
            da = da.mean(
                dim=dim_name,
                skipna=True,
            )

        else:
            ds.close()

            raise ValueError(
                f"存在未处理的额外维度：{dim_name}，"
                f"长度={da.sizes[dim_name]}\n"
                f"文件：{path}"
            )

    rename_dict = {}

    if time_name != "time":
        rename_dict[time_name] = "time"

    if lat_name != "lat":
        rename_dict[lat_name] = "lat"

    if lon_name != "lon":
        rename_dict[lon_name] = "lon"

    if rename_dict:
        da = da.rename(rename_dict)

    lat_values = np.asarray(da["lat"].values)

    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(LAT_MAX, LAT_MIN)
    else:
        lat_slice = slice(LAT_MIN, LAT_MAX)

    da = da.sel(
        time=slice(START_TIME, END_TIME),
        lat=lat_slice,
        lon=slice(LON_MIN, LON_MAX),
    )

    da = da.transpose("time", "lat", "lon")

    if da.sizes["time"] == 0:
        ds.close()
        raise ValueError(
            f"时间选择后没有数据：{path}"
        )

    if da.sizes["lat"] == 0 or da.sizes["lon"] == 0:
        ds.close()
        raise ValueError(
            f"空间裁剪后没有数据：{path}"
        )

    print("\n已打开：", path)
    print("变量名称：", var_name)
    print("选择后形状：", da.shape)

    print(
        "时间范围：",
        str(da["time"].values[0]),
        "至",
        str(da["time"].values[-1]),
    )

    print(
        "纬度范围：",
        float(da["lat"].min()),
        "至",
        float(da["lat"].max()),
    )

    print(
        "经度范围：",
        float(da["lon"].min()),
        "至",
        float(da["lon"].max()),
    )

    return ds, da


def empty_stats():
    """初始化面积加权统计量。"""
    return {
        "weighted_sum": 0.0,
        "weight_sum": 0.0,
        "hist": np.zeros(
            len(BIN_EDGES) - 1,
            dtype=np.float64,
        ),
        "valid_number": 0,
        "observed_max": -np.inf,
        "overflow_number": 0,
    }


def update_stats(speed, area_weights_2d, stats):
    """
    更新面积加权均值和面积加权第99百分位数所需统计量。

    参数
    ----
    speed:
        shape=(time, lat, lon)

    area_weights_2d:
        shape=(lat, lon)
    """
    valid = np.isfinite(speed)

    if not np.any(valid):
        return

    weights_3d = np.broadcast_to(
        area_weights_2d[None, :, :],
        speed.shape,
    )

    values = speed[valid].astype(
        np.float64,
        copy=False,
    )

    weights = weights_3d[valid].astype(
        np.float64,
        copy=False,
    )

    stats["weighted_sum"] += np.sum(
        values * weights
    )

    stats["weight_sum"] += np.sum(weights)

    stats["valid_number"] += int(values.size)

    this_max = float(np.nanmax(values))

    if this_max > stats["observed_max"]:
        stats["observed_max"] = this_max

    overflow = values >= BIN_MAX
    stats["overflow_number"] += int(
        np.sum(overflow)
    )

    values_for_hist = np.clip(
        values,
        BIN_MIN,
        BIN_MAX - 1.0e-10,
    )

    hist, _ = np.histogram(
        values_for_hist,
        bins=BIN_EDGES,
        weights=weights,
    )

    stats["hist"] += hist


def weighted_percentile_from_hist(
    hist,
    edges,
    percentile=99.0,
):
    """
    根据面积加权直方图计算百分位数。
    """
    hist = np.asarray(
        hist,
        dtype=np.float64,
    )

    cumulative_weight = np.cumsum(hist)

    if (
        cumulative_weight.size == 0
        or cumulative_weight[-1] <= 0
    ):
        return np.nan

    target_weight = (
        percentile / 100.0
        * cumulative_weight[-1]
    )

    index = int(
        np.searchsorted(
            cumulative_weight,
            target_weight,
            side="left",
        )
    )

    if index <= 0:
        previous_weight = 0.0
    else:
        previous_weight = cumulative_weight[index - 1]

    if index >= len(hist):
        return float(edges[-1])

    current_bin_weight = hist[index]

    if current_bin_weight <= 0:
        return float(edges[index])

    fraction = (
        target_weight - previous_weight
    ) / current_bin_weight

    percentile_value = (
        edges[index]
        + fraction
        * (edges[index + 1] - edges[index])
    )

    return float(percentile_value)


def finalize_stats(stats):
    """完成面积加权均值和第99百分位数计算。"""
    if stats["weight_sum"] <= 0:
        mean_speed = np.nan
    else:
        mean_speed = (
            stats["weighted_sum"]
            / stats["weight_sum"]
        )

    p99_speed = weighted_percentile_from_hist(
        hist=stats["hist"],
        edges=BIN_EDGES,
        percentile=99.0,
    )

    return {
        "area_weighted_mean_wind_speed_m_s": mean_speed,
        "area_weighted_p99_wind_speed_m_s": p99_speed,
        "valid_value_number": stats["valid_number"],
        "observed_max_for_check_m_s": stats["observed_max"],
        "histogram_overflow_number": stats["overflow_number"],
    }


datasets = []
wind_arrays = {}

try:
    for level in LEVELS:
        for component in ["u", "v"]:

            ds, da = open_selected_dataarray(
                path=FILES[(component, level)],
                level_hpa=level,
            )

            datasets.append(ds)
            wind_arrays[(component, level)] = da

    keys = list(wind_arrays.keys())

    aligned_arrays = xr.align(
        *[wind_arrays[key] for key in keys],
        join="inner",
        copy=False,
    )

    wind_arrays = {
        key: da
        for key, da in zip(keys, aligned_arrays)
    }

    reference_da = wind_arrays[("u", 850)]

    time_values = reference_da["time"].values

    lat_values = np.asarray(
        reference_da["lat"].values,
        dtype=np.float64,
    )

    lon_values = np.asarray(
        reference_da["lon"].values,
        dtype=np.float64,
    )

    n_time = reference_da.sizes["time"]
    n_lat = reference_da.sizes["lat"]
    n_lon = reference_da.sizes["lon"]

    print("\n" + "=" * 70)
    print("六个U/V数据对齐完成")
    print("=" * 70)

    print("共同时间长度：", n_time)
    print("纬度格点数：", n_lat)
    print("经度格点数：", n_lon)

    print(
        "共同时间范围：",
        str(time_values[0]),
        "至",
        str(time_values[-1]),
    )

    lat_weights = np.cos(
        np.deg2rad(lat_values)
    ).astype(np.float64)

    area_weights_2d = np.broadcast_to(
        lat_weights[:, None],
        (n_lat, n_lon),
    ).copy()

    level_stats = {
        level: empty_stats()
        for level in LEVELS
    }

    vertical_mean_stats = empty_stats()

    spatial_sum = np.zeros(
        (n_lat, n_lon),
        dtype=np.float64,
    )

    spatial_square_sum = np.zeros(
        (n_lat, n_lon),
        dtype=np.float64,
    )

    spatial_count = np.zeros(
        (n_lat, n_lon),
        dtype=np.int64,
    )


    for start_index in tqdm(
        range(0, n_time, CHUNK_HOURS),
        desc="计算长江流域低层风速",
    ):
        end_index = min(
            start_index + CHUNK_HOURS,
            n_time,
        )

        speed_each_level = []

        for level in LEVELS:

            u_chunk = wind_arrays[
                ("u", level)
            ].isel(
                time=slice(
                    start_index,
                    end_index,
                )
            ).values.astype(
                np.float32,
                copy=False,
            )

            v_chunk = wind_arrays[
                ("v", level)
            ].isel(
                time=slice(
                    start_index,
                    end_index,
                )
            ).values.astype(
                np.float32,
                copy=False,
            )

            speed = np.hypot(
                u_chunk,
                v_chunk,
            ).astype(
                np.float32,
                copy=False,
            )

            update_stats(
                speed=speed,
                area_weights_2d=area_weights_2d,
                stats=level_stats[level],
            )

            speed_each_level.append(speed)

            del u_chunk
            del v_chunk

        speed_stack = np.stack(
            speed_each_level,
            axis=0,
        )

        vertical_mean_speed = np.nanmean(
            speed_stack,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        update_stats(
            speed=vertical_mean_speed,
            area_weights_2d=area_weights_2d,
            stats=vertical_mean_stats,
        )

        valid = np.isfinite(
            vertical_mean_speed
        )

        speed64 = vertical_mean_speed.astype(
            np.float64,
            copy=False,
        )

        spatial_sum += np.sum(
            np.where(
                valid,
                speed64,
                0.0,
            ),
            axis=0,
        )

        spatial_square_sum += np.sum(
            np.where(
                valid,
                speed64 ** 2,
                0.0,
            ),
            axis=0,
        )

        spatial_count += np.sum(
            valid,
            axis=0,
        )

        del speed_each_level
        del speed_stack
        del vertical_mean_speed
        del speed64
        del valid


    result_rows = []

    for level in LEVELS:

        result = finalize_stats(
            level_stats[level]
        )

        result_rows.append({
            "wind_field": f"{level} hPa",
            **result,
        })

    vertical_result = finalize_stats(
        vertical_mean_stats
    )

    result_rows.append({
        "wind_field": (
            "850-1000 hPa vertically averaged"
        ),
        **vertical_result,
    })

    result_df = pd.DataFrame(result_rows)


    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        spatial_mean = np.divide(
            spatial_sum,
            spatial_count,
            out=np.full(
                spatial_sum.shape,
                np.nan,
                dtype=np.float64,
            ),
            where=spatial_count > 0,
        )

        spatial_mean_square = np.divide(
            spatial_square_sum,
            spatial_count,
            out=np.full(
                spatial_square_sum.shape,
                np.nan,
                dtype=np.float64,
            ),
            where=spatial_count > 0,
        )

        spatial_variance = (
            spatial_mean_square
            - spatial_mean ** 2
        )

        spatial_variance = np.maximum(
            spatial_variance,
            0.0,
        )

        spatial_std = np.sqrt(
            spatial_variance
        )


    if (
        vertical_result[
            "histogram_overflow_number"
        ] > 0
    ):
        print(
            "\nWarning：存在风速超过 "
            f"{BIN_MAX} m/s 的值。"
        )
        print(
            "请增大BIN_MAX后重新运行，"
            "否则第99百分位数可能受到影响。"
        )


    result_df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    output_ds = xr.Dataset(
        data_vars={
            "vertically_averaged_wind_speed_mean": (
                ("lat", "lon"),
                spatial_mean.astype(np.float32),
            ),

            "vertically_averaged_wind_speed_std": (
                ("lat", "lon"),
                spatial_std.astype(np.float32),
            ),

            "valid_time_count": (
                ("lat", "lon"),
                spatial_count.astype(np.int32),
            ),
        },

        coords={
            "lat": lat_values.astype(np.float32),
            "lon": lon_values.astype(np.float32),
        },

        attrs={
            "description": (
                "Temporal mean and standard deviation "
                "of vertically averaged horizontal wind "
                "speed at 850, 925, and 1000 hPa."
            ),

            "wind_speed_definition": (
                "sqrt(u^2+v^2) calculated separately "
                "at each pressure level, followed by "
                "arithmetic averaging over 850, 925, "
                "and 1000 hPa."
            ),

            "study_area": (
                "24.0-36.0 degrees North, "
                "108.0-126.0 degrees East"
            ),

            "start_time": str(time_values[0]),
            "end_time": str(time_values[-1]),

            "standard_deviation_ddof": 0,

            "area_weighted_mean_wind_speed_m_s": (
                vertical_result[
                    "area_weighted_mean_wind_speed_m_s"
                ]
            ),

            "area_weighted_p99_wind_speed_m_s": (
                vertical_result[
                    "area_weighted_p99_wind_speed_m_s"
                ]
            ),

            "p99_histogram_bin_width_m_s": BIN_STEP,
        },
    )

    output_ds[
        "vertically_averaged_wind_speed_mean"
    ].attrs = {
        "long_name": (
            "Temporal mean of vertically averaged "
            "low-level horizontal wind speed"
        ),
        "units": "m s-1",
    }

    output_ds[
        "vertically_averaged_wind_speed_std"
    ].attrs = {
        "long_name": (
            "Temporal standard deviation of vertically "
            "averaged low-level horizontal wind speed"
        ),
        "units": "m s-1",
    }

    output_ds[
        "valid_time_count"
    ].attrs = {
        "long_name": (
            "Number of valid hourly wind-speed values"
        ),
        "units": "1",
    }

    encoding = {
        "vertically_averaged_wind_speed_mean": {
            "zlib": True,
            "complevel": 9,
            "dtype": "float32",
        },

        "vertically_averaged_wind_speed_std": {
            "zlib": True,
            "complevel": 9,
            "dtype": "float32",
        },

        "valid_time_count": {
            "zlib": True,
            "complevel": 9,
            "dtype": "int32",
        },
    }

    output_ds.to_netcdf(
        OUT_CLIM_NC,
        mode="w",
        format="NETCDF4",
        encoding=encoding,
    )

    output_ds.close()

finally:
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass

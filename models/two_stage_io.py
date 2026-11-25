import gc
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


def rounded_hour_times(values, tolerance_ns=1_000_000):
    """Return unique, increasing datetime64[ns] hours, tolerating tiny NC drift."""
    if (
        isinstance(tolerance_ns, (bool, np.bool_))
        or not isinstance(tolerance_ns, (int, np.integer))
        or not 0 <= tolerance_ns < 1_800_000_000_000
    ):
        raise ValueError("tolerance_ns must be an integer from 0 to half an hour")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("time must be a nonempty one-dimensional coordinate")
    if raw.dtype.kind in "biufc":
        raise TypeError("time must contain datetimes, not undecoded numeric values")
    try:
        times = raw.astype("datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("time cannot be decoded as datetime64[ns]") from exc
    if np.isnat(times).any():
        raise ValueError("time contains NaT")
    nanos = times.view(np.int64)
    hour_ns = 3_600_000_000_000
    remainder = np.remainder(nanos, hour_ns)
    offsets = np.where(remainder <= hour_ns // 2, -remainder, hour_ns - remainder)
    if np.any(np.abs(offsets) > tolerance_ns):
        raise ValueError("time differs from its nearest hour by more than tolerance_ns")
    rounded = nanos + offsets
    if np.any(((offsets > 0) & (rounded < nanos)) | ((offsets < 0) & (rounded > nanos))):
        raise ValueError("rounded time exceeds the datetime64[ns] range")
    if np.any(rounded[1:] <= rounded[:-1]):
        raise ValueError("rounded time must be strictly increasing and unique")
    return rounded.view("datetime64[ns]")


def _wind_coordinate(values, name, length):
    """Validate a finite, real, one-dimensional spatial coordinate."""
    coordinate = np.asarray(values)
    if coordinate.ndim != 1 or coordinate.size != length:
        raise ValueError(f"{name} must have shape ({length},)")
    if coordinate.dtype.kind not in "iuf" or not np.isfinite(coordinate).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if np.any(np.abs(coordinate.astype(np.float64)) > np.finfo(np.float32).max):
        raise ValueError(f"{name} exceeds the float32 range")
    result = coordinate.astype(np.float32)
    if np.unique(result).size != length:
        raise ValueError(f"{name} must remain unique after float32 conversion")
    return result


def _wind_nc_attributes(attrs):
    """Normalize scalar metadata to the string/numeric types accepted by NetCDF."""
    result = {}
    for key, value in (attrs or {}).items():
        if not isinstance(key, str) or not key:
            raise TypeError("NetCDF attribute names must be nonempty strings")
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        elif isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if not isinstance(value, (str, int, float)):
            raise TypeError(f"NetCDF attribute {key!r} must be a scalar string or number")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"NetCDF attribute {key!r} must be finite")
        result[key] = value
    return result


def write_wind_nc(
    path, variable, values, times, lat, lon, levels,
    mean=None, std=None, attrs=None, chunk_size=128,
):
    """Atomically write N,L,H,W winds, inverse-normalizing one time chunk at a time.

    ``mean`` and ``std`` must both be omitted or both have shape L,H,W. Metadata
    in ``attrs`` is saved globally; standard field metadata is also attached to
    the wind variable. Supply w units and its saved -0.1*omega convention there.
    No NaN, infinity, masked cell, or float32 overflow is replaced or filled.
    """
    import netCDF4

    if not isinstance(variable, str) or not variable or variable in {
        "time", "level", "latitude", "longitude",
    }:
        raise ValueError("variable must be a nonempty field name, distinct from coordinates")
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    data = np.asanyarray(values)
    if data.ndim != 4 or any(size == 0 for size in data.shape):
        raise ValueError("values must have nonempty shape (time, level, latitude, longitude)")
    if data.dtype.kind not in "iuf":
        raise TypeError("values must have a real numeric dtype")
    count, depth, height, width = data.shape
    time_values = rounded_hour_times(times)
    if time_values.size != count:
        raise ValueError("time length does not match values.shape[0]")
    lat_values = _wind_coordinate(lat, "latitude", height)
    lon_values = _wind_coordinate(lon, "longitude", width)
    level_values = _wind_coordinate(levels, "level", depth)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    if mean is not None:
        mean_values = np.asarray(mean, dtype=np.float64)
        std_values = np.asarray(std, dtype=np.float64)
        if mean_values.shape != data.shape[1:] or std_values.shape != data.shape[1:]:
            raise ValueError("mean and std must both have shape (level, latitude, longitude)")
        if not np.isfinite(mean_values).all() or not np.isfinite(std_values).all():
            raise ValueError("mean and std must be finite")
        if np.any(std_values <= 0):
            raise ValueError("std must be strictly positive")
    metadata = _wind_nc_attributes(attrs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
    )
    os.close(descriptor)
    try:
        with netCDF4.Dataset(temporary_name, "w", format="NETCDF4") as dataset:
            dimensions = ("time", "level", "latitude", "longitude")
            for name, size in zip(dimensions, data.shape):
                dataset.createDimension(name, size)
            dataset.setncatts({"Conventions": "CF-1.8", **metadata})
            time_var = dataset.createVariable("time", "i8", ("time",), fill_value=False)
            time_var.setncatts({
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "standard", "standard_name": "time", "axis": "T",
            })
            time_var[:] = time_values.astype("datetime64[s]").astype(np.int64)
            coordinate_specs = (
                ("level", level_values, {"units": "hPa", "long_name": "pressure level", "positive": "down", "axis": "Z"}),
                ("latitude", lat_values, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
                ("longitude", lon_values, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}),
            )
            for name, coordinate, coordinate_attrs in coordinate_specs:
                coordinate_var = dataset.createVariable(name, "f4", (name,), fill_value=False)
                coordinate_var.setncatts(coordinate_attrs)
                coordinate_var[:] = coordinate
            field = dataset.createVariable(
                variable, "f4", dimensions, zlib=True, complevel=1, shuffle=True,
                chunksizes=(min(int(chunk_size), count), 1, height, width),
                fill_value=False,
            )
            if variable.lower().split("_")[-1] in {"u", "v"}:
                field.setncattr("units", "m s-1")
            field_metadata_keys = {
                "units", "long_name", "standard_name", "comment", "vertical_velocity_note",
            }
            field.setncatts({key: value for key, value in metadata.items() if key in field_metadata_keys})
            limit = np.finfo(np.float32).max
            for start in range(0, count, int(chunk_size)):
                stop = min(start + int(chunk_size), count)
                source = data[start:stop]
                if np.ma.isMaskedArray(source) and np.ma.getmaskarray(source).any():
                    raise ValueError(f"masked wind data in time slice {start}:{stop}")
                chunk = np.array(source, dtype=np.float64, copy=True)
                if not np.isfinite(chunk).all():
                    raise ValueError(f"nonfinite wind data in time slice {start}:{stop}")
                if mean is not None:
                    with np.errstate(over="ignore", invalid="ignore"):
                        chunk *= std_values
                        chunk += mean_values
                if not np.isfinite(chunk).all() or np.any(np.abs(chunk) > limit):
                    raise ValueError(f"wind data exceeds finite float32 range in time slice {start}:{stop}")
                field[start:stop] = chunk.astype(np.float32)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output_path


def read_wind_array(ds, variable):
    """Return a lazy DataArray with canonical axes and exact hourly timestamps."""
    field = ds[variable]
    aliases = {old: new for old, new in (("lat", "latitude"), ("lon", "longitude")) if old in field.dims}
    if aliases:
        field = field.rename(aliases)
    dimensions = ("time", "level", "latitude", "longitude")
    if field.ndim != 4 or set(field.dims) != set(dimensions):
        raise ValueError(f"{variable} must have exactly the dimensions {dimensions}; found {field.dims}")
    field = field.transpose(*dimensions)
    if any(field.sizes[name] == 0 for name in dimensions):
        raise ValueError(f"{variable} has an empty dimension")
    if field.dtype.kind not in "iuf":
        raise TypeError(f"{variable} must have a real numeric dtype")
    for name in dimensions:
        if name not in field.coords or field.coords[name].dims != (name,):
            raise ValueError(f"{variable} requires a one-dimensional {name} coordinate")
        if field.coords[name].size != field.sizes[name]:
            raise ValueError(f"{variable} has a mismatched {name} coordinate")
    times = rounded_hour_times(field.coords["time"].values)
    for name in dimensions[1:]:
        _wind_coordinate(field.coords[name].values, name, field.sizes[name])
    return field.assign_coords(time=times)


def clear_model_memory(model=None):
    """Clear only this model's ordinary caches, then release unused CUDA memory.

    Registered parameters, buffers, and submodules are preserved, and the model
    is moved to CPU. The caller must remove its own model/optimizer references.
    """
    try:
        import torch
    except ImportError:
        if model is not None:
            raise
        gc.collect()
        return
    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        pending = [model]
        visited = set()
        while pending:
            current = pending.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            pending.extend(current._modules.values())
            pending = [module for module in pending if module is not None]
            for name, value in list(current.__dict__.items()):
                if name in {"_parameters", "_buffers", "_modules"}:
                    continue
                if name in {"self", "align_residual"} or isinstance(value, (torch.Tensor, np.ndarray)):
                    current.__dict__.pop(name, None)
            del current, value
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def checkpoint_sha256(path):
    """Compute a checkpoint provenance digest without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

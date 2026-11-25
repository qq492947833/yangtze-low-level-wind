def create_nc(times, lat, lon, levels, v, timename, latname, lonname, v_name, iflevel, levelname, iftime, iflat, iflon, newfilename, time0_ref_date=None):
    import netCDF4
    import numpy as np
    from datetime import datetime

    newnc = netCDF4.Dataset(newfilename, 'w', format='NETCDF4')
    if iftime != 'no':
        newnc.createDimension(timename, len(times))
    if iflat != 'no':
        newnc.createDimension(latname, len(lat))
    if iflon != 'no':
        newnc.createDimension(lonname, len(lon))
    if iflevel != 'no':
        newnc.createDimension(levelname, len(levels))
    if iflat != 'no':
        lat_var = newnc.createVariable(latname, np.float32, (latname,))
        lat_var[:] = lat
        lat_var.units = 'degrees_north'
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'latitude'
    if iflon != 'no':
        lon_var = newnc.createVariable(lonname, np.float32, (lonname,))
        lon_var[:] = lon
        lon_var.units = 'degrees_east'
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'longitude'

    if iflevel != 'no':
        level_var = newnc.createVariable(levelname, np.float32, (levelname,))
        level_var[:] = levels
    if iftime != 'no':
        if len(times) == 0:
            raise ValueError("The 'times' array is empty, cannot determine a reference date or write time data.")
        time_var = newnc.createVariable(timename, np.float64, (timename,))
        if time0_ref_date is None:
            time0_ref_np = times[0]
            if np.isnat(time0_ref_np):
                raise ValueError("The first time point in 'times' array is 'Not a Time' (NaT), cannot use as reference date.")
            time0_ref = datetime.utcfromtimestamp(time0_ref_np.astype('int64') / 1e9)

        elif isinstance(time0_ref_date, str):
            try:
                time0_ref = datetime.fromisoformat(time0_ref_date.replace(' ', 'T'))
            except ValueError:
                time0_ref = datetime.strptime(time0_ref_date, '%Y-%m-%d %H:%M:%S')
        else:
            time0_ref = time0_ref_date
        units_str = f"days since {time0_ref.strftime('%Y-%m-%d %H:%M:%S')}"
        datetime_objects_list = []
        for dt64_val in times:
            if np.isnat(dt64_val):
                raise ValueError(f"Found 'Not a Time' (NaT) in the 'times' array. Please handle NaT values before passing them to create_nc.")
            datetime_objects_list.append(datetime.utcfromtimestamp(dt64_val.astype('int64') / 1e9))
        numeric_times = netCDF4.date2num(datetime_objects_list, units=units_str, calendar='standard')
        time_var[:] = numeric_times
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

    vs = newnc.createVariable(v_name, np.float32, tuple(v_dims_ordered))
    vs[:] = v

    newnc.close()

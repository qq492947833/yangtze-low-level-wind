"""Select the study's paired initialization times and their physical ERA5 inputs."""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path(os.getenv("YANGTZE_MODEL_DIR", "models")))
    parser.add_argument("--data", type=Path, default=Path(os.getenv("YANGTZE_DATA_DIR", "data")))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hour = np.timedelta64(1, "h")
    train_pool = test_pool = None
    for lead in range(1, 7):
        scaler = args.models / f"DA_TransUNet_{lead}_24year_changjiang_residual_normalization.npz"
        with np.load(scaler, allow_pickle=False) as z:
            train = z["residual_target_times"][:int(z["train_stop"])].astype("datetime64[ns]") - lead * hour
            test = z["test_times"].astype("datetime64[ns]") - lead * hour
            if lead == 1:
                latitude, longitude = z["latitude"], z["longitude"]
            elif not (np.array_equal(latitude, z["latitude"]) and np.array_equal(longitude, z["longitude"])):
                raise ValueError("Attribution grid differs between leads")
        train_pool = train if train_pool is None else np.intersect1d(train_pool, train)
        test_pool = test if test_pool is None else np.intersect1d(test_pool, test)
    rng = np.random.default_rng(25)
    bg_index = np.sort(rng.choice(len(train_pool), 100, replace=False))
    sample_index = np.sort(rng.choice(len(test_pool), 100, replace=False))
    selected = dict(background=train_pool[bg_index], sample=test_pool[sample_index])
    result = dict(background_pool_index=bg_index, sample_pool_index=sample_index)
    for group, times in selected.items():
        result[group + "_init_time"] = times
        result[group + "_history_time"] = times[:, None] + np.arange(-10, 1) * hour
        result[group + "_valid_time"] = times[:, None] + np.arange(1, 7) * hour
    if (len(latitude), len(longitude)) != (49, 73):
        raise ValueError("Expected the 49 by 73 regional model grid")
    values = np.empty((200, 11, 3, 49, 73), dtype=np.float64)
    histories = np.concatenate((result["background_history_time"], result["sample_history_time"]))
    for c, (name, var) in enumerate((("Surface-pressure.nc", "sp"), ("2m-temperature.nc", "t2m"),
                                      ("2m-dewpoint-temperature.nc", "d2m"))):
        with xr.open_dataset(args.data / name) as ds:
            for j, times in enumerate(histories):
                values[j, :, c] = ds[var].sel(time=times, latitude=latitude, longitude=longitude).transpose(
                    "time", "latitude", "longitude").values
    if not np.isfinite(values).all():
        raise ValueError("Missing values in the sampled histories")
    # The regional mean uses the complete original rectangular model domain, equally weighted.
    result.update(output_valid_mask=np.ones((9, 49, 73), dtype=bool), latitude=latitude, longitude=longitude,
                  background_values=values[:100], x_values=values[100:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)


if __name__ == "__main__":
    main()

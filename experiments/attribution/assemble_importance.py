"""Assemble the stage-specific relative-importance figure inputs from converged EG blocks."""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def assemble(stage, blocks, sample_file, output):
    leads = np.arange(1, 7)
    targets = [c + str(p) for c in ("u", "v", "omega") for p in (850, 925, 1000)]
    features = ["sp", "t2m", "d2m"] + (["pred_" + c for c in targets] if stage == 2 else [])
    dims = ("sample", "forecast_time", "y_feature")
    shape = (100, 6, 9)
    phi = np.empty(shape + (len(features),), dtype=np.float64)
    scalars = {k: np.empty(shape, dtype=np.float64) for k in (
        "prediction", "expected_value", "attribution_sum", "efficiency_error", "efficiency_tolerance",
        "convergence_l1", "convergence_feature_share_l1", "signed_feature_convergence_l1")}
    flags = {k: np.empty(shape, dtype=np.int8) for k in ("quality_pass", "convergence_robust_order", "convergence_pass")}
    counts = np.empty(shape, dtype=np.int32)
    complete = np.zeros((100, 6), dtype=np.int8)
    with np.load(sample_file, allow_pickle=False) as raw:
        times = {k: raw[k] for k in (
            "sample_init_time", "sample_valid_time", "sample_pool_index", "background_init_time", "background_pool_index")}
    for i in range(100):
        for h in leads:
            cell = (i, h - 1)
            with np.load(blocks / f"stage{stage}_lead{h}_sample{i:03d}.npz", allow_pickle=False) as z:
                if (int(z["stage"]), int(z["lead"]), int(z["sample"])) != (stage, h, i):
                    raise ValueError(f"Unexpected block at {cell}")
                if (z["init_time"] != times["sample_init_time"][i] or
                        z["valid_time"] != times["sample_valid_time"][cell] or
                        z["pool_index"] != times["sample_pool_index"][i]):
                    raise ValueError(f"Unmatched initialization at {cell}")
                phi[cell] = z["phi_feature"]
                for k, a in (*scalars.items(), *flags.items()):
                    a[cell] = z[k]
                counts[cell] = z["nsamples"]
                if not all(a[cell].all() for a in flags.values()):
                    raise ValueError(f"Unconverged attribution at {cell}")
                complete[cell] = 1
    if not np.isfinite(phi).all() or not all(np.isfinite(a).all() for a in scalars.values()):
        raise ValueError("Nonfinite attribution or diagnostic")
    importance = np.abs(phi).mean(axis=0)
    unit = ["m s-1"] * 6 + ["Pa s-1"] * 3
    res = xr.Dataset(
        data_vars={**{k: (dims, v) for k, v in {**scalars, **flags}.items()},
                   "nsamples": (dims, counts), "shap_values": ((*dims, "x_feature"), phi),
                   "importance": (("forecast_time", "y_feature", "x_feature"), importance),
                   "block_complete": (("sample", "forecast_time"), complete),
                   "sample_init_time": ("sample", times["sample_init_time"]),
                   "sample_target_time": (("sample", "forecast_time"), times["sample_valid_time"]),
                   "sample_pool_index": ("sample", times["sample_pool_index"]),
                   "background_init_time": ("background", times["background_init_time"]),
                   "background_pool_index": ("background", times["background_pool_index"]),
                   "x_feature_units": ("x_feature", ["Pa", "K", "K"] + (unit if stage == 2 else [])),
                   "y_feature_units": ("y_feature", unit), "y_pressure_hpa": ("y_feature", [850, 925, 1000] * 3)},
        coords={"sample": np.arange(100), "background": np.arange(100), "forecast_time": leads,
                "y_feature": targets, "x_feature": features, "lag_time": np.arange(-5, 1)},
        attrs={"stage": stage, "method_id": "expected_gradients_stagewise_physical_region_mean_v1",
               "target_definition": "main_model_forecast" if stage == 1 else "residual_correction_only",
               "importance_definition": "Absolute signed feature sums averaged over samples for each lead."})
    res["forecast_time"].attrs["long_name"] = "forecast lead in hours"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"DA_TransUNet_stage{stage}_shap_values.nc"
    tmp = path.with_suffix(".pending.nc")
    res.to_netcdf(tmp, engine="netcdf4")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--blocks", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    assemble(args.stage, args.blocks, args.samples, args.results)

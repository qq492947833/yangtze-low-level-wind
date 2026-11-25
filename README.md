# Yangtze low-level wind forecasts

Code for forecasting three-dimensional wind over the middle and lower Yangtze River basin at lead times of 1–6 h. The direct networks use hourly surface pressure, 2 m temperature and 2 m dew-point temperature. Their targets are eastward wind, northward wind and pressure vertical velocity at 850, 925 and 1000 hPa on a 49 × 73 grid. Physical outputs use m s⁻¹ for horizontal winds and Pa s⁻¹ for pressure vertical velocity.

## Entry points

| Task | Start here |
| --- | --- |
| Train the six-hour-input DA-TransUNet (Direct6) | [models/direct6_1h.ipynb](models/direct6_1h.ipynb); use the corresponding notebook for each lead. The six other backbone families are in the same directory. |
| Train the residual stage (TwoStage) | [models/two_stage.ipynb](models/two_stage.ipynb), after training the matching Direct6 model. Select its forecast lead. The final prediction adds the predicted correction to the first-stage forecast. |
| Train an eleven-hour-input single-stage model (Direct11) | [direct11/](direct11/). Prepare the input arrays once, then train and export one model per lead. |
| Statistical baselines and additional experiments | [experiments/README.md](experiments/README.md) and the `pinn_*` notebooks in [models/](models/). |
| Paper figures | [figures/](figures/). For the comparison of Direct6, Direct11 and TwoStage, open [FigS24_Direct6_Direct11_TwoStage.ipynb](figures/FigS24_Direct6_Direct11_TwoStage.ipynb). |

## Data and use

Install the packages in [requirements.txt](requirements.txt) and a PyTorch build appropriate for your hardware. Provide your own hourly ERA5 fields on the study grid. The source file conventions are `Surface-pressure.nc`, `2m-temperature.nc`, `2m-dewpoint-temperature.nc` and one pressure-level file per component and level (for example, `U-component-of-wind-850hpa.nc`). Set `YANGTZE_DATA_DIR`, `YANGTZE_MODEL_DIR`, `YANGTZE_RESULTS_DIR` and `YANGTZE_WORK_DIR` to absolute directories before starting Jupyter or Python. Open each notebook with its own directory as the working directory so that local imports resolve. Model weights, reanalysis data, saved predictions and figure inputs are not bundled.

The original regional input grid spans 24–36°N and 108–126°E at 0.25°. For the study-period Direct11 preparation, provide the hourly record from 2000-01-01 00:00 through 2023-12-31 23:00 UTC; the first six hours supply context before the Direct6 time origin. Run the two-stage notebook for each matching first-stage checkpoint. The plotting notebooks read the products for their figure. If a saved figure input is missing, use its corresponding `prepare_*` entry in `figures/`. For Fig. S24, set `YANGTZE_SCORE_DIR` if the per-lead score tables are available outside the default results tree; the notebook also supplies the prediction-file route. The release covers scientific plotting code; the original presentation-drawn schematics for Figs. 2–4 are not included.

For Direct11, run the following commands from `direct11/`. The example is for the 1 h lead; repeat the last two commands for leads 2–6. Set `CUDA_VISIBLE_DEVICES` to the chosen training device. `DIRECT11_MICROBATCH` controls the in-memory batch size; an optimizer step still corresponds to the original group of up to 500 samples.

```sh
python direct11_data.py
python direct11_worker.py train --lead 1
python direct11_worker.py export --lead 1
```

After producing the six direct and residual forecast sets, run `python direct11_compare.py run` from the same directory. The comparison uses common valid times, the raw ERA5 reference, and paired 72 h time blocks. The legacy field name `RMSPE` in these files denotes gridpoint RMSE divided by the time range of the reference series; it is dimensionless and is not multiplied by 100. Direct11 NetCDFs already store physical vertical velocity; do not apply an additional scale factor to them. Figure scripts retain their original metric definitions.

**Version note.** The revised DA training entries in v1.0.0 fit gridpoint scaling statistics on the training inputs and training targets, then reuse them on held-out samples. Earlier DA preprocessing code computed statistics over the full lead-aligned record. The release change was not used to regenerate or verify the previously reported forecasts. Use the matching scaler with each checkpoint; applying release scalers to historical weights does not reproduce the historical workflow.

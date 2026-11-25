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

**Version note.** The DA training entries in this release fit gridpoint means and standard deviations on each experiment's original training inputs and training labels, then reuse them on validation and test samples. This applies to existing standardization of VMD modes, surface-gradient channels, forecast inputs to the residual stage, and residual labels. VMD decomposition operates on the split-specific signals in physical units. Score calculations retain the test-reference range in the denominator of RMSPE.


## Figure-to-code index

The paths below follow the figure numbers in the manuscript and supplement. Read each row from left to right. Shared programs handle several figures. The three architectural and design schematics were drawn in presentation software. The linked entries show the implementations used in those schematics.

### Main text

| Figure | Quantity | Experiment and score or diagnostic | Plot |
| --- | --- | --- | --- |
| 1 | Regional wind climatology | [climatology][f1p] | [Fig1_wind_climatology.py][f1] |
| 2 | ST-TransUNet architecture | [ST-TransUNet][st] | Hand-drawn schematic |
| 3 | CNN-Transformer architecture | [CNN-Transformer][tr] | Hand-drawn schematic |
| 4 | Experimental design | [seven backbones][bb] · [two-stage forecasts][ts] · [Ridge][ri] | Hand-drawn schematic |
| 5 | Seven-backbone skill by lead | [seven backbones][bb] · [seven-backbone metrics][f5p] | [Fig5_S1_backbone_skill.ipynb][f5] |
| 6 | Spatial change in range-normalized RMSE after correction | [Direct6][d6] · [two-stage forecasts][ts] · [paired residual maps][f6p] | [Fig6_S8_residual_maps.py][f6] |
| 7 | Two-stage, Ridge and persistence skill | [two-stage forecasts][ts] · [Ridge][ri] · [persistence and baseline scoring][f7p] | [Fig7_S13_baselines.py][f7] |
| 8 | Spatial correlation and range-normalized RMSE | [two-stage forecasts][ts] · [gridded metrics and mean vectors][f8p] | [Fig8_Fig9_S14_spatial_winds.py][f8] |
| 9 | Mean horizontal vectors at 6 h | [two-stage forecasts][ts] · [gridded metrics and mean vectors][f8p] | [Fig8_Fig9_S14_spatial_winds.py][f8] |
| 10 | Conditional skill at training-period extreme thresholds | [two-stage forecasts][ts] · [training thresholds and extreme metrics][f10p] | [Fig10_S15_extreme_winds.py][f10] |
| 11 | Typhoon Lekima, 850 hPa | [two-stage forecasts][ts] · [case extraction][f11p] | [Fig11_lekima_case.py][f11] |
| 12 | Discussion: PINN continuity, gradient and VMD gains across stages | [paired seeds][seeds] · [continuity decomposition][pc] · [gradient forecasting stages][gt] · [conditional scores][gc] · [cross-stage aggregation][ga] · [VMD forecasting stages][vt] · [temporal bands][vb] | [Fig12_S16_S19_mechanisms.ipynb][f12] |
| 13 | Discussion: first-stage predictor importance | [Direct6][d6] · [stage inputs and attribution protocol][at] · [expected gradients][eg] · [signed channel aggregation][ea] | [Fig13_S20_shap_importance.ipynb][f13] |

### Supplementary material

| Figure | Quantity | Experiment and score or diagnostic | Plot |
| --- | --- | --- | --- |
| S1 | Component scores for seven backbones | [seven backbones][bb] · [seven-backbone metrics][f5p] | [Fig5_S1_backbone_skill.ipynb][f5] |
| S2 | Five 1 h physics-loss configurations | [Direct6][d6] · [physical loss][loss] · [loss-weight search][tune] · [full PINN][pfull] · [loss ablations][ab] · [ablation scores][asc] | [FigS2_pinn_ablation.py][s2] |
| S3 | Continuity, vorticity and shear diagnostics | [Direct6][d6] · [physical loss][loss] · [full PINN][pfull] · [loss ablations][ab] · [physical diagnostics, bootstrap and checkpoint gradients][s3p] | [FigS3_S5_physics.py][s3] |
| S4 | Paired changes in errors and physical diagnostics | [Direct6][d6] · [physical loss][loss] · [full PINN][pfull] · [loss ablations][ab] · [physical diagnostics, bootstrap and checkpoint gradients][s3p] | [FigS3_S5_physics.py][s3] |
| S5 | Checkpoint loss terms and parameter-gradient diagnostics | [physical loss][loss] · [full PINN][pfull] · [loss ablations][ab] · [physical diagnostics, bootstrap and checkpoint gradients][s3p] | [FigS3_S5_physics.py][s3] |
| S6 | 1 h paired-seed skill by component | [paired seeds][seeds] · [paired-seed scores][ssc] | [FigS6_S7_seed_metrics.py][s6] |
| S7 | 1 h paired-seed skill by pressure level | [paired seeds][seeds] · [paired-seed scores][ssc] | [FigS6_S7_seed_metrics.py][s6] |
| S8 | Spatial change in temporal correlation after correction | [Direct6][d6] · [two-stage forecasts][ts] · [paired residual maps][f6p] | [Fig6_S8_residual_maps.py][f6] |
| S9 | VMD gains in corrected forecasts | [Direct6][d6] · [two-stage forecasts][ts] · [mode search and decomposition][vf] · [VMD forecasting stages][vt] · [common-time VMD scores][s9p] | [FigS9_S10_vmd_gains.py][s9] |
| S10 | VMD gains in first-stage forecasts | [Direct6][d6] · [mode search and decomposition][vf] · [VMD forecasting stages][vt] · [common-time VMD scores][s9p] | [FigS9_S10_vmd_gains.py][s9] |
| S11 | Gradient gains in corrected forecasts | [two-stage forecasts][ts] · [surface-field derivatives][gf] · [gradient forecasting stages][gt] · [paired gradient scores][s11p] | [FigS11_S12_gradient_gains.py][s11] |
| S12 | Gradient gains in first-stage forecasts | [Direct6][d6] · [surface-field derivatives][gf] · [gradient forecasting stages][gt] · [paired gradient scores][s11p] | [FigS11_S12_gradient_gains.py][s11] |
| S13 | Dimensional errors against Ridge and persistence | [two-stage forecasts][ts] · [Ridge][ri] · [persistence and baseline scoring][f7p] | [Fig7_S13_baselines.py][f7] |
| S14 | Spatial RMSE and MAE | [two-stage forecasts][ts] · [gridded metrics and mean vectors][f8p] | [Fig8_Fig9_S14_spatial_winds.py][f8] |
| S15 | Equitable threat scores across thresholds | [two-stage forecasts][ts] · [training thresholds and extreme metrics][f10p] | [Fig10_S15_extreme_winds.py][f10] |
| S16 | Gradient gains by input-gradient regime | [two-stage forecasts][ts] · [surface-field derivatives][gf] · [gradient forecasting stages][gt] · [regimes][gr] · [conditional scores][gc] · [cross-stage aggregation][ga] | [Fig12_S16_S19_mechanisms.ipynb][f12] |
| S17 | Level-averaged VMD gains by temporal band | [two-stage forecasts][ts] · [mode search and decomposition][vf] · [VMD forecasting stages][vt] · [temporal bands][vb] | [Fig12_S16_S19_mechanisms.ipynb][f12] |
| S18 | Gradient gains by spatial scale | [two-stage forecasts][ts] · [surface-field derivatives][gf] · [gradient forecasting stages][gt] · [conditional scores][gc] · [cross-stage aggregation][ga] | [Fig12_S16_S19_mechanisms.ipynb][f12] |
| S19 | Level-resolved VMD gains by temporal band | [two-stage forecasts][ts] · [mode search and decomposition][vf] · [VMD forecasting stages][vt] · [temporal bands][vb] | [Fig12_S16_S19_mechanisms.ipynb][f12] |
| S20 | Residual-stage predictor importance | [Direct6][d6] · [two-stage forecasts][ts] · [stage inputs and attribution protocol][at] · [expected gradients][eg] · [signed channel aggregation][ea] | [Fig13_S20_shap_importance.ipynb][f13] |
| S21 | Extreme-event skill relative to Ridge and persistence | [two-stage forecasts][ts] · [Ridge][ri] · [baseline ETS][s21p] | [FigS21_extreme_baselines.py][s21] |
| S22 | Correlation change with ZWD | [ZWD pairing][zw] · [A/B training][zt] · [paired ZWD scores][zc] | [FigS22_S23_zwd.ipynb][s22] |
| S23 | Range-normalized RMSE change with ZWD | [ZWD pairing][zw] · [A/B training][zt] · [paired ZWD scores][zc] | [FigS22_S23_zwd.ipynb][s22] |
| S24 | Direct6, Direct11 and TwoStage | [Direct6][d6] · [Direct11][d11] · [two-stage forecasts][ts] · [paired comparison][dcomp] | [FigS24_Direct6_Direct11_TwoStage.ipynb][s24] |

Fig. S24 uses the 42,070 hourly target times from 2019-03-15 02:00 through 2023-12-31 23:00 UTC. If supplying precomputed score tables, use absolute component-by-level scores from that window. The alternative route selects the target times from the prediction files and ERA5 and writes the component-by-level score table. The Direct11 comparison program also produces paired summaries on its own documented support.

[bb]: models/
[d6]: models/direct6_1h.ipynb
[ts]: models/two_stage.ipynb
[d11]: direct11/direct11_worker.py
[dcomp]: direct11/direct11_compare.py
[ri]: experiments/ridge/run_ridge.py
[loss]: models/pinn_loss.py
[pfull]: models/pinn_full_1h.ipynb
[ab]: models/pinn_term_ablation.ipynb
[tune]: models/pinn_weight_search.ipynb
[seeds]: models/pinn_paired_seeds.ipynb
[asc]: experiments/pinn_scores/score_ablation.py
[ssc]: experiments/pinn_scores/score_paired_seeds.py
[vf]: experiments/vmd/decompose.ipynb
[vt]: experiments/vmd/train_residual.ipynb
[vb]: experiments/mechanisms/vmd/timescale.py
[gf]: experiments/gradient/gradient_features.py
[gt]: experiments/gradient/gradient_pipeline.py
[gr]: experiments/mechanisms/gradient/regimes.py
[gc]: experiments/mechanisms/gradient/case_scores.py
[ga]: experiments/mechanisms/gradient/assemble.py
[pc]: experiments/mechanisms/pinn/continuity.py
[at]: experiments/attribution/README.md
[eg]: experiments/attribution/stagewise.py
[ea]: experiments/attribution/assemble_importance.py
[zw]: experiments/zwd/prepare_zwd.py
[zt]: experiments/zwd/train_zwd.py
[zc]: experiments/zwd/compare_zwd.py
[f1p]: figures/prepare_Fig1_climatology.py
[f1]: figures/Fig1_wind_climatology.py
[st]: models/st_transunet_1h.ipynb
[tr]: models/cnn_transformer_1h.ipynb
[f5p]: figures/prepare_Fig5_S1_backbone_scores.ipynb
[f5]: figures/Fig5_S1_backbone_skill.ipynb
[f6p]: figures/prepare_Fig6_S8_residual_scores.ipynb
[f6]: figures/Fig6_S8_residual_maps.py
[f7p]: figures/prepare_Fig7_S13_baseline_scores.ipynb
[f7]: figures/Fig7_S13_baselines.py
[f8p]: figures/prepare_Fig8_Fig9_S14_spatial_metrics.ipynb
[f8]: figures/Fig8_Fig9_S14_spatial_winds.py
[f10p]: figures/prepare_Fig10_S15_extreme_metrics.ipynb
[f10]: figures/Fig10_S15_extreme_winds.py
[f11p]: figures/prepare_Fig11_lekima_case.ipynb
[f11]: figures/Fig11_lekima_case.py
[f12]: figures/Fig12_S16_S19_mechanisms.ipynb
[f13]: figures/Fig13_S20_shap_importance.ipynb
[s2]: figures/FigS2_pinn_ablation.py
[s3p]: figures/prepare_FigS3_S5_physics.ipynb
[s3]: figures/FigS3_S5_physics.py
[s6]: figures/FigS6_S7_seed_metrics.py
[s9p]: figures/prepare_FigS9_S10_vmd_scores.py
[s9]: figures/FigS9_S10_vmd_gains.py
[s11p]: figures/prepare_FigS11_S12_gradient_scores.ipynb
[s11]: figures/FigS11_S12_gradient_gains.py
[s21p]: figures/prepare_FigS21_extreme_scores.ipynb
[s21]: figures/FigS21_extreme_baselines.py
[s22]: figures/FigS22_S23_zwd.ipynb
[s24]: figures/FigS24_Direct6_Direct11_TwoStage.ipynb


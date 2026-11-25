# One-hour PINN scores

Set `YANGTZE_RESULTS_DIR` to the directory containing the experiment outputs. Both scripts write their products there.

| Comparison | Training entries in `models/` | Score producer | Figure inputs |
| --- | --- | --- | --- |
| Data loss only, full PINN and the three single-term omissions | `direct6_1h.ipynb`, `pinn_full_1h.ipynb`, `pinn_term_ablation.ipynb` | `score_ablation.py` | Figure S2 reads `da_transunet_1hour_pinn_comparison_metrics.nc`. |
| Matched initializations at seeds 42 and 2026 | `pinn_paired_seeds.ipynb` writes `da_transunet_1hour_pinn_seed_repeats_v1_index.json`. | `score_paired_seeds.py` | Figures S6 and S7 read `da_transunet_1hour_pinn_seed_metrics_v1_summary.csv`. |

The single-term experiment sets one coefficient to zero per run, initializes each of the three runs with seed 25, and writes a distinct prediction and truth file for each wind component. Its no-shear output supplies the five-way comparison. The separate no-shear training entry uses the same filename convention. In the paired-seed experiment, the no-shear condition retains continuity and vorticity, and its data-only partner disables the combined physics term. The seed scorer checks the initial model hash for each pair.

Both scorers evaluate each run against its own test truth on common times and finite observations, then average grid-point scores over metric-specific shared spatial masks. The quantity named `rmspe` is RMSE divided by the local observed range. The five-way scorer takes that range from finite truth values during the shared period. The paired-seed scorer takes it from times that are finite in all four runs. Absolute RMSE for saved vertical wind uses the stored field `w = -0.1 × omega`. When reporting physical pressure velocity, the seed plotting entry converts this RMSE to Pa s⁻¹.

The seed scorer also writes grid-level scores, time alignment, truth differences, within-seed changes and cross-seed direction counts. All share the `da_transunet_1hour_pinn_seed_metrics_v1_` filename prefix.

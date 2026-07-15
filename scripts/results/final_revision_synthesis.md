# Final Revision Result Synthesis

Date: 2026-07-15

## Sources

- 1 km SBCV benchmark: `scripts/results/benchmark/benchmark_results_ceacce0.csv`
- 100 km SBCV benchmark: `scripts/results/benchmark/benchmark_results_d0848f6.csv`
- Figure 4 EVA scale-binned NRMSE: `figures/figure_4/figure_4_normalized_rmse_by_area.csv`
- Training-fraction scaling: `scripts/results/training_fraction_scaling/ceacce0/training_fraction_scaling_results.csv`
- GIFT asymptote audits: `scripts/results/gift_asymptote_evaluation/ceacce0/gift_asymptote_evaluation_results.csv` and `scripts/results/gift_asymptote_evaluation/d0848f6/gift_asymptote_evaluation_results.csv`

## Benchmark Summary

Mean NRMSE across five folds, calculated as RMSE divided by mean observed richness for each model and fold and reported as a percentage. Values are mean +/- standard deviation.

| Dataset | Model | SBCV test NRMSE | GIFT NRMSE | Mean GIFT R2 | Mean GIFT median relative bias |
|---|---:|---:|---:|---:|---:|
| 1 km `ceacce0` | MuScaRi_ClimateDEM | 18.166 +/- 1.028% | 29.473 +/- 4.448% | 0.721 | 0.052 |
| 1 km `ceacce0` | MuScaRi_ClimateDEM_Area | 17.682 +/- 0.615% | 34.587 +/- 2.199% | 0.621 | -0.177 |
| 1 km `ceacce0` | MuScaRi_Area | 59.905 +/- 1.244% | 51.728 +/- 1.585% | 0.154 | -0.225 |
| 1 km `ceacce0` | FFNN_ClimateDEM_Area | 17.594 +/- 1.220% | 163.717 +/- 31.520% | -7.715 | 1.618 |
| 1 km `ceacce0` | Linear_ClimateDEM_Area | 57.489 +/- 0.789% | 8226.337 +/- 730.640% | -21503.281 | 32.759 |
| 100 km `d0848f6` | MuScaRi_ClimateDEM_Area | 32.942 +/- 6.353% | 32.900 +/- 1.697% | 0.657 | -0.072 |
| 100 km `d0848f6` | MuScaRi_ClimateDEM | 34.763 +/- 8.085% | 38.016 +/- 10.569% | 0.515 | 0.267 |
| 100 km `d0848f6` | MuScaRi_Area | 55.376 +/- 4.511% | 50.147 +/- 5.072% | 0.199 | -0.028 |
| 100 km `d0848f6` | FFNN_ClimateDEM_Area | 34.536 +/- 7.911% | 135.870 +/- 25.410% | -4.993 | 1.427 |
| 100 km `d0848f6` | Linear_ClimateDEM_Area | 51.196 +/- 3.730% | 5929.150 +/- 285.516% | -11120.659 | 25.372 |

## Interpretation Notes

- On 1 km SBCV, `MuScaRi_ClimateDEM` has the best asymptotic GIFT NRMSE, while `MuScaRi_ClimateDEM_Area` has slightly lower SBCV test NRMSE. This supports using climate and topography as the main extrapolative signal for the final GIFT-facing model.
- On 100 km SBCV, `MuScaRi_ClimateDEM_Area` has the best aggregate asymptotic GIFT NRMSE and bias calibration, but the improvement over `MuScaRi_ClimateDEM` is not uniform across folds. This should be framed as aggregate robustness, not as a fold-wise dominance claim.
- `MuScaRi_Area` is far weaker on SBCV test sets and GIFT than the environmental MuScaRi models, but it remains useful for the scale-binned diagnostic in Figure 4.
- `FFNN_ClimateDEM_Area` matches MuScaRi-like SBCV test NRMSE but extrapolates poorly to GIFT. This supports the role of the MuScaRi accumulation-curve structure rather than a generic feed-forward predictor.
- The corrected `Linear_ClimateDEM_Area` baseline has reasonable SBCV test NRMSE relative to `MuScaRi_Area`, but catastrophic GIFT extrapolation. This should be described as a log-linear extrapolation failure mode rather than ordinary low performance.

## Figure 4 Diagnostic

Mean EVA test-set NRMSE across area bins, normalized independently within each fold and bin:

| Model | Mean NRMSE |
|---|---:|
| MuScaRi_ClimateDEM | 33.381% |
| MuScaRi_Area | 54.107% |

This supports replacing Shapley-value language with a direct predictive diagnostic: climate and topography carry substantially more interpolation-scale information than area alone across the EVA test-set area range.

## Training-Fraction Scaling

The environment-and-area MuScaRi model was evaluated over nine fractions from $10^{-4}$ to 1 using five folds per fraction. Mean interpolation NRMSE decreased from 146.264% at $10^{-4}$ to 27.922% at $10^{-2}$, then improved more gradually to 19.848% at 0.316 and 17.682% with the full training set. The full-data point reuses the identical selected-model benchmark; the other 40 model-fold points were trained specifically for this diagnostic.

## GIFT Asymptote Audit

Uniform ensemble NRMSE on GIFT:

| Dataset | Model | Finite full-area NRMSE | Asymptotic total NRMSE |
|---|---|---:|---:|
| `ceacce0` | MuScaRi_Area | 51.544% | 51.544% |
| `ceacce0` | MuScaRi_ClimateDEM | 25.997% | 26.136% |
| `ceacce0` | MuScaRi_ClimateDEM_Area | 32.644% | 32.661% |
| `d0848f6` | MuScaRi_Area | 43.952% | 43.976% |
| `d0848f6` | MuScaRi_ClimateDEM | 34.571% | 35.752% |
| `d0848f6` | MuScaRi_ClimateDEM_Area | 31.143% | 31.253% |

The true asymptotic prediction is close to the finite full-area prediction on GIFT, so the GIFT conclusions are not driven by the observed-area convention. Manuscript performance tables report fold-wise means and standard deviations for asymptotic MuScaRi GIFT metrics; this audit table reports uniform ensemble diagnostics.

## Provenance Caveats

- The benchmark script uses GIFT dataset ID `da569da`. The project guide names `418c563` as canonical. The two local compiled GIFT parquet files have identical shape, columns, non-geometry values, CRS, and geometries, but different parquet byte sizes.
- The 1 km benchmark CSV originally had incomplete linear baseline primary metrics. The linear rows were recomputed with the corrected `scripts/benchmark.py` logic and merged into `benchmark_results_ceacce0.csv` without rerunning neural models.

# Final Revision Result Synthesis

Date: 2026-06-26

## Sources

- 1 km SBCV benchmark: `scripts/results/benchmark/benchmark_results_ceacce0.csv`
- 100 km SBCV benchmark: `scripts/results/benchmark/benchmark_results_d0848f6.csv`
- Figure 4 EVA scale-binned relative RMSE: `figures/figure_4/figure_4_relative_rmse_by_area.csv`
- GIFT asymptote audit: `scripts/results/gift_asymptote_evaluation/ceacce0/gift_asymptote_evaluation_results.csv`

## Benchmark Summary

Mean RMSE across five folds. Values are mean +/- standard deviation.

| Dataset | Model | SBCV test RMSE | GIFT RMSE | Mean GIFT R2 | Mean GIFT median relative bias |
|---|---:|---:|---:|---:|---:|
| 1 km `ceacce0` | MuScaRi_ClimateDEM | 48.244 +/- 1.916 | 458.834 +/- 66.273 | 0.725 | 0.051 |
| 1 km `ceacce0` | MuScaRi_ClimateDEM_Area | 46.982 +/- 1.393 | 540.815 +/- 32.646 | 0.623 | -0.178 |
| 1 km `ceacce0` | MuScaRi_Area | 157.827 +/- 4.981 | 810.648 +/- 24.853 | 0.154 | -0.225 |
| 1 km `ceacce0` | FFNN_ClimateDEM_Area | 46.726 +/- 2.673 | 2565.690 +/- 493.967 | -7.715 | 1.618 |
| 1 km `ceacce0` | Linear_ClimateDEM_Area | 152.814 +/- 4.651 | 128918.722 +/- 11450.202 | -21503.281 | 32.759 |
| 100 km `d0848f6` | MuScaRi_ClimateDEM_Area | 92.916 +/- 14.535 | 513.718 +/- 26.076 | 0.660 | -0.074 |
| 100 km `d0848f6` | MuScaRi_ClimateDEM | 97.929 +/- 18.847 | 578.796 +/- 155.840 | 0.544 | 0.262 |
| 100 km `d0848f6` | MuScaRi_Area | 156.781 +/- 26.540 | 782.893 +/- 74.557 | 0.206 | -0.036 |
| 100 km `d0848f6` | FFNN_ClimateDEM_Area | 97.465 +/- 19.232 | 2129.284 +/- 398.210 | -4.993 | 1.427 |
| 100 km `d0848f6` | Linear_ClimateDEM_Area | 146.103 +/- 22.859 | 92918.445 +/- 4474.452 | -11120.659 | 25.372 |

## Interpretation Notes

- On 1 km SBCV, `MuScaRi_ClimateDEM` has the best GIFT RMSE, while `MuScaRi_ClimateDEM_Area` has slightly lower SBCV test RMSE. This supports using climate and topography as the main extrapolative signal for the final GIFT-facing model.
- On 100 km SBCV, `MuScaRi_ClimateDEM_Area` has the best aggregate GIFT RMSE and bias calibration, but the improvement over `MuScaRi_ClimateDEM` is not uniform across folds. This should be framed as aggregate robustness, not as a fold-wise dominance claim.
- `MuScaRi_Area` is far weaker on SBCV test sets and GIFT than the environmental MuScaRi models, but it remains useful for the scale-binned diagnostic in Figure 4.
- `FFNN_ClimateDEM_Area` matches MuScaRi-like SBCV test RMSE but extrapolates poorly to GIFT. This supports the role of the MuScaRi accumulation-curve structure rather than a generic feed-forward predictor.
- The corrected `Linear_ClimateDEM_Area` baseline has reasonable SBCV test RMSE relative to `MuScaRi_Area`, but catastrophic GIFT extrapolation. This should be described as a log-linear extrapolation failure mode rather than ordinary low performance.

## Figure 4 Diagnostic

Mean EVA test-set relative RMSE across area bins:

| Model | Mean relative RMSE |
|---|---:|
| MuScaRi_ClimateDEM | 33.381% |
| MuScaRi_Area | 54.107% |

This supports replacing Shapley-value language with a direct predictive diagnostic: climate and topography carry substantially more interpolation-scale information than area alone across the EVA test-set area range.

## GIFT Asymptote Audit

Uniform ensemble RMSE on GIFT:

| Model | Finite full-area RMSE | Asymptotic total RMSE |
|---|---:|---:|
| MuScaRi_Area | 807.766 | 807.768 |
| MuScaRi_ClimateDEM | 407.409 | 409.584 |
| MuScaRi_ClimateDEM_Area | 511.574 | 511.838 |

The true asymptotic prediction is essentially identical to finite full-area prediction on GIFT, so the GIFT conclusions are not driven by the observed-area convention.

## Provenance Caveats

- The benchmark script uses GIFT dataset ID `da569da`. The project guide names `418c563` as canonical. The two local compiled GIFT parquet files have identical shape, columns, non-geometry values, CRS, and geometries, but different parquet byte sizes.
- The 1 km benchmark CSV originally had incomplete linear baseline primary metrics. The linear rows were recomputed with the corrected `scripts/benchmark.py` logic and merged into `benchmark_results_ceacce0.csv` without rerunning neural models.

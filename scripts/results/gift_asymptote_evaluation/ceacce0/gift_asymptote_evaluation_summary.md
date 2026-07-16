# GIFT Asymptote Evaluation

Dataset: `ceacce0`
GIFT dataset: `418c563`
Observed-area policy: `observed_area_set_to_sp_unit_area`

Uniform ensemble metrics:

| model_name | prediction_mode | nrmse_percent | r2 | median_relative_bias | bias_slope_log_area | pred_mean |
| --- | --- | --- | --- | --- | --- | --- |
| MuScaRi_Area | finite_full_area | 51.544 | 0.161 | -0.226 | 0.092 | 1302.282 |
| MuScaRi_Area | asymptotic_total | 51.544 | 0.161 | -0.226 | 0.092 | 1302.290 |
| MuScaRi_ClimateDEM | finite_full_area | 25.997 | 0.787 | 0.063 | -0.036 | 1642.313 |
| MuScaRi_ClimateDEM | asymptotic_total | 26.136 | 0.784 | 0.066 | -0.038 | 1645.045 |
| MuScaRi_ClimateDEM_Area | finite_full_area | 32.644 | 0.664 | -0.163 | 0.000 | 1323.890 |
| MuScaRi_ClimateDEM_Area | asymptotic_total | 32.661 | 0.663 | -0.163 | -0.001 | 1325.329 |

Full results: `scripts/results/gift_asymptote_evaluation/ceacce0/gift_asymptote_evaluation_results.csv`

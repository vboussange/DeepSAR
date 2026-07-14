# GIFT Asymptote Evaluation

Dataset: `d0848f6`
GIFT dataset: `da569da`
Observed-area policy: `observed_area_set_to_sp_unit_area`

Uniform ensemble metrics:

| model_name | prediction_mode | rmse | r2 | median_relative_bias | bias_slope_log_area | pred_mean |
| --- | --- | --- | --- | --- | --- | --- |
| MuScaRi_Area | finite_full_area | 688.786 | 0.390 | -0.036 | 0.060 | 1515.728 |
| MuScaRi_Area | asymptotic_total | 689.163 | 0.389 | -0.022 | 0.055 | 1522.103 |
| MuScaRi_ClimateDEM | finite_full_area | 541.783 | 0.623 | 0.249 | -0.092 | 1921.843 |
| MuScaRi_ClimateDEM | asymptotic_total | 560.292 | 0.596 | 0.250 | -0.106 | 1937.478 |
| MuScaRi_ClimateDEM_Area | finite_full_area | 488.052 | 0.694 | -0.083 | -0.024 | 1435.287 |
| MuScaRi_ClimateDEM_Area | asymptotic_total | 489.776 | 0.692 | -0.080 | -0.028 | 1438.951 |

Full results: `scripts/results/gift_asymptote_evaluation/d0848f6/gift_asymptote_evaluation_results.csv`

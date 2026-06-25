# Ensemble Weighting Analysis

Artifact: `scripts/results/benchmark/artifacts/MuScaRi_ClimateDEM/dae0789a3c87`

## Recommendation

Use uniform averaging for this artifact. It gives the best GIFT RMSE among the tested strategies.

## GIFT Performance

| weighting_strategy | rmse | mape | r2 | d2 | mean_relative_bias | effective_n |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 407.4089 | 0.3406 | 0.7866 | 0.5472 | 0.2349 | 5.0000 |
| inverse_val_rmse | 408.1616 | 0.3411 | 0.7858 | 0.5464 | 0.2355 | 4.9953 |
| inverse_val_rmse_squared | 408.9281 | 0.3415 | 0.7850 | 0.5457 | 0.2360 | 4.9817 |

## Interpolation Performance

| weighting_strategy | rmse | mape | r2 | d2 | mean_relative_bias | effective_n |
| --- | --- | --- | --- | --- | --- | --- |
| inverse_val_rmse_squared | 43.4189 | 0.5021 | 0.9897 | 0.8660 | 0.3229 | 4.9817 |
| inverse_val_rmse | 43.4386 | 0.5021 | 0.9897 | 0.8660 | 0.3229 | 4.9953 |
| uniform | 43.4598 | 0.5021 | 0.9897 | 0.8659 | 0.3229 | 5.0000 |

## Ensemble Standard Deviation

| split | weighting_strategy | ensemble_std_mean | ensemble_std_median | mean_std_ratio_to_uniform_mean | spearman_std_abs_error | effective_n |
| --- | --- | --- | --- | --- | --- | --- |
| gift | uniform | 175.1582 | 131.1658 | 1.0000 | 0.2396 | 5.0000 |
| gift | inverse_val_rmse | 175.6060 | 130.9537 | 1.0026 | 0.2454 | 4.9953 |
| gift | inverse_val_rmse_squared | 175.9980 | 131.5401 | 1.0048 | 0.2513 | 4.9817 |
| interpolation | uniform | 10.4108 | 7.5834 | 1.0000 | 0.3540 | 5.0000 |
| interpolation | inverse_val_rmse | 10.4107 | 7.5812 | 1.0000 | 0.3541 | 4.9953 |
| interpolation | inverse_val_rmse_squared | 10.4073 | 7.5738 | 0.9997 | 0.3542 | 4.9817 |

## Notes

- Best GIFT RMSE delta vs uniform: 0.0000 (0.000%).
- Current validation-RMSE-squared weighting effective_n: 4.982.
- Current GIFT mean ensemble-SD ratio vs uniform: 1.0048.
- Current vs uniform GIFT Spearman(std, abs error): 0.2513 vs 0.2396.
- GIFT is used only for evaluation in this script, not for fitting weights.

## Weight Table

| weighting_strategy | fold | val_rmse | weight | effective_n |
| --- | --- | --- | --- | --- |
| uniform | 0 | 46.892141 | 0.200000 | 5.000000 |
| uniform | 1 | 46.946412 | 0.200000 | 5.000000 |
| uniform | 2 | 50.357918 | 0.200000 | 5.000000 |
| uniform | 3 | 49.196778 | 0.200000 | 5.000000 |
| uniform | 4 | 46.635280 | 0.200000 | 5.000000 |
| inverse_val_rmse | 0 | 46.892141 | 0.204554 | 4.995313 |
| inverse_val_rmse | 1 | 46.946412 | 0.204318 | 4.995313 |
| inverse_val_rmse | 2 | 50.357918 | 0.190476 | 4.995313 |
| inverse_val_rmse | 3 | 49.196778 | 0.194972 | 4.995313 |
| inverse_val_rmse | 4 | 46.635280 | 0.205681 | 4.995313 |
| inverse_val_rmse_squared | 0 | 46.892141 | 0.209016 | 4.981652 |
| inverse_val_rmse_squared | 1 | 46.946412 | 0.208533 | 4.981652 |
| inverse_val_rmse_squared | 2 | 50.357918 | 0.181236 | 4.981652 |
| inverse_val_rmse_squared | 3 | 49.196778 | 0.189892 | 4.981652 |
| inverse_val_rmse_squared | 4 | 46.635280 | 0.211324 | 4.981652 |

# Fold-paired 100 km GIFT statistics

Source: `scripts/results/gift_asymptote_evaluation/d0848f6/gift_asymptote_evaluation_results.csv` (`asymptotic_total`, `fold_member` rows).
All models use folds 0--4 and the same 178 GIFT samples. The results file records GIFT dataset `da569da`; its compiled data are exactly equivalent in columns, values, CRS, and geometry to canonical dataset `418c563`.

## Environment and area versus environment only

Fold NRMSE values (%) were 55.41, 32.81, 40.61, 31.42, 29.83 for environment only and 33.94, 30.22, 34.33, 33.78, 32.24 for environment and area. Their fold means were 38.02% and 32.90%, respectively. The mean paired difference (environment and area minus environment only) was -5.12 percentage points (95% CI [-17.34, 7.11]; raw paired t-test P=0.310; Holm-adjusted P=0.310). The exact sign-flip P value was 0.3125.

Fold-level absolute median relative biases were 0.467, 0.196, 0.357, 0.204, 0.111 for environment only and 0.117, 0.012, 0.081, 0.123, 0.025 for environment and area. The mean paired difference was -0.196 (95% CI [-0.342, -0.049]; raw paired t-test P=0.021; Holm-adjusted P=0.062). The exact sign-flip P value was 0.0625.

The Holm-adjusted paired tests and exact sign-flip sensitivity checks agree qualitatively at alpha=0.05 for both targeted contrasts. The combined model has lower aggregate NRMSE and lower absolute fold-level median relative bias, but neither contrast meets the adjusted significance threshold with five folds.

## Full pairwise results

See `scripts/results/gift_asymptote_evaluation/d0848f6/gift_100km_pairwise_statistics.csv` for all three pairwise comparisons within each metric-specific Holm family, including fold values and exact sign-flip results.

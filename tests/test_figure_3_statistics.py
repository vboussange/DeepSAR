import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from figures.figure_3.figure_3 import (
    LABEL_MAP,
    analyze_performance_panels,
    load_benchmark_results,
    paired_model_comparisons,
)


class Figure3StatisticsTest(unittest.TestCase):
    def setUp(self):
        values = {
            "A": [10.0, 12.0, 9.0, 11.0, 10.0],
            "B": [13.0, 11.0, 12.0, 15.0, 14.0],
            "C": [20.0, 22.0, 21.0, 23.0, 20.0],
        }
        self.df = pd.DataFrame(
            [
                {"experiment": model, "fold": fold, "interp_nrmse_percent": value}
                for model, model_values in values.items()
                for fold, value in enumerate(model_values)
            ]
        )

    def test_matches_folds_and_is_invariant_to_row_order(self):
        shuffled = self.df.sample(frac=1, random_state=7).reset_index(drop=True)
        result = paired_model_comparisons(shuffled, "interp", ["A", "B", "C"])
        ordered_result = paired_model_comparisons(self.df, "interp", ["A", "B", "C"])

        pd.testing.assert_frame_equal(result.fold_values, ordered_result.fold_values)
        pd.testing.assert_frame_equal(result.pairwise_results, ordered_result.pairwise_results)

        row = result.pairwise_results.query("model == 'A' and reference_model == 'B'").iloc[0]
        expected = stats.ttest_rel(
            result.fold_values["A"].to_numpy(),
            result.fold_values["B"].to_numpy(),
        )
        self.assertAlmostEqual(row["t_statistic"], expected.statistic)
        self.assertAlmostEqual(row["p_value_raw"], expected.pvalue)
        self.assertEqual(row["fold_ids"], "0;1;2;3;4")
        self.assertEqual(row["n"], 5)
        self.assertEqual(row["df"], 4)

    def test_rejects_duplicate_missing_and_nonfinite_model_fold_values(self):
        duplicated = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate model-fold"):
            paired_model_comparisons(duplicated, "interp", ["A", "B", "C"])

        missing = self.df.drop(self.df.query("experiment == 'B' and fold == 3").index)
        with self.assertRaisesRegex(ValueError, "has folds"):
            paired_model_comparisons(missing, "interp", ["A", "B", "C"])

        nonfinite = self.df.copy()
        nonfinite.loc[
            (nonfinite["experiment"] == "C") & (nonfinite["fold"] == 2),
            "interp_nrmse_percent",
        ] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            paired_model_comparisons(nonfinite, "interp", ["A", "B", "C"])

    def test_holm_family_matrix_decisions_and_letters_agree(self):
        result = paired_model_comparisons(self.df, "interp", ["A", "B", "C"])
        expected_reject, expected_adjusted, _, _ = multipletests(
            result.pairwise_results["p_value_raw"], alpha=0.05, method="holm"
        )

        np.testing.assert_allclose(result.pairwise_results["p_value_holm"], expected_adjusted)
        np.testing.assert_array_equal(result.pairwise_results["reject_holm"], expected_reject)
        self.assertEqual(len(result.pairwise_results), 3)

        for row in result.pairwise_results.itertuples(index=False):
            self.assertAlmostEqual(
                result.adjusted_p_matrix.loc[row.model, row.reference_model],
                row.p_value_holm,
            )
            self.assertEqual(
                result.rejection_matrix.loc[row.model, row.reference_model],
                row.reject_holm,
            )
            shared_letters = set(result.letters[row.model]) & set(result.letters[row.reference_model])
            self.assertEqual(bool(shared_letters), not row.reject_holm)

    def test_models_without_endpoint_values_are_not_in_the_family(self):
        unavailable = pd.DataFrame(
            {
                "experiment": ["D"] * 5,
                "fold": range(5),
                "interp_nrmse_percent": [np.nan] * 5,
            }
        )
        result = paired_model_comparisons(
            pd.concat([self.df, unavailable], ignore_index=True),
            "interp",
            ["A", "B", "C", "D"],
        )
        self.assertEqual(list(result.fold_values), ["A", "B", "C"])
        self.assertNotIn("D", result.adjusted_p_matrix.index)

    def test_real_figure_families_have_complete_five_fold_inputs(self):
        df = load_benchmark_results()
        df = df[df["experiment"].isin(LABEL_MAP)].copy()
        analyses = analyze_performance_panels(df, list(LABEL_MAP))

        self.assertEqual(analyses["interp"].fold_values.shape, (5, 5))
        self.assertEqual(analyses["extrap"].fold_values.shape, (5, 6))
        self.assertEqual(len(analyses["interp"].pairwise_results), 10)
        self.assertEqual(len(analyses["extrap"].pairwise_results), 15)
        self.assertTrue((analyses["interp"].pairwise_results[["n", "df"]] == [5, 4]).all().all())
        self.assertTrue((analyses["extrap"].pairwise_results[["n", "df"]] == [5, 4]).all().all())
        self.assertEqual(analyses["interp"].metric_column, "interp_nrmse_percent")
        self.assertEqual(analyses["extrap"].metric_column, "extrap_nrmse_percent")

        interpolation = analyses["interp"].pairwise_results
        focal = interpolation[
            (interpolation["model"] == "MuScaRi_ClimateDEM_Area")
            & (interpolation["reference_model"] == "MuScaRi_ClimateDEM")
        ].iloc[0]
        self.assertAlmostEqual(focal["mean_paired_difference"], -0.483969, places=5)
        self.assertAlmostEqual(focal["ci95_lower"], -1.038713, places=5)
        self.assertAlmostEqual(focal["ci95_upper"], 0.070775, places=5)
        self.assertAlmostEqual(focal["p_value_holm"], 0.145177, places=5)


if __name__ == "__main__":
    unittest.main()

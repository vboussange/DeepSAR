import unittest

import numpy as np

from figures.figure_4.figure_4 import MODEL_SPECS, normalized_rmse


class Figure4NormalizedRmseTest(unittest.TestCase):
    def test_canonical_figure_uses_all_muscari_feature_variants(self):
        self.assertEqual(
            [(spec["name"], spec["label"]) for spec in MODEL_SPECS],
            [
                ("MuScaRi_Area", "Area only"),
                ("MuScaRi_ClimateDEM", "Environment only"),
                ("MuScaRi_ClimateDEM_Area", "Environment + area"),
            ],
        )

    def test_matches_previous_mean_normalized_definition_within_bin(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([13.0, 18.0, 35.0])

        rmse, mean_sr, nrmse = normalized_rmse(y_true, y_pred)

        self.assertAlmostEqual(rmse, np.sqrt(38.0 / 3.0))
        self.assertEqual(mean_sr, 20.0)
        self.assertAlmostEqual(nrmse, rmse / mean_sr)

    def test_rejects_invalid_within_bin_denominator(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            normalized_rmse(np.array([-1.0, 1.0]), np.array([0.0, 0.0]))

    def test_rejects_mismatched_empty_or_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "same non-empty shape"):
            normalized_rmse(np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "same non-empty shape"):
            normalized_rmse(np.array([]), np.array([]))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            normalized_rmse(np.array([1.0]), np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()

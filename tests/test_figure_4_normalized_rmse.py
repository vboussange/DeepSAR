import unittest

import numpy as np

from figures.figure_4.figure_4 import normalized_rmse


class Figure4NormalizedRmseTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

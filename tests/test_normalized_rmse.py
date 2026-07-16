import unittest

import numpy as np

from muscari.utils import compute_metrics


class NormalizedRmseTest(unittest.TestCase):
    def test_mean_normalized_rmse_and_percentage_conversion(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])

        metrics = compute_metrics(y_true, y_pred)

        expected_rmse = np.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], expected_rmse)
        self.assertAlmostEqual(metrics["nrmse"], expected_rmse / 2.0)
        self.assertAlmostEqual(100.0 * metrics["nrmse"], 40.8248290463863)

    def test_inputs_are_flattened(self):
        flat = compute_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        column = compute_metrics(
            np.array([[1.0], [2.0]]),
            np.array([[1.5], [2.5]]),
        )
        self.assertEqual(flat, column)

    def test_rejects_empty_or_nonpositive_mean_target(self):
        with self.assertRaisesRegex(ValueError, "empty target"):
            compute_metrics(np.array([]), np.array([]))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            compute_metrics(np.array([-1.0, 1.0]), np.array([0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            compute_metrics(np.array([np.nan, 1.0]), np.array([0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

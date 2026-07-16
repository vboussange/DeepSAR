import unittest

import numpy as np

from scripts.chao2_estimator import estimate_sr


class Chao2EstimatorTest(unittest.TestCase):
    def test_zero_doubleton_correction_deduplicates_plot_incidences(self):
        species = {
            1: ["a", "a", "b"],
            2: ["a", "c"],
            3: ["a", "d"],
        }

        self.assertEqual(estimate_sr([1, 2, 3], species), 6.0)

    def test_doubleton_branch_uses_unique_sampling_units(self):
        species = {
            1: ["a", "b", "c"],
            2: ["a", "b"],
            3: ["a", "d"],
            4: ["a"],
        }

        self.assertEqual(estimate_sr([1, 1, 2, 3, 4], species), 5.5)

    def test_empty_incidence_data_returns_nan(self):
        self.assertTrue(np.isnan(estimate_sr([1, 2], {})))


if __name__ == "__main__":
    unittest.main()

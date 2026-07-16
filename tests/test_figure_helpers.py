import tempfile
import unittest
from pathlib import Path

import pandas as pd

from figures.SI.datasets.split_sample_counts import load_fold_counts, render_latex_table
from figures.figure_5.calculate_sar import coordinate_slice


class FigureHelperTest(unittest.TestCase):
    def test_fold_counts_reject_incomplete_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing fold 0 input files"):
                load_fold_counts(Path(tmpdir))

    def test_fold_count_table_contains_all_partitions(self):
        table = render_latex_table(
            pd.DataFrame([{"Fold": 0, "Train": 10, "Validation": 2, "Test": 3}])
        )
        self.assertIn("Fold & Train & Validation & Test", table)
        self.assertIn("0 & 10 & 2 & 3", table)

    def test_coordinate_slice_respects_axis_direction(self):
        ascending = coordinate_slice([0.0, 10.0], center=5.0, window_size=4.0)
        descending = coordinate_slice([10.0, 0.0], center=5.0, window_size=4.0)
        self.assertEqual((ascending.start, ascending.stop), (3.0, 7.0))
        self.assertEqual((descending.start, descending.stop), (7.0, 3.0))


if __name__ == "__main__":
    unittest.main()

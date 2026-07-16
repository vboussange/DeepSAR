"""Compatibility copy of the canonical three-variant Figure 4."""
from __future__ import annotations

import figure_4 as base


base.FIGURE_PATH = base.OUTPUT_DIR / "figure_4_variant.pdf"
base.PAPER_FIGURE_PATH = None
base.CSV_PATH = base.OUTPUT_DIR / "figure_4_variant_normalized_rmse_by_area.csv"


if __name__ == "__main__":
    base.main()

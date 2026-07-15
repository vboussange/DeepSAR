"""Variant of Figure 4 including the ClimateDEM + area model."""
from __future__ import annotations

import figure_4 as base


base.FIGURE_PATH = base.OUTPUT_DIR / "figure_4_variant.pdf"
base.PAPER_FIGURE_PATH = None
base.CSV_PATH = base.OUTPUT_DIR / "figure_4_variant_normalized_rmse_by_area.csv"
base.MODEL_SPECS = base.MODEL_SPECS + [
    {
        "name": "MuScaRi_ClimateDEM_Area",
        "label": "Climate + DEM + area",
        "config_hash": "ad74a3020281",
        "color": "#4cc9f0",
        "marker": "s",
    },
]


if __name__ == "__main__":
    base.main()

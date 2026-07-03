"""Plot EVA scale-binned relative RMSE for MuScaRi variants."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).parent
CSV_PATH = OUTPUT_DIR / "figure_4_relative_rmse_by_area.csv"
FIGURE_PATH = OUTPUT_DIR / "figure_4.pdf"

MODEL_SPECS = [
    {
        "name": "MuScaRi_Area",
        "label": "MuScaRi (area only)",
        "color": "#f72585",
        "marker": "o",
    },
    {
        "name": "MuScaRi_ClimateDEM",
        "label": "MuScaRi (env. only)",
        "color": "#3a0ca3",
        "marker": "^",
    },
    {
        "name": "MuScaRi_ClimateDEM_Area",
        "label": "MuScaRi (env. + area)",
        "color": "#4cc9f0",
        "marker": "s",
    },
]


def summarize_for_plot(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["model_name", "area_bin"], sort=True)
    summary = grouped.agg(
        area_center_km2=("area_center_km2", "mean"),
        relative_rmse_percent_mean=("relative_rmse_percent", "mean"),
        relative_rmse_percent_std=("relative_rmse_percent", "std"),
    )
    return summary.reset_index()


def validate_results(results: pd.DataFrame) -> None:
    expected_models = {spec["name"] for spec in MODEL_SPECS}
    observed_models = set(results["model_name"].unique())
    missing_models = expected_models - observed_models
    if missing_models:
        raise ValueError(f"Missing model rows in {CSV_PATH}: {sorted(missing_models)}")
    if results["relative_rmse_percent"].isna().any():
        raise ValueError("Relative RMSE contains missing values.")
    if (results["n_samples"] <= 0).any():
        raise ValueError("At least one model/fold/area bin has no samples.")


def plot_results(results: pd.DataFrame) -> None:
    summary = summarize_for_plot(results)
    fig, ax = plt.subplots(figsize=(4, 4))

    for spec in MODEL_SPECS:
        data = summary[summary["model_name"] == spec["name"]].sort_values("area_bin")
        x = data["area_center_km2"].to_numpy(dtype=float)
        y = data["relative_rmse_percent_mean"].to_numpy(dtype=float)
        y_std = data["relative_rmse_percent_std"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker=spec["marker"],
            markersize=4,
            linestyle="-",
            color=spec["color"],
            label=spec["label"],
            alpha=0.9,
        )
        ax.fill_between(
            x,
            np.maximum(y - y_std, 1e-9),
            y + y_std,
            alpha=0.18,
            color=spec["color"],
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Relative RMSE (%)")
    ax.set_xlabel(r"Spatial unit area, $A$ (km$^2$)")
    ax.legend(frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.22), loc="center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")


def main() -> None:
    results = pd.read_csv(CSV_PATH)
    validate_results(results)
    plot_results(results)
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()

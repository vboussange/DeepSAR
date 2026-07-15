"""Plot interpolation NRMSE against the proportion of training samples."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[3]
RESULTS_PATH = (
    ROOT
    / "scripts/results/training_fraction_scaling/ceacce0/training_fraction_scaling_results.csv"
)
FIGURE_PATH = Path(__file__).with_name("scaling_exp.pdf")
PAPER_FIGURE_PATH = ROOT / "paper/figures/SI/scaling_exp.pdf"


def main() -> None:
    results = pd.read_csv(RESULTS_PATH)
    required = {"train_frac", "fold", "interp_nrmse"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing scaling-result columns: {sorted(missing)}")
    if results.duplicated(["train_frac", "fold"]).any():
        raise ValueError("Duplicate training-fraction/fold rows.")

    fractions = np.sort(results["train_frac"].unique())
    fig, ax = plt.subplots(figsize=(4, 3))
    rng = np.random.default_rng(42)
    medians = []
    for fraction in fractions:
        values = 100.0 * results.loc[
            results["train_frac"] == fraction,
            "interp_nrmse",
        ].to_numpy(dtype=float)
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"Expected five finite folds for training fraction {fraction:g}.")
        medians.append(float(np.median(values)))
        jitter = np.exp(rng.normal(0.0, 0.035, size=len(values)))
        ax.scatter(fraction * jitter, values, alpha=0.6, s=10, color="#f72585", zorder=3)

    ax.plot(fractions, medians, "--", color="#f72585", linewidth=1, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training samples / available training plots")
    ax.set_ylabel("NRMSE (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_FIGURE_PATH, dpi=300, bbox_inches="tight")
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {PAPER_FIGURE_PATH}")


if __name__ == "__main__":
    main()

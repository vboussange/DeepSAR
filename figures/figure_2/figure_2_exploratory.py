"""
Plotting figure 2 'prediction power of climate, area, and both on SR'.
Simplified to performance-only plots using benchmark and Chao2 results.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    benchmark_path = Path(__file__).parents[2] / "scripts" / "results" / "benchmark" / "benchmark_results.csv"
    chao2_path = Path(__file__).parents[2] / "scripts" / "results" / "benchmark" / "benchmark_chao2_results.csv"

    df_bench = pd.read_csv(benchmark_path)
    df_bench = df_bench[df_bench["experiment"] != "DeepSAR_All_frac_0.01"]  # Remove low-data experiment
    df_chao2 = pd.read_csv(chao2_path)

    df_bench = df_bench.copy()
    df_bench["model"] = df_bench["experiment"]

    df_chao2 = df_chao2.copy()
    df_chao2["model"] = "chao2_estimator"

    metric = "rmse"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    datasets = ["interp", "extrap"]
    titles = ["Interpolation (SBCV test)", "Extrapolation (GIFT)"]
    colors = ["#f72585", "#4cc9f0"]
    axes = [ax1, ax2]

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):
        if dataset == "interp":
            df_plot = df_bench
        else:
            df_plot = pd.concat([df_chao2, df_bench], ignore_index=True)

        models = df_plot["model"].dropna().unique().tolist()
        if dataset == "extrap" and "chao2_estimator" not in models:
            models = ["chao2_estimator"] + models
        models = sorted(models, key=str.lower)

        metric_col = f"{dataset}_{metric}"
        box_data = []
        for model in models:
            data = df_plot[df_plot["model"] == model][metric_col].dropna().values
            box_data.append(data)

        bplot = ax.boxplot(box_data, patch_artist=True, widths=0.6, showfliers=False)

        color = colors[j]
        for i, data in enumerate(box_data):
            if len(data) == 0:
                continue
            x = np.random.normal(i + 1, 0.06, size=len(data))
            ax.scatter(x, data, alpha=0.6, s=10, color=color, zorder=3)

        for patch in bplot["boxes"]:
            patch.set_facecolor("none")
            patch.set_edgecolor("none")
        for item in ["caps", "whiskers"]:
            for element in bplot[item]:
                element.set_color("none")
        for element in bplot["medians"]:
            element.set_color("black")

        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(metric.upper()) if j == 0 else None
        ax.set_title(titles[j])

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    ax1.text(0.1, 0.1, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")
    ax2.text(0.1, 0.1, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches="tight")
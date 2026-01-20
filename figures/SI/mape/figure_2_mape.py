"""
Supplementary figure: MAPE performance for interpolation/extrapolation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



ROOT = Path(__file__).parents[2]
BENCHMARK_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_results.csv"
CHAO2_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_chao2_results.csv"


def load_benchmark_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_nw = pd.read_csv(BENCHMARK_RESULTS)
    df_chao2 = pd.read_csv(CHAO2_RESULTS)
    return df_nw, df_chao2


def add_performance_panels(
    df_deepsar: pd.DataFrame,
    df_chao2: pd.DataFrame,
    metric: str,
    label_map: dict[str, str] | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    datasets = ["interp", "extrap"]
    titles_main = [
        "Interpolation performance",
        "Extrapolation performance",
    ]
    titles_sub = [
        "(Spatial block cross evaluation with EVA dataset)",
        "(independent evaluation with GIFT dataset)",
    ]
    colors = ["#f72585", "#4cc9f0"]
    axes = [ax1, ax2]

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):
        if dataset == "interp":
            experiments = ["MLP_All", "DeepSAR_Area", "DeepSAR_ClimateDEM_Landcover", "DeepSAR_All"]
            df_plot = df_deepsar
        else:
            experiments = [
                "MLP_All",
                "chao2_estimator",
                "DeepSAR_Area",
                "DeepSAR_ClimateDEM_Landcover",
                "DeepSAR_All",
            ]
            df_plot = pd.concat([df_chao2, df_deepsar], ignore_index=True)

        box_data = []
        for experiment in experiments:
            exp_data = df_plot[df_plot["experiment"] == experiment]
            metric_col = f"{dataset}_{metric}"
            data = exp_data[metric_col].values
            box_data.append(data)

        bplot = ax.boxplot(box_data, patch_artist=True, widths=0.6, showfliers=False)

        color = colors[j]
        for i, data in enumerate(box_data):
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

        ax.set_xticks(range(1, len(experiments) + 1))
        display_labels = [label_map.get(e, e) if label_map else e for e in experiments]
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(f"{metric.upper()}") if j == 0 else None
        ax.set_title(titles_main[j], fontsize=12, fontweight="bold", pad=18)
        ax.text(
            0.5,
            1.02,
            titles_sub[j],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.3 * y_range)

        alpha = 0.05
        flat_data = []
        group_labels = []
        for i, data in enumerate(box_data):
            flat_data.extend(data)
            group_labels.extend([experiments[i]] * len(data))

        # if len(set(group_labels)) > 1:
        #     mc = MultiComparison(flat_data, group_labels)
        #     test_results = mc.allpairtest(stats.ttest_ind, alpha=alpha)

        #     comp_matrix = create_comp_matrix_allpair_t_test(test_results)
        #     letters = multcomp_letters(comp_matrix < alpha)

        #     for i, experiment in enumerate(experiments):
        #         if experiment in letters:
        #             data_vals = box_data[i]
        #             q75 = np.percentile(data_vals, 75)
        #             iqr = np.percentile(data_vals, 75) - np.percentile(data_vals, 25)
        #             whisker_top = q75 + 1.5 * iqr
        #             ypos = min(whisker_top, max(data_vals)) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

        #             median_val = np.median(data_vals)
        #             ax.text(
        #                 i + 0.7,
        #                 median_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
        #                 letters[experiment],
        #                 ha="left",
        #                 va="bottom",
        #                 fontsize=10,
        #                 color="black",
        #             )

    return fig, ax1, ax2


if __name__ == "__main__":
    df_deepsar, df_chao2 = load_benchmark_results()

    label_map = {
        "DeepSAR_Area": "DeepSAR\n(area only)",
        "DeepSAR_ClimateDEM_Landcover": "DeepSAR\n(environment\nonly)",
        "DeepSAR_All": "DeepSAR",
        "MLP_All": "MLP",
        "chao2_estimator": "Chao2 estimator",
    }

    df_deepsar = df_deepsar[df_deepsar["experiment"].isin(label_map)].copy()

    metric = "mape"
    fig, ax1, ax2 = add_performance_panels(df_deepsar, df_chao2, metric, label_map=label_map)

    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches="tight")


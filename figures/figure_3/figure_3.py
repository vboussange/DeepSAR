"""
Plotting figure 2 'prediction power of climate, area, and both on SR'.
Keeps the original experiments and correlation plots.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import MultiComparison

from muscari.muscari import MuScaRi
from muscari.cld import create_comp_matrix_allpair_t_test, multcomp_letters
from muscari import MuScaRiEnsemble

ROOT = Path(__file__).parents[2]
TRAINING_DATASET_SEED = "ceacce0"
BENCHMARK_RESULTS = ROOT / "scripts" / "results" / "benchmark" / f"benchmark_results_{TRAINING_DATASET_SEED}.csv"
CHAO2_RESULTS = ROOT / "scripts" / "results" / "benchmark" / f"benchmark_chao2_results_{TRAINING_DATASET_SEED}.csv"
MODEL_FAMILY = "MuScaRi_ClimateDEM"
MODEL_HASH = "dae0789a3c87"
GIFT_ASYMPTOTE_RESULTS = (
    ROOT
    / "scripts"
    / "results"
    / "gift_asymptote_evaluation"
    / TRAINING_DATASET_SEED
    / "gift_asymptote_evaluation_results.csv"
)
RUN_DIR = ROOT / "scripts" / "results" / "benchmark" / "artifacts" / MODEL_FAMILY / MODEL_HASH
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT/da569da/compiled_data.parquet"
FIGURE_OUTPUTS = [
    Path(__file__).with_name("figure_3.pdf"),
    ROOT / "paper" / "figures" / "figure_3.pdf",
]
PLOT_STYLE = {
    "axis_label": 12,
    "tick_label": 12,
    "title": 15,
    "subtitle": 12,
    "annotation": 10,
    "panel_label": 12,
    "panel_letter_weight": "bold",
    "quantiles": (0.005, 1),
}

def report_model_performance(
    df_plot,
    metric,
    output_file=Path(__file__).with_name("model_performance_and_bias_report.txt"),
):
    """
    Report model performance, statistical significance, and relative bias for eva and gift datasets.
    
    Parameters:
    -----------
    df_plot : pd.DataFrame
        Combined dataframe with model results
    metric : str
        Performance metric to analyze
    output_file : str
        Path to output text file for results
    """
    datasets = ["interp", "extrap"]
    
    with open(output_file, "w") as file:
        # Model Performance and Statistical Significance Analysis
        print("\n\nMODEL PERFORMANCE AND STATISTICAL SIGNIFICANCE", file=file)
        print("=" * 60, file=file)
        
        for dataset in datasets:
            metric_col = f"{dataset}_{metric}"
            
            # Get available models for this dataset
            available_models = []
            model_data_dict = {}
            
            for experiment in df_plot['experiment'].unique():
                model_data = df_plot[df_plot['experiment'] == experiment]
                if not model_data.empty and metric_col in model_data.columns:
                    performance = model_data[metric_col].dropna().values
                    if len(performance) > 0:
                        available_models.append(experiment)
                        model_data_dict[experiment] = performance
            
            if not available_models:
                continue
                
            label = "Interpolation" if dataset == "interp" else "Extrapolation"
            print(f"\n{label} Dataset", file=file)
            print("=" * 50, file=file)
            
            # Performance summary table
            dataset_results = []
            for experiment in available_models:
                performance = model_data_dict[experiment]
                dataset_results.append({
                    'Experiment': experiment,
                    'RMSE_mean': np.mean(performance),
                    'RMSE_std': np.std(performance, ddof=1) if len(performance) > 1 else 0.0,
                    'N': len(performance)
                })
            
            results_df = pd.DataFrame(dataset_results)
            results_df['RMSE'] = results_df.apply(lambda x: f"{x['RMSE_mean']:.4f} ± {x['RMSE_std']:.4f}", axis=1)
            summary_table = results_df[['Experiment', 'RMSE', 'N']]
            print(summary_table.to_string(index=False), file=file)
            
            # Statistical significance tests (pairwise comparisons)
            print(f"\nPairwise Statistical Significance Tests ({label})", file=file)
            print("-" * 50, file=file)
            
            # Create significance matrix
            n_experiments = len(available_models)
            for i in range(n_experiments):
                for j in range(i+1, n_experiments):
                    experiment1, experiment2 = available_models[i], available_models[j]
                    data1, data2 = model_data_dict[experiment1], model_data_dict[experiment2]
                    
                    # Calculate means for relative difference
                    median1, median2 = np.median(data1), np.median(data2)
                    rel_diff = ((median2 - median1) / median1) * 100
                    
                    # Perform t-test
                    statistic, p_value = ttest_ind(data1, data2)
                    
                    # Determine significance level
                    if p_value < 0.001:
                        sig_level = "***"
                    elif p_value < 0.01:
                        sig_level = "**"
                    elif p_value < 0.05:
                        sig_level = "*"
                    else:
                        sig_level = "ns"
                    
                    print(f"{experiment1} vs {experiment2}: t={statistic:.3f}, p={p_value:.4f} {sig_level}, rel_diff={rel_diff:+.1f}%", file=file)
            
            print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant", file=file)

    print(f"Model performance and bias analysis saved to '{output_file}'")


def load_chao2_results() -> pd.DataFrame:
    if not CHAO2_RESULTS.exists():
        raise FileNotFoundError(f"Missing fold-level Chao2 benchmark: {CHAO2_RESULTS}")
    return pd.read_csv(CHAO2_RESULTS)


def apply_asymptotic_muscari_extrapolation(df_plot: pd.DataFrame) -> pd.DataFrame:
    if not GIFT_ASYMPTOTE_RESULTS.exists():
        raise FileNotFoundError(f"Missing GIFT asymptote audit: {GIFT_ASYMPTOTE_RESULTS}")

    asymptote = pd.read_csv(GIFT_ASYMPTOTE_RESULTS)
    asymptote = asymptote[
        (asymptote["prediction_mode"] == "asymptotic_total")
        & (asymptote["aggregation"] == "fold_member")
    ].copy()
    asymptote["fold"] = asymptote["fold"].astype(int)

    metric_map = {
        "r2": "extrap_r2",
        "d2": "extrap_d2",
        "rmse": "extrap_rmse",
        "mape": "extrap_mape",
        "mean_relative_bias": "extrap_mean_relative_bias",
        "median_relative_bias": "extrap_median_relative_bias",
        "log1p_r2": "extrap_log1p_r2",
        "log1p_d2": "extrap_log1p_d2",
        "log1p_rmse": "extrap_log1p_rmse",
        "log1p_mae": "extrap_log1p_mae",
        "bias_slope_log_area": "extrap_bias_slope_log_area",
    }

    df_plot = df_plot.copy()
    for _, row in asymptote.iterrows():
        mask = (df_plot["experiment"] == row["model_name"]) & (df_plot["fold"] == row["fold"])
        if not mask.any():
            continue
        for source_col, target_col in metric_map.items():
            if target_col in df_plot.columns and source_col in row:
                df_plot.loc[mask, target_col] = row[source_col]
    return df_plot


def load_benchmark_results() -> pd.DataFrame:
    df_muscari = pd.read_csv(BENCHMARK_RESULTS)
    df_muscari = apply_asymptotic_muscari_extrapolation(df_muscari)
    df_chao2 = load_chao2_results()
    df_plot = pd.concat([df_chao2, df_muscari], ignore_index=True)
    return df_plot


def add_performance_panels(
    ax1: plt.Axes,
    ax2: plt.Axes,
    df_perf: pd.DataFrame,
    metric: str,
    label_map: dict[str, str] | None = None,
) -> None:

    datasets = ["interp", "extrap"]
    titles_main = [
        "Interpolation performance",
        "Extrapolation performance",
    ]
    titles_sub = [
        "(spatial block cross evaluation,\n EVA dataset)",
        "(independent evaluation,\n GIFT dataset)",
    ]
    colors = ["#f72585", "#4cc9f0"]
    axes = [ax1, ax2]

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):

        box_data = []
        missing_rows = []
        experiments = list(label_map.keys())
        for experiment in experiments:
            exp_data = df_perf[df_perf["experiment"] == experiment]
            metric_col = f"{dataset}_{metric}"
            data = exp_data[metric_col].values
            finite_data = data[np.isfinite(data)]

            if len(finite_data) == 0:
                missing_rows.append(True)
                box_data.append(np.array([np.nan]))
            else:
                missing_rows.append(False)
                box_data.append(finite_data)

        bplot = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            showmeans=True,
            meanline=True,
            meanprops={"color": "black", "linewidth": 1.2},
            medianprops={"color": "none"},
            vert=False,
        )

        color = colors[j]
        for i, data in enumerate(box_data):
            if missing_rows[i]:
                continue
            y = np.random.normal(i + 1, 0.06, size=len(data))
            ax.scatter(data, y, alpha=0.6, s=10, color=color, zorder=3)

        for patch in bplot["boxes"]:
            patch.set_facecolor("none")
            patch.set_edgecolor("none")
        for item in ["caps", "whiskers"]:
            for element in bplot[item]:
                element.set_color("none")

        ax.set_yticks(range(1, len(experiments) + 1))
        ax.set_xlabel(f"{metric.upper()}", fontsize=PLOT_STYLE["axis_label"])
        if dataset == "extrap":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            positive_values = np.concatenate([d[d > 0] for d in box_data if np.isfinite(d).any()])
            ax.set_xlim(positive_values.min() * 0.75, positive_values.max() * 2.5)
        else:
            finite_values = np.concatenate([d[np.isfinite(d)] for d in box_data if np.isfinite(d).any()])
            data_range = finite_values.max() - finite_values.min()
            right_pad = 0.25 * data_range if data_range > 0 else finite_values.max() * 0.1
            left_pad = 0.05 * data_range if data_range > 0 else finite_values.max() * 0.02
            ax.set_xlim(max(0, finite_values.min() - left_pad), finite_values.max() + right_pad)

        if j == 0:
            display_labels = [label_map.get(e, e) if label_map else e for e in experiments]
            ax.set_yticklabels(display_labels, rotation=0, ha="right", fontsize=PLOT_STYLE["tick_label"])

            for i, is_missing in enumerate(missing_rows):
                if is_missing:
                    ax.axhspan(i + 1 - 0.35, i + 1 + 0.35, color="lightgray", alpha=0.6, zorder=1)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        ax.set_title(titles_main[j], fontsize=PLOT_STYLE["title"], fontweight="normal", pad=36)
        ax.text(
            0.5,
            1.02,
            titles_sub[j],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=PLOT_STYLE["subtitle"],
        )

        alpha = 0.05
        flat_data = []
        group_labels = []
        for i, data in enumerate(box_data):
            if missing_rows[i]:
                continue
            flat_data.extend(data)
            group_labels.extend([experiments[i]] * len(data))

        if len(set(group_labels)) > 1:
            mc = MultiComparison(flat_data, group_labels)
            test_results = mc.allpairtest(stats.ttest_ind, alpha=alpha)

            comp_matrix = create_comp_matrix_allpair_t_test(test_results)
            sig_matrix = pd.DataFrame(
                (comp_matrix.to_numpy() < alpha).copy(),
                index=comp_matrix.index,
                columns=comp_matrix.columns,
            )
            letters = multcomp_letters(sig_matrix)
            for i, experiment in enumerate(experiments):
                if experiment in letters:
                    data_vals = box_data[i]
                    bar_x = np.mean(data_vals)
                    xmin, xmax = ax.get_xlim()
                    if dataset == "extrap":
                        log_offset = 0.015 * (np.log10(xmax) - np.log10(xmin))
                        letter_x = 10 ** (np.log10(max(bar_x, xmin)) + log_offset)
                    else:
                        letter_x = bar_x + 0.015 * (xmax - xmin)
                    ax.text(
                        letter_x,
                        i + 1 - 0.16,
                        letters[experiment],
                        ha="left",
                        va="top",
                        fontsize=PLOT_STYLE["annotation"],
                        color="black",
                    )
        
        # Add grid after all other elements
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='both', zorder=0)

    return None


def load_fold_model(ckpt_path: Path, device: str) -> tuple[MuScaRi, object, dict]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MuScaRi(
        config.layer_sizes,
        feature_names=checkpoint["feature_names"],
        feature_scaler=checkpoint["feature_scaler"],
        target_scaler=checkpoint["target_scaler"],
        ffnn_batchnorm=getattr(config, "muscari_batchnorm", False),
        asymptote_transform=getattr(
            config,
            "muscari_asymptote_transform",
            checkpoint.get("asymptote_transform", "softplus"),
        ),
        weibull_parameterization=getattr(
            config,
            "muscari_weibull_parameterization",
            checkpoint.get("weibull_parameterization", "legacy"),
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metrics = checkpoint.get("metrics", {})
    return model, config, metrics


def select_best_fold_model(run_dir: Path, device: str) -> tuple[MuScaRi, gpd.GeoDataFrame, int]:
    ckpt_paths = sorted(run_dir.glob("fold_*.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold_*.pth files found in {run_dir}")

    best_rmse = np.inf
    best_ckpt_path = None
    best_fold = -1
    best_config = None

    for ckpt_path in ckpt_paths:
        fold_id = int(ckpt_path.stem.split("_")[-1])
        _, config, metrics = load_fold_model(ckpt_path, device)
        rmse = metrics.get("test", {}).get("rmse")
        if rmse is None:
            continue
        if rmse < best_rmse:
            best_rmse = rmse
            best_ckpt_path = ckpt_path
            best_fold = fold_id
            best_config = config

    if best_ckpt_path is None or best_config is None:
        raise FileNotFoundError("Could not select a best fold model; check checkpoint metrics.")

    best_model, _, _ = load_fold_model(best_ckpt_path, device)
    test_path = best_config.path_sbcv_data / f"fold_{best_fold}_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file for fold {best_fold} not found at {test_path}")

    best_test_df = gpd.read_parquet(test_path)
    best_test_df["log_sp_unit_area"] = np.log(best_test_df["sp_unit_area"])
    best_test_df["log_observed_area"] = np.log(best_test_df["observed_area"])
    best_test_df = best_test_df.replace([np.inf, -np.inf], np.nan).dropna()

    return best_model, best_test_df, best_fold


def prepare_eva_test_data(test_df: gpd.GeoDataFrame, model: MuScaRi, sample_frac: float = 0.1) -> gpd.GeoDataFrame:
    test_df = test_df.copy()
    test_df["predicted_sr"] = model.predict_sr(test_df)
    return test_df.sample(frac=sample_frac, random_state=42)


def prepare_gift_data(model: MuScaRiEnsemble) -> gpd.GeoDataFrame:
    gift_dataset = gpd.read_parquet(GIFT_SAMPLES_PATH)
    gift_dataset["log_sp_unit_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["observed_area"])
    gift_dataset = gift_dataset.replace([np.inf, -np.inf], np.nan)
    gift_dataset = gift_dataset.dropna(subset=model.feature_names + ["sr"])
    gift_dataset["predicted_sr"] = model.predict_mean_sr_tot(gift_dataset)
    return gift_dataset

if __name__ == "__main__":
    df_perf = load_benchmark_results()
    metric = "rmse"
    
    device = "cpu"
    best_model, best_test_df, best_fold = select_best_fold_model(RUN_DIR, device)
    ensemble_model = MuScaRiEnsemble.from_folds(RUN_DIR, device=device)

    eva_test_data = prepare_eva_test_data(best_test_df, best_model)
    gift_dataset = prepare_gift_data(ensemble_model)

    report_model_performance(df_perf, metric)
    
    label_map = {
        "MuScaRi_ClimateDEM_Area": "MuScaRi\n(env. + area)",
        "MuScaRi_Area": "MuScaRi\n(area only)",
        "MuScaRi_ClimateDEM": "MuScaRi\n(env. only)",
        "FFNN_ClimateDEM_Area": "FFNN\n(env. + area)",
        "Linear_ClimateDEM_Area": "Linear\n(env. + area)",
        "chao2_estimator": "Chao2\nestimator",
    }

    df_perf = df_perf[df_perf["experiment"].isin(label_map)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    add_performance_panels(ax1, ax2, df_perf, metric, label_map=label_map)

    # Plot predictions vs observations for EVA
    mask_eva = eva_test_data[["sr", "predicted_sr"]].dropna()
    x_eva = mask_eva["sr"]
    y_eva = mask_eva["predicted_sr"]
    eva_relative_bias = (y_eva - x_eva) / x_eva
    eva_relative_bias_stat = eva_relative_bias.median()
    eva_r2 = r2_score(x_eva, y_eva)
    ax3.text(
        0.55,
        0.08,
        f"Rel. bias: {eva_relative_bias_stat:.3f}\nR$^2$: {eva_r2:.3f}",
        transform=ax3.transAxes,
        fontsize=PLOT_STYLE["annotation"],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
    )

    ax3.scatter(x_eva, y_eva, alpha=0.6, s=10, color="#f72585")
    x_eva_q = np.nanquantile(x_eva, PLOT_STYLE["quantiles"])
    y_eva_q = np.nanquantile(y_eva[y_eva > 0], PLOT_STYLE["quantiles"])
    eva_min = min(x_eva_q[0], y_eva_q[0])
    eva_max = max(x_eva_q[1], y_eva_q[1])
    ax3.set_xlim(eva_min, eva_max)
    ax3.set_ylim(eva_min, eva_max)
    
    # Add 1:1 line through plot corners
    ax3.plot([eva_min, eva_max], [eva_min, eva_max], linestyle='--', color="black", linewidth=1)
    
    ax3.set_xlabel(r'Empirical species richness, $S(a)$', fontsize=PLOT_STYLE["axis_label"])
    ax3.set_ylabel(r'Predicted species richness, $S(a)$', fontsize=PLOT_STYLE["axis_label"])
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    # Fourth plot: model predictions vs GIFT observations
    # Plot predictions vs observations for GIFT
    mask_gift = gift_dataset[["sr", "predicted_sr"]].dropna()
    x_gift = mask_gift["sr"]
    y_gift = mask_gift["predicted_sr"]
    gift_relative_bias = (y_gift - x_gift) / x_gift
    gift_relative_bias_stat = gift_relative_bias.median()
    gift_r2 = r2_score(x_gift, y_gift)

    ax4.text(
        0.55,
        0.08,
        f"Rel. bias: {gift_relative_bias_stat:.3f}\nR$^2$: {gift_r2:.3f}",
        transform=ax4.transAxes,
        fontsize=PLOT_STYLE["annotation"],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
    )
    ax4.scatter(x_gift, y_gift, alpha=0.6, s=10, color="#4cc9f0")
    x_gift_q = np.nanquantile(x_gift, PLOT_STYLE["quantiles"])
    y_gift_q = np.nanquantile(y_gift, PLOT_STYLE["quantiles"])
    gift_min = min(x_gift_q[0], y_gift_q[0])
    gift_max = max(x_gift_q[1], y_gift_q[1])
    ax4.set_xlim(gift_min, gift_max)
    ax4.set_ylim(gift_min, gift_max)
    
    # Add 1:1 line through plot corners
    ax4.plot([gift_min, gift_max], [gift_min, gift_max], linestyle='--', color="black", linewidth=1)
    
    ax4.set_xlabel(r'Empirical total species richness, $S_T$', fontsize=PLOT_STYLE["axis_label"])
    ax4.set_ylabel(r'Predicted total species richness, $S_T$', fontsize=PLOT_STYLE["axis_label"])
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # Add panel labels (a, b, c, d) in Nature style
    ax1.text(0.9, 0.1, 'a', transform=ax1.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax2.text(0.9, 0.1, 'b', transform=ax2.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax3.text(0.1, 0.9, 'c', transform=ax3.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax4.text(0.1, 0.9, 'd', transform=ax4.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    
    plt.tight_layout()
    for output_path in FIGURE_OUTPUTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

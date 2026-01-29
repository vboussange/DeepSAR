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

import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import MultiComparison

from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.cld import create_comp_matrix_allpair_t_test, multcomp_letters
from deepsar.utils import load_ensemble_from_folds

ROOT = Path(__file__).parents[2]
BENCHMARK_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_results_95c85d6.csv"
CHAO2_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_chao2_results.csv"
RUN_DIR = ROOT / "scripts" / "results" / "train" / "95c85d6_no_lc_features"
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT/1cb3898/compiled_data.parquet"
PLOT_STYLE = {
    "axis_label": 12,
    "tick_label": 12,
    "title": 15,
    "subtitle": 12,
    "annotation": 10,
    "panel_label": 12,
    "panel_letter_weight": "bold",
    "quantiles": (0.005, 0.995),
}

def report_model_performance_and_bias(df_plot, eva_test_data, gift_dataset, metric, output_file="model_performance_and_bias_report.txt"):
    """
    Report model performance, statistical significance, and relative bias for eva and gift datasets.
    
    Parameters:
    -----------
    df_plot : pd.DataFrame
        Combined dataframe with model results
    eva_test_data : pd.DataFrame
        EVA test dataset with observed and predicted SR values
    gift_dataset : pd.DataFrame
        GIFT dataset with observed and predicted SR values
    metric : str
        Performance metric to analyze
    output_file : str
        Path to output text file for results
    """
    datasets = ["interp", "extrap"]
    
    with open(output_file, "w") as file:
        print("Relative bias calculated as (predicted - observed) / observed", file=file)
        print("Positive values indicate model overestimation, negative values indicate underestimation\n", file=file)

        # Relative Bias Analysis
        print("RELATIVE BIAS ANALYSIS", file=file)
        print("=" * 50, file=file)
        
        # EVA dataset bias
        eva_mask = eva_test_data[["sr", "predicted_sr"]].dropna()
        eva_observed = eva_mask["sr"]
        eva_predicted = eva_mask["predicted_sr"]
        eva_relative_bias = (eva_predicted - eva_observed) / eva_observed
        
        print("EVA Dataset", file=file)
        print("-" * 20, file=file)
        print(f"Mean relative bias: {eva_relative_bias.mean():.4f}", file=file)
        print(f"Median relative bias: {eva_relative_bias.median():.4f}", file=file)
        print(f"Std relative bias: {eva_relative_bias.std():.4f}", file=file)
        print(f"Min relative bias: {eva_relative_bias.min():.4f}", file=file)
        print(f"Max relative bias: {eva_relative_bias.max():.4f}", file=file)
        print(f"N observations: {len(eva_relative_bias)}", file=file)
        
        # GIFT dataset bias
        gift_mask = gift_dataset[["sr", "predicted_sr"]].dropna()
        gift_observed = gift_mask["sr"]
        gift_predicted = gift_mask["predicted_sr"]
        gift_relative_bias = (gift_predicted - gift_observed) / gift_observed
        
        print("\nGIFT Dataset", file=file)
        print("-" * 20, file=file)
        print(f"Mean relative bias: {gift_relative_bias.mean():.4f}", file=file)
        print(f"Median relative bias: {gift_relative_bias.median():.4f}", file=file)
        print(f"Std relative bias: {gift_relative_bias.std():.4f}", file=file)
        print(f"Min relative bias: {gift_relative_bias.min():.4f}", file=file)
        print(f"Max relative bias: {gift_relative_bias.max():.4f}", file=file)
        print(f"N observations: {len(gift_relative_bias)}", file=file)
        
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
                    'RMSE_std': np.std(performance),
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


def load_benchmark_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_nw = pd.read_csv(BENCHMARK_RESULTS)
    df_chao2 = pd.read_csv(CHAO2_RESULTS)
    return df_nw, df_chao2


def add_performance_panels(
    ax1: plt.Axes,
    ax2: plt.Axes,
    df_deepsar: pd.DataFrame,
    df_chao2: pd.DataFrame,
    metric: str,
    label_map: dict[str, str] | None = None,
) -> None:

    datasets = ["interp", "extrap"]
    titles_main = [
        "Interpolation performance",
        "Extrapolation performance",
    ]
    titles_sub = [
        "(spatial block cross evaluation with EVA dataset)",
        "(independent evaluation with GIFT dataset)",
    ]
    colors = ["#f72585", "#4cc9f0"]
    axes = [ax1, ax2]

    for j, (dataset, ax) in enumerate(zip(datasets, axes)):
        if dataset == "interp":
            experiments = ["DeepSAR_Area", "DeepSAR_ClimateDEM_Landcover", "DeepSAR_All", "MLP_All"]
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

        bplot = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            showmeans=True,
            meanline=True,
            meanprops={"color": "black", "linewidth": 1.2},
            medianprops={"color": "none"},
        )

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

        ax.set_xticks(range(1, len(experiments) + 1))
        display_labels = [label_map.get(e, e) if label_map else e for e in experiments]
        ax.set_xticklabels(display_labels, rotation=45, ha="center", fontsize=PLOT_STYLE["tick_label"])
        ax.set_ylabel(f"{metric.upper()}", fontsize=PLOT_STYLE["axis_label"]) if j == 0 else None
        ax.set_title(titles_main[j], fontsize=PLOT_STYLE["title"], fontweight="bold", pad=24)
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
            flat_data.extend(data)
            group_labels.extend([experiments[i]] * len(data))

        if len(set(group_labels)) > 1:
            mc = MultiComparison(flat_data, group_labels)
            test_results = mc.allpairtest(stats.ttest_ind, alpha=alpha)

            comp_matrix = create_comp_matrix_allpair_t_test(test_results)
            letters = multcomp_letters(comp_matrix < alpha)

            for i, experiment in enumerate(experiments):
                if experiment in letters:
                    data_vals = box_data[i]
                    q75 = np.percentile(data_vals, 75)
                    iqr = np.percentile(data_vals, 75) - np.percentile(data_vals, 25)
                    whisker_top = q75 + 1.5 * iqr
                    ypos = min(whisker_top, max(data_vals)) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

                    mean_val = np.mean(data_vals)
                    ax.text(
                        i + 0.7,
                        mean_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                        letters[experiment],
                        ha="left",
                        va="bottom",
                        fontsize=PLOT_STYLE["annotation"],
                        color="black",
                    )

    return None


def load_fold_model(ckpt_path: Path, device: str) -> tuple[Deep4PWeibull, object, dict]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Deep4PWeibull(
        config.layer_sizes,
        feature_names=checkpoint["feature_names"],
        feature_scaler=checkpoint["feature_scaler"],
        target_scaler=checkpoint["target_scaler"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metrics = checkpoint.get("metrics", {})
    return model, config, metrics


def select_best_fold_model(run_dir: Path, device: str) -> tuple[Deep4PWeibull, gpd.GeoDataFrame, int]:
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


def prepare_eva_test_data(test_df: gpd.GeoDataFrame, model: Deep4PWeibull, sample_frac: float = 0.1) -> gpd.GeoDataFrame:
    test_df = test_df.copy()
    test_df["predicted_sr"] = model.predict_sr(test_df)
    return test_df.sample(frac=sample_frac, random_state=42)


def prepare_gift_data(model: Deep4PWeibull) -> gpd.GeoDataFrame:
    gift_dataset = gpd.read_parquet(GIFT_SAMPLES_PATH)
    gift_dataset["log_sp_unit_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["observed_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    gift_dataset["predicted_sr"] = model.predict_mean_sr_tot(gift_dataset)
    return gift_dataset

if __name__ == "__main__":
    df_deepsar, df_chao2 = load_benchmark_results()

    label_map = {
        "DeepSAR_Area": "DeepSAR\n(area only)",
        "DeepSAR_ClimateDEM_Landcover": "DeepSAR\n(env. only)",
        "DeepSAR_All": "DeepSAR",
        "MLP_All": "MLP",
        "chao2_estimator": "Chao2\nestimator",
    }

    df_deepsar = df_deepsar[df_deepsar["experiment"].isin(label_map)].copy()

    metric = "rmse"
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    add_performance_panels(ax1, ax2, df_deepsar, df_chao2, metric, label_map=label_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model, best_test_df, best_fold = select_best_fold_model(RUN_DIR, device)
    ensemble_model = load_ensemble_from_folds(RUN_DIR, device=device)

    eva_test_data = prepare_eva_test_data(best_test_df, best_model)

    # Plot predictions vs observations for EVA
    mask_eva = eva_test_data[["sr", "predicted_sr"]].dropna()
    x_eva = mask_eva["sr"]
    y_eva = mask_eva["predicted_sr"]
    eva_relative_bias = (y_eva - x_eva) / x_eva
    eva_median_bias = eva_relative_bias.median()
    eva_r2 = r2_score(x_eva, y_eva)
    ax3.text(
        0.55,
        0.08,
        f"Rel. bias: {eva_median_bias:.3f}\nR$^2$: {eva_r2:.3f}",
        transform=ax3.transAxes,
        fontsize=PLOT_STYLE["annotation"],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
    )

    ax3.scatter(x_eva, y_eva, alpha=0.6, s=10, color="#f72585")
    x_eva_q = np.nanquantile(x_eva, PLOT_STYLE["quantiles"])
    y_eva_q = np.nanquantile(y_eva, PLOT_STYLE["quantiles"])
    eva_min = min(x_eva_q[0], y_eva_q[0])
    eva_max = max(x_eva_q[1], y_eva_q[1])
    ax3.set_xlim(eva_min, eva_max)
    ax3.set_ylim(eva_min, eva_max)
    
    # Add 1:1 line through plot corners
    ax3.plot([eva_min, eva_max], [eva_min, eva_max], linestyle='--', color="black", linewidth=1)
    
    ax3.set_xlabel('EVA observed SR', fontsize=PLOT_STYLE["axis_label"])
    ax3.set_ylabel('Predicted SR', fontsize=PLOT_STYLE["axis_label"])
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    # Fourth plot: model predictions vs GIFT observations
    gift_dataset = prepare_gift_data(ensemble_model)

    # Plot predictions vs observations for GIFT
    mask_gift = gift_dataset[["sr", "predicted_sr"]].dropna()
    x_gift = mask_gift["sr"]
    y_gift = mask_gift["predicted_sr"]
    gift_relative_bias = (y_gift - x_gift) / x_gift
    gift_median_bias = gift_relative_bias.median()
    gift_r2 = r2_score(x_gift, y_gift)

    ax4.text(
        0.55,
        0.08,
        f"Rel. bias: {gift_median_bias:.3f}\nR$^2$: {gift_r2:.3f}",
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
    
    ax4.set_xlabel('GIFT observed SR', fontsize=PLOT_STYLE["axis_label"])
    ax4.set_ylabel('Predicted SR', fontsize=PLOT_STYLE["axis_label"])
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # Add grid lines to ax1 and ax2
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add panel labels (a, b, c, d) in Nature style
    ax1.text(0.1, 0.1, 'a', transform=ax1.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax2.text(0.1, 0.1, 'b', transform=ax2.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax3.text(0.1, 0.9, 'c', transform=ax3.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    ax4.text(0.1, 0.9, 'd', transform=ax4.transAxes, fontsize=PLOT_STYLE["panel_label"], fontweight=PLOT_STYLE["panel_letter_weight"], va='top', ha='right')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{Path(__file__).stem}.pdf", dpi=300, bbox_inches='tight')

    report_model_performance_and_bias(df_deepsar, eva_test_data, gift_dataset, metric)
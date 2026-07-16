"""Generate Figure 3 performance comparisons and prediction diagnostics."""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import shutil

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests

from muscari.muscari import MuScaRi
from muscari.cld import multcomp_letters
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
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT/418c563/compiled_data.parquet"
PAIRWISE_RESULTS_OUTPUT = Path(__file__).with_name("model_performance_pairwise_results.csv")
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
ALPHA = 0.05
EXPECTED_FOLDS = tuple(range(5))
LABEL_MAP = {
    "MuScaRi_ClimateDEM_Area": "MuScaRi\n(env. + area)",
    "MuScaRi_Area": "MuScaRi\n(area only)",
    "MuScaRi_ClimateDEM": "MuScaRi\n(env. only)",
    "FFNN_ClimateDEM_Area": "FFNN\n(env. + area)",
    "Linear_ClimateDEM_Area": "Linear\n(env. + area)",
    "chao2_estimator": "Chao2\nestimator",
}


@dataclass(frozen=True)
class PairedComparisonAnalysis:
    endpoint: str
    metric_column: str
    fold_values: pd.DataFrame
    pairwise_results: pd.DataFrame
    adjusted_p_matrix: pd.DataFrame
    rejection_matrix: pd.DataFrame
    letters: dict[str, str]


def _validated_fold_table(
    df: pd.DataFrame,
    metric_column: str,
    models: list[str],
    expected_folds: tuple[int, ...] = EXPECTED_FOLDS,
) -> pd.DataFrame:
    required_columns = {"experiment", "fold", metric_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    selected = df[df["experiment"].isin(models)].copy()
    absent_models = [model for model in models if model not in set(selected["experiment"])]
    if absent_models:
        raise ValueError(f"Requested models are absent: {absent_models}")

    duplicated = selected.duplicated(["experiment", "fold"], keep=False)
    if duplicated.any():
        duplicate_keys = (
            selected.loc[duplicated, ["experiment", "fold"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate model-fold rows: {duplicate_keys}")

    expected_fold_set = set(expected_folds)
    participating_models = []
    for model in models:
        model_rows = selected[selected["experiment"] == model]
        values = pd.to_numeric(model_rows[metric_column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.any():
            continue
        if not finite.all():
            raise ValueError(f"{model} has missing or non-finite {metric_column} values")

        model_folds = set(model_rows["fold"])
        if model_folds != expected_fold_set:
            raise ValueError(
                f"{model} has folds {sorted(model_folds)}, expected {sorted(expected_fold_set)}"
            )
        participating_models.append(model)

    if len(participating_models) < 2:
        raise ValueError(f"Fewer than two models have finite {metric_column} values")

    fold_table = selected[selected["experiment"].isin(participating_models)].pivot(
        index="fold", columns="experiment", values=metric_column
    )
    fold_table = fold_table.reindex(index=expected_folds, columns=participating_models)
    if not np.isfinite(fold_table.to_numpy(dtype=float)).all():
        raise ValueError(f"Could not construct a complete matched-fold table for {metric_column}")
    fold_table.index.name = "fold"
    return fold_table


def _adjusted_p_matrix(
    pairwise_results: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
    for row in pairwise_results.itertuples(index=False):
        matrix.loc[row.model, row.reference_model] = row.p_value_holm
        matrix.loc[row.reference_model, row.model] = row.p_value_holm
    return matrix


def paired_model_comparisons(
    df: pd.DataFrame,
    endpoint: str,
    models: list[str],
    metric: str = "nrmse_percent",
    expected_folds: tuple[int, ...] = EXPECTED_FOLDS,
    alpha: float = ALPHA,
) -> PairedComparisonAnalysis:
    """Compare models by matched cross-validation fold with panel-wise Holm correction.

    Differences use the sign convention ``model - reference_model``.
    """
    metric_column = f"{endpoint}_{metric}"
    fold_values = _validated_fold_table(df, metric_column, models, expected_folds)
    participating_models = list(fold_values.columns)
    fold_ids = ";".join(str(fold) for fold in fold_values.index)

    records = []
    for model, reference_model in combinations(participating_models, 2):
        model_values = fold_values[model].to_numpy(dtype=float)
        reference_values = fold_values[reference_model].to_numpy(dtype=float)
        differences = model_values - reference_values
        n = len(differences)
        df_t = n - 1
        statistic, p_value = stats.ttest_rel(model_values, reference_values)
        if not np.isfinite([statistic, p_value]).all():
            raise ValueError(
                f"Non-finite paired t-test for {model} and {reference_model} on {metric_column}"
            )

        mean_difference = differences.mean()
        standard_error = stats.sem(differences)
        ci_half_width = stats.t.ppf(0.975, df_t) * standard_error
        reference_mean = reference_values.mean()
        records.append(
            {
                "endpoint": endpoint,
                "metric_column": metric_column,
                "model": model,
                "reference_model": reference_model,
                "fold_ids": fold_ids,
                "n": n,
                "df": df_t,
                "model_mean": model_values.mean(),
                "reference_mean": reference_mean,
                "mean_paired_difference": mean_difference,
                "percentage_difference_vs_reference": 100 * mean_difference / reference_mean,
                "ci95_lower": mean_difference - ci_half_width,
                "ci95_upper": mean_difference + ci_half_width,
                "t_statistic": statistic,
                "p_value_raw": p_value,
            }
        )

    pairwise_results = pd.DataFrame.from_records(records)
    reject, p_values_holm, _, _ = multipletests(
        pairwise_results["p_value_raw"].to_numpy(), alpha=alpha, method="holm"
    )
    pairwise_results["p_value_holm"] = p_values_holm
    pairwise_results["reject_holm"] = reject
    pairwise_results["alpha"] = alpha

    adjusted_p_matrix = _adjusted_p_matrix(pairwise_results, participating_models)
    rejection_matrix = (adjusted_p_matrix < alpha).copy()
    for model in rejection_matrix.index:
        rejection_matrix.loc[model, model] = False
    if not np.array_equal(
        pairwise_results["reject_holm"].to_numpy(),
        pairwise_results["p_value_holm"].to_numpy() < alpha,
    ):
        raise AssertionError("Holm rejection decisions disagree with adjusted P values")
    letters = multcomp_letters(rejection_matrix)

    return PairedComparisonAnalysis(
        endpoint=endpoint,
        metric_column=metric_column,
        fold_values=fold_values,
        pairwise_results=pairwise_results,
        adjusted_p_matrix=adjusted_p_matrix,
        rejection_matrix=rejection_matrix,
        letters=letters,
    )


def analyze_performance_panels(
    df: pd.DataFrame,
    models: list[str],
    metric: str = "nrmse_percent",
) -> dict[str, PairedComparisonAnalysis]:
    return {
        endpoint: paired_model_comparisons(df, endpoint, models, metric=metric)
        for endpoint in ("interp", "extrap")
    }


def report_model_performance(
    analyses: dict[str, PairedComparisonAnalysis],
    output_file=Path(__file__).with_name("model_performance_and_bias_report.txt"),
):
    """Write the same paired results used to construct the figure letters."""
    output_file = Path(output_file)
    with output_file.open("w") as file:
        print("MODEL PERFORMANCE AND PAIRED STATISTICAL ANALYSIS", file=file)
        print("=" * 60, file=file)
        print(
            "Two-sided paired t-tests use matched fold-level NRMSE percentages. "
            "Differences are model - reference_model; Holm adjustment is applied "
            "separately within each panel.",
            file=file,
        )

        for endpoint, analysis in analyses.items():
            label = "Interpolation" if endpoint == "interp" else "Extrapolation"
            print(f"\n{label} Dataset", file=file)
            print("=" * 50, file=file)

            summary = pd.DataFrame(
                {
                    "Experiment": analysis.fold_values.columns,
                    "NRMSE_mean": analysis.fold_values.mean(axis=0).to_numpy(),
                    "NRMSE_std": analysis.fold_values.std(axis=0, ddof=1).to_numpy(),
                    "N": analysis.fold_values.count(axis=0).to_numpy(),
                }
            )
            summary["NRMSE (%)"] = summary.apply(
                lambda row: f"{row['NRMSE_mean']:.4f} ± {row['NRMSE_std']:.4f}", axis=1
            )
            print(summary[["Experiment", "NRMSE (%)", "N"]].to_string(index=False), file=file)
            print(
                f"\nPairwise paired t-tests "
                f"({len(analysis.pairwise_results)}-comparison Holm family)",
                file=file,
            )
            print("-" * 50, file=file)
            report_columns = [
                "model",
                "reference_model",
                "fold_ids",
                "n",
                "df",
                "mean_paired_difference",
                "percentage_difference_vs_reference",
                "ci95_lower",
                "ci95_upper",
                "t_statistic",
                "p_value_raw",
                "p_value_holm",
                "reject_holm",
            ]
            print(
                analysis.pairwise_results[report_columns].to_string(index=False, float_format="%.6g"),
                file=file,
            )
            print(f"\nCompact-letter display: {analysis.letters}", file=file)

    print(f"Paired model performance analysis saved to '{output_file}'")


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
        "nrmse": "extrap_nrmse",
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
    for endpoint in ("interp", "extrap"):
        df_plot[f"{endpoint}_nrmse_percent"] = 100.0 * df_plot[f"{endpoint}_nrmse"]
    return df_plot


def add_performance_panels(
    ax1: plt.Axes,
    ax2: plt.Axes,
    analyses: dict[str, PairedComparisonAnalysis],
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
        analysis = analyses[dataset]
        box_data = []
        missing_rows = []
        experiments = list(label_map.keys())
        rng = np.random.default_rng(42 + j)
        for experiment in experiments:
            if experiment not in analysis.fold_values.columns:
                missing_rows.append(True)
                box_data.append(np.array([np.nan]))
            else:
                missing_rows.append(False)
                box_data.append(analysis.fold_values[experiment].to_numpy(dtype=float))

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
            y = rng.normal(i + 1, 0.06, size=len(data))
            ax.scatter(data, y, alpha=0.6, s=10, color=color, zorder=3)

        for patch in bplot["boxes"]:
            patch.set_facecolor("none")
            patch.set_edgecolor("none")
        for item in ["caps", "whiskers"]:
            for element in bplot[item]:
                element.set_color("none")

        ax.set_yticks(range(1, len(experiments) + 1))
        metric_label = "NRMSE (%)" if metric == "nrmse_percent" else metric.upper()
        ax.set_xlabel(metric_label, fontsize=PLOT_STYLE["axis_label"])
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

        letters = analysis.letters
        if letters:
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
    metric = "nrmse_percent"
    df_perf = df_perf[df_perf["experiment"].isin(LABEL_MAP)].copy()
    analyses = analyze_performance_panels(df_perf, list(LABEL_MAP), metric=metric)
    pairwise_results = pd.concat(
        [analysis.pairwise_results for analysis in analyses.values()], ignore_index=True
    )
    pairwise_results.to_csv(PAIRWISE_RESULTS_OUTPUT, index=False)
    
    device = "cpu"
    best_model, best_test_df, best_fold = select_best_fold_model(RUN_DIR, device)
    ensemble_model = MuScaRiEnsemble.from_folds(RUN_DIR, device=device)

    eva_test_data = prepare_eva_test_data(best_test_df, best_model)
    gift_dataset = prepare_gift_data(ensemble_model)

    report_model_performance(analyses)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    add_performance_panels(ax1, ax2, analyses, metric, label_map=LABEL_MAP)

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
    fig.savefig(FIGURE_OUTPUTS[0], dpi=300, bbox_inches="tight")
    for output_path in FIGURE_OUTPUTS[1:]:
        shutil.copyfile(FIGURE_OUTPUTS[0], output_path)

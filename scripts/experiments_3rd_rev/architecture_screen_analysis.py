"""Summarize architecture screen results with tables and compact comparison plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
RESULTS_DIR = ROOT / "scripts/results/architecture_screen_stable_small" / SBCV_DATASET_ID
RESULTS_PATH = RESULTS_DIR / f"architecture_screen_results_{SBCV_DATASET_ID}.csv"

VARIANT_ORDER = [
    "softplus_abs",
    "exp_abs",
    "softplus_rel",
    "exp_rel",
    "stable_abs",
    "stable_coverage",
]
LABEL_MAP = {
    "softplus_abs": "Softplus asymptote + absolute effort",
    "exp_abs": "Exp asymptote + absolute effort",
    "softplus_rel": "Softplus asymptote + relative effort",
    "exp_rel": "Exp asymptote + relative effort",
    "stable_abs": "Stable Weibull + absolute effort + log target",
    "stable_coverage": "Stable Weibull + log coverage + log target",
}
EFFORT_LABELS = {
    "absolute": "Absolute",
    "relative": "Relative",
    "coverage": "Log coverage",
}
ASYMPTOTE_LABELS = {
    "softplus": "Softplus",
    "exp": "Exponential",
}
METRICS = ["rmse", "mape", "median_relative_bias", "r2", "d2", "bias_slope_log_area"]
PLOT_METRICS = [
    ("interp_rmse", "Interpolation RMSE", False),
    ("extrap_rmse", "GIFT RMSE", False),
    ("extrap_median_relative_bias", "GIFT median relative bias", True),
    ("extrap_bias_slope_log_area", "GIFT residual-bias slope", True),
]
COLORS = {
    "softplus_abs": "#1f77b4",
    "exp_abs": "#ff7f0e",
    "softplus_rel": "#2a9d8f",
    "exp_rel": "#e76f51",
    "stable_abs": "#6a4c93",
    "stable_coverage": "#8ac926",
}


def asymptote_label(exp_data: pd.DataFrame) -> str:
    if exp_data.get("weibull_parameterization", pd.Series(["legacy"])).iloc[0] == "stable":
        return "Stable span"
    return ASYMPTOTE_LABELS[exp_data["asymptote_transform"].iloc[0]]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["experiment"].isin(VARIANT_ORDER)].copy()
    df["experiment"] = pd.Categorical(df["experiment"], categories=VARIANT_ORDER, ordered=True)
    return df.sort_values(["experiment", "fold"])


def format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.3f} ± {std:.3f}"


def better_by_absolute_distance(metric_name: str) -> bool:
    return metric_name in {"median_relative_bias", "bias_slope_log_area"}


def higher_is_better(metric_name: str) -> bool:
    return metric_name in {"r2", "d2"}


def build_dataset_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for experiment in VARIANT_ORDER:
        exp_data = df[df["experiment"] == experiment]
        if exp_data.empty:
            continue

        row = {
            "_experiment": experiment,
            "Architecture": LABEL_MAP[experiment],
            "Effort": EFFORT_LABELS[exp_data["effort_transform"].iloc[0]],
            "Asymptote": asymptote_label(exp_data),
            "Folds": int(exp_data["fold"].nunique()),
        }
        for metric in METRICS:
            column = f"{dataset}_{metric}"
            values = exp_data[column].dropna().to_numpy()
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values)) if len(values) else np.nan
            row[metric] = format_mean_std(mean, std)
            row[f"_{metric}_mean"] = mean
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for metric in METRICS:
        mean_col = f"_{metric}_mean"
        valid = table[mean_col].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        if better_by_absolute_distance(metric):
            best_index = (table[mean_col].abs()).idxmin()
        elif higher_is_better(metric):
            best_index = table[mean_col].idxmax()
        else:
            best_index = table[mean_col].idxmin()
        table.at[best_index, metric] = f"\\textbf{{{table.at[best_index, metric]}}}"

    return table.drop(columns=[f"_{metric}_mean" for metric in METRICS] + ["_experiment"])


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    header = (
        "\\begin{table}\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\begin{tabularx}{\\textwidth}{l l l c >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}\n"
        "    \\toprule\n"
        "    Architecture & Effort & Asymptote & Folds & RMSE & MAPE & Rel. Bias & $R^2$ & $D^2$ & Bias slope \\\\\n"
        "    \\midrule\n"
    )
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "    "
            f"{row['Architecture']} & {row['Effort']} & {row['Asymptote']} & {row['Folds']} & "
            f"{row['rmse']} & {row['mape']} & {row['median_relative_bias']} & {row['r2']} & {row['d2']} & {row['bias_slope_log_area']} \\\\\n"
        )
    footer = (
        "    \\bottomrule\n"
        "    \\end{tabularx}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return header + "".join(rows) + footer


def save_tables(interp_table: pd.DataFrame, extrap_table: pd.DataFrame) -> None:
    interp_caption = (
        "Architecture screen results on the EVA interpolation folds (mean ± standard deviation across folds). "
        "Lower RMSE, MAPE, relative bias, and residual-bias slope magnitudes are better; higher $R^2$ and $D^2$ are better."
    )
    extrap_caption = (
        "Architecture screen results on the GIFT extrapolation dataset (mean ± standard deviation across folds). "
        "Lower RMSE, MAPE, relative bias, and residual-bias slope magnitudes are better; higher $R^2$ and $D^2$ are better."
    )

    (RESULTS_DIR / "architecture_screen_interp_summary.tex").write_text(
        render_latex_table(interp_table, interp_caption, "tab:architecture_screen_interp")
    )
    (RESULTS_DIR / "architecture_screen_extrap_summary.tex").write_text(
        render_latex_table(extrap_table, extrap_caption, "tab:architecture_screen_extrap")
    )
    interp_table.to_csv(RESULTS_DIR / "architecture_screen_interp_summary.csv", index=False)
    extrap_table.to_csv(RESULTS_DIR / "architecture_screen_extrap_summary.csv", index=False)


def bar_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, center_zero: bool) -> None:
    grouped = df.groupby("experiment", observed=True)[metric]
    means = grouped.mean().reindex(VARIANT_ORDER)
    stds = grouped.std().reindex(VARIANT_ORDER).fillna(0.0)
    y_positions = np.arange(len(VARIANT_ORDER))
    colors = [COLORS[name] for name in VARIANT_ORDER]

    ax.barh(
        y_positions,
        means.to_numpy(),
        xerr=stds.to_numpy(),
        height=0.68,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
    )

    rng = np.random.default_rng(42)
    for index, experiment in enumerate(VARIANT_ORDER):
        fold_values = df.loc[df["experiment"] == experiment, metric].dropna().to_numpy()
        if len(fold_values) == 0:
            continue
        jitter = rng.normal(0.0, 0.06, size=len(fold_values))
        ax.scatter(
            fold_values,
            np.full(len(fold_values), index) + jitter,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            s=28,
            zorder=3,
        )

    if center_zero:
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABEL_MAP[name] for name in VARIANT_ORDER], fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, (metric, title, center_zero) in zip(axes.flat, PLOT_METRICS):
        bar_panel(ax, df, metric, title, center_zero)

    axes[0, 0].text(0.02, 1.04, "a", transform=axes[0, 0].transAxes, fontsize=12, fontweight="bold")
    axes[0, 1].text(0.02, 1.04, "b", transform=axes[0, 1].transAxes, fontsize=12, fontweight="bold")
    axes[1, 0].text(0.02, 1.04, "c", transform=axes[1, 0].transAxes, fontsize=12, fontweight="bold")
    axes[1, 1].text(0.02, 1.04, "d", transform=axes[1, 1].transAxes, fontsize=12, fontweight="bold")

    figure_path = RESULTS_DIR / "architecture_screen_summary.pdf"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    ranking = (
        df.groupby("experiment", observed=True)
        .agg(
            interp_rmse=("interp_rmse", "mean"),
            extrap_rmse=("extrap_rmse", "mean"),
            extrap_bias_slope_log_area=("extrap_bias_slope_log_area", lambda s: np.mean(np.abs(s))),
        )
        .reindex(VARIANT_ORDER)
    )
    ranking.index = [LABEL_MAP[name] for name in ranking.index]
    print("Architecture ranking summary")
    print(ranking.sort_values(["extrap_rmse", "interp_rmse"]).round(3).to_string())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    interp_table = build_dataset_table(df, "interp")
    extrap_table = build_dataset_table(df, "extrap")
    save_tables(interp_table, extrap_table)
    save_figure(df)
    print_summary(df)
    print("Saved architecture analysis to", RESULTS_DIR)


if __name__ == "__main__":
    main()

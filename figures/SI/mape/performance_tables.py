"""Utilities for building LaTeX performance tables."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[3]
BENCHMARK_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_results.csv"
CHAO2_RESULTS = ROOT / "scripts" / "results" / "benchmark" / "benchmark_chao2_results.csv"


def load_benchmark_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_nw = pd.read_csv(BENCHMARK_RESULTS)
    df_chao2 = pd.read_csv(CHAO2_RESULTS)
    return df_nw, df_chao2


def format_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "-"
    return f"{mean:.3f} ± {std:.3f}"


def apply_best_bold(values: dict[str, float], higher_is_better: bool) -> dict[str, str]:
    valid_items = {k: v for k, v in values.items() if np.isfinite(v)}
    if not valid_items:
        return {k: "-" for k in values}
    best_value = max(valid_items.values()) if higher_is_better else min(valid_items.values())
    result = {}
    for key, value in values.items():
        if not np.isfinite(value):
            result[key] = "-"
        elif value == best_value:
            result[key] = "best"
        else:
            result[key] = ""
    return result


def build_performance_table(
    df_deepsar: pd.DataFrame,
    df_chao2: pd.DataFrame,
    dataset: str,
    label_map: dict[str, str],
    predictor_map: dict[str, str],
    metrics: list[str],
) -> pd.DataFrame:
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

    rows = []
    for experiment in experiments:
        row = {
            "model": label_map.get(experiment, experiment),
            "Predictors": predictor_map.get(experiment, "-"),
        }
        exp_data = df_plot[df_plot["experiment"] == experiment]
        for metric in metrics:
            metric_col = f"{dataset}_{metric}"
            values = exp_data[metric_col].dropna().values
            if len(values) == 0:
                mean_val = np.nan
                std_val = np.nan
            else:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
            row[metric] = format_mean_std(mean_val, std_val)
            row[f"_{metric}_mean"] = mean_val
        rows.append(row)

    df = pd.DataFrame(rows)

    best_flags = {}
    for metric in metrics:
        means = df.set_index("model")[f"_{metric}_mean"].to_dict()
        best_flags[metric] = apply_best_bold(
            means,
            higher_is_better=metric in {"r2", "d2"},
        )

    for metric in metrics:
        for idx, model_name in df["model"].items():
            if best_flags[metric].get(model_name) == "best":
                df.at[idx, metric] = f"\\textbf{{{df.at[idx, metric]}}}"

    df = df.drop(columns=[f"_{m}_mean" for m in metrics])
    return df


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l l c c c c",
    )
    return latex


def write_latex_table(df: pd.DataFrame, caption: str, label: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = render_latex_table(df, caption, label)
    output_path.write_text(latex)


if __name__ == "__main__":
    df_deepsar, df_chao2 = load_benchmark_results()

    label_map = {
        "DeepSAR_Area": "DeepSAR\n(area only)",
        "DeepSAR_ClimateDEM_Landcover": "DeepSAR\n(environment\nonly)",
        "DeepSAR_All": "DeepSAR",
        "MLP_All": "MLP",
        "chao2_estimator": "Chao2 estimator",
    }

    predictor_map = {
        "DeepSAR_Area": "Area",
        "DeepSAR_ClimateDEM_Landcover": "Environment",
        "DeepSAR_All": "Area + Environment",
        "MLP_All": "Area + Environment",
        "chao2_estimator": "Incidence (Chao2)",
    }

    df_deepsar = df_deepsar[df_deepsar["experiment"].isin(label_map)].copy()

    metrics = ["rmse", "mape", "r2", "d2"]
    interp_table = build_performance_table(
        df_deepsar,
        df_chao2,
        dataset="interp",
        label_map=label_map,
        predictor_map=predictor_map,
        metrics=metrics,
    )
    extrap_table = build_performance_table(
        df_deepsar,
        df_chao2,
        dataset="extrap",
        label_map=label_map,
        predictor_map=predictor_map,
        metrics=metrics,
    )

    interp_caption = "Interpolation performance (mean ± std across folds)."
    extrap_caption = "Extrapolation performance (mean ± std across folds)."
    interp_label = "tab:interp_performance"
    extrap_label = "tab:extrap_performance"

    interp_tex = render_latex_table(interp_table, caption=interp_caption, label=interp_label)
    extrap_tex = render_latex_table(extrap_table, caption=extrap_caption, label=extrap_label)
    print(interp_tex)
    print(extrap_tex)

    write_latex_table(
        interp_table,
        caption=interp_caption,
        label=interp_label,
        output_path=Path(__file__).with_name("interp_performance.tex"),
    )
    write_latex_table(
        extrap_table,
        caption=extrap_caption,
        label=extrap_label,
        output_path=Path(__file__).with_name("extrap_performance.tex"),
    )

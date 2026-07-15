"""Generate the four final-revision NRMSE performance tables."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[3]
BENCHMARK_DIR = ROOT / "scripts" / "results" / "benchmark"
GIFT_AUDIT_DIR = ROOT / "scripts" / "results" / "gift_asymptote_evaluation"
DATASET_IDS = ("ceacce0", "d0848f6")
CHAO2_RESULTS = BENCHMARK_DIR / "benchmark_chao2_results_ceacce0.csv"
MODEL_ORDER = [
    "FFNN_ClimateDEM_Area",
    "chao2_estimator",
    "Linear_ClimateDEM_Area",
    "MuScaRi_Area",
    "MuScaRi_ClimateDEM",
    "MuScaRi_ClimateDEM_Area",
]
LABEL_MAP = {
    "FFNN_ClimateDEM_Area": "FFNN",
    "chao2_estimator": "Chao2 estimator",
    "Linear_ClimateDEM_Area": "Linear",
    "MuScaRi_Area": "MuScaRi",
    "MuScaRi_ClimateDEM": "MuScaRi",
    "MuScaRi_ClimateDEM_Area": "MuScaRi",
}
PREDICTOR_MAP = {
    "FFNN_ClimateDEM_Area": "Env. + Area",
    "chao2_estimator": "--",
    "Linear_ClimateDEM_Area": "Env. + Area",
    "MuScaRi_Area": "Area",
    "MuScaRi_ClimateDEM": "Env.",
    "MuScaRi_ClimateDEM_Area": "Env. + Area",
}
METRICS = ["nrmse", "mape", "median_relative_bias", "r2", "d2"]


def load_benchmark_results(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(BENCHMARK_DIR / f"benchmark_results_{dataset_id}.csv")
    audit = pd.read_csv(
        GIFT_AUDIT_DIR / dataset_id / "gift_asymptote_evaluation_results.csv"
    )
    audit = audit[
        (audit["prediction_mode"] == "asymptotic_total")
        & (audit["aggregation"] == "fold_member")
    ]
    for _, row in audit.iterrows():
        mask = (models["experiment"] == row["model_name"]) & (
            models["fold"] == int(row["fold"])
        )
        for metric in METRICS:
            models.loc[mask, f"extrap_{metric}"] = row[metric]
    chao2 = pd.read_csv(CHAO2_RESULTS) if dataset_id == "ceacce0" else pd.DataFrame()
    return models, chao2


def format_mean_std(values: pd.Series, percent: bool = False) -> tuple[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return "-", np.nan
    scale = 100.0 if percent else 1.0
    mean = float(np.mean(values)) * scale
    std = float(np.std(values, ddof=1)) * scale
    return f"{mean:.3f} ± {std:.3f}", mean


def build_performance_table(
    models: pd.DataFrame,
    chao2: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    data = pd.concat([models, chao2], ignore_index=True)
    experiments = [
        name
        for name in MODEL_ORDER
        if name in set(data["experiment"])
        and data.loc[data["experiment"] == name, f"{endpoint}_nrmse"].notna().any()
    ]
    rows = []
    for experiment in experiments:
        experiment_data = data[data["experiment"] == experiment]
        row = {
            "experiment": experiment,
            "model": LABEL_MAP[experiment],
            "predictors": PREDICTOR_MAP[experiment],
        }
        for metric in METRICS:
            row[metric], row[f"_{metric}_mean"] = format_mean_std(
                experiment_data[f"{endpoint}_{metric}"],
                percent=metric == "nrmse",
            )
        rows.append(row)
    table = pd.DataFrame(rows)

    for metric in METRICS:
        means = table[f"_{metric}_mean"]
        score = means.abs() if metric == "median_relative_bias" else means
        best = score.idxmax() if metric in {"r2", "d2"} else score.idxmin()
        table.loc[best, metric] = f"\\textbf{{{table.loc[best, metric]}}}"
    return table


def render_latex_table(table: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[h!]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{tabularx}{\\textwidth}{l l >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}",
        "    \\toprule",
        "    Model & Predictors & NRMSE (\\%) & MAPE & Rel. Bias & $R^2$ & $D^2$ \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"    {row['model']} & {row['predictors']} & {row['nrmse']} & {row['mape']} & "
            f"{row['median_relative_bias']} & {row['r2']} & {row['d2']} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabularx}",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    description = (
        "NRMSE: root mean squared error divided by mean observed richness for each model and split, "
        "reported as a percentage; "
        "MAPE: mean absolute percentage error; Rel.\\ Bias: median relative bias, defined as "
        "$\\mathrm{median}[(\\hat{S}-S)/S]$; $R^2$: coefficient of determination; $D^2$: fraction "
        "of deviance explained. Lower NRMSE, MAPE, and Rel.\\ Bias (closer to 0), and higher $R^2$ "
        "and $D^2$, indicate better performance."
    )
    for dataset_id in DATASET_IDS:
        models, chao2 = load_benchmark_results(dataset_id)
        suffix = "" if dataset_id == "ceacce0" else "_100km"
        block_text = " using 100\\,km spatial blocks" if suffix else ""
        for endpoint, dataset_name in (("interp", "EVA test"), ("extrap", "GIFT")):
            table = build_performance_table(models, chao2, endpoint)
            endpoint_label = "Interpolation" if endpoint == "interp" else "Extrapolation"
            caption = (
                f"{endpoint_label} performance on the {dataset_name} dataset{block_text} "
                "(mean ± standard deviation across splits). " + description
            )
            label = f"tab:{endpoint}_performance{suffix}"
            latex = render_latex_table(table, caption, label)
            output = BENCHMARK_DIR / f"benchmark_results_{dataset_id}_{endpoint}_performance.tex"
            output.write_text(latex)
            print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Add mean-normalized RMSE to existing final-revision result artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from muscari.utils import add_effort_columns


ROOT = Path(__file__).parents[2]
SBCV_DATASET_IDS = ("ceacce0", "d0848f6")
GIFT_DATASET_ID = "418c563"
GIFT_DATA_PATH = (
    ROOT / "data" / "processed" / "test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
)
BENCHMARK_DIR = ROOT / "scripts" / "results" / "benchmark"
ARTIFACT_ROOT = BENCHMARK_DIR / "artifacts"
GIFT_AUDIT_ROOT = ROOT / "scripts" / "results" / "gift_asymptote_evaluation"
SPLITS = ("train", "val", "test")


def load_model_config(experiment: str, config_hash: str) -> dict:
    path = ARTIFACT_ROOT / experiment / config_hash / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark config: {path}")
    with open(path) as handle:
        return json.load(handle)


def add_nrmse(rmse: float, y_true: pd.Series) -> float:
    mean_observed = float(y_true.mean())
    if not np.isfinite(mean_observed) or mean_observed <= 0:
        raise ValueError("Mean observed richness must be finite and positive.")
    return float(rmse) / mean_observed


def prepared_targets(path: Path, feature_names: list[str], effort_transform: str) -> pd.Series:
    columns = list(
        dict.fromkeys(
            ["sr", "observed_area", "sp_unit_area"]
            + [
                name
                for name in feature_names
                if name not in {"log_observed_area", "log_sp_unit_area"}
            ]
        )
    )
    df = pd.read_parquet(path, columns=columns)
    df = add_effort_columns(df, effort_transform)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sr", "log_observed_area"] + feature_names)
    return df["sr"]


def backfill_benchmark(dataset_id: str) -> None:
    path = BENCHMARK_DIR / f"benchmark_results_{dataset_id}.csv"
    results = pd.read_csv(path)
    combined = results[results["experiment"] == "MuScaRi_ClimateDEM_Area"].iloc[0]
    combined_config = load_model_config(combined["experiment"], combined["config_hash"])
    linear_features = combined_config["features_and_labels"]["feature_columns"]

    target_cache: dict[tuple, pd.Series] = {}
    for index, row in results.iterrows():
        experiment = row["experiment"]
        if experiment == "Linear_ClimateDEM_Area":
            feature_names = linear_features
        else:
            config = load_model_config(experiment, row["config_hash"])
            feature_names = config["features_and_labels"]["feature_columns"]
        effort_transform = row["effort_transform"]

        for split in SPLITS:
            cache_key = (dataset_id, row["fold"], split, tuple(feature_names), effort_transform)
            if cache_key not in target_cache:
                split_path = (
                    ROOT
                    / "data"
                    / "processed"
                    / "training_samples"
                    / "sbcv"
                    / dataset_id
                    / f"fold_{int(row['fold'])}_{split}.parquet"
                )
                target_cache[cache_key] = prepared_targets(
                    split_path,
                    feature_names,
                    effort_transform,
                )
            results.loc[index, f"{split}_nrmse"] = add_nrmse(
                row[f"{split}_rmse"],
                target_cache[cache_key],
            )

        gift_key = (GIFT_DATASET_ID, tuple(feature_names), effort_transform)
        if gift_key not in target_cache:
            target_cache[gift_key] = prepared_targets(
                GIFT_DATA_PATH,
                feature_names,
                effort_transform,
            )
        results.loc[index, "gift_nrmse"] = add_nrmse(
            row["gift_rmse"],
            target_cache[gift_key],
        )
        results.loc[index, "interp_nrmse"] = results.loc[index, "test_nrmse"]
        results.loc[index, "extrap_nrmse"] = results.loc[index, "gift_nrmse"]

    results.to_csv(path, index=False)
    print(f"Updated {path.relative_to(ROOT)}")


def backfill_gift_audit(dataset_id: str) -> None:
    path = GIFT_AUDIT_ROOT / dataset_id / "gift_asymptote_evaluation_results.csv"
    results = pd.read_csv(path)
    if (results["sr_mean"] <= 0).any() or not np.isfinite(results["sr_mean"]).all():
        raise ValueError(f"Invalid observed-richness mean in {path}")
    results["nrmse"] = results["rmse"] / results["sr_mean"]
    results.to_csv(path, index=False)
    ensemble = results[results["aggregation"] == "uniform_ensemble"].copy()
    display = ensemble[
        [
            "model_name",
            "prediction_mode",
            "nrmse",
            "r2",
            "median_relative_bias",
            "bias_slope_log_area",
            "pred_mean",
        ]
    ].copy()
    display["nrmse"] *= 100.0
    display = display.rename(columns={"nrmse": "nrmse_percent"})
    summary_path = path.with_name("gift_asymptote_evaluation_summary.md")
    markdown_table = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        markdown_table.append(
            "| "
            + " | ".join(
                f"{row[column]:.3f}" if isinstance(row[column], (float, np.floating)) else str(row[column])
                for column in display.columns
            )
            + " |"
        )
    lines = [
        "# GIFT Asymptote Evaluation",
        "",
        f"Dataset: `{dataset_id}`",
        f"GIFT dataset: `{GIFT_DATASET_ID}`",
        "",
        "Uniform ensemble metrics:",
        "",
        "\n".join(markdown_table),
        "",
        f"Full results: `{path.relative_to(ROOT)}`",
        "",
    ]
    summary_path.write_text("\n".join(lines))
    print(f"Updated {path.relative_to(ROOT)}")


def main() -> None:
    for dataset_id in SBCV_DATASET_IDS:
        backfill_benchmark(dataset_id)
        backfill_gift_audit(dataset_id)


if __name__ == "__main__":
    main()

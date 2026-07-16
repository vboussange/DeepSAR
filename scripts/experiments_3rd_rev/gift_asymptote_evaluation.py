"""Audit finite full-area versus asymptotic GIFT predictions.

The benchmark currently evaluates GIFT with observed_area equal to the spatial
unit area. This script compares that finite full-area prediction to the true
asymptotic richness returned by ``predict_sr_tot`` for existing MuScaRi
benchmark artifacts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from muscari import MuScaRiEnsemble
from muscari.utils import (
    add_effort_columns,
    compute_log1p_metrics,
    compute_metrics,
    residual_bias_slope,
    setup_logger,
)


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
GIFT_DATASET_ID = "418c563"
GIFT_DATA_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
ARTIFACT_ROOT = ROOT / "scripts/results/benchmark/artifacts"
OUTPUT_DIR = ROOT / "scripts/results/gift_asymptote_evaluation" / SBCV_DATASET_ID
RESULTS_PATH = OUTPUT_DIR / "gift_asymptote_evaluation_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "gift_asymptote_evaluation_summary.md"

MODEL_SPECS = [
    ("MuScaRi_Area", "ac733d9bd2f6"),
    ("MuScaRi_ClimateDEM", "dae0789a3c87"),
    ("MuScaRi_ClimateDEM_Area", "ad74a3020281"),
]

DEVICE = os.environ.get("MUSCARI_GIFT_AUDIT_DEVICE", "cpu")
OBSERVED_AREA_POLICY = "observed_area_set_to_sp_unit_area"
logger = setup_logger("gift_asymptote_evaluation")


def unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing benchmark config: {config_path}")
    with open(config_path) as handle:
        return json.load(handle)


def prepare_gift_data(feature_names: list[str], effort_transform: str) -> pd.DataFrame:
    df = gpd.read_parquet(GIFT_DATA_PATH)
    df = df.copy()
    df["observed_area"] = df["sp_unit_area"]
    df = add_effort_columns(df, effort_transform)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Use one complete-case cohort so metrics remain comparable across feature sets.
    df = df.dropna()

    required = ["sr", "sp_unit_area", "observed_area", "log_sp_unit_area", "log_observed_area"]
    required = unique_columns(required + feature_names)
    df = df.dropna(subset=required)

    columns = ["sr", "sp_unit_area", "observed_area", "log_sp_unit_area", "log_observed_area"]
    columns = unique_columns(columns + feature_names)
    return pd.DataFrame(df[columns]).copy()


def metric_row(
    *,
    config: dict,
    model_name: str,
    config_hash: str,
    run_dir: Path,
    prediction_mode: str,
    aggregation: str,
    fold: int | None,
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict:
    y_true = df["sr"].to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    row = {
        "dataset_id": SBCV_DATASET_ID,
        "gift_dataset_id": GIFT_DATASET_ID,
        "model_name": model_name,
        "config_hash": config_hash,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "prediction_mode": prediction_mode,
        "observed_area_policy": OBSERVED_AREA_POLICY,
        "aggregation": aggregation,
        "fold": fold,
        "n_samples": len(df),
        "feature_count": len(config["features_and_labels"]["feature_columns"]),
        "feature_names": ";".join(config["features_and_labels"]["feature_columns"]),
        "architecture_variant": config["model"]["architecture_variant"],
        "layer_sizes": ";".join(str(size) for size in config["model"]["layer_sizes"]),
        "effort_transform": config["model"]["effort_transform"],
        "asymptote_transform": config["model"]["asymptote_transform"],
        "weibull_parameterization": config["model"]["weibull_parameterization"],
        "target_transform": config["training"]["target_transform"],
        "sr_mean": float(np.mean(y_true)),
        "sr_median": float(np.median(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_median": float(np.median(y_pred)),
        "bias_slope_log_area": residual_bias_slope(
            y_true,
            y_pred,
            df["log_sp_unit_area"].to_numpy(dtype=float),
        ),
    }
    row.update(compute_metrics(y_true, y_pred))
    row.update(compute_log1p_metrics(y_true, y_pred))
    return row


def fold_ids_for_run(run_dir: Path) -> list[int]:
    fold_paths = sorted(run_dir.glob("fold_*.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {run_dir}")
    return [int(path.stem.split("_")[-1]) for path in fold_paths]


def evaluate_model(model_name: str, config_hash: str) -> list[dict]:
    run_dir = ARTIFACT_ROOT / model_name / config_hash
    config = load_config(run_dir)
    ensemble = MuScaRiEnsemble.from_folds(
        run_dir,
        device=DEVICE,
        use_validation_weights=False,
    )
    fold_ids = fold_ids_for_run(run_dir)
    feature_names = list(ensemble.feature_names)
    effort_transform = config["model"]["effort_transform"]
    gift_df = prepare_gift_data(feature_names, effort_transform)

    rows = []
    predictions_by_mode = {
        "finite_full_area": ensemble.predict_members_sr(gift_df),
        "asymptotic_total": ensemble.predict_members_sr_tot(gift_df),
    }
    weights = np.asarray(ensemble.ensemble_weights, dtype=float)

    for prediction_mode, predictions in predictions_by_mode.items():
        logger.info(
            "Evaluating %s %s on %d GIFT rows.",
            model_name,
            prediction_mode,
            len(gift_df),
        )
        for idx, fold_id in enumerate(fold_ids):
            rows.append(
                metric_row(
                    config=config,
                    model_name=model_name,
                    config_hash=config_hash,
                    run_dir=run_dir,
                    prediction_mode=prediction_mode,
                    aggregation="fold_member",
                    fold=fold_id,
                    df=gift_df,
                    y_pred=predictions[idx],
                )
            )
        rows.append(
            metric_row(
                config=config,
                model_name=model_name,
                config_hash=config_hash,
                run_dir=run_dir,
                prediction_mode=prediction_mode,
                aggregation="uniform_ensemble",
                fold=None,
                df=gift_df,
                y_pred=np.average(predictions, axis=0, weights=weights),
            )
        )
    return rows


def write_summary(results: pd.DataFrame) -> None:
    ensemble_rows = results[results["aggregation"] == "uniform_ensemble"].copy()
    display_cols = [
        "model_name",
        "prediction_mode",
        "nrmse",
        "r2",
        "median_relative_bias",
        "bias_slope_log_area",
        "pred_mean",
    ]
    table = ensemble_rows[display_cols].copy()
    table["nrmse"] *= 100.0
    table = table.rename(columns={"nrmse": "nrmse_percent"})
    for col in ["nrmse_percent", "r2", "median_relative_bias", "bias_slope_log_area", "pred_mean"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    markdown_table = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        markdown_table.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")

    lines = [
        "# GIFT Asymptote Evaluation",
        "",
        f"Dataset: `{SBCV_DATASET_ID}`",
        f"GIFT dataset: `{GIFT_DATASET_ID}`",
        f"Observed-area policy: `{OBSERVED_AREA_POLICY}`",
        "",
        "Uniform ensemble metrics:",
        "",
        "\n".join(markdown_table),
        "",
        f"Full results: `{RESULTS_PATH.relative_to(ROOT)}`",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, config_hash in MODEL_SPECS:
        rows.extend(evaluate_model(model_name, config_hash))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    write_summary(results)
    logger.info("Wrote %s", RESULTS_PATH)
    logger.info("Wrote %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()

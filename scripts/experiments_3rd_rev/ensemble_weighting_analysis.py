"""Compare ensemble weighting choices for the selected benchmark artifact."""
from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from muscari.muscari import MuScaRi
from muscari.utils import (
    add_effort_columns,
    compute_log1p_metrics,
    compute_metrics,
    residual_bias_slope,
    setup_logger,
)


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
GIFT_DATASET_ID = "da569da"
MODEL_NAME = "MuScaRi_ClimateDEM"
CONFIG_HASH = "dae0789a3c87"
RUN_DIR = ROOT / "scripts/results/benchmark/artifacts" / MODEL_NAME / CONFIG_HASH
CONFIG_PATH = RUN_DIR / "config.json"
SBCV_DATA_DIR = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
GIFT_DATA_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
OUTPUT_DIR = ROOT / "scripts/results/ensemble_weighting_analysis" / SBCV_DATASET_ID
RESULTS_PATH = OUTPUT_DIR / "ensemble_weighting_results.csv"
WEIGHTS_PATH = OUTPUT_DIR / "ensemble_weighting_weights.csv"
STD_SUMMARY_PATH = OUTPUT_DIR / "ensemble_weighting_std_summary.csv"
SUMMARY_PATH = OUTPUT_DIR / "ensemble_weighting_summary.md"

DEVICE = os.environ.get("MUSCARI_ENSEMBLE_ANALYSIS_DEVICE", "cpu")
WEIGHTING_STRATEGIES = [
    "uniform",
    "inverse_val_rmse",
    "inverse_val_rmse_squared",
]
SPLITS = {
    "gift": "GIFT extrapolation",
    "interpolation": "combined SBCV test folds",
}
REQUIRED_COLUMNS = ["sr", "sp_unit_area", "observed_area"]
logger = setup_logger("ensemble_weighting_analysis")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark config: {CONFIG_PATH}")
    with open(CONFIG_PATH) as handle:
        return json.load(handle)


def fold_id_from_path(path: Path) -> int:
    return int(path.stem.split("_", maxsplit=1)[1])


def validation_rmse(metrics: dict | None) -> float:
    if not metrics:
        return float("nan")
    if "val_rmse" in metrics:
        return float(metrics["val_rmse"])
    val_metrics = metrics.get("val")
    if isinstance(val_metrics, dict) and "rmse" in val_metrics:
        return float(val_metrics["rmse"])
    return float("nan")


def split_rmse(metrics: dict | None, split: str) -> float:
    if not metrics:
        return float("nan")
    split_metrics = metrics.get(split)
    if isinstance(split_metrics, dict) and "rmse" in split_metrics:
        return float(split_metrics["rmse"])
    return float("nan")


def load_models() -> tuple[list[MuScaRi], pd.DataFrame]:
    checkpoint_paths = sorted(RUN_DIR.glob("fold_*.pth"), key=fold_id_from_path)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {RUN_DIR}")

    models = []
    rows = []
    for checkpoint_path in checkpoint_paths:
        fold_id = fold_id_from_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = MuScaRi.initialize(checkpoint, device=DEVICE).eval()
        models.append(model)
        metrics = checkpoint.get("metrics")
        rows.append({
            "fold": fold_id,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "val_rmse": validation_rmse(metrics),
            "test_rmse": split_rmse(metrics, "test"),
            "gift_rmse": split_rmse(metrics, "gift"),
        })
    return models, pd.DataFrame(rows)


def normalized_weights(val_rmse: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        raw = np.ones_like(val_rmse, dtype=float)
    elif strategy == "inverse_val_rmse":
        raw = 1.0 / np.maximum(val_rmse, 1e-12)
    elif strategy == "inverse_val_rmse_squared":
        raw = 1.0 / np.maximum(val_rmse, 1e-12) ** 2
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    valid = np.isfinite(raw) & (raw > 0)
    if not valid.all():
        raw = np.where(valid, raw, np.nanmedian(raw[valid]))
    return raw / raw.sum()


def effective_n(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights ** 2))


def expand_weight_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    val_rmse = fold_metrics["val_rmse"].to_numpy(dtype=float)
    rows = []
    for strategy in WEIGHTING_STRATEGIES:
        weights = normalized_weights(val_rmse, strategy)
        eff_n = effective_n(weights)
        for idx, source in fold_metrics.iterrows():
            rows.append({
                "dataset_id": SBCV_DATASET_ID,
                "gift_dataset_id": GIFT_DATASET_ID,
                "model_name": MODEL_NAME,
                "config_hash": CONFIG_HASH,
                "weighting_strategy": strategy,
                "fold": int(source["fold"]),
                "val_rmse": float(source["val_rmse"]),
                "test_rmse": float(source["test_rmse"]),
                "gift_rmse": float(source["gift_rmse"]),
                "weight": float(weights[idx]),
                "effective_n": eff_n,
                "checkpoint_path": source["checkpoint_path"],
            })
    return pd.DataFrame(rows)


def load_split_data(config: dict, feature_names: list[str]) -> dict[str, pd.DataFrame]:
    effort_transform = config["model"]["effort_transform"]
    required = REQUIRED_COLUMNS + feature_names

    def prepare(path: Path, source_fold: int | None = None) -> pd.DataFrame:
        df = gpd.read_parquet(path)
        df = add_effort_columns(df, effort_transform)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=required + ["log_sp_unit_area", "log_observed_area"])
        columns = ["sr", "log_sp_unit_area", "log_observed_area"] + feature_names
        out = pd.DataFrame(df[columns]).copy()
        if source_fold is not None:
            out["source_fold"] = source_fold
        return out

    gift = prepare(GIFT_DATA_PATH)
    test_frames = [
        prepare(SBCV_DATA_DIR / f"fold_{fold_id}_test.parquet", source_fold=fold_id)
        for fold_id in range(5)
    ]
    return {
        "gift": gift,
        "interpolation": pd.concat(test_frames, ignore_index=True),
    }


def member_prediction_matrix(models: list[MuScaRi], df: pd.DataFrame) -> np.ndarray:
    predictions = []
    for idx, model in enumerate(models):
        logger.info("Predicting with member %d on %d rows.", idx, len(df))
        predictions.append(model.predict_sr(df).reshape(-1))
    return np.asarray(predictions, dtype=float)


def weighted_mean(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(predictions, axis=0, weights=weights)


def weighted_std(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weighted_mean(predictions, weights)
    variance = np.average((predictions - mean) ** 2, axis=0, weights=weights)
    return np.sqrt(np.maximum(variance, 0.0))


def metric_row(
    *,
    config: dict,
    split: str,
    strategy: str,
    weights: np.ndarray,
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, float | str | int]:
    y_true = df["sr"].to_numpy(dtype=float)
    row = {
        "dataset_id": SBCV_DATASET_ID,
        "gift_dataset_id": GIFT_DATASET_ID,
        "split": split,
        "split_description": SPLITS[split],
        "model_name": MODEL_NAME,
        "config_hash": CONFIG_HASH,
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "weighting_strategy": strategy,
        "n_models": len(weights),
        "effective_n": effective_n(weights),
        "n_samples": len(df),
        "feature_count": len(config["features_and_labels"]["feature_columns"]),
        "feature_names": ";".join(config["features_and_labels"]["feature_columns"]),
        "architecture_variant": config["model"]["architecture_variant"],
        "layer_sizes": ";".join(str(size) for size in config["model"]["layer_sizes"]),
        "effort_transform": config["model"]["effort_transform"],
        "target_transform": config["model"]["target_transform"],
        "asymptote_transform": config["model"]["asymptote_transform"],
        "weibull_parameterization": config["model"]["weibull_parameterization"],
    }
    row.update(compute_metrics(y_true, y_pred))
    row.update(compute_log1p_metrics(y_true, y_pred))
    row["bias_slope_log_area"] = residual_bias_slope(
        y_true,
        y_pred,
        df["log_sp_unit_area"].to_numpy(dtype=float),
    )
    return row


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return float("nan")
    return float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method=method))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def quartile_rmse_columns(y_true: np.ndarray, y_pred: np.ndarray, ensemble_std: np.ndarray) -> dict[str, float | int]:
    residual = y_pred - y_true
    out: dict[str, float | int] = {}
    try:
        bins = pd.qcut(ensemble_std, q=4, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.full(len(ensemble_std), np.nan))
    bins = pd.Series(bins)
    for quartile in range(4):
        mask = bins == quartile
        label = quartile + 1
        out[f"std_q{label}_n"] = int(mask.sum())
        out[f"std_q{label}_rmse"] = rmse(residual[mask.to_numpy()]) if mask.any() else float("nan")
    return out


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    def format_value(value) -> str:
        if isinstance(value, float) or isinstance(value, np.floating):
            if np.isnan(value):
                return "nan"
            return f"{float(value):.{float_digits}f}"
        return str(value)

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def std_summary_row(
    *,
    split: str,
    strategy: str,
    weights: np.ndarray,
    df: pd.DataFrame,
    y_pred: np.ndarray,
    ensemble_std: np.ndarray,
    uniform_std: np.ndarray,
) -> dict[str, float | str | int]:
    y_true = df["sr"].to_numpy(dtype=float)
    abs_error = np.abs(y_pred - y_true)
    ratio = np.divide(
        ensemble_std,
        uniform_std,
        out=np.full_like(ensemble_std, np.nan, dtype=float),
        where=uniform_std > 0,
    )
    row = {
        "dataset_id": SBCV_DATASET_ID,
        "gift_dataset_id": GIFT_DATASET_ID,
        "split": split,
        "split_description": SPLITS[split],
        "model_name": MODEL_NAME,
        "config_hash": CONFIG_HASH,
        "weighting_strategy": strategy,
        "n_models": len(weights),
        "effective_n": effective_n(weights),
        "n_samples": len(df),
        "ensemble_std_mean": float(np.mean(ensemble_std)),
        "ensemble_std_median": float(np.median(ensemble_std)),
        "ensemble_std_std": float(np.std(ensemble_std)),
        "ensemble_std_q05": float(np.quantile(ensemble_std, 0.05)),
        "ensemble_std_q25": float(np.quantile(ensemble_std, 0.25)),
        "ensemble_std_q75": float(np.quantile(ensemble_std, 0.75)),
        "ensemble_std_q95": float(np.quantile(ensemble_std, 0.95)),
        "mean_ratio_to_uniform_std": float(np.nanmean(ratio)),
        "median_ratio_to_uniform_std": float(np.nanmedian(ratio)),
        "mean_std_ratio_to_uniform_mean": float(np.mean(ensemble_std) / np.mean(uniform_std)),
        "median_std_ratio_to_uniform_median": float(np.median(ensemble_std) / np.median(uniform_std)),
        "pearson_std_abs_error": safe_corr(ensemble_std, abs_error, "pearson"),
        "spearman_std_abs_error": safe_corr(ensemble_std, abs_error, "spearman"),
    }
    row.update(quartile_rmse_columns(y_true, y_pred, ensemble_std))
    return row


def write_summary(results: pd.DataFrame, weights: pd.DataFrame, std_summary: pd.DataFrame) -> None:
    gift = results[results["split"] == "gift"].sort_values("rmse")
    interpolation = results[results["split"] == "interpolation"].sort_values("rmse")
    uniform_gift = gift[gift["weighting_strategy"] == "uniform"].iloc[0]
    best_gift = gift.iloc[0]
    current_gift = gift[gift["weighting_strategy"] == "inverse_val_rmse_squared"].iloc[0]
    current_eff_n = float(current_gift["effective_n"])
    gift_std = std_summary[std_summary["split"] == "gift"].set_index("weighting_strategy")

    rmse_delta = float(best_gift["rmse"] - uniform_gift["rmse"])
    rmse_delta_pct = 100.0 * rmse_delta / float(uniform_gift["rmse"])
    current_std_ratio = float(gift_std.loc["inverse_val_rmse_squared", "mean_std_ratio_to_uniform_mean"])
    current_calib = float(gift_std.loc["inverse_val_rmse_squared", "spearman_std_abs_error"])
    uniform_calib = float(gift_std.loc["uniform", "spearman_std_abs_error"])

    if best_gift["weighting_strategy"] == "uniform":
        recommendation = "Use uniform averaging for this artifact."
        reason = "It gives the best GIFT RMSE among the tested strategies."
    elif abs(rmse_delta_pct) < 1.0 and current_eff_n < 4.75:
        recommendation = "Prefer uniform averaging unless a separate validation set justifies weighting."
        reason = (
            "The weighted GIFT RMSE change is below 1%, while validation weighting "
            "reduces the effective ensemble size."
        )
    else:
        recommendation = f"Use {best_gift['weighting_strategy']} if GIFT RMSE is the priority."
        reason = "It gives the lowest GIFT RMSE in this diagnostic comparison."

    lines = [
        "# Ensemble Weighting Analysis",
        "",
        f"Artifact: `{RUN_DIR.relative_to(ROOT)}`",
        "",
        "## Recommendation",
        "",
        f"{recommendation} {reason}",
        "",
        "## GIFT Performance",
        "",
        markdown_table(
            gift[["weighting_strategy", "rmse", "mape", "r2", "d2", "mean_relative_bias", "effective_n"]]
        ),
        "",
        "## Interpolation Performance",
        "",
        markdown_table(
            interpolation[
                ["weighting_strategy", "rmse", "mape", "r2", "d2", "mean_relative_bias", "effective_n"]
            ]
        ),
        "",
        "## Ensemble Standard Deviation",
        "",
        markdown_table(
            std_summary[
                [
                    "split",
                    "weighting_strategy",
                    "ensemble_std_mean",
                    "ensemble_std_median",
                    "mean_std_ratio_to_uniform_mean",
                    "spearman_std_abs_error",
                    "effective_n",
                ]
            ]
        ),
        "",
        "## Notes",
        "",
        f"- Best GIFT RMSE delta vs uniform: {rmse_delta:.4f} ({rmse_delta_pct:.3f}%).",
        f"- Current validation-RMSE-squared weighting effective_n: {current_eff_n:.3f}.",
        f"- Current GIFT mean ensemble-SD ratio vs uniform: {current_std_ratio:.4f}.",
        f"- Current vs uniform GIFT Spearman(std, abs error): {current_calib:.4f} vs {uniform_calib:.4f}.",
        "- GIFT is used only for evaluation in this script, not for fitting weights.",
        "",
        "## Weight Table",
        "",
        markdown_table(
            weights[["weighting_strategy", "fold", "val_rmse", "weight", "effective_n"]],
            float_digits=6,
        ),
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading config from %s.", CONFIG_PATH)
    config = load_config()
    feature_names = config["features_and_labels"]["feature_columns"]
    logger.info("Loading fold models from %s on %s.", RUN_DIR, DEVICE)
    models, fold_metrics = load_models()
    weights_df = expand_weight_table(fold_metrics)
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    split_data = load_split_data(config, feature_names)
    result_rows = []
    std_rows = []
    val_rmse = fold_metrics["val_rmse"].to_numpy(dtype=float)

    for split, df in split_data.items():
        logger.info("Evaluating split %s with %d rows.", split, len(df))
        predictions = member_prediction_matrix(models, df)
        uniform_weights = normalized_weights(val_rmse, "uniform")
        uniform_std = weighted_std(predictions, uniform_weights)
        for strategy in WEIGHTING_STRATEGIES:
            weights = normalized_weights(val_rmse, strategy)
            y_pred = weighted_mean(predictions, weights)
            ensemble_std = weighted_std(predictions, weights)
            result_rows.append(
                metric_row(
                    config=config,
                    split=split,
                    strategy=strategy,
                    weights=weights,
                    df=df,
                    y_pred=y_pred,
                )
            )
            std_rows.append(
                std_summary_row(
                    split=split,
                    strategy=strategy,
                    weights=weights,
                    df=df,
                    y_pred=y_pred,
                    ensemble_std=ensemble_std,
                    uniform_std=uniform_std,
                )
            )

    results = pd.DataFrame(result_rows)
    std_summary = pd.DataFrame(std_rows)
    results.to_csv(RESULTS_PATH, index=False)
    std_summary.to_csv(STD_SUMMARY_PATH, index=False)
    write_summary(results, weights_df, std_summary)
    logger.info("Wrote %s.", RESULTS_PATH)
    logger.info("Wrote %s.", WEIGHTS_PATH)
    logger.info("Wrote %s.", STD_SUMMARY_PATH)
    logger.info("Wrote %s.", SUMMARY_PATH)


if __name__ == "__main__":
    main()

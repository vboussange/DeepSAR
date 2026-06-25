"""Aggregate third-revision architecture screen results."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from muscari.utils import feature_sets, symmetric_arch


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
RESULTS_ROOT = ROOT / "scripts/results"
INPUT_GLOB = f"architecture_*/{SBCV_DATASET_ID}/architecture_screen_results_{SBCV_DATASET_ID}.csv"
SBCV_SAMPLES_DIR = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
OUTPUT_DIR = RESULTS_ROOT / "aggregate_screen_results" / SBCV_DATASET_ID
OUTPUT_PATH = OUTPUT_DIR / f"aggregate_screen_results_{SBCV_DATASET_ID}.csv"

BIOCLIMATE_VARS = [
    "bio1",
    "pet_penman_mean",
    "sfcWind_mean",
    "bio4",
    "rsds_1981-2010_range_V.2.1",
    "bio12",
    "bio15",
]
FEATURE_CONFIGS = {
    "env_area": {
        "bioclimate_vars": [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            "bio12",
        ],
        "include_elevation": True,
        "include_landcover": False,
    },
    "env_area_full_lc": {
        "bioclimate_vars": BIOCLIMATE_VARS,
        "include_elevation": True,
        "include_landcover": True,
    },
}
FEATURE_GROUP = "env_area"
LAYER_SIZE_CONFIGS = {
    "architecture_screen": ("large", 128, symmetric_arch(6, base=128, factor=4)),
    "architecture_screen_legacyonly_large": ("large", 128, symmetric_arch(6, base=128, factor=4)),
    "architecture_screen_legacy_log1p_max": ("large", 128, symmetric_arch(6, base=128, factor=4)),
    "architecture_screen_stable": ("large", 128, symmetric_arch(6, base=128, factor=4)),
    "architecture_screen_stable_maxabs_large": ("large", 128, symmetric_arch(6, base=128, factor=4)),
    "architecture_screen_stable_small": ("small", 32, symmetric_arch(6, base=32, factor=4)),
    "architecture_screen_stable_small_full_covariates": (
        "small",
        32,
        symmetric_arch(6, base=32, factor=4),
    ),
}

DATASETS = ["interp", "extrap"]
METRICS = [
    "rmse",
    "mape",
    "r2",
    "d2",
    "mean_relative_bias",
    "median_relative_bias",
    "bias_slope_log_area",
    "log1p_rmse",
    "log1p_mae",
    "log1p_r2",
    "log1p_d2",
]
METADATA_COLUMNS = [
    "model_family",
    "architecture_variant",
    "effort_transform",
    "asymptote_transform",
    "weibull_parameterization",
    "target_transform",
    "feature_set",
]
GROUP_COLUMNS = [
    "dataset_id",
    "source_screen",
    "model_type",
    "model_family",
    "architecture_variant",
    "layer_size_label",
    "layer_size_base",
    "layer_sizes",
    "effort_transform",
    "asymptote_transform",
    "weibull_parameterization",
    "target_transform",
    "covariates_used",
]
IDENTITY_COLUMNS = GROUP_COLUMNS + ["feature_count", "feature_names", "fold_count", "folds"]
NOT_RECORDED = "not_recorded"


def result_paths() -> list[Path]:
    paths = sorted(RESULTS_ROOT.glob(INPUT_GLOB))
    if not paths:
        raise FileNotFoundError(f"No architecture screen result CSVs matched {RESULTS_ROOT / INPUT_GLOB}")
    return paths


def schema_dataframe() -> pd.DataFrame:
    sample_paths = sorted(SBCV_SAMPLES_DIR.glob("*_train.parquet"))
    if not sample_paths:
        raise FileNotFoundError(f"No training parquet files found under {SBCV_SAMPLES_DIR}")
    schema = pq.read_schema(sample_paths[0])
    return pd.DataFrame(columns=schema.names)


def covariate_features() -> dict[str, list[str]]:
    sample_df = schema_dataframe()
    features = {}
    for name, config in FEATURE_CONFIGS.items():
        features[name] = feature_sets(
            sample_df,
            config["bioclimate_vars"],
            include_elevation=config["include_elevation"],
            include_landcover=config["include_landcover"],
        )[FEATURE_GROUP]
    return features


def normalize_layer_sizes(value) -> tuple[str, list[int]]:
    if isinstance(value, float) and np.isnan(value):
        return NOT_RECORDED, []
    if isinstance(value, str):
        value = value.strip()
        if not value or value == NOT_RECORDED:
            return NOT_RECORDED, []
        if ";" in value:
            sizes = [int(size) for size in value.split(";")]
            return value, sizes
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value, []
        if isinstance(parsed, (list, tuple)):
            sizes = [int(size) for size in parsed]
            return ";".join(str(size) for size in sizes), sizes
    if isinstance(value, (list, tuple)):
        sizes = [int(size) for size in value]
        return ";".join(str(size) for size in sizes), sizes
    return str(value), []


def label_layer_sizes(text: str, sizes: list[int]) -> tuple[str, str | int]:
    if text == NOT_RECORDED:
        return NOT_RECORDED, NOT_RECORDED
    for label, base, known_sizes in LAYER_SIZE_CONFIGS.values():
        if sizes == known_sizes:
            return label, base
    return "custom", sizes[0] if sizes else NOT_RECORDED


def fallback_layer_size_metadata(path: Path) -> tuple[str, str | int, str]:
    layer_label, layer_base, layer_sizes = LAYER_SIZE_CONFIGS.get(
        path.parent.parent.name,
        (NOT_RECORDED, NOT_RECORDED, []),
    )
    text = ";".join(str(size) for size in layer_sizes) if layer_sizes else NOT_RECORDED
    return layer_label, layer_base, text


def load_result_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset_id"] = SBCV_DATASET_ID
    df["source_screen"] = path.parent.parent.name
    fallback_label, fallback_base, fallback_sizes = fallback_layer_size_metadata(path)
    if "layer_sizes" in df.columns:
        labels = []
        bases = []
        layer_sizes_text = []
        for value in df["layer_sizes"]:
            text, sizes = normalize_layer_sizes(value)
            if text == NOT_RECORDED:
                labels.append(fallback_label)
                bases.append(fallback_base)
                layer_sizes_text.append(fallback_sizes)
                continue
            label, base = label_layer_sizes(text, sizes)
            labels.append(label)
            bases.append(base)
            layer_sizes_text.append(text)
        df["layer_size_label"] = labels
        df["layer_size_base"] = bases
        df["layer_sizes"] = layer_sizes_text
    else:
        df["layer_size_label"] = fallback_label
        df["layer_size_base"] = fallback_base
        df["layer_sizes"] = fallback_sizes
    for column in METADATA_COLUMNS:
        if column not in df.columns:
            df[column] = NOT_RECORDED
        df[column] = df[column].fillna(NOT_RECORDED).astype(str)
    df["model_type"] = df["architecture_variant"]
    df["covariates_used"] = df["feature_set"]
    return df


def load_results() -> pd.DataFrame:
    return pd.concat([load_result_file(path) for path in result_paths()], ignore_index=True)


def format_folds(values: pd.Series) -> str:
    folds = pd.to_numeric(values, errors="coerce").dropna().astype(int).sort_values().unique()
    return ",".join(str(fold) for fold in folds)


def summarize_values(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        return np.nan, np.nan
    return float(np.mean(numeric)), float(np.std(numeric, ddof=0))


def summarize_results(df: pd.DataFrame, features: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    for key, group in grouped:
        row = dict(zip(GROUP_COLUMNS, key))
        covariates = row["covariates_used"]
        feature_names = features.get(covariates)
        row["feature_count"] = len(feature_names) if feature_names is not None else np.nan
        row["feature_names"] = ";".join(feature_names) if feature_names is not None else NOT_RECORDED
        row["fold_count"] = int(pd.to_numeric(group["fold"], errors="coerce").nunique())
        row["folds"] = format_folds(group["fold"])

        for dataset in DATASETS:
            for metric in METRICS:
                column = f"{dataset}_{metric}"
                mean, std = summarize_values(group[column]) if column in group.columns else (np.nan, np.nan)
                row[f"{column}_mean"] = mean
                row[f"{column}_std"] = std
        rows.append(row)

    summary = pd.DataFrame(rows)
    metric_columns = [
        f"{dataset}_{metric}_{stat}"
        for dataset in DATASETS
        for metric in METRICS
        for stat in ["mean", "std"]
    ]
    summary = summary[IDENTITY_COLUMNS + metric_columns]
    return summary.sort_values(
        ["covariates_used", "extrap_rmse_mean", "interp_rmse_mean", "source_screen", "model_type"],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(load_results(), covariate_features())
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(summary)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

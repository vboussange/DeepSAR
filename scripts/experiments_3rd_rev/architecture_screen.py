"""
Private architecture screen for the third MuScaRi revision.

"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch

from muscari.benchmarker import BenchmarkConfig, Benchmarker
from muscari.muscari import MuScaRi
from muscari.utils import (
    feature_sets,
    setup_logger,
    symmetric_arch,
)


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
GIFT_DATASET_ID = "da569da"
SBCV_SAMPLES_PATH = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
SCREEN_NAME = "architecture_screen_stable_maxabs_large"
RUN_FOLDER = ROOT / "scripts/results" / SCREEN_NAME / SBCV_DATASET_ID
FEATURE_GROUP = "env_area"

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
    # "env_area_full_lc": {
    #     "bioclimate_vars": BIOCLIMATE_VARS,
    #     "include_elevation": True,
    #     "include_landcover": True,
    # },
    # "env_area_full_no_lc": {
    #     "bioclimate_vars": BIOCLIMATE_VARS,
    #     "include_elevation": True,
    #     "include_landcover": False,
    # },
    # "env_area_full_lc_no_dem": {
    #     "bioclimate_vars": BIOCLIMATE_VARS,
    #     "include_elevation": False,
    #     "include_landcover": True,
    # },
}
SELECTED_FEATURE_CONFIG = "env_area"

USE_WANDB = True
WANDB_PROJECT = "muscari-third-revision"
WANDB_GROUP = f"{SCREEN_NAME}_{SBCV_DATASET_ID}"
WANDB_TAGS = [SCREEN_NAME, SBCV_DATASET_ID, SELECTED_FEATURE_CONFIG]

SMOKE_TEST = os.environ.get("MUSCARI_SMOKE_TEST", "0") == "1"
FOLD_IDS = [0] if SMOKE_TEST else list(range(5))
N_EPOCHS = 1 if SMOKE_TEST else 500
TRAIN_FRAC = 0.002 if SMOKE_TEST else 1.0
BATCH_SIZE = 1024
LR = 1e-3
LAYER_SIZES = symmetric_arch(6, base=128, factor=4)

VARIANTS = [
    # {
    #     "name": "legacy_abs",
    #     "effort_transform": "absolute",
    #     "asymptote_transform": "identity",
    #     "weibull_parameterization": "legacy",
    #     "target_transform": "maxabs",
    # },
    # {
    #     "name": "softplus_abs",
    #     "effort_transform": "absolute",
    #     "asymptote_transform": "softplus",
    #     "weibull_parameterization": "legacy",
    #     "target_transform": "maxabs",
    # },
    # {
    #     "name": "exp_abs",
    #     "effort_transform": "absolute",
    #     "asymptote_transform": "exp",
    #     "weibull_parameterization": "legacy",
    #     "target_transform": "maxabs",
    # },
    # {
    #     "name": "stable_abs",
    #     "effort_transform": "absolute",
    #     "asymptote_transform": "softplus",
    #     "weibull_parameterization": "stable",
    #     "target_transform": "log1p_max",
    # },
    {
        "name": "stable_abs_maxabs",
        "effort_transform": "absolute",
        "asymptote_transform": "softplus",
        "weibull_parameterization": "stable",
        "target_transform": "maxabs",
    },
    # {
    #     "name": "stable_coverage",
    #     "effort_transform": "coverage",
    #     "asymptote_transform": "softplus",
    #     "weibull_parameterization": "stable",
    #     "target_transform": "log1p_max",
    # },
]

logger = setup_logger("architecture_screen")


def resolve_feature_config() -> dict:
    try:
        return FEATURE_CONFIGS[SELECTED_FEATURE_CONFIG]
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_CONFIGS))
        raise ValueError(
            f"Unknown feature config '{SELECTED_FEATURE_CONFIG}'. Expected one of: {valid}"
        ) from exc


@dataclass
class MuScaRiArchitectureInit:
    feature_names: list[str]
    asymptote_transform: str
    weibull_parameterization: str = "legacy"
    architecture: list[int] = field(default_factory=lambda: LAYER_SIZES.copy())

    def __call__(self, **kwargs):
        return MuScaRi(
            layer_sizes=self.architecture,
            feature_names=self.feature_names,
            ffnn_batchnorm=False,
            asymptote_transform=self.asymptote_transform,
            weibull_parameterization=self.weibull_parameterization,
            **kwargs,
        )


def discover_devices() -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def default_num_workers(devices: list[str]) -> int:
    return 0


def torch_threads_per_fold(devices: list[str]) -> int:
    worker_budget = os.cpu_count() or 1
    return max(1, min(4, worker_budget // max(1, len(devices))))


def build_config(variant: dict, feature_names: list[str], devices: list[str]) -> BenchmarkConfig:
    return BenchmarkConfig(
        devices=devices,
        path_gift_data=GIFT_SAMPLES_PATH,
        path_sbcv_data=SBCV_SAMPLES_PATH,
        num_workers=default_num_workers(devices),
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        lr=LR,
        torch_num_threads=torch_threads_per_fold(devices),
        effort_transform=variant["effort_transform"],
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_group=WANDB_GROUP,
        wandb_tags=WANDB_TAGS + [SBCV_DATASET_ID, variant["name"]],
        target_transform=variant["target_transform"],
        wandb_config={
            "dataset_id": SBCV_DATASET_ID,
            "gift_dataset_id": GIFT_DATASET_ID,
            "feature_set": SELECTED_FEATURE_CONFIG,
            "feature_names": feature_names,
            "architecture_variant": variant["name"],
            "asymptote_transform": variant["asymptote_transform"],
            "weibull_parameterization": variant["weibull_parameterization"],
            "target_transform": variant["target_transform"],
            "model_family": "MuScaRi",
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epoch_limit": N_EPOCHS,
            "run_folder": str(RUN_FOLDER),
        },
        fold_ids=FOLD_IDS,
    )


def run_variant(variant: dict, feature_names: list[str], devices: list[str]) -> pd.DataFrame:
    config = build_config(variant, feature_names, devices)
    model_init = MuScaRiArchitectureInit(
        feature_names=feature_names,
        asymptote_transform=variant["asymptote_transform"],
        weibull_parameterization=variant["weibull_parameterization"],
    )
    bench = Benchmarker(config)
    logger.info(
        "Running %s on %s with %s dataloader workers per fold",
        variant["name"],
        config.devices,
        config.num_workers,
    )
    results = bench.run(
        variant["name"],
        model_init,
        feature_names,
        train_frac=TRAIN_FRAC,
    )
    if results.empty:
        logger.warning("No results produced for %s", variant["name"])
        return results
    results["architecture_variant"] = variant["name"]
    results["effort_transform"] = variant["effort_transform"]
    results["asymptote_transform"] = variant["asymptote_transform"]
    results["weibull_parameterization"] = variant["weibull_parameterization"]
    results["target_transform"] = variant["target_transform"]
    results["feature_set"] = SELECTED_FEATURE_CONFIG
    results["model_family"] = "MuScaRi"
    results["n_epochs"] = N_EPOCHS
    results["batch_size"] = BATCH_SIZE
    return results


if __name__ == "__main__":
    RUN_FOLDER.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    sample_df = gpd.read_parquet(next(SBCV_SAMPLES_PATH.glob("*_train.parquet")))
    feature_config = resolve_feature_config()
    feature_names = feature_sets(
        sample_df,
        feature_config["bioclimate_vars"],
        include_elevation=feature_config["include_elevation"],
        include_landcover=feature_config["include_landcover"],
    )[FEATURE_GROUP]
    logger.info(
        "Architecture screen feature config %s (%d features): %s",
        SELECTED_FEATURE_CONFIG,
        len(feature_names),
        feature_names,
    )
    devices = discover_devices()
    logger.info("Architecture screen devices: %s", devices)

    rows = []
    for variant in VARIANTS:
        variant_results = run_variant(variant, feature_names, devices)
        if not variant_results.empty:
            rows.append(variant_results)

    if not rows:
        raise RuntimeError("No architecture screen results were produced.")
    results = pd.concat(rows, ignore_index=True).sort_values(["experiment", "fold"])
    suffix = "_smoke" if SMOKE_TEST else ""
    output_file = RUN_FOLDER / f"architecture_screen_results_{SBCV_DATASET_ID}{suffix}.csv"
    results.to_csv(output_file, index=False)
    logger.info("Saved architecture screen results to %s", output_file)

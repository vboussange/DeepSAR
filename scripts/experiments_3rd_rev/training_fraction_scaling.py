"""Evaluate interpolation NRMSE across training-sample fractions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from muscari.muscari import MuScaRi
from muscari.trainer import TrainConfig, Trainer
from muscari.utils import environmental_features, symmetric_arch


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
GIFT_DATASET_ID = "418c563"
SBCV_PATH = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
GIFT_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
OUTPUT_DIR = ROOT / "scripts/results/training_fraction_scaling" / SBCV_DATASET_ID
OUTPUT_PATH = OUTPUT_DIR / "training_fraction_scaling_results.csv"
ARTIFACT_ROOT = OUTPUT_DIR / "artifacts"
FINAL_BENCHMARK_PATH = (
    ROOT / "scripts/results/benchmark" / f"benchmark_results_{SBCV_DATASET_ID}.csv"
)

TRAIN_FRACTIONS = np.logspace(-4, 0, 9)
FOLD_IDS = list(range(5))
DEVICES = []  # Let TrainConfig discover each available accelerator once.
BIOCLIMATE_VARS = ["bio1", "pet_penman_mean", "sfcWind_mean", "bio12"]
LAYER_SIZES = symmetric_arch(6, base=32, factor=4)
N_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EFFORT_TRANSFORM = "absolute"
TARGET_TRANSFORM = "maxabs"
ASYMPTOTE_TRANSFORM = "absolute"
WEIBULL_PARAMETERIZATION = "legacy"
ARCHITECTURE_VARIANT = "legacy_maxabs_absolute_l32"
USE_WANDB = False
WANDB_PROJECT = "muscari-third-revision"
WANDB_GROUP = f"training_fraction_scaling_{SBCV_DATASET_ID}"


@dataclass
class MuScaRiInit:
    feature_names: list[str]
    architecture: list[int] = field(default_factory=lambda: LAYER_SIZES.copy())

    def __call__(self, **kwargs) -> MuScaRi:
        return MuScaRi(
            layer_sizes=list(self.architecture),
            feature_names=self.feature_names,
            ffnn_batchnorm=False,
            asymptote_transform=ASYMPTOTE_TRANSFORM,
            weibull_parameterization=WEIBULL_PARAMETERIZATION,
            **kwargs,
        )


def feature_names() -> list[str]:
    sample_path = next(SBCV_PATH.glob("fold_*_train.parquet"))
    sample = gpd.read_parquet(sample_path)
    return environmental_features(
        sample,
        BIOCLIMATE_VARS,
        include_elevation=True,
        include_landcover=False,
    ) + ["log_sp_unit_area"]


def validate_final_benchmark(benchmark: pd.DataFrame) -> None:
    if set(benchmark["fold"].astype(int)) != set(FOLD_IDS):
        raise ValueError("Final benchmark does not contain the expected folds.")
    if benchmark["interp_nrmse"].isna().any():
        raise ValueError("Final benchmark contains missing interpolation NRMSE values.")
    expected = {
        "model_family": "MuScaRi",
        "architecture_variant": ARCHITECTURE_VARIANT,
        "effort_transform": EFFORT_TRANSFORM,
        "asymptote_transform": ASYMPTOTE_TRANSFORM,
        "weibull_parameterization": WEIBULL_PARAMETERIZATION,
        "target_transform": TARGET_TRANSFORM,
        "layer_sizes": ";".join(map(str, LAYER_SIZES)),
    }
    for column, value in expected.items():
        if column not in benchmark or not benchmark[column].astype(str).eq(value).all():
            raise ValueError(f"Final benchmark has unexpected {column} metadata.")
    if not np.isclose(benchmark["train_frac"].astype(float), 1.0).all():
        raise ValueError("Final benchmark is not a full-training-set result.")


def train_fraction(fraction: float, features: list[str]) -> pd.DataFrame:
    config = TrainConfig(
        devices=DEVICES,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        run_root=ARTIFACT_ROOT,
        path_sbcv_data=SBCV_PATH,
        path_gift_data=GIFT_PATH,
        effort_transform=EFFORT_TRANSFORM,
        target_transform=TARGET_TRANSFORM,
        layer_sizes=LAYER_SIZES.copy(),
        fold_ids=FOLD_IDS,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_group=WANDB_GROUP,
        wandb_tags=["third-revision", "training-fraction-scaling", SBCV_DATASET_ID],
        wandb_config={
            "dataset_id": SBCV_DATASET_ID,
            "gift_dataset_id": GIFT_DATASET_ID,
            "fold": "set per run",
            "feature_set": "env_area",
            "architecture_variant": ARCHITECTURE_VARIANT,
            "effort_transform": EFFORT_TRANSFORM,
            "asymptote_transform": ASYMPTOTE_TRANSFORM,
            "weibull_parameterization": WEIBULL_PARAMETERIZATION,
            "target_transform": TARGET_TRANSFORM,
            "model_family": "MuScaRi",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epoch_limit": N_EPOCHS,
            "output_path": str(OUTPUT_PATH),
        },
        save_checkpoints=True,
        write_summary=True,
        overwrite=True,
        architecture_variant=ARCHITECTURE_VARIANT,
        muscari_asymptote_transform=ASYMPTOTE_TRANSFORM,
        muscari_weibull_parameterization=WEIBULL_PARAMETERIZATION,
        feature_set="env_area",
        metadata={
            "script": str(Path(__file__)),
            "dataset_id": SBCV_DATASET_ID,
            "gift_dataset_id": GIFT_DATASET_ID,
            "purpose": "supplementary_training_fraction_scaling",
        },
    )
    trainer = Trainer(config)
    results = trainer.run(
        "MuScaRi_ClimateDEM_Area",
        MuScaRiInit(features),
        features,
        train_frac=float(fraction),
        model_metadata={
            "model_family": "MuScaRi",
            "architecture_variant": ARCHITECTURE_VARIANT,
            "effort_transform": EFFORT_TRANSFORM,
            "asymptote_transform": ASYMPTOTE_TRANSFORM,
            "weibull_parameterization": WEIBULL_PARAMETERIZATION,
            "target_transform": TARGET_TRANSFORM,
            "layer_sizes": LAYER_SIZES.copy(),
        },
    )
    if len(results) != len(FOLD_IDS):
        raise RuntimeError(f"Fraction {fraction:g} produced {len(results)} of {len(FOLD_IDS)} folds.")
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = feature_names()
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        if "scaling_result_source" not in existing:
            existing["scaling_result_source"] = "dedicated_scaling_run"
        else:
            existing["scaling_result_source"] = existing["scaling_result_source"].fillna(
                "dedicated_scaling_run"
            )
        completed = [existing]
    else:
        existing = pd.DataFrame()
        completed = []
    for fraction in TRAIN_FRACTIONS:
        existing_fraction = (
            existing[np.isclose(existing["train_frac"], fraction)]
            if not existing.empty
            else existing
        )
        complete_folds = (
            set(existing_fraction["fold"].astype(int)) == set(FOLD_IDS)
            if not existing_fraction.empty
            else False
        )
        if len(existing_fraction) == len(FOLD_IDS) and complete_folds:
            print(f"Reusing completed fraction {fraction:g}", flush=True)
            continue
        if not existing_fraction.empty:
            print(f"Discarding partial fraction {fraction:g} before rerunning", flush=True)
            existing = existing[~np.isclose(existing["train_frac"], fraction)].copy()
            completed = [existing] if not existing.empty else []
        if np.isclose(fraction, 1.0):
            benchmark = pd.read_csv(FINAL_BENCHMARK_PATH)
            benchmark = benchmark[benchmark["experiment"] == "MuScaRi_ClimateDEM_Area"].copy()
            validate_final_benchmark(benchmark)
            benchmark["train_frac"] = 1.0
            benchmark["scaling_result_source"] = "final_benchmark"
            completed.append(benchmark)
            pd.concat(completed, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
            print(f"Reused final benchmark for fraction {fraction:g}", flush=True)
            continue
        print(f"Training fraction {fraction:g}", flush=True)
        result = train_fraction(float(fraction), features)
        result["scaling_result_source"] = "dedicated_scaling_run"
        completed.append(result)
        pd.concat(completed, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
        print(f"Updated {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

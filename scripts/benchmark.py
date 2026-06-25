"""
This script benchmarks the MuScaRi model on SBCV datasets, evaluating both
Interpolation (on SBCV test set) and Extrapolation (on GIFT dataset).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from muscari.utils import (
    add_effort_columns,
    compute_metrics,
    environmental_features,
    finish_wandb,
    landcover_fraction_features,
    log_wandb_metrics,
    maybe_wandb_logger,
    setup_logger,
    symmetric_arch,
)
from muscari.trainer import TrainConfig, Trainer
from muscari.muscari import MuScaRi
from muscari.ffnn import FFNNExp
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[1]
GIFT_DATASET_ID = "da569da"
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"

SBCV_ID = "ceacce0"
SBCV_SAMPLES_PATH = ROOT / "data/processed/training_samples/sbcv" / SBCV_ID

BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            # "bio4",
            # "rsds_1981-2010_range_V.2.1",
            "bio12",
            # "bio15",
        ]
INCLUDE_LANDCOVER_EXPERIMENTS = False

# TODO(agent): update these constants once the full architecture screen selects
# the final architecture.
SELECTED_ARCHITECTURE_NAME = "stable_maxabs_softplus"
EFFORT_TRANSFORM = "absolute"
ASYMPTOTE_TRANSFORM = "softplus"
WEIBULL_PARAMETERIZATION = "stable"
TARGET_TRANSFORM = "maxabs"

USE_WANDB = True
WANDB_PROJECT = "muscari-third-revision"
WANDB_TAGS = ["benchmark", "third-revision"]

SMOKE_TEST = os.environ.get("MUSCARI_SMOKE_TEST", "0") == "1"
FOLD_IDS = [0] if SMOKE_TEST else list(range(5))
N_EPOCHS = 1 if SMOKE_TEST else 100
TRAIN_FRAC = 0.002 if SMOKE_TEST else 1.0
RIDGE_ALPHAS = np.logspace(-6, 6, 13)
logger = setup_logger("benchmark")

@dataclass
class MuScaRiInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=128, factor=4))
    asymptote_transform: str = ASYMPTOTE_TRANSFORM
    weibull_parameterization: str = WEIBULL_PARAMETERIZATION
    def __call__(self, **kwargs):
        return MuScaRi(layer_sizes=self.architecture,
                       feature_names=self.feature_names,
                       ffnn_batchnorm=False,
                       asymptote_transform=self.asymptote_transform,
                       weibull_parameterization=self.weibull_parameterization,
                       **kwargs)

class WrappedFFNNExp(FFNNExp):
    def __init__(self, input_dim, layer_sizes, feature_names=None, feature_scaler=None, target_scaler=None, batchnorm=True):
        super().__init__(input_dim, layer_sizes, batchnorm=batchnorm)
        self.feature_names = feature_names or []
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

@dataclass
class FFNNExpInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=128, factor=4))
    batchnorm: bool = True
    def __call__(self, **kwargs):
        # input_dim = len(feature_names) + 1 (for log_observed_area)
        return WrappedFFNNExp(len(self.feature_names) + 1, self.architecture, batchnorm=self.batchnorm, **kwargs)


def log_linear_wandb(metrics, dataset_id, fold_id, feature_names):
    wandb_logger = maybe_wandb_logger(
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        group=f"benchmark_{dataset_id}",
        tags=WANDB_TAGS + [dataset_id, "linear"],
        name=f"Linear_ClimateDEM_Area_fold_{fold_id}",
        config={
            "dataset_id": dataset_id,
            "gift_dataset_id": GIFT_DATASET_ID,
            "fold": fold_id,
            "feature_names": feature_names,
            "model_family": "linear",
            "architecture_variant": SELECTED_ARCHITECTURE_NAME,
            "effort_transform": EFFORT_TRANSFORM,
            "ridge_alphas": RIDGE_ALPHAS.tolist(),
        },
    )
    log_wandb_metrics(wandb_logger, metrics)
    finish_wandb(wandb_logger)


def run_linear_baseline(config, dataset_id, feature_names, train_frac=1.0):
    rows = []
    columns = ["log_observed_area"] + feature_names
    gift_df = add_effort_columns(gpd.read_parquet(config.path_gift_data), config.effort_transform)
    gift_df = gift_df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ["sr"])

    for fold_id in config.fold_ids:
        train_df = add_effort_columns(
            gpd.read_parquet(config.path_sbcv_data / f"fold_{fold_id}_train.parquet"),
            config.effort_transform,
        )
        val_df = add_effort_columns(
            gpd.read_parquet(config.path_sbcv_data / f"fold_{fold_id}_val.parquet"),
            config.effort_transform,
        )
        test_df = add_effort_columns(
            gpd.read_parquet(config.path_sbcv_data / f"fold_{fold_id}_test.parquet"),
            config.effort_transform,
        )
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ["sr"])
        val_df = val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ["sr"])
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ["sr"])

        if train_frac < 1.0:
            train_df = train_df.sample(frac=train_frac, random_state=config.seed)

        best_model = None
        best_alpha = None
        best_rmse = np.inf
        for alpha in RIDGE_ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
            model.fit(train_df[columns].to_numpy(), np.log1p(train_df["sr"].to_numpy()))
            val_pred = np.clip(np.expm1(model.predict(val_df[columns].to_numpy())), 0, None)
            val_rmse = compute_metrics(val_df["sr"].to_numpy(), val_pred)["rmse"]
            if val_rmse < best_rmse:
                best_model = model
                best_alpha = float(alpha)
                best_rmse = val_rmse

        metrics = {
            "experiment": "Linear_ClimateDEM_Area",
            "fold": fold_id,
            "train_frac": train_frac,
            "n_train_samples": len(train_df),
            "ridge_alpha": best_alpha,
        }
        for prefix, df in [
            ("interp", test_df),
            ("extrap", gift_df),
        ]:
            y_true = df["sr"].to_numpy()
            y_pred = np.clip(np.expm1(best_model.predict(df[columns].to_numpy())), 0, None)
            for key, value in compute_metrics(y_true, y_pred).items():
                metrics[f"{prefix}_{key}"] = value

        log_linear_wandb(metrics, dataset_id, fold_id, feature_names)
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_experiments(climate_dem_feats, landcover_feats, all_env_feats):
    experiments = []

    experiments.append({
        "name": "MuScaRi_Area",
        "model_init": MuScaRiInit(feature_names=["log_sp_unit_area"]),
        "feature_names": ["log_sp_unit_area"],
        "train_frac": TRAIN_FRAC,
    })

    experiments.append({
        "name": "MuScaRi_ClimateDEM",
        "model_init": MuScaRiInit(feature_names=climate_dem_feats),
        "feature_names": climate_dem_feats,
        "train_frac": TRAIN_FRAC,
    })

    experiments.append({
        "name": "MuScaRi_ClimateDEM_Area",
        "model_init": MuScaRiInit(feature_names=climate_dem_feats + ["log_sp_unit_area"]),
        "feature_names": climate_dem_feats + ["log_sp_unit_area"],
        "train_frac": TRAIN_FRAC,
    })

    if INCLUDE_LANDCOVER_EXPERIMENTS:
        experiments.extend([
            {
                "name": "MuScaRi_Landcover",
                "model_init": MuScaRiInit(feature_names=landcover_feats),
                "feature_names": landcover_feats,
                "train_frac": TRAIN_FRAC,
            },
            {
                "name": "MuScaRi_ClimateDEM_Landcover",
                "model_init": MuScaRiInit(feature_names=climate_dem_feats + landcover_feats),
                "feature_names": climate_dem_feats + landcover_feats,
                "train_frac": TRAIN_FRAC,
            },
            {
                "name": "MuScaRi_Landcover_Area",
                "model_init": MuScaRiInit(feature_names=landcover_feats + ["log_sp_unit_area"]),
                "feature_names": landcover_feats + ["log_sp_unit_area"],
                "train_frac": TRAIN_FRAC,
            },
            {
                "name": "MuScaRi_All",
                "model_init": MuScaRiInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
                "feature_names": all_env_feats + ["log_sp_unit_area"],
                "train_frac": TRAIN_FRAC,
            },
            {
                "name": "FFNN_All",
                "model_init": FFNNExpInit(feature_names=all_env_feats + ["log_sp_unit_area"], batchnorm=True),
                "feature_names": all_env_feats + ["log_sp_unit_area"],
                "train_frac": TRAIN_FRAC,
            },
        ])

    experiments.append({
        "name": "FFNN_ClimateDEM_Area",
        "model_init": FFNNExpInit(feature_names=climate_dem_feats + ["log_sp_unit_area"], batchnorm=True),
        "feature_names": climate_dem_feats + ["log_sp_unit_area"],
        "train_frac": TRAIN_FRAC,
    })
    return experiments


if __name__ == "__main__":
    root_folder = Path(__file__).parent / Path("results", "benchmark")
    root_folder.mkdir(parents=True, exist_ok=True)
    artifact_root = root_folder / "artifacts"

    dataset_id = SBCV_ID
    config = TrainConfig(
        run_root=artifact_root,
        path_gift_data=GIFT_SAMPLES_PATH,
        path_sbcv_data=SBCV_SAMPLES_PATH,
        n_epochs=N_EPOCHS,
        effort_transform=EFFORT_TRANSFORM,
        target_transform=TARGET_TRANSFORM,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_group=f"benchmark_{dataset_id}",
        wandb_tags=WANDB_TAGS + [dataset_id, SELECTED_ARCHITECTURE_NAME],
        fold_ids=FOLD_IDS,
        architecture_variant=SELECTED_ARCHITECTURE_NAME,
        muscari_asymptote_transform=ASYMPTOTE_TRANSFORM,
        muscari_weibull_parameterization=WEIBULL_PARAMETERIZATION,
        save_checkpoints=True,
        write_summary=True,
        overwrite=SMOKE_TEST,
        metadata={
            "script": str(Path(__file__)),
            "dataset_id": dataset_id,
            "gift_dataset_id": GIFT_DATASET_ID,
        },
    )

    sample_file = next(config.path_sbcv_data.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)
    landcover_feats = landcover_fraction_features(df)
    climate_dem_feats = environmental_features(
        df,
        BIOCLIMATE_VARS,
        include_elevation=True,
        include_landcover=False,
    )
    all_env_feats = climate_dem_feats + landcover_feats
    logger.info("Identified %d environmental features.", len(all_env_feats))

    experiments = build_experiments(climate_dem_feats, landcover_feats, all_env_feats)
    trainer = Trainer(config)
    results = []
    logger.info("Running neural benchmarks (interpolation and extrapolation).")
    for exp in experiments:
        logger.info("Running experiment: %s", exp["name"])
        model_family = "FFNN" if exp["name"].startswith("FFNN") else "MuScaRi"
        config.run_root = artifact_root / exp["name"]
        results.append(
            trainer.run(
                exp["name"],
                exp["model_init"],
                exp["feature_names"],
                train_frac=exp["train_frac"],
                model_metadata={
                    "model_family": model_family,
                    "architecture_variant": SELECTED_ARCHITECTURE_NAME,
                    "asymptote_transform": ASYMPTOTE_TRANSFORM,
                    "weibull_parameterization": WEIBULL_PARAMETERIZATION,
                    "effort_transform": EFFORT_TRANSFORM,
                    "target_transform": TARGET_TRANSFORM,
                },
            )
        )

    logger.info("Running regularized linear baseline.")
    results.append(
        run_linear_baseline(
            config,
            dataset_id,
            climate_dem_feats + ["log_sp_unit_area"],
            train_frac=TRAIN_FRAC,
        )
    )

    df_results = pd.concat(results, ignore_index=True)
    suffix = "_smoke" if SMOKE_TEST else ""
    output_file = root_folder / f"benchmark_results_{dataset_id}{suffix}.csv"
    df_results.to_csv(output_file, index=False)
    logger.info("Benchmark completed, output saved at %s.", output_file)

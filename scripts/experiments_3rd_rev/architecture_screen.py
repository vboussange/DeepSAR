"""
Private architecture screen for the third MuScaRi revision.

Keep constants in this file explicit. This screen is not intended to generate
paper-facing figures; it selects the architecture used by later scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from muscari.dataset import create_dataloader
from muscari.muscari import MuScaRi
from muscari.trainer import MuScaRiLitModule, TrainConfig
from muscari.utils import (
    add_effort_columns,
    choose_accelerator,
    compute_metrics,
    evaluate_lit_model,
    feature_sets,
    finish_wandb,
    log_wandb_metrics,
    make_trainer,
    maybe_wandb_logger,
    residual_bias_slope,
    setup_logger,
    symmetric_arch,
)


ROOT = Path(__file__).parents[2]
SBCV_DATASET_ID = "ceacce0"
GIFT_DATASET_ID = "418c563"
SBCV_SAMPLES_PATH = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
GIFT_SAMPLES_PATH = ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
RUN_FOLDER = ROOT / "scripts/results/architecture_screen" / SBCV_DATASET_ID

BIOCLIMATE_VARS = [
    "bio1",
    "pet_penman_mean",
    "sfcWind_mean",
    "bio12",
]

USE_WANDB = False
WANDB_PROJECT = "muscari-third-revision"
WANDB_GROUP = f"architecture_screen_{SBCV_DATASET_ID}"
WANDB_TAGS = ["architecture-screen", SBCV_DATASET_ID]

SMOKE_TEST = os.environ.get("MUSCARI_SMOKE_TEST", "0") == "1"
FOLD_IDS = [0] if SMOKE_TEST else list(range(5))
N_EPOCHS = 1 if SMOKE_TEST else 100
TRAIN_FRAC = 0.002 if SMOKE_TEST else 1.0
BATCH_SIZE = 1024
NUM_WORKERS = 0 if SMOKE_TEST else 4
LAYER_SIZES = symmetric_arch(6, base=128, factor=4)

VARIANTS = [
    {
        "name": "current_abs",
        "effort_transform": "absolute",
        "asymptote_transform": "identity",
    },
    {
        "name": "exp_abs",
        "effort_transform": "absolute",
        "asymptote_transform": "exp",
    },
    {
        "name": "current_rel",
        "effort_transform": "relative",
        "asymptote_transform": "identity",
    },
    {
        "name": "exp_rel",
        "effort_transform": "relative",
        "asymptote_transform": "exp",
    },
]

logger = setup_logger("architecture_screen")


def prepare_split(path: Path, feature_names: list[str], effort_transform: str) -> gpd.GeoDataFrame:
    df = gpd.read_parquet(path)
    df = add_effort_columns(df, effort_transform)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=["log_observed_area", "sr"] + feature_names, inplace=True)
    return df


def train_and_evaluate_fold(variant: dict, fold_id: int, feature_names: list[str]) -> dict:
    pl.seed_everything(1 + fold_id)

    train_df = prepare_split(
        SBCV_SAMPLES_PATH / f"fold_{fold_id}_train.parquet",
        feature_names,
        variant["effort_transform"],
    )
    val_df = prepare_split(
        SBCV_SAMPLES_PATH / f"fold_{fold_id}_val.parquet",
        feature_names,
        variant["effort_transform"],
    )
    test_df = prepare_split(
        SBCV_SAMPLES_PATH / f"fold_{fold_id}_test.parquet",
        feature_names,
        variant["effort_transform"],
    )
    gift_df = prepare_split(GIFT_SAMPLES_PATH, feature_names, variant["effort_transform"])

    if TRAIN_FRAC < 1.0:
        train_df = train_df.sample(frac=TRAIN_FRAC, random_state=1 + fold_id)

    config = TrainConfig(
        path_sbcv_data=SBCV_SAMPLES_PATH,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        muscari_batchnorm=False,
        muscari_asymptote_transform=variant["asymptote_transform"],
        effort_transform=variant["effort_transform"],
        layer_sizes=LAYER_SIZES,
        run_folder=RUN_FOLDER / variant["name"],
    )

    train_loader, feature_scaler, target_scaler = create_dataloader(
        train_df, feature_names, config.batch_size, config.num_workers
    )
    val_loader, _, _ = create_dataloader(
        val_df,
        feature_names,
        config.batch_size,
        config.num_workers,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        shuffle=False,
    )
    test_loader, _, _ = create_dataloader(
        test_df,
        feature_names,
        config.batch_size,
        config.num_workers,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        shuffle=False,
    )

    model = MuScaRi(
        config.layer_sizes,
        feature_names=feature_names,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ffnn_batchnorm=config.muscari_batchnorm,
        asymptote_transform=variant["asymptote_transform"],
    )
    lit_model = MuScaRiLitModule(model, config, torch.nn.MSELoss())

    run_name = f"{variant['name']}_fold_{fold_id}"
    wandb_logger = maybe_wandb_logger(
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        tags=WANDB_TAGS + [variant["name"]],
        name=run_name,
        config={
            "dataset_id": SBCV_DATASET_ID,
            "gift_dataset_id": GIFT_DATASET_ID,
            "fold": fold_id,
            "feature_set": "env_area",
            "feature_names": feature_names,
            "architecture_variant": variant["name"],
            "effort_transform": variant["effort_transform"],
            "asymptote_transform": variant["asymptote_transform"],
            "n_epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "train_frac": TRAIN_FRAC,
            "run_folder": str(config.run_folder),
        },
    )

    trainer = make_trainer(config.n_epochs, wandb_logger, config.lr_scheduler_patience)
    trainer.fit(lit_model, train_loader, val_loader)
    _, _, eval_device = choose_accelerator()

    y_train, yhat_train = evaluate_lit_model(lit_model, train_loader, eval_device)
    y_val, yhat_val = evaluate_lit_model(lit_model, val_loader, eval_device)
    y_test, yhat_test = evaluate_lit_model(lit_model, test_loader, eval_device)

    model.eval().to(eval_device)
    y_gift = gift_df["sr"].to_numpy()
    yhat_gift = model.predict_sr_tot(gift_df)

    metrics = {
        "experiment": variant["name"],
        "fold": fold_id,
        "n_train_samples": len(train_df),
        "train_frac": TRAIN_FRAC,
        "effort_transform": variant["effort_transform"],
        "asymptote_transform": variant["asymptote_transform"],
    }
    for prefix, y, yhat, df in [
        ("train", y_train, yhat_train, train_df),
        ("val", y_val, yhat_val, val_df),
        ("test", y_test, yhat_test, test_df),
        ("gift", y_gift, yhat_gift, gift_df),
    ]:
        for key, value in compute_metrics(y, yhat).items():
            metrics[f"{prefix}_{key}"] = value
        metrics[f"{prefix}_bias_slope_log_area"] = residual_bias_slope(
            y, yhat, df["log_sp_unit_area"].to_numpy()
        )

    config.run_folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": feature_names,
            "config": config,
            "variant": variant,
            "asymptote_transform": variant["asymptote_transform"],
            "metrics": metrics,
        },
        config.run_folder / f"fold_{fold_id}.pth",
    )
    log_wandb_metrics(wandb_logger, metrics)
    finish_wandb(wandb_logger)
    return metrics


if __name__ == "__main__":
    RUN_FOLDER.mkdir(parents=True, exist_ok=True)
    sample_df = gpd.read_parquet(next(SBCV_SAMPLES_PATH.glob("*_train.parquet")))
    feature_names = feature_sets(sample_df, BIOCLIMATE_VARS)["env_area"]
    logger.info("Architecture screen features: %s", feature_names)

    rows = []
    for variant in VARIANTS:
        for fold_id in FOLD_IDS:
            logger.info("Running %s fold %s", variant["name"], fold_id)
            rows.append(train_and_evaluate_fold(variant, fold_id, feature_names))

    results = pd.DataFrame(rows)
    suffix = "_smoke" if SMOKE_TEST else ""
    output_file = RUN_FOLDER / f"architecture_screen_results_{SBCV_DATASET_ID}{suffix}.csv"
    results.to_csv(output_file, index=False)
    logger.info("Saved architecture screen results to %s", output_file)

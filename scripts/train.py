"""
Training MuScaRi models on CV folds.
"""
import logging
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import geopandas as gpd
from pathlib import Path

from muscari.muscari import MuScaRi
from muscari.trainer import MuScaRiLitModule, TrainConfig
from muscari.dataset import create_dataloader
from muscari.utils import (
    add_effort_columns,
    choose_accelerator,
    climate_dem_features,
    compute_metrics,
    evaluate_lit_model,
    finish_wandb,
    log_wandb_metrics,
    make_trainer,
    maybe_wandb_logger,
    symmetric_arch,
)

SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/ceacce0"
RUN_FOLDER = Path(__file__).parent / f"results/train/{SBCV_SAMPLES_PATH.name}"
BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            # "bio4",
            # "rsds_1981-2010_range_V.2.1",
            "bio12",
            # "bio15",
        ]

SELECTED_ARCHITECTURE_NAME = "current_abs"
EFFORT_TRANSFORM = "absolute"
ASYMPTOTE_TRANSFORM = "identity"

USE_WANDB = False
WANDB_PROJECT = "muscari-third-revision"
WANDB_GROUP = f"train_{SBCV_SAMPLES_PATH.name}"
WANDB_TAGS = ["train", "third-revision", SBCV_SAMPLES_PATH.name]

SMOKE_TEST = os.environ.get("MUSCARI_SMOKE_TEST", "0") == "1"
FOLD_IDS = [0] if SMOKE_TEST else list(range(5))
N_EPOCHS = 1 if SMOKE_TEST else 100
TRAIN_FRAC = 0.002 if SMOKE_TEST else 1.0

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_wandb_logger(fold_id, feature_names, config):
    return maybe_wandb_logger(
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        tags=WANDB_TAGS,
        name=f"{SBCV_SAMPLES_PATH.name}_fold_{fold_id}",
        config={
            "dataset_id": SBCV_SAMPLES_PATH.name,
            "fold": fold_id,
            "feature_names": feature_names,
            "architecture_variant": SELECTED_ARCHITECTURE_NAME,
            "effort_transform": EFFORT_TRANSFORM,
            "asymptote_transform": ASYMPTOTE_TRANSFORM,
            "n_epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "train_frac": TRAIN_FRAC,
            "run_folder": str(config.run_folder),
        },
    )


def train_fold(config: TrainConfig, fold_id, feature_names):
    logger.info(f"Training Fold {fold_id}...")
    pl.seed_everything(config.seed + fold_id)
    
    train_path = config.path_sbcv_data / f"fold_{fold_id}_train.parquet"
    val_path = config.path_sbcv_data / f"fold_{fold_id}_val.parquet"
    test_path = config.path_sbcv_data / f"fold_{fold_id}_test.parquet"
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        logger.warning(f"Fold {fold_id} data not found. Skipping.")
        return

    train_df = gpd.read_parquet(train_path)
    val_df = gpd.read_parquet(val_path)
    test_df = gpd.read_parquet(test_path)
    if TRAIN_FRAC < 1.0:
        train_df = train_df.sample(frac=TRAIN_FRAC, random_state=config.seed + fold_id)
    
    train_df = add_effort_columns(train_df, EFFORT_TRANSFORM)
    val_df = add_effort_columns(val_df, EFFORT_TRANSFORM)
    test_df = add_effort_columns(test_df, EFFORT_TRANSFORM)
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)
        
    # Create dataloaders
    train_loader, feature_scaler, target_scaler = create_dataloader(
        train_df, feature_names, config.batch_size, config.num_workers
    )
    val_loader, _, _ = create_dataloader(
        val_df, feature_names, config.batch_size, config.num_workers,
        feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
    )
    
    # Initialize model
    model = MuScaRi(config.layer_sizes, 
                    feature_names=feature_names,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    ffnn_batchnorm=config.muscari_batchnorm,
                    asymptote_transform=ASYMPTOTE_TRANSFORM)
    
    _, _, eval_device = choose_accelerator()

    lit_model = MuScaRiLitModule(model, config, nn.MSELoss())
    wandb_logger = make_wandb_logger(fold_id, feature_names, config)

    trainer = make_trainer(config.n_epochs, wandb_logger, config.lr_scheduler_patience)
    trainer.fit(lit_model, train_loader, val_loader)

    test_loader, _, _ = create_dataloader(
        test_df, feature_names, config.batch_size, config.num_workers,
        feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
    )

    y_true_train, y_pred_train = evaluate_lit_model(lit_model, train_loader, eval_device)
    y_true_val, y_pred_val = evaluate_lit_model(lit_model, val_loader, eval_device)
    y_true_test, y_pred_test = evaluate_lit_model(lit_model, test_loader, eval_device)

    metrics = {
        "train": compute_metrics(y_true_train, y_pred_train),
        "val": compute_metrics(y_true_val, y_pred_val),
        "test": compute_metrics(y_true_test, y_pred_test),
    }
    if wandb_logger is not None:
        log_wandb_metrics(wandb_logger, {
            f"{split}_{metric}": value
            for split, split_metrics in metrics.items()
            for metric, value in split_metrics.items()
        })
        finish_wandb(wandb_logger)
    
    # Save model
    save_path = config.run_folder / f"fold_{fold_id}.pth"
    logger.info(f"Saving model to {save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "config": config,
        "metrics": metrics,
    }, save_path)

if __name__ == "__main__":
    RUN_FOLDER.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(path_sbcv_data=SBCV_SAMPLES_PATH,
                         num_workers=4,
                         n_epochs=N_EPOCHS,
                         muscari_batchnorm=False,
                         muscari_asymptote_transform=ASYMPTOTE_TRANSFORM,
                         effort_transform=EFFORT_TRANSFORM,
                         layer_sizes=symmetric_arch(6, base=128, factor=4),
                         run_folder=RUN_FOLDER)
    
    sample_file = next(config.path_sbcv_data.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)
    
    feature_names = climate_dem_features(df, BIOCLIMATE_VARS) + ["log_sp_unit_area"]
    logger.info(f"Training with features: {feature_names}")
        
    for fold_id in FOLD_IDS:
        train_fold(config, fold_id, feature_names)

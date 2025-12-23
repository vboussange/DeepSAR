"""
Training DeepSAR models on CV folds.
"""
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass, field

from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.trainer import DeepSARLitModule
from deepsar.dataset import create_dataloader
from deepsar.utils import symmetric_arch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HASH = "0b85791"

def train_fold(config, fold_id, feature_names):
    logger.info(f"Training Fold {fold_id}...")
    pl.seed_everything(config.seed + fold_id)
    
    train_path = config.cv_data_path / f"fold_{fold_id}_train.parquet"
    val_path = config.cv_data_path / f"fold_{fold_id}_test.parquet"
    
    if not train_path.exists() or not val_path.exists():
        logger.warning(f"Fold {fold_id} data not found. Skipping.")
        return

    train_df = gpd.read_parquet(train_path)
    val_df = gpd.read_parquet(val_path)
    
    # Preprocess
    for df in [train_df, val_df]:
        df["log_observed_area"] = np.log(df["observed_area"])
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
    model = Deep4PWeibull(config.layer_sizes, 
                          feature_names=feature_names,
                          feature_scaler=feature_scaler,
                          target_scaler=target_scaler)
    
    # Determine accelerator
    device = config.devices[fold_id % len(config.devices)]
    if "cuda" in device:
        accelerator = "gpu"
        devices = [int(device.split(":")[1])]
    elif "mps" in device:
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
        
    lit_model = DeepSARLitModule(model, config, nn.MSELoss())
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=config.lr_scheduler_patience * 2)],
        enable_progress_bar=True,
    )
    
    trainer.fit(lit_model, train_loader, val_loader)
    
    # Save model
    save_path = config.run_folder / f"fold_{fold_id}.pth"
    logger.info(f"Saving model to {save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "config": config
    }, save_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
        
    config = TrainConfig(devices=devices, hash_data=HASH)
    
    # Identify features from first fold
    try:
        sample_file = next(config.cv_data_path.glob("*_train.parquet"))
        df = gpd.read_parquet(sample_file)
        
        climate_feats = config.climate_variables + [f"std_{v}" for v in config.climate_variables]
        dem_feats = ["elevation", "std_elevation"]
        lc_feats = [c for c in df.columns if c.startswith("lc_frac_")]
        
        climate_feats = [c for c in climate_feats if c in df.columns]
        dem_feats = [c for c in dem_feats if c in df.columns]
        
        all_env_feats = climate_feats + dem_feats + lc_feats
        feature_names = all_env_feats + ["log_sp_unit_area"]
        logger.info(f"Training with features: {feature_names}")
        
    except StopIteration:
        logger.error(f"No training files found in {config.cv_data_path}.")
        exit(1)
        
    for fold_id in range(5):
        train_fold(config, fold_id, feature_names)

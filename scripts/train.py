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
from deepsar.trainer import DeepSARLitModule, TrainConfig
from deepsar.dataset import create_dataloader
from deepsar.utils import symmetric_arch

SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/606e055"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_fold(config: TrainConfig, fold_id, feature_names):
    logger.info(f"Training Fold {fold_id}...")
    pl.seed_everything(config.seed + fold_id)
    
    train_path = config.sbcv_path / f"fold_{fold_id}_train.parquet"
    val_path = config.sbcv_path / f"fold_{fold_id}_val.parquet"
    
    if not train_path.exists() or not val_path.exists():
        logger.warning(f"Fold {fold_id} data not found. Skipping.")
        return

    train_df = gpd.read_parquet(train_path)
    val_df = gpd.read_parquet(val_path)
    
    # Preprocess
    for df in [train_df, val_df]:
        df["log_observed_area"] = np.log(df["observed_area"])
        df["log_sp_unit_area"] = np.log(df["sp_unit_area"])
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
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1 # TODO: you could set this to torch.cuda.device_count()
    elif torch.backends.mps.is_available():
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
    config = TrainConfig(sbcv_path=SBCV_SAMPLES_PATH,
                         num_workers=4)
    
    sample_file = next(config.sbcv_path.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)
    
    climate_feats = config.climate_variables + [f"std_{v}" for v in config.climate_variables]
    dem_feats = ["elevation", "std_elevation"]
    lc_feats = [c for c in df.columns if c.startswith("lc_frac_")]
    
    climate_feats = [c for c in climate_feats if c in df.columns]
    dem_feats = [c for c in dem_feats if c in df.columns]
    
    all_env_feats = climate_feats + dem_feats + lc_feats
    feature_names = all_env_feats + ["log_sp_unit_area"]
    logger.info(f"Training with features: {feature_names}")
        
    for fold_id in range(5):
        train_fold(config, fold_id, feature_names)

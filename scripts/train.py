"""
Training MuScaRi models on CV folds.
"""
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.metrics import (
    d2_absolute_error_score,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from muscari.muscari import MuScaRi
from muscari.trainer import MuScaRiLitModule, TrainConfig
from muscari.dataset import create_dataloader
from muscari.utils import symmetric_arch

SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/ceacce0"
RUN_FOLDER = Path(__file__).parent / f"results/train/{SBCV_SAMPLES_PATH.name}_full_clim_features"
BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            "bio4",
            "rsds_1981-2010_range_V.2.1",
            "bio12",
            "bio15",
        ]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(lit_model, data_loader, device):
    lit_model.eval()
    lit_model.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y_pred = lit_model(X)
            preds.append(y_pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    if lit_model.model.target_scaler:
        preds = lit_model.model.target_scaler.inverse_transform(preds)
        targets = lit_model.model.target_scaler.inverse_transform(targets.reshape(-1, 1))

    return targets.flatten(), preds.flatten()


def compute_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "d2": d2_absolute_error_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


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
    
    # Preprocess
    for df in [train_df, val_df, test_df]:
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
    model = MuScaRi(config.layer_sizes, 
                    feature_names=feature_names,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler)
    
    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1 # TODO: you could set this to torch.cuda.device_count()
        eval_device = "cuda:0"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        eval_device = "mps"
    else:
        accelerator = "cpu"
        devices = 1
        eval_device = "cpu"
        
    lit_model = MuScaRiLitModule(model, config, nn.MSELoss())
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=config.lr_scheduler_patience * 2)],
        enable_progress_bar=False,
    )
    
    trainer.fit(lit_model, train_loader, val_loader)

    test_loader, _, _ = create_dataloader(
        test_df, feature_names, config.batch_size, config.num_workers,
        feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
    )

    y_true_train, y_pred_train = evaluate_model(lit_model, train_loader, eval_device)
    y_true_val, y_pred_val = evaluate_model(lit_model, val_loader, eval_device)
    y_true_test, y_pred_test = evaluate_model(lit_model, test_loader, eval_device)

    metrics = {
        "train": compute_metrics(y_true_train, y_pred_train),
        "val": compute_metrics(y_true_val, y_pred_val),
        "test": compute_metrics(y_true_test, y_pred_test),
    }
    
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
                         layer_sizes=symmetric_arch(6, base=128, factor=4),
                         run_folder=RUN_FOLDER)
    
    sample_file = next(config.path_sbcv_data.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)
    
    climate_feats = BIOCLIMATE_VARS + [f"std_{v}" for v in BIOCLIMATE_VARS]
    dem_feats = ["elevation", "std_elevation"]
    lc_feats = [c for c in df.columns if c.startswith("lc_frac_")]
    
    climate_feats = [c for c in climate_feats if c in df.columns]
    dem_feats = [c for c in dem_feats if c in df.columns]
    
    feature_names = climate_feats + dem_feats + ["log_sp_unit_area"] # + lc_feats 
    logger.info(f"Training with features: {feature_names}")
        
    for fold_id in range(5):
        train_fold(config, fold_id, feature_names)

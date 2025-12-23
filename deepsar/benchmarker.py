import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import (d2_absolute_error_score, root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from deepsar.dataset import create_dataloader
from deepsar.trainer import DeepSARLitModule
import torch.multiprocessing as mp
import logging

@dataclass
class BenchmarkConfig:
    devices: list = field(default_factory=lambda: [])
    seed: int = 1
    nruns: int = 5
    hash_data: str = ""
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    val_size: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    climate_variables: list = field(
        default_factory=lambda: []
    )
    run_name: str = ""
    path_sbcv_data: Path = None
    path_gift_data: Path = None

    def __post_init__(self):
        root = Path(__file__).parent
        if self.path_sbcv_data is None:
            self.path_sbcv_data = (
                root
                / "../data"
                / "processed"
                / "training_samples"
                / "sbcv"
                / self.hash_data
            )
        if self.path_gift_data is None:
            self.path_gift_data = (
                root
                / "../data"
                / "processed"
                / "GIFT_CHELSA_compilation"
                / "6c2d61d" # TODO: make this configurable or dynamic
                / "sp_unit_data.parquet"
            )


class Benchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        pl.seed_everything(config.seed)
        self.devices = config.devices
        self.nruns = config.nruns
        self.sbcv_path = config.path_sbcv_data
        self.gift_path = config.path_gift_data
        
        # Load GIFT once
        self.gift_df = gpd.read_parquet(self.gift_path)
        self.gift_df["log_observed_area"] = np.log(self.gift_df["observed_area"])
        self.gift_df.dropna(inplace=True)

    def _train(self, model, train_loader, val_loader, loss_fn, device):
        lit_model = DeepSARLitModule(model, self.config, loss_fn)
        
        # Determine accelerator and device
        if "cuda" in device:
            accelerator = "gpu"
            devices = [int(device.split(":")[1])]
        elif "mps" in device:
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = 1

        trainer = pl.Trainer(
            max_epochs=self.config.n_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_checkpointing=False,
            logger=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=self.config.lr_scheduler_patience * 2)],
            enable_progress_bar=False,
        )
        
        trainer.fit(lit_model, train_loader, val_loader)
        return lit_model

    def _evaluate(self, lit_model, test_loader, device):
        lit_model.eval()
        lit_model.to(device)
        
        preds = []
        targets = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y_pred = lit_model(X)
                preds.append(y_pred.cpu())
                targets.append(y.cpu())
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        
        # Inverse transform
        if lit_model.model.target_scaler:
            preds = lit_model.model.target_scaler.inverse_transform(preds)
            targets = lit_model.model.target_scaler.inverse_transform(targets.reshape(-1, 1))
            
        return targets.flatten(), preds.flatten()

    def _compute_metrics(self, y_true, y_pred):
        return {
            "r2": r2_score(y_true, y_pred),
            "d2": d2_absolute_error_score(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }

    def run(self, experiment_name, model_init, feature_names, train_frac=1.0):
        results = []
        
        # Iterate over folds (assuming 5 folds as per compilation scripts)
        for fold_id in range(5):
            # Load data
            train_path = self.sbcv_path / f"fold_{fold_id}_train.parquet"
            test_path = self.sbcv_path / f"fold_{fold_id}_test.parquet"
            
            if not train_path.exists() or not test_path.exists():
                logging.warning(f"Fold {fold_id} data not found at {self.sbcv_path}. Skipping.")
                continue
                
            train_df = gpd.read_parquet(train_path)
            test_df = gpd.read_parquet(test_path)
            
            # Subsample training data if needed
            if train_frac < 1.0:
                train_df = train_df.sample(frac=train_frac, random_state=self.config.seed)
            
            # Split train into train/val
            train_df, val_df = train_test_split(train_df, test_size=self.config.val_size, random_state=self.config.seed)
            
            # Preprocess (log area)
            for df in [train_df, val_df, test_df]:
                df["log_observed_area"] = np.log(df["observed_area"])
                # Ensure feature columns exist and handle NaNs
                df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)
                
            # Ensure GIFT has features
            gift_df = self.gift_df.copy()
            gift_df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)

            # Run multiple runs for this fold
            for run_id in range(self.nruns):
                device = self.devices[run_id % len(self.devices)]
                
                # Create dataloaders
                train_loader, feature_scaler, target_scaler = create_dataloader(
                    train_df, feature_names, self.config.batch_size, self.config.num_workers
                )
                val_loader, _, _ = create_dataloader(
                    val_df, feature_names, self.config.batch_size, self.config.num_workers, 
                    feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
                )
                test_loader_interp, _, _ = create_dataloader(
                    test_df, feature_names, self.config.batch_size, self.config.num_workers,
                    feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
                )
                test_loader_extrap, _, _ = create_dataloader(
                    gift_df, feature_names, self.config.batch_size, self.config.num_workers,
                    feature_scaler=feature_scaler, target_scaler=target_scaler, shuffle=False
                )
                
                # Initialize model
                model = model_init(feature_names=feature_names, feature_scaler=feature_scaler, target_scaler=target_scaler)
                loss_fn = torch.nn.MSELoss() # Default loss
                
                # Train
                lit_model = self._train(model, train_loader, val_loader, loss_fn, device)
                
                # Evaluate (Interpolation)
                y_true_interp, y_pred_interp = self._evaluate(lit_model, test_loader_interp, device)
                
                # Evaluate (Extrapolation)
                y_true_extrap, y_pred_extrap = self._evaluate(lit_model, test_loader_extrap, device)
                
                # Metrics
                metrics_interp = self._compute_metrics(y_true_interp, y_pred_interp)
                metrics_extrap = self._compute_metrics(y_true_extrap, y_pred_extrap)
                
                # Combine results
                combined_metrics = {
                    "experiment": experiment_name,
                    "fold": fold_id,
                    "run": run_id,
                    "train_frac": train_frac,
                    "n_train_samples": len(train_df)
                }
                
                for k, v in metrics_interp.items():
                    combined_metrics[f"interp_{k}"] = v
                for k, v in metrics_extrap.items():
                    combined_metrics[f"extrap_{k}"] = v
                    
                results.append(combined_metrics)
                
        return pd.DataFrame(results)

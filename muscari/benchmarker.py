from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import geopandas as gpd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from muscari.dataset import create_dataloader
from muscari.trainer import MuScaRiLitModule
from muscari.utils import (
    add_effort_columns,
    compute_metrics,
    evaluate_lit_model,
    finish_wandb,
    log_wandb_metrics,
    maybe_wandb_logger,
)
import torch.multiprocessing as mp
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

@dataclass
class BenchmarkConfig:
    devices: list = field(default_factory=lambda: [])
    seed: int = 1
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    climate_variables: list = field(
        default_factory=lambda: []
    )
    path_sbcv_data: Path = None
    path_gift_data: Path = None
    effort_transform: str = "absolute"
    use_wandb: bool = False
    wandb_project: str = "muscari-third-revision"
    wandb_group: str = "benchmark"
    wandb_tags: list = field(default_factory=lambda: [])
    fold_ids: list = field(default_factory=lambda: list(range(5)))

    def __post_init__(self):
        if not self.devices:
            if torch.cuda.is_available():
                self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif torch.backends.mps.is_available():
                self.devices = ["mps"]
            else:
                self.devices = ["cpu"]


class Benchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        pl.seed_everything(config.seed)
        self.devices = config.devices
        self.sbcv_path = config.path_sbcv_data
        self.gift_path = config.path_gift_data
        
        # Load GIFT once
        self.gift_df = gpd.read_parquet(self.gift_path)
        self.gift_df = self._add_effort_columns(self.gift_df)
        self.gift_df.dropna(inplace=True)

    def _add_effort_columns(self, df):
        return add_effort_columns(df, self.config.effort_transform)

    def _make_wandb_logger(self, experiment_name, fold_id, feature_names, train_frac):
        return maybe_wandb_logger(
            use_wandb=self.config.use_wandb,
            project=self.config.wandb_project,
            group=self.config.wandb_group,
            tags=self.config.wandb_tags + [experiment_name],
            name=f"{experiment_name}_fold_{fold_id}",
            config={
                "experiment": experiment_name,
                "fold": fold_id,
                "feature_names": feature_names,
                "train_frac": train_frac,
                "path_sbcv_data": str(self.config.path_sbcv_data),
                "path_gift_data": str(self.config.path_gift_data),
                "effort_transform": self.config.effort_transform,
            },
        )

    def _train(self, model, train_loader, val_loader, loss_fn, device, wandb_logger=False):
        lit_model = MuScaRiLitModule(model, self.config, loss_fn)
        
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
            logger=wandb_logger if wandb_logger is not None else False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=self.config.lr_scheduler_patience * 2)],
            enable_progress_bar=False,
        )
        
        trainer.fit(lit_model, train_loader, val_loader)
        return lit_model

    def _evaluate(self, lit_model, test_loader, device):
        return evaluate_lit_model(lit_model, test_loader, device)

    def _compute_metrics(self, y_true, y_pred):
        return compute_metrics(y_true, y_pred)

    def _train_fold(self, fold_id, device, experiment_name, model_init, feature_names, train_frac):
        """Train and evaluate a single fold on a specific device."""
        try:
            # Set seed for reproducibility
            pl.seed_everything(self.config.seed + fold_id)
            
            # Load data
            train_path = self.sbcv_path / f"fold_{fold_id}_train.parquet"
            val_path = self.sbcv_path / f"fold_{fold_id}_val.parquet"
            test_path = self.sbcv_path / f"fold_{fold_id}_test.parquet"
            
            if not train_path.exists() or not test_path.exists():
                logging.warning(f"Fold {fold_id} data not found at {self.sbcv_path}. Skipping.")
                return None
                
            train_df = gpd.read_parquet(train_path)
            val_df = gpd.read_parquet(val_path)
            test_df = gpd.read_parquet(test_path)
            
            # Subsample training data if needed
            if train_frac < 1.0:
                train_df = train_df.sample(frac=train_frac, random_state=self.config.seed)
            
            # Preprocess (log area)
            train_df = self._add_effort_columns(train_df)
            val_df = self._add_effort_columns(val_df)
            test_df = self._add_effort_columns(test_df)
            for df in [train_df, val_df, test_df]:
                # Ensure feature columns exist and handle NaNs
                df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)
                
            # Ensure GIFT has features
            gift_df = self.gift_df.copy()
            gift_df.dropna(subset=["log_observed_area"] + feature_names, inplace=True)
            
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
            model = model_init(feature_scaler=feature_scaler, target_scaler=target_scaler)
            loss_fn = torch.nn.MSELoss() # Default loss
            wandb_logger = self._make_wandb_logger(
                experiment_name,
                fold_id,
                feature_names,
                train_frac,
            )
            
            # Train
            lit_model = self._train(model, train_loader, val_loader, loss_fn, device, wandb_logger)
            
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
                "train_frac": train_frac,
                "n_train_samples": len(train_df)
            }
            
            for k, v in metrics_interp.items():
                combined_metrics[f"interp_{k}"] = v
            for k, v in metrics_extrap.items():
                combined_metrics[f"extrap_{k}"] = v
            if wandb_logger is not None:
                log_wandb_metrics(wandb_logger, combined_metrics)
                finish_wandb(wandb_logger)
            
            logging.info(f"Completed fold {fold_id} on device {device}")
            return combined_metrics
            
        except Exception as e:
            logging.error(f"Error processing fold {fold_id} on device {device}: {e}")
            return None

    def run(self, experiment_name, model_init, feature_names, train_frac=1.0):
        """Run benchmark across all folds, utilizing multiple GPUs in parallel."""
        results = []
        
        # Prepare fold configurations
        fold_configs = []
        for fold_id in self.config.fold_ids:
            device = self.devices[fold_id % len(self.devices)]
            fold_configs.append((fold_id, device))
        
        # If only one device or sequential execution requested, run sequentially
        if len(self.devices) == 1:
            logging.info("Single device detected. Running folds sequentially.")
            for fold_id, device in fold_configs:
                result = self._train_fold(fold_id, device, experiment_name, model_init, feature_names, train_frac)
                if result is not None:
                    results.append(result)
        else:
            # Parallel execution across multiple GPUs
            logging.info(f"Multiple devices detected ({len(self.devices)}). Running folds in parallel.")
            
            # Use ProcessPoolExecutor to train folds in parallel
            max_workers = min(len(self.devices), len(fold_configs))
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
                # Submit all fold training jobs
                future_to_fold = {
                    executor.submit(
                        self._train_fold, 
                        fold_id, 
                        device, 
                        experiment_name, 
                        model_init, 
                        feature_names, 
                        train_frac
                    ): fold_id
                    for fold_id, device in fold_configs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_fold):
                    fold_id = future_to_fold[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Fold {fold_id} generated an exception: {e}")
                
        return pd.DataFrame(results)

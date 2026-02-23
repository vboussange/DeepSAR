import torch
from torch import nn
from torch import multiprocessing as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from dataclasses import dataclass, field
from muscari.deep4pweibull import Deep4PWeibull
from muscari.trainer import MuScaRiLitModule
from muscari.dataset import create_dataloader
from muscari.utils import symmetric_arch
from muscari.ensemble_model import MuScaRiEnsembleModel

@dataclass
class EnsembleConfig:
    devices: list = field(default_factory=lambda: [])
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    val_size: float = 0.1
    lr: float = 1e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    weight_decay: float = 1e-4
    seed: int = 1
    hash_data: str = ""
    predictors: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    n_ensembles: int = 5  # Number of models in the ensemble
    run_name: str = ""
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: symmetric_arch(6, base=32, factor=4))
    path_eva_data: str = ""

class EnsembleTrainer:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.devices = config.devices
        self.n_ensembles = config.n_ensembles

    def run(self, feature_names):
        df_test = self.df[self.df["test"] == True]
        df_train_val = self.df[self.df["test"] == False]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.n_ensembles) as pool:
            args = [
                (self.config, df_train_val, df_test, feature_names, i, self.devices[i % len(self.devices)])
                for i in range(self.n_ensembles)
            ]
            results = pool.starmap(self._single_run, args)
        
        models = [r["model"] for r in results]
        logs = [r["log"] for r in results]
        feature_scalers = [r["model"].feature_scaler for r in results]
        target_scalers = [r["model"].target_scaler for r in results]
        ensemble_model = MuScaRiEnsembleModel(models)
        
        pred_s = ensemble_model.predict_mean_sr(df_test)
        ensemble_mse = root_mean_squared_error(df_test["sr"], pred_s)
        ensemble_r2 = r2_score(df_test["sr"], pred_s)

        return {
            "ensemble_model_state_dict": ensemble_model.state_dict(),
            "logs": logs,
            "feature_scalers": feature_scalers,
            "target_scalers": target_scalers,
            "mean_squared_error_test": ensemble_mse,
            "r2_test": ensemble_r2,
            "feature_names": feature_names,
        }

    @staticmethod
    def _single_run(config, df_train_val, df_test, feature_names, ensemble_idx, device):

        pl.seed_everything(config.seed + ensemble_idx)

        train_idx, val_idx = train_test_split(
            df_train_val.index,
            test_size=config.val_size,
            random_state=config.seed + ensemble_idx,
        )
        df_train, df_val = df_train_val.loc[train_idx], df_train_val.loc[val_idx]

        train_loader, feature_scaler, target_scaler = create_dataloader(df_train, feature_names, config.batch_size, config.num_workers)
        val_loader, _, _ = create_dataloader(df_val, feature_names, config.batch_size, config.num_workers, feature_scaler, target_scaler)
        test_loader, _, _ = create_dataloader(df_test, feature_names, config.batch_size, config.num_workers, feature_scaler, target_scaler)

        model = Deep4PWeibull(config.layer_sizes, 
                              feature_names=feature_names,
                              feature_scaler=feature_scaler,
                              target_scaler=target_scaler)

        # Determine accelerator
        if "cuda" in device:
            accelerator = "gpu"
            devices = [int(device.split(":")[1])]
        elif "mps" in device:
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = 1

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

        return {
            "model": model.cpu(),
            "log": {k: v.item() for k, v in trainer.callback_metrics.items()}
        }

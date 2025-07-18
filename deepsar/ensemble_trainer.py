import torch
from torch import nn
from torch import multiprocessing as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import random

from dataclasses import dataclass, field
from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.trainer import Trainer
from deepsar.dataset import create_dataloader
from deepsar.utils import symmetric_arch
from deepsar.ensemble_model import EnsembleModel

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
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
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

    def run(self, predictors):
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.n_ensembles) as pool:
            args = [
                (self.config, self.df, predictors, i, self.devices[i % len(self.devices)])
                for i in range(self.n_ensembles)
            ]
            results = pool.starmap(self._single_run, args)

        models = [r["model"] for r in results]
        logs = [r["log"] for r in results]
        feature_scaler = results[0]["feature_scaler"]
        target_scaler = results[0]["target_scaler"]
        test_loader = results[0]["test_loader"]

        # Create ensemble model
        ensemble_model = EnsembleModel(models)
        ensemble_trainer = Trainer(
            config=self.config,
            model=ensemble_model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            train_loader=None,
            val_loader=None,
            test_loader=test_loader,
            compute_loss=nn.MSELoss(),
            device=self.devices[0]
        )
        targets, preds = ensemble_trainer.get_model_predictions(test_loader)
        pred_trs = target_scaler.inverse_transform(preds)
        target_trs = target_scaler.inverse_transform(targets)
        ensemble_mse = root_mean_squared_error(target_trs, pred_trs)
        ensemble_r2 = r2_score(target_trs, pred_trs)

        return {
            "ensemble_model_state_dict": ensemble_model.state_dict(),
            "logs": logs,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "mean_squared_error_test": ensemble_mse,
            "r2_test": ensemble_r2,
            "predictors": predictors,
        }

    @staticmethod
    def _single_run(config, df, predictors, ensemble_idx, device):

        gdf_train_val = df[df["test"] == False]
        gdf_test = df[df["test"] == True]
        train_val_loader, feature_scaler, target_scaler = create_dataloader(
            gdf_train_val, predictors, config.batch_size, config.num_workers
        )
        test_loader, _, _ = create_dataloader(
            gdf_test, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler
        )

        torch.manual_seed(config.seed + ensemble_idx)
        np.random.seed(config.seed + ensemble_idx)
        random.seed(config.seed + ensemble_idx)


        train_idx, val_idx = train_test_split(
            gdf_train_val.index,
            test_size=config.val_size,
            random_state=config.seed + ensemble_idx,
        )
        gdf_train, gdf_val = gdf_train_val.loc[train_idx], gdf_train_val.loc[val_idx]

        train_loader, _, _ = create_dataloader(
            gdf_train, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler
        )
        val_loader, _, _ = create_dataloader(
            gdf_val, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler
        )

        e0 = train_val_loader.dataset.features[:, 0].median()
        c0 = train_val_loader.dataset.targets.max()
        d0 = train_val_loader.dataset.targets.min()
        p0 = [1e-1, c0, d0, e0]
        model = Deep4PWeibull(len(predictors) - 1, config.layer_sizes, p0).to(device)

        trainer = Trainer(
            config=config,
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            compute_loss=nn.MSELoss(),
            device=device,
        )
        best_model, log = trainer.train(n_epochs=config.n_epochs, metrics=["root_mean_squared_error", "r2_score"])
        best_model = best_model.to("cpu")
        return {
            "model": best_model,
            "log": log,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "test_loader": test_loader,
        }
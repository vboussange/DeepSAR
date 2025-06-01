"""
Training ensemble model for the different datasets.
"""


import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass, field
from src.neural_4pweibull import Neural4PWeibull, MSELogLoss
from src.trainer import Trainer
from src.ensemble_model import EnsembleModel
from src.dataset import AugmentedDataset, create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ARCHITECTURE = {"small":[16, 16, 16],
                    "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                    "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]}
MODEL = "large"
HASH = "0b85791"
@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 10
    val_size: float = 0.1
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    n_epochs: int = 100
    weight_decay: float = 1e-3
    seed: int = 1
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    n_ensembles: int = 5  # Number of models in the ensemble
    run_name: str = f"checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL])
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_megaplot_data.parquet"


def train_and_evaluate_ensemble(config, df):
    
    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    predictors = ["log_observed_area", "log_megaplot_area"] + climate_features
            
    gdf_train_val = df[df["test"] == False]
    gdf_test = df[df["test"] == True]
    train_val_loader, feature_scaler, target_scaler = create_dataloader(gdf_train_val, predictors, config.batch_size, config.num_workers)
    test_loader, _, _ = create_dataloader(gdf_test, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)

    models = []
    logs = []
    for ensemble_idx in range(config.n_ensembles):
        logger.info(f"Training model {ensemble_idx + 1}/{config.n_ensembles}")

        # train val test split
        torch.manual_seed(config.seed + ensemble_idx)
        np.random.seed(config.seed + ensemble_idx)
        random.seed(config.seed + ensemble_idx)

        train_idx, val_idx = train_test_split(gdf_train_val.index,
                                            test_size=config.val_size,
                                            random_state=config.seed + ensemble_idx,)
        gdf_train, gdf_val = gdf_train_val.loc[train_idx], gdf_train_val.loc[val_idx]

        train_loader, _, _ = create_dataloader(gdf_train, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)
        val_loader, _, _ = create_dataloader(gdf_val, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)

        # Model initialization 
        e0 = train_val_loader.dataset.features[:,0].median()
        c0 = train_val_loader.dataset.targets.max()
        d0 = train_val_loader.dataset.targets.min()
        p0 = [1e-1, c0, d0, e0]
        model = Neural4PWeibull(len(predictors)-1, config.layer_sizes, p0)

        trainer = Trainer(config=config, 
                          model=model, 
                          feature_scaler=feature_scaler, 
                          target_scaler=target_scaler, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          test_loader=test_loader, 
                          compute_loss=MSELogLoss(),
                          device=config.device,
                        # compute_loss = nn.MSELoss()
                          )
        best_model, log = trainer.train(n_epochs=config.n_epochs, metrics=["mean_squared_error", "r2_score"])
        models.append(best_model)
        logs.append(log)

    # Create ensemble model
    ensemble_model = EnsembleModel(models)
    ensemble_trainer = Trainer(config=config,
                               model=ensemble_model,
                               feature_scaler=feature_scaler,
                               target_scaler=target_scaler,
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                test_loader=test_loader, 
                                compute_loss=MSELogLoss(),
                                device=config.device,
                               )
    # evaluating model ensemble predictions
    targets, preds = ensemble_trainer.get_model_predictions(test_loader)
    pred_trs = target_scaler.inverse_transform(preds)
    target_trs = target_scaler.inverse_transform(targets)
    ensemble_mse = mean_squared_error(target_trs, pred_trs)
    ensemble_r2 = r2_score(target_trs, pred_trs)
    logger.info(f"Ensemble MSE on test dataset: {ensemble_mse:.4f}")

    return {
        "ensemble_model_state_dict": ensemble_model.state_dict(),
        "logs": logs,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "mean_squared_error_test": ensemble_mse,
        "r2_test": ensemble_r2,
        "predictors": predictors,
    }
        
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    config = Config(device=device)
    config.run_folder = Path(Path(__file__).parent, 'results', f"{Path(__file__).stem}_seed_{config.seed}")
    config.run_folder.mkdir(exist_ok=True, parents=True)
    
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset = eva_dataset.dropna() #TODO: to improve
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    eva_dataset["log_sr"] = np.log(eva_dataset["sr"])

    results = train_and_evaluate_ensemble(config, eva_dataset)

    results["config"] = config
    logger.info(f"Saving results in {config.run_folder}")
    torch.save(results, config.run_folder / f"{config.run_name}.pth")
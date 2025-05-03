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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass, field
from src.mlp import MLP, CustomMSELoss
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
HASH = "fb8bc71"
@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 10
    test_size: float = 0.1
    val_size: float = 0.1
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    n_epochs: int = 100
    dSRdA_weight: float = 1e0
    weight_decay: float = 1e-3
    seed: int = 1
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T", "Q", "S", "R"])
    n_ensembles: int = 5  # Number of models in the ensemble
    run_name: str = f"checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL]) # [16, 16, 16] # [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7] # [2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]
    # test_partitions: list = field(default_factory=lambda: [684, 546, 100, 880, 1256, 296]) # ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_augmented_data.pkl"
    path_gift_data: str = Path(__file__).parent / f"../data/processed/GIFT_CHELSA_compilation/{HASH}/megaplot_data.gpkg"


def train_and_evaluate_ensemble(config, df):
    
    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    predictors = ["log_area", "log_megaplot_area"] + climate_features
            
    train_val_idx, test_idx = train_test_split(df.index,
                                            test_size= config.test_size,
                                            random_state=config.seed)
    gdf_train_val = df.loc[train_val_idx]
    _, feature_scaler, target_scaler = create_dataloader(gdf_train_val, predictors, config.batch_size, config.num_workers)

    models = []

    for ensemble_idx in range(config.n_ensembles):
        logger.info(f"Training model {ensemble_idx + 1}/{config.n_ensembles}")

        # Set random seeds for reproducibility
        torch.manual_seed(config.seed + ensemble_idx)
        np.random.seed(config.seed + ensemble_idx)
        random.seed(config.seed + ensemble_idx)

        train_idx, val_idx = train_test_split(train_val_idx,
                                            test_size=config.val_size,
                                            random_state=config.seed + ensemble_idx,)
        gdf_train, gdf_val, gdf_test = df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

        train_loader, _, _ = create_dataloader(gdf_train, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)
        val_loader, _, _ = create_dataloader(gdf_val, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)
        test_loader, _, _ = create_dataloader(gdf_test, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)

        model = MLP(len(predictors), config.layer_sizes)

        trainer = Trainer(config=config, 
                          model=model, 
                          feature_scaler=feature_scaler, 
                          target_scaler=target_scaler, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          test_loader=test_loader, 
                          compute_loss=CustomMSELoss(config.dSRdA_weight),
                          device=config.device,
                        # compute_loss = nn.MSELoss()
                          )
        best_model, _ = trainer.train(n_epochs=config.n_epochs, metrics=["mean_squared_error", "r2_score"])
        models.append(best_model)

    # Create ensemble model
    ensemble_model = EnsembleModel(models)
    ensemble_trainer = Trainer(config=config,
                               model=ensemble_model,
                               feature_scaler=feature_scaler,
                               target_scaler=target_scaler,
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                test_loader=test_loader, 
                                compute_loss=CustomMSELoss(config.dSRdA_weight),
                                device=config.device,
                               )
    targets, preds = ensemble_trainer.get_model_predictions(test_loader)
    ensemble_mse = mean_squared_error(targets, preds)
    ensemble_r2 = r2_score(targets, preds)
    logger.info(f"Ensemble MSE on test dataset: {ensemble_mse:.4f}")

    return {
        "ensemble_model_state_dict": ensemble_model.state_dict(),
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
    config.run_folder = Path(Path(__file__).parent, 'results', f"{Path(__file__).stem}_dSRdA_weight_{config.dSRdA_weight:.0e}_seed_{config.seed}")
    config.run_folder.mkdir(exist_ok=True, parents=True)
    
    augmented_dataset = AugmentedDataset(path_eva_data = config.path_eva_data,
                                         path_gift_data = config.path_gift_data,
                                         seed = config.seed)
    
    
    results_all = {}
    for hab in config.habitats:
        logger.info(f"Training ensemble model with habitat {hab}")
        df = augmented_dataset.compile_training_data(hab)
        results = train_and_evaluate_ensemble(config, df)
        results_all[hab] = results

    results_all["config"] = config
    logger.info(f"Saving results in {config.run_folder}")
    torch.save(results_all, config.run_folder / f"{config.run_name}.pth")
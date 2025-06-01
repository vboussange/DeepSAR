"""
Cross-validation script for training and evaluating models on the EVA-CHELSA dataset for different habitats and predictors.

# TODO: significant work is performed on CPU, consider identifying and moving to GPU
# TODO: WORK IN PROGRESS, script must be checked
"""
import copy
import random
import logging
import sys
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from joblib import Parallel, delayed
import geopandas as gpd

from src.mlp import MLP, CustomMSELoss, inverse_transform_scale_feature_tensor
from src.dataset import create_dataloader, AugmentedDataset
from src.plotting import read_result
from src.trainer import Trainer
# TODO: can we use logging.info instead of logger.info?
def setup_logger():
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s  %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

MODEL_ARCHITECTURE = {
                      "small":[16, 16, 16],
                      "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4],
                      "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                        }
MODEL = "large"
HASH = "fb8bc71" 
@dataclass
class Config:
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100 #todo: to change
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    dSRdA_weight: float = 1e0
    weight_decay: float = 1e-3
    val_size: float = 0.2
    seed: int = 2
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T", "Q", "S", "R"])
    run_name: str = f"checkpoint_{MODEL}_model_cross_validation_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL])
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_augmented_data.pkl"
    path_gift_data: str = Path(__file__).parent / f"../data/processed/GIFT_CHELSA_compilation/{HASH}/megaplot_data.gpkg"


class ParallelCrossValidator:
    def __init__(self, config: Config):
        self.config = config
        self.augmented_dataset = AugmentedDataset(path_eva_data = config.path_eva_data,
                                                path_gift_data = config.path_gift_data,
                                                seed = config.seed)
        self.devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]

    def run_CV_for_habitat(self, hab):
        climate_vars = self.config.climate_variables
        std_climate_vars = ["std_" + env for env in climate_vars]
        climate_features = climate_vars + std_climate_vars
        num_climate_features = len(climate_features)

        model_list = {
            "power_law": (MLP(2, []), ["log_area", "log_megaplot_area"], torch.nn.MSELoss(), False),
            "area": (MLP(2, config.layer_sizes), ["log_area", "log_megaplot_area"], CustomMSELoss(config.dSRdA_weight), False),
            "climate": (MLP(num_climate_features, config.layer_sizes), climate_features, torch.nn.MSELoss(), False),
            "area+climate": (MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), False),
            "area+climate, no physics": (MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, torch.nn.MSELoss(), False),
        }

        if hab != "all":
            model_list["area+climate, habitat agnostic"] = (
                MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), True
            )

        gdf = self.augmented_dataset.compile_training_data(hab)

        cv_results = {}
        for predictors_name, (model, predictors, compute_loss, agnostic) in model_list.items():
            gdf_train_val, gdf_test = (self.augmented_dataset.compile_training_data("all"), gdf) if agnostic else (gdf, gdf)

            logger.info(f"Training model with predictors: {predictors_name}")

            res = self.train_and_evaluate(
                model,
                predictors,
                compute_loss,
                gdf_train_val,
                gdf_test,
            )
            cv_results[predictors_name] = res
        return cv_results
        
    
    def train_and_evaluate(self, model, predictors, compute_loss, gdf_train_val, gdf_test):
        kfold = GroupKFold(n_splits=self.config.k_folds)

        results = Parallel(n_jobs=len(self.devices))(
            delayed(self.train_and_evaluate_fold)(
                train_idx,
                test_idx,
                gdf_train_val,
                gdf_test,
                model,
                predictors,
                compute_loss,
                self.devices[fold % len(self.devices)], 
            )
            for fold, (train_idx, test_idx) in enumerate(kfold.split(gdf_train_val, groups=gdf_train_val.partition))
        )
        
        # results = [
        #     self.train_and_evaluate_fold(
        #     train_idx,
        #     test_idx,
        #     gdf_train_val,
        #     gdf_test,
        #     model,
        #     predictors,
        #     compute_loss,
        #     optimizer_cls,
        #     scheduler_cls,
        #     self.devices[fold % len(self.devices)], 
        #     fold
        #     )
        #     for fold, (train_idx, test_idx) in enumerate(kfold.split(gdf_train_val, groups=gdf_train_val.partition))
        # ]

        aggregated_results = {key: [result[key] for result in results] for key in results[0]}
        aggregated_results["predictors"] = predictors

        return aggregated_results
    
    def train_and_evaluate_fold(self, train_idx, test_idx, gdf_train_val, gdf_test, model, predictors, compute_loss, device):
        torch.cuda.set_device(device)
        model = copy.deepcopy(model)

        gdf_train_val_fold = gdf_train_val.iloc[train_idx]
        ttrain_idx, val_idx = next(GroupShuffleSplit(test_size=self.config.val_size, random_state=self.config.seed).split(gdf_train_val_fold, groups=gdf_train_val_fold.partition))
        gdf_train_fold = gdf_train_val_fold.iloc[ttrain_idx]
        gdf_val_fold = gdf_train_val_fold.iloc[val_idx]
        
        val_partitions = gdf_val_fold.partition.unique()
        # NOTE: we gather partitions that must be used for test set, and define the test 
        # dataset with gdf_test. This is essential to cover the agnostic model case (no use otherwise)
        test_partitions = gdf_train_val.iloc[test_idx].partition.unique()
        test_idx_fold = gdf_test.partition.isin(test_partitions)
        gdf_test_fold = gdf_test[test_idx_fold]

        train_loader, feature_scaler, target_scaler = create_dataloader(gdf_train_fold, predictors, self.config.batch_size, self.config.num_workers)
        val_loader, _, _ = create_dataloader(gdf_val_fold, predictors, self.config.batch_size, self.config.num_workers, feature_scaler, target_scaler)
        test_loader, _, _ = create_dataloader(gdf_test_fold, predictors, self.config.batch_size, self.config.num_workers, feature_scaler, target_scaler)

        trainer = Trainer(config=config, 
                          model=model, 
                          feature_scaler=feature_scaler, 
                          target_scaler=target_scaler, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          test_loader=test_loader, 
                          compute_loss=compute_loss,
                          device=device)
        
        best_model, best_metrics = trainer.train(n_epochs=config.n_epochs, metrics=["mean_squared_error", "r2_score"])

        results = {
            # "train_loss": best_metrics['train_mean_squared_error'],
            # "val_MSE": best_metrics['val_mean_squared_error'],
            "test_MSE": best_metrics['test_mean_squared_error'],
            "test_R2": best_metrics['test_r2_score'],
            "test_partition": test_partitions.tolist(),
            "val_partition": val_partitions.tolist(),
            "model_state_dict": best_model.state_dict(),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler
        }

        # calculating the test loss for the best model
        best_model.eval()
        best_model.to(device)
        for loss_name, compute_loss in zip(["test_physics_informed_loss", "test_standard_loss"], [CustomMSELoss(self.config.dSRdA_weight), torch.nn.MSELoss()]):
            val_loss = 0.0
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                if isinstance(compute_loss, torch.nn.MSELoss):
                        outputs = best_model(X)
                        batch_loss = compute_loss(outputs, y)
                else:
                    X = X.requires_grad_(True)
                    outputs = best_model(X)
                    batch_loss = compute_loss(best_model, outputs, X, y)
                    
                val_loss +=  batch_loss.item() * X.size(0)
            avg_val_loss = val_loss / len(test_loader.dataset)
            results[loss_name] = avg_val_loss


        return results

        
        

if __name__ == "__main__":
    config = Config()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    config.run_folder = Path(Path(__file__).parent,'results', f"{Path(__file__).stem}_dSRdA_weight_{config.dSRdA_weight:.0e}_seed_{config.seed}")
    config.run_folder.mkdir(exist_ok=True)
    
    validator = ParallelCrossValidator(config)
    
    results_all = {}
    for hab in config.habitats:
        logger.info(f"Training models for habitat: {hab}")
        cv_results = validator.run_CV_for_habitat(hab)
        results_all[hab] = cv_results
        
        # checkpointing
        save_path = config.run_folder / f"{config.run_name}_{hab}.pth"
        torch.save(cv_results, save_path)
        
    results_all["config"] = config
    logger.info(f"Saving results in {config.run_folder}")
    torch.save(results_all, config.run_folder / f"{config.run_name}.pth")



    # # testing
    # compiled_data = trainer.compile_training_data("all", config)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # data = compiled_data["log_megaplot_area"]
    # # data = np.log(trainer.gift_data["area"])
    # plt.hist(data, bins=50, log=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.xlabel("Area")
    # plt.ylabel("Frequency (log scale)")
    # plt.title("Distribution of Area")
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    
    
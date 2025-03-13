"""
Cross-validation script for training and evaluating models on the EVA-CHELSA dataset for different habitats and predictors.

# TODO: significant work is performed on CPU, consider identifying and moving to GPU
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

from src.mlp import MLP, CustomMSELoss, inverse_transform_scale_feature_tensor
from src.dataset import create_dataloader
from src.plotting import read_result

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

MODEL_ARCHITECTURE = {
                      "small":[16, 16, 16],
                      "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4],
                      "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                        }
MODEL = "large"
HASH = "7ad505e"
@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 0
    k_folds: int = 10
    n_epochs: int = 100
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    dSRdA_weight: float = 1e0
    weight_decay: float = 1e-3
    val_size: float = 0.2
    seed: int = 2
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"])
    run_name: str = f"checkpoint_{MODEL}_model_cross_validation_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL])
    path_augmented_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_raw_compilation/EVA_CHELSA_raw_random_state_2_{HASH}.pkl"


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.devices = ["cuda:0", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
        self.results = {}
        self.habitat_agnostic_model = None
        self.habitat_agnostic_scalers = {}
        self.data = read_result(config.path_augmented_data)
        
    def train(self):
        self.prepare_run()
        for hab in self.config.habitats:
            print(f"Training models for habitat: {hab}")
            self.results[hab] = {}
            self.train_models_for_habitat(hab)
            self.save_results() # checkpointing after each habitat

    def prepare_run(self):
        self.config.run_folder = Path(Path(__file__).parent,'results', f"{Path(__file__).stem}_dSRdA_weight_{self.config.dSRdA_weight:.0e}_seed_{self.config.seed}")
        self.config.run_folder.mkdir(exist_ok=True)

    def train_models_for_habitat(self, hab):
        climate_vars = self.config.climate_variables
        std_climate_vars = ["std_" + env for env in climate_vars]
        climate_features = climate_vars + std_climate_vars
        num_climate_features = len(climate_features)

        predictors_list = {
            "power_law": (MLP(2, []), ["log_area", "log_megaplot_area"], torch.nn.MSELoss(), False),
            "area": (MLP(2, config.layer_sizes), ["log_area", "log_megaplot_area"], CustomMSELoss(config.dSRdA_weight), False),
            "climate": (MLP(num_climate_features, config.layer_sizes), climate_features, torch.nn.MSELoss(), False),
            "area+climate": (MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), False),
            "area+climate, no physics": (MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, torch.nn.MSELoss(), False),
        }

        if hab != "all":
            predictors_list["area+climate, habitat agnostic"] = (
                MLP(num_climate_features + 2, config.layer_sizes), ["log_area", "log_megaplot_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), True
            )

        gdf = compile_training_data(self.data, hab, config)

        for predictors_name, (model, predictors, criterion, agnostic) in predictors_list.items():
            gdf_train_val, gdf_test = (compile_training_data(self.data, "all", config), gdf) if agnostic else (gdf, gdf)

            print(f"Training model with predictors: {predictors_name}")

            trained_model, metrics = self.train_and_evaluate(
                model,
                predictors,
                criterion,
                optim.AdamW,
                ReduceLROnPlateau,
                gdf_train_val,
                gdf_test,
            )

            self.results[hab][predictors_name] = metrics


    
    def train_and_evaluate_fold(self, train_idx, test_idx, gdf_train_val, gdf_test, model, predictors, criterion, optimizer_cls, scheduler_cls, device, fold):
        torch.cuda.set_device(device)
        model = copy.deepcopy(model).to(device)
        criterion = criterion.to(device)
        optimizer = optimizer_cls(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = scheduler_cls(optimizer, factor=self.config.lr_scheduler_factor, patience=self.config.lr_scheduler_patience)

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

        best_model, epoch_metrics = self.train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, target_scaler, device, fold)
        best_epoch = np.argmin(epoch_metrics['val_MSE'])

        return {
            "train_MSE": epoch_metrics['train_MSE'][best_epoch],
            "val_MSE": epoch_metrics['val_MSE'][best_epoch],
            "test_MSE": epoch_metrics['test_MSE'][best_epoch],
            "test_partition": test_partitions.tolist(),
            "val_partition": val_partitions.tolist(),
            "model_state_dict": best_model.state_dict(),
            "epoch_metrics": epoch_metrics,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler
        }

    def train_and_evaluate(self, model, predictors, criterion, optimizer_cls, scheduler_cls, gdf_train_val, gdf_test):
        kfold = GroupKFold(n_splits=self.config.k_folds)

        results = Parallel(n_jobs=len(self.devices))(
            delayed(self.train_and_evaluate_fold)(
                train_idx,
                test_idx,
                gdf_train_val,
                gdf_test,
                model,
                predictors,
                criterion,
                optimizer_cls,
                scheduler_cls,
                self.devices[fold % len(self.devices)], 
                fold
            )
            for fold, (train_idx, test_idx) in enumerate(kfold.split(gdf_train_val, groups=gdf_train_val.partition))
        )

        aggregated_results = {key: [result[key] for result in results] for key in results[0]}
        aggregated_results["predictors"] = predictors

        return model, aggregated_results
    
    def train_model(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, target_scaler, device, fold):
        logger = setup_logger()

        
        epoch_metrics = {
            "train_loss": [],
            "train_MSE": [],
            "val_MSE": [],
            "test_MSE": []
        }
        best_val_MSE = float('inf')
        best_model = None

        for epoch in range(self.config.n_epochs):
            model.train()
            running_train_loss = 0.0
            running_train_MSE = 0.0

            for log_sr, inputs in train_loader:
                inputs, log_sr = inputs.to(device), log_sr.to(device)
                if isinstance(criterion, torch.nn.MSELoss):
                    outputs = model(inputs)
                    loss = criterion(outputs, log_sr)
                else:
                    inputs.requires_grad_(True)
                    outputs = model(inputs)
                    loss = criterion(model, outputs, inputs, log_sr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

                with torch.no_grad():
                    model.eval()
                    y_pred = inverse_transform_scale_feature_tensor(outputs, target_scaler)
                    y_true = inverse_transform_scale_feature_tensor(log_sr, target_scaler)
                    train_MSE = torch.mean((y_pred - y_true) ** 2) * inputs.size(0)
                    running_train_MSE += train_MSE.item()

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            avg_train_MSE = running_train_MSE / len(train_loader.dataset)
            epoch_metrics["train_loss"].append(avg_train_loss)
            epoch_metrics["train_MSE"].append(avg_train_MSE)

            # Validation and Test
            model.eval()
            avg_val_MSE = self.evaluate_model(model, val_loader, target_scaler, device)
            avg_test_MSE = self.evaluate_model(model, test_loader, target_scaler, device)
            epoch_metrics["val_MSE"].append(avg_val_MSE)
            epoch_metrics["test_MSE"].append(avg_test_MSE)

            logger.info(f"Device: {device} | Fold: {fold} || Epoch {epoch + 1}/{self.config.n_epochs} | Training Loss: {avg_train_loss:.4f} | Training MSE: {avg_train_MSE:.4f} | Validation MSE: {avg_val_MSE:.4f} | Test MSE: {avg_test_MSE:.4f}")

            if avg_val_MSE < best_val_MSE:
                best_val_MSE = avg_val_MSE
                best_model = copy.deepcopy(model).to("cpu")

            scheduler.step(avg_val_MSE)

        return best_model, epoch_metrics

    def evaluate_model(self, model, loader, target_scaler, device):
        running_MSE = 0.0
        with torch.no_grad():
            for log_sr, inputs in loader:
                inputs, log_sr = inputs.to(device), log_sr.to(device)
                outputs = model(inputs)
                y_pred = inverse_transform_scale_feature_tensor(outputs, target_scaler)
                y_true = inverse_transform_scale_feature_tensor(log_sr, target_scaler)
                MSE = torch.mean((y_pred - y_true) ** 2) * inputs.size(0)
                running_MSE += MSE.item()
        avg_MSE = running_MSE / len(loader.dataset)
        return avg_MSE

    def save_results(self):
        self.results["config"] = self.config
        save_path = self.config.run_folder / f"{self.config.run_name}.pth"
        print(f"Saving results to {save_path}")
        torch.save(self.results, save_path)
        
        
def compile_training_data(data, hab, config):
    megaplot_data = data["megaplot_data"][data["megaplot_data"]["habitat_id"] == hab]
    if hab == "all":
        plot_data = data["plot_data_all"]
    else:
        plot_data = data["plot_data_all"][data["plot_data_all"]["habitat_id"] == hab]
        
    augmented_data = pd.concat([plot_data, megaplot_data], ignore_index=True)
    
    # stack with raw plot data
    augmented_data.loc[:, "log_area"] = np.log(augmented_data["area"].astype(np.float32))  # area
    augmented_data.loc[:, "log_megaplot_area"] = np.log(augmented_data["megaplot_area"].astype(np.float32))  # area
    augmented_data.loc[:, "log_sr"] = np.log(augmented_data["sr"].astype(np.float32))  # area
    augmented_data = augmented_data.dropna()
    augmented_data = augmented_data.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    return augmented_data

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:2"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    config = Config(device=device)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainer = Trainer(config)
    trainer.train()

"""
cross_validate.py
This script performs cross-validation for training and evaluating machine learning models on habitat-specific data.
It supports multiple model architectures and configurations, including habitat-agnostic variants.
Classes:
    Config: Configuration dataclass for setting up training parameters and model configurations.
    Trainer: Class responsible for training and evaluating models across different habitats.
Functions:
    Trainer.train: Main method to initiate the training process for all habitats.
    Trainer.prepare_run: Prepares the run directory for saving results.
    Trainer.train_models_for_habitat: Trains models for a specific habitat.
    Trainer.train_and_evaluate: Trains and evaluates models using k-fold cross-validation.
    Trainer.train_model: Trains a single model and tracks performance metrics.
    Trainer.evaluate_model: Evaluates a model on a given dataset.
    Trainer.save_results: Saves the training results to a file.
Usage:
    Run the script directly to start the training process with the specified configuration.
    
# TODO: significant work is performed on CPU, consider identifying and moving to GPU
"""
import copy
import random
import logging
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, GroupKFold
from pathlib import Path
from dataclasses import dataclass, field
from src.mlp import MLP, CustomMSELoss, inverse_transform_scale_feature_tensor
from src.sar import SAR
from src.dataset import create_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data

MODEL_ARCHITECTURE = {
                      "small":[16, 16, 16],
                      "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4],
                      "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                        }
MODEL = "large"
HASH = "71f9fc7"
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
    data_seed: int = 2
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"])
    run_name: str = f"checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL])

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.results = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self.habitat_agnostic_model = None
        self.habitat_agnostic_scalers = {}

    def train(self):
        self.prepare_run()
        for hab in self.config.habitats:
            logger.info(f"Training models for habitat: {hab}")
            self.results[hab] = {}
            self.train_models_for_habitat(hab)
        self.save_results()

    def prepare_run(self):
        self.config.run_folder = Path(Path(__file__).parent,'results', f"{Path(__file__).stem}_dSRdA_weight_{self.config.dSRdA_weight:.0e}_seed_{self.config.seed}")
        self.config.run_folder.mkdir(exist_ok=True)

    def train_models_for_habitat(self, hab):
        # Define common variables for climate-related predictors
        climate_vars = self.config.climate_variables
        std_climate_vars = ["std_" + env for env in climate_vars]
        climate_features = climate_vars + std_climate_vars
        num_climate_features = len(climate_features)

        # Define predictor configurations with corresponding models, input features, loss functions, and physics flag
        predictors_list = {
            "power_law": (SAR(), ["log_area"], torch.nn.MSELoss(), False),
            "area": (MLP(1, config.layer_sizes), ["log_area"], CustomMSELoss(config.dSRdA_weight), False),
            "climate": (MLP(num_climate_features, config.layer_sizes), climate_features, torch.nn.MSELoss(), False),
            "area+climate": (MLP(num_climate_features + 1, config.layer_sizes), ["log_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), False),
            "area+climate, no physics": (MLP(num_climate_features + 1, config.layer_sizes), ["log_area"] + climate_features, torch.nn.MSELoss(), False),
        }
        
        # Add habitat agnostic variant if applicable
        if hab != "all":
            predictors_list["area+climate, habitat agnostic"] = (
                MLP(num_climate_features + 1, config.layer_sizes), ["log_area"] + climate_features, CustomMSELoss(config.dSRdA_weight), True
            )

        gdf = load_preprocessed_data(hab, self.config.hash_data, self.config.data_seed)
        for predictors_name, (model, predictors, criterion, agnostic) in predictors_list.items():
            if agnostic:
                gdf_all = load_preprocessed_data("all", config.hash_data, config.data_seed)
                gdf_train_val, gdf_test = gdf_all, gdf
            else:
                gdf_train_val, gdf_test = gdf, gdf
                
            logger.info(f"Training model with predictors: {predictors_name}")
            model = copy.deepcopy(model).to(self.device)
            criterion = CustomMSELoss(self.config.dSRdA_weight).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, factor=self.config.lr_scheduler_factor, patience=self.config.lr_scheduler_patience)

            trained_model, metrics, scalers = self.train_and_evaluate(model, predictors, criterion, optimizer, scheduler, gdf_train_val, gdf_test, agnostic)
            self.results[hab][predictors_name] = metrics
            self.feature_scalers[hab] = scalers[0]
            self.target_scalers[hab] = scalers[1]

    def train_and_evaluate(self, model, predictors, criterion, optimizer, scheduler, gdf_train_val, gdf_test, agnostic):
        kfold = GroupKFold(n_splits=self.config.k_folds)
        results = {
            "train_MSE": [],
            "val_MSE": [],
            "test_MSE": [],
            "test_idx": [],
            "model_state_dict": [],
            "epoch_metrics": [],
            "feature_scaler": [],
            "target_scaler": [], 
            "predictors": predictors
        }

        for fold, (train_idx, test_idx) in enumerate(kfold.split(gdf_train_val, groups=gdf_train_val.partition)):
            logger.info(f"Fold {fold + 1}/{self.config.k_folds}")
            try:
                gdf_train_val_fold = gdf_train_val.iloc[train_idx]
                gdf_train_fold, gdf_val_fold = train_test_split(gdf_train_val_fold, test_size=config.val_size, random_state=config.seed)
                # we subset in the test data the partitions corresponding to the ones from the fold, indexed in test_idx
                test_partitions = gdf_train_val.iloc[test_idx].partition.unique()
                test_idx_fold = gdf_test.partition.isin(test_partitions)
                gdf_test_fold = gdf_test[test_idx_fold]

                train_loader, feature_scaler, target_scaler = create_dataloader(gdf_train_fold, predictors, self.config.batch_size, self.config.num_workers)
                val_loader, _, _ = create_dataloader(gdf_val_fold, predictors, self.config.batch_size, self.config.num_workers, feature_scaler, target_scaler)
                test_loader, _, _ = create_dataloader(gdf_test_fold, predictors, self.config.batch_size, self.config.num_workers, feature_scaler, target_scaler)

                best_model, epoch_metrics = self.train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, target_scaler)
                best_epoch = np.argmin(epoch_metrics['val_MSE'])
                best_train_MSE = epoch_metrics['train_MSE'][best_epoch]
                best_val_MSE = epoch_metrics['val_MSE'][best_epoch]
                best_test_MSE = epoch_metrics['test_MSE'][best_epoch]
                best_model_state = best_model.state_dict()

                results["train_MSE"].append(best_train_MSE)
                results["val_MSE"].append(best_val_MSE)
                results["test_MSE"].append(best_test_MSE)
                results["test_idx"].append(gdf_test_fold.index.tolist())
                results["model_state_dict"].append(best_model_state)
                results["epoch_metrics"].append(epoch_metrics)
                results["feature_scaler"].append(feature_scaler)
                results["target_scaler"].append(target_scaler)
            except Exception as e:
                logger.error(f"Problem with fold {fold + 1}: {e}")
                traceback.print_exc()
                results["train_MSE"].append(float("nan"))
                results["val_MSE"].append(float("nan"))
                results["test_MSE"].append(float("nan"))
                results["feature_scaler"].append(None)
                results["target_scaler"].append(None)

        return model, results, (feature_scaler, target_scaler)

    def train_model(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, target_scaler):
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
                inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
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
            avg_val_MSE = self.evaluate_model(model, val_loader, target_scaler)
            avg_test_MSE = self.evaluate_model(model, test_loader, target_scaler)
            epoch_metrics["val_MSE"].append(avg_val_MSE)
            epoch_metrics["test_MSE"].append(avg_test_MSE)

            logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs} | Training Loss: {avg_train_loss:.4f} | Training MSE: {avg_train_MSE:.4f} | Validation MSE: {avg_val_MSE:.4f} | Test MSE: {avg_test_MSE:.4f}")

            if avg_val_MSE < best_val_MSE:
                best_val_MSE = avg_val_MSE
                best_model = copy.deepcopy(model).to("cpu")

            scheduler.step(avg_val_MSE)

        return best_model, epoch_metrics

    def evaluate_model(self, model, loader, target_scaler):
        running_MSE = 0.0
        with torch.no_grad():
            for log_sr, inputs in loader:
                inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
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
        logger.info(f"Saving results to {save_path}")
        torch.save(self.results, save_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:1"
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

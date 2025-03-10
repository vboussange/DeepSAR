import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from src.mlp import MLP, CustomMSELoss, inverse_transform_scale_feature_tensor
from src.ensemble_model import EnsembleModel
from src.dataset import create_dataloader
from src.plotting import read_result

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ARCHITECTURE = {"small":[16, 16, 16],
                    "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                    "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]}
MODEL = "large"
HASH = "71f9fc7"
@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 0
    test_size: float = 0.1
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    n_epochs: int = 100
    dSRdA_weight: float = 1e0
    weight_decay: float = 1e-3
    seed: int = 1
    data_seed: int = 1
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]) # ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    n_ensembles: int = 5  # Number of models in the ensemble
    run_name: str = f"checkpoint_{MODEL}_model_full_physics_informed_constraint_plot_only_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL]) # [16, 16, 16] # [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7] # [2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.models = []
        self.feature_scaler = None
        self.target_scaler = None

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        best_val_MSE = float('inf')
        best_model = None
        epoch_metrics = {'train_loss': [], 'train_MSE': [], 'val_MSE': []}

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
                    y_pred = inverse_transform_scale_feature_tensor(outputs, self.target_scaler)
                    y_true = inverse_transform_scale_feature_tensor(log_sr, self.target_scaler)
                    train_MSE = torch.mean((y_pred - y_true) ** 2) * inputs.size(0)
                    running_train_MSE += train_MSE.item()

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            avg_train_MSE = running_train_MSE / len(train_loader.dataset)
            epoch_metrics['train_loss'].append(avg_train_loss)
            epoch_metrics['train_MSE'].append(avg_train_MSE)

            # Validation
            model.eval()
            running_val_MSE = 0.0
            with torch.no_grad():
                for log_sr, inputs in val_loader:
                    inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
                    outputs = model(inputs)

                    y_pred = inverse_transform_scale_feature_tensor(outputs, self.target_scaler)
                    y_true = inverse_transform_scale_feature_tensor(log_sr, self.target_scaler)
                    val_MSE = torch.mean((y_pred - y_true) ** 2) * inputs.size(0)
                    running_val_MSE += val_MSE.item()

            avg_val_MSE = running_val_MSE / len(val_loader.dataset)
            epoch_metrics['val_MSE'].append(avg_val_MSE)

            logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs} | Training Loss: {avg_train_loss:.4f} | Training MSE: {avg_train_MSE:.4f} | Validation MSE: {avg_val_MSE:.4f}")

            if avg_val_MSE < best_val_MSE:
                best_val_MSE = avg_val_MSE
                best_model = copy.deepcopy(model).to('cpu')

            scheduler.step(avg_val_MSE)

        return best_model, best_val_MSE, epoch_metrics

    def train_and_evaluate_ensemble(self, gdf_full, predictors, criterion):
        self.models = []
        ensemble_metrics = []

        for ensemble_idx in range(self.config.n_ensembles):
            logger.info(f"Training model {ensemble_idx + 1}/{self.config.n_ensembles}")

            # Set random seeds for reproducibility
            torch.manual_seed(self.config.seed + ensemble_idx)
            np.random.seed(self.config.seed + ensemble_idx)
            random.seed(self.config.seed + ensemble_idx)

            train_idx, test_idx = train_test_split(
                gdf_full.index,
                test_size=self.config.test_size,
                random_state=self.config.seed + ensemble_idx,
            )

            gdf_train, gdf_test = gdf_full.loc[train_idx], gdf_full.loc[test_idx]

            train_loader, self.feature_scaler, self.target_scaler = create_dataloader(gdf_train, predictors, self.config.batch_size, self.config.num_workers)
            val_loader, _, _ = create_dataloader(gdf_test, predictors, self.config.batch_size, self.config.num_workers, self.feature_scaler, self.target_scaler)


            model = MLP(len(predictors), config.layer_sizes).to(self.device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
            )

            best_model, best_val_MSE, epoch_metrics = self.train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
            self.models.append(best_model)
            ensemble_metrics.append(epoch_metrics)

        # Create ensemble model
        ensemble_model = EnsembleModel(self.models).to(self.device)

        # Evaluate ensemble model on full data
        loader, _, _ = create_dataloader(gdf_full, predictors, self.config.batch_size, self.config.num_workers, self.feature_scaler, self.target_scaler)

        ensemble_model.eval()
        running_MSE = 0.0
        with torch.no_grad():
            for log_sr, inputs in loader:
                inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
                outputs = ensemble_model(inputs)

                y_pred = inverse_transform_scale_feature_tensor(outputs, self.target_scaler)
                y_true = inverse_transform_scale_feature_tensor(log_sr, self.target_scaler)
                val_MSE = torch.mean((y_pred - y_true) ** 2) * inputs.size(0)
                running_MSE += val_MSE.item()

        avg_MSE = running_MSE / len(loader.dataset)
        logger.info(f"Ensemble MSE: {avg_MSE:.4f}")

        return {
            "ensemble_model_state_dict": ensemble_model.state_dict(),
            "best_validation_loss": avg_MSE,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "predictors": predictors,
            "ensemble_metrics": ensemble_metrics,
            "train_idx": train_idx
        }
        
def compile_training_data(data, hab):
    if hab == "all":
        augmented_data = data["plot_data_all"]
    else:
        augmented_data = data["plot_data_all"][data["plot_data_all"]["habitat_id"] == hab]
        
    augmented_data.loc[:, "log_area"] = np.log(augmented_data["area"].astype(np.float32))  # area
    augmented_data.loc[:, "log_megaplot_area"] = np.log(augmented_data["megaplot_area"].astype(np.float32))  # area
    augmented_data.loc[:, "log_sr"] = np.log(augmented_data["sr"].astype(np.float32))  # area
    augmented_data = augmented_data.dropna()
    climate_predictors = config.climate_variables + ["std_" + env for env in config.climate_variables]
    predictors = ["log_area", "log_megaplot_area"] + climate_predictors
    return augmented_data, predictors


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create Config instance
    config = Config(device=device)
    config.run_folder = Path(Path(__file__).parent, 'results', f"{Path(__file__).stem}_dSRdA_weight_{config.dSRdA_weight:.0e}_seed_{config.seed}")
    config.run_folder.mkdir(exist_ok=True, parents=True)

    path_augmented_data = Path("/home/boussang/DeepSAR/data/processed/EVA_CHELSA_raw_compilation/EVA_CHELSA_raw_random_state_2_cf6ea5c.pkl")
    data = read_result(path_augmented_data)

    results_all = {}
    for hab in config.habitats:
        logger.info(f"Training ensemble model with habitat {hab}")
        augmented_data, predictors = compile_training_data(data, hab)

        trainer = Trainer(config)
        results = trainer.train_and_evaluate_ensemble(
            augmented_data,
            predictors,
            CustomMSELoss(config.dSRdA_weight).to(device),
        )
        results_all[hab] = results

    results_all["config"] = config
    logger.info(f"Saving results in {config.run_folder}")
    torch.save(results_all, config.run_folder / f"{config.run_name}.pth")
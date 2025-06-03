"""
Evaluating model performance on the EVA-CHELSA and GIFT-CHELSA datasets for 
- different predictors, 
- different dataset sizes.

Should rely on train.py to train models.
"""
import copy
import random
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, d2_absolute_error_score

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from joblib import Parallel, delayed
import geopandas as gpd

from src.dataset import create_dataloader
from src.trainer import Trainer
from src.neural_4pweibull import Neural4PWeibull

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
HASH = "0b85791" 
@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 10
    n_epochs: int = 1 #todo: to change
    val_size: float = 0.2
    lr: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    weight_decay: float = 1e-4
    seed: int = 1
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    run_name: str = f"checkpoint_{MODEL}_ablation_{HASH}"
    run_folder: str = ""
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL])
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_megaplot_data.parquet"
    path_gift_data: str = Path(__file__).parent / f"../data/processed/GIFT_CHELSA_compilation/6c2d61d/megaplot_data.parquet"


class Benchmark:
    def __init__(self, config: Config):
        self.config = config
        eva_data = pd.read_parquet(config.path_eva_data)
        eva_data["log_megaplot_area"] = np.log(eva_data["megaplot_area"])
        eva_data["log_observed_area"] = np.log(eva_data["megaplot_area"])
        eva_data = eva_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.eva_data = eva_data
        gift_data = gpd.read_parquet(config.path_gift_data)
        gift_data["log_megaplot_area"] = np.log(gift_data["megaplot_area"])
        gift_data["log_observed_area"] = np.log(gift_data["megaplot_area"])
        gift_data = gift_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.gift_data = gift_data
        self.devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
        self.nruns = 5  # Number of runs for each fold

    def run(self, predictors, layer_sizes, compute_loss, train_size):

        results = Parallel(n_jobs=len(self.devices))(
            delayed(self.train_and_evaluate_fold)(
                predictors = predictors,
                compute_loss = compute_loss,
                layer_sizes = layer_sizes,
                train_size = train_size,
                seed = i,
                device = self.devices[i % len(self.devices)], 
            ) for i in range(self.nruns))


        # compute number of parameters
        model = Neural4PWeibull(len(predictors) - 1, layer_sizes, np.ones(4))
        num_params = sum(p.numel() for p in model.parameters())
        del model  # Free memory
        
        aggregated_results = {
            "logs": [result["log"] for result in results],
            "r2_test_eva": [result["r2_test_eva"] for result in results],
            "d2_test_eva": [result["d2_test_eva"] for result in results],
            "rmse_test_eva": [result["rmse_test_eva"] for result in results],
            "r2_test_gift": [result["r2_test_gift"] for result in results],
            "d2_test_gift": [result["d2_test_gift"] for result in results],
            "rmse_test_gift": [result["rmse_test_gift"] for result in results],
            "num_params": num_params,
        }

        return aggregated_results
    
    def train_and_evaluate_fold(self, 
                                predictors, 
                                layer_sizes,
                                compute_loss,
                                train_size,
                                seed,
                                device):
        
        eva_gdf_test, eva_gdf_train_val = self.eva_data[self.eva_data["test"] == True].copy(), self.eva_data[self.eva_data["test"] == False].copy()
        eva_gdf_train_val = eva_gdf_train_val.sample(frac=train_size, random_state=seed)
        train_idx, val_idx = train_test_split(eva_gdf_train_val.index,
                                    test_size=self.config.val_size,
                                    random_state=seed)
        gdf_train, gdf_val = eva_gdf_train_val.loc[train_idx], eva_gdf_train_val.loc[val_idx]
        train_loader, feature_scaler, target_scaler = create_dataloader(gdf_train, predictors, self.config.batch_size, self.config.num_workers)
        val_loader, _, _ = create_dataloader(gdf_val, predictors, self.config.batch_size, self.config.num_workers, feature_scaler, target_scaler)

        # Model initialization 
        e0 = train_loader.dataset.features[:,0].median()
        c0 = train_loader.dataset.targets.max()
        d0 = train_loader.dataset.targets.min()
        p0 = [1e-1, c0, d0, e0]
        model = Neural4PWeibull(len(predictors)-1, layer_sizes, p0)

        trainer = Trainer(config=self.config, 
                          model=model, 
                          feature_scaler=feature_scaler, 
                          target_scaler=target_scaler, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          compute_loss=compute_loss,
                          device=device)
        
        best_model, log = trainer.train(n_epochs=self.config.n_epochs)
        best_model.eval()

        
        # evaluating model predictions against EVA test set
        X = eva_gdf_test[predictors].copy()
        X = torch.tensor(feature_scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            y_pred = best_model(X).numpy()
            y_pred = target_scaler.inverse_transform(y_pred).squeeze()
        y_true = eva_gdf_test["sr"].values
        
        r2_test_eva = r2_score(y_true, y_pred)
        d2_test_eva = d2_absolute_error_score(y_true, y_pred)
        rmse_test_eva = np.sqrt(mean_squared_error(y_true, y_pred))

        # evaluating model predictions against GIFT test set
        X = self.gift_data[predictors].copy()
        X = torch.tensor(feature_scaler.transform(X), dtype=torch.float32)
        X = X[:,1:] # removing the log_observed_area feature
        with torch.no_grad():
            y_pred = best_model.predict_sr(X).numpy()
            y_pred = target_scaler.inverse_transform(y_pred).squeeze()
        y_true = self.gift_data["sr"].values
        
        r2_test_gift = r2_score(y_true, y_pred)
        d2_test_gift = d2_absolute_error_score(y_true, y_pred)
        rmse_test_gift = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            "log": log,
            "r2_test_eva": r2_test_eva,
            "d2_test_eva": d2_test_eva,
            "rmse_test_eva": rmse_test_eva,
            "r2_test_gift": r2_test_gift,
            "d2_test_gift": d2_test_gift,
            "rmse_test_gift": rmse_test_gift
        }

        
def generate_symmetric_architecture(n, base_size=32, growth_factor=2):
    """
    Generates symmetric hidden layer sizes where capacity increases with n.
    
    Args:
        n: Complexity parameter (higher = more capacity)
        base_size: Starting number of neurons
        growth_factor: Multiplier for neuron count scaling
        
    Returns:
        List of layer sizes (symmetric and increasing/decreasing)
    """
    # Build the first half of layers by geometric growth
    half = (n + 1) // 2
    layers = [base_size * (growth_factor ** i) for i in range(half)]
    # Mirror for symmetry; exclude the middle layer when n is odd
    mirror_section = layers[:-1] if n % 2 else layers
    return layers + mirror_section[::-1]

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    config = Config(device=device)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    config.run_folder = Path(Path(__file__).parent, 'results', f"{Path(__file__).stem}_seed_{config.seed}")
    config.run_folder.mkdir(exist_ok=True)
    
    benchmark = Benchmark(config)
    
    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    
    
    base_architecture = generate_symmetric_architecture(8, base_size=32, growth_factor=4)
    model_list = {
            "area": (["log_observed_area", "log_megaplot_area"], 
                     nn.MSELoss(), 
                     base_architecture, 
                     1),
            "climate": (["log_observed_area"] + climate_features, 
                        nn.MSELoss(), 
                        base_architecture, 
                        1),
        }
    for frac in np.logspace(np.log(0.1), np.log(1.0), 5):        
        model_list["area+climate"] = (["log_observed_area", "log_megaplot_area"] + climate_features, 
                                                  nn.MSELoss(), 
                                                  base_architecture,
                                                  frac)
        
    for n in [0, 2, 4, 6]:
        model_list["area+climate"] = (["log_observed_area", "log_megaplot_area"] + climate_features, 
                                        nn.MSELoss(), 
                                        generate_symmetric_architecture(n, base_size=32, growth_factor=4),
                                        1)
    
    
    
results_data = []

for model_name, (predictors, compute_loss, layer_sizes, train_size) in model_list.items():
    logger.info(f"Running benchmark for {model_name}")
    
    results = benchmark.run(
        predictors=predictors,
        layer_sizes=layer_sizes,
        compute_loss=compute_loss,
        train_size=train_size
    )
    
    # Add results to the list with model information
    for i in range(benchmark.nruns):
        results_data.append({
            'model_name': model_name,
            'run_id': i,
            'layer_sizes': str(layer_sizes),
            'train_size': train_size,
            'num_params': results['num_params'],
            'r2_test_eva': results['r2_test_eva'][i],
            'd2_test_eva': results['d2_test_eva'][i],
            'rmse_test_eva': results['rmse_test_eva'][i],
            'r2_test_gift': results['r2_test_gift'][i],
            'd2_test_gift': results['d2_test_gift'][i],
            'rmse_test_gift': results['rmse_test_gift'][i]
        })
    
    logger.info(f"Completed {model_name}")

# Create DataFrame and save to CSV
df_results = pd.DataFrame(results_data)
csv_path = config.run_folder / "benchmark_results.csv"
df_results.to_csv(csv_path, index=False)
logger.info(f"Results saved to {csv_path}")
"""
This script benchmarks the DeepSAR model on SBCV datasets, evaluating both
Interpolation (on SBCV test set) and Extrapolation (on GIFT dataset).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
from deepsar.utils import symmetric_arch
from deepsar.benchmarker import BenchmarkConfig, Benchmarker
from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.mlp import MLP
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "test"
GIFT_SAMPLES_PATH = Path(__file__).parent / "../data/processed/GIFT_test_samples/384b9c9/compiled_data.parquet"
SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/5c8274a"

def setup_logger():
    log = logging.getLogger("benchmark")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        h.setFormatter(fmt)
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log

logger = setup_logger()

@dataclass
class Deep4PWeibullInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=32, factor=4))
    def __call__(self, **kwargs):
        return Deep4PWeibull(feature_names=self.feature_names, 
                             layer_sizes=self.architecture, 
                             **kwargs)

class WrappedMLP(MLP):
    def __init__(self, input_dim, layer_sizes, feature_names=[], feature_scaler=None, target_scaler=None):
        super().__init__(input_dim, layer_sizes)
        self.feature_names = feature_names
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

@dataclass
class MLPInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: [128, 64, 32])
    def __call__(self, **kwargs):
        # input_dim = len(feature_names) + 1 (for log_observed_area)
        return WrappedMLP(len(self.feature_names) + 1, self.architecture, **kwargs)

if __name__ == "__main__":
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
        
    root_folder = Path(__file__).parent / Path('results', 'benchmark')
    root_folder.mkdir(parents=True, exist_ok=True)
    
    config = BenchmarkConfig(devices=devices,
                             path_gift_data= GIFT_SAMPLES_PATH,
                             path_sbcv_data= SBCV_SAMPLES_PATH,)
    
    # Inspect one file to get feature names
    sbcv_path = config.path_sbcv_data
    try:
        sample_file = next(sbcv_path.glob("*_train.parquet"))
        df = gpd.read_parquet(sample_file)
        
        # Identify features
        climate_feats = config.climate_variables + [f"std_{v}" for v in config.climate_variables]
        dem_feats = ["elevation", "std_elevation"]
        lc_feats = [c for c in df.columns if c.startswith("lc_frac_")]
        
        # Filter to what is actually present
        climate_feats = [c for c in climate_feats if c in df.columns]
        dem_feats = [c for c in dem_feats if c in df.columns]
        
        all_env_feats = climate_feats + dem_feats + lc_feats
        logger.info(f"Identified features: {len(all_env_feats)} environmental features.")
        
    except StopIteration:
        logger.error(f"No training files found in {sbcv_path}. Cannot determine features.")

    experiments = []
    
    # 1. DeepSAR Area Only
    experiments.append({
        "name": "DeepSAR_Area",
        "model_init": Deep4PWeibullInit(feature_names=["log_sp_unit_area"]),
        "feature_names": ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # 2. DeepSAR All Env
    experiments.append({
        "name": "DeepSAR_Env",
        "model_init": Deep4PWeibullInit(feature_names=all_env_feats),
        "feature_names": all_env_feats,
        "train_frac": 1.0
    })
    
    # 3. DeepSAR All Env + Area
    experiments.append({
        "name": "DeepSAR_All",
        "model_init": Deep4PWeibullInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
        "feature_names": all_env_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # 4. Varying training samples (on All Env + Area)
    for frac in [0.1, 0.5]:
        experiments.append({
            "name": f"DeepSAR_All_frac_{frac}",
            "model_init": Deep4PWeibullInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
            "feature_names": all_env_feats + ["log_sp_unit_area"],
            "train_frac": frac
        })
        
    # 5. Varying architecture (on All Env + Area)
    # Base 64
    experiments.append({
        "name": "DeepSAR_All_Base64",
        "model_init": Deep4PWeibullInit(feature_names=all_env_feats + ["log_sp_unit_area"], 
                                        architecture=symmetric_arch(6, base=64, factor=4)),
        "feature_names": all_env_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # 6. MLP (on All Env + Area)
    experiments.append({
        "name": "MLP_All",
        "model_init": MLPInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
        "feature_names": all_env_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # Run Benchmark
    logger.info("Running Benchmark (Interpolation & Extrapolation)...")
    bench = Benchmarker(config)
    results = []
    for exp in experiments:
        logger.info(f"Running experiment: {exp['name']}")
        res = bench.run(exp["name"], exp["model_init"], exp["feature_names"], exp["train_frac"])
        results.append(res)
    
    df_results = pd.concat(results)
    df_results.to_csv(root_folder / "benchmark_results.csv", index=False)
    
    logger.info("Benchmark completed.")

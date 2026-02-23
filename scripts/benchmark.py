"""
This script benchmarks the MuScaRi model on SBCV datasets, evaluating both
Interpolation (on SBCV test set) and Extrapolation (on GIFT dataset).
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

from muscari.utils import symmetric_arch
from muscari.benchmarker import BenchmarkConfig, Benchmarker
from muscari.muscari import MuScaRi
from muscari.ffnn import FFNNExp
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

GIFT_SAMPLES_PATH = Path(__file__).parent / "../data/processed/test_samples_GIFT/1085825/compiled_data.parquet"
SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/ceacce0"
BIOCLIMATE_VARS = [
            "bio1",
            "pet_penman_mean",
            "sfcWind_mean",
            # "bio4",
            # "rsds_1981-2010_range_V.2.1",
            "bio12",
            # "bio15",
        ]

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
class MuScaRiInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=128, factor=4))
    def __call__(self, **kwargs):
        return MuScaRi(layer_sizes=self.architecture,
                       feature_names=self.feature_names,
                       **kwargs)

class WrappedFFNNExp(FFNNExp):
    def __init__(self, input_dim, layer_sizes, feature_names=[], feature_scaler=None, target_scaler=None):
        super().__init__(input_dim, layer_sizes)
        self.feature_names = feature_names
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

@dataclass
class FFNNExpInit():
    feature_names: list
    architecture: list = field(default_factory=lambda: symmetric_arch(6, base=64, factor=4))
    def __call__(self, **kwargs):
        # input_dim = len(feature_names) + 1 (for log_observed_area)
        return WrappedFFNNExp(len(self.feature_names) + 1, self.architecture, **kwargs)

if __name__ == "__main__":
    root_folder = Path(__file__).parent / Path('results', 'benchmark')
    root_folder.mkdir(parents=True, exist_ok=True)
    
    config = BenchmarkConfig(path_gift_data= GIFT_SAMPLES_PATH,
                             path_sbcv_data= SBCV_SAMPLES_PATH,
                             )
    
    # Inspect one file to get feature names
    sample_file = next(config.path_sbcv_data.glob("*_train.parquet"))
    df = gpd.read_parquet(sample_file)
    
    # Identify features
    climate_feats = BIOCLIMATE_VARS + [f"std_{v}" for v in BIOCLIMATE_VARS]
    dem_feats = ["elevation", "std_elevation"]
    lc_feats = [c for c in df.columns if c.startswith("lc_frac_")]
    
    # Filter to what is actually present
    climate_feats = [c for c in climate_feats if c in df.columns]
    dem_feats = [c for c in dem_feats if c in df.columns]
    
    climate_dem_feats = climate_feats + dem_feats
    landcover_feats = lc_feats
    all_env_feats = climate_dem_feats + landcover_feats
    logger.info(f"Identified features: {len(all_env_feats)} environmental features.")
        
    experiments = []
    
    # MuScaRi Area Only
    experiments.append({
        "name": "MuScaRi_Area",
        "model_init": MuScaRiInit(feature_names=["log_sp_unit_area"]),
        "feature_names": ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # MuScaRi Climate+DEM
    experiments.append({
        "name": "MuScaRi_ClimateDEM",
        "model_init": MuScaRiInit(feature_names=climate_dem_feats),
        "feature_names": climate_dem_feats,
        "train_frac": 1.0
    })

    # MuScaRi Landcover only
    experiments.append({
        "name": "MuScaRi_Landcover",
        "model_init": MuScaRiInit(feature_names=landcover_feats),
        "feature_names": landcover_feats,
        "train_frac": 1.0
    })

    # MuScaRi Climate+DEM + Landcover
    experiments.append({
        "name": "MuScaRi_ClimateDEM_Landcover",
        "model_init": MuScaRiInit(feature_names=climate_dem_feats + landcover_feats),
        "feature_names": climate_dem_feats + landcover_feats,
        "train_frac": 1.0
    })
    
    # MuScaRi Climate+DEM + Area
    experiments.append({
        "name": "MuScaRi_ClimateDEM_Area",
        "model_init": MuScaRiInit(feature_names=climate_dem_feats + ["log_sp_unit_area"]),
        "feature_names": climate_dem_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })

    # MuScaRi Landcover + Area
    experiments.append({
        "name": "MuScaRi_Landcover_Area",
        "model_init": MuScaRiInit(feature_names=landcover_feats + ["log_sp_unit_area"]),
        "feature_names": landcover_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })

    # MuScaRi All Env + Area
    experiments.append({
        "name": "MuScaRi_All",
        "model_init": MuScaRiInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
        "feature_names": all_env_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    # Varying training samples (on All Env + Area)
    # for frac in [0.01, 0.1]:
    #     experiments.append({
    #         "name": f"MuScaRi_All_frac_{frac}",
    #         "model_init": MuScaRiInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
    #         "feature_names": all_env_feats + ["log_sp_unit_area"],
    #         "train_frac": frac
    #     })
        
    # # Varying architecture (on All Env + Area)
    # # Base 64
    # experiments.append({
    #     "name": "MuScaRi_All_Base64",
    #     "model_init": MuScaRiInit(feature_names=all_env_feats + ["log_sp_unit_area"], 
    #                                     architecture=symmetric_arch(6, base=64, factor=4)),
    #     "feature_names": all_env_feats + ["log_sp_unit_area"],
    #     "train_frac": 1.0
    # })
    
    # FFNNExp (on All Env + Area)
    experiments.append({
        "name": "FFNN_All",
        "model_init": FFNNExpInit(feature_names=all_env_feats + ["log_sp_unit_area"]),
        "feature_names": all_env_feats + ["log_sp_unit_area"],
        "train_frac": 1.0
    })
    
    experiments.append({
        "name": "FFNN_ClimateDEM_Area",
        "model_init": FFNNExpInit(feature_names=climate_dem_feats + ["log_sp_unit_area"]),
        "feature_names": climate_dem_feats + ["log_sp_unit_area"],
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
    output_file = root_folder / f"benchmark_results_{SBCV_SAMPLES_PATH.name}_reduced_bioclim_vars.csv"
    df_results.to_csv(output_file, index=False)
    
    logger.info(f"Benchmark completed, output saved at {output_file}.")
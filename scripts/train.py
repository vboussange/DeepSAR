"""
Training an ensemble `Deep4PWeibull` model.
"""
import logging
import torch
from deepsar.ensemble_trainer import EnsembleConfig, EnsembleTrainer
import numpy as np

import pandas as pd
import geopandas as gpd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HASH = "0b85791"

if __name__ == "__main__":
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]

    config = EnsembleConfig(devices=devices, 
                            hash_data=HASH, 
                            run_name=f"checkpoint_deep4pweibull_basearch6_{HASH}", 
                            path_eva_data=Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_megaplot_data.parquet")
    config.run_folder = Path(Path(__file__).parent, 'results', f"{Path(__file__).stem}")
    config.run_folder.mkdir(exist_ok=True, parents=True)

    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset = eva_dataset.dropna()
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["megaplot_area"]) # TODO: legacy name, to be changed in the future
    eva_dataset = eva_dataset[eva_dataset["num_plots"] > 2]  # TODO: to change

    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    predictors = ["log_observed_area", "log_sp_unit_area"] + climate_features

    ensemble_trainer = EnsembleTrainer(config, eva_dataset)
    results = ensemble_trainer.run(predictors)

    results["config"] = config
    logger.info(f"Saving results in {config.run_folder}")
    torch.save(results, config.run_folder / f"{config.run_name}.pth")
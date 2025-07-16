""""
Plotting environmental feature correlation structure.'
"""
import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

from deepsar.deep4pweibull import initialize_ensemble_model
from deepsar.plotting import CMAP_BR
import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from deepsar.deep4pweibull import initialize_ensemble_model
from train import Config, Trainer

MODEL_NAME = "MSEfit_lowlr_nosmallsp_units2_basearch6_0b85791"

def load_data_and_model():
    """Load model and data."""
    path_results = Path(__file__).parent / f"../../../scripts/results/train/checkpoint_{MODEL_NAME}.pth"
    results_fit_split = torch.load(path_results, map_location="cpu")
    config = results_fit_split["config"]
    
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    
    model = initialize_ensemble_model(
        results_fit_split["ensemble_model_state_dict"], 
        results_fit_split["predictors"], 
        config
    )
    
    return model, results_fit_split, eva_dataset, config

if __name__ == "__main__":    
    model, results_fit_split, eva_dataset, config = load_data_and_model()

    config = results_fit_split["config"]
    predictors = results_fit_split["predictors"][1:]

    
    # Load all data to fit PCA globally
    eva_dataset = eva_dataset[predictors]
    eva_dataset = eva_dataset.rename(columns={"log_sp_unit_area": "log_area"})
    corr_matrix = eva_dataset.corr()
    
    fig, ax = plt.subplots(figsize=(15, 12))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap=CMAP_BR, square=True, ax=ax, cbar_kws={'label': 'Correlation', 'ticks': [i/10 for i in range(-10, 11)]})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig("correlation_feature_EVA.pdf", transparent=True, dpi=300, bbox_inches='tight')
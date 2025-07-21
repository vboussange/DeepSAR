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

from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.plotting import CMAP_BR
from deepsar.ensemble_trainer import EnsembleConfig

MODEL_NAME = "deep4pweibull_basearch6_0b85791"

def load_data_and_model():
    """Load model and data."""
    path_results = Path(__file__).parent / f"../../../scripts/results/train/checkpoint_{MODEL_NAME}.pth"
    checkpoint = torch.load(path_results, map_location="cpu")
    config = checkpoint["config"]
    
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    
    model = Deep4PWeibull.initialize_ensemble(checkpoint)
    
    return model, checkpoint, eva_dataset, config

if __name__ == "__main__":    
    model, checkpoint, eva_dataset, config = load_data_and_model()

    config = checkpoint["config"]
    predictors = checkpoint["predictors"][1:]

    
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
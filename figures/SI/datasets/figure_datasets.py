"""Plotting species richness against area and coverage for EVA and GIFT datasets."""

import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

from deepsar.neural_4pweibull import initialize_ensemble_model
from deepsar.plotting import CMAP_BR
import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from deepsar.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer

# Assuming load_preprocessed_data is already defined
MODEL_NAME = "MSEfit_lowlr_nosmallsp_units2_basearch6_0b85791"

def load_data():
    
    # Load model and data for EVA predictions
    path_results = Path(f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")
        
    # Load model results
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]

    # Load EVA dataset
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["coverage"] = eva_dataset["log_observed_area"] / eva_dataset["log_sp_unit_area"]
    eva_dataset = eva_dataset.sample(n=5000, random_state=42)  # Sample 1000 points for visualization
    gift_data_dir = Path("../../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")
    
    # Load GIFT dataset
    gift_dataset = gpd.read_parquet(gift_data_dir / "sp_unit_data.parquet")
    gift_dataset["log_sp_unit_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate log-transformed species richness (log_sr) for both datasets
    eva_dataset["log_sr"] = np.log(eva_dataset["sr"])
    gift_dataset["log_sr"] = np.log(gift_dataset["sr"])
    
    return eva_dataset, gift_dataset


if __name__ == "__main__":
    eva_dataset, gift_dataset = load_data()
    colors = ["#f72585","#4cc9f0"]
    
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot EVA dataset
    scatter = ax.scatter(
        np.exp(eva_dataset["log_sp_unit_area"]) / 1e6,
        np.exp(eva_dataset["log_sr"]),
        c=eva_dataset["coverage"],
        cmap=CMAP_BR,
        alpha=0.6,
        vmax = 1,
        label="EVA Dataset",
        s=10
    )
    # Add a colorbar to indicate log_observed_area
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Sampling effort")

    # Plot GIFT dataset
    ax.scatter(
        np.exp(gift_dataset["log_sp_unit_area"]) / 1e6,
        np.exp(gift_dataset["log_sr"]),
        color=colors[0],
        alpha=1,
        label="GIFT Dataset",
        s=20,
        marker="x",  # Use a different marker
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Area (km²)")
    ax.set_ylabel("Species richness")
    ax.legend()
    fig.savefig("figure_datasets.pdf", dpi=300, bbox_inches="tight")
    

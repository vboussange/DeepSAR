import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

from src.neural_4pweibull import initialize_ensemble_model
from src.plotting import CMAP_BR
import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from src.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer

# Assuming load_preprocessed_data is already defined

def load_data():
    
    # Load model and data for EVA predictions
    path_results = Path("../../../scripts/results/train_seed_1/checkpoint_MSEfit_large_0b85791.pth")
    
    # Load model results
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]

    # Load EVA dataset
    eva_dataset = gpd.read_parquet(config.path_eva_data)
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["coverage"] = eva_dataset["log_observed_area"] / eva_dataset["log_megaplot_area"]
    eva_dataset = eva_dataset.sample(n=5000, random_state=42)  # Sample 1000 points for visualization
    gift_data_dir = Path("../../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")
    
    # Load GIFT dataset
    gift_dataset = gpd.read_parquet(gift_data_dir / "megaplot_data.parquet")
    gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate log-transformed species richness (log_sr) for both datasets
    eva_dataset["log_sr"] = np.log(eva_dataset["sr"])
    gift_dataset["log_sr"] = np.log(gift_dataset["sr"])
    
    return eva_dataset, gift_dataset

def create_subplot_for_habitat(ax, gdf, hab_name):
    """Creates a scatter plot and histograms for a specific habitat."""
    # Binning log-transformed area and calculating std of species richness (log_sr)
    # gdf['log_area_bins'] = pd.cut(gdf['log_megaplot_area'], bins=100, labels=False)
    # std_sr = gdf.groupby('log_area_bins')['log_sr'].std()
    # log_megaplot_area = gdf.groupby('log_area_bins')['log_megaplot_area'].mean()
    # Scatter plot using hexbin for faster rendering with many points
    hexbin = ax.hexbin(np.exp(gdf.log_megaplot_area), 
                       np.exp(gdf.log_sr), 
                       gridsize=30, 
                       cmap='viridis', 
                       xscale="log",
                       yscale = "log",
                       mincnt=1)

    ax.set_xlabel("Area")
    ax.set_ylabel("Species Richness")
    # Add a colorbar to indicate density
    cbar = plt.colorbar(hexbin, ax=ax, shrink=0.5)
    cbar.set_label('Data density')

    # Set the title with habitat name and the number of data points
    ax.set_title(f"{hab_name}")

if __name__ == "__main__":
    eva_dataset, gift_dataset = load_data()
    colors = ["#f72585","#4cc9f0"]
    
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot EVA dataset
    scatter = ax.scatter(
        np.exp(eva_dataset["log_megaplot_area"]) / 1e6,
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
        np.exp(gift_dataset["log_megaplot_area"]) / 1e6,
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
    

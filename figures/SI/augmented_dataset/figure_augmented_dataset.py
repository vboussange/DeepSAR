import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import sys
PATH_MLP_TRAINING = Path("../../../scripts/")
sys.path.append(str(Path(__file__).parent / PATH_MLP_TRAINING))
from train import Config, compile_training_data
from src.plotting import read_result

# Assuming load_preprocessed_data is already defined

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
    # # Add a secondary y-axis to show standard deviation
    # ax_std = ax.twinx()
    # ax_std.plot(np.exp(log_megaplot_area), std_sr, color='red', linestyle='--', label='Std of SR', alpha=0.7)
    # ax_std.set_ylabel("Std of SR", color='red')
    # ax_std.tick_params(axis='y', labelcolor='red')
    
    # # Add a histogram for the area
    # ax_histx = ax.inset_axes([0.1, 0.85, 0.8, 0.15])  # X-axis histogram inset
    # ax_histx.hist(gdf['log_megaplot_area'], bins=30, color='gray')
    # # ax_histx.set_xscale('log')
    # ax_histx.axis('off')  # Hide axes for the histograms
    # # Add an inset histogram for the distribution of the species richness
    # ax_histy = ax.inset_axes([0.85, 0.1, 0.15, 0.8])  # Y-axis histogram inset
    # ax_histy.hist(gdf['log_sr'], bins=30, orientation='horizontal', color='gray')
    # # ax_histy.set_yscale('log')
    # ax_histy.axis('off')  # Hide axes for the histograms

    # Set the title with habitat name and the number of data points
    ax.set_title(f"{hab_name}")

if __name__ == "__main__":
    seed = 1
    MODEL = "large"
    HASH = "ee40db7"
    path_eva_data = Path(__file__).parent / f"../../../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_augmented_data.pkl"
    path_gift_data = Path(__file__).parent / f"../../../data/processed/GIFT_CHELSA_compilation/{HASH}/megaplot_data.gpkg"

    eva_data = read_result(path_eva_data)
    gift_data = gpd.read_file(path_gift_data)
    habitats = ["all", "T", "R", "Q", "S"]

    # Create a grid for plotting
    n_habs = len(habitats)
    n_rows = int(np.ceil(n_habs / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    # Iterate over habitats and plot each in its corresponding grid cell
    for i, hab in enumerate(habitats):
        augmented_data = compile_training_data(eva_data, gift_data, hab, 1)
        create_subplot_for_habitat(axes[i], augmented_data, hab)

    # Remove any empty subplots if the number of habitats doesn't fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.savefig(Path(__file__).stem + ".png", dpi=300, transparent=True)

    plt.show()

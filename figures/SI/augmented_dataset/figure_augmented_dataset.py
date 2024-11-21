import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/eva_processing/")))
from preprocess_eva_CHELSA_EUNIS_plot_megaplot_ratio_1_1 import load_preprocessed_data

# Assuming load_preprocessed_data is already defined

def create_subplot_for_habitat(ax, gdf, hab_name):
    """Creates a scatter plot and histograms for a specific habitat."""
    # Binning log-transformed area and calculating std of species richness (log_sr)
    gdf['log_area_bins'] = pd.cut(gdf['log_area'], bins=100, labels=False)
    std_sr = gdf.groupby('log_area_bins')['log_sr'].std()
    log_area = gdf.groupby('log_area_bins')['log_area'].mean()

    # Scatter plot
    scatter = ax.scatter(np.exp(gdf.log_area), np.exp(gdf.log_sr), label="SR vs Area", alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Area")
    ax.set_ylabel("Species Richness")
    
    # Add a secondary y-axis to show standard deviation
    ax_std = ax.twinx()
    ax_std.plot(np.exp(log_area), std_sr, color='red', linestyle='--', label='Std of SR', alpha=0.7)
    ax_std.set_ylabel("Std of SR", color='red')
    ax_std.tick_params(axis='y', labelcolor='red')
    
    # Add a histogram for the area
    ax_histx = ax.inset_axes([0.1, 0.85, 0.8, 0.15])  # X-axis histogram inset
    ax_histx.hist(gdf['log_area'], bins=30, color='gray')
    # ax_histx.set_xscale('log')
    ax_histx.axis('off')  # Hide axes for the histograms
    # Add an inset histogram for the distribution of the species richness
    ax_histy = ax.inset_axes([0.85, 0.1, 0.15, 0.8])  # Y-axis histogram inset
    ax_histy.hist(gdf['log_sr'], bins=30, orientation='horizontal', color='gray')
    # ax_histy.set_yscale('log')
    ax_histy.axis('off')  # Hide axes for the histograms

    # Set the title with habitat name and the number of data points
    ax.set_title(f"{hab_name}")

if __name__ == "__main__":
    seed = 1
    checkpoint_path = f"../../../scripts/MLP3/results/MLP_fit_torch_all_habs_dSRdA_weight_1e+00_seed_{seed}/checkpoint.pth"
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")

    config = results_fit_split_all["config"]
    habitats = ["T1", "T3", "R1", "R2", "Q2", "Q5", "S2", "S3", "all"]

    # Create a grid for plotting
    n_habs = len(habitats)
    n_rows = int(np.ceil(n_habs / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    # Iterate over habitats and plot each in its corresponding grid cell
    for i, hab in enumerate(habitats):
        gdf = load_preprocessed_data(hab, config["hash"], config["data_seed"])
        create_subplot_for_habitat(axes[i], gdf, hab)

    # Remove any empty subplots if the number of habitats doesn't fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.savefig(Path(__file__).stem + ".png", dpi=300, transparent=True)

    plt.show()

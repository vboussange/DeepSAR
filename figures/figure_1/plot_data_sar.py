"""
Plotting the raw data for 'all' habitat, differentiating between GIFT, EVA and
augmented datasets, and partial surveys
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import sys
from src.plotting import read_result
from src.dataset import AugmentedDataset


if __name__ == "__main__":
    MODEL = "large"
    HASH = "627173c"
    path_eva_data = Path(__file__).parent / f"../../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_augmented_data.pkl"
    path_gift_data = Path(__file__).parent / f"../../data/processed/GIFT_CHELSA_compilation/{HASH}/megaplot_data.gpkg"
    
    augmented_dataset = AugmentedDataset(path_eva_data = path_eva_data,
                                        path_gift_data = path_gift_data,
                                        seed = 1)
    
    df = augmented_dataset.compile_training_data("all")
    
    # calculating percentage of survey cover
    df["coverage"] = df["area"] / df["megaplot_area"]

    fig, ax = plt.subplots()

    markers = {'GIFT': 'o', 'EVA_raw': 's', 'EVA_megaplot': '^'}
    norm = mcolors.LogNorm(vmin=df['coverage'].min(), vmax=df['coverage'].max())

    for t in df['type'].unique():
        
        df_subset = df[df['type'] == t]
        if len(df_subset) > 300:
            df_subset = df_subset.sample(n=300, random_state=1) # Use a fixed random state for reproducibility
        scatter = ax.scatter(df_subset['megaplot_area'], df_subset['sr'],
                            c=df_subset['coverage'],
                            marker=markers[t],
                            norm=norm,
                            label=t,
                            s = 25,
                            alpha=0.7,
                            cmap='magma',
                            edgecolor='k',
                            linewidth=0.5,
                            )

    ax.set_xlabel("Area (m$^2$)")
    ax.set_ylabel("Species richness")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Survey coverage (%)")
    
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f"data_sar.png", dpi=300, transparent=True)


#######
# quick test


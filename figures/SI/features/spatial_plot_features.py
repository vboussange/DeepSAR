""""
Not up to date.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt


from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_landcover import CopernicusDataset
import sys

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_EVA_Copernicus import (
    process_results,
)

if __name__ == "__main__":
    ncells = 10
    
    dataset = process_results()
    env_vars = dataset.config["env_vars"]
    
    env_pred_dataset = CHELSADataset()
    CHELSA_arr = env_pred_dataset.load()
    coarse = CHELSA_arr.coarsen(x=ncells, y=ncells, boundary="trim")
    coarse_mean = coarse.mean().to_dataset(dim="variable")
    coarse_std = coarse.std().to_dataset(dim="variable")
    
    fig, axs = plt.subplots(len(env_vars), 2, figsize=(7, 14), sharex=True, sharey=True)
    for i, var in enumerate(env_vars):
        coarse_mean[var].plot(ax=axs[i, 0], cmap="coolwarm", label = "mean " + var)
        coarse_std[var].plot(ax=axs[i, 1], cmap="OrRd")

        # axs[i, 0].set_title("mean " + var)
        # axs[i, 1].set_title("std " + var)
        for ax in axs[i,:2]:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

    axs[0, 0].set_title("10x10km2 block mean")
    axs[0, 1].set_title("10x10km2 block St.d.")
    fig.tight_layout()
    fig.savefig("figSI_feature_maps.png", dpi=300, transparent=True)

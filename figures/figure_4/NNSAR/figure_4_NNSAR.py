"""
This script compares the performance of a XGBoost model and a NN-based model
We do a spatial block cross validation.
"""
import matplotlib.pyplot as plt
import pickle
import xarray as xr

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from xgboost import XGBRegressor
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from src.plotting import COLOR_PALETTE
from src.NNSAR import NNSAR2
from src.utils import save_to_pickle
from src.data_processing.utils_env_pred import CHELSADataset, CHELSA_PATH

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../figure_2/")))
from NNSAR_figure_2 import (
    process_results,
)

def plot_raster(rast, label, ax, cmap, vmin=None, vmax=None):
        print("Plotting...")
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":"","pad":0.05, "location":"bottom"} #if display_cbar else {}
        # rolling window for smoothing
        rast.rolling(x=2, y=2, center=True, min_periods=1).mean().where(rast > 0.).plot(ax=ax,
                                                                                        cmap=cmap, 
                                                                                        cbar_kwargs=cbar_kwargs, 
                                                                                        vmin=vmin, 
                                                                                        vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")

            
result_path = Path(str(Path(__file__).parent), "../../scripts/NNSAR/NNSAR_project_coarse_nruns.pkl")
with open(result_path, 'rb') as file:
    full_maps = pickle.load(file)


# plotting
fig_mean_run, axs_mean_run = plt.subplots(3, 3, figsize=(8, 9))
fig_std_run, axs_std_run = plt.subplots(3, 3, figsize=(8, 9))

plot_metadata = {"sr": "SR", "logc": "$\log(c)$\n(Local species richness)", "z": "$z$\n(Species turnover)"}
for i, ncells  in enumerate([20, 50, 100]):
    for j, quantity in enumerate(["sr", "logc", "z"]):
        rast = xr.concat(full_maps[ncells][quantity], dim="runs")
        mean_rast = rast.median(dim="runs")
        mean_rast = np.exp(mean_rast) if quantity == "sr" or quantity == "logc" else mean_rast
        std_rast = rast.std(dim="runs")
        titles = plot_metadata[quantity] if i == 0 else ""
        cmap = "OrRd" if quantity == "z" else "PuBu" if quantity == "logc" else "YlGn"
        plot_raster(mean_rast, titles, axs_mean_run[i, j], cmap)
        plot_raster(std_rast, titles, axs_std_run[i, j], cmap)
        for ax in [axs_mean_run[i, j], axs_std_run[i, j]]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        if j == 0:
            axs_mean_run[i, j].set_ylabel(f"Resolution: {ncells}km$^2$")
            axs_std_run[i, j].set_ylabel(f"Resolution: {ncells}km$^2$")

fig_mean_run.tight_layout()
fig_std_run.tight_layout()
fig_mean_run.savefig("mean_run_c_z_sr_coarsening_hig_med_low_res.png", transparent=True, dpi=300)
fig_mean_run.savefig("std_run_c_z_sr_coarsening_hig_med_low_res.png", transparent=True, dpi=300)

# for ax in axs.flatten():
#     ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    
    # fig.tight_layout()
    # fig
    # fig.savefig(str(Path(__file__).parent / Path(__file__).stem) + ".png", dpi=300)
    
    
    # # fig, ax = plt.subplots()
    # # ax.scatter(sr, beta)
    # # ax.set_xlabel("sr")
    # # ax.set_ylabel("beta")
    
    # # fig, ax = plt.subplots()
    # # ax.scatter(alpha, beta)
    # # ax.set_xlabel("alpha")
    # # ax.set_ylabel("beta")
    
    # print("Correlation betwee alpha and beta", xr.corr(alpha_rast, beta_rast))
    
    


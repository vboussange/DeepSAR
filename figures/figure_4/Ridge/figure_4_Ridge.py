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

import sys
from pathlib import Path

def plot_raster(rast, label, ax, vmin=None, vmax=None):
        print("Plotting...")
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":label,"pad":0.05} #if display_cbar else {}
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        # rolling window for smoothing
        rast.rolling(x=5, y=5, center=True, min_periods=1).mean().plot(ax=ax,cmap="OrRd", cbar_kwargs=cbar_kwargs, vmin=vmin, vmax=vmax)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

            
result_path = Path(str(Path(__file__).parent), "../../scripts/GLM/Ridge_project_coarse.pkl")
with open(result_path, 'rb') as file:
    full_maps = pickle.load(file)


# plotting
fig, axs = plt.subplots(3, 3, figsize=(8, 9))

plot_metadata = {"sr": "SR", "logc": "$\log(c)$\n(Local species richness)", "z": "$z$\n(Species turnover)"}
for i, ncells  in enumerate([25, 50, 100]):
    for j, quantity in enumerate(["sr", "logc", "z"]):
        ax = axs[i, j]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
        rast = full_maps[ncells][quantity]

        titles = plot_metadata[quantity] if i == 2 else ""
        plot_raster(rast, titles, ax)

        if j == 1:
            axs[i, j].set_title(f"Resolution: {ncells}km$^2$")

fig.tight_layout()
fig.savefig("Ridge_c_z_sr_coarsening_hig_med_low_res.png", transparent=True, dpi=300)

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
    
    


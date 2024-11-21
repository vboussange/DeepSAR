# in v2 we use plot:megaplot ratio 1:1 for augmented dataset
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
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


from src.plotting import COLOR_PALETTE
from src.NNSAR import NNSAR2
from src.utils import save_to_pickle
from src.data_processing.utils_env_pred import CHELSADataset, CHELSA_PATH

import sys
from pathlib import Path

WORLD = gpd.read_file("/home/boussang/SAR_modelling/data/naturalearth/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

def create_X_map(climate_predictors, ncells=5):
    env_pred_dataset = CHELSADataset()
    CHELSA_arr = env_pred_dataset.load()
    if ncells > 0:
        coarse = CHELSA_arr.coarsen(x=ncells, y=ncells, boundary="trim")
        coarse_mean = coarse.mean().to_dataset(dim="variable")
        coarse_std = coarse.std().to_dataset(dim="variable")
        df_mean = coarse_mean.to_dataframe()
        df_std = coarse_std.to_dataframe()
    else:
        coarse = CHELSA_arr
        coarse_mean = coarse.to_dataset(dim="variable")
        df_mean = coarse_mean.to_dataframe()
        df_std = df_mean.copy()
        df_std[:] = 0
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    return X_map[climate_predictors].dropna()

def create_raster(X_map, ypred):
    Xy_map = X_map.copy()
    Xy_map["pred"] = ypred
    return Xy_map[["pred"]].to_xarray()["pred"].sortby(["y", "x"])


def plot_raster(rast, title, label, ax, cmap, vmin=None, vmax=None, ncells=2):
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":label,"pad":0.05, "location":"bottom"} #if display_cbar else {}
        # rolling window for smoothing
        rast.rolling(x=ncells, y=ncells, center=True, min_periods=1).mean().where(rast > 0.).plot(ax=ax,
                                                                                        cmap=cmap, 
                                                                                        cbar_kwargs=cbar_kwargs, 
                                                                                        vmin=vmin, 
                                                                                        vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

if __name__ == "__main__":
    
    result_path = Path(__file__).parent / Path("../../../../scripts/XGBoost/XGBoost_fit_simple_plot_megaplot.pkl")
    
    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]

    climate_predictors  = results_fit_split["climate_predictors"]
    hab = "all"
    reg = results_fit_split[hab]["reg"]
    gdf = results_fit_split[hab]["gdf"]
    
    params = [{"res": int(1e2), 
                "ncells_coarse":0, 
                "ncells_rolling" : 2,
                "title": "Resolution: 0.01km2"},
              {"res": int(1e3), 
                "ncells_coarse":0, 
                "ncells_rolling" : 2,
                "title": "Resolution: 1km2"},
              {"res": int(1e4), 
                "ncells_coarse":10, 
                "ncells_rolling" : 2,
                "title": "Resolution: 100km2"}]

    fig, axs = plt.subplots(2, 3, figsize=(7, 6))

    for i,p in enumerate(params):
        res = p["res"] #m2
        area_m2 = res**2
        ncells_coarse = p["ncells_coarse"]
        ncells_rolling = p["ncells_rolling"]
        print(f"Creation of X_map with ncells = {ncells_coarse}")
        X_map = create_X_map(climate_predictors, ncells=ncells_coarse)
        # assigning 0 to std
        
        X_map["log_area"] = 0. # must be expressed in m2
        X_map = X_map[["log_area"] + climate_predictors]
        # X_map[[c for c in climate_predictors if "std_" in c]] = 0

        print(f"Computing sr, logc and z for  ncells = {ncells_coarse}")
        # SR
        X_sr = X_map.copy()
        X_sr["log_area"] = np.log(area_m2) # must be expressed in m2
        log_sr = reg.predict(X_sr)
        
        # z
        log_a1 = np.log(area_m2*10)
        log_a0 = np.log(area_m2*0.1)
        X_map0 = X_map.copy()
        X_map0["log_area"] = log_a0
        # X_map0[[c for c in climate_predictors if "std_" in c]] = 0
        X_map1 = X_map.copy()
        X_map1["log_area"] = log_a1
        z = (reg.predict(X_map1) - reg.predict(X_map0)) / (log_a1 - log_a0)
        
        print(f"Projecting on a map")
        rast_sr = create_raster(X_map, np.exp(log_sr))
        if ncells_coarse == 0:
            rast_sr = rast_sr.coarsen(x=10, y=10, boundary="pad").median()
        plot_raster(rast_sr,
                    p["title"],
                    "",
                    axs[0,i],
                    "BuGn",
                    ncells=ncells_rolling)
        rast_dsr = create_raster(X_map, z)
        if ncells_coarse == 0:
            rast_dsr = rast_dsr.coarsen(x=10, y=10, boundary="pad").median()
        plot_raster(rast_dsr,
                    "",
                    "",
                    axs[1,i],
                    "OrRd",
                    ncells=ncells_rolling)

    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        WORLD.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)
    axs[0,0].set_ylabel("Species richness (SR)")
    axs[1,0].set_ylabel("$\\frac{d \\log SR}{d \\log A}$")

    fig.tight_layout()
    fig.savefig(Path(__file__).stem + ".png", transparent=True, dpi=300)
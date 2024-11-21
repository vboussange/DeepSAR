"""
Projecting model predictions spatially.
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
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


from MLP import MLP, get_gradient, inverse_transform_scale_feature_tensor, scale_feature_tensor
from utils_env_pred import CHELSADataset

import sys
from pathlib import Path

def load_model(result, device):
    """Load the model and scalers from the saved checkpoint."""
    
    # Load the model architecture
    predictors = result['predictors']
    model = MLP(len(predictors)).to(device)
    
    # Load model weights and other components
    model.load_state_dict(result['model_state_dict'])
    model.eval()
    return model

def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    res_climate_pixel = abs(climate_dataset.rio.resolution()[0])  # X resolution (in meters)
    return climate_dataset, res_climate_pixel

def update_area(X_map, res_sr_map):
    X_map[["max_lon_diff", "max_lat_diff"]] = res_sr_map
    X_map["log_area"] = np.log(res_sr_map**2)

def create_X_map(env_predictors, ncells, climate_dataset):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
    coarse = climate_dataset.coarsen(x=ncells, y=ncells, boundary="trim")
    
    # See: https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.reproject_match
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035").rio.reproject_match(climate_dataset)
    coarse_std = coarse.std().rio.write_crs("EPSG:3035").rio.reproject_match(climate_dataset)
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)

    return X_map[env_predictors]
    
def get_SR_dSR(model, X_map, res_sr_map, env_predictors, feature_scaler, target_scaler):
        log_area = np.log(res_sr_map**2)
        log_area_tensor = torch.tensor(log_area, dtype=torch.float32, requires_grad=True).expand(len(X_map)).unsqueeze(1)
        env_features = torch.tensor(X_map[env_predictors].values.astype(np.float32), dtype=torch.float32)
        features = torch.cat([log_area_tensor, env_features], dim=1)

        X = scale_feature_tensor(features, feature_scaler)
        y = model(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        dlogSR_dlogA = get_gradient(log_SR, log_area_tensor).detach().numpy()
        log_SR = log_SR.detach().numpy()
        return log_SR, dlogSR_dlogA

def create_raster(X_map, ypred):
    Xy_map = X_map.copy()
    Xy_map["pred"] = ypred
    rast = Xy_map["pred"].to_xarray().sortby(["y","x"])
    rast = xr.DataArray(rast.values.transpose()[::-1,:], dims=["y", "x"], coords={
                            "y": rast.y.values[::-1],  # X coordinates (easting)
                            "x": rast.x.values,  # Y coordinates (northing)
                        },
                        name="pred")
    rast = rast.rio.write_crs("EPSG:3035")
    return rast


def plot_raster(rast, label, ax, cmap, vmin=None, vmax=None, ncells=2):
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":"","pad":0.05, "location":"bottom"} #if display_cbar else {}
        # rolling window for smoothing
        rast.rolling(x=ncells, y=ncells, center=True, min_periods=1).mean().where(rast > 0.).plot(ax=ax,
                                                                                        cmap=cmap, 
                                                                                        cbar_kwargs=cbar_kwargs, 
                                                                                        vmin=vmin, 
                                                                                        vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 2
    checkpoint_path = Path(f"./results/MWE_2_dSRdA_weight_1e+00_seed_{seed}/checkpoint.pth")
    xmap_dict_path = Path("./results/X_maps/X_maps_dict.pkl")
    results_path = Path("./results/MLP_project_simple_full_grad/MLP_project_simple_full_grad.pkl")
    results_fit_split = torch.load(checkpoint_path, map_location="cpu")
    model = load_model(results_fit_split, "cpu")
    

    env_predictors = results_fit_split["predictors"][1:]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    climate_dataset, res_climate_pixel = load_chelsa_and_reproject(env_predictors)
   
    print("Preparing X_map data...")
    X_map = create_X_map(env_predictors, 1, climate_dataset)

    
    SR_dSR_rast_dict = {}
    plotting = False
    print("Calculating SR and dlogSR/dlogA...")
    for res_sr_map in [1e2, 5e2]:
        log_SR, dlogSR_dlogA = get_SR_dSR(model, X_map, res_sr_map, env_predictors, feature_scaler, target_scaler)
        

        ncells_rolling = 4
        log_SR_rast = create_raster(X_map, log_SR)
        dlog_SR_dlogA_rast = create_raster(X_map, dlogSR_dlogA)
        SR_dSR_rast_dict[f"{res_sr_map:.0e}"] = {"log_SR": log_SR_rast, "dlogSR_dlogA": dlog_SR_dlogA_rast}
        
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        print(f"Plotting SR...")
        plot_raster(log_SR_rast,
                    f"SR resolved at {res_sr_map/1e3}km",
                    axs[0],
                    "BuGn",
                    ncells=ncells_rolling)
        print(f"Plotting dlogSR/dlogA...")
        plot_raster(dlog_SR_dlogA_rast,
                    "$\\frac{d \log SR}{d \log A}$",
                    axs[1],
                    "OrRd",
                    vmin=0,
                    vmax=.8,
                    ncells=ncells_rolling)

        for ax in axs.flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        print(f"Saving figure...")
        fig.savefig(Path(__file__).stem + f"_res_{res_sr_map/1e3}km.png", transparent=True, dpi=300)
    
    print(f"Saving projections at {results_path}")

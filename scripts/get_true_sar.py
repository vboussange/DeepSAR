"""
Calculate true SAR at different locations, where environmental variables values change with areas 

Using Ensemble model.

TODO: continue
"""
import torch
import numpy as np
import pandas as pd
from src.utils import save_to_pickle
from src.data_processing.utils_env_pred import CHELSADataset
import matplotlib.pyplot as plt
import xarray as xr
from src.ensemble_model import initialize_ensemble_model

from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient
# from scripts.train import Config

from pathlib import Path
from pyproj import Transformer


def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    res_climate_pixel = abs(climate_dataset.rio.resolution()[0])  # X resolution (in meters)
    return climate_dataset, res_climate_pixel


def update_area(X_map, res_sr_map):
    X_map[["max_lon_diff", "max_lat_diff"]] = res_sr_map
    X_map["log_area"] = np.log(res_sr_map**2)

# `ref` is the reference where all raster data are reprojected
def create_X_map(predictor_labels, ncells, climate_dataset, res_climate_pixel, ref):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
    coarse = climate_dataset.coarsen(x=ncells, y=ncells, boundary="trim")
    
    # See: https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.reproject_match
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035").rio.reproject_match(ref)
    coarse_std = coarse.std().rio.write_crs("EPSG:3035").rio.reproject_match(ref)
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    
    # heterogeity is 0 for ncells < 2 with res_climate_map = 600m, 
    # but becomes > 0 as soon as ncells >= 2
    # to make a more smooth transition, we can interpolate, 
    # assuming that heterogeneity is 0 for a smaller area than 1km2, that we call `default_sr_map`.
    default_res_sr_map = 1e1
    if ncells == 1:
        res_sr_map = default_res_sr_map
    else:
        res_sr_map = ncells * res_climate_pixel

    update_area(X_map, res_sr_map)
    return X_map[predictor_labels]

def interpolate_features(X_map_dict, 
                         log_area_tensor, 
                         res_climate_pixel, 
                         predictors, 
                         batch_index=None):
        with torch.no_grad():
            res_sr_map = np.sqrt(np.exp(log_area_tensor[0].numpy()))
        ncells = 2**max(0, int(np.floor(np.log(res_sr_map /res_climate_pixel) / np.log(2))))
        # print(f"Interpolating raster for area {np.exp(log_area_tensor.detach().cpu().numpy()):.2f} based on values calculated by coarsening with {ncells} and {2*ncells} cells")
        if batch_index is None:
            X_map1 = X_map_dict[ncells]
            X_map2 = X_map_dict[2*ncells]
        else:
            X_map1 = X_map_dict[ncells].iloc[batch_index,:]
            X_map2 = X_map_dict[2*ncells].iloc[batch_index, :]

        features1 = torch.tensor(X_map1[predictors].values.astype(np.float32), dtype=torch.float32)
        features2 = torch.tensor(X_map2[predictors].values.astype(np.float32), dtype=torch.float32)

        env_features = features1[:, 1:] + (features2[:, 1:] - features1[:, 1:]) / (torch.exp(features2[:,:1]) - torch.exp(features1[:, :1])) * (torch.exp(log_area_tensor) - torch.exp(features1[:,:1]))
        features = torch.cat([log_area_tensor, env_features], dim=1)
        return features

# we use batches, otherwise model and data may not fit in memory
def get_SR_dSR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler, batch_size=1024):
    log_area = np.log(res_sr_map**2)
    log_SR_list = []
    dlogSR_dlogA_list = []
    total_length = len(X_map_dict[1])
    
    for i in range(0, total_length, batch_size):
        current_batch_size = min(batch_size, total_length - i)
        log_area_tensor = torch.tensor(log_area, dtype=torch.float32, requires_grad=True).expand(current_batch_size).unsqueeze(1)
        features = interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
        X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
        y = model(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        dlogSR_dlogA = get_gradient(log_SR, log_area_tensor).detach().cpu().numpy()
        log_SR = log_SR.detach().cpu().numpy()
        log_SR_list.append(log_SR)
        dlogSR_dlogA_list.append(dlogSR_dlogA)

    log_SR = np.concatenate(log_SR_list, axis=0)
    dlogSR_dlogA = np.concatenate(dlogSR_dlogA_list, axis=0)
    return log_SR, dlogSR_dlogA

def get_std_SR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler, batch_size=1024):
    log_area = np.log(res_sr_map**2)
    log_SR_list = []
    total_length = len(X_map_dict[1])
    
    for i in range(0, total_length, batch_size):
        current_batch_size = min(batch_size, total_length - i)
        log_area_tensor = torch.tensor(log_area, dtype=torch.float32, requires_grad=True).expand(current_batch_size).unsqueeze(1)
        features = interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
        X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
        y = model.std(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        log_SR = log_SR.detach().cpu().numpy()
        log_SR_list.append(log_SR)

    log_SR = np.concatenate(log_SR_list, axis=0)
    return log_SR
    
if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"
    ncells_ref = 20 # used for coarsening
    true_SAR_path = Path(f".results/true_SAR/true_SAR_ensemble_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")
    xmap_dict_path = Path("./results/X_maps/X_maps_dict.pkl")
    checkpoint_path = Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
    results_fit_split = results_fit_split_all["all"]
    model = initialize_ensemble_model(results_fit_split, results_fit_split_all["config"], "cuda")

    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    climate_dataset, res_climate_pixel = load_chelsa_and_reproject(predictors)
    
    dict_SAR = {"loc1": {"coords": (45.1, 6.3), #lat, long
                       "log_SR_median": [],
                       "log_SR_first_quantile": [],
                       "log_SR_third_quantile": []},
              "loc2": {"coords": (52.9, 8.4),
                       "log_SR_median": [],
                       "log_SR_first_quantile": [],
                       "log_SR_third_quantile": []},
            "loc3": {"coords": (42.1, -5),
                       "log_SR_median": [],
                       "log_SR_first_quantile": [],
                       "log_SR_third_quantile": []}
            }
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035")

    window_size = 2e4
    log_area_array = np.linspace(np.log(1e3), np.log(1e8), 100)
    for loc in dict_SAR:
        print(loc)
        y, x = transformer.transform(*dict_SAR[loc]["coords"])
        reduced_climate_dataset = climate_dataset.sel(
                                            x=slice(x, x + window_size),
                                            y=slice(y, y-window_size)
                                            )

        X_map_dict = {}
        for ncells in 2**np.array(range(5)):
            print(f"Calculating X_map for ncells: {ncells}")
            X_map_dict[ncells] = create_X_map(predictors, ncells, reduced_climate_dataset, res_climate_pixel, reduced_climate_dataset)
        
        for log_area in log_area_array:
            res_sr_map = np.sqrt(np.exp(log_area))
            log_SR, dlogSR_dlogA = get_SR_dSR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler)
            
            minx, miny, maxx, maxy = x, y - window_size, x + window_size, y
            dict_SAR[loc]["coords_epsg_3035"] = (minx, miny, maxx, maxy)
            dict_SAR[loc]["log_SR_median"].append(np.nanmedian(log_SR))
            dict_SAR[loc]["log_SR_first_quantile"].append(np.nanquantile(log_SR, 0.25))
            dict_SAR[loc]["log_SR_third_quantile"].append(np.nanquantile(log_SR, 0.75))
            
    dict_SAR["log_area"] = log_area_array
    
    fig, ax = plt.subplots()
    dict_plot = {"loc1": {"c":"tab:blue"}, "loc2": {"c":"tab:red"}, "loc3": {"c":"tab:purple"}}
    for loc in dict_plot:
        d = dict_SAR[loc]
        arg_plot = dict_plot[loc]
        ax.plot(np.exp(log_area_array), np.exp(d["log_SR_median"]), c=arg_plot["c"])
        ax.fill_between(np.exp(log_area_array), 
                np.exp(d["log_SR_first_quantile"]), np.exp(d["log_SR_third_quantile"]), 
                # label="Neural network",
                linestyle="-", 
                color = arg_plot["c"],
                alpha = 0.4,)
    ax.set_xscale("log")
    ax.set_yscale("log")

    save_to_pickle(true_SAR_path, dict_SAR=dict_SAR)


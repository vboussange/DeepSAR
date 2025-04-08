"""
Calculate true SAR at different locations, where environmental variables values change with areas 

Using Ensemble model.

TODO: this script seems to work, but it needs to be checked.
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
from scripts.train import Config

from pathlib import Path
from pyproj import Transformer


def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    res_climate_pixel = abs(climate_dataset.rio.resolution()[0])  # X resolution (in meters)
    return climate_dataset, res_climate_pixel

    
if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 1
    MODEL = "large"
    HASH = "a53390d"
    path_results = Path(__file__).parent / Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(path_results, map_location="cpu")
    config = results_fit_split_all["config"]
    results_fit_split = results_fit_split_all["all"]
    model = initialize_ensemble_model(results_fit_split, config, "cuda")
    
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    climate_dataset, res_climate_pixel = load_chelsa_and_reproject(predictors)
    
    dict_SAR = {"loc1": {"coords": (45.1, 6.3), #lat, long
                       "log_SR": [],},
              "loc2": {"coords": (53, 8.4),
                       "log_SR": [],},
            "loc3": {"coords": (42.1, -5),
                       "log_SR": [],}
            }
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035")
    window_sizes = np.logspace(np.log10(1e2), np.log10(1e5), 100)
    for loc in dict_SAR:
        print(loc)
        y, x = transformer.transform(*dict_SAR[loc]["coords"])
        for window_size in window_sizes:
            # predictor compilation
            if window_size < res_climate_pixel:
                reduced_climate_dataset = climate_dataset.sel(
                                    x=x,
                                    y=y,
                                    method="nearest",
                                    )
            else:
                reduced_climate_dataset = climate_dataset.sel(
                                                    x=slice(x, x + window_size),
                                                    y=slice(y, y-window_size),
                                                    )
            df_mean = pd.DataFrame({var: [reduced_climate_dataset[var].mean().item()] for var in reduced_climate_dataset.data_vars})
            df_std = pd.DataFrame({f"std_{var}": [reduced_climate_dataset[var].std().item()] for var in reduced_climate_dataset.data_vars})
            X_map = pd.concat([df_mean, df_std], axis=1)
            X_map = X_map.assign(log_area=np.log(window_size**2), log_megaplot_area=np.log(window_size**2))
            X_map = X_map[predictors]
            
            # predictions
            X = torch.tensor(X_map.values, dtype=torch.float32)
            X = scale_feature_tensor(X, feature_scaler).to(next(model.parameters()).device)
            X = X.requires_grad_(True)
            log_SR = model(X)
            log_SR = inverse_transform_scale_feature_tensor(log_SR, target_scaler).detach().cpu().numpy()
            # dlogSR_dlogA = get_gradient(log_SR, X).detach().cpu().numpy()[:,:2].sum(axis=1)
            dict_SAR[loc]["log_SR"].append(log_SR)
        dict_SAR[loc]["log_SR"] = np.concatenate(dict_SAR[loc]["log_SR"], axis=0)

        minx, miny, maxx, maxy = x, y - window_size, x + window_size, y
        dict_SAR[loc]["coords_epsg_3035"] = (minx, miny, maxx, maxy)
            
    dict_SAR["log_area"] = np.log(window_sizes**2)
    
    fig, ax = plt.subplots()
    dict_plot = {"loc1": {"c":"tab:blue"}, "loc2": {"c":"tab:red"}, "loc3": {"c":"tab:purple"}}
    for loc in dict_plot:
        d = dict_SAR[loc]
        arg_plot = dict_plot[loc]
        ax.plot(np.exp(dict_SAR["log_area"]), np.exp(d["log_SR"]), c=arg_plot["c"])
        # ax.fill_between(np.exp(dict_SAR["log_area"]), 
        #         np.exp(d["log_SR_first_quantile"]), np.exp(d["log_SR_third_quantile"]), 
        #         # label="Neural network",
        #         linestyle="-", 
        #         color = arg_plot["c"],
        #         alpha = 0.4,)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(Path(f"results/true_SAR/true_SAR_ensemble_seed_{seed}_model_{MODEL}_hash_{HASH}.pdf"), dpi=300, bbox_inches="tight")
    save_to_pickle(Path(f"results/true_SAR/true_SAR_ensemble_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl"), dict_SAR=dict_SAR)


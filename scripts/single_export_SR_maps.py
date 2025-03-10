"""
Projecting spatially MLP and saving to geotiff files.
# TODO: this version is more recent than export_SR_maps.py, but 
# 1. simplifies the code, with no interpolation of features
# 2. loads output from train_single_habitat; should be modified to load from train.py
# 3. We should also export sensitivity maps, which are not exported here.
"""
import torch
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from train_single_habitat import Config
from src.ensemble_model import initialize_ensemble_model
import get_true_sar
from pathlib import Path
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient
from src.data_processing.utils_env_pred import CHELSADataset
import pandas as pd
from tqdm import tqdm

def create_raster(X_map, ypred):
    Xy_map = X_map.copy()
    Xy_map["pred"] = ypred
    rast = Xy_map["pred"].to_xarray().sortby(["y","x"])
    rast = xr.DataArray(rast.values, dims=["y", "x"], coords={
                            "x": rast.x.values,  # X coordinates (easting)
                            "y": rast.y.values,  # Y coordinates (northing)
                        },
                        name="pred")
    rast = rast.rio.write_crs("EPSG:3035")
    return rast

def plot_raster(rast, label, ax, cmap, vmin=None, vmax=None):
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":"","pad":0.05, "location":"bottom"} #if display_cbar else {}
        # rolling window for smoothing
        rast.where(rast > 0.).plot(ax=ax,
                                    cmap=cmap, 
                                    cbar_kwargs=cbar_kwargs, 
                                    vmin=vmin, 
                                    vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        
def create_features(predictor_labels, climate_dataset, res):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
    resolution = abs(climate_dataset.rio.resolution()[0])
    ncells = int(res / resolution)
    coarse = climate_dataset.coarsen(x=ncells, y=ncells, boundary="trim")
    
    # See: https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.reproject_match
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035")#.rio.reproject_match(ref)
    coarse_std = coarse.std().rio.write_crs("EPSG:3035")#.rio.reproject_match(ref)
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    
    X_map = X_map.assign(log_area=np.log(res**2), log_megaplot_area=np.log(res**2))
    return X_map[predictor_labels]
        
# we use batches, otherwise model and data may not fit in memory
def get_SR_std_SR(model, climate_dataset, res, predictors, feature_scaler, target_scaler, batch_size=2048):
    mean_log_SR_list = []
    std_log_SR_list = []
    features = create_features(predictors, climate_dataset, res)
    total_length = len(features)

    for i in tqdm(range(0, total_length, batch_size), desc = "Calculating SR and stdSR"):
        current_batch_size = min(batch_size, total_length - i)
        # features = get_true_sar.interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
        X = features.iloc[i:i+current_batch_size,:]
        X = torch.tensor(X.values, dtype=torch.float32)
        X = scale_feature_tensor(X, feature_scaler).to(next(model.parameters()).device)
        ys = [m(X) for m in model.models]
        log_SRs = [np.exp(inverse_transform_scale_feature_tensor(y, target_scaler).detach().cpu().numpy()) for y in ys]
        mean_log_SR = np.mean(log_SRs, axis=0)
        std_log_SR = np.std(log_SRs, axis=0)
        mean_log_SR_list.append(mean_log_SR)
        std_log_SR_list.append(std_log_SR)

    mean_log_SR = np.concatenate(mean_log_SR_list, axis=0)
    std_log_SR = np.concatenate(std_log_SR_list, axis=0)

    return features, mean_log_SR, std_log_SR
        
        
def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    return climate_dataset

if __name__ == "__main__":
    results_path = Path("./results/projections")
    checkpoint_path = Path("/home/boussang/DeepSAR/scripts/results/train_single_habitat_dSRdA_weight_1e+00_seed_1/checkpoint_large_model_full_physics_informed_constraint_71f9fc7.pth")    
    
    results_path.mkdir(parents=True, exist_ok=True)
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
    model_params = results_fit_split_all["results"]
    model = initialize_ensemble_model(model_params, results_fit_split_all["config"], "cuda")
    

    predictors = model_params["predictors"]
    feature_scaler = model_params["feature_scaler"]
    target_scaler = model_params["target_scaler"]
    
    
    climate_dataset = load_chelsa_and_reproject(predictors)

    res = 5e3
    print(f"Calculating SR, and stdSR for resolution {res}")
    features, SR, std_SR = get_SR_std_SR(model, climate_dataset, res, predictors, feature_scaler, target_scaler)

    print("Projection predictions on map.")
    SR_rast = create_raster(features, SR)
    std_SR_rast = create_raster(features, std_SR)
    
    SR_rast.rio.to_raster(results_path/f"SR_raster_{res:.0f}m.tif")
    std_SR_rast.rio.to_raster(results_path/f"std_SR_raster_{res:.0f}m.tif")
    
    # fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    # print(f"Plotting SR...")
    # plot_raster(SR_rast.coarsen(x=20, y=20, boundary="trim").mean(),
    #             f"SR resolved at {res_sr_map/1e3}km",
    #             axs[0],
    #             "BuGn")
    # print(f"Plotting dlogSR/dlogA...")
    # plot_raster(std_SR_rast.coarsen(x=20, y=20,  boundary="trim").mean(),
    #             "Std. SR",
    #             axs[1],
    #             "OrRd",)

    # for ax in axs.flatten():
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # fig.tight_layout()
    # fig.savefig(results_path / f"rel_std_SR_{res_sr_map:.0f}m.png")
    
    # relative_std_SR = std_SR / SR

    # relative_std_SR_rast = create_raster(X_map_dict[1], relative_std_SR)

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # print(f"Plotting relative std. SR...")
    # plot_raster(relative_std_SR_rast.coarsen(x=20, y=20, boundary="trim").mean(),
    #             "Relative Std. SR",
    #             ax,
    #             "Reds",)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.tight_layout()
    # fig.savefig(results_path / f"rel_std_SR_{res_sr_map:.0f}m.png")
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
from train import Config
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
    ncells = max(1, int(res / resolution))
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
def get_SR_std_SR_dSR(model, climate_dataset, res, predictors, feature_scaler, target_scaler, batch_size=4096):
    mean_log_SR_list = []
    std_log_SR_list = []
    dlogSR_dlogA_list = []
    features = create_features(predictors, climate_dataset, res)
    total_length = len(features)

    percent_step = max(1, total_length // batch_size // 100)
    
    for i in tqdm(range(0, total_length, batch_size), desc = "Calculating SR and stdSR", miniters=percent_step, maxinterval=float("inf")):
        with torch.no_grad():
            current_batch_size = min(batch_size, total_length - i)
            # features = get_true_sar.interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
            X = features.iloc[i:i+current_batch_size,:]
            X = feature_scaler.transform(X.values)
            X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
            ys = [m(X) for m in model.models]
            log_SRs = [np.exp(target_scaler.inverse_transform(y.cpu().numpy())) for y in ys]
            mean_log_SR = np.mean(log_SRs, axis=0)
            std_log_SR = np.std(log_SRs, axis=0)
            mean_log_SR_list.append(mean_log_SR)
            std_log_SR_list.append(std_log_SR)
        
        # grad: dSR/dA, getting predictions without statistics
        X = X.requires_grad_(True)
        log_SR = model(X)
        dlogSR_dlogA = get_gradient(log_SR, X).detach().cpu().numpy()[:,:2].sum(axis=1)
        dlogSR_dlogA_list.append(dlogSR_dlogA)
        
    mean_log_SR = np.concatenate(mean_log_SR_list, axis=0)
    std_log_SR = np.concatenate(std_log_SR_list, axis=0)
    dlogSR_dlogA = np.concatenate(dlogSR_dlogA_list, axis=0)    
    return features, mean_log_SR, std_log_SR, dlogSR_dlogA
        
        
def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    return climate_dataset

if __name__ == "__main__":
    results_path = Path("./results/projections")
    results_path.mkdir(parents=True, exist_ok=True)

    seed = 1
    MODEL = "large"
    HASH = "fb8bc71"  
    path_results = Path(__file__).parent / Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(path_results, map_location="cpu")
    config = results_fit_split_all["config"]
    results_fit_split = results_fit_split_all["all"]
    
    # TODO: this should be in `results_fit_split`, next commit should fix this
    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    results_fit_split["predictors"] = ["log_area", "log_megaplot_area"] + climate_features
    
    model = initialize_ensemble_model(results_fit_split, config, "cuda")
    

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    
    
    climate_dataset = load_chelsa_and_reproject(predictors)

    for res in [100, 1000, 10000, 100000]:
        print(f"Calculating SR, and stdSR for resolution: {res}m")
        features, SR, std_SR, dlogSR_dlogA = get_SR_std_SR_dSR(model, climate_dataset, res, predictors, feature_scaler, target_scaler)

        SR_rast = create_raster(features, SR)
        SR_rast.rio.to_raster(results_path/ HASH / f"SR_raster_{res:.0f}m.tif")

        std_SR_rast = create_raster(features, std_SR)
        std_SR_rast.rio.to_raster(results_path/ HASH / f"std_SR_raster_{res:.0f}m.tif")
        
        dlogSR_dlogA_rast = create_raster(features, dlogSR_dlogA)
        dlogSR_dlogA_rast.rio.to_raster(results_path/ HASH / f"dlogSR_dlogA_raster_{res:.0f}m.tif")
        print(f"Saved SR, std_SR, dlogSR_dlogA for resolution: {res}m in {results_path}")
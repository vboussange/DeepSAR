"""
Projecting spatially MLP and saving to geotiff files.
"""
import torch
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from pathlib import Path
from deepsar.data_processing.utils_env_pred import CHELSADataset
from deepsar.deep4pweibull import Deep4PWeibull
import pandas as pd
from tqdm import tqdm

from deepsar.ensemble_trainer import EnsembleConfig

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
        
        
def coarsen_climate_data(climate_dataset, ncells):
    """Coarsen climate data to specified resolution."""
    coarse = climate_dataset.coarsen(x=ncells, y=ncells, boundary="trim")
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035")
    coarse_std = coarse.std().rio.write_crs("EPSG:3035")
    return coarse_mean, coarse_std

def create_features(predictor_labels, coarse_mean, coarse_std, res):
    """Create features dataframe from coarsened climate data."""
    
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    
    X_map = X_map.assign(log_observed_area=np.log(res**2), log_sp_unit_area=np.log(res**2))
    return X_map[predictor_labels]
        
# we use batches, otherwise model and data may not fit in memory
def get_SR_dSR_stats(model, climate_dataset, res0, predictors, feature_scaler, target_scaler, batch_size=4096):
    """
    Calculate SR, std_SR and dlogSR_dlogA for the given model and climate
    dataset at a specified resolution. dSR is obtained as a gradient of SR with
    respect to log_sp_unit_area. Does not account for changes in climate
    features with area.
    """
    
    resolution = abs(climate_dataset.rio.resolution()[0])
    ncells0 = max(1, int(res0 / resolution))
    ncells1 = ncells0 + 1
    res1 = ncells1 * resolution
    
    coarse_mean0, coarse_std0 = coarsen_climate_data(climate_dataset, ncells0)
    coarse_mean1, coarse_std1 = coarsen_climate_data(climate_dataset, ncells1)
    # see https://github.com/corteva/rioxarray/issues/298
    coarse_mean1 = coarse_mean1.rio.reproject_match(coarse_mean0).assign_coords(x=coarse_mean0.x, y=coarse_mean0.y)
    coarse_std1 = coarse_std1.rio.reproject_match(coarse_std0).assign_coords(x=coarse_std0.x, y=coarse_std0.y)

    features0 = create_features(predictors, coarse_mean0, coarse_std0, res0)
    features1 = create_features(predictors, coarse_mean1, coarse_std1, res1)

    total_length = len(features0)

    percent_step = max(1, total_length // batch_size // 100)
    
    SR01_list = []
    for features in [features0, features1]:
        SR_list = []
        for i in tqdm(range(0, total_length, batch_size), desc = "Calculating SR and stdSR", miniters=percent_step, maxinterval=float("inf")):
            with torch.no_grad():
                current_batch_size = min(batch_size, total_length - i)
                # features = get_true_sar.interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
                X = features.iloc[i:i+current_batch_size,:]
                X = feature_scaler.transform(X.values)
                X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
                ys = [m.predict_sr(X[:, 1:]) for m in model.models] # predicting asymptote, no need to feed log_observed_area
                SRs = [target_scaler.inverse_transform(y.cpu().numpy()) for y in ys]
                SR_list.append(np.concatenate(SRs, axis=1))
        SR01_list.append(np.concatenate(SR_list, axis=0))



    mean_SR = np.mean(SR01_list[0], axis=1)
    std_SR = np.std(SR01_list[0], axis=1)
    
    mean_SR1 = np.mean(SR01_list[1], axis=1)
    
    dSR_dlogA = (SR01_list[1] - SR01_list[0]) / (res1 - res0)
    mean_dSR_dlogA = np.nanmean(dSR_dlogA, axis=1)
    std_dSR_dlogA = np.std(dSR_dlogA, axis=1)
    return features0, mean_SR, std_SR, mean_dSR_dlogA, std_dSR_dlogA


def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    return climate_dataset

if __name__ == "__main__":
    seed = 1
    MODEL_NAME = "deep4pweibull_basearch6_0b85791"
    plotting = True
    
    projection_path = Path(__file__).parent / Path(f"projections/")
    projection_path.mkdir(parents=True, exist_ok=True)
    
    path_results = Path(__file__).parent / Path(f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")
    checkpoint = torch.load(path_results, map_location="cpu")
    config = checkpoint["config"]    

    predictors = checkpoint["predictors"]
    feature_scaler = checkpoint["feature_scaler"]
    target_scaler = checkpoint["target_scaler"]
    
    model = Deep4PWeibull.initialize_ensemble(checkpoint["ensemble_model_state_dict"], predictors, config)
    
    climate_dataset = load_chelsa_and_reproject(predictors)

    for res in [50000, 1000]:
        print(f"Calculating SR, and stdSR for resolution: {res}m")

        features0, mean_SR, std_SR, mean_dSR_dlogA, std_dSR_dlogA = get_SR_dSR_stats(model, climate_dataset, res, predictors, feature_scaler, target_scaler)

        # Create and save rasters
        raster_configs = [
            ("SR", mean_SR, "SR"),
            ("std_SR", std_SR, "Standard Deviation of SR"),
            ("dSR_dlogA", mean_dSR_dlogA, "dSR/dlogA"),
            ("std_dSR_dlogA", std_dSR_dlogA, "Standard Deviation of dSR/dlogA")
        ]
        
        for raster_name, data, plot_title in raster_configs:
            # Create and save raster
            rast = create_raster(features0, data)
            rast.rio.to_raster(projection_path / f"{raster_name}_raster_{res:.0f}m.tif")
                        
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
            custom_cmap = LinearSegmentedColormap.from_list("species_richness", colors[::-1])
            
            rast_renamed = rast.rename(plot_title)
            rast_renamed.plot(ax=ax, cmap=custom_cmap, vmin=rast.quantile(0.01), vmax=rast.quantile(0.99))
            ax.set_title(f"{plot_title} - Res: {res}m")
            
            fig.savefig(projection_path / f"{raster_name}_raster_{MODEL_NAME}_{res:.0f}m.png", 
                   dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved rasters in {projection_path}")
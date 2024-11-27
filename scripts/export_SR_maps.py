"""
Projecting spatially MLP and saving to geotiff files.

TODO: this should eventually replace project.py, which export obscure .pkl files.
"""
import torch
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.utils import save_to_pickle
from train import Config
from src.ensemble_model import initialize_ensemble_model
import get_true_sar
from pathlib import Path
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient

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
        
        
# we use batches, otherwise model and data may not fit in memory
def get_SR_std_SR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler, batch_size=1024):
    log_area = np.log(res_sr_map**2)
    mean_log_SR_list = []
    std_log_SR_list = []
    total_length = len(X_map_dict[1])
    
    for i in range(0, total_length, batch_size):
        current_batch_size = min(batch_size, total_length - i)
        log_area_tensor = torch.tensor(log_area, dtype=torch.float32, requires_grad=True).expand(current_batch_size).unsqueeze(1)
        features = get_true_sar.interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
        X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
        ys = [m(X) for m in model.models]
        log_SRs = [np.exp(inverse_transform_scale_feature_tensor(y, target_scaler).detach().cpu().numpy()) for y in ys]
        mean_log_SR = np.mean(log_SRs, axis=0)
        std_log_SR = np.std(log_SRs, axis=0)
        mean_log_SR_list.append(mean_log_SR)
        std_log_SR_list.append(std_log_SR)

    mean_log_SR = np.concatenate(mean_log_SR_list, axis=0)
    std_log_SR = np.concatenate(std_log_SR_list, axis=0)

    return mean_log_SR, std_log_SR
        
if __name__ == "__main__":
    if True:
        # creating X_maps for different resolutions
        seed = 1
        MODEL = "large"
        HASH = "71f9fc7"
        
        results_path = Path("./results/projections")
        checkpoint_path = Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
        
        results_path.mkdir(parents=True, exist_ok=True)
        results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
        results_fit_split = results_fit_split_all["all"]
        model = initialize_ensemble_model(results_fit_split, results_fit_split_all["config"], "cuda")
        

        predictors = results_fit_split["predictors"]
        feature_scaler = results_fit_split["feature_scaler"]
        target_scaler = results_fit_split["target_scaler"]
        
        
        climate_dataset, res_climate_pixel = get_true_sar.load_chelsa_and_reproject(predictors)

        print("Calculating climate feature data...")
        ref = climate_dataset["bio1"]
    
        X_map_dict = {}
        for ncells in 2**np.array(range(2)):
            print(f"Calculating X_map for ncells: {ncells}")
            X_map_dict[ncells] = get_true_sar.create_X_map(predictors, ncells, climate_dataset, res_climate_pixel, ref)
        
    
    res_sr_map = 1e3
    print(f"Calculating SR, and stdSR for resolution {res_sr_map}")
    SR, std_SR = get_SR_std_SR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler)

    print("Projection predictions on map.")
    SR_rast = create_raster(X_map_dict[1], SR)
    std_SR_rast = create_raster(X_map_dict[1], std_SR)
    
    SR_rast.rio.to_raster(results_path/f"SR_raster_{res_sr_map:.0f}m.tif")
    std_SR_rast.rio.to_raster(results_path/f"std_SR_raster_{res_sr_map:.0f}m.tif")
    
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    print(f"Plotting SR...")
    plot_raster(SR_rast.coarsen(x=20, y=20, boundary="trim").mean(),
                f"SR resolved at {res_sr_map/1e3}km",
                axs[0],
                "BuGn")
    print(f"Plotting dlogSR/dlogA...")
    plot_raster(std_SR_rast.coarsen(x=20, y=20,  boundary="trim").mean(),
                "Std. SR",
                axs[1],
                "OrRd",)

    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(results_path / f"rel_std_SR_{res_sr_map:.0f}m.png")
    
    relative_std_SR = std_SR / SR

    relative_std_SR_rast = create_raster(X_map_dict[1], relative_std_SR)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    print(f"Plotting relative std. SR...")
    plot_raster(relative_std_SR_rast.coarsen(x=20, y=20, boundary="trim").mean(),
                "Relative Std. SR",
                ax,
                "Reds",)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(results_path / f"rel_std_SR_{res_sr_map:.0f}m.png")
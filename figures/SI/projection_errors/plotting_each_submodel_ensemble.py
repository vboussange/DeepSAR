"""
Plotting SR and dlogSR/dlogA for each submodel in the ensemble.
"""
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
import torch
from src.utils import save_to_pickle
import sys
PATH_MLP3 = "../../../scripts/MLP3"
sys.path.append(PATH_MLP3)
from MLP_fit_torch_all_habs_ensemble import Config
import get_true_SAR_ensemble

from pathlib import Path

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
        
if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 1
    ncells_ref = 20 # used for coarsening
    MODEL = "large"
    HASH = "71f9fc7"
    
    checkpoint_path = Path(PATH_MLP3 + f"/results/MLP_fit_torch_all_habs_ensemble_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")
    xmap_dict_path = Path(PATH_MLP3 + "/results/X_maps/X_maps_dict.pkl")
    result_all = torch.load(checkpoint_path, map_location="cpu")
    results_fit_split = result_all["all"]
    config = result_all["config"]
    model = get_true_SAR_ensemble.load_model(results_fit_split, result_all["config"], "cuda")
    

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    
    # set True at first run
    if True:
        climate_dataset, res_climate_pixel = get_true_SAR_ensemble.load_chelsa_and_reproject(predictors)

        print("Loading X_map data...")
        if not xmap_dict_path.is_file():
            ref = climate_dataset["bio1"].coarsen(x=ncells_ref, y=ncells_ref, boundary="trim").mean()
    
            X_map_dict = {}
            for ncells in 2**np.array(range(7)):
                print(f"Calculating X_map for ncells: {ncells}")
                X_map_dict[ncells] = get_true_SAR_ensemble.create_X_map(predictors, ncells, climate_dataset, res_climate_pixel, ref)
                save_to_pickle(xmap_dict_path, X_map_dict=X_map_dict)
        else:
            with open(xmap_dict_path, 'rb') as pickle_file:
                X_map_dict = pickle.load(pickle_file)["X_map_dict"]
    
    
    SR_dSR_rast_dict = {}
    plotting = True
    print("Calculating SR and dlogSR/dlogA...")
    for res_sr_map in [1e3, 1e4]:
        print(f"Calculating SR, dSR and stdSR for resolution {res_sr_map}")
        log_SR_list = []
        for m in model.models:
            log_SR, dlogSR_dlogA = get_true_SAR_ensemble.get_SR_dSR(m, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler)
            log_SR_rast = create_raster(X_map_dict[1], log_SR)
            log_SR_list.append(log_SR_rast)
        
        if plotting:
            fig, axs = plt.subplots(1, len(log_SR_list), figsize=(12, 4))
            print(f"Plotting SR...")
            for i, ax in enumerate(axs):
                log_SR_rast = log_SR_list[i]
                plot_raster(log_SR_rast,
                            f"SR resolved at {res_sr_map/1e3}km",
                            axs[i],
                            "BuGn",
                            vmin = 5, 
                            vmax=8)

            for ax in axs.flatten():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(Path(checkpoint_path).stem + f"_res_{res_sr_map/1e3}km.png", transparent=True, dpi=300)
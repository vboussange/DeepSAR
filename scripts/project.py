"""
Projecting spatially MLP, 
where 
- heterogeneity value is interpolated to have more sensible predictions
- dlogSR/dlogA is calculated by accounting for the change in heterogeneity with the change in area

Using ensemble methods.
"""
import torch
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.utils import save_to_pickle
from scripts.train import Config
from src.ensemble_model import initialize_ensemble_model
import scripts.get_true_sar as get_true_sar
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
    MODEL = "large"
    HASH = "71f9fc7"
    ncells_ref = 20 # used for coarsening
    results_path = Path("./results/results/MLP_project_simple_full_grad_ensemble/MLP_projections_rasters_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")
    
    xmap_dict_path = Path("./results/X_maps/X_maps_dict.pkl")
    checkpoint_path = Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
    results_fit_split = results_fit_split_all["all"]
    model = initialize_ensemble_model(results_fit_split, results_fit_split_all["config"], "cuda")
    

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    
    
    if True:
        climate_dataset, res_climate_pixel = get_true_sar.load_chelsa_and_reproject(predictors)

    print("Loading X_map data...")
    if not xmap_dict_path.is_file():
        ref = climate_dataset["bio1"].coarsen(x=ncells_ref, y=ncells_ref, boundary="trim").mean()
   
        X_map_dict = {}
        for ncells in 2**np.array(range(7)):
            print(f"Calculating X_map for ncells: {ncells}")
            X_map_dict[ncells] = get_true_sar.create_X_map(predictors, ncells, climate_dataset, res_climate_pixel, ref)
            save_to_pickle(xmap_dict_path, X_map_dict=X_map_dict)
    else:
        with open(xmap_dict_path, 'rb') as pickle_file:
            X_map_dict = pickle.load(pickle_file)["X_map_dict"]
    
    
    SR_dSR_rast_dict = {}
    plotting = True
    print("Calculating SR and dlogSR/dlogA...")
    for res_sr_map in [1e0, 1e1, 1e2, 1e3, 1e4]:
        print(f"Calculating SR, dSR and stdSR for resolution {res_sr_map}")
        log_SR, dlogSR_dlogA = get_true_sar.get_SR_dSR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler)
        std_log_SR = get_true_sar.get_std_SR(model, X_map_dict, res_climate_pixel, res_sr_map, predictors, feature_scaler, target_scaler)

        print("Projection predictions on map.")
        log_SR_rast = create_raster(X_map_dict[1], log_SR)
        dlog_SR_dlogA_rast = create_raster(X_map_dict[1], dlogSR_dlogA)
        std_log_SR_rast = create_raster(X_map_dict[1], std_log_SR)
        SR_dSR_rast_dict[f"{res_sr_map:.0e}"] = {"log_SR": log_SR_rast, "dlogSR_dlogA": dlog_SR_dlogA_rast, "std_log_SR": std_log_SR_rast}
        SR_rast = np.exp(log_SR_rast)
        SR_rast.rio.to_raster("results/" + Path(__file__).stem + f"_res_{res_sr_map:.0e}m.tif")
        
        if plotting:
            fig, axs = plt.subplots(1, 2, figsize=(6, 4))
            print(f"Plotting SR...")
            plot_raster(log_SR_rast,
                        f"SR resolved at {res_sr_map/1e3}km",
                        axs[0],
                        "BuGn")
            print(f"Plotting dlogSR/dlogA...")
            plot_raster(dlog_SR_dlogA_rast,
                        "$\\frac{d \log SR}{d \log A}$",
                        axs[1],
                        "OrRd",)

            for ax in axs.flatten():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.savefig("results/" + Path(__file__).stem + f"_res_{res_sr_map/1e3}km.png", transparent=True, dpi=300)
    
    print(f"Saving projections at {results_path}")
    save_to_pickle(results_path, SR_dSR_rast_dict=SR_dSR_rast_dict)

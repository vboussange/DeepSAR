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
from src.neural_4pweibull import initialize_ensemble_model

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
    MODEL_NAME = "MSEfit_lowlr_nosmallmegaplots2_basearch6_0b85791"
    output_dir = Path("SARs")
    output_dir.mkdir(parents=True, exist_ok=True)


    path_results = Path(__file__).parent / Path(f"../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")
    results_fit_split = torch.load(path_results, map_location="cpu")
    config = results_fit_split["config"]    

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    
    model = initialize_ensemble_model(results_fit_split["ensemble_model_state_dict"], predictors, config)
    
    climate_dataset, res_climate_pixel = load_chelsa_and_reproject(predictors)

    
    dict_SAR = {"loc1": {"coords": (45.1, 6.3), #lat, long
                       "SRs": [],},
              "loc2": {"coords": (53, 8.4),
                       "SRs": [],},
            "loc3": {"coords": (42.1, -5),
                       "SRs": [],}
            }
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035")
    window_sizes = np.logspace(np.log10(1e2), np.log10(1e6), 100)
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
            features = pd.concat([df_mean, df_std], axis=1)
            features = features.assign(log_observed_area=np.log(window_size**2), log_megaplot_area=np.log(window_size**2))
            features = features[predictors]
            
            # predictions
            X = features.values
            X = feature_scaler.transform(X)
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
                ys = np.concatenate([m.predict_sr(X[:, 1:]).cpu().numpy() for m in model.models], axis=1) # predicting asymptote, no need to feed log_observed_area
                SRs = target_scaler.inverse_transform(ys.T).T # inverse transform to get back to original scale
                
            dict_SAR[loc]["SRs"].append(SRs[0])  # SRs[0] since we have only one sample

        dict_SAR[loc]["coords_epsg_3035"] = (x, y)
        
        # Convert to numpy array with shape (len(window_sizes), len(model.models))
        dict_SAR[loc]["SRs"] = np.array(dict_SAR[loc]["SRs"])
            
    dict_SAR["log_area"] = np.log(window_sizes**2)
    
    fig, ax = plt.subplots()
    dict_plot = {"loc1": {"c":"tab:blue"}, "loc2": {"c":"tab:red"}, "loc3": {"c":"tab:purple"}}
    for loc in dict_plot:
        d = dict_SAR[loc]
        arg_plot = dict_plot[loc]
        ax.plot(np.exp(dict_SAR["log_area"]), d["SRs"], c=arg_plot["c"])
        # ax.fill_between(np.exp(dict_SAR["log_area"]), 
        #     np.array(d["SR"]) - np.array(d["std_SR"]), 
        #     np.array(d["SR"]) + np.array(d["std_SR"]), 
        #     color=arg_plot["c"],
        #     alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(output_dir / "SARs.pdf", dpi=300, bbox_inches="tight")
    save_to_pickle(output_dir / "SARs.pkl", dict_SAR=dict_SAR)

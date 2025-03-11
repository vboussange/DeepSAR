# evaluating model trained only with raw data on the raw data

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
from src.dataset import scale_features_targets
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient

from sklearn.metrics import mean_squared_error
from src.plotting import read_result
from tqdm import tqdm
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"
    path_augmented_data = Path(__file__).parent / "../../../data/processed/EVA_CHELSA_raw_compilation/EVA_CHELSA_raw_random_state_2_d84985e.pkl"
    data = read_result(path_augmented_data)
    hab = "all"
    checkpoint_path = Path(f"../../../scripts/results/train_raw_plot_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_plot_only_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
    
    
    results_fit_split = results_fit_split_all[hab]
    model = initialize_ensemble_model(results_fit_split, results_fit_split_all["config"], "cuda:0")

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    

    gdf_test = data["plot_data_all"]
    if hab != "all":
        gdf_test = gdf_test[gdf_test["habitat_id"] == hab]
    
    gdf_test["log_area"] = np.log(gdf_test["area"])
    gdf_test["log_megaplot_area"] = np.log(gdf_test["megaplot_area"])
    gdf_test["log_sr"] = np.log(gdf_test["sr"])
    X_test, _, _, _ = scale_features_targets(gdf_test, predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    ys = np.array([])
    batch_size = 1000
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(X_test), batch_size), desc="Calculating SR and stdSR"):
        X_batch = X_test[i:i + batch_size, :].to(device)
        y_batch = model(X_batch).detach().cpu()
        y_batch = target_scaler.inverse_transform(y_batch).flatten()
        ys = np.concatenate([ys, y_batch])
        
    gdf_test["log_sr_pred"] = ys
    
    gdf_test.dropna(inplace=True)
    r2 = r2_score(gdf_test["log_sr"], gdf_test["log_sr_pred"])
    print(f"R^2 score: {r2}")
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(gdf_test["log_sr"], gdf_test["log_sr_pred"], alpha=0.5)
    # ax.plot([gdf_test["log_sr"].min(), gdf_test["log_sr"].max()], 
    #         [gdf_test["log_sr"].min(), gdf_test["log_sr"].max()], 
    #         'r--', lw=2)
    ax.set_xlabel("True log(SR)")
    ax.set_ylabel("Predicted log(SR)")
    ax.set_title(f"True vs Predicted log(SR) (R^2 score: {r2:.2f})")
    ax.grid(True)
    plt.show()

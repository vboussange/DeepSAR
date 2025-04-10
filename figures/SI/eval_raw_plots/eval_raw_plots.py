# evaluating model trained on the full augmented data on the raw data
# TODO: fix data leak by using models trained on train/test data

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
from src.mlp import load_model_checkpoint
import sys
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/")))
from cross_validate_parallel import Config, compile_training_data

from sklearn.metrics import mean_squared_error
from src.plotting import read_result
from tqdm import tqdm
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # creating X_maps for different resolutions
    seed = 2
    MODEL = "large"
    HASH = "a53390d" 
    path_results = Path(f"../../../scripts/results/cross_validate_parallel_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_cross_validation_{HASH}.pth")    
    
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]
    data = read_result(config.path_augmented_data)
    
    hab = "R2"
    fold=3
    checkpoint = result_modelling[hab]["area+climate"]
    gdf = compile_training_data(data, hab, config)
    gdf_test = gdf[gdf.megaplot_area < 1e4]

    predictors = checkpoint["predictors"]
    test_partition = checkpoint["test_partition"][fold]
    feature_scaler = checkpoint["feature_scaler"][fold]
    target_scaler = checkpoint["target_scaler"][fold]
    gdf_test = gdf_test[gdf_test.partition.isin(test_partition)]
    X_test, y_test, _, _ = scale_features_targets(gdf_test, predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    log_area =  gdf_test["log_area"].values
    
    model_state = checkpoint["model_state_dict"][fold]
    model = load_model_checkpoint(model_state, predictors, layer_sizes=config.layer_sizes)
    with torch.no_grad():
        y = target_scaler.inverse_transform(model(X_test))
        y_test = target_scaler.inverse_transform(y_test)
        mse = mean_squared_error(y, y_test)
        print(f"MSE: {mse:.4f}")
        r2 = r2_score(y_test, y)
        print(f"R²: {r2:.4f}")
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_test, alpha=0.5)
    # ax.plot([gdf_test["log_sr"].min(), gdf_test["log_sr"].max()], 
    #         [gdf_test["log_sr"].min(), gdf_test["log_sr"].max()], 
    #         'r--', lw=2)
    ax.set_xlabel("True log(SR)")
    ax.set_ylabel("Predicted log(SR)")
    ax.set_title(f"True vs predicted log(SR) (R^2 score: {r2:.2f})")
    # Add the x=y line
    lims = [1, 5]
    ax.plot(lims, lims, 'r--', lw=2, label='x = y')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    plt.show()

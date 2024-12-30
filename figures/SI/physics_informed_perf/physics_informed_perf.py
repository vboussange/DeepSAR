""""
Plotting figure 2 'prediction power of climate, area, and both on SR'
# TODO: fix the residual plot
"""
import torch
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from src.plotting import boxplot_bypreds

import scipy.stats as stats

import sys
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/")))
from cross_validate import Config
from src.mlp import load_model_checkpoint
from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data
from src.dataset import scale_features_targets


def evaluate_residuals(gdf, checkpoint, fold, config):
    predictors = checkpoint["predictors"]
    test_idx = np.random.choice(checkpoint["test_idx"][fold], 1000, replace=False)
    feature_scaler = checkpoint["feature_scaler"][fold]
    target_scaler = checkpoint["target_scaler"][fold]
    gdf_test = gdf.loc[test_idx,:]
    X_test, y_test, _, _ = scale_features_targets(gdf_test, predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    log_area =  gdf_test["log_area"].values
    
    model_state = checkpoint["model_state_dict"][fold]
    model = load_model_checkpoint(model_state, predictors, layer_sizes=config.layer_sizes)
    with torch.no_grad():
        y = target_scaler.inverse_transform(model(X_test))
        y_test = target_scaler.inverse_transform(y_test)
        res = y - y_test
        print(f"MSE: {mean_squared_error(y, y_test):.4f}")
    return log_area, res.flatten()

def evaluate_model_all_residuals(gdf, result_modelling, hab, config):
    fold = 5
    result_all = {}
    for scenario in ["area", "climate", "area+climate"]:
        checkpoint = result_modelling[hab][scenario]
        log_area, residuals = evaluate_residuals(gdf, checkpoint, fold, config)
        result_all[scenario] = {}
        result_all[scenario]["residuals"] = residuals
        result_all[scenario]["log_area"] = log_area
    return result_all


if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    seed = 2
    MODEL = "large"
    HASH = "71f9fc7"    
    path_results = Path(f"../../../scripts/results/cross_validate_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    
    result_modelling = torch.load(path_results, map_location="cpu")
        
    for hab in habitats:
        val = result_modelling[hab]
        mse_arr = []
        # removing nan values
        for val2 in val.values():
            mse = val2["test_MSE"]
            if len(mse) > 0:
                mse_arr.append(mse)
        mse = np.stack(mse_arr)
        non_nan_columns = ~np.isnan(mse).any(axis=0)
        matrix_no_nan_columns = mse[:, non_nan_columns]
        for i, val2 in enumerate(val.values()):
            if len(val2["test_MSE"]) > 0:
                val2["test_MSE"] = mse[i,:]
            else:
                val2["test_MSE"] = np.array([])
                
    # Calculating residuals
    hab = "all"
    config = result_modelling["config"]
    
    gdf = load_preprocessed_data(hab, config.hash_data, config.data_seed)    
        
    result_modelling["all"]['area+climate, habitat agnostic'] = {"test_MSE":[]}
    PREDICTORS = [
                #   "area", 
                #   "climate", 
                #   'area+climate, habitat agnostic', 
                  "area+climate", 
                  "area+climate, no physics"
                  ]

    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(result_modelling.keys()))
    # habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["T1", "T3"]
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])
    
    # first axis
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_ylim(0., 0.8)
    boxplot_bypreds(
        result_modelling,
        ax=ax1,
        spread=0.2,
        colormap="Set2",
        legend=True,
        xlab="",
        ylab="MSE",
        yscale="linear",
        yname="test_MSE",
        habitats=habitats,
        predictors=PREDICTORS,
        widths=0.15,
    )

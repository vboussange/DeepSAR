""""
Calculating MSE for the model on raw plots.
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
from cross_validate_parallel import Config, compile_training_data
from src.mlp import load_model_checkpoint
from src.dataset import scale_features_targets
from src.plotting import read_result
from sklearn.metrics import r2_score


def evaluate_residuals(gdf, checkpoint, fold, config):
    predictors = checkpoint["predictors"]
    test_partition = checkpoint["test_partition"][fold]
    feature_scaler = checkpoint["feature_scaler"][fold]
    target_scaler = checkpoint["target_scaler"][fold]
    gdf_test = gdf[gdf.partition.isin(test_partition)]
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
    return mse, r2

def evaluate_model_all_residuals(gdf, checkpoint, config):
    res = {}
    for scenario in ["area", "climate", "area+climate"]:
        print(f"Scenario: {scenario}")
        res[scenario] = {"mse": [], "r2":[]}
        for fold in range(len(checkpoint[scenario]["test_MSE"])):
            print(f"Fold: {fold}")
            scen = checkpoint[scenario]
            mse, r2 = evaluate_residuals(gdf, scen, fold, config)
            res[scenario]["mse"].append(mse)
            res[scenario]["r2"].append(r2)
            
        res[scenario]["mse"] = np.array(res[scenario]["mse"])
        res[scenario]["r2"] = np.array(res[scenario]["r2"])
            
    # Convert results directly to numpy arrays
    return res


if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["all"]
    seed = 2
    MODEL = "large"
    HASH = "a53390d" 
    path_results = Path(f"../../../scripts/results/cross_validate_parallel_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_cross_validation_{HASH}.pth")    
    
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]
    data = read_result(config.path_augmented_data)

    # Calculating MSE on subset of the data
    r2_dict = {}
    for hab in habitats:
        checkpoint = result_modelling[hab]
        gdf = compile_training_data(data, hab, config)
        gdf_test = gdf[gdf.megaplot_area < 1e4]
        # gdf_test = gdf
        post_crossval = evaluate_model_all_residuals(gdf_test, checkpoint, config)
        r2_dict[hab] = post_crossval
    
    metric = "r2"
    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(post_crossval.keys()))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])
    
    # first axis
    ax1 = fig.add_subplot(gs[0, :])
    # ax1.set_ylim(0., 0.8)
    boxplot_bypreds(
        r2_dict,
        ax=ax1,
        spread=0.6,
        colormap="Set2",
        legend=True,
        xlab="",
        ylab=metric.upper(),
        yscale="linear",
        yname=metric,
        habitats=habitats,
        predictors=["area", "climate", "area+climate"],
        widths=0.1,
    )
    if metric == "mse":
        label_l1 = ["Forests", "Grasslands", "Wetlands", "Shrublands"]
        for i,x in enumerate(np.arange(1, len(habitats), step=2)):
            ax1.text(x+0.5, -0.01, label_l1[i], ha='center', va='bottom', fontsize=10, color='black')
    else:
        ax1.set_ylim(-0.2, 0.5)
    fig.savefig(Path(__file__).stem + "_" + metric + ".pdf", 
                transparent=True, 
                dpi=300,
                bbox_inches='tight')
    
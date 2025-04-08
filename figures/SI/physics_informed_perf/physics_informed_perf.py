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
from cross_validate_parallel import Config, compile_training_data
from src.mlp import load_model_checkpoint
from src.dataset import scale_features_targets
from src.plotting import read_result

if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    seed = 2
    MODEL = "large"
    HASH = "a53390d"    
    path_results = Path(f"../../../scripts/results/cross_validate_parallel_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_cross_validation_{HASH}.pth")    
    
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]
        
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
    
    data = read_result(config.path_augmented_data)
    gdf = compile_training_data(data, hab, config)    
        
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
    fig.savefig(Path(__file__).stem + ".pdf", dpi=300, transparent=True, bbox_inches='tight')


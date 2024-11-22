"""
Plotting loss of against iterations for training and validation sets for all habitats.

Using ensemble methods.
"""
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
import sys
import torch
from pathlib import Path
PATH_MLP_TRAINING = Path("../../../scripts/")
sys.path.append(str(Path(__file__).parent / PATH_MLP_TRAINING))
from scripts.train import Config

if __name__ == "__main__":
    # creating X_maps for different resolutions
    # creating X_maps for different resolutions
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"
    ncells_ref = 20 # used for coarsening
    
    checkpoint_path = PATH_MLP_TRAINING / Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth") 
    result_all = torch.load(checkpoint_path, map_location="cpu")    
    config = result_all["config"]

    fig, axs = plt.subplots(3,3, figsize=(9,9))
    for i, hab in enumerate(config.habitats):
        ax = axs.flatten()[i]
        ax.set_yscale("log")
        ax.set_title(hab,  fontweight="bold")
        ax.set_ylim(5e-2, 1e3)
        for j, metrics in enumerate(result_all[hab]["ensemble_metrics"]):
            train_MSE = metrics["train_MSE"]
            val_MSE = metrics["val_MSE"]
            label = "Training loss" if j==0 else ""
            ax.plot(train_MSE, c = "tab:blue", label = label, alpha=0.5)
            label = "Validation loss" if j==0 else ""
            ax.plot(val_MSE, c = "tab:red", label = label, alpha=0.5,)
            
    axs.flatten()[0].legend()
    fig.savefig("convergence.pdf", transparent=True, dpi=300)
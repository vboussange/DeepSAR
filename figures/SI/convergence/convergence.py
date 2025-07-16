"""
Plotting loss of against iterations for training and validation sets.

Using ensemble methods.
"""
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from train import Config


if __name__ == "__main__":
    path_results = Path("../../../scripts/results/train/checkpoint_MSEfit_lowlr_nosmallmegaplots2_basearch6_0b85791.pth")
    
    # Load model results
    result_modelling = torch.load(path_results, map_location="cpu")
    config = result_modelling["config"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale("log")
    # ax.set_ylim(5e-2, 1e3)

    for j, metrics in enumerate(result_modelling["logs"]):
        train_MSE = metrics["train_losses"]
        val_MSE = metrics["val_losses"]
        label = "Training loss" if j==0 else ""
        ax.plot(train_MSE, c = "#4cc9f0", label = label)
        label = "Validation loss" if j==0 else ""
        ax.plot(val_MSE, c = "#f72585", label = label)
            
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle='-', linewidth=0.5)
    ax.legend()
    fig.savefig("convergence.pdf", transparent=True, dpi=300)
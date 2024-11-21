"""
!!! Problem of evaluation of out[:, 0] and out[:, 1]. When evaluated in global
scope, works, but does not work when evaluated in local scope. This seems like a
bug from skorch or pytorch, as depending on the number of layers, we do not get
the same behavior
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


from src.plotting import COLOR_PALETTE
from src.utils import save_to_pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../scripts/NNSAR/")))
import SAR_fit_simple
from SAR_fit_simple import CONFIG

if __name__ == "__main__":
    result_path = Path(__file__).parent / Path("../../scripts/NNSAR/SAR_fit_simple.pkl")
    
    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]
        
    climate_predictors  = results_fit_split["climate_predictors"]
    hab = "all"
    gdf = results_fit_split[hab]["gdf"]
    gdf_test = gdf[~results_fit_split[hab]["train_idx"]]
    reg = results_fit_split[hab]["reg"]
    feature_scaler, target_scaler = results_fit_split[hab]["scalers"]
    X, y, _, _ = SAR_fit_simple.get_Xy_scaled(gdf_test, climate_predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)
    
    fig, ax = plt.subplots()
    log_sr_pred = target_scaler.inverse_transform(reg.predict(X["log_area"])).flatten()
    ax.scatter(gdf_test["log_area"], gdf_test["log_sr"], alpha=0.5, c="tab:red", label="SR true")
    ax.scatter(gdf_test["log_area"], log_sr_pred, alpha=0.5, c="tab:blue", label="SR predicted")
    # ax.scatter(gdf_test["log_sr"], log_sr_pred, alpha=0.5, c="tab:blue")
    # ax.plot([-1 ,8], [-1, 8], c="tab:grey")
    ax.legend()
    ax.set_title(f"SAR regression, MSE={mean_squared_error(gdf_test["log_sr"], log_sr_pred)}")

    # Plot as horizontal bars the standardized effects of climate predictors on
    # c and z, where respective effect on c and z are plotted on the same ax
    # with different colors
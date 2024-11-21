"""
Plotting c vs z and MSE for SAR4
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
from src.NNSAR import NNSAR2
from src.utils import save_to_pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../scripts/NNSAR/")))
import NNSAR4_fit_simple
from NNSAR4_fit_simple import CONFIG

def calculate_sr_logc_z(reg, gdf, X):
    reg = reg.set_params(device="cpu")
    # reg.module_ = reg.module_.to("cpu")

    neural_net = reg.module_.nn.to(torch.device("cpu"))
    
        
    with torch.no_grad():
        out = neural_net(torch.tensor(X["env_pred"])).numpy()
        logc = out[:,0]
        z = out[:, 1] * (out[:, 1] > 0)
        # print(torch.tensor(X["env_pred"]).to(CONFIG["torch_params"]["device"]))
        # print(list(reg.module_.parameters()))
        # print(reg.predict(SliceDict(**X)))
        logc = target_scaler.scale_[0] * (logc - z * feature_scaler.mean_[0] / feature_scaler.scale_[0]) + target_scaler.mean_[0]
        z = target_scaler.scale_[0] * z / feature_scaler.scale_[0]
        
        y_indir_transform = logc + z * gdf["log_area"]
        y_dir_transform = target_scaler.inverse_transform(reg.predict(SliceDict(**X))).flatten()
        assert np.allclose(y_indir_transform, y_dir_transform, atol=1e-5)
    return y_dir_transform, logc, z

if __name__ == "__main__":
    result_path = Path(__file__).parent / Path("../../scripts/NNSAR/NNSAR4_fit_simple.pkl")
    
    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]
        
    climate_predictors  = results_fit_split["climate_predictors"]
    hab = "all"
    gdf = results_fit_split[hab]["gdf"]
    gdf_test = gdf[~results_fit_split[hab]["train_idx"]]
    reg = results_fit_split[hab]["reg"]
    feature_scaler, target_scaler = results_fit_split[hab]["scalers"]
    X, y, _, _ = NNSAR4_fit_simple.get_Xy_scaled(gdf_test, climate_predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)


    # log_sr_pred, logc, z = calculate_sr_logc_z(reg, gdf_test, X)
    
    # gdf_test["log_sr_pred"] = log_sr_pred
    # gdf_test["logc"] = logc
    # gdf_test["z"] = z
    # gdf_test.to_csv(str(Path(__file__).parent / Path(__file__).stem) + ".csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    
    log_sr_pred = target_scaler.inverse_transform(reg.predict(SliceDict(**X))).flatten()

    # ax = axes[0]
    # ax.scatter(logc, z, alpha=0.5, c=gdf_test.log_area)
    # ax.set_xlabel('logc')
    # ax.set_ylabel('z')
    
    ax = axes[1]
    ax.scatter(gdf_test["log_area"], gdf_test["log_sr"], alpha=0.5, c="tab:red", label="SR true")
    ax.scatter(gdf_test["log_area"], log_sr_pred, alpha=0.5, c="tab:blue", label="SR predicted")
    # ax.scatter(gdf_test["log_sr"], log_sr_pred, alpha=0.5, c="tab:blue")
    # ax.plot([-1 ,8], [-1, 8], c="tab:grey")
    ax.legend()
    ax.set_title(f"NNSAR regression, MSE={mean_squared_error(gdf_test["log_sr"], log_sr_pred)}")

    # Plot as horizontal bars the standardized effects of climate predictors on
    # c and z, where respective effect on c and z are plotted on the same ax
    # with different colors
"""
Performing a simple training run with train test split for each habitat

!!! this scrip is similar to NNSAR_fit_simple - it should ideally use its outputs to have only plot_SAR function and output
"""
import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import LRScheduler,EarlyStopping

from sklearn.model_selection import cross_validate, GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from src.plotting import COLOR_PALETTE
from src.NNSAR import NNSAR2
from src.utils import save_to_pickle
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../figures/figure_2/")))
import NNSAR_figure_2


CONFIG = {
    "torch_params": {
        "optimizer":optim.AdamW,
        "lr":1e-3,
        "batch_size":256,
        "max_epochs":300,
        # "callbacks":[("early_stopping", EarlyStopping(monitor='train_loss', patience=20))],
        "callbacks":[("lr_scheduler", LRScheduler(policy = "ReduceLROnPlateau", monitor='train_loss', patience=20, factor=0.5))],
        # "optimizer__weight_decay":1e-4,
        "train_split": None,
        "device": torch.device("cuda")
    },
}

def get_Xy_scaled(gdf, predictors, feature_scaler=None, target_scaler=None):
    
    features = gdf[predictors].values.astype(np.float32)
    target = gdf["log_sr"].values.astype(np.float32)
    
    if feature_scaler == None:
        # Standardize features and target
        feature_scaler = StandardScaler()
        feature_scaler.fit(features)

        target_scaler = StandardScaler()
        target_scaler.fit(target.reshape(-1,1))

    features_scaled = feature_scaler.transform(features)
    target_scaled = target_scaler.transform(target.reshape(-1, 1))

    # discarding small areas
    # idx = features_scaled[:, -1] > pd.DataFrame(features_scaled[:, -1]).quantile(0.25)[0]

    # Prepare training data
    X_train = {
        "env_pred": features_scaled[:, 1:],
        "log_area": features_scaled[:, 0].reshape(-1, 1)
    }
    
    return X_train, target_scaled, feature_scaler, target_scaler

def evaluate_model_per_hab(dataset, predictors, torch_params, habitats):
    """
    Evaluating SBCV for different combinations of features, for each habitat.
    """
    result_all = {}
    climate_predictors = dataset.aggregate_labels
    gdf_full = dataset.gdf.sample(frac=1, random_state=42).reset_index(drop=True)

    train_cv_partition_idx, test_cv_partition_idx = train_test_split(
        gdf_full.partition.unique(), test_size=0.2, random_state=42
    )
    
    for hab in habitats:
        print("Training", hab)
        
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        gdf_train = gdf[gdf.partition.isin(train_cv_partition_idx)]
        gdf_test = gdf[gdf.partition.isin(test_cv_partition_idx)]

        X_train, y_train, feature_scaler, target_scaler = get_Xy_scaled(gdf_train, climate_predictors)
        X_test, y_test, _, _ = get_Xy_scaled(gdf_test, climate_predictors, feature_scaler, target_scaler)

        
        print("Evaluation of model with area and climate")
        reg = NeuralNetRegressor(module=NNSAR2,
                                module__input_dim=X_train["env_pred"].shape[1],
                                **CONFIG["torch_params"])
        
        reg.fit(SliceDict(**X_train), y_train)
        y_pred = reg.predict(SliceDict(**X_test))
        mse = mean_squared_error(y_test, y_pred)

        result_all[hab] = {}
        result_all[hab]["reg"] = reg
        result_all[hab]["X_test"] = X_test
        result_all[hab]["y_test"] = y_test
        result_all[hab]["y_pred"] = y_pred
        result_all[hab]["mse"] = mse

    return result_all


def plot_SAR(dataset, result_modelling, habitats, color_palette, predictors):
    fig, axs = plt.subplots(2, 5, figsize=(8, 4), sharex=True, sharey=True)
    gdf_full = dataset.gdf

    for i, hab in enumerate(habitats):
        reg = result_modelling[hab]["reg"]
        
        gdf_hab = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        
        X, y, _, _ = get_Xy_scaled(gdf_hab, predictors, *result_modelling[hab]["scalers"])
        
        constant_preds = np.median(X["env_pred"], axis=0)
        mask_str = ["std" in pred for pred in predictors[1:]]
        
        XX = {}
        XX["log_area"] = np.linspace(X["log_area"].min(), X["log_area"].max(), 100).reshape(-1,1)

        # plotting for non corrected SAR
        constant_preds[mask_str] = np.quantile(X["env_pred"][:,mask_str], 0.1, axis=0)
        XX["env_pred"] = np.tile(constant_preds, (100, 1))
        ypred = reg.predict(SliceDict(**XX))
        
        
        ax = axs.flatten()[i]
        ax.set_title(hab, fontweight="bold")
        # gdf_sub = gdf_hab.sample(n=5000)
        ax.scatter(np.exp(X["log_area"]), np.exp(y), s = 0.1, alpha=0.1, color=color_palette[i])
        ax.plot(np.exp(XX["log_area"]), np.exp(ypred), color="tab:blue", label=hab, alpha=0.9)

        # plotting for corrected SAR
        constant_preds[mask_str] = np.quantile(X["env_pred"][:,mask_str], 0.9, axis=0)
        XX["env_pred"] = np.tile(constant_preds, (100, 1))
        ypred = reg.predict(SliceDict(**XX))
        ax.plot(np.exp(XX["log_area"]), np.exp(ypred), color="tab:red", alpha=0.9, linestyle="--")
        
    return fig, axs


if __name__ == "__main__":
    # habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3", ]
    habitats = ["T1"]

    result_path = Path(str(Path(__file__).stem) + ".pkl")
    dataset = NNSAR_figure_2.process_results()
    predictors = ["log_area"] + dataset.aggregate_labels
    
    if True:

        start_time = time.time()
        result_modelling = evaluate_model_per_hab(
            dataset, predictors, CONFIG["torch_params"], habitats
        )
        end_time = time.time()

        # Calculate the elapsed time
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")

        save_to_pickle(result_path ,result_modelling=result_modelling)
    
    else:
        with open(result_path, 'rb') as pickle_file:
            result_modelling = pickle.load(pickle_file)["result_modelling"]
        
        
    fig, axs = plot_SAR(dataset, 
            result_modelling, 
            habitats, 
            COLOR_PALETTE, 
            predictors)
    
    for ax in axs.flatten():
        ax.set_yscale("log")
        ax.set_xscale("log")

    fig.supylabel("SR")
    fig.supxlabel("$A$")
    
    plain_line = plt.Line2D([0,1], [0,0], color='tab:grey', label = "low env. het.")
    dashed_line = plt.Line2D([0,1], [0,0], color='tab:grey', label = "high env. het.", linestyle="--")
    # trained_data = plt.Line2D([0], [0], marker='s', color='w', label='test data',
    # # Add the legend
    axs[-1,-1].legend(handles=[dashed_line,plain_line ], loc='center', bbox_to_anchor=(0.4, 0.4))
    fig.tight_layout()
    axs.flatten()[-1].axis("off")
    fig.savefig(str(Path(__file__).parent / Path(__file__).stem) + ".png", dpi=300)


""""
metal: 18.95 sec
cpu: 7.8 sec
cuda: 6sec
"""
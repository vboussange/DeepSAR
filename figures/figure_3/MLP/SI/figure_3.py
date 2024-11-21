"""
TODO: 
* Same as ../figure_3.py script
* Here we make the same figure for other habitats
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec

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

import shap

from src.plotting import COLOR_PALETTE
from src.MLP import MLP, scale_feature_tensor, get_gradient
from src.utils import save_to_pickle

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / Path("../../../../scripts/MLP3/simple_train_test_split")))
from MLP_fit_torch import process_results, preprocess_gdf_hab

from captum.attr import ShapleyValueSampling


def scale_features_targets(gdf, predictors, feature_scaler=None, target_scaler=None):
    features = gdf[predictors].values.astype(np.float32)
    target = gdf["log_sr"].values.astype(np.float32)

    if feature_scaler is None:
        feature_scaler, target_scaler = StandardScaler(), StandardScaler()
        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def get_df_shap_val(model, results_fit_split, gdf):
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    gdf = gdf.sample(10000)
    features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
    X = scale_feature_tensor(features, feature_scaler)
    def forward_fn(X):
        with torch.no_grad():
            return model(X)
    explainer = ShapleyValueSampling(forward_fn)
    shap_val = explainer.attribute(X)
    shap_abs = np.abs(shap_val.numpy())

    df_shap_values = pd.DataFrame(shap_abs, columns=predictors)
    df_shap_values = df_shap_values / df_shap_values.max().max()
    df_shap_values["log_sr"] = gdf["log_sr"].values
    return df_shap_values


def update_area(X_map, log_area):
    res_sr_map = np.sqrt(np.exp(log_area))
    X_map["max_lon_diff"] = res_sr_map
    X_map["max_lat_diff"] = res_sr_map
    X_map["log_area"] = log_area
    
def plot_SAR(ax, model, results_fit_split, gdf):
    # fig, ax = plt.subplots(1)
    
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]

    gdf_plot = gdf[gdf.num_plots == 1]
    gdf_megaplot = gdf[gdf.num_plots > 1]

    max_log_area_train = gdf["log_area"].max()
    npoints = 150
    log_area = np.linspace(gdf["log_area"].min(), 1.5 * max_log_area_train, npoints)
    log_area_train_idx = log_area < max_log_area_train

    mydict = {"plot": {"gdf": gdf_plot, "linestyle": "-", "c": "tab:blue", "label":"No env. het."}, 
              "megaplot": {"gdf": gdf_megaplot, "linestyle": "-", "c": "tab:red", "label":"Env. het."}}
    
    for plottype in mydict:
        gdf = mydict[plottype]["gdf"]
        ys = []
        for i in range(100):
            plot_row = gdf.sample(1)
            _gdf = pd.concat([plot_row]*npoints, ignore_index=True)
            update_area(_gdf, log_area)
            XX, y, _, _ = scale_features_targets(_gdf, predictors, feature_scaler=feature_scaler, target_scaler=target_scaler)
            
            with torch.no_grad():
                y_pred = model(XX)
                y_pred = target_scaler.inverse_transform(y_pred.numpy()).flatten()
            # y_pred = pd.Series(reg.predict(XX)).rolling(window=30).mean().values
            ys.append(y_pred)
            ax.plot(np.exp(log_area)[log_area_train_idx], 
                    np.exp(y_pred)[log_area_train_idx], 
                    c=mydict[plottype]["c"], 
                    linestyle="-",
                    alpha=0.1)
            ax.plot(np.exp(log_area)[~log_area_train_idx], 
                    np.exp(y_pred)[~log_area_train_idx], 
                    c=mydict[plottype]["c"], 
                    linestyle="--",
                    alpha=0.1)

        # print(ys)
        stacked_ys = np.stack(ys, axis=0)
        median_ys = np.median(stacked_ys, axis=0)
        ax.plot(np.exp(log_area)[log_area_train_idx], 
                np.exp(median_ys)[log_area_train_idx], 
                c=mydict[plottype]["c"], 
                linestyle=mydict[plottype]["linestyle"],
                alpha=1.,
                label=mydict[plottype]["label"],
                linewidth=2.)
        ax.plot(np.exp(log_area)[~log_area_train_idx], 
                    np.exp(median_ys)[~log_area_train_idx], 
                    c=mydict[plottype]["c"], 
                    linestyle="--",
                    alpha=1.,
                    label=mydict[plottype]["label"],
                    linewidth=2.)
        ax.set_yscale("log")
        ax.set_xscale("log")
        # ax.set_xlabel("Area (m2)")
        ax.set_ylabel("Species Richness")
        
def scale_feature_tensor(x, scaler):
    assert len(x.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32).reshape(1, -1)
    features_scaled = (x - mean_tensor) / scale_tensor
    return features_scaled

def inverse_transform_scale_feature_tensor(y, scaler):
    assert len(y.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32).reshape(1, -1)
    invy = y * scale_tensor + mean_tensor
    return invy

def get_SR_dSR(model, gdf, predictors, feature_scaler, target_scaler):
        features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
        X = scale_feature_tensor(features, feature_scaler)
        y = model(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        dlogSR_dlogA = get_gradient(log_SR, features).detach().numpy()[:, 0]
        log_SR = log_SR.detach().numpy()
        return log_SR, dlogSR_dlogA

def plot_dSAR(ax, model, results_fit_split, gdf):
    # fig, ax = plt.subplots(1)
    
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]

    gdf_plot = gdf[gdf.num_plots == 1]
    gdf_megaplot = gdf[gdf.num_plots > 1]

    max_log_area_train = gdf["log_area"].max()
    npoints = 150
    log_area = np.linspace(gdf["log_area"].min(), 1.5 * max_log_area_train, npoints)
    log_area_train_idx = log_area < max_log_area_train

    mydict = {"plot": {"gdf": gdf_plot, "linestyle": "-", "c": "tab:blue", "label":"No env. het."}, 
              "megaplot": {"gdf": gdf_megaplot, "linestyle": "-", "c": "tab:red", "label":"Env. het."}}
    
    for plottype in mydict:
        gdf = mydict[plottype]["gdf"]
        ys = []
        for i in range(100):
            plot_row = gdf.sample(1)
            _gdf = pd.concat([plot_row]*npoints, ignore_index=True)
            _gdf["log_area"] = log_area
            log_SR, dlogSR_dlogA = get_SR_dSR(model, _gdf, predictors, feature_scaler, target_scaler)
            # applying moving window
            # dlogSR_dlogA = pd.Series(dlogSR_dlogA).rolling(window=5).mean().values

            ys.append(dlogSR_dlogA)
            # ax.plot(np.exp(log_area)[log_area_train_idx], 
            #         np.exp(dlogSR_dlogA)[log_area_train_idx], 
            #         c=mydict[plottype]["c"], 
            #         linestyle="-",
            #         alpha=0.1)
            # ax.plot(np.exp(log_area)[~log_area_train_idx], 
            #         np.exp(dlogSR_dlogA)[~log_area_train_idx], 
            #         c=mydict[plottype]["c"], 
            #         linestyle="--",
            #         alpha=0.1)

        # print(ys)
        stacked_ys = np.stack(ys, axis=0)
        median_ys = pd.Series(np.median(stacked_ys, axis=0)).rolling(window=5).mean().values
        quantile_1 = pd.Series(np.quantile(stacked_ys, 0.25, axis=0)).rolling(window=5).mean().values
        quantile_3 = pd.Series(np.quantile(stacked_ys, 0.75, axis=0)).rolling(window=5).mean().values

        ax.plot(np.exp(log_area)[log_area_train_idx], 
                median_ys[log_area_train_idx], 
                c=mydict[plottype]["c"], 
                linestyle=mydict[plottype]["linestyle"],
                alpha=1.,
                label=mydict[plottype]["label"],
                linewidth=2.)
        ax.fill_between(np.exp(log_area), 
            quantile_1, 
            quantile_3, 
            color=mydict[plottype]["c"], 
            alpha = 0.4,
            linewidth=0.3)
        ax.plot(np.exp(log_area)[~log_area_train_idx], 
                    median_ys[~log_area_train_idx], 
                    c=mydict[plottype]["c"], 
                    linestyle="--",
                    alpha=1.,
                    label=mydict[plottype]["label"],
                    linewidth=2.)
        # ax.set_xlabel("Area (m2)")
        ax.set_ylabel("dlogSR/dlogA")

def load_model_checkpoint(checkpoint_path, device):
        """Load the model and scalers from the saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load the model architecture
        predictors = checkpoint['predictors']
        model = MLP(len(predictors)).to(device)
        
        # Load model weights and other components
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    
if __name__ == "__main__":
    hab = "T1"
    checkpoint_path = f"../../../../scripts/MLP3/simple_train_test_split/results/MLP_fit_torch_{hab}_dSRdA_weight_1e+01_seed_1/checkpoint_best.pth"
    model, results_fit_split = load_model_checkpoint(checkpoint_path, "cpu")
    dataset = process_results()
    gdf = preprocess_gdf_hab(dataset.gdf, hab)
    
    print(f"MSE val: {results_fit_split["best_validation_loss"]}")

    fig = plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.8)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom left
    ax3 = fig.add_subplot(gs[:, 1])   
    
    # # SAR
    plot_SAR(ax1, model, results_fit_split, gdf)
    # ax.set_ylim(1, 1e2)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend()

    # # dSAR
    plot_dSAR(ax2, model, results_fit_split, gdf)
    ax2.set_xscale("log")
    
    # SHAP  
    df_shap_values = get_df_shap_val(model, results_fit_split, gdf)
    sns_kwargs = {"alpha": 0.5, "palette": "icefire", "legend": False, "s": 2.}
    _df_shap = df_shap_values.drop("log_area", axis=1)
    df_melted = pd.melt(_df_shap, id_vars=["log_sr"], var_name="Feature", value_name="SHAP values")
    sns.stripplot(data=df_melted, 
                    x="SHAP values", 
                    y="Feature", 
                    hue = "log_sr",
                    ax=ax3, 
                    **sns_kwargs
                    )
    ax3.set_ylabel("")
    
    # Separate the columns into 'std_' variables and others
    std_columns = [col for col in _df_shap.columns if col.startswith("std_")]
    mean_columns = [col for col in _df_shap.columns if col not in std_columns and col != "log_sr"]

    # Calculate the sum for std_ and non-std variables
    sum_std_shap = df_shap_values[std_columns].median().median()  # Sum of all std_ columns' SHAP values
    sum_non_std_shap = df_shap_values[mean_columns].median().median()  # Sum of all non-std SHAP values

    # Step 4: Overlay bar plots on the same axes
    bar_width = 7  # You can adjust this to match your stripplot width

    # Get the current y-ticks (features) in the stripplot
    y_ticks = ax3.get_yticks()
    std_y_pos = np.mean(y_ticks[:len(std_columns)])  # Middle of the std_ variables
    non_std_y_pos = np.mean(y_ticks[len(std_columns):])  # Middle of the non-std variables

    # Plot the bars
    ax3.barh(std_y_pos, sum_std_shap, height=bar_width, color='tab:blue', alpha=0.6, label='std_ sum')
    ax3.barh(non_std_y_pos, sum_non_std_shap, height=bar_width, color='tab:red', alpha=0.6, label='non-std sum')
    fig.tight_layout()
    fig.savefig(f"figure_3_{hab}.png", transparent=True, dpi=300)

    # ax.set_xscale("log")
    # ax.set_xlim(1e-3, 1e0)
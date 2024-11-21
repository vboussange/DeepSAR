"""
TODO: 
* make Shapley values more robust
* Refactor to simplify the code
* Make sure that the model converges to robust results
    - It seems that there are two types of convergence: one where dlogSR/dlogA is = 0.3 at large A, and the other where it convergence to 0. This could be due to a lack of constraints
* Add variations as a fill_between plot
* Evaluate model on trained data, for now we do not make a distinction between the two.

Using ensemble model
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns

import numpy as np
import pandas as pd
import torch


from sklearn.preprocessing import StandardScaler

from src.plotting import COLOR_PALETTE
from src.mlp import MLP, scale_feature_tensor, get_gradient
from src.ensemble_model import initialize_ensemble_model

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / Path("../../../../scripts/MLP3")))
from MLP_fit_torch_all_habs_ensemble import *
sys.path.append(str(Path(__file__).parent / Path("../../../../scripts/eva_processing/")))
from preprocess_eva_CHELSA_EUNIS_plot_megaplot_ratio_1_1 import load_preprocessed_data

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
        X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
        y = model(X).to("cpu")
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        dlogSR_dlogA = get_gradient(log_SR, features).detach().numpy()[:, 0]
        log_SR = log_SR.detach().numpy().flatten()
        return log_SR, dlogSR_dlogA

def plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, dict_styles):
    # fig, ax = plt.subplots(1)
    
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]

    gdf_plot = gdf[gdf.num_plots == 1]
    gdf_megaplot = gdf[gdf.num_plots > 1]

    max_log_area_train = gdf["log_area"].max()
    npoints = 150
    log_area = np.linspace(np.log(1e1), np.log(1e9), npoints)
    log_area_train_idx = log_area < max_log_area_train

    dict_styles["plot"]["gdf"] = gdf_plot
    dict_styles["megaplot"]["gdf"] = gdf_megaplot

    for plottype in dict_styles:
        gdf = dict_styles[plottype]["gdf"]
        ys_SAR = []
        ys_dSAR = []
        for i in range(100):
            plot_row = gdf.sample(1)
            _gdf = pd.concat([plot_row]*npoints, ignore_index=True)
            _gdf["log_area"] = log_area
            log_SR, dlogSR_dlogA = get_SR_dSR(model, _gdf, predictors, feature_scaler, target_scaler)
            # applying moving window
            # dlogSR_dlogA = pd.Series(dlogSR_dlogA).rolling(window=5).mean().values

            ys_SAR.append(np.exp(log_SR))
            ys_dSAR.append(dlogSR_dlogA)

        # print(ys)
        for ax, ys in zip((ax_SAR, ax_dSAR), [ys_SAR, ys_dSAR]):
            stacked_ys = np.stack(ys, axis=0)
            median_ys = pd.Series(np.median(stacked_ys, axis=0)).rolling(window=5).mean().values
            quantile_1 = pd.Series(np.quantile(stacked_ys, 0.25, axis=0)).rolling(window=5).mean().values
            quantile_3 = pd.Series(np.quantile(stacked_ys, 0.75, axis=0)).rolling(window=5).mean().values

            ax.plot(np.exp(log_area)[log_area_train_idx], 
                    median_ys[log_area_train_idx], 
                    c=dict_styles[plottype]["c"], 
                    linestyle=dict_styles[plottype]["linestyle"],
                    alpha=1.,
                    label=dict_styles[plottype]["label"],
                    linewidth=2.)
            ax.fill_between(np.exp(log_area), 
                quantile_1, 
                quantile_3, 
                color=dict_styles[plottype]["c"], 
                alpha = 0.4,
                linewidth=0.3)
            ax.plot(np.exp(log_area)[~log_area_train_idx], 
                        median_ys[~log_area_train_idx], 
                        c=dict_styles[plottype]["c"], 
                        linestyle="--",
                        alpha=1.,
                        linewidth=2.)
            # ax.set_xlabel("Area (m2)")
            # ax.set_ylabel(label)
            ax.set_xscale("log")
    ax_SAR.set_yscale("log")
    ax_dSAR.set_ylim(0, 0.7)

    return ax_SAR, ax_dSAR

def load_model(result, config, device):
        """Load the model and scalers from the saved checkpoint."""
        
        # Load the model architecture
        predictors = result['predictors']
        model = initialize_ensemble_model(config.n_ensembles, len(predictors)).to(device)
        
        # Load model weights and other components
        model.load_state_dict(result['ensemble_model_state_dict'])
        model.eval()
        return model
    
if __name__ == "__main__":
    seed = 2
    ncells_ref = 20 # used for coarsening
    MODEL = "large"
    HASH = "71f9fc7"
    
    
    checkpoint_path = f"../../../../scripts/MLP3/results/MLP_fit_torch_all_habs_ensemble_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth"
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")
    config = results_fit_split_all["config"]


    fig_SAR, axs_SAR = plt.subplots(2, 5, figsize=(8, 4), sharex=True, sharey=True)
    fig_dSAR, axs_dSAR = plt.subplots(2, 5, figsize=(8, 4), sharex=True, sharey=True)
    habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3", ]
    dict_styles = {"plot": {"linestyle": "-", "c": "tab:blue", "label":"No env. het."}, 
                    "megaplot": {"linestyle": "-", "c": "tab:red", "label":"Env. het."}}
    
    for i, hab in enumerate(habitats):
        print(hab)
        results_fit_split = results_fit_split_all[hab]
        model = load_model(results_fit_split, config, "cuda")
        print(f"MSE val: {results_fit_split["best_validation_loss"]}")
        gdf = load_preprocessed_data(hab, config.hash, config.data_seed)
        gdf_train = gdf.loc#[results_fit_split["train_idx"]] #todo: to fix
        ax_SAR = axs_SAR.flatten()[i]
        ax_dSAR = axs_dSAR.flatten()[i]
        plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, dict_styles)
        for ax in [ax_SAR, ax_dSAR]:
            if hab == "all":
                hab = "habitat agnostic"
            ax.set_title(hab, fontweight="bold")
            
            
    legs = [plt.Line2D([0,1], [0,0], color=e["c"], label = e["label"]) for e in dict_styles.values()]

    # trained_data = plt.Line2D([0], [0], marker='s', color='w', label='test data',
    # # Add the legend
        
    for fig, axs, label in zip([fig_SAR, fig_dSAR], (axs_SAR, axs_dSAR), ("SR", "dlogSR_dlogA")):
        fig.supylabel(label)
        fig.supxlabel("Area (m2)")
        fig.tight_layout()
        axs.flatten()[-1].axis("off")
        axs.flatten()[-1].legend(handles=legs, loc='center', bbox_to_anchor=(0.4, 0.4))
        fig.savefig(f"figure_3_{label}.png", dpi=300, transparent=True)


"""
Here we plot corrected and not corrected SAR and dSAR.
TODO: since heterogeneity does not seem to be a big player, we may just remove the colors and just talk about the variability of the SAR.
"""
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient
from src.plotting import read_result
from src.ensemble_model import initialize_ensemble_model
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).parent / Path("../../scripts/")))
from train import Config, compile_training_data


def get_SR_dSR(model, gdf, predictors, feature_scaler, target_scaler):
    features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
    X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
    y = model(X).to("cpu")
    log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
    dlogSR_dlogA = get_gradient(log_SR, features).detach().numpy()[:, 0]
    log_SR = log_SR.detach().numpy().flatten()
    return log_SR, dlogSR_dlogA
    

def plot_SAR_dSAR_combined(ax, model, results_fit_split, gdf):    
    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    pca = PCA(n_components=1)
    gdf["pca_component"] = pca.fit_transform(gdf[predictors])
    
    # Stratified sampling along the first PCA component
    gdf['pca_component_bins'] = pd.cut(gdf['pca_component'], bins=100, labels=False)
    
    max_log_area_train = gdf["log_area"].max()
    min_log_area_train = gdf["log_area"].min()
    npoints = 150
    log_area = np.linspace(min_log_area_train, max_log_area_train, npoints)

    ys_SAR = []
    ys_dSAR = []
    for _, _gdf in gdf.groupby('pca_component_bins'):
        plot_row = _gdf.sample(1)
        _gdf = pd.concat([plot_row]*npoints, ignore_index=True)
        _gdf["log_area"] = log_area
        log_SR, dlogSR_dlogA = get_SR_dSR(model, _gdf, predictors, feature_scaler, target_scaler)

        ys_SAR.append(np.exp(log_SR))
        ys_dSAR.append(dlogSR_dlogA)

    stacked_ys_SAR = np.stack(ys_SAR, axis=0)
    stacked_ys_dSAR = np.stack(ys_dSAR, axis=0)
    
    median_SAR = pd.Series(np.median(stacked_ys_SAR, axis=0)).rolling(window=5).mean().values
    q1_SAR = pd.Series(np.quantile(stacked_ys_SAR, 0.25, axis=0)).rolling(window=5).mean().values
    q3_SAR = pd.Series(np.quantile(stacked_ys_SAR, 0.75, axis=0)).rolling(window=5).mean().values
    
    median_dSAR = pd.Series(np.median(stacked_ys_dSAR, axis=0)).rolling(window=5).mean().values
    q1_dSAR = pd.Series(np.quantile(stacked_ys_dSAR, 0.25, axis=0)).rolling(window=5).mean().values
    q3_dSAR = pd.Series(np.quantile(stacked_ys_dSAR, 0.75, axis=0)).rolling(window=5).mean().values
    
    # Primary y-axis for SR
    ax.plot(np.exp(log_area), median_SAR, c="tab:blue", linestyle="-", alpha=1.0, linewidth=2.0, label="SR")
    ax.fill_between(np.exp(log_area), q1_SAR, q3_SAR, color="tab:blue", alpha=0.4, linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Secondary y-axis for derivative
    ax2 = ax.twinx()
    ax2.plot(np.exp(log_area), median_dSAR, c="tab:red", linestyle="-", alpha=1.0, linewidth=2.0, label=r"$\frac{d \log(SR)}{d \log(A)}$")
    ax2.fill_between(np.exp(log_area), q1_dSAR, q3_dSAR, color="tab:red", alpha=0.4, linewidth=0.3)
    ax2.set_ylim(0, 0.7)
    
    # Add legends and labels will be handled in the main function
    
    return ax, ax2

if __name__ == "__main__":
    seed = 1
    MODEL = "large"
    HASH = "a53390d"  
    path_results = Path(__file__).parent / Path(f"../../scripts/results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(path_results, map_location="cpu")
    config = results_fit_split_all["config"]
    data = read_result(config.path_augmented_data)

    # Create a single figure with subplots
    fig, axs = plt.subplots(2, 5, figsize=(12, 6), sharex=True)
    habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3", ]
    
    ax2s = []  # Store the secondary axes for later reference
    
    for i, hab in enumerate(habitats):
        print(hab)
        results_fit_split = results_fit_split_all[hab]
        model = initialize_ensemble_model(results_fit_split, config, "cuda")
        print(f"MSE val: {results_fit_split['best_validation_loss']}")
        gdf = compile_training_data(data, hab, config)

        ax = axs.flatten()[i]
        ax, ax2 = plot_SAR_dSAR_combined(ax, model, results_fit_split, gdf)
        ax2s.append(ax2)
        
        if hab == "all":
            hab = ""
        ax.set_xlabel(hab)
    
    # Add a common legend
    lines_1, labels_1 = axs.flatten()[0].get_legend_handles_labels()
    lines_2, labels_2 = ax2s[0].get_legend_handles_labels()
    
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    
    # Add global labels
    fig.supxlabel("Area (m²)", fontsize=16)
    fig.text(0.04, 0.5, "Species Richness (SR)", fontsize=16, va='center', rotation='vertical')
    fig.text(0.96, 0.5, r"$\frac{d \log(SR)}{d \log(A)}$", fontsize=16, va='center', rotation='vertical')
    
    # Hide the last subplot if needed
    axs.flatten()[-1].axis("off")
    ax2s[-1].axis("off")
    
    fig.tight_layout()
    fig.savefig("figure_4_combined.pdf", dpi=300, transparent=True, bbox_inches="tight")

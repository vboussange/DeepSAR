"""
TODO: 
- make a color scale based on heterogeneity level
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient
from src.ensemble_model import initialize_ensemble_model
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).parent / Path("../../scripts/")))
from train import Config
from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data


def get_SR_dSR(model, gdf, predictors, feature_scaler, target_scaler):
    features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
    X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
    y = model(X).to("cpu")
    log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
    dlogSR_dlogA = get_gradient(log_SR, features).detach().numpy()[:, 0]
    log_SR = log_SR.detach().numpy().flatten()
    return log_SR, dlogSR_dlogA
    
def generate_SAR_dSAR(model, gdf, predictors, feature_scaler, target_scaler, log_area, pca, subsample_size=10):
    ys_SAR = []
    ys_dSAR = []
    pca_components = []
    
    # Stratified sampling along the first PCA component
    gdf['pca_component_bins'] = pd.cut(gdf['pca_component'], bins=100, labels=False)
    for _, _gdf in gdf.groupby('pca_component_bins'):
        bin_ys_SAR = []
        bin_ys_dSAR = []
        _gdf_sample = _gdf.sample(n=subsample_size, random_state=42) if len(_gdf) > subsample_size else _gdf
        for _, row in _gdf_sample.iterrows():
            _gdf_row = pd.concat([row.to_frame().T]*len(log_area), ignore_index=True)
            _gdf_row["log_area"] = log_area
            log_SR, dlogSR_dlogA = get_SR_dSR(model, _gdf_row, predictors, feature_scaler, target_scaler)
            bin_ys_SAR.append(np.exp(log_SR))
            bin_ys_dSAR.append(dlogSR_dlogA)
        ys_SAR.append(np.median(bin_ys_SAR, axis=0))
        ys_dSAR.append(np.median(bin_ys_dSAR, axis=0))
        pca_components.append(_gdf['pca_component'].median())
    
    return ys_SAR, ys_dSAR, pca_components

def plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, pca, global_pca_min, global_pca_max):
    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]

    max_log_area_train = gdf["log_area"].max()
    min_log_area_train = gdf["log_area"].min()
    npoints = 150
    log_area = np.linspace(min_log_area_train, max_log_area_train, npoints)

    ys_SAR, ys_dSAR, pca_axis = generate_SAR_dSAR(model, gdf, predictors, feature_scaler, target_scaler, log_area, pca)

    # Normalize PCA components for color mapping using global min and max
    norm = plt.Normalize(global_pca_min, global_pca_max)
    cmap = plt.get_cmap('coolwarm')

    # Function to apply moving window smoothing
    def smooth(y, window_size=5):
        return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    for ax, ys in zip((ax_SAR, ax_dSAR), [ys_SAR, ys_dSAR]):
        for i, z in enumerate(pca_axis):
            smoothed_ys = smooth(ys[i])
            smoothed_log_area = smooth(np.exp(log_area))
            ax.plot(smoothed_log_area, 
                    smoothed_ys, 
                    alpha=0.5,
                    color=cmap(norm(z)), 
                    linewidth=0.5)
        ax.set_xscale("log")
    ax_SAR.set_yscale("log")
    ax_dSAR.set_ylim(0, 0.7)

    return ax_SAR, ax_dSAR, norm, cmap
    
if __name__ == "__main__":
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"    
    checkpoint_path = Path(f"../../scripts/results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")
    config = results_fit_split_all["config"]

    fig_SAR, axs_SAR = plt.subplots(2, 5, figsize=(8, 4), sharex=True, sharey=True)
    fig_dSAR, axs_dSAR = plt.subplots(2, 5, figsize=(8, 4), sharex=True, sharey=True)
    habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3", ]
    
    # Load all data to fit PCA globally
    all_gdf = pd.concat([load_preprocessed_data(hab, config.hash_data, config.data_seed) for hab in habitats])
    std_predictors = [col for col in all_gdf.columns if col.startswith('std_')]
    pca = PCA(n_components=1)
    all_gdf['pca_component'] = pca.fit_transform(all_gdf[std_predictors])
    
    global_pca_min = all_gdf["pca_component"].min()
    global_pca_max = all_gdf["pca_component"].max()
    
    for i, (hab, gdf) in enumerate(all_gdf.groupby("habitat_id")):
        print(hab)
        results_fit_split = results_fit_split_all[hab]
        model = initialize_ensemble_model(results_fit_split, config, "cuda")
        print(f"MSE val: {results_fit_split['best_validation_loss']}")
        gdf_train = gdf.loc#[results_fit_split["train_idx"]] #todo: to fix
        ax_SAR = axs_SAR.flatten()[i]
        ax_dSAR = axs_dSAR.flatten()[i]
        plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, pca, global_pca_min, global_pca_max)
        for ax in [ax_SAR, ax_dSAR]:
            if hab == "all":
                hab = "habitat agnostic"
            ax.set_title(hab, fontweight="bold")
    
    # Add color bar to the last axis in axs_SAR
    norm, cmap = plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, pca, global_pca_min, global_pca_max)[2:]
    cbar_ax = axs_SAR.flatten()[-1]
    cbar = fig_SAR.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=cbar_ax, orientation='vertical')
    cbar.set_label('Climate features\n(PCA1)')
    cbar_ax.axis("off")

    for fig, axs, label in zip([fig_SAR, fig_dSAR], (axs_SAR, axs_dSAR), ("SR", r"$\frac{d \log(SR)}{d \log(A)}$")):
        fig.supylabel(label, fontsize=12)
        fig.supxlabel("Area (m2)")
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        axs.flatten()[-1].axis("off")
        fig.savefig(f"figure_3_{label}.png", dpi=300, transparent=True)

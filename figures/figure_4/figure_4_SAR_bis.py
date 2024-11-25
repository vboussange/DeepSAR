"""
Here we plot corrected and not corrected SAR and dSAR.
"""
import copy
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
    

def plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, dict_styles):    
    predictors  = results_fit_split["predictors"]
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

    std_predictors = [col for col in gdf.columns if col.startswith('std_')]
    gdf_corrected = copy.deepcopy(gdf)
    gdf_corrected[std_predictors] = 0.
    
    max_log_area_train = gdf["log_area"].max()
    min_log_area_train = gdf["log_area"].min()
    npoints = 150
    log_area = np.linspace(min_log_area_train, max_log_area_train, npoints)

    dict_styles["corrected"]["gdf"] = gdf_corrected
    dict_styles["not_corrected"]["gdf"] = gdf

    for plottype in dict_styles:
        gdf = dict_styles[plottype]["gdf"]
        ys_SAR = []
        ys_dSAR = []
        for _, _gdf in gdf.groupby('pca_component_bins'):
            plot_row = _gdf.sample(1)
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

            ax.plot(np.exp(log_area), 
                    median_ys, 
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

            # ax.set_xlabel("Area (m2)")
            # ax.set_ylabel(label)
            ax.set_xscale("log")
    ax_SAR.set_yscale("log")
    ax_dSAR.set_ylim(0, 0.7)

    return ax_SAR, ax_dSAR
    
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
    dict_styles = {"corrected": {"linestyle": "-", "c": "tab:blue", "label":"No clim. het."}, 
                    "not_corrected": {"linestyle": "-", "c": "tab:red", "label":"Clim. het."}}
    
    for i, hab in enumerate(habitats):
        print(hab)
        results_fit_split = results_fit_split_all[hab]
        model = initialize_ensemble_model(results_fit_split, config, "cuda")
        print(f"MSE val: {results_fit_split['best_validation_loss']}")
        gdf = load_preprocessed_data(hab, config.hash_data, config.data_seed)

        ax_SAR = axs_SAR.flatten()[i]
        ax_dSAR = axs_dSAR.flatten()[i]
        plot_SAR_dSAR(ax_SAR, ax_dSAR, model, results_fit_split, gdf, dict_styles)
        for ax in [ax_SAR, ax_dSAR]:
            if hab == "all":
                hab = ""
            ax.set_xlabel(hab)
            
            
    legs = [plt.Line2D([0,1], [0,0], color=e["c"], label = e["label"]) for e in dict_styles.values()]

    # trained_data = plt.Line2D([0], [0], marker='s', color='w', label='test data',
    # # Add the legend
    label_l1 = ["Forests", "Grasslands", "Mires", "Shrublands", "Habitat agnostic"]
    for i, (fig, axs, label) in enumerate(zip([fig_SAR, fig_dSAR], (axs_SAR, axs_dSAR), ("SR", r"$\frac{d \log(SR)}{d \log(A)}$"))):
        fig.supylabel(label, fontsize=16)
        fig.tight_layout()
        if i == 0:
            axs.flatten()[-1].legend(handles=legs, loc='center', bbox_to_anchor=(0.4, 0.4))
            for i, ax in enumerate(axs[0, :]):
                ax.set_title(label_l1[i], fontweight="bold")
        else:
            fig.supxlabel("Area (m2)", fontsize=16)
        axs.flatten()[-1].axis("off")
        fig.tight_layout()
        fig.savefig(f"figure_4_{i}.png", dpi=300, transparent=True)


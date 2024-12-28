"""
This script generates a plot of relative absolute Shapley values versus area for
different habitats.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ensemble_model import initialize_ensemble_model
from src.mlp import scale_feature_tensor
from captum.attr import ShapleyValueSampling
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / Path("../../scripts/")))
from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data
from train import Config

def get_df_shap_val(model, results_fit_split, gdf):
    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    
    gdf = gdf.copy()
    gdf['log_area_bins'] = pd.cut(gdf['log_area'], bins=100, labels=False)
    gdf = gdf.groupby('log_area_bins', group_keys=False).apply(lambda x: x.sample(min(10, len(x))))

    features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
    X = scale_feature_tensor(features, feature_scaler).to(next(model.parameters()).device)
    
    def forward_fn(X):
        with torch.no_grad():
            return model(X).flatten()
    
    explainer = ShapleyValueSampling(forward_fn)
    shap_val = explainer.attribute(X)
    shap_abs = shap_val.cpu().numpy()

    df_shap_values = pd.DataFrame(shap_abs, columns=predictors)
    df_shap_values["log_sr_values"] = gdf["log_sr"].values
    df_shap_values["log_area_values"] = gdf["log_area"].values

    return df_shap_values

def compute_shap_values(seed, model_name, hash_value, habitats):
    checkpoint_path = Path(f"../../scripts/results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{model_name}_model_full_physics_informed_constraint_{hash_value}.pth")
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")
    config = results_fit_split_all["config"]

    shap_values = {}
    for hab in habitats:
        results_fit_split = results_fit_split_all[hab]
        model = initialize_ensemble_model(results_fit_split, config, "cuda")
        gdf = load_preprocessed_data(hab, config.hash_data, config.data_seed)
        df_shap = get_df_shap_val(model, results_fit_split, gdf)
        
        std_labs = ["std_" + env for env in config.climate_variables]
        mean_labs = config.climate_variables
        df_shap["Climate heterogeneity"] = np.abs(df_shap[std_labs]).sum(axis=1)
        df_shap["Mean climate"] = np.abs(df_shap[mean_labs]).sum(axis=1)
        df_shap["Area"] = np.abs(df_shap["log_area"])
        
        df_shap[["Area", "Climate heterogeneity", "Mean climate"]] = df_shap[["Area", "Climate heterogeneity", "Mean climate"]].values / df_shap[["Area", "Climate heterogeneity", "Mean climate"]].sum(axis=1).values.reshape(-1, 1)
        
        shap_values[hab] = df_shap
    
    return shap_values, config

def plot_shap_values(shap_values, config, habitats, config_plot):
    fig, axs = plt.subplots(2, 5, figsize=(8, 4))
    
    for i, hab in enumerate(habitats):
        ax = axs.flatten()[i]
        df_shap = shap_values[hab]
        
        for var, col in config_plot:
            df_shap['log_area_bins'] = pd.cut(df_shap['log_area_values'], bins=20, labels=False)
            std_var = df_shap.groupby('log_area_bins')[var].std()
            mean_var = df_shap.groupby('log_area_bins')[var].mean()
            log_area = df_shap.groupby('log_area_bins')['log_area_values'].mean()
            ax.errorbar(np.exp(log_area), mean_var, yerr=std_var, fmt='o', c=col)
        
        ax.set_xscale("log")
        ax.set_xlabel(hab)
        ax.set_ylabel("")

    fig.supylabel("Relative absolute Shapley values")
    fig.supxlabel("Area (m2)")
    label_l1 = ["Forests", "Grasslands", "Mires", "Shrublands", "Habitat agnostic"]
    for i, ax in enumerate(axs[0, :]):
        ax.set_title(label_l1[i], fontweight="bold")
    fig.tight_layout()
    axs.flatten()[-1].axis("off")
    
    legs = [plt.Line2D([0, 1], [0, 0], color=e[1], label=e[0]) for e in config_plot]
    axs.flatten()[-1].legend(handles=legs, loc='center', bbox_to_anchor=(0.4, 0.4))
    
    fig.savefig(Path(__file__).stem + ".pdf", dpi=300, transparent=True)

if __name__ == "__main__":
    seed = 1
    model_name = "large"
    hash_value = "71f9fc7"
    habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3"]
    config_plot = [("Area", "tab:green"), ("Mean climate", "tab:blue"), ("Climate heterogeneity", "tab:red")]

    shap_values, config = compute_shap_values(seed, model_name, hash_value, habitats)
    plot_shap_values(shap_values, config, habitats, config_plot)
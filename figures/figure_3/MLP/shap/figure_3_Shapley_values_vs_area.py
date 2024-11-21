"""
TODO: 
* make Shapley values more robust
* Refactor to simplify the code
* Make sure that the model converges to robust results
    - It seems that there are two types of convergence: one where dlogSR/dlogA is = 0.3 at large A, and the other where it convergence to 0. This could be due to a lack of constraints
* Add variations as a fill_between plot
* Evaluate model on trained data, for now we do not make a distinction between the two.
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

import seaborn as sns
import pickle

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.plotting import COLOR_PALETTE
from src.MLP import MLP, scale_feature_tensor, get_gradient
from src.utils import save_to_pickle

from captum.attr import ShapleyValueSampling

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / Path("../../../../scripts/MLP3")))
from MLP_fit_torch import process_results, preprocess_gdf_hab

sys.path.append(str(Path(__file__).parent / Path("../../../../scripts/eva_processing/")))
from preprocess_eva_CHELSA_EUNIS_plot_megaplot_ratio_1_1 import load_preprocessed_data


def get_df_shap_val(model, results_fit_split, gdf):
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    
    gdf = gdf.copy()
    gdf['log_area_bins'] = pd.cut(gdf['log_area'], bins=100, labels=False)
    gdf = gdf.groupby('log_area_bins', group_keys=False).apply(lambda x: x.sample(min(10, len(x))))

    features = torch.tensor(gdf[predictors].values.astype(np.float32), dtype=torch.float32).requires_grad_(True)
    X = scale_feature_tensor(features, feature_scaler)
    def forward_fn(X):
        with torch.no_grad():
            return model(X)
    explainer = ShapleyValueSampling(forward_fn)
    shap_val = explainer.attribute(X)
    # shap_abs = np.abs(shap_val.numpy())
    shap_abs = shap_val.numpy()

    df_shap_values = pd.DataFrame(shap_abs, columns=predictors)
    # df_shap_values = df_shap_values / df_shap_values.abs().max().max()
    df_shap_values["log_sr_values"] = gdf["log_sr"].values
    df_shap_values["log_area_values"] = gdf["log_area"].values

    return df_shap_values

def load_model(result, device):
        """Load the model and scalers from the saved checkpoint."""
        
        # Load the model architecture
        predictors = result['predictors']
        model = MLP(len(predictors)).to(device)
        
        # Load model weights and other components
        model.load_state_dict(result['model_state_dict'])
        model.eval()
        return model

if __name__ == "__main__":
    seed = 1
    checkpoint_path = f"../../../../scripts/MLP3/results/MLP_fit_torch_all_habs_dSRdA_weight_1e+00_seed_{seed}/checkpoint.pth"
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")

    config = results_fit_split_all["config"]

    fig, axs = plt.subplots(2, 5, figsize=(8, 4))
    habitats = ["T1", "R1", "Q5", "S2", "all", "T3", "R2", "Q2", "S3", ]
    config_plot = [("abs_log_area", "tab:green"),  ("mean_climate_vars", "tab:blue"), ("std_climate_vars", "tab:red"),]

    # habitats = [ "S2"]

    for i, hab in enumerate(habitats):
        ax = axs.flatten()[i]
        # shap_vals = []
        results_fit_split = results_fit_split_all[hab]
        model = load_model(results_fit_split, "cpu")
        print(f"MSE val: {results_fit_split["best_validation_loss"]}")
        gdf = load_preprocessed_data(hab, config["hash"], config["data_seed"])
        _gdf = gdf#[gdf.num_plots > 10]
        df_shap = get_df_shap_val(model, results_fit_split, _gdf)
                

        std_labs = ["std_" + env for env in config["climate_variables"]]
        mean_labs =config["climate_variables"]
        df_shap["std_climate_vars"] = np.abs(df_shap[std_labs]).sum(axis=1)
        df_shap["mean_climate_vars"] = np.abs(df_shap[mean_labs]).sum(axis=1)
        df_shap["abs_log_area"] = np.abs(df_shap["log_area"])
        # rescaling
        df_shap[["abs_log_area", "std_climate_vars", "mean_climate_vars"]] = df_shap[["abs_log_area", "std_climate_vars", "mean_climate_vars"]].values / df_shap[["abs_log_area", "std_climate_vars", "mean_climate_vars"]].sum(axis=1).values.reshape(-1,1)

        for var, col in config_plot:
            # ax.scatter(df_shap.log_area_values, df_shap[var], label = var)
            df_shap['log_area_bins'] = pd.cut(df_shap['log_area'], bins=20, labels=False)
            std_var = df_shap.groupby('log_area_bins')[var].std()
            mean_var = df_shap.groupby('log_area_bins')[var].mean()
            log_area = df_shap.groupby('log_area_bins')['log_area'].mean()
            ax.errorbar(np.exp(log_area), mean_var, yerr=std_var, fmt='o', c =col)
        # ax.legend()
        ax.set_xscale("log")
        ax.set_title(hab, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supylabel("Relative absolute Shapley values")
    fig.supxlabel("Area (m2)")
    fig.tight_layout()
    axs.flatten()[-1].axis("off")
    
    legs = [plt.Line2D([0,1], [0,0], color=e[1], label = e[0]) for e in config_plot]

    # trained_data = plt.Line2D([0], [0], marker='s', color='w', label='test data',
    # # Add the legend
    axs.flatten()[-1].legend(handles=legs, loc='center', bbox_to_anchor=(0.4, 0.4))
    
    fig.savefig(Path(__file__).stem + ".png", dpi=300, transparent=True)
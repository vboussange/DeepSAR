
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from captum.attr import ShapleyValueSampling

from src.plotting import COLOR_PALETTE
from src.sar import NNSAR2
from src.utils import save_to_pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/NNSAR/")))
import NNSAR_fit_simple
from NNSAR_fit_simple import CONFIG

def calculate_shapley_values(model, X):
    module = model.module_.to(torch.device("cpu"))
    neural_net = module.nn
    env_pred = torch.tensor(X["env_pred"], dtype=torch.float32)
    log_area = torch.tensor(X["log_area"], dtype=torch.float32)

    shap_value_dict = {}
    shap_value_dict["log_area"] = X["log_area"]
    for i, param in enumerate(["logc", "z"]):
        def forward_fn(env_pred):
            with torch.no_grad():
                return neural_net(env_pred)[:, i]
        explainer = ShapleyValueSampling(forward_fn)
        shap_value_dict[param] = explainer.attribute(env_pred)
        
    # SR
    def forward_fn(env_pred, log_area):
            with torch.no_grad():
                return module(env_pred, log_area)
    explainer = ShapleyValueSampling(forward_fn)
    shap_value_dict["log_sr"] = explainer.attribute((env_pred, log_area))
    
    return shap_value_dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_shapley_values(shap_values, feature_names):
    def process_data(shap_values, param, feature_names):
        """Process and normalize the SHAP values based on the parameter."""
        if param == "log_sr":
            df = pd.DataFrame(shap_values["log_sr"][0].numpy(), columns=feature_names)
            df["shap_log_area"] = shap_values["log_sr"][1].numpy()
        else:
            df = pd.DataFrame(shap_values[param].numpy(), columns=feature_names)
        
        # Normalize the SHAP values
        df = df.sample(frac=0.1)
        df = df / df.abs().max().max()
        df["log_area"] = shap_values["log_area"][df.index]
        return df

    def create_strip_and_bar_plot(ax, df, title, sns_kwargs):
        """Create strip and bar plots for the SHAP values."""
        df_melted = pd.melt(df, id_vars="log_area", var_name="Feature", value_name="Shapley_Value")
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        sns.violinplot(
                        data=df_melted, 
                        x="Shapley_Value", 
                        y="Feature", 
                        ax=ax, 
                        scale="width",   # Controls the scaling of the violin plot (can be "area" as well)
                        inner=None,      # Remove the inner box plot or quartile lines
                        bw=0.2,          # Adjust the bandwidth, controls smoothness of the plot
                        cut=0,           # No extension of violins beyond extreme data points
                        width=0.6,       # Controls the width of the violin plots,
                        alpha=0.3,
                        color="grey",
            )
    
        sns.stripplot(
                    data=df_melted, 
                    x="Shapley_Value", 
                    y="Feature", 
                    hue="log_area",
                    ax=ax, 
                    **sns_kwargs
                    )
        ax.set_title(title)
        # ax.set_ylabel("Feature")

    def split_features(df):
        """Split the DataFrame into 'std' and non-'std' features."""
        std_features = [col for col in feature_names if col.startswith("std")]
        non_std_features = [col for col in feature_names if not col.startswith("std")]
        if "shap_log_area" in df.columns:
            return df[std_features + ["log_area"]], df[non_std_features + ["shap_log_area", "log_area"]]
        else:
            return df[std_features + ["log_area"]], df[non_std_features + ["log_area"]]


    # Create subplots for vertical arrangement
    fig, axs = plt.subplots(nrows=2, ncols=3, sharey="row", sharex=True, figsize=(8, 6))
    sns_kwargs = {"alpha": 0.5, "palette": "icefire", "legend": False, "s": 2.}
    
    params = ["log_sr", "logc", "z"]
    titles = ["$\log(sr)$", "$\log(c)$", "$z$"]

    for i, param in enumerate(params):
        df = process_data(shap_values, param, feature_names)
        
        # Split features into 'std' and 'non-std'
        std_df, non_std_df = split_features(df)
        
        # Plot for 'std' features (top row)
        create_strip_and_bar_plot(axs[0, i], std_df, titles[i], sns_kwargs)
        
        # Plot for non-'std' features (bottom row)
        create_strip_and_bar_plot(axs[1, i], non_std_df, "", sns_kwargs)

    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    axs[1,1].set_xlabel("Shapley values")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    result_path = Path(__file__).parent / Path("../../../scripts/NNSAR/NNSAR_fit_simple.pkl")
    
    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]
        
    climate_predictors  = results_fit_split["climate_predictors"]
    hab = "all"
    gdf = results_fit_split[hab]["gdf"]
    gdf_test = gdf[~results_fit_split[hab]["train_idx"]]
    gdf_train = gdf[results_fit_split[hab]["train_idx"]]
    reg = results_fit_split[hab]["reg"]
    feature_scaler, target_scaler = results_fit_split[hab]["scalers"]
    X, y, _, _ = NNSAR_fit_simple.get_Xy_scaled(gdf_test, climate_predictors, feature_scaler=feature_scaler,  target_scaler=target_scaler)

    # Compute Shapley values for logc and z
    shap_values = calculate_shapley_values(reg, X)

    # Plot the Shapley values
    fig, axs = plot_shapley_values(shap_values, climate_predictors)
    fig.tight_layout()

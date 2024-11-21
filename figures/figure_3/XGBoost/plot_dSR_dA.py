"""
figure 3 with SAR, shapley values and dSAR
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
from src.NNSAR import NNSAR2
from src.utils import save_to_pickle

import sys
from pathlib import Path

def get_df_shap_val(results_fit_split, hab):
    climate_predictors  = results_fit_split["climate_predictors"]
    gdf_test = results_fit_split[hab]["gdf"][~results_fit_split[hab]["train_idx"]].sample(1000)
    
    X_test = gdf_test[["log_area"] + climate_predictors]
    reg = results_fit_split[hab]["reg"]
    explainer = shap.Explainer(reg)
    shap_res = explainer(X_test)

    shap_abs = np.abs(shap_res.values)
    df_shap_values = pd.DataFrame(
        np.stack(shap_abs),
        columns=["log_area"] + climate_predictors,
    )
    df_shap_values = df_shap_values / df_shap_values.max().max()
    df_shap_values["log_sr"] = gdf_test["log_sr"].values
    return df_shap_values

def plot_SAR(ax, results_fit_split, hab):
    # fig, ax = plt.subplots(1)
    climate_predictors  = results_fit_split["climate_predictors"]
    gdf = results_fit_split[hab]["gdf"][results_fit_split[hab]["train_idx"]]
    gdf_plot = gdf[gdf.num_plots == 1]
    gdf_megaplot = gdf[gdf.num_plots > 1]
    reg = results_fit_split[hab]["reg"]
    X_plot = gdf_plot[["log_area"] + climate_predictors]
    X_megaplot = gdf_megaplot[["log_area"] + climate_predictors]
    log_area = np.linspace(gdf["log_area"].min(), gdf["log_area"].max(), 100)
    mydict = {"plot": {"X": X_plot, "linestyle": "-", "c": "tab:blue", "label":"No env. het."}, 
              "megaplot": {"X": X_megaplot, "linestyle": "-", "c": "tab:red", "label":"Env. het."}}
    
    for plottype in mydict:
        X = mydict[plottype]["X"]
        ys = []
        for i in range(100):
            XX = pd.concat([X.sample(1)]*100, ignore_index=True)
            XX["log_area"] = log_area
            y_pred = pd.Series(reg.predict(XX)).rolling(window=30).mean().values
            ys.append(y_pred)
            ax.plot(np.exp(XX["log_area"]), 
                    np.exp(y_pred), 
                    c=mydict[plottype]["c"], 
                    linestyle=mydict[plottype]["linestyle"],
                    alpha=0.1)

        stacked_ys = np.stack(ys, axis=0)
        median_ys = np.median(stacked_ys, axis=0)
        ax.plot(np.exp(log_area), 
                np.exp(median_ys), 
                c=mydict[plottype]["c"], 
                linestyle=mydict[plottype]["linestyle"],
                alpha=1.,
                label=mydict[plottype]["label"],
                linewidth=2.)
        ax.set_yscale("log")
        ax.set_xscale("log")
        # ax.set_xlabel("Area (m2)")
        ax.set_ylabel("Species Richness")
        
def plot_dSAR(ax, results_fit_split, hab):
    # fig, ax = plt.subplots(1)
    climate_predictors  = results_fit_split["climate_predictors"]
    gdf = results_fit_split[hab]["gdf"][results_fit_split[hab]["train_idx"]]
    gdf_plot = gdf[gdf.num_plots == 1]
    gdf_megaplot = gdf[gdf.num_plots > 1]
    reg = results_fit_split[hab]["reg"]
    X_plot = gdf_plot[["log_area"] + climate_predictors]
    X_megaplot = gdf_megaplot[["log_area"] + climate_predictors]
    log_area = np.linspace(gdf["log_area"].min(), gdf["log_area"].max(), 100)
    mydict = {"plot": {"X": X_plot, "linestyle": "-", "c": "tab:blue", "label":"No env. het."}, 
              "megaplot": {"X": X_megaplot, "linestyle": "-", "c": "tab:red", "label":"Env. het."}}
    
    for plottype in mydict:
        X = mydict[plottype]["X"]
        dys = []
        for i in range(100):
            XX = pd.concat([X.sample(1)]*100, ignore_index=True)
            XX["log_area"] = log_area
            y_pred = pd.Series(reg.predict(XX)).rolling(window=20).mean().values
            dy_da = (y_pred[2:] - y_pred[0:-2])/ 2 / (log_area[2] - log_area[0])
            dy_da = pd.Series(dy_da).rolling(window=30).mean().values
            dys.append(dy_da)
            ax.plot(np.exp(log_area[1:-1]), 
                    dy_da, 
                    c=mydict[plottype]["c"], 
                    linestyle=mydict[plottype]["linestyle"],
                    alpha=0.1)

        stacked_dys = np.stack(dys, axis=0)
        median_dys = np.median(stacked_dys, axis=0)
        ax.plot(np.exp(log_area[1:-1]), 
                median_dys, 
                c=mydict[plottype]["c"], 
                linestyle=mydict[plottype]["linestyle"],
                alpha=1.,
                label=mydict[plottype]["label"],
                linewidth=2.)
        # ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Area (m2)")
        ax.set_ylabel("dSR/dA")


if __name__ == "__main__":
    result_path = Path(__file__).parent / Path("../../../scripts/XGBoost/XGBoost_fit_simple_plot_megaplot.pkl")
    
    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]
        
    fig = plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.8)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom left
    ax3 = fig.add_subplot(gs[:, 1])   
    
    # SAR
    plot_SAR(ax1, results_fit_split, "all")
    ax1.legend()

    # dSAR
    plot_dSAR(ax2, results_fit_split, "all")
    # SHAP  
    df_shap_values = get_df_shap_val(results_fit_split, "all")
    sns_kwargs = {"alpha": 0.5, "palette": "icefire", "legend": False, "s": 2.}
    _df_shap = df_shap_values.drop("log_area", axis=1)
    df_melted = pd.melt(_df_shap, id_vars=["log_sr"], var_name="Feature", value_name="SHAP values")
    sns.stripplot(
                    data=df_melted, 
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
    fig.savefig("figure_3.png", transparent=True, dpi=300)

    # ax.set_xscale("log")
    # ax.set_xlim(1e-3, 1e0)
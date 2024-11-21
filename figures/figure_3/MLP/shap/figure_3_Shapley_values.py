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
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.plotting import COLOR_PALETTE
from src.MLP import MLP, scale_feature_tensor, get_gradient
from src.utils import save_to_pickle

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / Path("../../../scripts/MLP3")))
from MLP_fit_torch import process_results, preprocess_gdf_hab

from captum.attr import ShapleyValueSampling


def get_df_shap_val(model, results_fit_split, gdf):
    predictors  = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    gdf = gdf.sample(min(len(gdf), 10000))
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

def load_model(result, device):
        """Load the model and scalers from the saved checkpoint."""
        
        # Load the model architecture
        predictors = result['predictors']
        model = MLP(len(predictors)).to(device)
        
        # Load model weights and other components
        model.load_state_dict(result['model_state_dict'])
        model.eval()
        return model

def boxplot_byclass(
    df=None,
    ax=None,
    spread=None,
    color_palette=None,
    legend=False,
    xlab=None,
    ylab=None,
    habitats=None,
    predictors=None,
    predictor_labels = None,
    widths=0.1,
):
    if not habitats:
        habitats = list(df.habitat_names.unique())
    N = len(habitats)  # number of habitats
    M = len(predictors)  # number of groups

    dfg = df.groupby("habitat_names")
    for j, hab in enumerate(habitats):
        dfhab = dfg.get_group(hab).drop(columns=["habitat_names"])
        # dfhab = (dfhab - dfhab.values.flatten().mean()) / dfhab.values.flatten().std()
        dfhab = dfhab / dfhab.values.flatten().max()
        y = [np.abs(dfhab[k]) for k in predictors]
        xx = (
            np.arange(1, M + 1) + (j - (N + 1) / 2) * spread / N
        )  # artificially shift the x values to better visualise the std
        box_parts = ax.boxplot(
            y, positions=xx, widths=widths, vert=False, patch_artist=True,
                    showfliers=False,

        )
        for patch in box_parts['boxes']:
            patch.set_facecolor(color_palette[j])
            patch.set_edgecolor(color_palette[j])
        for whisker in box_parts['whiskers']:
            whisker.set_color(color_palette[j])
        for cap in box_parts['caps']:
            cap.set_color(color_palette[j])
        for median in box_parts['medians']:
            median.set_color('black')

    ax.set_xlabel(ylab, fontsize=14)
    x = predictors
    ax.set_yticks(np.arange(1, len(x) + 1))
    ax.set_yticklabels(predictor_labels, fontsize=14)
    if legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=color_palette[i], label=habitats[i],)
                for i in range(len(habitats))
            ],
             fontsize=10
        )
    plt.show()


if __name__ == "__main__":
    dataset = process_results()
    seed = 9
    checkpoint_path = f"../../../scripts/MLP3/results/MLP_fit_torch_all_habs_dSRdA_weight_1e+00_seed_{seed}/checkpoint.pth"
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")

    habitats = ["T1", "T3", "R1", "R2", "Q2", "Q5", "S2", "S3",  "all",  ]

    shap_vals = []
    for i, hab in enumerate(habitats):
        results_fit_split = results_fit_split_all[hab]
        model = load_model(results_fit_split, "cpu")
        print(f"MSE val: {results_fit_split["best_validation_loss"]}")
        gdf = preprocess_gdf_hab(dataset.gdf, hab, seed)
        df_shap = get_df_shap_val(model, results_fit_split, gdf)
        df_shap["habitat_names"] = hab
        shap_vals.append(df_shap)
    df_shap_vals = pd.concat(shap_vals, axis=0)
            

    color_palette = [(0.09019607843137255, 0.41568627450980394, 0.050980392156862744, .9),
                    (0.08627450980392157, 0.23921568627450981, 0.07058823529411765, .9),
                    (0.8274509803921568, 0.7372549019607844, 0.14901960784313725, .9),
                    (0.7215686274509804, 0.8313725490196079, 0.10196078431372549, .9),
                    (0.06666666666666667, 0.4, 0.6196078431372549),
                    (0.11372549019607843, 0.8470588235294118, 0.6274509803921569, .9),
                    (0.3803921568627451, 0.0196078431372549, 0.0196078431372549, .9),
                    (0.6823529411764706, 0.0784313725490196, 0.043137254901960784, .9),
                    (0.9058823529411765, 0.5411764705882353, 0.7647058823529411, .9)]

    nclasses = len(habitats)
    
    std_labs = [lab for lab in dataset.aggregate_labels if "std" in lab]
    mean_labs = [lab for lab in dataset.aggregate_labels if not "std" in lab]
    df_shap_vals["std_climate_vars"] = np.abs(df_shap_vals[std_labs]).sum(axis=1)
    df_shap_vals["mean_climate_vars"] = np.abs(df_shap_vals[mean_labs]).sum(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))

    boxplot_byclass(
        df=df_shap_vals,
        ax=ax,
        spread=0.9,
        color_palette=color_palette,
        legend=True,
        # xlab="Predictors",
        ylab="Normalised Shapley values",
        habitats=habitats,
        predictors=["log_area", "std_climate_vars", "mean_climate_vars"],
        widths=0.08,
        predictor_labels = ["Area", "Climatic heterogeneity", "Climate", ]
    )
    fig.tight_layout()
    # ax.legend()
    # plotting violin plot of climate variables per cohort

    # plotting climate var 2 effect

    fig.savefig(Path(__file__).stem + ".png", dpi=300, transparent=True)
""""
Plotting figure 2 'Feature importance'
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetRegressor
from skorch.helper import SliceDict
from skorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from src.model_validation import get_spatial_block_cv_index
from xgboost import XGBRegressor

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_polygons import partition_polygon_gdf
from src.SAR import NNSAR
from NNSAR_fit_simple import get_Xy_scaled
from src.utils import save_to_pickle
from src.plotting import COLOR_PALETTE

import shap

import pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / Path("../../figures/figure_2/")))
sys.path.append(str(Path(__file__).parent / Path("../../figures/figure_3/")))

from figure_2_EVA_EUNIS import (
    process_results,
)

from figure_3_Shap_EVA_EUNIS import (
    get_df_shap_val,
)

CONFIG = {
    "torch_params": {
        "optimizer":optim.Adam,
        "lr":5e-2,
        "batch_size":4096,
        "max_epochs":300,
        "callbacks":[("early_stopping", EarlyStopping(patience=20))],
        "optimizer__weight_decay":1e-4,
    },
}

def evaluate_model_shap(dataset, predictors, habitats):
    result_all = {}
    gdf_full = dataset.gdf
    
    for hab in habitats:
        print("Training", hab)
        results = {}
        result_all[hab] = results
        
        # here we filter the df and shuffle the rows, so that the folds are randomly selected
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        gdf = gdf_full[gdf_full.habitat_id == hab].sample(frac=1).reset_index(drop=True)

        train_cv_partition_idx, test_cv_partition_idx = train_test_split(
            gdf.partition.unique(), test_size=0.3, random_state=42
        )
        train_idx = gdf.index[gdf.partition.isin(train_cv_partition_idx)]
        test_idx = gdf.index[gdf.partition.isin(test_cv_partition_idx)]

        X, y, feature_scaler, target_scaler = get_Xy_scaled(gdf, predictors)

        reg = NeuralNetRegressor(module=NNSAR,
                                module__input_dim=X["env_pred"].shape[1],
                                **CONFIG["torch_params"])
        
        print("Training model...")
        reg.fit(SliceDict(**X)[train_idx], y[train_idx],)

        # background points for shap
        print("Finding background points...")
        background = shap.kmeans(X["env_pred"][train_idx], 50)

        print("Calculating shapley values for nn1")
        nn = reg.module_.nn1
        def model_wrapper(x):
            with torch.no_grad():
                return nn(torch.tensor(x, dtype=torch.float)).numpy()
        # model_wrapper(X["env_pred"][train_idx])

        explainer = shap.KernelExplainer(model_wrapper, background)
        results["shap_res_nn1"] = explainer.shap_values(X["env_pred"][test_idx], nsamples=500)


        print("Calculating shapley values for nn2")
        nn = reg.module_.nn2
        def model_wrapper(x):
            with torch.no_grad():
                return nn(torch.tensor(x, dtype=torch.float)).numpy()
        # model_wrapper(X["env_pred"][train_idx])

        explainer = shap.KernelExplainer(model_wrapper, background)
        results["shap_res_nn2"] = explainer.shap_values(X["env_pred"][test_idx], nsamples=500)

    return result_all

def get_df_shap_val(result_modelling, key, feature_names):
    shap_vals = []
    for hab in result_modelling.keys():
        shap_res = result_modelling[hab][key]
        # Calculate the mean absolute Shapley values across all classes and samples
        df_shap_values = pd.DataFrame(
            np.stack(shap_res).reshape(-1, len(feature_names)),
            columns=feature_names,
        )  # Mean across all samples and classes

        shap_vals.append(df_shap_values)

    df_shap_vals = pd.concat(shap_vals, axis=0)
    return df_shap_vals

def boxplot_byclass(
    df=None,
    ax=None,
    spread=None,
    color_palette=None,
    legend=False,
    xlab=None,
    ylab=None,
    groups=None,
    predictors=None,
    predictor_labels = None,
    widths=0.1,
):

    N = len(groups)
    M = len(predictors)  # number of groups

    dfg = df.groupby("group")
    for j, hab in enumerate(groups):
        dfhab = dfg.get_group(hab).drop(columns=["group", "habitat_names"])
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
    ax.set_yticks(np.arange(0.5, len(x)+0.5))
    ax.set_yticklabels(predictor_labels, fontsize=14)
    if legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=color_palette[i], label=groups[i],)
                for i in range(len(groups))
            ],
             fontsize=10
        )
    plt.show()


if __name__ == "__main__":    
    habitats = ["all"]
    dataset = process_results()
    predictors = ["log_area"] + dataset.aggregate_labels
    result_path = Path(str(Path(__file__).parent / Path(__file__).stem) + ".pkl")

    # with open("NNSAR_fit_simple.pkl", 'rb') as pickle_file:
    #     result_modelling = pickle.load(pickle_file)["result_modelling"]

    if False:
        results_shap = evaluate_model_shap(dataset, predictors, habitats)
        save_to_pickle(result_path ,results_shap=results_shap)

    else:
        with open(result_path, 'rb') as file:
            results_shap = pickle.load(file)["results_shap"]

    df_shap_vals_nn1 = get_df_shap_val(results_shap, "shap_res_nn1", dataset.aggregate_labels)
    df_shap_vals_nn1["group"] = "alpha"
    df_shap_vals_nn2 = get_df_shap_val(results_shap, "shap_res_nn2", dataset.aggregate_labels)
    df_shap_vals_nn2["group"] = "beta"
    df_shap_vals = pd.concat([df_shap_vals_nn1, df_shap_vals_nn2])

    std_labs = [lab for lab in dataset.aggregate_labels if "std" in lab]
    mean_labs = [lab for lab in dataset.aggregate_labels if not "std" in lab]
    df_shap_vals["std_climate_vars"] = np.abs(df_shap_vals[std_labs]).sum(axis=1)
    df_shap_vals["mean_climate_vars"] = np.abs(df_shap_vals[mean_labs]).sum(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))

    boxplot_byclass(
        df=df_shap_vals,
        ax=ax,
        spread=0.9,
        color_palette=["tab:blue", "tab:red"],
        legend=True,
        # xlab="Predictors",
        ylab="Normalised Shapley values",
        groups=["alpha", "beta"],
        predictors=["std_climate_vars", "mean_climate_vars"],
        widths=0.12,
        predictor_labels = ["Climate heterogeneity", "Climate", ]
    )
    fig.tight_layout()
    # plotting violin plot of climate variables per cohort

    # plotting climate var 2 effect

    fig.savefig(Path(__file__).stem + ".png", dpi=300, transparent=True)

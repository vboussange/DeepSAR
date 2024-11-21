""""
Plotting figure 2 'prediction power of climate, area, and both on SR'
"""
# V2 uses EVA_EUNIS_Chelsa_20000_block_relative_extent_0.2_v2 augmented dataset (plot/megaplot 1:1 ratio)


import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from sklearn.model_selection import KFold, cross_validate, GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import pickle
from src.utils import save_to_pickle

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.plotting import read_result, ResultData, boxplot_bypreds

import scipy.stats as stats

import sys
sys.path.append(str(Path(__file__).parent / Path("../../../scripts/XGBoost/")))
import XGBoost_fit_pred_comb_SBCV_plot_megaplot as XGBoost_fit

PREDICTORS = ["area", "climate", "area+climate, habitat agnostic", "area+climate"]
CONFIG = {
    "test_split": 0.3,  # not used
    "xgb_params": {
        "booster": "gbtree",
        "learning_rate": 0.05,
        "max_depth": 12,
        "lambda": 10,
        "objective": "reg:squarederror",  # can be reg:squarederror, reg:squaredlogerror
        "min_child_weight": 1.0,
        "device" : "cuda",
        "tree_method": "hist",
    },
    "kfold": GroupKFold(
        n_splits=10,
    ),
    "scoring": [
        "r2",
        "neg_mean_squared_error",
    ],
}


def process_results(path_results):
    """
    Reading and processing compiled data.
    """
    results = read_result(path_results)
    gdf_full = results["SAR_data"]
    config = results["config"]
    aggregate_labels = config["env_vars"] + ["std_" + env for env in config["env_vars"]]

    gdf_full = gdf_full.reset_index()
    gdf_nonan = gdf_full.dropna(axis=0, how="any")
    print("we have", len(gdf_full), "entries")
    print("we have", len(gdf_nonan), "entries with non nans")
    print("Filtering rows with nans")
    gdf_full = gdf_nonan

    gdf_full["log_area"] = np.log(gdf_full["area"].astype(np.float64))  # area
    gdf_full["log_sr"] = np.log(gdf_full["sr"].astype(np.float64))

    gdf_full.reset_index(inplace=True, drop=True)
    return ResultData(gdf_full, config, aggregate_labels, None)


def evaluate_model_per_hab_residuals(gdf_train, gdf_test, xgb_params, predictors):
    result_all = {}

    X_train = gdf_train[predictors]  # env vars
    y_train = gdf_train["log_sr"]
    reg = XGBRegressor(**xgb_params)
    reg.fit(X_train, y_train)
    X_test = gdf_test[predictors]  # env vars

    result_all["residuals"] = reg.predict(X_test) -gdf_test["log_sr"]
    # result_all["residuals"] = np.exp(reg.predict(X_test)) - np.exp(gdf_test["log_sr"])
    result_all["log_area"] = gdf_test["log_area"]
    result_all["reg"] = reg
    
    print(hab)
    print("Test dataset partition indices:", gdf_test.partition.unique(), "\nlength:", len(gdf_test))
    print("Train dataset partition indices:", gdf_train.partition.unique(), "\nlength:", len(gdf_train))
        
    return result_all


def preprocess_gdf_hab(gdf_full, hab):
    
    if hab == "all":
        gdf = pd.concat([gdf_full[gdf_full.habitat_id == hab], gdf_full[gdf_full.num_plots == 1]])
    else:
        gdf = gdf_full[gdf_full.habitat_id == hab] 
        
    plot_gdf = gdf[gdf.num_plots == 1]
    # taking idx of single plots, filtering out possible doublon
    # plot_idxs = plot_gdf['plot_idxs'].value_counts()[plot_gdf['plot_idxs'].value_counts() == 1].index
    plot_idxs = plot_gdf.index.to_series()
    megaplot_idxs = gdf[gdf.num_plots > 1].index.to_series()
    megaplot_idxs = megaplot_idxs.sample(min(len(plot_idxs), len(megaplot_idxs)), replace=False)
    plot_megaplot_gdf_hab = gdf_full.loc[pd.concat([plot_idxs, megaplot_idxs])]
    return plot_megaplot_gdf_hab.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    with open(XGBoost_fit.PATH_RESULTS, 'rb') as file:
        result_modelling = pickle.load(file)["result_modelling"]

    for hab, val in result_modelling.items():
        print(hab)
        mse_arr = []
        # removing nan values
        for val2 in val.values():
            neg_mse = val2["test_neg_mean_squared_error"]
            if neg_mse.size > 0:
                mse_arr.append(-neg_mse)
        mse = np.stack(mse_arr)
        non_nan_columns = ~np.isnan(mse).any(axis=0)
        matrix_no_nan_columns = mse[:, non_nan_columns]
        for i, val2 in enumerate(val.values()):
            if val2["test_neg_mean_squared_error"].size > 0:
                val2["test_MSE"] = mse[i,:]
            else:
                val2["test_MSE"] = np.array([])
        

    PREDICTORS = ["area", "climate", "area+climate, habitat agnostic", "area+climate"]

    # plotting results for test data
    fig = plt.figure(figsize=(6, 6))
    nclasses = len(list(result_modelling.keys()))
    habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    # habitats = ["T1", "T3", "S2", "S3"]
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])
    
    # first axis
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_ylim(0.1, 0.8)
    boxplot_bypreds(
        result_modelling,
        ax=ax1,
        spread=0.5,
        colormap="Set2",
        legend=True,
        xlab="",
        ylab="MSE",
        yscale="linear",
        yname="test_MSE",
        habitats=habitats,
        predictors=PREDICTORS,
        widths=0.1,
    )
    label_l1 = ["Forests", "Grasslands", "Mires", "Shrublands"]
    for i,x in enumerate(np.arange(1, len(habitats), step=2)):
        ax1.text(x+0.5, -0., label_l1[i], ha='center', va='bottom', fontsize=10, color='black')
    fig.savefig(Path(__file__).stem + "_model_score.png", transparent=True, dpi=300)
    
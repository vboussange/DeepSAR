""""
Analysing synthetic data from a simple SAR + climate model
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from src.model_validation import get_spatial_block_cv_index
from xgboost import XGBRegressor
import pickle

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_polygons import partition_polygon_gdf
from src.plotting import read_result, ResultData, boxplot_bypreds

import scipy.stats as stats

PATH_RESULTS = Path(
    "/home/boussang/SAR_modelling/python/results/Synthetic_polygons_CHELSA/Synthetic_Copernicus_CHELSA_20000.pkl"
)

PREDICTORS = ["area", "CHELSA", "area+CHELSA"]
CONFIG = {"test_split": 0.3, # not used
        "xgb_params": {
            "booster": "gbtree",
            "learning_rate": 0.05,
            "max_depth": 4,
            "lambda": 10,
            "objective": "reg:squarederror",  # can be reg:squarederror, reg:squaredlogerror
            "min_child_weight": 1.0,
            # "device" : "cuda",
            "tree_method": "hist",
        },
        "kfold": KFold(
            n_splits=10,
            shuffle=True,
            #    random_state=42
        ),
        "scoring": [
            "r2",
            "neg_mean_squared_error",
        ],
    }

def process_results(path_results = PATH_RESULTS, n_partition=100):
    results = read_result(path_results)
    gdf_full = results["SAR_data"]
    config = results["config"]
    aggregate_labels = config["env_vars"] + ["std_" + env for env in config["env_vars"]]

    gdf_full = gdf_full.reset_index()
    gdf_nonan = gdf_full.dropna(axis=0, how="any")
    print("we have", len(gdf_full), "entries")
    print("we have", len(gdf_nonan), "entries with non nans")
    print("Filtering rows with nans")
    gdf_full = gdf_nonan  #[gdf_full.density_obs > gdf_full.density_obs.median()].reset_index()

    print("Partitioning")
    gdf_full["centroids"] = gdf_full.to_crs(epsg=3035).centroid
    gdf_full = gpd.GeoDataFrame(pd.concat(partition_polygon_gdf(gdf_full, n_partition)))
    
    gdf_full["log_area"] = np.log(gdf_full["area"].astype(np.float64))  # area
    gdf_full["log_sr"] = np.log(gdf_full['sr'].astype(np.float64))

    gdf_full.reset_index(inplace=True, drop=True)
    return ResultData(gdf_full, config, aggregate_labels, None)


def plot_partitioning(gdf_full):
    if "partition_modulo" not in gdf_full.columns:
        gdf_full["partition_modulo"] = gdf_full["partition"].mod(4)
        gdf_full.set_geometry("centroids").plot(column="partition_modulo",
                                                cmap="rainbow",
                                                s=1)

def evaluate_model(*, gdf, kfold, predictor_labels, xgb_params, scoring):
    X = gdf[predictor_labels]  # env vars
    # X = pd.DataFrame()
    y = gdf["log_sr"]
    reg = XGBRegressor(**xgb_params)
    cv_idxs = get_spatial_block_cv_index(gdf, kfold)

    return cross_validate(reg,
                          X,
                          y,
                          cv=cv_idxs,
                          scoring=scoring,
                          return_train_score=True)


def evaluate_model_per_hab(res, kfold, xgb_params, scoring, habitats):
    result_all = {}
    gdf_full = res.gdf
    aggregate_labels = res.aggregate_labels
    for hab in habitats:
        print("Training", hab)
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        result_all[hab] = {}
        results = result_all[hab]
        results["area"] = evaluate_model(gdf=gdf,
                                         kfold=kfold,
                                         predictor_labels=["log_area"],
                                         xgb_params=xgb_params,
                                         scoring=scoring)

        results["CHELSA"] = evaluate_model(gdf=gdf,
                                    kfold=kfold,
                                    predictor_labels=aggregate_labels,
                                    xgb_params=xgb_params,
                                    scoring=scoring)
        
        results["area+CHELSA"] = evaluate_model(gdf=gdf,
                                    kfold=kfold,
                                    predictor_labels=aggregate_labels + ["log_area"],
                                    xgb_params=xgb_params,
                                    scoring=scoring)
    return result_all

if __name__ == "__main__":

    dataset = process_results()
    plot_partitioning(dataset.gdf)
    dataset.config["habitats"] = ["DBF_closed"]
    dataset.gdf["habitat_id"] = "DBF_closed"
    
    # fitting models
    result_modelling = evaluate_model_per_hab(dataset, CONFIG["kfold"], CONFIG["xgb_params"],
                                              CONFIG["scoring"], dataset.config["habitats"])
    
    # plotting results for test data
    fig, ax = plt.subplots(figsize=(7,4))
    nclasses = len(list(result_modelling.keys()))
    print(result_modelling.keys())
    boxplot_bypreds(result_modelling,
                    ax=ax,
                    spread=.4,
                    colormap="Set2",
                    legend=True,
                    xlab="Habitats",
                    ylab="MSE",
                    yscale="log",
                    yname="test_neg_mean_squared_error",
                    habitats=dataset.config["habitats"],
                    predictors=PREDICTORS,
                    widths=0.08)
    fig.tight_layout()
    fig.savefig("figure_2_test.png", dpi=300, transparent=True)

    with open("paired_t_test.txt", "w") as file:
        for j, c in enumerate(dataset.config["habitats"]):
            print(c, file=file)
            performance_baseline = -result_modelling[c]["area"]["test_neg_mean_squared_error"]
            performance_new_features = -result_modelling[c]["area+CHELSA"]["test_neg_mean_squared_error"]
            t_statistic, p_value = stats.ttest_rel(performance_baseline, performance_new_features)
            print("T-statistic:", t_statistic, file=file)
            print("P-value:", p_value, file=file)
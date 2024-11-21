""""
Plotting figure 3 'prediction power of climate, area, and both on SR'
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
import shap

from src.data_processing.utils_env_pred import calculate_aggregates, CHELSADataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_landcover import CopernicusDataset
import sys

sys.path.append(str(Path(__file__).parent / Path("../figure_2/")))
from figure_2_synthetic_experiment import (
    process_results,
    CONFIG,
)


def evaluate_model_per_hab(res, xgb_params, habitats, predictor_labels):
    result_all = {}
    gdf_full = res.gdf
    for hab in habitats:
        print("Training", hab)
        gdf = gdf_full[gdf_full.habitat_id == hab].reset_index(drop=True)
        X = gdf[predictor_labels]  # env vars
        y = gdf["log_sr"]
        reg = XGBRegressor(**xgb_params)
        reg.fit(X, y)
        result_all[hab] = {}
        result_all[hab]["reg"] = reg
        result_all[hab]["explainer"] = shap.Explainer(reg)
    return result_all

def create_X_map(X, ncells=10):
    env_pred_dataset = CHELSADataset()
    CHELSA_arr = env_pred_dataset.load()
    coarse = CHELSA_arr.coarsen(x=ncells, y=ncells, boundary="trim")
    coarse_mean = coarse.mean().to_dataset(dim="variable")
    coarse_std = coarse.std().to_dataset(dim="variable")
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    X_map["log_area"] = X["log_area"].median()
    X_map = X_map[X.columns]
    assert X_map.shape[1] == X.shape[1]
    return X_map

def create_raster_sr(X_map, explainer, pred):
    shap_values = explainer(X_map)
    Xy_map = X_map.copy()
    Xy_map["sensitivity"] = shap_values[:, pred].values
    sr_raster = Xy_map[["sensitivity"]].to_xarray()
    return sr_raster["sensitivity"].sortby(["y", "x"])


if __name__ == "__main__":
    dataset = process_results()
    dataset.config["habitats"] = [" DBF_closed"]
    dataset.gdf["habitat_id"] = " DBF_closed"
    predictors = ["log_area"] + dataset.aggregate_labels
    result_modelling = evaluate_model_per_hab(
        dataset, CONFIG["xgb_params"], dataset.config["habitats"], predictors
    )

    lc_dataset = CopernicusDataset()
    lc_raster = lc_dataset.load_landcover_level3_1km()
    inv_legend = {v: k for k, v in lc_dataset.legend_l3.items()}
    
    X = dataset.gdf[predictors]
    X_map = create_X_map(X, ncells=10)

    projected_results = {}
    for hab in dataset.config["habitats"]:
        print("Processing", hab)
        hab_id = inv_legend[hab]
        ncells = 10
        # calculating raster
        # TODO: this is not efficient, since we discard most of the values later on
        rast = create_raster_sr(
            X_map,
            result_modelling[hab]["explainer"],
            "log_area",
        )

        # masking areas
        lcb = lc_raster == hab_id
        coarse = lcb.coarsen(x=ncells, y=ncells, boundary="trim")
        coarse_max = coarse.max().to_dataset()
        coarse_max = coarse_max.interp_like(rast)

        rast_masked = rast.where(coarse_max["Copernicus1KML3"])
        projected_results[hab] = rast_masked
        # # plotting raster
        # fig, ax = plt.subplots()
        # rast_masked.plot(ax=ax)
        # ax.set_title(hab)
        # fig.tight_layout()
        # fig.savefig(f"map_GBIF_{hab}.png", transparent=True)

    fig, ax = plt.subplots()
    for i, (hab, raster) in enumerate(projected_results.items()):
        raster.plot(ax=ax, cmap="coolwarm")
        ax.set_title(hab)

    fig.tight_layout()
    fig.savefig("figure_5_EVA_Copernicus.png", dpi=300, transparent=True)

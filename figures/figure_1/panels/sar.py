""""
Plotting figure 2 'prediction power of climate, area, and both on SR'
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import KFold, cross_validate, GroupShuffleSplit
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
    "/home/boussang/SAR_modelling/python/results/EVA_polygons_CHELSA/EVA_EUNIS_CHELSA/EVA_EUNIS_Chelsa_20000_block_relative_extent_0.2.pkl"
)

def process_results(path_results = PATH_RESULTS):
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
    gdf_full["log_sr"] = np.log(gdf_full['sr'].astype(np.float64))

    gdf_full.reset_index(inplace=True, drop=True)
    return ResultData(gdf_full, config, aggregate_labels, None)



if __name__ == "__main__":

    if True:
        dataset = process_results()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    habitat_gdf = dataset.gdf[dataset.gdf.habitat_id == "T1"]
    gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=1)
    train_index, test_index = next(gss.split(habitat_gdf, groups=habitat_gdf.partition))
    habitat_gdf_train = habitat_gdf.iloc[train_index]
    habitat_gdf_test = habitat_gdf.iloc[test_index]

    Xr = habitat_gdf_test.log_area
    Yr = habitat_gdf_test.bio1
    Zr = habitat_gdf_test.log_sr
    ax.scatter(Xr, Yr, Zr,  s=1, c = "tab:red", label="Testing data")
    ax.scatter(habitat_gdf_train.log_area, habitat_gdf_train.bio1, habitat_gdf_train.log_sr,  s=1, c= "tab:blue", label="Training data")
    ax.legend()
    ax.set_xlabel("log(A)")
    ax.set_zlabel("log(SR)")
    ax.set_ylabel("Mean bio1")
    
        # Plot the 2D projections using `plot`. This is the piece I'd like to improve 
    ax.scatter(Xr, Yr, s=.2, color='tab:grey', zdir='z', zs=0.)
    ax.scatter(Xr, Zr, s=.2, color='tab:grey', zdir='y', zs=20.)
    ax.scatter(Yr, Zr, s=.2, color='tab:grey', zdir='x', zs=-5.)
    

    # Set the limits to make the axes visible; adjust these as necessary
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_zlim(0, 10)
    # Set the azimuth and elevation angles
    ax.view_init(azim=-30, elev=30, vertical_axis="z")
    
    # Optional: Customize the appearance of the axes
    # ax.xaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.5)
    # ax.yaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.5)
    # ax.zaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.5)

    # Hide the panes to make only the axes visible
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    # Hide the grid
    ax.grid(False)

    fig.tight_layout()
    fig.savefig("sar.png", transparent=True, dpi=300)
    
"""
!!! Problem of evaluation of out[:, 0] and out[:, 1]. When evaluated in global
scope, works, but does not work when evaluated in local scope.
This seems like a bug from skorch or pytorch, as depending on the number of layers, we do not get the same behavior
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd


from src.plotting import COLOR_PALETTE
from src.utils import save_to_pickle
from src.data_processing.utils_polygons import create_grid

import sys
from pathlib import Path

if __name__ == "__main__":
    result_path = Path(__file__).parent / Path("../../../scripts/XGBoost/XGBoost_fit_simple_plot_megaplot.pkl")

    with open(result_path, 'rb') as file:
        results_fit_split = pickle.load(file)["result_modelling"]
        
    climate_predictors  = results_fit_split["climate_predictors"]
    
    gdf = results_fit_split["all"]["gdf"]
    gdf = gdf[gdf.num_plots==1]
    
    block_length = (gdf.total_bounds[2] - gdf.total_bounds[0]) * 0.01
    grid = create_grid(gdf, block_length)
    joined = gpd.sjoin(gdf, grid, how='left', predicate='within')
    aggregated = joined.groupby('index_right').agg({'sr': 'mean'}).reset_index()
    grid['sr'] = 0.
    grid['sr'].loc[aggregated["index_right"]] = aggregated['sr'].values.astype(np.float64)

    fig, ax =plt.subplots()
    grid.plot(column='sr', ax=ax, legend=True, cmap='Greens')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Raw plot species richness, aggregated")
    fig.savefig("raw_SR_map.png", dpi=300, transparent=True)
    
    # exporting gdf
    _gdf =  gdf[["sr", "habitat_id"]]
    _gdf["longitude"] = gdf.to_crs(epsg=4326).geometry.x
    _gdf["latitude"] = gdf.to_crs(epsg=4326).geometry.y
    _gdf.to_csv('plot_SR.csv', index=False)


"""
Illustrating the generation procedure of mega plots
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import math
from tqdm import tqdm
import seaborn as sns

from src.generate_sar_data_eva import clip_EVA_SR_gpu
from src.generate_SAR_data_GBIF import generate_random_boxes
from src.data_processing.utils_eva import EVADataset
from src.data_processing.utils_env_pred import CHELSADataset
from src.utils import save_to_pickle
from src.data_processing.utils_polygons import (
    partition_polygon_gdf,
)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

plot_gdf, dict_sp = EVADataset().load()
plot_gdf = plot_gdf.to_crs(epsg=3035)


num_polygons = 10000
area_range = (1e4, 1e11)  # in m2
side_range = (1e2, 1e6) # in m
polygons_gdf = generate_random_boxes(
    plot_gdf,
    num_polygons,
    area_range,
    side_range,
)

def draw_map(ax_map):
    ax_map.add_feature(cfeature.COASTLINE)
    # ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, edgecolor="black")
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")


    # Add labels and a legend
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # ax_map.set_title("Plot Locations")
    ax_map.legend(loc="best")
    
    
polygons_gdf.geometry[3]
    
    
fig = plt.figure(figsize = (10, 5))
ax_map = fig.add_subplot(121, projection=ccrs.epsg(3035))
draw_map(ax_map)
ax_map.add_geometries(polygons_gdf.geometry, crs=ccrs.epsg(3035), facecolor='blue', edgecolor='red', alpha=0.5)

ax = fig.add_subplot(122)
polygons_gdf["area"] = polygons_gdf.area
sns.kdeplot(polygons_gdf, 
            x = 'area', 
            ax=ax, 
            log_scale=(True, False))

fig.savefig("boxes_generation_EVA.png", dpi = 300)
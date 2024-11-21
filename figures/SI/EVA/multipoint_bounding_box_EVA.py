"""
Plotting bounding boxes of multipoints generated from EVA dataset
"""
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import box
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).parent / Path("../../figure_2/")))
from figure_2_EVA import (
    process_results,
    CLASSES
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
    
# loading
result_data = process_results(path_results = "/home/boussang/SAR_modelling/python/results/EVA_polygons_CHELSA/EVA_landsys_CHELSA/EVA_Chelsa_EUNIS_1000.pkl")
df = result_data.gdf
df = df[df.habitat_id.isin(CLASSES)]
df["n_EVA_entries"] = [len(p.geoms) for p in df.geometry]
df = df[df["n_EVA_entries"] > 1] # filtering out only monopoints
df["box_multipoint"] = [box(*p.bounds) for p in df.geometry]

df_by_hab = df.groupby("habitat_id")


color_palette = sns.color_palette("Set2", len(CLASSES))
fig = plt.figure(figsize = (10, 10))
ax_map = fig.add_subplot(111, projection=ccrs.epsg(3035))
draw_map(ax_map)

for i, hab in enumerate(CLASSES):
    # plotting area, sampling effort density and sr
    df = df_by_hab.get_group(hab)
    ax_map.add_geometries(df["box_multipoint"], 
                          crs=ccrs.epsg(3035), 
                          facecolor=color_palette[i], 
                          edgecolor=color_palette[i], 
                          alpha=0.5,
                          label=hab)
ax_map.legend(handles=[
            Line2D([0], [0], color=color_palette[i], label=CLASSES[i], linewidth=8)
            for i in range(len(CLASSES))
        ])
fig.savefig("multipoint_bounding_box_EVA.png", dpi = 300)
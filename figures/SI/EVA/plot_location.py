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
from src.data_processing.utils_eva import EVADataset

color_palette = [(0.09019607843137255, 0.41568627450980394, 0.050980392156862744, 1.0),
                (0.08627450980392157, 0.23921568627450981, 0.07058823529411765, 1.0),
                (0.8274509803921568, 0.7372549019607844, 0.14901960784313725, 1.0),
                (0.7215686274509804, 0.8313725490196079, 0.10196078431372549, 1.0),
                (0.06666666666666667, 0.4, 0.6196078431372549),
                (0.11372549019607843, 0.8470588235294118, 0.6274509803921569, 1.0),
                (0.3803921568627451, 0.0196078431372549, 0.0196078431372549, 1.0),
                (0.5450980392156862, 0.03529411764705882, 0.03529411764705882, 1.0),
                (0.9058823529411765, 0.5411764705882353, 0.7647058823529411)]
habitats = ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]

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
plot_gdf, dict_sp = EVADataset().load()
plot_gdf = plot_gdf.to_crs(3035)
df_by_hab = plot_gdf.groupby("Level_2")

fig = plt.figure(figsize = (5, 7))
ax_map = fig.add_subplot(111, projection=ccrs.epsg(3035))
draw_map(ax_map)

# plotting all points
xs = [point.x for point in plot_gdf["geometry"]]
ys = [point.y for point in plot_gdf["geometry"]]
ax_map.scatter(xs,ys, 
                c=color_palette[-1], 
                alpha=1,
                label="all",
                s=0.1)


for i, hab in enumerate(habitats[:-1]):
    # plotting area, sampling effort density and sr
    df = df_by_hab.get_group(hab)
    xs = [point.x for point in df["geometry"]]
    ys = [point.y for point in df["geometry"]]
    ax_map.scatter(xs,ys, 
                    c=color_palette[i], 
                    alpha=0.5,
                    label=hab,
                    s=0.1)
    
    
ax_map.legend(handles=[
            Line2D([0], [0], color=color_palette[i], label=habitats[i], linewidth=8)
            for i in range(len(habitats))
        ])

fig.savefig("plot_location.png", dpi = 300, transparent=True)
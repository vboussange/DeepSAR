import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

from src.plotting import CMAP_BR
from src.data_processing.utils_eva import EVADataset
from scipy.ndimage import gaussian_filter

def load_and_preprocess_data():
    plot_gdf, _ = EVADataset().load()
    plot_gdf = plot_gdf.set_index("plot_id")
    plot_gdf = plot_gdf.to_crs("EPSG:3035")
    return plot_gdf

if __name__ == "__main__":
    plot_gdf = load_and_preprocess_data()
    world = gpd.read_file("../../../data/raw/NaturalEarth/ne_10m_admin_0_countries.shp")
    
    # extract x and y coordinates of each plot
    xs = plot_gdf.geometry.x.values
    ys = plot_gdf.geometry.y.values
    
    fig, ax = plt.subplots(figsize=(6, 4))

    # plot the density using hexbin
    gridsize=75
    hb = ax.hexbin(
        xs, ys,
        gridsize=gridsize,            # similar resolution to n_bins
        cmap=CMAP_BR,
        mincnt=1,                # hide empty bins if desired
        linewidths=0.,
        edgecolors='none',
        vmin=0,
        vmax=2000,
    )

    # Get the resolution (width) of a single hexagon in data units
    # The width is the distance between centers of two adjacent hexagons horizontally
    hex_width = (xs.max() - xs.min()) / gridsize
    print(f"Hexagon width (data units): {hex_width}")

    # add a colorbar for counts
    cb = fig.colorbar(hb, ax=ax, shrink=0.5, aspect=10)
    cb.set_label('Number of samples')

    # plot borders of all countries
    world = world.to_crs(plot_gdf.crs)  # ensure CRS matches
    world.boundary.plot(ax=ax, linewidth=0.4, color='black')

    # Restrict the plot to the extent of plot_gdf
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    ax.set_xlim(minx*0.9, maxx)
    ax.set_ylim(miny, maxy*0.9)
    ax.set_aspect('auto')
    hb.set_rasterized(True)

    ax.set_axis_off()
    fig.savefig("figure_density.pdf", dpi=300, transparent=True)

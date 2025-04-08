import matplotlib.pyplot as plt
import pickle
import xarray as xr
import rioxarray
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import matplotlib.colors as colors

rcparams = {
            "font.size": 9,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 16,
            "lines.markersize": 3
        }
plt.rcParams.update(rcparams)


seed = 1
MODEL = "large"
HASH = "a53390d"

# Constants for file paths
RAST_PATH = Path(f"../../scripts/results/projections")
SAR_PATH = Path(f"../../scripts/results/true_SAR/true_SAR_ensemble_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")


def load_data(rast_path=RAST_PATH, sar_path=SAR_PATH):
    """Load data from pickle files."""
    # Find and read all raster files containing MODEL and HASH in the path
    rast_dict = {}
    for file_path in rast_path.glob(f"*.tif"):
        # Extract resolution from filename (assuming it's part of the filename pattern)
        # Typically files might be named something like "projection_1e+03_large_71f9fc7.tiff"
        name = file_path.stem
        # Open the raster file using xarray with rioxarray
        raster_data = rioxarray.open_rasterio(file_path)
        rast_dict[name] = raster_data
    with open(sar_path, 'rb') as pickle_file:
        dict_sar = pickle.load(pickle_file)["dict_SAR"]
    return rast_dict, dict_sar

# def preprocess_raster(rast, coarsen_factor=0, rolling_window=2):
#     """Preprocess raster data by coarsening and smoothing."""
#     if coarsen_factor > 0:
#         rast = rast.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean() 
#     if rolling_window > 0:
#         rast = rast.rolling(x=rolling_window, y=rolling_window, 
#                             center=False, 
#                             # min_periods=4
#                             ).mean()
#     return rast

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, title='', **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title(title)
    ax.set_axis_off()

def plot_sar(ax, dict_sar, dict_plot, area):
    """Plot SAR data on the central plot."""
    for loc, loc_info in dict_plot.items():
        sar_data = dict_sar[loc]
        color = loc_info['c']
        ax.plot(area, np.exp(sar_data["log_SR"]), c=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Area", bbox=dict(facecolor='white', edgecolor='none', pad=3))
    ax.set_ylabel("Species Richness (SR)", bbox=dict(facecolor='white', edgecolor='none', pad=3, alpha=0.5))

def plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=100000):
    """Plot bounding boxes on corner plots."""
    for loc, loc_info in dict_plot.items():
        sar_data = dict_sar[loc]
        color = loc_info['c']

        minx, miny, maxx, maxy = sar_data['coords_epsg_3035']
        bbox = box(minx, miny, maxx, maxy)
        gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:3035')
        centroid_proj = gdf_bbox.centroid.geometry.iloc[0]

        x_centroid, y_centroid = centroid_proj.coords[0]
        minx_proj = x_centroid - buffer_size_meters
        maxx_proj = x_centroid + buffer_size_meters
        miny_proj = y_centroid - buffer_size_meters
        maxy_proj = y_centroid + buffer_size_meters
        bbox_proj = box(minx_proj, miny_proj, maxx_proj, maxy_proj)
        x, y = bbox_proj.exterior.xy

        # Plot the bbox with white border first, then the colored line
        col = []
        col.append(ax.plot(x, y, color='white', linewidth=4)[0])  # Wider white border
        col.append(ax.plot(x, y, color=color, linewidth=2)[0])    # Colored line on top
        for c in col:
            c.set_rasterized(True)
    

if __name__ == '__main__':
    # Load data
    rast_dict, dict_sar = load_data()
    dict_plot = {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}}

    Path("panels").mkdir(exist_ok=True)

    # Plot species richness at resolution 1km
    fig, ax = plt.subplots()
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}
    name = "SR_raster_100m"
    rast_sr_100m = rast_dict[name]
    # reducing the size of the raster for faster plotting
    rast_sr_100m = rast_sr_100m.coarsen(x=10, y=10, boundary="trim").mean()
    # rast = preprocess_raster(rast)
    # norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast_sr_100m, 
                cmap="viridis", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = 100m$^2$')
    fig.tight_layout()
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot species richness at resolution 10km
    fig, ax = plt.subplots()
    cbar_kwargs['location'] = 'right'
    name = "SR_raster_10000m"
    rast_sr_10000m = rast_dict[name]
    # rast = preprocess_raster(rast)
    # norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast_sr_10000m, 
                cmap="viridis", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = 10km$^2$')
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 100m
    fig, ax = plt.subplots()
    cbar_kwargs['label'] = 'dlogSR/dlogA'
    cbar_kwargs['location'] = 'left'
    name = "dlogSR_dlogA_raster_100m"
    rast_dsr_100m = np.maximum(0.,rast_dict[name])
    # Filter to retain only positive values
    rast_dsr_100m = rast_dsr_100m.coarsen(x=10, y=10, boundary="trim").mean()
    rast_dsr_100m = rast_dsr_100m.where(rast_sr_100m > 0)

    # rast = preprocess_raster(rast)
    plot_raster(ax, 
                rast_dsr_100m, 
                cmap="plasma", 
                cbar_kwargs=cbar_kwargs, 
                vmax=rast_dsr_100m.quantile(0.9)
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 10km
    fig, ax = plt.subplots()
    cbar_kwargs['location'] = 'right'
    name = "dlogSR_dlogA_raster_10000m"
    rast_dsr_10000m = np.maximum(0.,rast_dict[name])
    rast_dsr_10000m = rast_dsr_10000m.where(rast_sr_10000m > 0)
    plot_raster(ax, 
                rast_dsr_10000m, 
                cmap="plasma", 
                cbar_kwargs=cbar_kwargs, 
                vmax=rast_dsr_10000m.quantile(0.9)
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot SAR data on central plot
    fig, ax = plt.subplots()
    area = np.exp(dict_sar["log_area"])
    plot_sar(ax, dict_sar, dict_plot, area)
    fig.tight_layout()
    fig.savefig("panels/SAR.png", dpi=300, transparent=True)
    fig.savefig("panels/SAR.svg", dpi=300, transparent=True)

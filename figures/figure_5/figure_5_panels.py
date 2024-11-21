import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

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
HASH = "71f9fc7"

results_path = Path(f"./results/MLP_project_simple_full_grad_ensemble/" + "MLP_projections_rasters_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")

# Constants for file paths
SR_DSR_RAST_DICT_PATH = Path("../../../scripts/MLP3/results/MLP_project_simple_full_grad_ensemble/MLP_projections_rasters_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")
SAR_DICT_PATH = Path(f"../../../scripts/MLP3/results/true_SAR/true_SAR_ensemble_seed_{seed}_model_{MODEL}_hash_{HASH}.pkl")


def load_data(sr_dsr_rast_dict_path, sar_dict_path):
    """Load data from pickle files."""
    with open(sr_dsr_rast_dict_path, 'rb') as pickle_file:
        sr_dsr_rast_dict = pickle.load(pickle_file)["SR_dSR_rast_dict"]
    with open(sar_dict_path, 'rb') as pickle_file:
        dict_sar = pickle.load(pickle_file)["dict_SAR"]
    return sr_dsr_rast_dict, dict_sar

def preprocess_raster(rast, coarsen_factor=0, rolling_window=2):
    """Preprocess raster data by coarsening and smoothing."""
    if coarsen_factor > 0:
        rast = rast.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean() 
    if rolling_window > 0:
        rast = rast.rolling(x=rolling_window, y=rolling_window, 
                            center=False, 
                            # min_periods=4
                            ).mean()
    return rast

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
        ax.plot(area, np.exp(sar_data["log_SR_median"]), c=color)
        ax.fill_between(
            area,
            np.exp(sar_data["log_SR_first_quantile"]),
            np.exp(sar_data["log_SR_third_quantile"]),
            color=color,
            alpha=0.4,
            linewidth=0.3
        )
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

        col = ax.plot(x, y, color=color, linewidth=2)
        for c in col:
            c.set_rasterized(True)
    

if __name__ == '__main__':
    # Load data
    sr_dsr_rast_dict, dict_sar = load_data(SR_DSR_RAST_DICT_PATH, SAR_DICT_PATH)
    dict_plot = {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}}

    # Plot species richness at resolution 1km
    fig, ax = plt.subplots()
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}
    rast = np.exp(sr_dsr_rast_dict["1e+03"]["log_SR"])
    rast = preprocess_raster(rast)
    norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap="BuGn", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = $2.5 \cdot 10^5$ m2')
    fig.tight_layout()
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.savefig("panels/log_SR_fine.pdf", dpi=300, transparent=True)
    fig.savefig("panels/log_SR_fine.png", dpi=300, transparent=True)
    # fig.savefig("panels/log_SR_fine.svg", dpi=300, transparent=True)

    # Plot species richness at resolution 10km
    fig, ax = plt.subplots()
    cbar_kwargs['location'] = 'right'
    rast = np.exp(sr_dsr_rast_dict["1e+04"]["log_SR"])
    rast = preprocess_raster(rast)
    norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap="BuGn", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = $2.5 \cdot 10^7$ m2')
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig("panels/log_SR_coarse.pdf", dpi=300, transparent=True)
    # fig.savefig("panels/log_SR_coarse.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 1km
    fig, ax = plt.subplots()
    cbar_kwargs['label'] = 'dlogSR/dlogA'
    cbar_kwargs['location'] = 'left'
    rast = np.maximum(0., sr_dsr_rast_dict["1e+03"]["dlogSR_dlogA"])
    rast = preprocess_raster(rast)
    plot_raster(ax, 
                rast, 
                cmap="OrRd", 
                cbar_kwargs=cbar_kwargs, 
                # vmax=1.
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig("panels/dlog_SR_fine.pdf", dpi=300, transparent=True)
    fig.savefig("panels/dlog_SR_fine.png", dpi=300, transparent=True)
    # fig.savefig("panels/dlog_SR_fine.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 10km
    fig, ax = plt.subplots()
    cbar_kwargs['location'] = 'right'
    rast = np.maximum(0., sr_dsr_rast_dict["1e+04"]["dlogSR_dlogA"])
    rast = preprocess_raster(rast)
    plot_raster(ax, 
                rast, 
                cmap="OrRd", 
                cbar_kwargs=cbar_kwargs, 
                vmax=0.8
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot)
    fig.tight_layout()
    fig.savefig("panels/dlog_SR_coarse.pdf", dpi=300, transparent=True)
    fig.savefig("panels/dlog_SR_coarse.png", dpi=300, transparent=True)
    # fig.savefig("panels/dlog_SR_coarse.svg", dpi=300, transparent=True)

    # Plot SAR data on central plot
    # TODO: to fix, it seems that this is not the right plot
    fig, ax = plt.subplots()
    area = np.exp(dict_sar["log_area"])
    plot_sar(ax, dict_sar, dict_plot, area)
    fig.tight_layout()
    fig.savefig("panels/SAR.png", dpi=300, transparent=True)
    # fig.savefig("panels/SAR.svg", dpi=300, transparent=True)

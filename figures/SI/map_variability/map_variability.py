import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

# Constants for file paths
SR_DSR_RAST_DICT_PATH = Path("../../../scripts/MLP3/results/MLP_project_simple_full_grad_ensemble/MLP_project_simple_full_grad_ensemble.pkl")

def load_data(sr_dsr_rast_dict_path):
    """Load data from pickle files."""
    with open(sr_dsr_rast_dict_path, 'rb') as pickle_file:
        sr_dsr_rast_dict = pickle.load(pickle_file)["SR_dSR_rast_dict"]
    return sr_dsr_rast_dict

def preprocess_raster(rast, coarsen_factor=0, rolling_window=0):
    """Preprocess raster data by coarsening and smoothing."""
    if coarsen_factor > 0:
        rast = rast.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean() 
    if rolling_window > 0:
        rast = rast.rolling(x=rolling_window, y=rolling_window, center=True, min_periods=4).mean()
    return rast

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, title='', **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title(title)
    ax.set_axis_off()

if __name__ == '__main__':
    # Load data
    sr_dsr_rast_dict = load_data(SR_DSR_RAST_DICT_PATH)
    dict_plot = {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}}

    # Plot species richness at resolution 0.5km
    fig, ax = plt.subplots()
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}
    rast = np.exp(sr_dsr_rast_dict["5e+02"]["std_log_SR"])
    rast = preprocess_raster(rast)
    norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap="BuGn", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = $2.5 \cdot 10^5$ m2')
    fig.tight_layout()
    fig.savefig("panels/std_log_SR_5e02.pdf", dpi=300, transparent=True)

    # Plot species richness at resolution 5km
    fig, ax = plt.subplots()
    cbar_kwargs['location'] = 'right'
    rast = np.exp(sr_dsr_rast_dict["5e+03"]["std_log_SR"])
    rast = preprocess_raster(rast)
    norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap="BuGn", 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                title='Area = $2.5 \cdot 10^7$ m2')
    fig.tight_layout()
    fig.savefig("panels/std_log_SR_5e03.pdf", dpi=300, transparent=True)
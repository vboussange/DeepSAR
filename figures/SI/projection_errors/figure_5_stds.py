import matplotlib.pyplot as plt
import pickle
import xarray as xr
import rioxarray
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from scipy import stats
from deepsar.plotting import CMAP_BR, CMAP_DSR
from matplotlib import colors
import seaborn as sns
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
HASH = "a53390d"

# Constants for file paths
RAST_PATH = Path(f"../../figure_5/projections")

def load_data(rast_path=RAST_PATH):
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

    return rast_dict

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

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title('')
    ax.set_axis_off()

if __name__ == '__main__':
    # Load data
    rast_dict = load_data()
    # Download higher resolution Natural Earth data
    world = gpd.read_file("../../../data/raw/NaturalEarth/ne_10m_admin_0_countries.shp")
    europe = world[world.CONTINENT == 'Europe'].to_crs('EPSG:3035')
    europe_geom = europe.geometry
    
    dict_plot = {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}}
    Path("panels").mkdir(exist_ok=True)

    # Load required rasters
    rolling_kwargs = {'x': 2, 'y': 2, 'center': False, 'min_periods': 2}
    
    # Define raster processing parameters
    raster_configs = [
        ('sr_1000', "SR_raster_1000m"),
        ('sr_50000', "SR_raster_50000m"),
        ('std_sr_1000', "std_SR_raster_1000m"),
        ('std_sr_50000', "std_SR_raster_50000m"),
    ]
    
    rasters = {}
    for key, filename in raster_configs:
        if filename not in rast_dict:
            print(f"Warning: {filename} not found in raster data")
            continue
            
        raster = (rast_dict[filename]
                 .rolling(**rolling_kwargs)
                 .mean()
                 .rio.clip(europe_geom, drop=True)
                 .rio.write_crs("EPSG:3035")
                 )
        rasters[key] = raster
    
    # Coarsen rasters for faster plotting
    for key in rasters:
        if '1000' in key:
            factor = 5
            rasters[key] = rasters[key].coarsen(x=factor, y=factor, boundary="trim").mean()

    # Plot standard deviations
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.4, 'aspect': 40,
                   'label': 'Standard deviation', 'pad': 0.05}

    fig, (ax_std_sr1, ax_std_sr2) = plt.subplots(1, 2, figsize=(10, 6))
    # Compute and plot relative standard deviations (RSD)
    cbar_kwargs['label'] = 'Relative std. of\npredicted SR (%)'

    # RSD of SR at 1000m resolution
    name = "std_sr_1000"
    rast = rasters[name]
    mean_rast = rast_dict["SR_raster_1000m"]
    rsd_rast = (rast / mean_rast) * 100
    cbar_kwargs['location'] = 'left'
    plot_raster(ax_std_sr1, 
                rsd_rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rsd_rast.quantile(0.01), 
                vmax=rsd_rast.quantile(0.99))

    ax_std_sr1.set_aspect('equal')

    # RSD of SR at 50000m resolution
    name = "std_sr_50000"
    rast = rasters[name]
    mean_rast = rast_dict["SR_raster_50000m"]
    rsd_rast = (rast / mean_rast) * 100
    cbar_kwargs['location'] = 'right'
    plot_raster(ax_std_sr2, 
                rsd_rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rsd_rast.quantile(0.01), 
                vmax=rsd_rast.quantile(0.99))

    ax_std_sr2.set_aspect('equal')

    ax_std_sr1.set_title('Area = 1 km$^2$', y=-0.3)
    ax_std_sr2.set_title('Area = 2500 km$^2$', y=-0.3)


    # Save the figure
    fig.savefig("figure_rstd.pdf", dpi=300, transparent=True)

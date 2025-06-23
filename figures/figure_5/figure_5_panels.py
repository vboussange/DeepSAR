import matplotlib.pyplot as plt
import pickle
import xarray as xr
import rioxarray
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from scipy import stats
from src.plotting import CMAP_BR, CMAP_DSR
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
RAST_PATH = Path(f"./projections")
SAR_PATH = Path(f"./SARs")


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
    with open(sar_path / "SARs.pkl", 'rb') as pickle_file:
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

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title('')
    ax.set_axis_off()

def plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=100000):
    """Plot bounding boxes on corner plots."""
    for loc, loc_info in dict_plot.items():
        sar_data = dict_sar[loc]
        color = loc_info['c']

        x_centroid, y_centroid = sar_data['coords_epsg_3035']
        minx_proj = x_centroid - buffer_size_meters
        maxx_proj = x_centroid + buffer_size_meters
        miny_proj = y_centroid - buffer_size_meters
        maxy_proj = y_centroid + buffer_size_meters
        bbox_proj = box(minx_proj, miny_proj, maxx_proj, maxy_proj)
        x, y = bbox_proj.exterior.xy

        # Plot the bbox with white border first, then the colored line
        col = []
        col.append(ax.plot(x, y, color='white', linewidth=4, alpha=0.8)[0])  # Wider white border
        col.append(ax.plot(x, y, color=color, linewidth=2)[0])    # Colored line on top
    

if __name__ == '__main__':
    # Load data
    rast_dict, dict_sar = load_data()
    # Download higher resolution Natural Earth data
    world = gpd.read_file("../../data/raw/NaturalEarth/ne_10m_admin_0_countries.shp")
    europe = world[world.CONTINENT == 'Europe'].to_crs('EPSG:3035')
    europe_geom = europe.geometry
    
    dict_plot = {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}}
    Path("panels").mkdir(exist_ok=True)

    # Load required rasters
    kwargs = {'x': 2, 'y': 2, 'center': False, 'min_periods': 2}
    # Define raster processing parameters
    rolling_kwargs = {'x': 2, 'y': 2, 'center': False, 'min_periods': 2}
    
    # Load and process rasters
    raster_configs = [
        ('sr_1000', "SR_raster_1000m"),
        ('sr_50000', "SR_raster_50000m"),
        ('dsr_1000', "dSR_dlogA_raster_1000m"),
        ('dsr_50000', "dSR_dlogA_raster_50000m")
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

    # Plot species richness at resolution 1km
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}
    
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[0.4, 1, 1])

    # Top row: SAR curves (3 panels, each spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    area = np.exp(dict_sar["log_area"])

    # Plot each location on a separate axis
    locations = list(dict_plot.keys())
    axes = [ax1, ax2, ax3]
        
    # Add labels under the vertical lines
    labels = ['A', 'B', 'C']
    
    for i, (loc, ax) in enumerate(zip(locations, axes)):
        loc_info = dict_plot[loc]
        sar_data = dict_sar[loc]
        color = loc_info['c']
        
        # Calculate median and quantiles
        median_sr = np.median(sar_data["SRs"], axis=1)
        q05_sr = np.quantile(sar_data["SRs"], 0.05, axis=1)
        q95_sr = np.quantile(sar_data["SRs"], 0.95, axis=1)
        
        # Apply rolling window for smoothness
        window_size = 10
        median_sr_smooth = np.convolve(median_sr, np.ones(window_size)/window_size, mode='valid')
        q05_sr_smooth = np.convolve(q05_sr, np.ones(window_size)/window_size, mode='valid')
        q95_sr_smooth = np.convolve(q95_sr, np.ones(window_size)/window_size, mode='valid')
        area_smooth = area[window_size-1:] / 1e6  # Convert area to km² for plotting
        
        ax.plot(area_smooth, median_sr_smooth,
                color=color, 
                linewidth=2)
        ax.fill_between(area_smooth,
                        q05_sr_smooth,
                        q95_sr_smooth,
                        color=color, alpha=0.2)
        
        # Add vertical lines
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=2.5e3, color='gray', linestyle='--', alpha=0.7)

        ax.text(1, 2200, f'{labels[i]}1', ha='center', va='top', fontsize=10, weight='bold')
        ax.text(2.5e3, 2200, f'{labels[i]}2', ha='center', va='top', fontsize=10, weight='bold')
        
        ax.set_xscale('log')
        
        if i == 0:
            ax.set_ylabel("Species richness")
        if i == 1:
            ax.set_xlabel("Area (km$^2$)")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(500, 2000)
        
        # Remove xtick labels at both ends
        xticks = ax.get_xticks()
        xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
        if len(xlabels) > 2:
            xlabels[0:2] = ['', '']  # Remove first two labels
            xlabels[-2:] = ['', '']  # Remove last two labels
            ax.set_xticklabels(xlabels)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        if i > 0:
            ax.set_yticklabels([])

    # Second row: Species richness maps (2 panels, each spanning 3 columns)
    ax_sr1 = fig.add_subplot(gs[1, 0:3])
    ax_sr2 = fig.add_subplot(gs[1, 3:6])

    # SR at 1000m resolution
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}

    name = "sr_1000"
    rast = rasters[name]
    plot_raster(ax_sr1, 
                rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99))
    plot_bounding_boxes(ax_sr1, dict_sar, dict_plot, buffer_size_meters=20000)
    ax_sr1.set_aspect('equal')

    # SR at 50000m resolution
    name = "sr_50000"
    rast = rasters[name]
    cbar_kwargs['location'] = 'right'
    plot_raster(ax_sr2, 
                rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99))
    plot_bounding_boxes(ax_sr2, dict_sar, dict_plot, buffer_size_meters=50000)
    ax_sr2.set_aspect('equal')

    # Third row: dSR maps (2 panels, each spanning 3 columns)
    ax_dsr1 = fig.add_subplot(gs[2, 0:3])
    ax_dsr2 = fig.add_subplot(gs[2, 3:6])

    cbar_kwargs['label'] = 'Sensitivity of SR\nto area ($\\frac{d S}{d A}$)'
    cbar_kwargs['location'] = 'left'

    # dSR at 1000m resolution
    name = "dsr_1000"
    rast = np.maximum(0., rasters[name])
    rast = rast.where(rasters[name[1:]] > 0)
    cbar_kwargs['location'] = 'left'
    plot_raster(ax_dsr1, 
                rast, 
                cmap=CMAP_DSR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99))
    ax_dsr1.set_aspect('equal')
    ax_dsr1.set_title('Area = 1 km$^2$', y=-0.1)

    # dSR at 50000m resolution
    name = "dsr_50000"
    rast = np.maximum(0., rasters[name])
    rast = rast.where(rasters[name[1:]] > 0)
    cbar_kwargs['location'] = 'right'
    plot_raster(ax_dsr2, 
                rast, 
                cmap=CMAP_DSR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99))
    ax_dsr2.set_aspect('equal')
    ax_dsr2.set_title('Area = 2500 km$^2$', y=-0.1)

    # fig.tight_layout()
    fig.savefig("figure_5_combined.pdf", dpi=300, transparent=True)
    fig.savefig("figure_5_combined.png", dpi=300, transparent=True)
    fig.savefig("figure_5_combined.svg", dpi=300, transparent=True)

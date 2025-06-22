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

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, title='', **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title(title)
    ax.set_axis_off()

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
        col.append(ax.plot(x, y, color='white', linewidth=4, alpha=0.8)[0])  # Wider white border
        col.append(ax.plot(x, y, color=color, linewidth=2)[0])    # Colored line on top
        for c in col:
            c.set_rasterized(True)
    

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
    rasters = {
        'sr_1000': rast_dict["SR_raster_1000m"].rolling(**kwargs).mean().rio.clip(europe_geom, drop=True),
        'sr_50000': rast_dict["SR_raster_50000m"].rolling(**kwargs).mean().rio.clip(europe_geom, drop=True), 
        'dsr_1000': rast_dict["dSR_dlogA_raster_1000m"].rolling(**kwargs).mean().rio.clip(europe_geom, drop=True),
        'dsr_50000': rast_dict["dSR_dlogA_raster_50000m"].rolling(**kwargs).mean().rio.clip(europe_geom, drop=True)
    }
    
    # Coarsen rasters for faster plotting
    for key in rasters:
        if '1000' in key:
            factor = 5
            rasters[key] = rasters[key].coarsen(x=factor, y=factor, boundary="trim").mean()

    # Plot species richness at resolution 1km
    cbar_kwargs = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                   'label': 'Species richness', 'pad': 0.05, 'location': 'left'}
    
    fig, ax = plt.subplots()
    
    name = "sr_1000"
    rast = rasters[name]
    # reducing the size of the raster for faster plotting

    # rast = preprocess_raster(rast)
    # norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99),
                title='Area = $10^6$m$^2$')
    fig.tight_layout()
    plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=20000)
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot species richness at resolution 50km
    fig, ax = plt.subplots()

    name = "sr_50000"
    rast = rasters[name]

    # rast = preprocess_raster(rast)
    # norm = colors.LogNorm(vmin=rast.min().item(), vmax=rast.max().item())
    plot_raster(ax, 
                rast, 
                cmap=CMAP_BR, 
                cbar_kwargs=cbar_kwargs, 
                # norm=norm, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99),
                title='Area = $25\cdot 10^8$m$^2$')
    fig.tight_layout()
    plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=50000)
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 1000m
    fig, ax = plt.subplots()
    cbar_kwargs['label'] = 'dlogSR/dlogA'
    cbar_kwargs['location'] = 'left'
    name = "dsr_1000"
    rast = np.maximum(0.,rasters[name])
    rast = rast.where(rasters[name[1:]] > 0)

    # rast = preprocess_raster(rast)
    plot_raster(ax, 
                rast, 
                cmap=CMAP_DSR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99),
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=20000)
    fig.tight_layout()
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)

    # Plot dlogSR/dlogA at resolution 50km
    fig, ax = plt.subplots()
    cbar_kwargs['label'] = 'dlogSR/dlogA'
    cbar_kwargs['location'] = 'left'
    name = "dsr_50000"
    rast = np.maximum(0.,rasters[name])
    rast = rast.where(rasters[name[1:]] > 0)

    # rast = preprocess_raster(rast)
    plot_raster(ax, 
                rast, 
                cmap=CMAP_DSR, 
                cbar_kwargs=cbar_kwargs, 
                vmin=rast.quantile(0.01), 
                vmax=rast.quantile(0.99),
                )
    plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=50000)
    fig.tight_layout()
    fig.savefig(f"panels/{name}.pdf", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.png", dpi=300, transparent=True)
    fig.savefig(f"panels/{name}.svg", dpi=300, transparent=True)
    
    
    
    # Plot SAR curves on three different axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3), sharey=True, sharex=True)
    area = np.exp(dict_sar["log_area"])

    # Plot each location on a separate axis
    locations = list(dict_plot.keys())
    axes = [ax1, ax2, ax3]

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
        area_smooth = area[window_size-1:]
        
        ax.plot(area_smooth, median_sr_smooth,
                color=color, 
                linewidth=2, 
                # marker='o', 
                # markersize=3
                )
        ax.fill_between(area_smooth,
                        q05_sr_smooth,
                        q95_sr_smooth,
                        color=color, alpha=0.2)
        
        # Add vertical lines
        ax.axvline(x=1e6, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=25e8, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xscale('log')
        # ax.set_yscale('log')
        
        if i == 0:
            ax.set_ylabel("SR")
        if i == 1:
            ax.set_xlabel("Area")
        ax.set_xlim(1e5, 1e10)
        ax.set_ylim(500, 2000)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig("panels/sar_curves.pdf", dpi=300, transparent=True)
    fig.savefig("panels/sar_curves.png", dpi=300, transparent=True)
    fig.savefig("panels/sar_curves.svg", dpi=300, transparent=True)



    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
    area = np.exp(dict_sar["log_area"])
    
    # Row 1: Three panels for correlations and SAR (2 grid cells each)
    ax1 = fig.add_subplot(gs[0, 0:2])  # Left correlation
    ax2 = fig.add_subplot(gs[0, 2:4])  # Middle SAR
    ax3 = fig.add_subplot(gs[0, 4:6])  # Right correlation
    
    # Left: Correlation dSR vs SR (1000m)
    valid_mask = (rasters['sr_1000'] > 0) & (rasters['dsr_1000'] > 0)
    sr_flat = rasters['sr_1000'].where(valid_mask).values.flatten()
    dsr_flat = rasters['dsr_1000'].where(valid_mask).values.flatten()
    valid_idx = ~(np.isnan(sr_flat) | np.isnan(dsr_flat))
    
    ax1.scatter(sr_flat[valid_idx], dsr_flat[valid_idx], alpha=0.5, s=1)
    
    # Calculate and plot linear trend
    # Calculate correlation coefficient
    correlation = np.corrcoef(np.log10(sr_flat[valid_idx]), np.log10(dsr_flat[valid_idx]))[0, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(sr_flat[valid_idx]), np.log10(dsr_flat[valid_idx]))
    x_trend = np.logspace(np.log10(sr_flat[valid_idx].min()), np.log10(sr_flat[valid_idx].max()), 100)
    y_trend = 10**(slope * np.log10(x_trend) + intercept)
    ax1.plot(x_trend, y_trend, 'r-', linewidth=2, label=f'Linear fit (r={r_value:.3f})')
    
    
    ax1.set_xlabel('Species Richness (1km)')
    ax1.set_ylabel('dlogSR/dlogA')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('Correlation: 1km resolution')
    ax1.legend()
    
    # Middle: SAR curves
    plot_sar(ax2, dict_sar, dict_plot, area)
    ax2.set_title('Species-Area Relationships')
    
    # Right: Correlation dSR vs SR (50000m)
    valid_mask = (rasters['sr_50000'] > 0) & (rasters['dsr_50000'] > 0)
    sr_flat = rasters['sr_50000'].where(valid_mask).values.flatten()
    dsr_flat = rasters['dsr_50000'].where(valid_mask).values.flatten()
    valid_idx = ~(np.isnan(sr_flat) | np.isnan(dsr_flat))
    correlation = np.corrcoef(np.log10(sr_flat[valid_idx]), np.log10(dsr_flat[valid_idx]))[0, 1]

    ax3.scatter(sr_flat[valid_idx], dsr_flat[valid_idx], alpha=0.5, s=1)
    ax3.set_xlabel('Species Richness (50km)')
    ax3.set_ylabel('dlogSR/dlogA')
    ax3.set_title('Correlation: 50km resolution')
    
    # Row 2: Species richness rasters (3 grid cells each)
    ax4 = fig.add_subplot(gs[1, 0:3])  # SR 1km
    ax5 = fig.add_subplot(gs[1, 3:6])  # SR 50km
    
    cbar_kwargs_sr = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                      'label': 'Species richness', 'pad': 0.05}
    
    plot_raster(ax4, rasters['sr_1000'], cmap="viridis", 
                cbar_kwargs=cbar_kwargs_sr, title='SR: 1km²')
    plot_bounding_boxes(ax4, dict_sar, dict_plot, buffer_size_meters=50000)
    
    plot_raster(ax5, rasters['sr_50000'], cmap="viridis", 
                cbar_kwargs=cbar_kwargs_sr, title='SR: 50km²')
    plot_bounding_boxes(ax5, dict_sar, dict_plot, buffer_size_meters=200000)
    
    # Row 3: dlogSR/dlogA rasters (3 grid cells each)
    ax6 = fig.add_subplot(gs[2, 0:3])  # dSR 1km
    ax7 = fig.add_subplot(gs[2, 3:6])  # dSR 50km
    
    cbar_kwargs_dsr = {'orientation': 'vertical', 'shrink': 0.6, 'aspect': 40,
                       'label': 'dlogSR/dlogA', 'pad': 0.05}
    
    dsr_1000_filtered = np.maximum(0., rasters['dsr_1000']).where(rasters['sr_1000'] > 0)
    plot_raster(ax6, dsr_1000_filtered, cmap="plasma", 
                cbar_kwargs=cbar_kwargs_dsr, vmax=dsr_1000_filtered.quantile(0.9),
                title='dlogSR/dlogA: 1km²')
    plot_bounding_boxes(ax6, dict_sar, dict_plot, buffer_size_meters=50000)
    
    dsr_50000_filtered = np.maximum(0., rasters['dsr_50000']).where(rasters['sr_50000'] > 0)
    plot_raster(ax7, dsr_50000_filtered, cmap="plasma", 
                cbar_kwargs=cbar_kwargs_dsr, vmax=dsr_50000_filtered.quantile(0.9),
                title='dlogSR/dlogA: 50km²')
    plot_bounding_boxes(ax7, dict_sar, dict_plot, buffer_size_meters=200000)
    
    fig.savefig("panels/figure_5_complete.pdf", dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig("panels/figure_5_complete.png", dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig("panels/figure_5_complete.svg", dpi=300, bbox_inches='tight', transparent=True)

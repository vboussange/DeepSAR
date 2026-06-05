"""Generates figure 5 panels; the figure is further processed in Inkscape."""


import matplotlib.pyplot as plt
import pickle
import rioxarray
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from muscari.plotting import CMAP_BR, CMAP_DSR
from matplotlib.gridspec import GridSpec
import xarray as xr

rcparams = {
    "font.size": 9,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 16,
    "lines.markersize": 3,
}
plt.rcParams.update(rcparams)

ROOT = Path(__file__).parents[2]
TRAINING_DATASET_SEED = "ceacce0"

CONFIG = {
    "model_name": f"{TRAINING_DATASET_SEED}",
    "sar_path": Path(__file__).parent / "SARs",
    "panels_dir": Path(__file__).parent / "panels",
    "rolling_kwargs": {"x": 2, "y": 2, "center": False, "min_periods": 2},
    "sr_panels": [
        {"key": "sr_5000", "res_m": 5000, "buffer": 20000, "cbar_loc": "left"},
        {"key": "sr_50000", "res_m": 50000, "buffer": 50000, "cbar_loc": "right"},
    ],
    "dsr_panels": [
        {"key": "dsr_5000", "res_m": 5000, "cbar_loc": "left", "title": "Area = 25 km$^2$"},
        {"key": "dsr_50000", "res_m": 50000, "cbar_loc": "right", "title": "Area = 2500 km$^2$"},
    ],
    "dict_plot": {"loc1": {"c": "tab:blue"}, "loc2": {"c": "tab:red"}, "loc3": {"c": "tab:purple"}},
    "figure_size": (8, 10),
    "cbar_base": {
        "orientation": "vertical",
        "shrink": 0.6,
        "aspect": 40,
        "pad": 0.05,
    },
    "sar_window": 10,
    "sar_xlim": (2e3**2/1e6, 2e5**2/1e6), # km2
    "sar_ylim": (500, 2200),
    "sar_vlines": [5e3**2/1e6, 5e4**2/1e6],
    "sar_labels": ["A", "B", "C"],
}

RAST_PATH = ROOT / "data" / "processed" / "projections" / CONFIG["model_name"]


def load_data(rast_path=None, sar_path=None):
    """Load rasters and SAR curves from disk."""
    rast_path = rast_path or RAST_PATH
    sar_path = sar_path or CONFIG["sar_path"]
    rast_dict = {}
    for file_path in rast_path.glob("*.tif"):
        raster_data = rioxarray.open_rasterio(file_path).squeeze(drop=True)
        rast_dict[file_path.stem] = raster_data

    with open(sar_path / "SARs.pkl", "rb") as pickle_file:
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


def require_raster(rasters, name):
    if name not in rasters:
        raise FileNotFoundError(f"Missing raster: {name}")
    return rasters[name]


def raster_name(prefix: str, res_m: int) -> str:
    return f"{prefix}_raster_{CONFIG['model_name']}_{int(res_m)}m"

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
        ax.plot(x, y, color="white", linewidth=4, alpha=0.8)  # Wider white border
        ax.plot(x, y, color=color, linewidth=2)  # Colored line on top
    

if __name__ == "__main__":
    # Load data
    rast_dict, dict_sar = load_data()
    # Download higher resolution Natural Earth data
    world = gpd.read_file(ROOT / "data" / "raw" / "NaturalEarth" / "ne_10m_admin_0_countries.shp")
    europe = world[world.CONTINENT == "Europe"].to_crs("EPSG:3035")
    europe_geom = europe.geometry
    
    dict_plot = CONFIG["dict_plot"]
    CONFIG["panels_dir"].mkdir(parents=True, exist_ok=True)

    # Define raster processing parameters
    rolling_kwargs = CONFIG["rolling_kwargs"]
    
    # Load and process rasters
    raster_configs = [
        (panel["key"], raster_name("SR", panel["res_m"]))
        for panel in CONFIG["sr_panels"]
    ] + [
        (panel["key"], raster_name("dSR_dlogA", panel["res_m"]))
        for panel in CONFIG["dsr_panels"]
    ]
    
    rasters = {}
    for key, filename in raster_configs:
        if filename not in rast_dict:
            print(f"Warning: {filename} not found in raster data")
            continue

        raster = (
            rast_dict[filename]
            .rolling(**rolling_kwargs)
            .mean()
            .rio.clip(europe_geom, drop=True)
            .rio.write_crs("EPSG:3035")
        )
        rasters[key] = raster
    
    # Coarsen rasters for faster plotting
    # for key in rasters:
    #     if '5000' in key:
    #         factor = 5
    #         rasters[key] = rasters[key].coarsen(x=factor, y=factor, boundary="trim").mean()

    # Plot species richness at resolution 5km
    cbar_kwargs = {**CONFIG["cbar_base"], "label": "Predicted species\nrichness ($S_T$)", "location": "left"}
    
    fig = plt.figure(figsize=CONFIG["figure_size"])
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
    labels = CONFIG["sar_labels"]
    
    for i, (loc, ax) in enumerate(zip(locations, axes)):
        loc_info = dict_plot[loc]
        sar_data = dict_sar[loc]
        color = loc_info['c']
        
        central_sr = np.asarray(sar_data["SR"])
        q05_sr = central_sr - np.asarray(sar_data["std_SR"])
        q95_sr = central_sr + np.asarray(sar_data["std_SR"])
    
        # Apply rolling window for smoothness
        window_size = CONFIG["sar_window"]
        kernel = np.ones(window_size) / window_size
        central_sr_smooth = np.convolve(central_sr, kernel, mode="valid")
        q05_sr_smooth = np.convolve(q05_sr, kernel, mode="valid")
        q95_sr_smooth = np.convolve(q95_sr, kernel, mode="valid")
        area_smooth = area[window_size-1:] / 1e6  # Convert area to km² for plotting
        
        ax.plot(area_smooth, central_sr_smooth, color=color, linewidth=2)
        ax.fill_between(area_smooth, q05_sr_smooth, q95_sr_smooth, color=color, alpha=0.2)
        
        # Add vertical lines
        for xline in CONFIG["sar_vlines"]:
            ax.axvline(x=xline, color="gray", linestyle="--", alpha=0.7)

        ax.text(CONFIG["sar_vlines"][0], 2500, f"{labels[i]}1", ha="center", va="top", fontsize=10, weight="bold")
        ax.text(CONFIG["sar_vlines"][1], 2500, f"{labels[i]}2", ha="center", va="top", fontsize=10, weight="bold")
        
        ax.set_xscale('log')
        
        if i == 0:
            ax.set_ylabel("Predicted species\nrichness ($S_T$)")
        if i == 1:
            ax.set_xlabel("Area (km$^2$)")
        ax.set_xlim(*CONFIG["sar_xlim"])
        ax.set_ylim(*CONFIG["sar_ylim"])
        
        # Remove xtick labels at both ends
        xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
        if len(xlabels) > 2:
            xlabels[0:2] = ["", ""]  # Remove first two labels
            xlabels[-2:] = ["", ""]  # Remove last two labels
            ax.set_xticklabels(xlabels)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        if i > 0:
            ax.set_yticklabels([])

    # Second row: Species richness maps (2 panels, each spanning 3 columns)
    ax_sr1 = fig.add_subplot(gs[1, 0:3])
    ax_sr2 = fig.add_subplot(gs[1, 3:6])

    cbar_kwargs = {**CONFIG["cbar_base"], "label": "Predicted species\nrichness ($S_T$)", "location": "left"}

    sr_axes = [ax_sr1, ax_sr2]
    for ax, panel in zip(sr_axes, CONFIG["sr_panels"]):
        rast = require_raster(rasters, panel["key"])
        cbar_kwargs["location"] = panel["cbar_loc"]
        plot_raster(
            ax,
            rast,
            cmap=CMAP_BR,
            cbar_kwargs=cbar_kwargs,
            vmin=rast.quantile(0.01),
            vmax=rast.quantile(0.99),
        )
        plot_bounding_boxes(ax, dict_sar, dict_plot, buffer_size_meters=panel["buffer"])
        ax.set_aspect("equal")

    # Third row: dSR maps (2 panels, each spanning 3 columns)
    ax_dsr1 = fig.add_subplot(gs[2, 0:3])
    ax_dsr2 = fig.add_subplot(gs[2, 3:6])

    cbar_kwargs["label"] = "Species accumulation\nrate ($\\frac{d S}{d A}$)"

    dsr_axes = [ax_dsr1, ax_dsr2]
    for ax, panel in zip(dsr_axes, CONFIG["dsr_panels"]):
        base_rast = require_raster(rasters, panel["key"])
        # ensure no negative values
        rast = xr.where((base_rast >= 0) | (~np.isfinite(base_rast)), base_rast, 0)
        cbar_kwargs["location"] = panel["cbar_loc"]
        plot_raster(
            ax,
            rast,
            cmap=CMAP_DSR,
            cbar_kwargs=cbar_kwargs,
            vmin=rast.quantile(0.01),
            vmax=rast.quantile(0.99),
        )
        ax.set_aspect("equal")
        ax.set_title(panel["title"], y=-0.1)

    # fig.tight_layout()
    fig.savefig("figure_5_panels.pdf", dpi=300, transparent=True)
    fig.savefig("figure_5_panels.png", dpi=300, transparent=True)
    fig.savefig("figure_5_panels.svg", dpi=300, transparent=True)

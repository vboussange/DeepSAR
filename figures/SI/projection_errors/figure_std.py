import matplotlib.pyplot as plt
import rioxarray
import numpy as np
import geopandas as gpd
from pathlib import Path
from muscari.plotting import CMAP_BR

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


MODEL_NAME = "ceacce0"
RESOLUTIONS_M = (5_000, 50_000)

# Constants for file paths
ROOT = Path(__file__).resolve().parents[3]
RAST_PATH = ROOT / "data" / "processed" / "projections" / MODEL_NAME

def load_data(rast_path=RAST_PATH):
    """Load all geotiff rasters in projection directory."""
    rast_dict = {}
    for file_path in rast_path.glob("*.tif"):
        name = file_path.stem
        raster_data = rioxarray.open_rasterio(file_path)
        rast_dict[name] = raster_data

    return rast_dict

def get_raster_name(raster_prefix, resolution_m):
    return f"{raster_prefix}_raster_{MODEL_NAME}_{int(resolution_m)}m"


def get_raster(rast_dict, raster_prefix, resolution_m):
    expected_name = get_raster_name(raster_prefix, resolution_m)
    if expected_name in rast_dict:
        return rast_dict[expected_name]

    candidates = [
        name for name in rast_dict
        if name.startswith(f"{raster_prefix}_raster_") and name.endswith(f"_{int(resolution_m)}m")
    ]
    if len(candidates) == 1:
        return rast_dict[candidates[0]]

    raise KeyError(
        f"Could not uniquely find raster for prefix={raster_prefix}, resolution={resolution_m}m. "
        f"Expected {expected_name}."
    )


def preprocess_raster(rast, europe_geom, rolling_kwargs, coarsen_factor=None):
    raster = (
        rast
        .rolling(**rolling_kwargs)
        .mean()
        .rio.clip(europe_geom, drop=True)
        .rio.write_crs("EPSG:3035")
    )
    if coarsen_factor is not None and coarsen_factor > 1:
        raster = raster.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()
    return raster

def plot_raster(ax, rast, cmap, cbar_kwargs, norm=None, **kwargs):
    """Plot raster data on a given axis."""
    rast.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm, **kwargs).set_rasterized(True)
    ax.set_title('')
    ax.set_axis_off()


def relative_std(std_rast, mean_rast, denominator_abs=False):
    denominator = np.abs(mean_rast) if denominator_abs else mean_rast
    return ((std_rast / denominator) * 100).where(np.abs(denominator) > 1e-12)


def area_label(resolution_m):
    area_km2 = (resolution_m / 1000) ** 2
    return f"{area_km2:g}"

if __name__ == '__main__':
    # Load data
    if not RAST_PATH.exists():
        raise FileNotFoundError(f"Projection folder does not exist: {RAST_PATH}")

    rast_dict = load_data()
    # Download higher resolution Natural Earth data
    world = gpd.read_file(ROOT / "data" / "raw" / "NaturalEarth" / "ne_10m_admin_0_countries.shp")
    europe = world[world.CONTINENT == 'Europe'].to_crs('EPSG:3035')
    europe_geom = europe.geometry

    Path("panels").mkdir(exist_ok=True)

    rolling_kwargs = {'x': 2, 'y': 2, 'center': False, 'min_periods': 2}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_sr_low, ax_sr_high = axes[0]
    ax_dsr_low, ax_dsr_high = axes[1]

    for idx, res in enumerate(RESOLUTIONS_M):
        coarsen_factor = 2 if res <= 5_000 else None

        sr_mean = preprocess_raster(get_raster(rast_dict, "SR", res), europe_geom, rolling_kwargs, coarsen_factor)
        sr_std = preprocess_raster(get_raster(rast_dict, "std_SR", res), europe_geom, rolling_kwargs, coarsen_factor)
        dsr_mean = preprocess_raster(get_raster(rast_dict, "dSR_dlogA", res), europe_geom, rolling_kwargs, coarsen_factor)
        dsr_std = preprocess_raster(get_raster(rast_dict, "std_dSR_dlogA", res), europe_geom, rolling_kwargs, coarsen_factor)

        std_sr = sr_std
        std_dsr = dsr_std

        sr_ax = ax_sr_low if idx == 0 else ax_sr_high
        dsr_ax = ax_dsr_low if idx == 0 else ax_dsr_high

        sr_cbar_kwargs = {
            'orientation': 'vertical',
            'shrink': 0.45,
            'aspect': 35,
            'label': 'Std. of\npredicted $S_T$',
            'pad': 0.04,
            'location': 'left' if idx == 0 else 'right',
        }
        dsr_cbar_kwargs = {
            'orientation': 'vertical',
            'shrink': 0.45,
            'aspect': 35,
            'label': 'Std. of species\nacc. rate $\\frac{dS_T}{d\\log A}$',
            'pad': 0.04,
            'location': 'left' if idx == 0 else 'right',
        }

        plot_raster(
            sr_ax,
            std_sr,
            cmap=CMAP_BR,
            cbar_kwargs=sr_cbar_kwargs,
            vmin=std_sr.quantile(0.01),
            vmax=std_sr.quantile(0.99),
        )
        plot_raster(
            dsr_ax,
            std_dsr,
            cmap=CMAP_BR,
            cbar_kwargs=dsr_cbar_kwargs,
            vmin=std_dsr.quantile(0.01),
            vmax=std_dsr.quantile(0.99),
        )

        sr_ax.set_aspect('equal')
        dsr_ax.set_aspect('equal')
        sr_ax.set_title(f'SR | Area = {area_label(res)} km$^2$', y=-0.22)
        dsr_ax.set_title(f'dSR/dlogA | Area = {area_label(res)} km$^2$', y=-0.22)

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.)
    fig.savefig("figure_rstd.pdf", dpi=300, transparent=True)

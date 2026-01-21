"""
Plot example spatial units (training samples) centered on Switzerland.
"""
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box, LineString
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.plotting import COLORS_BR

SBCV_PATH = Path(__file__).parents[3] / "data" / "processed" / "training_samples" / "sbcv" / "606e055"
SAMPLE_FILE = SBCV_PATH / "fold_0_train.parquet"
VAL_SAMPLE_FILE = SBCV_PATH / "fold_0_val.parquet"
TEST_SAMPLE_FILE = SBCV_PATH / "fold_0_test.parquet"
N_SAMPLES = 150
N_PLOTS = 100
BOX_HALF_SIZE_KM = 20
BLOCK_SIZE_M = 20_000

COUNTRY_DATA = Path(__file__).parents[3] / "data" / "raw" / "NaturalEarth" / "ne_10m_admin_0_countries.shp"
COUNTRY_NAME = "Switzerland"

OUTPUT_PATH = Path(__file__).parent / "figure_samples.pdf"


if __name__ == "__main__":

    samples = gpd.read_parquet(SAMPLE_FILE)
    samples = samples.set_crs(epsg=3035, allow_override=True)

    val_samples = gpd.read_parquet(VAL_SAMPLE_FILE)
    val_samples = val_samples.set_crs(samples.crs, allow_override=True)

    test_samples = gpd.read_parquet(TEST_SAMPLE_FILE)
    test_samples = test_samples.set_crs(samples.crs, allow_override=True)

    eva_plots = EVADataset().read_plot_data()
    eva_plots = eva_plots.to_crs(samples.crs)

    countries = gpd.read_file(COUNTRY_DATA)
    countries = countries.to_crs(samples.crs)
    switzerland = countries[countries["NAME_EN"] == COUNTRY_NAME]

    switz_geom = switzerland.unary_union
    bounds = switz_geom.bounds
    bbox_geom = box(*bounds).buffer(BOX_HALF_SIZE_KM * 1000)
    samples = gpd.GeoDataFrame(
        pd.concat([samples, val_samples], ignore_index=True),
        crs=samples.crs,
    )

    samples_ch = samples[samples.intersects(bbox_geom)]
    samples_ch = samples_ch[samples_ch.sp_unit_area <= 1e9]  # Filter out very large spatial units

    test_samples_ch = test_samples[test_samples.intersects(bbox_geom)]
    test_samples_ch = test_samples_ch[test_samples_ch.sp_unit_area <= 1e9]

    eva_plots_ch = eva_plots[eva_plots.intersects(bbox_geom)]

    neighbors = countries[countries.geometry.touches(switz_geom)]

    if len(samples_ch) > N_SAMPLES:
        samples_ch = samples_ch.sample(N_SAMPLES, random_state=42)

    if len(test_samples_ch) > N_SAMPLES:
        test_samples_ch = test_samples_ch.sample(N_SAMPLES, random_state=42)

    if len(eva_plots_ch) > N_PLOTS:
        eva_plots_ch = eva_plots_ch.sample(N_PLOTS, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    samples_ch.plot(ax=ax, color="#4361ee", alpha=0.6, linewidth=0, zorder=1)
    test_samples_ch.plot(ax=ax, color="#ff7a00", alpha=0.7, linewidth=0, zorder=2)

    x_min, y_min, x_max, y_max = samples_ch.total_bounds
    minx, miny, maxx, maxy = eva_plots.total_bounds

    grid_x_min = np.floor((x_min - minx) / BLOCK_SIZE_M) * BLOCK_SIZE_M + minx
    grid_x_max = np.ceil((x_max - minx) / BLOCK_SIZE_M) * BLOCK_SIZE_M + minx
    grid_y_min = np.floor((y_min - miny) / BLOCK_SIZE_M) * BLOCK_SIZE_M + miny
    grid_y_max = np.ceil((y_max - miny) / BLOCK_SIZE_M) * BLOCK_SIZE_M + miny

    verticals = [
        LineString([(x, grid_y_min), (x, grid_y_max)])
        for x in np.arange(grid_x_min, grid_x_max + BLOCK_SIZE_M, BLOCK_SIZE_M)
    ]
    horizontals = [
        LineString([(grid_x_min, y), (grid_x_max, y)])
        for y in np.arange(grid_y_min, grid_y_max + BLOCK_SIZE_M, BLOCK_SIZE_M)
    ]
    grid = gpd.GeoSeries(verticals + horizontals, crs=samples.crs)

    grid.plot(ax=ax, color="#9aa0a6", linewidth=0.4, alpha=0.6, zorder=0)
    eva_plots_ch.plot(ax=ax, color=COLORS_BR[0], markersize=4, zorder=3)

    countries.boundary.plot(ax=ax, color="#1f1f1f", linewidth=0.6, zorder=5)
    legend_handles = [
        Patch(facecolor="#4361ee", edgecolor="none", alpha=0.6, label="Train samples"),
        Patch(facecolor="#ff7a00", edgecolor="none", alpha=0.7, label="Test samples"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS_BR[0],
               markeredgecolor="none", markersize=6, label="Vegetation plots"),
        Patch(facecolor="none", edgecolor="#9aa0a6", linewidth=1.2, label="Spatial blocks"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=2,
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.8,
        columnspacing=1.4,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", transparent=True)
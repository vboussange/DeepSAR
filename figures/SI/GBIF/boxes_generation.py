"""
Illustrating the generation procedure of mega plots
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import seaborn as sns

from src.generate_SAR_data_GBIF import generate_random_boxes_on_grid
from src.data_processing.utils_gbif_local import GBIFDataset
from src.data_processing.utils_landcover import CopernicusDataset
from src.data_processing.utils_env_pred import CHELSADataset

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

num_polygons = int(2e4)
CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../results/GBIF_polygons_CHELSA/GBIF_Copernicus_CHELSA/GBIF_Copernicus_CHELSA_cinf_{num_polygons}.pkl",
    ),
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
    ],
    "n_partitions": 100,  # test
    "batch_size": 20,
    "area_range": (1e2, 1e9),  # in n pixels or 100m, here equivalent to 1000km2
    "side_range": (1, 1e4),
    "num_polygons": num_polygons,
    "crs": "EPSG:3035",
}

def load_and_preprocess_data():
    """
    Load and preprocess GBIF and landcover data.
    Returns processed GBIF and landcover datasets.
    """
    logging.info("Loading GBIF data...")
    gbif_data = GBIFDataset().load()
    
    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info("Loading landcover raster...")
    lc_dataset = CopernicusDataset()
    lc_raster = lc_dataset.load_landcover_level3()
    
    logging.info("Cropping GBIF data to study spatial extent")
    grid_bounds = lc_raster.rio.bounds()
    gbif_data = gbif_data.cx[
        grid_bounds[0] : grid_bounds[2], grid_bounds[1] : grid_bounds[3]
    ]

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    gbif_data = gbif_data.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    lc_raster = lc_raster.rio.reproject(CONFIG["crs"]).sortby("y")
    
    return gbif_data, climate_raster, lc_raster, lc_dataset

def assign_landcover_types(gbif_data, lc_raster):
    """
    Assign landcover types to GBIF points.
    Returns updated GBIF data.
    """
    logging.info("Assigning landcover types to GBIF points...")
    yy = gbif_data.geometry.y
    xx = gbif_data.geometry.x
    lct = lc_raster.sel(
        x=xr.DataArray(xx, dims="z"), y=xr.DataArray(yy, dims="z"), method="nearest"
    ).astype(np.int8)
    gbif_data["Copernicus_landcover_type_id"] = lct
    return gbif_data

def draw_map(ax_map):
    ax_map.add_feature(cfeature.COASTLINE)
    # ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, edgecolor="black")
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")


    # Add labels and a legend
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # ax_map.set_title("Plot Locations")
    ax_map.legend(loc="best")
    
    
if __name__ == "__main__":
    hab = 114 # " DBF_closed"

    gbif_data, climate_raster, lc_raster, lc_dataset = load_and_preprocess_data()
    gbif_data = assign_landcover_types(gbif_data, lc_raster)

    lc_binary = lc_raster == hab
    
    gbif_gdf_by_hab = gbif_data.groupby("Copernicus_landcover_type_id")
    gdf_hab = gbif_gdf_by_hab.get_group(hab)
    
    polygons_gdf = generate_random_boxes_on_grid(
        gdf_hab,
        lc_binary,
        CONFIG["num_polygons"],
        CONFIG["area_range"],
        CONFIG["side_range"],
    )

    polygons_gdf.geometry[3]
        
        
    fig = plt.figure(figsize = (10, 5))
    ax_map = fig.add_subplot(121, projection=ccrs.epsg(3035))
    draw_map(ax_map)
    ax_map.add_geometries(polygons_gdf.geometry, crs=ccrs.epsg(3035), facecolor='blue', edgecolor='red', alpha=0.5)

    ax = fig.add_subplot(122)
    polygons_gdf["area"] = polygons_gdf.area
    sns.kdeplot(polygons_gdf, 
                x = 'area', 
                ax=ax, 
                log_scale=(True, False))

    fig.savefig("boxes_generation_GBIF.png", dpi = 300)
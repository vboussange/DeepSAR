import pandas as pd
import numpy as np
from pathlib import Path
from src.generate_SAR_data_GBIF import (create_random_box_on_grid,
                                        crop_raster)
from src.data_processing.utils_polygons import place_randomly_rectangle
import rioxarray
import xarray as xr
from shapely.geometry import Point, box
import geopandas as gpd

def create_lcb_raster():
    # Create a 100x100 binary landcover raster where 1=land and 0=water
    data = np.random.randint(2, size=(101, 101))  # Random binary data

    # Convert the numpy array to an xarray DataArray
    raster = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.linspace(0.5, 200.5, 101) + 0.1, "x": np.linspace(100.5, 300.5, 101) + 0.1}
    )

    # Convert the DataArray to a rioxarray object and set a CRS
    raster.rio.write_crs("epsg:3035", inplace=True)
    return raster

def test_create_random_box_on_grid():
    lc_binary = create_lcb_raster()
    minx, miny, maxx, maxy = lc_binary.rio.bounds()
    raster_extent = box(minx, miny, maxx, maxy)
    POLY_RANGE = (1, 1, 10, 10)
    
    # check at lower left
    points = [Point(lc_binary.x[0], lc_binary.y[0])] 
    so_data = gpd.GeoDataFrame(crs = "epsg:3035", geometry=points)
    geom = create_random_box_on_grid(so_data, lc_binary, 100, POLY_RANGE).geometry[0]
    geom_extent = box(*geom.bounds)
    assert geom_extent.within(raster_extent)
    assert geom.area > 0
    
    # check at upper right, we expect that a point is created
    points = [Point(lc_binary.x[-1], lc_binary.y[-1])] 
    so_data = gpd.GeoDataFrame(crs = "epsg:3035", geometry=points)
    geom = create_random_box_on_grid(so_data, lc_binary, 100, POLY_RANGE).geometry[0]
    geom_extent = box(*geom.bounds)
    assert geom.area == 0
    
    # check at upper right, we expect that polygon of area 4 is created
    points = [Point(lc_binary.x[-2], lc_binary.y[-2])] 
    so_data = gpd.GeoDataFrame(crs = "epsg:3035", geometry=points)
    geom = create_random_box_on_grid(so_data, lc_binary, 100, POLY_RANGE).geometry[0]
    geom_extent = box(*geom.bounds)
    assert geom_extent.within(raster_extent)
    assert geom.area == 4
    
def test_crop_raster():
    lc_binary = create_lcb_raster()
    minx, miny, maxx, maxy = lc_binary.rio.bounds()
    points = [Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)) for _ in range(100)]
    so_data = gpd.GeoDataFrame(crs = "epsg:3035", geometry=points)
    POLY_RANGE = (1, 1, 10, 10)
    poly_gdf = create_random_box_on_grid(so_data, lc_binary, 100, POLY_RANGE)
    for geom in poly_gdf.geometry:
        cropped = crop_raster(lc_binary, geom)

        assert POLY_RANGE[0] <= cropped.shape[0] <= POLY_RANGE[2]
        assert POLY_RANGE[1] <= cropped.shape[1] <= POLY_RANGE[3]
        
        assert cropped.x[0].item() == geom.bounds[0]
        assert cropped.x[-1].item() == geom.bounds[2]
        assert cropped.y[0].item() == geom.bounds[1]
        assert cropped.y[-1].item() == geom.bounds[3]

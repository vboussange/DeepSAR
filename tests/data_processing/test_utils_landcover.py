import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
from shapely.geometry import Polygon, Point, box
import numpy as np
import os
from pathlib import Path
import rioxarray
import xarray as xr
from scipy import stats

from src.data_processing.utils_landcover import LandSysDataset, CopernicusDataset, EXTENT_DATASET

def test_landsys_is_land():
    dataset = LandSysDataset()
    assert dataset.is_land(0) == False

def test_landsys_is_land():
    dataset = LandSysDataset()
    geo_bounds = dataset.load_shapefile(4).total_bounds
    raster = dataset.get_land_support(geo_bounds)
    # visual assertion
    # raster.coarsen(x=100, y = 100, boundary="pad").mean().plot()
    assert raster.sum() > 0
    
def test_copernicus_is_land():
    dataset = CopernicusDataset()
    assert dataset.is_land(200) == False

def test_landsys_is_land():
    dataset = CopernicusDataset()
    raster = dataset.get_land_support(EXTENT_DATASET)
    # visual assertion
    # raster.coarsen(x=100, y = 100, boundary="pad").mean().plot()
    assert raster.sum() > 0
        
        
def test_load_landcover_level2():
    dataset = CopernicusDataset()
    lcl2 = dataset.load_landcover_level2()

    assert (lcl2 < 100).all()
    lcl2 = lcl2.rio.clip_box(minx=8.5, miny=47.3, maxx=8.6, maxy=47.4)
    # visual plot
    lcl2.plot()
    np.unique(lcl2)
    
def test_load_landcover_level3_1km():
    dataset = CopernicusDataset()
    extent = (34, 46, 34.1, 46.1)
    lcl31k = dataset.load_landcover_level3_1km(extent)
    lcl3 = dataset.load_landcover_level3(extent)
    
    val = stats.mode(lcl3.values[0:10,0:10], axis=None, nan_policy="omit")[0]
    assert lcl31k[0,0] == val
    
    val = stats.mode(lcl3.values[0:10,50:60], axis=None, nan_policy="omit")[0]
    assert lcl31k[0,6] == val

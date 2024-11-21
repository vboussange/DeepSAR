import dask.dataframe as dd
import os
from pathlib import Path
import geopandas as gpd
import warnings
from datetime import datetime
import pandas as pd
from shapely.geometry import box


from src.data_processing.utils_landcover import load_landsys_data, crs_transform_and_area
from src.data_processing.utils_gbif_local import load_gbif_data, process_partition
from src.data_processing.utils_polygons import create_habitat_free_polygons, partition_polygon_gdf

os.chdir(Path(__file__).parent)
pd.options.mode.chained_assignment = None  # see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

BATCH_SIZE = 20
N_PARTITIONS = 100
GBIF_PARQUET_PATH = Path("/lud11/boussang/data/gbif/gbif_EUROPE2.parquet")
CONNECTIVITY_TEMPLATE = 8
# warnings.filterwarnings('ignore')

def test_extent_gbif_landsys():
    """
    testing if extent of gbif is beyond extent of polygon_gdf
    """
    gbif_data = gpd.read_parquet(GBIF_PARQUET_PATH) # 1 min on Sauron
    gdf_landsys = load_landsys_data(CONNECTIVITY_TEMPLATE)
    gdf_landsys.to_crs(epsg=4326, inplace=True)
    
    gbif_extent = box(*gbif_data.total_bounds)
    gdf_landsys = box(*gdf_landsys.total_bounds)
    assert gdf_landsys.within(gbif_extent)




    

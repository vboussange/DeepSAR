import geopandas as gpd
import pandas as pd
import numpy as np

def read_data(folder_path):
    all_files = folder_path.glob("*.pkl")
    gdf_list = [pd.read_pickle(fp) for fp in all_files]
    return gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
import socket
import rioxarray
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import os
import socket

from src.data_processing.utils_landcover import EXTENT_DATASET

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_CHELSA = Path(FILE_PATH, 'CHELSA/CHELSA_EU.nc')


class EnvironmentalFeatureDataset():
    def __init__(self, tif_path, cache_path):
        self.tif_path = tif_path
        self.cache_path = cache_path
        
class CHELSADataset(EnvironmentalFeatureDataset):
    def __init__(self, tif_path="", cache_path=CACHE_CHELSA):
        CACHE_CHELSA.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(tif_path, cache_path)


    def load(self, extent = EXTENT_DATASET):
        """
        Loads and combines environmental raster data into a single xarray Dataset.

        Note:
            Caching not working
            
        Args:
            worldclim_path (Path): Path to WorldClim TIFF files.

        Returns:
            xr.Dataset: Combined dataset of all WorldClim variables.
        """
        if self.cache_path.is_file():
            with xr.open_dataset(self.cache_path) as ds:
                return ds.to_array()
        
        data_arrays = []
        for tiff_path in self.tif_path.glob("*.tif"):
            with rioxarray.open_rasterio(tiff_path, mask_and_scale=True) as da:
                # we load slightly more than the extent to be able to correctly interpolate
                dx = 0.1
                cda = da.rio.clip_box(minx=extent[0] - dx, miny=extent[1] - dx, maxx=extent[2]+ dx, maxy=extent[3]+ dx)
                cda = cda.sel(band=1)
                cda = cda.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
                
                # extracting name and renaming
                name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010_V.2.1")[0]
                cda = cda.rename(name)
                
                data_arrays.append(cda)
                
        ref_array = [ar for ar in data_arrays if ar.name == "bio1"][0]
        for i in range(len(data_arrays)):
            if not data_arrays[i].coords.equals(ref_array): 
                data_arrays[i] = data_arrays[i].interp_like(ref_array)

        dataset = xr.merge(data_arrays, join="left")
        # now we clip the dataset
        dataset = dataset.sel(x=slice(extent[0], extent[2]), y = slice(extent[3],extent[1]))
        
        # # caching
        dataset.to_netcdf(self.cache_path)
        
        return dataset.to_array()


def get_mean_std_env_pred(row):
    _m = np.nanmean(row["env_pred"], axis = (1,2))
    _std = np.nanstd(row["env_pred"], axis = (1,2))
    return np.concatenate([_m,_std])

def calculate_aggregates(gdf, env_vars):
    
    # prepare aggregate values for predictors
    env_pred_arr = gdf.apply(get_mean_std_env_pred, axis=1)

    # Generate column names
    mean_labels = env_vars
    std_labels = [f"std_{var}" for var in env_vars]
    column_names = np.hstack((mean_labels, std_labels))

    # Convert the result to a DataFrame
    env_pred_df = pd.DataFrame(env_pred_arr.tolist(), columns=column_names)
    
    gdf = gdf.join(env_pred_df)
    return gdf, column_names
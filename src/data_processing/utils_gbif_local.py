import dask.dataframe as dd
import dask_geopandas
from pathlib import Path
import pandas as pd
import geopandas as gpd
import os 
from tqdm import tqdm
from shapely.geometry import box
import logging
import numpy as np

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
GBIF_PATH = Path("/lud11/boussang/data/gbif/EUROPE_2.csv")
CACHE_DIRECTORY_GBIF = Path("../../../data/gbif/.EUROPE2_cache")
GBIF_PARQUET_PATH = Path("/lud11/boussang/data/gbif/gbif_EUROPE2.parquet")

class GBIFDataset:
    # TODO: we should probably filter data, using rpy2
    # (https://rpy2.github.io/doc/latest/html/generated_rst/pandas.html) and
    # [CoordinateCleaner](https://cran.r-project.org/web/packages/CoordinateCleaner/vignettes/Cleaning_GBIF_data_with_CoordinateCleaner.html)
    def __init__(self, path=GBIF_PATH, cache_path = GBIF_PARQUET_PATH):
        self.gbif_path = path
        self.cache_path = cache_path

    def dask_load(gbif_path  = GBIF_PATH):
        """
        Load GBIF data using Dask for efficient parallel processing.
        """
        gbif_data = dd.read_csv(gbif_path, 
                                on_bad_lines='warn', 
                                sep='\t', 
                                encoding = "utf-8",  
                                usecols = ['speciesKey', 'countryCode', 'decimalLatitude', 'decimalLongitude'], 
                                dtype={'speciesKey': 'float64',
                                        },
                                quoting=3,
                                )
        gbif_data = gbif_data.set_geometry(
            dask_geopandas.points_from_xy(gbif_data, "decimalLongitude", "decimalLatitude", crs="EPSG:4326")
        )
        return gbif_data
        
    def load(self, *args, **kwargs):
        """
        Load GBIF data.
        """
        logging.debug("Loading GBIF data...")
        if not self.cache_path.is_file(): 
            gbif_data = pd.read_csv(self.gbif_path, 
                                    on_bad_lines='warn', 
                                    sep='\t', 
                                    encoding = "utf-8",  
                                    usecols = ['speciesKey', 'countryCode', 'decimalLatitude', 'decimalLongitude'], 
                                    dtype={'speciesKey': np.float64,
                                           'decimalLatitude': np.float32,
                                            'decimalLongitude': np.float32
                                            },
                                    quoting=3,
                    )
            gbif_data["geometry"] = gpd.points_from_xy(gbif_data.decimalLongitude, gbif_data.decimalLatitude, crs="EPSG:4326")
            gbif_gpd = gpd.GeoDataFrame(gbif_data, geometry="geometry")
            # filtering unidentified species
            gbif_gpd.dropna(inplace=True)
            gbif_gpd["speciesKey"] = gbif_gpd["speciesKey"].astype(np.int32)
            # gbif_gpd['decimalLatitude', 'decimalLongitude'] = gbif_gpd['decimalLatitude', 'decimalLongitude'].astype(np.float32)    
            gbif_gpd.to_parquet(self.cache_path)
        else:
            gbif_gpd = gpd.read_parquet(self.cache_path, *args, **kwargs) # 1 min on Sauron
            assert np.isnan(gbif_gpd.speciesKey).sum() == 0

        return gbif_gpd
        
def batch_indices(N, batch_size):
    """Yield successive batch-sized chunks of indices from 0 to N."""
    for i in range(0, N, batch_size):
        yield range(i, min(i + batch_size, N))

def process_partition(gbif_gdf, polygons_gdf, savepath, batchsize):
    try:
        import cuspatial
    except ImportError:
        raise ImportError("This function requires cuspatial for GPU acceleration. Please install cuspatial and try again.")
   
    # sending to gpu
    gbif_gdf_gpu = cuspatial.from_geopandas(gbif_gdf)
    # sending to gpu
    polygon_gdf_gpu = cuspatial.from_geopandas(polygons_gdf.geometry)
        
    polygons_gdf["sampling_effort"] = 0
    polygons_gdf["sr"] = 0e0 # species richness
    polygons_gdf.reset_index(drop=True, inplace=True)
    
    gen = batch_indices(len(polygons_gdf), batchsize)
    for batch in tqdm(gen):
        idxs = list(batch)
        pip = cuspatial.point_in_polygon(gbif_gdf_gpu.geometry, polygon_gdf_gpu.iloc[idxs])
        
        species_list = [gbif_gdf_gpu.speciesKey[pip[col]].unique() for col in pip.columns]
        
        species_richness = [len(sublist) for sublist in species_list]
        sampling_effort = pip.to_numpy().sum(axis=0)
        
        # see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        # We do something wrong here, but idxs is integer based and not index based
        polygons_gdf.loc[idxs, "sr"] = species_richness
        polygons_gdf.loc[idxs, "sampling_effort"] = sampling_effort
        
    # keeping polygons for which we have at least an entry
    polygons_gdf = polygons_gdf[polygons_gdf.sr > 0]
    
    
    if not polygons_gdf.empty:
        savepath.mkdir(exist_ok=True, parents=True)
        output_parquet_path = savepath / Path(f'partition_{polygons_gdf.partition.iloc[0]}.parquet')
        polygons_gdf.to_parquet(output_parquet_path)
        print(f"Polygon DataFrame saved at {output_parquet_path}")
        
def test_extent(gbif_data, polygon_data):
    gbif_extent = box(*gbif_data.total_bounds)
    polygon_extent = box(*polygon_data.total_bounds)
    assert polygon_extent.within(gbif_extent)
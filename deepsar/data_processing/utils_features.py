import rioxarray
import xarray as xr
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
from dask import delayed
import dask
from rasterio.enums import Resampling

# Default paths
CHELSA_PATH = Path(Path(__file__).parent, '../../data/raw/CHELSA/chelsav2/GLOBAL/climatologies/1981-2010/bio')
DEM_PATH = Path(Path(__file__).parent, '../../data/raw/EEA_DEM/eudem_dem_3035_europe_100m.tif')
LC_PATH = Path(Path(__file__).parent, '../../data/raw/Corine_Landcover/CLC2018_CLC2018_V2018_20.tif')
CACHE_DIR = Path(Path(__file__).parent, '../../data/processed/environmental_features')

# debug paths
# CHELSA_PATH = Path(Path(__file__).parent, '../../data/raw/CHELSA/debug')
# DEM_PATH = Path(Path(__file__).parent, '../../data/raw/EEA_DEM/eudem_debug_100m_ch.tif')
# LC_PATH = Path(Path(__file__).parent, '../../data/raw/Corine_Landcover/CLC2018_CLC2018_V2018_20_ch.tif')
# CACHE_DIR = Path(Path(__file__).parent, '../../data/processed/environmental_features_debug')

class EnvironmentalFeatureDataset():
    """
    Environmental feature dataset loader that combines CHELSA climate data,
    DEM elevation data, and Corine Land Cover data into a single aligned xarray Dataset.
    
    The dataset is suitable for downstream deep learning applications with:
    - All rasters aligned to the same spatial grid
    - Remapped landcover classes (stored as integers for memory efficiency)
    - Consistent coordinate reference system (default: EPSG:3035)
    """
    
    def __init__(self, 
                 chelsa_path=CHELSA_PATH, 
                 dem_path=DEM_PATH, 
                 lc_path=LC_PATH, 
                 cache_dir=CACHE_DIR):
        self.chelsa_path = Path(chelsa_path)
        self.dem_path = Path(dem_path)
        self.lc_path = Path(lc_path)
        self.cache_dir = Path(cache_dir)
        
        # Individual cache paths for each dataset
        self.chelsa_cache = self.cache_dir / 'chelsa_cache.nc'
        self.dem_cache = self.cache_dir / 'dem_cache.nc'
        self.lc_cache = self.cache_dir / 'landcover_cache.nc'        

    def load(self, use_cache=False):
        """
        Loads and combines environmental raster data into a single xarray Dataset.
        
        This method loads CHELSA climate variables, DEM elevation, and Corine Land Cover data,
        aligns them to a common grid, and one-hot encodes the landcover classes.
        All rasters are cropped to the extent of the smallest raster.
            
        Args:
            use_cache (bool): Whether to use cached data if available.

        Returns:
            xr.Dataset: Combined dataset with:
                - CHELSA bioclimatic variables (bio1, bio2, ..., bio19)
                - DEM elevation (elevation)
                - Remapped landcover classes (landcover) as integers
                All in target_crs coordinate system.
        """
        print("Loading and aligning environmental datasets...")
        
        # Load DEM first to get reference extent (DEM defines the European extent)
        dem_ds = self._load_dem(use_cache)
        
        # Use DEM as reference grid for alignment
        ref_da = dem_ds['elevation']
                
        # Load other datasets cropped to DEM extent
        chelsa_ds = self._load_chelsa(ref_da, use_cache)
        lc_ds = self._load_landcover(ref_da, use_cache)
                
        # we do not merge lc_ds here as it is np.int16
        chelsa_dem_ds = xr.merge([
            dem_ds,
            chelsa_ds
        ])
        
        # for var in chelsa_dem_ds.data_vars:
        #     chelsa_dem_ds[var] = chelsa_dem_ds[var].rio.write_nodata(np.nan)
        
        return chelsa_dem_ds, lc_ds
    
    def _load_chelsa(self, ref_da, use_cache=True):
        """Load CHELSA bioclimatic variables and reproject to target CRS.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in target_crs to crop the data.
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.chelsa_cache.is_file():
            with xr.open_dataset(self.chelsa_cache) as ds:
                if ds.rio.bounds() == ref_da.rio.bounds():
                    return ds
        
        data_arrays = []
        for tiff_path in tqdm(self.chelsa_path.glob("*.tif"), desc="Loading CHELSA variables"):
            print("Loading and interpolating", tiff_path)
            with rioxarray.open_rasterio(tiff_path, mask_and_scale=True) as da:
                da = da.sel(band=1, drop=True)
                # extracting name and renaming
                name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010_V.2.1")[0]
                da = da.rename(name)
                # interpolating and reprojecting
                da = da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
                # da = da.rio.write_nodata(np.nan)
                data_arrays.append(da)
                
        dataset = xr.merge(data_arrays, join="left")
        dataset.rio.write_crs(ref_da.rio.crs, inplace=True)
        
        # Remove _FillValue from attributes to avoid conflicts with encoding
        for var in dataset.data_vars:
            if '_FillValue' in dataset[var].attrs:
                del dataset[var].attrs['_FillValue']
        encoding = {var: {'dtype': 'float32', '_FillValue': np.nan} for var in dataset.data_vars}

        # caching
        if use_cache:
            self.chelsa_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching CHELSA to {self.chelsa_cache}")
            dataset.to_netcdf(self.chelsa_cache, engine='netcdf4', encoding=encoding)
        return dataset
            
    def _load_dem(self, use_cache=False):
        """Load DEM elevation data and reproject to target CRS.
        
        Args:
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.dem_cache.exists():
            print(f"Loading DEM from cache: {self.dem_cache}")
            dataset = xr.open_dataset(self.dem_cache)
            return dataset
        
        print("Loading DEM data...")
        
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
        
        with rioxarray.open_rasterio(self.dem_path, mask_and_scale=True) as da:
            dem_da = da.sel(band=1, drop=True)
            dem_da = xr.Dataset({'elevation': dem_da})
            
            # Save to cache with parallel computation
            if use_cache:
                self.dem_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching DEM to {self.dem_cache}")
                dem_da.to_netcdf(self.dem_cache)
            
            return dem_da
        
    
    def _load_landcover(self, ref_da, use_cache=False):
        """Load and remap Corine Land Cover data and reproject to target CRS.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in target_crs to crop the data.
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.lc_cache.exists():
            print(f"Loading landcover from cache: {self.lc_cache}")
            lc_ds = xr.open_dataset(self.lc_cache)
            if lc_ds.rio.bounds() == ref_da.rio.bounds():
                return lc_ds
            else:
                print("Cache does not match reference grid, recompiling.")
        
        print("Loading landcover data...")
        
        if not self.lc_path.exists():
            raise FileNotFoundError(f"Landcover file not found: {self.lc_path}")
        
        with rioxarray.open_rasterio(self.lc_path, mask_and_scale=True) as da:
            lc_da = da.sel(band=1, drop=True)
            lc_da = lc_da.rio.reproject_match(ref_da, resampling=Resampling.nearest).astype(np.int16)
            
            # Extract unique landcover classes from the raster
            print("Extracting unique landcover classes from raster...")
            unique_classes = np.unique(lc_da.values)
                        
            # Remap landcover classes to consecutive integers using vectorized operation
            class_mapping = {orig_class: idx for idx, orig_class in enumerate(unique_classes)}
            
            # Vectorized remapping using numpy's searchsorted (parallelized with dask)
            # Create lookup arrays for efficient remapping
            new_values = np.arange(len(unique_classes))
            
            # Vectorized remapping using searchsorted
            lc_remapped = lc_da.copy()
            flat_values = lc_da.values.flatten()
            indices = np.searchsorted(unique_classes, flat_values)
            remapped_flat = new_values[indices]
            lc_remapped.values = remapped_flat.reshape(lc_da.shape)
            
            # Cast to int16 to save memory (sufficient for landcover classes)
            # Use -9999 as a fill value for NaN before casting to int16
            lc_remapped = lc_remapped.fillna(-9999).astype(np.int16)
            
            # Store the mapping as attributes
            lc_remapped.attrs['class_mapping'] = str([int(k) for k in class_mapping.values()])
            lc_remapped.attrs['original_classes'] = str([int(k) for k in class_mapping.keys()])
                        
            lc_ds = xr.Dataset({'landcover': lc_remapped})
            lc_ds.rio.write_crs(ref_da.rio.crs, inplace=True)

            # Save to cache with parallel computation
            if use_cache:
                self.lc_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching landcover to {self.lc_cache}")
                # Specify encoding to handle int16 with fill value
                encoding = {'landcover': {'dtype': 'int16', '_FillValue': -9999}}
                lc_ds.to_netcdf(self.lc_cache, encoding=encoding)
            
            return lc_ds
            
        print(f"✓ Cached {len(dataset.data_vars)} variables to {self.cache_path}")
    
    def _load_from_cache(self):
        """Load dataset from cached COG file."""
        # Load multi-band COG as DataArray with chunking for parallel access
        stacked_da = rioxarray.open_rasterio(self.cache_path, mask_and_scale=True, chunks=self.chunk_size)
        
        # Convert DataArray back to Dataset
        var_names = stacked_da.coords['band'].values
        data_vars = {}
        for i, var_name in enumerate(var_names, start=1):
            data_vars[str(var_name)] = stacked_da.sel(band=i, drop=True)
        
        dataset = xr.Dataset(data_vars)
        print(f"✓ Loaded {len(dataset.data_vars)} variables from cache")
        
        return dataset


if __name__ == "__main__":
    features = EnvironmentalFeatureDataset()
    env_features_ds, lc_ds = features.load(use_cache=True)
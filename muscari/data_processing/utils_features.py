import rioxarray
import xarray as xr
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
from rasterio.enums import Resampling

# Default base paths (can be overridden via environment variables)
BASE_DIR = Path(os.getenv('MUSCARI_DATA_DIR', Path(__file__).parent.parent.parent / 'data'))

# Default paths with environment variable support
CHELSA_PATH = Path(os.getenv('CHELSA_PATH', BASE_DIR / 'raw/CHELSA/chelsav2/GLOBAL/climatologies/1981-2010/bio'))
DEM_PATH = Path(os.getenv('DEM_PATH', BASE_DIR / 'raw/EEA_DEM/eudem_dem_3035_europe_1000m.tif'))
LC_PATH = Path(os.getenv('LC_PATH', BASE_DIR / 'raw/Corine_Landcover/CLC2018_CLC2018_V2018_20.tif'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', BASE_DIR / 'processed/environmental_features'))

class EnvironmentalFeatureDataset():
    """
    Environmental feature dataset loader.

    Combines CHELSA climate variables, DEM elevation, and Corine Land Cover into a
    single aligned xarray Dataset on a shared grid.
    """
    
    # Compression settings for netCDF files (zlib with complevel 5 is a good balance)
    COMPRESSION_ENCODING = {'zlib': True, 'complevel': 5}
    
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

    def load(self, use_cache=True):
        """
        Loads and combines environmental raster data into a single xarray Dataset.
        
        This method loads CHELSA climate variables, DEM elevation, and Corine Land Cover data,
        and aligns them to a common grid.
        All rasters are cropped to the extent of the smallest raster.
            
        Args:
            use_cache (bool): Whether to use cached data if available.

                Returns:
                        tuple[xr.Dataset, xr.Dataset]:
                                - CHELSA + DEM dataset with bioclimatic variables (bio1..bio19)
                                    and elevation (elevation)
                                - Landcover dataset with remapped classes (landcover) as integers
                                Both in the target CRS.
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
        chelsa_dem_ds = xr.merge([dem_ds, chelsa_ds])
        
        return chelsa_dem_ds, lc_ds
    
    def _load_chelsa(self, ref_da, use_cache=True):
        """Load CHELSA bioclimatic variables and reproject to target CRS.
        
        Args:
            ref_da: Reference DataArray used to match grid, CRS, and extent.
            use_cache: Whether to use cached data when available.

        Returns:
            xr.Dataset: CHELSA variables aligned to the reference grid.
        """
        # Check cache
        if use_cache and self.chelsa_cache.is_file():
            print(f"Loading CHELSA from cache: {self.chelsa_cache}")
            ds = xr.open_dataset(self.chelsa_cache)
            if ds.rio.bounds() == ref_da.rio.bounds():
                return ds
            else:
                print("Cache does not match reference grid, recompiling.")
        
        data_arrays = []
        for tiff_path in tqdm(sorted(self.chelsa_path.glob("*.tif")), desc="Loading CHELSA variables"):
            with rioxarray.open_rasterio(tiff_path, mask_and_scale=True) as da:
                da = da.sel(band=1, drop=True)
                # Extract variable name and rename
                name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010_V.2.1")[0]
                da = da.rename(name)
                # Reproject to match reference grid
                da = da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
                data_arrays.append(da)
                
        dataset = xr.merge(data_arrays, join="left")
        dataset.rio.write_crs(ref_da.rio.crs, inplace=True)
        
        # Optimize encoding: float32 for data and coordinates, with compression
        encoding = self._get_optimized_encoding(dataset, dtype='float32')
        
        # Cache the dataset
        if use_cache:
            self.chelsa_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching CHELSA to {self.chelsa_cache}")
            dataset.to_netcdf(self.chelsa_cache, engine='netcdf4', encoding=encoding)
        return dataset
            
    def _load_dem(self, use_cache=False):
        """Load DEM elevation data.
        
        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            xr.Dataset: DEM elevation dataset with variable name `elevation`.
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
            dem_ds = xr.Dataset({'elevation': dem_da})
            dem_ds.rio.write_crs(dem_da.rio.crs, inplace=True)
            
            # Optimize encoding: float32 for data and coordinates, with compression
            encoding = self._get_optimized_encoding(dem_ds, dtype='float32')
            
            # Cache the dataset
            if use_cache:
                self.dem_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching DEM to {self.dem_cache}")
                dem_ds.to_netcdf(self.dem_cache, engine='netcdf4', encoding=encoding)
            
            return dem_ds
        
    
    def _load_landcover(self, ref_da, use_cache=False):
        """Load and remap Corine Land Cover data and reproject to target CRS.
        
        Args:
            ref_da: Reference DataArray used to match grid, CRS, and extent.
            use_cache: Whether to use cached data when available.

        Returns:
            xr.Dataset: Remapped landcover dataset aligned to the reference grid.
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
            lc_da = lc_da.rio.reproject_match(ref_da, resampling=Resampling.mode).astype(np.int16)
            
            # Extract unique landcover classes from the raster
            print("Extracting unique landcover classes from raster...")
            unique_classes = np.unique(lc_da.values)
                        
            # Remap landcover classes to consecutive integers
            class_mapping = {orig_class: idx for idx, orig_class in enumerate(unique_classes)}
            
            # Vectorized remapping using numpy's searchsorted
            new_values = np.arange(len(unique_classes))
            lc_remapped = lc_da.copy()
            flat_values = lc_da.values.flatten()
            indices = np.searchsorted(unique_classes, flat_values)
            remapped_flat = new_values[indices]
            lc_remapped.values = remapped_flat.reshape(lc_da.shape)
            
            # Use -9999 as fill value for int16 data
            lc_remapped = lc_remapped.fillna(-9999).astype(np.int16)
            
            # Store the mapping as attributes
            lc_remapped.attrs['class_mapping'] = str([int(k) for k in class_mapping.values()])
            lc_remapped.attrs['original_classes'] = str([int(k) for k in class_mapping.keys()])
                        
            lc_ds = xr.Dataset({'landcover': lc_remapped})
            lc_ds.rio.write_crs(ref_da.rio.crs, inplace=True)

            # Optimize encoding: int16 for data, float32 for coordinates, with compression
            encoding = self._get_optimized_encoding(lc_ds, dtype='int16', fill_value=-9999)
            
            # Cache the dataset
            if use_cache:
                self.lc_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching landcover to {self.lc_cache}")
                lc_ds.to_netcdf(self.lc_cache, engine='netcdf4', encoding=encoding)
            
            return lc_ds
    
    def _get_optimized_encoding(self, dataset, dtype='float32', fill_value=None):
        """Generate optimized encoding dict for netCDF with compression and coordinate precision.
        
        Args:
            dataset: xarray Dataset to generate encoding for.
            dtype: Data type for variables (default: 'float32').
            fill_value: Fill value for missing data (default: np.nan for float, -9999 for int).
            
        Returns:
            dict: Encoding dictionary with compression and coordinate settings.
        """
        # Set default fill value based on dtype
        if fill_value is None:
            fill_value = np.nan if 'float' in dtype else -9999
        
        # Encoding for data variables
        encoding = {}
        for var in dataset.data_vars:
            # Remove conflicting attributes
            if '_FillValue' in dataset[var].attrs:
                del dataset[var].attrs['_FillValue']
            
            encoding[var] = {
                'dtype': dtype,
                '_FillValue': fill_value,
                **self.COMPRESSION_ENCODING
            }
        
        # Encoding for coordinates (always float32 for precision/storage balance)
        for coord in dataset.coords:
            if coord not in ['band', 'spatial_ref']:  # Skip non-spatial coords
                encoding[coord] = {
                    'dtype': 'float32',
                    **self.COMPRESSION_ENCODING
                }
        
        return encoding


if __name__ == "__main__":
    features = EnvironmentalFeatureDataset()
    env_features_ds, lc_ds = features.load(use_cache=True)
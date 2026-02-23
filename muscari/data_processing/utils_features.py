import rioxarray
import xarray as xr
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
from rasterio.enums import Resampling
from muscari.data_processing._cache import MUSCARI_CACHE_DIR

# Default base paths (can be overridden via environment variables)
BASE_DIR = Path(os.getenv('MUSCARI_DATA_DIR', Path(__file__).parent.parent.parent / 'data'))

# Default paths with environment variable support
CHELSA_PATH = Path(os.getenv('CHELSA_PATH', BASE_DIR / 'raw/CHELSA/chelsav2/GLOBAL/climatologies/1981-2010/bio'))
DEM_PATH = Path(os.getenv('DEM_PATH', BASE_DIR / 'raw/EEA_DEM/eudem_dem_3035_europe_1000m.tif'))
LC_PATH = Path(os.getenv('LC_PATH', BASE_DIR / 'raw/Corine_Landcover/CLC2018_CLC2018_V2018_20.tif'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', BASE_DIR / 'processed/environmental_features'))
HF_DATASET_REPO = "vboussange/muscari-data"

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
                 cache_dir=None):
        self.chelsa_path = Path(chelsa_path)
        self.dem_path = Path(dem_path)
        self.lc_path = Path(lc_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else MUSCARI_CACHE_DIR / "environmental_features"
        
        # Cache paths
        self.chelsa_dem_cache = self.cache_dir / 'chelsa_dem_cache.nc'
        self.lc_cache = self.cache_dir / 'landcover_cache.nc'

    def push_to_hub(self, repo_id: str, token: str = None):
        """Upload the environmental feature caches to the Hugging Face Hub.

        Builds both caches first if they do not exist, then uploads
        ``chelsa_dem_cache.nc`` and ``landcover_cache.nc`` under
        ``environmental_features/`` in the repository.

        Args:
            repo_id: HF Hub repository id, e.g. ``"username/muscari-data"``.
            token: HF API token. Falls back to the cached login token when
                ``None``.
        """
        from huggingface_hub import HfApi

        # Ensure both caches exist
        if not self.chelsa_dem_cache.is_file() or not self.lc_cache.is_file():
            print("Cache files not found — building them now…")
            self.load()

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

        for local_path in (self.chelsa_dem_cache, self.lc_cache):
            path_in_repo = f"environmental_features/{local_path.name}"
            print(f"Uploading {path_in_repo} …")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  ✓ {path_in_repo} uploaded")

    @classmethod
    def from_hub(cls, repo_id: str, local_dir=None, token: str = None):
        """Download environmental feature caches from the Hugging Face Hub.

        Downloads ``chelsa_dem_cache.nc`` and ``landcover_cache.nc`` directly
        into the unified cache directory so that ``load()`` returns them
        immediately without rebuilding from raw rasters.

        Args:
            repo_id: HF Hub repository id, e.g. ``"username/muscari-data"``.
            local_dir: Override for the cache directory. When ``None`` the
                unified ``MUSCARI_CACHE_DIR/environmental_features`` is used.
            token: HF API token. Falls back to the cached login token when
                ``None``.

        Returns:
            EnvironmentalFeatureDataset: Instance whose ``load()`` immediately
                serves the downloaded caches.
        """
        import shutil
        from huggingface_hub import hf_hub_download

        instance = cls(cache_dir=local_dir) if local_dir is not None else cls()
        instance.cache_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("chelsa_dem_cache.nc", "landcover_cache.nc"):
            dest = instance.cache_dir / filename
            if not dest.exists():
                path_in_repo = f"environmental_features/{filename}"
                print(f"Downloading {path_in_repo} from {repo_id} …")
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    repo_type="dataset",
                    token=token,
                )
                shutil.copy(downloaded, dest)
                print(f"  ✓ Saved to {dest}")
            else:
                print(f"  {filename} already present, skipping download")

        return instance

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

        if use_cache and (not self.chelsa_dem_cache.is_file() or not self.lc_cache.is_file()):
            print(
                "Local environmental cache files are missing. "
                f"Trying Hugging Face ({HF_DATASET_REPO})..."
            )
            try:
                EnvironmentalFeatureDataset.from_hub(HF_DATASET_REPO, local_dir=self.cache_dir)
            except Exception as exc:
                print(f"Could not download environmental caches from Hugging Face: {exc}")

        chelsa_dem_ds = self._load_chelsa_dem(use_cache)

        # Use elevation as reference grid for landcover alignment
        ref_da = chelsa_dem_ds['elevation']
        lc_ds = self._load_landcover(ref_da, use_cache)

        return chelsa_dem_ds, lc_ds

    def _load_chelsa_dem(self, use_cache=True):
        """Load and combine CHELSA bioclimatic variables and DEM elevation into one dataset.

        DEM defines the reference grid; all CHELSA variables are reprojected to match it.

        Args:
            use_cache: Whether to use cached data when available.

        Returns:
            xr.Dataset: Merged dataset with elevation and bioclimatic variables.
        """
        if use_cache and self.chelsa_dem_cache.is_file():
            print(f"Loading CHELSA+DEM from cache: {self.chelsa_dem_cache}")
            return xr.open_dataset(self.chelsa_dem_cache)

        # --- Load DEM (defines the reference grid) ---
        print("Loading DEM data...")
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

        with rioxarray.open_rasterio(self.dem_path, mask_and_scale=True) as da:
            dem_da = da.sel(band=1, drop=True)
            dem_ds = xr.Dataset({'elevation': dem_da})
            dem_ds.rio.write_crs(dem_da.rio.crs, inplace=True)

        ref_da = dem_ds['elevation']

        # --- Load CHELSA variables ---
        data_arrays = []
        for tiff_path in tqdm(sorted(self.chelsa_path.glob("*.tif")), desc="Loading CHELSA variables"):
            with rioxarray.open_rasterio(tiff_path, mask_and_scale=True) as da:
                da = da.sel(band=1, drop=True)
                name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010_V.2.1")[0]
                da = da.rename(name)
                da = da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
                data_arrays.append(da)

        chelsa_ds = xr.merge(data_arrays, join="left")
        chelsa_ds.rio.write_crs(ref_da.rio.crs, inplace=True)

        # --- Merge DEM and CHELSA ---
        chelsa_dem_ds = xr.merge([dem_ds, chelsa_ds])

        # Cache the combined dataset
        if use_cache:
            encoding = self._get_optimized_encoding(chelsa_dem_ds, dtype='float32')
            self.chelsa_dem_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching CHELSA+DEM to {self.chelsa_dem_cache}")
            chelsa_dem_ds.to_netcdf(self.chelsa_dem_cache, engine='netcdf4', encoding=encoding)

        return chelsa_dem_ds
        
    
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
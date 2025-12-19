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
CACHE_PATH = CACHE_DIR / 'aligned_features_EPSG3035_cog.tif'


class EnvironmentalFeatureDataset():
    """
    Environmental feature dataset loader that combines CHELSA climate data,
    DEM elevation data, and Corine Land Cover data into a single aligned xarray Dataset.
    
    The dataset is suitable for downstream deep learning applications with:
    - All rasters aligned to the same spatial grid
    - Remapped landcover classes (stored as integers for memory efficiency)
    - Consistent coordinate reference system (default: EPSG:3035)
    """
    
    def __init__(self, chelsa_path=CHELSA_PATH, dem_path=DEM_PATH, lc_path=LC_PATH, cache_path=CACHE_PATH, target_crs='EPSG:3035', chunk_size='auto'):
        self.chelsa_path = Path(chelsa_path)
        self.dem_path = Path(dem_path)
        self.lc_path = Path(lc_path)
        self.cache_path = Path(cache_path)
        self.target_crs = target_crs
        self.chunk_size = chunk_size
        
        # Individual cache paths for each dataset
        cache_dir = self.cache_path.parent
        self.chelsa_cache = cache_dir / 'chelsa_cache.zarr'
        self.dem_cache = cache_dir / 'dem_cache.zarr'
        self.lc_cache = cache_dir / 'landcover_cache.zarr'
        

    def load(self, use_cache=True):
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
        # Check cache
        if use_cache and self.cache_path.is_file():
            print(f"Loading cached dataset from {self.cache_path}")
            return self._load_from_cache()
        
        print("Loading and aligning environmental datasets...")
        
        # Load DEM first to get reference extent (DEM defines the European extent)
        dem_da = self._load_dem(use_cache=False)
        
        # Extract DEM extent in its CRS
        dem_extent = dem_da.rio.bounds()
        print(f"DEM extent in {self.target_crs}: ({dem_extent[0]:.2f}, {dem_extent[1]:.2f}, {dem_extent[2]:.2f}, {dem_extent[3]:.2f})")
        
        # Load other datasets cropped to DEM extent
        chelsa_ds = self._load_chelsa(bbox=dem_extent, use_cache=True)
        lc_ds = self._load_landcover(bbox=dem_extent, use_cache=False)
        
        # Use DEM as reference grid for alignment
        ref_da = dem_da
        
        # Align CHELSA to reference grid using vectorized operation
        print("Aligning CHELSA to reference grid...")
        chelsa_ds_aligned = chelsa_ds.interp_like(ref_da, method='linear')
        
        # Align landcover to reference grid (use nearest neighbor for categorical data)
        print("Aligning landcover to reference grid...")
        lc_ds_aligned = lc_ds.interp_like(ref_da, method='nearest')
        
        # Merge all datasets and compute in parallel
        print("Merging datasets...")
        chelsa_ds_aligned = chelsa_ds_aligned.compute()
        lc_ds_aligned = lc_ds_aligned.compute()
        
        combined_ds = xr.merge([
            chelsa_ds_aligned,
            ref_da.to_dataset(name='elevation'),
            lc_ds_aligned
        ])
        
        # Cache the result
        if use_cache:
            print(f"Caching dataset to {self.cache_path}")
            self._save_to_cache(combined_ds)
        
        return combined_ds
    
    def _load_chelsa(self, bbox=None, use_cache=True):
        """Load CHELSA bioclimatic variables and reproject to target CRS.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in target_crs to crop the data.
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.chelsa_cache.exists():
            print(f"Loading CHELSA from cache: {self.chelsa_cache}")
            dataset = xr.open_dataset(self.chelsa_cache, chunks=self.chunk_size)
            # Crop to bbox if provided
            if bbox is not None:
                for var in dataset.data_vars:
                    dataset[var] = dataset[var].rio.write_crs(self.target_crs)
                    dataset[var] = dataset[var].rio.clip_box(*bbox)
            return dataset
        
        print("Loading CHELSA data...")
        
        if not self.chelsa_path.exists():
            raise FileNotFoundError(f"CHELSA path not found: {self.chelsa_path}")
        
        tiff_files = list(self.chelsa_path.glob("*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No CHELSA .tif files found in {self.chelsa_path}")
        
        # Define a helper function to process a single CHELSA file
        def process_chelsa_file(tiff_path):
            """Process a single CHELSA file with optimized reprojection."""
            name = tiff_path.stem.split("CHELSA_")[1].split("_1981-2010")[0]
            
            da = rioxarray.open_rasterio(tiff_path, mask_and_scale=True, chunks=self.chunk_size)
            cda = da.sel(band=1, drop=True)
            
            # Optimized reprojection with better parameters
            if da.rio.crs != self.target_crs:
                # Use bilinear resampling for climate data (faster than cubic, good enough for continuous data)
                # Set num_threads for parallel reprojection
                cda = cda.rio.reproject(
                    self.target_crs,
                    resampling=Resampling.bilinear,
                    num_threads=4
                )
                if bbox is not None:
                    cda = cda.rio.clip_box(*bbox)
            elif bbox is not None:
                cda = cda.rio.clip_box(*bbox)
            
            # Cast to float32 to reduce memory usage
            cda = cda.astype(np.float32)
            cda = cda.rename(name)
            
            return name, cda
        
        # Process files in parallel using Dask delayed
        print(f"Processing {len(tiff_files)} CHELSA files in parallel...")
        delayed_results = [delayed(process_chelsa_file)(tiff_path) for tiff_path in tiff_files]
        
        # Compute all files in parallel with progress bar
        with dask.config.set(scheduler='threads', num_workers=4):
            results = dask.compute(*delayed_results)
        
        # Separate names and data arrays
        data_arrays = [result[1] for result in results]
        var_names = [result[0] for result in results]
        
        print(f"Loaded CHELSA variables: {', '.join(var_names)}")
        
        # Align all arrays to a common grid (first array as reference)
        print("Aligning CHELSA arrays to common grid...")
        ref_array = data_arrays[0]
        for i in range(1, len(data_arrays)):
            if not data_arrays[i].coords.equals(ref_array.coords):
                data_arrays[i] = data_arrays[i].interp_like(ref_array)
        
        dataset = xr.merge(data_arrays)
        
        # Save to cache with parallel writing
        if use_cache:
            self.chelsa_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching CHELSA to {self.chelsa_cache}")
            # Compute arrays before saving
            dataset = dataset.compute()
            dataset.to_zarr(self.chelsa_cache)
        
        return dataset
    
    def _load_dem(self, use_cache=False):
        """Load DEM elevation data and reproject to target CRS.
        
        Args:
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.dem_cache.exists():
            print(f"Loading DEM from cache: {self.dem_cache}")
            dataset = xr.open_dataset(self.dem_cache, chunks=self.chunk_size)
            return dataset['elevation']
        
        print("Loading DEM data...")
        
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
        
        with rioxarray.open_rasterio(self.dem_path, mask_and_scale=True, chunks=self.chunk_size) as da:
            dem_da = da.sel(band=1, drop=True)
            
            # Reproject to target CRS if needed (DEM is typically already in EPSG:3035)
            if da.rio.crs != self.target_crs:
                dem_da = dem_da.rio.reproject(self.target_crs)
            
            # Cast to float32 to reduce memory usage
            dem_da = dem_da.astype(np.float32)
            
            dem_da = dem_da.rename('elevation')
            
            # Save to cache with parallel computation
            if use_cache:
                self.dem_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching DEM to {self.dem_cache}")
                dem_da = dem_da.compute()
                dem_da.to_dataset().to_zarr(self.dem_cache)
            
            return dem_da
    
    def _load_landcover(self, bbox=None, use_cache=False):
        """Load and remap Corine Land Cover data and reproject to target CRS.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in target_crs to crop the data.
            use_cache: Whether to use cached data if available.
        """
        # Check cache
        if use_cache and self.lc_cache.exists():
            print(f"Loading landcover from cache: {self.lc_cache}")
            lc_ds = xr.open_dataset(self.lc_cache, chunks=self.chunk_size)
            # Crop to bbox if provided
            if bbox is not None:
                lc_ds['landcover'] = lc_ds['landcover'].rio.write_crs(self.target_crs)
                lc_ds['landcover'] = lc_ds['landcover'].rio.clip_box(*bbox)
            return lc_ds
        
        print("Loading landcover data...")
        
        if not self.lc_path.exists():
            raise FileNotFoundError(f"Landcover file not found: {self.lc_path}")
        
        with rioxarray.open_rasterio(self.lc_path, mask_and_scale=True, chunks=self.chunk_size) as da:
            lc_da = da.sel(band=1, drop=True)
            
            # Reproject to target CRS if needed (Corine is typically already in EPSG:3035)
            if da.rio.crs != self.target_crs:
                lc_da = lc_da.rio.reproject(self.target_crs, nodata=999)
            
            # Crop to bbox if provided
            if bbox is not None:
                lc_da = lc_da.rio.clip_box(*bbox)
            
            # Extract unique landcover classes from the raster
            print("Extracting unique landcover classes from raster...")
            unique_classes = np.unique(lc_da.values)
            # Remove NODATA value (999) if present
            unique_classes = unique_classes[unique_classes != 999]
            # Remove NaN values if present
            unique_classes = unique_classes[~np.isnan(unique_classes)]
            # Convert to integers
            unique_classes = unique_classes.astype(int)
            
            print(f"Found {len(unique_classes)} unique landcover classes: {sorted(unique_classes)}")
            
            # Remap landcover classes to consecutive integers using vectorized operation
            print("Remapping landcover classes to consecutive integers...")
            # Create mapping from original class to remapped index
            sorted_classes = sorted(unique_classes)
            class_mapping = {orig_class: idx for idx, orig_class in enumerate(sorted_classes)}
            
            # Vectorized remapping using numpy's searchsorted (parallelized with dask)
            # Create lookup arrays for efficient remapping
            old_values = np.array(sorted_classes)
            new_values = np.arange(len(sorted_classes))
            
            # Vectorized remapping using searchsorted
            lc_remapped = lc_da.copy()
            flat_values = lc_da.values.flatten()
            indices = np.searchsorted(old_values, flat_values)
            remapped_flat = new_values[indices]
            lc_remapped.values = remapped_flat.reshape(lc_da.shape)
            
            # Cast to int16 to save memory (sufficient for landcover classes)
            lc_remapped = lc_remapped.astype(np.int16)
            
            # Store the mapping as attributes
            lc_remapped.attrs['class_mapping'] = str(class_mapping)
            lc_remapped.attrs['original_classes'] = str(sorted(unique_classes))
            
            print(f"Remapped {len(unique_classes)} classes to indices 0-{len(unique_classes)-1}")
            
            lc_ds = xr.Dataset({'landcover': lc_remapped})
            
            # Save to cache with parallel computation
            if use_cache:
                self.lc_cache.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching landcover to {self.lc_cache}")
                lc_ds = lc_ds.compute()
                lc_ds.to_zarr(self.lc_cache)
            
            return lc_ds
    
    def _save_to_cache(self, dataset):
        """Save dataset as single COG file with ZSTD compression."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(dataset.data_vars)} variables as COG with ZSTD compression...")
        
        # Convert dataset to DataArray by stacking variables as bands
        var_names = list(dataset.data_vars)
        data_arrays = [dataset[var] for var in var_names]
        stacked_da = xr.concat(data_arrays, dim='band')
        stacked_da = stacked_da.assign_coords(band=var_names)
        
        # Ensure CRS is set
        if stacked_da.rio.crs is None:
            stacked_da = stacked_da.rio.write_crs(self.target_crs)
        
        # Save as COG with ZSTD compression
        stacked_da.rio.to_raster(
            self.cache_path,
            driver='COG',
            compress='ZSTD',
            ZSTD_LEVEL=9,
            BIGTIFF='IF_SAFER'
        )
        
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
    dataset = EnvironmentalFeatureDataset()
    ds = dataset.load()

    # Test 2: Load dataset
    print("\n[Test 2] Loading dataset...")
    try:
        print(f"✓ Dataset loaded with {len(ds.data_vars)} variables")
        
        # Test 3: Check CRS consistency
        print("\n[Test 3] Checking CRS consistency...")
        all_vars = list(ds.data_vars)
        if all_vars:
            ref_crs = ds[all_vars[0]].rio.crs
            assert all(ds[var].rio.crs == ref_crs for var in all_vars), "All variables should have same CRS"
            print(f"✓ All variables in {ref_crs}")
        
        # Test 4: Check remapped landcover
        print("\n[Test 4] Checking remapped landcover...")
        assert 'landcover' in ds.data_vars, "Should have landcover variable"
        lc_values = np.unique(ds['landcover'].values)
        lc_values = lc_values[~np.isnan(lc_values)]  # Remove NaN if present
        print(f"✓ Landcover remapped to {len(lc_values)} classes (range: {int(lc_values.min())}-{int(lc_values.max())})")
        if 'class_mapping' in ds['landcover'].attrs:
            print(f"  Class mapping: {ds['landcover'].attrs['class_mapping'][:100]}...")
        
        # Test 5: Check data alignment
        print("\n[Test 5] Checking spatial alignment...")
        shapes = [ds[var].shape for var in all_vars]
        assert len(set(shapes)) == 1, "All variables should have same shape"
        print(f"✓ All variables aligned with shape {shapes[0]}")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"⚠ Test skipped: {e}")
        print("Note: Tests require data files to be present")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
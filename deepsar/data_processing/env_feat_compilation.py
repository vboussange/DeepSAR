"""
Environmental Feature Compilation Module

This module provides functions to compute environmental feature statistics (mean, std)
and landcover fractions for spatial units (polygons) using xarray rasters.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from shapely.geometry import box

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_polygon_env_feature_stats(
    bounds: tuple,
    env_raster: xr.Dataset,
    env_var_names: list,
) -> np.ndarray:
    """
    Compute mean and std of environmental variables for a single polygon.
    
    Args:
        bounds: Tuple (minx, miny, maxx, maxy)
        env_raster: xarray Dataset with environmental variables
        env_var_names: List of variable names to extract
        
    Returns:
        Array of [mean_var1, ..., mean_varN, std_var1, ..., std_varN]
    """
    minx, miny, maxx, maxy = bounds
    
    # Handle y-coordinate ordering (may be descending in rasters)
    y_ascending = env_raster.y.values[0] < env_raster.y.values[-1]
    if y_ascending:
        y_slice = slice(miny, maxy)
    else:
        y_slice = slice(maxy, miny)
    
    # Select spatial subset
    subset = env_raster.sel(x=slice(minx, maxx), y=y_slice)
    
    # Compute statistics for each variable
    means = []
    stds = []
    for var in env_var_names:
        if var in subset.data_vars:
            data = subset[var].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means.append(np.nanmean(data))
                stds.append(np.nanstd(data))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return np.array(means + stds, dtype=np.float32)


def compute_polygon_landcover_stats(
    bounds: tuple,
    lc_raster: xr.Dataset,
    num_lc_classes: int,
) -> np.ndarray:
    """
    Compute one-hot encoded landcover statistics for a single polygon.
    
    Uses memory-efficient incremental counting instead of full one-hot matrix.
    
    Args:
        bounds: Tuple (minx, miny, maxx, maxy)
        lc_raster: xarray Dataset with 'landcover' variable (int16)
        num_lc_classes: Number of landcover classes
        
    Returns:
        Array of [frac_class0, ..., frac_classN] (proportions of each class)
    """
    minx, miny, maxx, maxy = bounds
    
    # Handle y-coordinate ordering
    y_ascending = lc_raster.y.values[0] < lc_raster.y.values[-1]
    if y_ascending:
        y_slice = slice(miny, maxy)
    else:
        y_slice = slice(maxy, miny)
    
    # Select spatial subset
    subset = lc_raster.sel(x=slice(minx, maxx), y=y_slice)
    lc_values = subset['landcover'].values.flatten()
    
    # Remove invalid values (-9999 or negative)
    valid_mask = lc_values >= 0
    lc_values = lc_values[valid_mask]
    
    if len(lc_values) == 0:
        return np.full(num_lc_classes, np.nan, dtype=np.float32)
    
    # Count occurrences efficiently using bincount
    counts = np.bincount(lc_values.astype(np.int32), minlength=num_lc_classes)
    fractions = counts[:num_lc_classes] / len(lc_values)
    
    return fractions.astype(np.float32)


def run_environmental_features_compilation_parallel(
    sp_unit_data: gpd.GeoDataFrame,
    env_raster: xr.Dataset,
    lc_raster: xr.Dataset,
    env_var_names: list,
    num_workers: int = 4,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Compute environmental feature statistics for each spatial unit using parallel processing.
    
    Leverages Dask for chunked raster access and ThreadPoolExecutor for I/O-bound parallelism.
    
    Args:
        sp_unit_data: GeoDataFrame with polygon geometries
        env_raster: xarray Dataset with environmental feature variables
        lc_raster: xarray Dataset with landcover data
        env_var_names: List of environmental feature variable names
        num_workers: Number of parallel workers
        verbose: Whether to show progress
        
    Returns:
        GeoDataFrame with added environmental feature and landcover columns
    """
    logging.info("Compiling environmental features in parallel...")
    
    # Get landcover class info
    lc_attrs = lc_raster['landcover'].attrs
    if 'original_classes' in lc_attrs:
        lc_classes = eval(lc_attrs['original_classes'])
        num_lc_classes = len(lc_classes)
    
    # Generate column names
    env_feature_cols = env_var_names + [f"std_{v}" for v in env_var_names]
    lc_cols = [f"lc_frac_{i}" for i in range(num_lc_classes)]
    
    # Pre-allocate arrays for results
    n_units = len(sp_unit_data)
    env_feature_results = np.zeros((n_units, len(env_feature_cols)), dtype=np.float32)
    lc_results = np.zeros((n_units, num_lc_classes), dtype=np.float32)
    
    # Extract bounds for all polygons
    bounds_list = [geom.bounds for geom in sp_unit_data.geometry]
    
    def process_polygon(args):
        idx, bounds = args
        env_feature_stats = compute_polygon_env_feature_stats(bounds, env_raster, env_var_names)
        lc_stats = compute_polygon_landcover_stats(bounds, lc_raster, num_lc_classes)
        return idx, env_feature_stats, lc_stats
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [(i, b) for i, b in enumerate(bounds_list)]
        results = list(tqdm(
            executor.map(process_polygon, tasks),
            total=n_units,
            desc="Extracting environmental features",
            disable=not verbose,
            miniters=max(n_units // 100, 1),
        ))
    
    # Collect results
    for idx, env_feature_stats, lc_stats in results:
        env_feature_results[idx] = env_feature_stats
        lc_results[idx] = lc_stats
    
    # Add columns to dataframe
    for i, col in enumerate(env_feature_cols):
        sp_unit_data[col] = env_feature_results[:, i]
    for i, col in enumerate(lc_cols):
        sp_unit_data[col] = lc_results[:, i]
    
    return sp_unit_data


if __name__ == "__main__":
    from deepsar.data_processing.utils_eva import EVADataset
    from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
    from deepsar.data_processing.SR_compilation_ckdtree import run_SR_compilation_ckdtree
    
    logging.info("Running tests for env_feat_compilation module...")
    eva_dataset = EVADataset()
    coords, obs_areas, species_matrix, all_species = eva_dataset.load_species_matrix()
    logging.info(f"Loaded {len(coords):,} plots with {species_matrix.shape[1]:,} species")
    
    n_test_sp_units = 10000
    area_range = (1e4, 1e8)  # Smaller range for quick test
    
    test_gdf = run_SR_compilation_ckdtree(
        coords=coords,
        obs_areas=obs_areas,
        species_matrix=species_matrix,
        n_sp_units=n_test_sp_units,
        area_range=area_range,
        crs="EPSG:3035",
        verbose=True,
        random_state=42,
    )
    
    
    # Load actual environmental data
    logging.info("Loading environmental datasets...")
    env_features = EnvironmentalFeatureDataset()
    env_ds, lc_ds = env_features.load(use_cache=True)

    # Test single polygon stats
    logging.info("Testing single polygon stats...")
    bounds = test_gdf.geometry.iloc[1].bounds
    env_stats = compute_polygon_env_feature_stats(bounds, env_ds, list(env_ds.data_vars))
    logging.info(f"Env stats shape: {env_stats.shape}")
    logging.info(f"Env stats: {env_stats}")
    
    num_classes = len(eval(lc_ds["landcover"].attrs["class_mapping"]))
    lc_stats = compute_polygon_landcover_stats(bounds, lc_ds, num_classes)
    
    # Test parallel compilation
    logging.info("Testing parallel compilation...")
    result_gdf = run_environmental_features_compilation_parallel(
        test_gdf, env_ds, lc_ds, list(env_ds.data_vars), num_workers=100
    )
    
    logging.info("Result columns:")
    print(result_gdf.columns)
    logging.info("First row:")
    print(result_gdf.iloc[0])
    
    logging.info("Tests completed successfully.")

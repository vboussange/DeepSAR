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
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compute_polygon_env_feature_stats(
    bounds: tuple,
    env_raster: xr.Dataset,
    env_var_names: list,
) -> np.ndarray:
    """
    Compute raster mean and std within one axis-aligned bounding box.
    
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
    Compute land-cover fractions within one axis-aligned bounding box.
    
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
    
    Uses a thread pool to process spatial units concurrently.
    
    Args:
        sp_unit_data: GeoDataFrame with polygon geometries. Statistics are
            calculated over each geometry's axis-aligned bounding box, so they
            represent the geometry exactly only for axis-aligned rectangles.
        env_raster: xarray Dataset with environmental feature variables
        lc_raster: xarray Dataset with landcover data
        env_var_names: List of environmental feature variable names. If "landcover" is present, landcover fractions are computed.
        num_workers: Number of parallel workers
        verbose: Whether to show progress
        
    Returns:
        GeoDataFrame with added environmental feature and landcover columns
    """
    logging.info("Compiling environmental features in parallel...")
    
    compute_landcover = "landcover" in env_var_names
    # Filter out "landcover" from env_var_names for env_raster processing
    raster_env_vars = [v for v in env_var_names if v != "landcover"]
    
    # Get landcover class info
    num_lc_classes = 0
    lc_cols = []
    if compute_landcover:
        if lc_raster is None or "landcover" not in lc_raster:
            raise ValueError("lc_raster must contain 'landcover' when requested")
        lc_attrs = lc_raster['landcover'].attrs
        if 'original_classes' not in lc_attrs:
            raise ValueError("landcover raster is missing the 'original_classes' attribute")
        lc_classes = literal_eval(lc_attrs['original_classes'])
        num_lc_classes = len(lc_classes)
        if num_lc_classes == 0:
            raise ValueError("landcover raster contains no valid classes")
        lc_cols = [f"lc_frac_{i}" for i in range(num_lc_classes)]
    
    # Generate column names
    env_feature_cols = raster_env_vars + [f"std_{v}" for v in raster_env_vars]
    
    # Pre-allocate arrays for results
    n_units = len(sp_unit_data)
    env_feature_results = np.zeros((n_units, len(env_feature_cols)), dtype=np.float32)
    if compute_landcover:
        lc_results = np.zeros((n_units, num_lc_classes), dtype=np.float32)
    
    # Extract bounds for all polygons
    bounds_list = [geom.bounds for geom in sp_unit_data.geometry]
    
    def process_polygon(args):
        idx, bounds = args
        env_feature_stats = compute_polygon_env_feature_stats(bounds, env_raster, raster_env_vars)
        lc_stats = None
        if compute_landcover:
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
        if compute_landcover:
            lc_results[idx] = lc_stats
    
    # Add columns to dataframe
    for i, col in enumerate(env_feature_cols):
        sp_unit_data[col] = env_feature_results[:, i]
    if compute_landcover:
        for i, col in enumerate(lc_cols):
            sp_unit_data[col] = lc_results[:, i]
    
    return sp_unit_data

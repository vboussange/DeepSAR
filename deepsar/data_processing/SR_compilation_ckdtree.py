"""
Memory-Efficient Species Richness Computation using cKDTree
Spatial queries optimized for large-scale biodiversity datasets.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from shapely.geometry import box
from scipy.spatial import cKDTree

from deepsar.data_processing.utils_eva import EVADataset

def compute_single_square_stats_ckdtree(
    center_coords: np.ndarray,
    half_length: float,
    kdtree: cKDTree,
    coords: np.ndarray,
    obs_areas: np.ndarray,
    species_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """
    Compute SR, observed_area, sp_unit_area for a single square using cKDTree.
    
    Memory-efficient implementation using spatial indexing.
    
    Args:
        center_coords: Center coordinates [x, y]
        half_length: Half side length of the square
        kdtree: Pre-built cKDTree for spatial queries
        coords: All plot coordinates (N, 2) - needed for filtering if not using p=inf
        obs_areas: All observed areas (N,)
        species_matrix: Presence-absence matrix (N, M) - sparse or dense boolean
        rng: Numpy random generator for subsampling
        
    Returns:
        Tuple of (observed_area, sp_unit_area, sr, geometry)
    """
    cx, cy = center_coords
    
    # Define square bounds
    minx, miny = cx - half_length, cy - half_length
    maxx, maxy = cx + half_length, cy + half_length
    
    # Find all plots within the square using cKDTree
    # To strictly comply with "query_ball_point retrieves points within a ball",
    # we query a circumscribed ball (radius = half_length * sqrt(2)) and filter.
    # This guarantees we get all points in the square, then we remove those outside.
    radius = half_length * np.sqrt(2)
    indices_in_ball = kdtree.query_ball_point(center_coords, radius)
    
    if not indices_in_ball:
        geom = box(minx, miny, maxx, maxy)
        return 0.0, (2 * half_length) ** 2, 0, geom
        
    # Filter points to be within the square box
    indices_in_ball = np.array(indices_in_ball)
    points_in_ball = coords[indices_in_ball]
    
    mask = (
        (points_in_ball[:, 0] >= minx) &
        (points_in_ball[:, 0] <= maxx) &
        (points_in_ball[:, 1] >= miny) &
        (points_in_ball[:, 1] <= maxy)
    )
    
    indices_in_box = indices_in_ball[mask]
    num_plots_in_box = len(indices_in_box)
    
    if num_plots_in_box == 0:
        # No plots found - return zero values
        geom = box(minx, miny, maxx, maxy)
        return 0.0, (2 * half_length) ** 2, 0, geom
    
    # Subsample: sample log-uniform number of plots from those in box
    # Sample between 1 and num_plots_in_box (log-uniform)
    if num_plots_in_box == 1:
        n_sample = 1
    else:
        log_n = np.log(num_plots_in_box)
        u = rng.uniform(0, log_n)
        n_sample = max(1, int(np.floor(np.exp(u))))
    
    # Random subset selection
    if n_sample >= num_plots_in_box:
        selected_indices = indices_in_box
    else:
        selected_indices = rng.choice(indices_in_box, size=n_sample, replace=False)
    
    # Compute species richness: union of species across selected plots
    # More memory efficient: iterate if matrix is large
    if species_matrix.shape[0] > 10000:
        # For large matrices, compute union iteratively to save memory
        species_present = np.zeros(species_matrix.shape[1], dtype=bool)
        for idx in selected_indices:
            species_present |= species_matrix[idx]
        sr = np.sum(species_present)
    else:
        # For smaller matrices, vectorize
        species_present = np.any(species_matrix[selected_indices], axis=0)
        sr = np.sum(species_present)
    
    # Compute observed area (sum of areas of selected plots)
    observed_area = np.sum(obs_areas[selected_indices])
    
    # Compute sp_unit_area (square area)
    sp_unit_area = (2.0 * half_length) ** 2
    sp_unit_area = max(sp_unit_area, observed_area)
    
    # Create geometry
    geom = box(minx, miny, maxx, maxy)
    
    return observed_area, sp_unit_area, sr, geom


def run_SR_compilation_ckdtree(
    df: gpd.GeoDataFrame,
    n_sp_units: int,
    area_range: tuple,
    verbose: bool = True,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Compute species richness for random spatial units using cKDTree.
    
    Memory-efficient implementation that processes data in a streaming fashion
    without accumulating large intermediate arrays.
    
    Args:
        df: GeoDataFrame containing plot data (geometry, observed_area, species columns)
        n_sp_units: Number of spatial units to generate
        area_range: Tuple of (min_area, max_area) for random squares
        verbose: Whether to show progress bar
        random_state: Random seed
        
    Returns:
        GeoDataFrame with columns: observed_area, sp_unit_area, sr, geometry
    """
    # Extract arrays from DataFrame
    coords = np.column_stack((df.geometry.x, df.geometry.y))
    obs_areas = df['area_m2'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values

    logging.info("Building cKDTree for spatial queries...")
    kdtree = cKDTree(coords)
    
    # Initialize random generator
    rng = np.random.default_rng(random_state)
    
    n_plots = len(coords)
    log_area_min, log_area_max = np.log(area_range[0]), np.log(area_range[1])
    
    # Pre-allocate result arrays
    observed_areas = np.empty(n_sp_units, dtype=np.float64)
    sp_unit_areas = np.empty(n_sp_units, dtype=np.float64)
    srs = np.empty(n_sp_units, dtype=np.int32)
    geometries = []
    
    logging.info(f"Generating {n_sp_units} spatial units...")
    
    # Process each spatial unit one at a time (streaming approach)
    for i in tqdm(range(n_sp_units), desc="Compiling SR (cKDTree)", disable=not verbose):
        # Sample center plot index (uniform over all plots)
        center_idx = rng.integers(0, n_plots)
        center_coords = coords[center_idx]
        
        # Sample area from log-uniform distribution
        u = rng.uniform(0, 1)
        log_area = log_area_min + u * (log_area_max - log_area_min)
        area = np.exp(log_area)
        half_length = np.sqrt(area) / 2
        
        # Compute stats for this spatial unit
        obs_area, sp_unit_area, sr, geom = compute_single_square_stats_ckdtree(
            center_coords,
            half_length,
            kdtree,
            coords,
            obs_areas,
            species_matrix,
            rng,
        )
        
        # Store results
        observed_areas[i] = obs_area
        sp_unit_areas[i] = sp_unit_area
        srs[i] = sr
        geometries.append(geom)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "observed_area": observed_areas,
            "sp_unit_area": sp_unit_areas,
            "sr": srs,
        },
        geometry=geometries,
    )
    
    # Filter out any squares with sr=0
    initial_count = len(gdf)
    gdf = gdf[gdf.sr > 0].reset_index(drop=True)
    filtered_count = initial_count - len(gdf)
    
    if filtered_count > 0:
        logging.info(f"Filtered out {filtered_count} spatial units with SR=0")
    
    logging.info(f"Generated {len(gdf)} spatial units with SR > 0")
    return gdf



if __name__ == "__main__":
    eva_dataset = EVADataset()
    df = eva_dataset.load_species_matrix()
    coords = np.column_stack((df.geometry.x, df.geometry.y))
    obs_areas = df['area_m2'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values

    rng = np.random.default_rng(42)
    kdtree = cKDTree(coords)
    
    test_center_idx = 0
    test_half_length = 5000.0  # 5km half-length -> 10km x 10km square
    
    # Test 1: Single square computation
    obs_area, sp_unit_area, sr, geom = compute_single_square_stats_ckdtree(
        coords[test_center_idx],
        test_half_length,
        kdtree,
        coords,
        obs_areas,
        species_matrix,
        rng,
    )
    
    # Test 2: Small-scale compilation (streaming approach)    
    n_test_sp_units = 100000
    area_range = (1e4, 1e8)  # Smaller range for quick test
    
    test_gdf = run_SR_compilation_ckdtree(
        df=df,
        n_sp_units=n_test_sp_units,
        area_range=area_range,
        verbose=True,
        random_state=42,
    )
"""
JAX-accelerated Species Richness Computation for EVA Training Samples
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm

from equinox import filter_jit, filter_vmap
import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import sparse
import scipy.sparse

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# JAX-based Species Richness Computation
# =============================================================================

def compute_single_square_stats(
    center_idx: int,
    half_length: float,
    key: jax.Array,
    coords: jax.Array,
    obs_areas: jax.Array,
    species_matrix_T: sparse.BCOO,
) -> tuple:
    """
    Compute SR, observed_area, sp_unit_area for a single square centered on a plot.
    
    The square is centered on the plot at center_idx, guaranteeing at least one plot inside.
    
    Args:
        center_idx: Index of the center plot
        half_length: Half side length of the square
        key: JAX random key for subsampling
        coords: All plot coordinates (N, 2)
        obs_areas: All observed areas (N,)
        species_matrix_T: Transposed Presence-absence matrix (M, N) in sparse BCOO format
        
    Returns:
        Tuple of (observed_area, sp_unit_area, sr)
    """
    # Get center coordinates
    cx = coords[center_idx, 0]
    cy = coords[center_idx, 1]
    n_plots = coords.shape[0]
    
    # Find all plots within the square using simple boolean masking
    in_box = (
        (coords[:, 0] >= cx - half_length) &
        (coords[:, 0] <= cx + half_length) &
        (coords[:, 1] >= cy - half_length) &
        (coords[:, 1] <= cy + half_length)
    )
    
    # Count plots in box
    num_plots_in_box = jnp.sum(in_box)
    
    # Subsample: sample log-uniform number of plots from those in box
    key1, key2 = jax.random.split(key)
    log_n = jax.random.uniform(key1) * jnp.log(jnp.maximum(num_plots_in_box, 1).astype(jnp.float32))
    n_sample = jnp.maximum(1, jnp.floor(jnp.exp(log_n)).astype(jnp.int32))
    
    # Random subset selection using uniform noise ranking
    # Assign random scores to each plot, set out-of-box plots to -inf
    noise = jax.random.uniform(key2, shape=(n_plots,))
    scores = jnp.where(in_box, noise, -1.0)
    
    # Select top n_sample by finding threshold score
    # Sort scores descending and get the n_sample-th value as threshold
    sorted_scores = jnp.sort(scores)[::-1]
    # Use n_sample-1 because 0-indexed, clamp to valid range
    threshold_idx = jnp.minimum(n_sample - 1, num_plots_in_box - 1)
    threshold_idx = jnp.maximum(threshold_idx, 0)
    threshold = sorted_scores[threshold_idx]
    
    # Select plots with score >= threshold (handles ties by including all at threshold)
    selected_mask = (scores >= threshold) & in_box
    
    # Compute species richness: union of species across selected plots
    # Use sparse matrix multiplication for efficiency: (M, N) @ (N,) -> (M,)
    # species_matrix_T is sparse (M, N), selected_mask is dense (N,)
    species_counts = species_matrix_T @ selected_mask.astype(jnp.float32)
    sr = jnp.sum(species_counts > 0)
    
    # Compute observed area (sum of areas of selected plots)
    observed_area = jnp.sum(jnp.where(selected_mask, obs_areas, 0.0))
    
    # Compute sp_unit_area (square area)
    sp_unit_area = (2.0 * half_length) ** 2
    sp_unit_area = jnp.maximum(sp_unit_area, observed_area)
        
    return observed_area, sp_unit_area, sr


# Vectorized version over batch of squares
@filter_jit
def compute_batch_stats(
    center_indices: jax.Array,
    half_lengths: jax.Array,
    keys: jax.Array,
    coords: jax.Array,
    obs_areas: jax.Array,
    species_matrix_T: sparse.BCOO,
) -> tuple:
    """Vectorized computation over a batch of squares using vmap."""
    return vmap(
        compute_single_square_stats,
        in_axes=(0, 0, 0, None, None, None)
    )(center_indices, half_lengths, keys, coords, obs_areas, species_matrix_T)


def run_SR_compilation_jax(
    coords: np.ndarray,
    obs_areas: np.ndarray,
    species_matrix: np.ndarray,
    n_sp_units: int,
    area_range: tuple,
    crs: str = "EPSG:3035",
    batch_size: int = 100,
    verbose: bool = True,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Compute species richness for random spatial units using JAX acceleration.
    
    Generates random squares centered on existing plots (guaranteeing ≥1 plot per square),
    then computes SR and area statistics using vectorized JAX operations.
    
    Memory-efficient implementation that properly manages device/host memory transfers.
    
    Args:
        coords: Array (N, 2) of plot coordinates [x, y]
        obs_areas: Array (N,) of observed areas per plot
        species_matrix: Boolean array (N, M) presence-absence matrix
        n_sp_units: Number of spatial units to generate
        area_range: Tuple of (min_area, max_area) for random squares
        crs: Coordinate reference system
        batch_size: Batch size for processing
        verbose: Whether to show progress bar
        random_state: Random seed
        
    Returns:
        GeoDataFrame with columns: observed_area, sp_unit_area, sr, geometry
    """
    from shapely.geometry import box
    
    logging.info("Compiling SR using JAX vectorized operations...")
    
    n_plots = len(coords)
    log_area_min, log_area_max = np.log(area_range[0]), np.log(area_range[1])
    
    # Convert to JAX arrays (these stay on device)
    coords_jax = jnp.array(coords, dtype=jnp.float32)
    obs_areas_jax = jnp.array(obs_areas, dtype=jnp.float32)
    
    # Convert species matrix to sparse BCOO format
    logging.info("Converting species matrix to sparse format...")
    # Ensure input is numpy array
    if not isinstance(species_matrix, np.ndarray):
        species_matrix = np.array(species_matrix)
        
    sp_sparse = scipy.sparse.coo_matrix(species_matrix)
    values = sp_sparse.data
    shape = sp_sparse.shape
    
    # Create BCOO matrix (M, N) - Transposed for efficient column access
    # We want (M, N) so we can do (M, N) @ (N,) -> (M,)
    # Original is (N, M). Transpose swaps row/col.
    # So we use col as row, row as col.
    indices_T = np.vstack((sp_sparse.col, sp_sparse.row)).T
    shape_T = (shape[1], shape[0])
    
    species_matrix_T_jax = sparse.BCOO(
        (jnp.array(values, dtype=jnp.float32), jnp.array(indices_T, dtype=jnp.int32)),
        shape=shape_T
    )
    
    # Initialize random key
    rng_key = jax.random.key(random_state)
    
    # Pre-allocate numpy arrays for results (on host)
    observed_areas = np.empty(n_sp_units, dtype=np.float32)
    sp_unit_areas = np.empty(n_sp_units, dtype=np.float32)
    srs = np.empty(n_sp_units, dtype=np.int32)
    center_coords = np.empty((n_sp_units, 2), dtype=np.float32)
    half_lengths_all = np.empty(n_sp_units, dtype=np.float32)
    
    # Process in batches
    n_batches = (n_sp_units + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Compiling SR (JAX)", disable=not verbose):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, n_sp_units)
        current_batch_size = batch_end - batch_start
        
        # Generate random parameters for this batch
        rng_key, key1, key2, key3 = jax.random.split(rng_key, 4)
        
        # Sample center plot indices (uniform over all plots)
        center_indices = jax.random.randint(key1, (current_batch_size,), 0, n_plots)
        
        # Sample areas from log-uniform distribution
        u = jax.random.uniform(key2, (current_batch_size,))
        log_areas = log_area_min + u * (log_area_max - log_area_min)
        half_lengths = jnp.exp(log_areas / 2) / 2  # sqrt(area) / 2
        
        # Generate keys for subsampling within each square
        batch_keys = jax.random.split(key3, current_batch_size)
        
        # Compute stats for all squares in batch (vectorized)
        obs_areas_batch, sp_unit_areas_batch, srs_batch = compute_batch_stats(
            center_indices, half_lengths, batch_keys,
            coords_jax, obs_areas_jax, species_matrix_T_jax
        )
        
        # CRITICAL: Transfer results from device to host immediately using block_until_ready
        # This prevents memory accumulation on the device
        obs_areas_batch.block_until_ready()
        observed_areas[batch_start:batch_end] = np.asarray(obs_areas_batch)
        sp_unit_areas[batch_start:batch_end] = np.asarray(sp_unit_areas_batch)
        srs[batch_start:batch_end] = np.asarray(srs_batch)
        
        # Store geometry parameters
        center_coords[batch_start:batch_end] = np.asarray(coords_jax[center_indices])
        half_lengths_all[batch_start:batch_end] = np.asarray(half_lengths)
        
        # Delete JAX arrays to free device memory
        del obs_areas_batch, sp_unit_areas_batch, srs_batch
        del center_indices, half_lengths, batch_keys, u, log_areas
    
    # Clear JAX cache to free any remaining device memory
    jax.clear_caches()
    
    # Build geometries from stored parameters
    logging.info("Building geometries...")
    geometries = [
        box(cx - hl, cy - hl, cx + hl, cy + hl)
        for (cx, cy), hl in zip(center_coords, half_lengths_all)
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "observed_area": observed_areas,
            "sp_unit_area": sp_unit_areas,
            "sr": srs,
        },
        geometry=geometries,
        crs=crs,
    )
    
    # Filter out any squares with sr=0 (shouldn't happen but safety check)
    gdf = gdf[gdf.sr > 0].reset_index(drop=True)
    
    logging.info(f"Generated {len(gdf)} spatial units with SR > 0")
    return gdf


if __name__ == "__main__":
    logging.info("Starting test of JAX-based species richness computation...")
    
    # Load EVA dataset
    logging.info("Loading EVA data...")
    eva_dataset = EVADataset()
    df = eva_dataset.load_species_matrix()
    coords = np.column_stack((df.geometry.x, df.geometry.y))
    obs_areas = df['observed_area'].values
    species_list = df.attrs['species_list']
    species_matrix = df[species_list].values
    logging.info(f"Loaded {len(coords):,} plots with {species_matrix.shape[1]:,} species")
    
    # Convert to JAX arrays for single test
    coords_jax = jnp.array(coords, dtype=jnp.float32)
    obs_areas_jax = jnp.array(obs_areas, dtype=jnp.float32)
    
    sp_sparse = scipy.sparse.coo_matrix(species_matrix)
    indices_T = np.vstack((sp_sparse.col, sp_sparse.row)).T
    shape_T = (sp_sparse.shape[1], sp_sparse.shape[0])
    species_matrix_T_jax = sparse.BCOO(
        (jnp.array(sp_sparse.data, dtype=jnp.float32), jnp.array(indices_T, dtype=jnp.int32)),
        shape=shape_T
    )
    
    # Set test parameters
    test_center_idx = 0
    test_half_length = 5000.0  # 5km half-length -> 10km x 10km square
    test_key = jax.random.key(18)
    n_plots = len(coords)
    
    # Compute stats for a single square
    obs_area, sp_unit_area, sr = compute_single_square_stats(
        test_center_idx,
        test_half_length,
        test_key,
        coords_jax,
        obs_areas_jax,
        species_matrix_T_jax,
    )
    
    logging.info(f"Single square results:")
    logging.info(f"  Center coordinates: {coords[test_center_idx]}")
    logging.info(f"  Square half-length: {test_half_length} m")
    logging.info(f"  Observed area: {obs_area:.2f} m²")
    logging.info(f"  Spatial unit area: {sp_unit_area:.2f} m²")
    logging.info(f"  Species richness: {sr}")
    
    # Test 2: Full compilation with run_SR_compilation_jax
    logging.info("\n=== Test 2: Full compilation with run_SR_compilation_jax ===")
    
    # Generate a small test dataset
    n_test_sp_units = 1000
    area_range = (1e4, 1e8)  # Smaller range for quick test
    
    test_gdf = run_SR_compilation_jax(
        coords=coords,
        obs_areas=obs_areas,
        species_matrix=species_matrix,
        n_sp_units=n_test_sp_units,
        area_range=area_range,
        crs="EPSG:3035",
        batch_size=1000,
        verbose=True,
        random_state=42,
    )
    
    # Display summary statistics
    logging.info(f"\nCompilation results:")
    logging.info(f"  Total spatial units generated: {len(test_gdf)}")
    logging.info(f"  Species richness range: [{test_gdf['sr'].min()}, {test_gdf['sr'].max()}]")
    logging.info(f"  Mean SR: {test_gdf['sr'].mean():.2f}")
    logging.info(f"  Observed area range: [{test_gdf['observed_area'].min():.2e}, {test_gdf['observed_area'].max():.2e}] m²")
    logging.info(f"  Spatial unit area range: [{test_gdf['sp_unit_area'].min():.2e}, {test_gdf['sp_unit_area'].max():.2e}] m²")
    
    # Display first few rows
    logging.info(f"\nFirst few rows of compiled data:")
    print(test_gdf.head())
    
    logging.info("\n=== All tests completed successfully! ===")
    
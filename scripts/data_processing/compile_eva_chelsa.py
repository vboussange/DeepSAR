"""
Compiles training samples based on EVA and CHELSA data.

This script generates training data for species-area relationship models by:
1. Generating random spatial units (polygons) across the landscape
2. Computing species richness within each polygon using JAX-accelerated operations
3. Extracting environmental feature statistics (mean, std) for each polygon
4. Handling categorical landcover data via efficient one-hot encoding

Optimized for large-scale datasets using:
- JAX for vectorized species richness computations
- Dask for parallel raster processing  
- Memory-efficient one-hot encoding for landcover
- GeoParquet for efficient storage
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import json

import jax
import jax.numpy as jnp
from jax import jit, vmap

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset

import git
import random

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../data/processed/EVA_CHELSA_compilation/",
    ),
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
    ],
    "block_length": 1e6,  # in meters
    "area_range_train": (1e4, 1e12),  # in m2
    "area_range_test": (1e4, 1e10),  # in m2
    "sp_unit_test_size": 0.005,  # in fraction of total EVA records
    "sp_unit_train_size": 3,  # in fraction of total EVA train records
    "crs": "EPSG:3035",
    "random_state": 2,
    "verbose": True,
    "batch_size": 1000,  # batch size for JAX operations
    "dask_chunks": {"x": 2000, "y": 2000},  # chunk size for dask arrays
    "num_workers": 8,  # number of parallel workers for climate compilation
}

# Define covariate feature names based on environmental covariates
# mean_labels = CONFIG["env_vars"]
# std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
# CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()


# =============================================================================
# JAX-based Species Richness Computation
# =============================================================================

@partial(jit, static_argnums=(3,))
def compute_single_square_stats(
    center_idx: int,
    half_length: float,
    key: jax.Array,
    n_plots: int,
    coords: jax.Array,
    obs_areas: jax.Array,
    species_matrix: jax.Array,
) -> tuple:
    """
    Compute SR, observed_area, sp_unit_area for a single square centered on a plot.
    
    The square is centered on the plot at center_idx, guaranteeing at least one plot inside.
    
    Args:
        center_idx: Index of the center plot
        half_length: Half side length of the square
        key: JAX random key for subsampling
        n_plots: Total number of plots (static for JIT)
        coords: All plot coordinates (N, 2)
        obs_areas: All observed areas (N,)
        species_matrix: Presence-absence matrix (N, M)
        
    Returns:
        Tuple of (observed_area, sp_unit_area, sr, num_plots)
    """
    # Get center coordinates
    cx = coords[center_idx, 0]
    cy = coords[center_idx, 1]
    
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
    masked_species = species_matrix & selected_mask[:, None]
    species_present = jnp.any(masked_species, axis=0)
    sr = jnp.sum(species_present)
    
    # Compute observed area (sum of areas of selected plots)
    observed_area = jnp.sum(jnp.where(selected_mask, obs_areas, 0.0))
    
    # Compute sp_unit_area (square area)
    sp_unit_area = (2.0 * half_length) ** 2
    sp_unit_area = jnp.maximum(sp_unit_area, observed_area)
    
    num_selected = jnp.sum(selected_mask)
    
    return observed_area, sp_unit_area, sr, num_selected


# Vectorized version over batch of squares
@partial(jit, static_argnums=(3,))
def compute_batch_stats(
    center_indices: jax.Array,
    half_lengths: jax.Array,
    keys: jax.Array,
    n_plots: int,
    coords: jax.Array,
    obs_areas: jax.Array,
    species_matrix: jax.Array,
) -> tuple:
    """Vectorized computation over a batch of squares using vmap."""
    return vmap(
        lambda idx, hl, k: compute_single_square_stats(
            idx, hl, k, n_plots, coords, obs_areas, species_matrix
        )
    )(center_indices, half_lengths, keys)


def run_SR_compilation_jax(
    coords: np.ndarray,
    obs_areas: np.ndarray,
    species_matrix: np.ndarray,
    n_sp_units: int,
    area_range: tuple,
    crs: str = CONFIG["crs"],
    batch_size: int = CONFIG["batch_size"],
    verbose: bool = CONFIG["verbose"],
    random_state: int = CONFIG["random_state"],
) -> gpd.GeoDataFrame:
    """
    Compute species richness for random spatial units using JAX acceleration.
    
    Generates random squares centered on existing plots (guaranteeing ≥1 plot per square),
    then computes SR and area statistics using vectorized JAX operations.
    
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
        GeoDataFrame with columns: observed_area, sp_unit_area, sr, num_plots, geometry
    """
    from shapely.geometry import box
    
    logging.info("Compiling SR using JAX vectorized operations...")
    
    n_plots = len(coords)
    log_area_min, log_area_max = np.log(area_range[0]), np.log(area_range[1])
    
    # Convert to JAX arrays
    coords_jax = jnp.array(coords, dtype=jnp.float32)
    obs_areas_jax = jnp.array(obs_areas, dtype=jnp.float32)
    species_matrix_jax = jnp.array(species_matrix, dtype=bool)
    
    # Initialize random key
    rng_key = jax.random.PRNGKey(random_state)
    
    # Pre-allocate results
    all_observed_areas = []
    all_sp_unit_areas = []
    all_srs = []
    all_num_plots = []
    all_centers = []
    all_half_lengths = []
    
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
        obs_areas_batch, sp_unit_areas_batch, srs_batch, num_plots_batch = compute_batch_stats(
            center_indices, half_lengths, batch_keys,
            n_plots, coords_jax, obs_areas_jax, species_matrix_jax
        )
        
        # Store results
        all_observed_areas.append(np.array(obs_areas_batch))
        all_sp_unit_areas.append(np.array(sp_unit_areas_batch))
        all_srs.append(np.array(srs_batch))
        all_num_plots.append(np.array(num_plots_batch))
        all_centers.append(np.array(coords_jax[center_indices]))
        all_half_lengths.append(np.array(half_lengths))
    
    # Concatenate all batches
    observed_areas = np.concatenate(all_observed_areas)
    sp_unit_areas = np.concatenate(all_sp_unit_areas)
    srs = np.concatenate(all_srs)
    num_plots = np.concatenate(all_num_plots)
    centers = np.concatenate(all_centers)
    half_lengths = np.concatenate(all_half_lengths)
    
    # Build geometries (this part stays in numpy/shapely)
    geometries = [
        box(cx - hl, cy - hl, cx + hl, cy + hl)
        for (cx, cy), hl in zip(centers, half_lengths)
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "observed_area": observed_areas,
            "sp_unit_area": sp_unit_areas,
            "sr": srs,
            "num_plots": num_plots,
        },
        geometry=geometries,
        crs=crs,
    )
    
    # Filter out any squares with sr=0 (shouldn't happen but safety check)
    gdf = gdf[gdf.sr > 0].reset_index(drop=True)
    
    logging.info(f"Generated {len(gdf)} spatial units with SR > 0")
    return gdf


# =============================================================================
# Dask-parallel Climate Compilation
# =============================================================================

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
    env_var_names: list = CONFIG["env_vars"],
    num_workers: int = CONFIG["num_workers"],
    verbose: bool = CONFIG["verbose"],
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
    else:
        num_lc_classes = int(lc_raster['landcover'].max().values) + 1
    
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

# =============================================================================
# Main Compilation Pipeline
# =============================================================================

def run_sp_unit_compilation(
    coords: np.ndarray,
    obs_areas: np.ndarray,
    species_matrix: np.ndarray,
    env_raster: xr.Dataset,
    lc_raster: xr.Dataset,
    n_sp_units: int,
    area_range: tuple,
    crs: str = CONFIG["crs"],
    verbose: bool = CONFIG["verbose"],
) -> gpd.GeoDataFrame:
    """
    Full pipeline: generate spatial units, compute SR, and extract environmental features.
    
    Args:
        coords: Plot coordinates array
        obs_areas: Observed areas array
        species_matrix: Species presence-absence matrix
        env_raster: Climate xarray Dataset
        lc_raster: Landcover xarray Dataset
        n_sp_units: Number of spatial units to generate
        area_range: Area range for random squares
        crs: Coordinate reference system
        verbose: Whether to show progress
        
    Returns:
        Compiled GeoDataFrame with SR and environmental features
    """
    # Step 1: Generate spatial units and compute SR
    sp_unit_data = run_SR_compilation_jax(
        coords, obs_areas, species_matrix, n_sp_units, area_range, crs, verbose=verbose
    )
    
    # Step 2: Validate SR
    assert (sp_unit_data.sr > 0).all(), "Found spatial units with zero species richness"
    
    # Step 3: Extract environmental features
    sp_unit_data = run_environmental_features_compilation_parallel(
        sp_unit_data, env_raster, lc_raster, verbose=verbose
    )
    
    return sp_unit_data


def export_dataset_statistics(
    coords: np.ndarray,
    species_matrix: np.ndarray,
    output_file_path: Path,
) -> None:
    """
    Calculate and export dataset statistics to a text file.
    
    Args:
        coords: Plot coordinates
        species_matrix: Species presence-absence matrix
        output_file_path: Output directory path
    """
    logging.info("Calculating dataset statistics...")
    
    num_plots = len(coords)
    num_species = species_matrix.shape[1]
    num_presences = species_matrix.sum()
    avg_sr_per_plot = species_matrix.sum(axis=1).mean()
    
    stats_file_path = output_file_path / "dataset_statistics.txt"
    logging.info(f"Exporting dataset statistics to {stats_file_path}")
    
    with open(stats_file_path, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("==================\n")
        f.write(f"Number of plots: {num_plots:,}\n")
        f.write(f"Number of distinct species: {num_species:,}\n")
        f.write(f"Total presences: {num_presences:,}\n")
        f.write(f"Average species per plot: {avg_sr_per_plot:.1f}\n")
        f.write(f"Occupancy rate: {num_presences / (num_plots * num_species) * 100:.3f}%\n")


def save_compiled_data(
    sp_unit_data: gpd.GeoDataFrame,
    output_path: Path,
    partition_name: str = "train",
) -> None:
    """
    Save compiled spatial unit data to GeoParquet format.
    
    GeoParquet provides:
    - Efficient columnar storage with compression
    - Fast partial reads
    - Native geometry support
    - Compatibility with Dask-GeoPandas for distributed processing
    
    Args:
        sp_unit_data: GeoDataFrame to save
        output_path: Directory to save to
        partition_name: Name for the partition (e.g., 'train', 'test')
    """
    file_path = output_path / f"{partition_name}_data.parquet"
    logging.info(f"Saving compiled data to {file_path}")
    
    sp_unit_data.to_parquet(
        file_path,
        compression='zstd',  # Good compression ratio and speed
        index=False,
    )
    
    # Also save a lightweight summary
    summary_path = output_path / f"{partition_name}_summary.json"
    summary = {
        "n_samples": len(sp_unit_data),
        "columns": list(sp_unit_data.columns),
        "sr_range": [int(sp_unit_data.sr.min()), int(sp_unit_data.sr.max())],
        "area_range": [float(sp_unit_data.sp_unit_area.min()), float(sp_unit_data.sp_unit_area.max())],
        "crs": str(sp_unit_data.crs),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Set up random seeds for reproducibility
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    
    # Set up output directory with git hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    output_file_path = CONFIG["output_file_path"] / sha
    output_file_path.mkdir(parents=True, exist_ok=True)
    
    # Load EVA data (species matrix format)
    logging.info("Loading EVA data...")
    coords, obs_areas, species_matrix, species_list = EVADataset().load_species_matrix()
    logging.info(f"Loaded {len(coords):,} plots with {species_matrix.shape[1]:,} species")
    
    # Load environmental rasters
    logging.info("Loading environmental rasters...")
    env_features = EnvironmentalFeatureDataset()
    chelsa_dem_ds, lc_ds = env_features.load(use_cache=True)
    
    # Export dataset statistics
    export_dataset_statistics(coords, species_matrix, output_file_path)
    
    # Compute number of spatial units
    n_plots = len(coords)
    n_sp_units_test = int(n_plots * CONFIG["sp_unit_test_size"])
    n_sp_units_train = int(n_plots * CONFIG["sp_unit_train_size"])
    
    logging.info(f"Generating {n_sp_units_train:,} training and {n_sp_units_test:,} test spatial units...")
    
    # Generate training data
    logging.info("Compiling training data...")
    train_data = run_sp_unit_compilation(
        coords, obs_areas, species_matrix,
        chelsa_dem_ds, lc_ds,
        n_sp_units=n_sp_units_train,
        area_range=CONFIG["area_range_train"],
    )
    save_compiled_data(train_data, output_file_path, "train")
    
    # Generate test data
    logging.info("Compiling test data...")
    test_data = run_sp_unit_compilation(
        coords, obs_areas, species_matrix,
        chelsa_dem_ds, lc_ds,
        n_sp_units=n_sp_units_test,
        area_range=CONFIG["area_range_test"],
    )
    save_compiled_data(test_data, output_file_path, "test")
    
    # Final summary
    logging.info("=" * 60)
    logging.info("Compilation complete!")
    logging.info(f"Training samples: {len(train_data):,}")
    logging.info(f"Test samples: {len(test_data):,}")
    logging.info(f"Output directory: {output_file_path}")
    logging.info("=" * 60)

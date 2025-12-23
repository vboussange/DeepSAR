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
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor
import json

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
from deepsar.data_processing.SR_compilation_ckdtree import run_SR_compilation_ckdtree
from deepsar.data_processing.env_feat_compilation import run_environmental_features_compilation_parallel

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
        f"../../data/processed/training_samples/",
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
    "area_range": (1e4, 1e8),  # in m2
    "crs": "EPSG:3035",
    "random_state": 2,
    "verbose": True,
    "batch_size": 100,  # batch size for JAX operations
    "num_workers": 100,  # number of parallel workers for climate compilation
}

# Define covariate feature names based on environmental covariates
# mean_labels = CONFIG["env_vars"]
# std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
# CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()

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
    env_var_names: list = CONFIG["env_vars"],
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
        env_var_names: List of environmental variables to extract
        
    Returns:
        Compiled GeoDataFrame with SR and environmental features
    """
    # Step 1: Generate spatial units and compute SR
    sp_unit_data = run_SR_compilation_ckdtree(
        coords, obs_areas, species_matrix, n_sp_units, area_range, crs, verbose=verbose
    )
    
    # Step 2: Validate SR
    assert (sp_unit_data.sr > 0).all(), "Found spatial units with zero species richness"
    
    # Step 3: Extract environmental features
    sp_unit_data = run_environmental_features_compilation_parallel(
        sp_unit_data, env_raster, lc_raster, env_var_names, verbose=verbose
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
    coords, obs_areas, species_matrix, _ = EVADataset().load_species_matrix()
    logging.info(f"Loaded {len(coords):,} plots with {species_matrix.shape[1]:,} species")
    
    # Load environmental rasters
    logging.info("Loading environmental rasters...")
    env_features = EnvironmentalFeatureDataset()
    chelsa_dem_ds, lc_ds = env_features.load(use_cache=True)
    
    # Export dataset statistics
    export_dataset_statistics(coords, species_matrix, output_file_path)
    
    # Generate training data
    logging.info("Compiling training data...")
    train_data = run_sp_unit_compilation(
        coords, obs_areas, species_matrix,
        chelsa_dem_ds, lc_ds,
        n_sp_units=1000,
        area_range=CONFIG["area_range"],
        env_var_names=CONFIG["env_vars"],
    )
    save_compiled_data(train_data, output_file_path, "train")
    
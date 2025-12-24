"""
Compiles training samples based on EVA and environmental feature data using Spatial Block Cross-Validation.

This script generates training data by:
1. Loading EVA species data.
2. Assigning spatial blocks (checkerboard) to each plot.
3. Splitting data into `n_splits` folds based on spatial blocks.
4. For each fold:
    - Generating random spatial units (polygons) for training (using training plots).
    - Generating random spatial units (polygons) for testing (using testing plots).
    - Computing species richness within each polygon.
    - Extracting environmental feature statistics.
    - Saving the datasets.
"""

import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import json
import git
import random

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
from deepsar.data_processing.SR_compilation_ckdtree import run_SR_compilation_ckdtree
from deepsar.data_processing.env_feat_compilation import run_environmental_features_compilation_parallel

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../data/processed/training_samples/sbcv",
    ),
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
        "elevation",
        "landcover",
    ],
    "spunit_area_range_test": (2e3**2, 1e5**2),  # in m2
    "spunit_area_range_train": (2e3**2, 1e6**2),  # in m2
    "random_state": 2,
    "verbose": True,
    "num_workers": 100,  # number of parallel workers for env feature compilation
    "n_splits": 5, # number of spatial folds
    "block_size": 20_000, # Block size in meters (e.g., 20km x 20km)
    "ratio_samples_plots": 0.1, # ratio of genrated train/val/test samples to raw plots
}

def assign_checkerboard_folds(gdf, n_splits=5, block_size=10000):
    """
    Assigns spatial folds to a GeoDataFrame using a checkerboard pattern.
    
    Args:
        gdf: GeoDataFrame with plot data
        n_splits: Number of folds
        block_size: Size of the checkerboard blocks in meters (assuming projected CRS)
    """
    # Calculate bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Assign grid indices
    # Use floor to get grid index
    grid_x = np.floor((gdf.geometry.x - minx) / block_size).astype(int)
    grid_y = np.floor((gdf.geometry.y - miny) / block_size).astype(int)
    
    gdf['grid_x'] = grid_x
    gdf['grid_y'] = grid_y
    
    # Assign folds (checkerboard pattern)
    # (x + y) % n_splits creates diagonal stripes
    gdf['spatial_split'] = (gdf['grid_x'] + gdf['grid_y']) % n_splits
    
    return gdf

def run_sp_unit_compilation(
    df: gpd.GeoDataFrame,
    env_raster: xr.Dataset,
    lc_raster: xr.Dataset,
    n_sp_units: int,
    area_range: tuple,
    verbose: bool = CONFIG["verbose"],
    env_var_names: list = CONFIG["env_vars"],
) -> gpd.GeoDataFrame:
    """
    Full pipeline: generate spatial units, compute SR, and extract environmental features.
    """
    # Step 1: Generate spatial units and compute SR
    sp_unit_data = run_SR_compilation_ckdtree(
        df, n_sp_units, area_range, verbose=verbose
    )
    
    # Step 2: Validate SR
    assert (sp_unit_data.sr > 0).all(), "Found spatial units with zero species richness"
    
    # Step 3: Extract environmental features
    sp_unit_data = run_environmental_features_compilation_parallel(
        sp_unit_data, env_raster, lc_raster, env_var_names, num_workers=CONFIG["num_workers"], verbose=verbose
    )
    
    return sp_unit_data

def save_compiled_data(
    sp_unit_data: gpd.GeoDataFrame,
    output_path: Path,
    filename: str,
) -> None:
    """
    Save compiled spatial unit data to GeoParquet format.
    """
    file_path = output_path / f"{filename}.parquet"
    logging.info(f"Saving compiled data to {file_path}")
    
    sp_unit_data.to_parquet(
        file_path,
        compression='zstd',
        index=False,
    )
    
    # Also save a lightweight summary
    summary_path = output_path / f"{filename}_summary.json"
    summary = {
        "ratio_samples_plots": CONFIG["ratio_samples_plots"],
        "environmental_variables": CONFIG["env_vars"],
        "n_samples": len(sp_unit_data),
        "columns": list(sp_unit_data.columns),
        "sr_range": [int(sp_unit_data.sr.min()), int(sp_unit_data.sr.max())],
        "spunit_area_range": [float(sp_unit_data.sp_unit_area.min()), float(sp_unit_data.sp_unit_area.max())],
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
    df = EVADataset().load_species_matrix()
    logging.info(f"Loaded {len(df):,} plots")
    
    # Assign spatial folds
    logging.info("Assigning spatial folds...")
    df = assign_checkerboard_folds(
        df, 
        n_splits=CONFIG["n_splits"], 
        block_size=CONFIG["block_size"]
    )
    
    # Load environmental rasters
    logging.info("Loading environmental rasters...")
    env_features = EnvironmentalFeatureDataset()
    chelsa_dem_ds, lc_ds = env_features.load(use_cache=True)
    
    # Loop through folds
    for fold_id in range(CONFIG["n_splits"]):
        logging.info(f"Processing Fold {fold_id + 1}/{CONFIG['n_splits']}...")
        
        # Define fold indices
        test_fold_id = fold_id
        val_fold_id = (fold_id + 1) % CONFIG["n_splits"]
        
        # Split data
        test_df = df[df['spatial_split'] == test_fold_id]
        val_df = df[df['spatial_split'] == val_fold_id]
        train_df = df[(df['spatial_split'] != test_fold_id) & (df['spatial_split'] != val_fold_id)]
        
        logging.info(f"Fold {fold_id}: Train plots: {len(train_df):,}, Val plots: {len(val_df):,}, Test plots: {len(test_df):,}")
        
        # Generate Training Data
        logging.info(f"Generating training samples for Fold {fold_id}...")
        train_data = run_sp_unit_compilation(
            train_df,
            chelsa_dem_ds, lc_ds,
            n_sp_units=int(CONFIG["ratio_samples_plots"] * len(train_df)),
            area_range=CONFIG["spunit_area_range_train"],
            env_var_names=CONFIG["env_vars"],
        )
        save_compiled_data(train_data, output_file_path, f"fold_{fold_id}_train")
        
        # Generate Validation Data
        logging.info(f"Generating validation samples for Fold {fold_id}...")
        val_data = run_sp_unit_compilation(
            val_df,
            chelsa_dem_ds, lc_ds,
            n_sp_units=int(CONFIG["ratio_samples_plots"] * len(val_df)),
            area_range=CONFIG["spunit_area_range_train"],
            env_var_names=CONFIG["env_vars"],
        )
        save_compiled_data(val_data, output_file_path, f"fold_{fold_id}_val")
        
        # Generate Test Data
        logging.info(f"Generating test samples for Fold {fold_id}...")
        test_data = run_sp_unit_compilation(
            test_df,
            chelsa_dem_ds, lc_ds,
            n_sp_units=int(CONFIG["ratio_samples_plots"] * len(test_df)), # Smaller test set
            area_range=CONFIG["spunit_area_range_test"],
            env_var_names=CONFIG["env_vars"],
        )
        save_compiled_data(test_data, output_file_path, f"fold_{fold_id}_test")

    logging.info("Done!")

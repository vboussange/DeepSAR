"""
Compiles training samples based on EVA and CHELSA data using Simple Cross-Validation (Random Split).

NOTE: this script is currently not in use; we prefer spatial block CV (see compile_sbcv_datasets.py)
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import json
import random

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
from deepsar.data_processing.SR_compilation_ckdtree import run_SR_compilation_ckdtree
from deepsar.data_processing.env_feat_compilation import run_environmental_features_compilation_parallel
from deepsar.utils import get_git_hash

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../data/processed/training_samples/cv",
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
    "area_range": (1e4, 1e8),  # in m2
    "crs": "EPSG:3035",
    "random_state": 2,
    "verbose": True,
    "num_workers": 100,  # number of parallel workers for climate compilation
    "n_splits": 5,
    "ratio_samples_plots": 0.01, # ratio of genrated train/val/test samples to raw plots
}

def assign_random_folds(gdf, n_splits=5, random_state=42):
    """
    Assigns random folds to a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame with plot data
        n_splits: Number of folds
        random_state: Random seed
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(gdf)
    
    # Generate random fold assignments
    folds = rng.integers(0, n_splits, size=n_samples)
    
    gdf['spatial_split'] = folds # Keeping column name consistent for downstream logic
    
    return gdf

def run_sp_unit_compilation(
    df: gpd.GeoDataFrame,
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
    """
    # Step 1: Generate spatial units and compute SR
    sp_unit_data = run_SR_compilation_ckdtree(
        df, n_sp_units, area_range, crs, verbose=verbose
    )
    
    # Step 2: Validate SR
    assert (sp_unit_data.sr > 0).all(), "Found spatial units with zero species richness"
    
    # Step 3: Extract environmental features
    sp_unit_data = run_environmental_features_compilation_parallel(
        sp_unit_data, env_raster, lc_raster, env_var_names, verbose=verbose
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
    sha = get_git_hash()
    output_file_path = CONFIG["output_file_path"] / sha
    output_file_path.mkdir(parents=True, exist_ok=True)
    
    # Load EVA data (species matrix format)
    logging.info("Loading EVA data...")
    df = EVADataset().load_species_matrix()
    logging.info(f"Loaded {len(df):,} plots")
    
    # Assign random folds
    logging.info("Assigning random folds...")
    df = assign_random_folds(
        df, 
        n_splits=CONFIG["n_splits"], 
        random_state=CONFIG["random_state"]
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
        
        # Split data
        test_df = df[df['spatial_split'] == test_fold_id]
        train_df = df[df['spatial_split'] != test_fold_id]
        
        logging.info(f"Fold {fold_id}: Train plots: {len(train_df):,}, Test plots: {len(test_df):,}")
        
        # Generate Training Data
        logging.info(f"Generating training samples for Fold {fold_id}...")
        train_data = run_sp_unit_compilation(
            train_df,
            chelsa_dem_ds, lc_ds,
            n_sp_units=int(CONFIG["ratio_samples_plots"] * len(train_df)),
            area_range=CONFIG["area_range"],
            env_var_names=CONFIG["env_vars"],
        )
        save_compiled_data(train_data, output_file_path, f"fold_{fold_id}_train")
        
        # Generate Test Data
        logging.info(f"Generating test samples for Fold {fold_id}...")
        test_data = run_sp_unit_compilation(
            test_df,
            chelsa_dem_ds, lc_ds,
            n_sp_units=int(CONFIG["ratio_samples_plots"] * len(test_df)),
            area_range=CONFIG["area_range"],
            env_var_names=CONFIG["env_vars"],
        )
        save_compiled_data(test_data, output_file_path, f"fold_{fold_id}_test")

    logging.info("Done!")

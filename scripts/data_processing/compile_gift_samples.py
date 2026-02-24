"""
Compiles GIFT dataset with environmental features.

This script performs the following operations:
1. Loads processed GIFT species matrix data (plots with species presence-absence).
2. Calculates species richness from the species matrix.
3. Extracts environmental features (mean and std) for each plot from rasters.
4. Saves the compiled dataset to a parquet file.
"""

import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import json
import random

from muscari.data_processing.utils_gift import GIFTDataset
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari.data_processing.env_feat_compilation import run_environmental_features_compilation_parallel
from muscari.utils import get_git_hash

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

CONFIG = {
    "output_base_dir": Path(__file__).parent / "../../data/processed/test_samples_GIFT/",
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
    "random_state": 2,
    "verbose": True,
    "num_workers": 100,  # number of parallel workers for env feature compilation
}

def calculate_species_richness(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate species richness from species matrix.
    
    Args:
        df: GeoDataFrame with species columns (boolean presence-absence)
        
    Returns:
        GeoDataFrame with added 'sr' column
    """
    logging.info("Calculating species richness from species matrix...")
    species_list = df.attrs.get('species_list', [])
    
    if not species_list:
        raise ValueError("No species_list found in DataFrame attributes")
    
    # Calculate SR by summing species columns
    species_matrix = df[species_list].values
    df['sr'] = species_matrix.sum(axis=1)
    
    logging.info(f"Species richness range: {df['sr'].min()} - {df['sr'].max()}")
    
    return df

def save_compiled_data(df: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Save compiled data to GeoParquet format with summary statistics.
    """
    file_path = output_path / "compiled_data.parquet"
    logging.info(f"Saving compiled data to {file_path}")
    
    df.to_parquet(
        file_path,
        compression='zstd',
        index=False,
    )
    
    # Save summary statistics
    summary_path = output_path / "dataset_summary.json"
    summary = {
        "environmental_variables": CONFIG["env_vars"],
        "n_plots": len(df),
        "columns": list(df.columns),
        "sr_range": [int(df.sr.min()), int(df.sr.max())],
        "sp_unit_area_range": [float(df.sp_unit_area.min()), float(df.sp_unit_area.max())],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved dataset summary to {summary_path}")

if __name__ == "__main__":
    # Setup output directory with git hash
    sha = get_git_hash(fallback="unknown_commit")
    
    output_dir = CONFIG["output_base_dir"] / sha
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GIFT data (species matrix format)
    logging.info("Loading GIFT data...")
    df = GIFTDataset().load(use_cache=False)
    # Filter valid plots
    # Keep plots larger than 2000m x 2000m
    df = df[df.area_m2 > 2e3**2]  
    logging.info(f"Loaded {len(df):,} plots")
    
    # Calculate species richness
    df = calculate_species_richness(df)
    
    # Validate SR
    assert (df.sr > 0).all(), "Found plots with zero species richness"
    
    # Load environmental rasters
    logging.info("Loading environmental rasters...")
    env_features = EnvironmentalFeatureDataset()
    chelsa_dem_ds, lc_ds = env_features.load()

    # Extract environmental features
    logging.info("Extracting environmental features...")
    df = run_environmental_features_compilation_parallel(
        df, chelsa_dem_ds, lc_ds, 
        CONFIG["env_vars"], 
        num_workers=CONFIG["num_workers"], 
        verbose=CONFIG["verbose"]
    )
    
    # Save compiled data
    save_compiled_data(df, output_dir)
    
    logging.info(f"Compilation complete! Data saved at {output_dir}")

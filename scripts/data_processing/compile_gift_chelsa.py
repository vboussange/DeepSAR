"""
Compiles GIFT checklists with CHELSA climate predictors.

This script performs the following operations:
1. Loads processed GIFT plot data and CHELSA climate raster.
2. Extracts climate variables (mean and standard deviation) for each plot location.
3. Saves the augmented dataset with climate features to a parquet file.
4. Exports dataset statistics.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
import warnings
import git
import random

from deepsar.data_processing.utils_gift import GIFTDataset
from deepsar.data_processing.utils_features import CHELSADataset

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Suppress Numba logs
logging.getLogger("numba").setLevel(logging.WARNING)

# Configuration
CONFIG = {
    "gift_data_dir": Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered",
    "output_base_dir": Path(__file__).parent / "../../data/processed/GIFT_CHELSA_compilation/",
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
    ],
    "crs": "EPSG:3035",
    "random_state": 2,
}

# Generate column names for climate features
MEAN_LABELS = CONFIG["env_vars"]
STD_LABELS = [f"std_{var}" for var in CONFIG["env_vars"]]
CLIMATE_COL_NAMES = np.hstack((MEAN_LABELS, STD_LABELS)).tolist()

def load_and_preprocess_data():
    """
    Load and preprocess GIFT and climate data.
    
    Returns:
        tuple: (plot_gdf, species_df, climate_raster)
    """
    logging.info("Loading GIFT data...")
    plot_gdf = gpd.read_file(CONFIG["gift_data_dir"] / "plot_data.gpkg")
    species_df = pd.read_parquet(CONFIG["gift_data_dir"] / "species_data.parquet")
    
    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, species_df, climate_raster

def extract_climate_features(plot_gdf, climate_raster, verbose=False):
    """
    Extracts climate features (mean and std) for each plot from the climate raster.
    
    Args:
        plot_gdf (GeoDataFrame): Plots to process.
        climate_raster (DataArray): Climate data.
        verbose (bool): Whether to show progress bar.
        
    Returns:
        GeoDataFrame: Updated plot_gdf with climate columns.
    """
    # Initialize columns
    for col in CLIMATE_COL_NAMES:
        plot_gdf[col] = np.nan

    # Iterate over plots
    # Note: rio.clip per geometry can be slow for many geometries.
    # For point data, sampling is faster. For polygons, clipping is needed.
    # Assuming polygons here as per previous scripts.
    
    for i, row in tqdm(plot_gdf.iterrows(), total=len(plot_gdf), desc="Extracting climate features", disable=not verbose):
        try:
            # Clip raster to plot geometry
            env_vars = climate_raster.rio.clip([row.geometry], drop=True, all_touched=True)
            env_vars_np = env_vars.to_numpy()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Calculate mean and std across spatial dimensions (axis 1 and 2)
                # axis 0 is variables
                _m = np.nanmean(env_vars_np, axis=(1, 2))
                _std = np.nanstd(env_vars_np, axis=(1, 2))
            
            env_pred_stats = np.concatenate([_m, _std])
            plot_gdf.loc[i, CLIMATE_COL_NAMES] = env_pred_stats
            
        except Exception as e:
            if verbose:
                logging.warning(f"Failed to extract climate for plot {i}: {e}")

    return plot_gdf

def export_dataset_statistics(plot_gdf, species_df, output_dir):
    """
    Calculate and export dataset statistics to a text file.
    """
    logging.info("Calculating dataset statistics...")
    num_entries = len(plot_gdf)
    num_distinct_species = species_df['species_name'].nunique()

    stats_file_path = output_dir / "dataset_statistics.txt"
    logging.info(f"Exporting dataset statistics to {stats_file_path}")
    
    with open(stats_file_path, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("==================\n")
        f.write(f"Number of entries: {num_entries}\n")
        f.write(f"Number of distinct species: {num_distinct_species}\n")

def main():
    # Set seeds for reproducibility
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    
    # Setup output directory based on git commit
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.git.rev_parse(repo.head, short=True)
    except git.InvalidGitRepositoryError:
        sha = "unknown_commit"
        logging.warning("Not a git repository. Using 'unknown_commit' for output directory.")
        
    output_dir = CONFIG["output_base_dir"] / sha
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    plot_gdf, species_df, climate_raster = load_and_preprocess_data()
    
    # Export stats
    export_dataset_statistics(plot_gdf, species_df, output_dir)
    
    # Extract features
    plot_gdf = extract_climate_features(plot_gdf, climate_raster, verbose=True)

    # Save results
    output_path = output_dir / "compiled_data.parquet"
    logging.info(f"Exporting compiled data to {output_path}")
    plot_gdf.to_parquet(output_path)
    
    logging.info(f'Full compilation saved at {output_dir}.')

if __name__ == "__main__":
    main()

"""
Compiles GIFT checklists for different habitat types with climate predictors.
"""

# 1. Assign each EVA species a habitat
# 2. Retrieve GIFT data
# 3. For each habitat
#       - filter out species based on EVA species, 
#       - Calculate SR
#       - calculate associated climate vars

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
import warnings

from src.data_processing.utils_gift import GIFTDataset
from src.data_processing.utils_env_pred import CHELSADataset
from src.data_processing.utils_eunis import EUNISDataset, get_fraction_habitat_landcover
from src.utils import save_to_pickle
from src.data_processing.utils_polygons import (
    partition_polygon_gdf,
)
import git
import random

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(
    logging.WARNING
)  # see https://stackoverflow.com/questions/65398774/numba-printing-information-regarding-nvidia-driver-to-python-console-when-using

CONFIG = {
    "gift_data_dir": Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered",
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../data/processed/GIFT_CHELSA_compilation/",
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
    "block_length": 1e5, # in meters
    "crs": "EPSG:3035",
    "habitats" : ["all", "T", "Q", "S", "R"],
    # "habitats": ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"],
    "random_state": 2,
    
}
mean_labels = CONFIG["env_vars"]
std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()

# range to be investigated
# poly_range = (100, 100, 200e3, 200e3) # in meters

def load_and_preprocess_data():
    """
    Load and preprocess GIFT and climate data.
    Returns processed GIFT and climate datasets.
    """
    logging.info("Loading EVA data...")
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

# def clip_GIFT_SR(plot_gdf, species_dict, habitat_map):
#     for i, row in plot_gdf.iterrows():
#         plot_id = row["entity_ID"]
#         clipped_habitat_map = habitat_map.rio.clip([row.geometry], drop=True, all_touched=True)
#         proportion_area = get_fraction_habitat_landcover(clipped_habitat_map)
#         species = species_dict[plot_id]
#         sr = len(np.unique(species))
#         plot_gdf.loc[i, "sr"] = sr
#         plot_gdf.loc[i, "observed_area"] = row.geometry.area * proportion_area

#     return plot_gdf

def compile_climate_data_megaplot(megaplot_data, climate_raster, verbose=False):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    # only retain pixels which correspond to habitat map
    for i, row in tqdm(megaplot_data.iterrows(), total=megaplot_data.shape[0], desc="Compiling climate", disable=not verbose):
        # climate
        # Use the geometry directly to clip the climate raster
        env_vars = climate_raster.rio.clip([row.geometry], drop=True, all_touched=True)
        env_vars = env_vars.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _m = np.nanmean(env_vars, axis=(1, 2))
            _std = np.nanstd(env_vars, axis=(1, 2))
        env_pred_stats = np.concatenate([_m, _std])
        megaplot_data.loc[i, CLIMATE_COL_NAMES] = env_pred_stats
    return megaplot_data


def export_dataset_statistics(plot_gdf, species_df, output_file_path):
    """
    Calculate and export dataset statistics to a text file.
    
    Args:
        plot_gdf: GeoDataFrame containing plot data
        species_dict: Dictionary mapping plot IDs to species lists
        output_file_path: Path where statistics file should be saved
    """
    logging.info("Calculating dataset statistics...")
    num_entries = len(plot_gdf)
    all_species = species_df['work_species_cleaned'].unique()
    num_distinct_species = len(all_species)

    stats_file_path = output_file_path / "dataset_statistics.txt"
    logging.info(f"Exporting dataset statistics to {stats_file_path}")
    with open(stats_file_path, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n")
        f.write(f"Number of entries: {num_entries}\n")
        f.write(f"Number of distinct species: {num_distinct_species}\n")
        
if __name__ == "__main__":    
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    CONFIG["output_file_path"]  = CONFIG["output_file_path"] / sha
    CONFIG["output_file_path"].mkdir(parents=True, exist_ok=True)
    CONFIG["output_file_name"] = Path(f"augmented_data.pkl")
    
    plot_gdf, species_df, climate_raster = load_and_preprocess_data()
    export_dataset_statistics(plot_gdf, species_df, CONFIG["output_file_path"])
    plot_gdf = compile_climate_data_megaplot(plot_gdf, climate_raster, verbose=True)

    # exporting megaplot_data to gpkg
    output_path = CONFIG["output_file_path"] / "megaplot_data.parquet"
    print(f"Exporting {output_path}")
    plot_gdf.to_parquet(output_path)
    
    
    logging.info(f'Full compilation saved at {CONFIG["output_file_path"]}.')

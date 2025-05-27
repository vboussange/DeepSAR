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
    plot_gdf, species_dict = GIFTDataset().load()
    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    return plot_gdf, species_dict, climate_raster

def clip_GIFT_SR(plot_gdf, species_dict, habitat_map):
    for i, row in plot_gdf.iterrows():
        plot_id = row["entity_ID"]
        clipped_habitat_map = habitat_map.rio.clip([row.geometry], drop=True, all_touched=True)
        proportion_area = get_fraction_habitat_landcover(clipped_habitat_map)
        species = species_dict[plot_id]
        sr = len(np.unique(species))
        plot_gdf.loc[i, "sr"] = sr
        plot_gdf.loc[i, "observed_area"] = row.geometry.area * proportion_area

    return plot_gdf

def process_partition(partition, block_plot_gdf, species_dict, climate_raster, habitat_map):
    megaplot_data_partition = clip_GIFT_SR(block_plot_gdf, species_dict, habitat_map)
    # megaplot_data_partition["num_plots"] = megaplot_data_partition['geometry'].apply(lambda geom: len(geom.geoms) if geom.geom_type == 'MultiPoint' else 1)
    megaplot_data_partition["megaplot_area"] = block_plot_gdf.geometry.area
    megaplot_data_partition["geometry"] = block_plot_gdf.geometry
    megaplot_data_partition["partition"] = partition
    megaplot_data_partition = gpd.GeoDataFrame(megaplot_data_partition, crs = block_plot_gdf.crs, geometry="geometry")
    # print(f"Partition {partition}: Processing climate variables...")
    megaplot_data_partition = compile_climate_data_megaplot(megaplot_data_partition, climate_raster, habitat_map)
    return megaplot_data_partition

def generate_megaplots(plot_gdf, species_dict, climate_raster, habitat_map):
    """
    Process EVA data and generate synthetic megaplots data based on landcover.
    Returns GeoDataFrame of SAR data.
    """
    total = len(plot_gdf["partition"].unique())
    miniters = max(total // 100, 1)  # Refresh every 1%
    megaplot_data_hab_ar = []
    for partition, block_plot_gdf in tqdm(plot_gdf.groupby("partition"), desc="Processing partitions", total=total, miniters=miniters):
        megaplot_data_hab_ar.append(process_partition(partition, block_plot_gdf, species_dict, climate_raster, habitat_map))
                
    megaplot_data_hab = pd.concat(megaplot_data_hab_ar, ignore_index=True)

    logging.info(f"Nb. megaplots: {len(megaplot_data_hab)}")

    return megaplot_data_hab[["sr", "area", "megaplot_area", "geometry", "partition"] + CLIMATE_COL_NAMES]


def compile_climate_data_megaplot(megaplot_data, climate_raster, habitat_map, verbose=False):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    # only retain pixels which correspond to habitat map
    habitat_map = habitat_map.where(habitat_map > 0, np.nan)
    climate_raster = climate_raster * habitat_map
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

if __name__ == "__main__":    
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    CONFIG["output_file_path"]  = CONFIG["output_file_path"] / sha
    CONFIG["output_file_path"].mkdir(parents=True, exist_ok=True)
    CONFIG["output_file_name"] = Path(f"augmented_data.pkl")
    
    plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
    eunis = EUNISDataset()
    assert set(plot_gdf.entity_ID).issubset(set(species_dict.keys())), " plot_gdf.entity_ID is not a subset of species_dict.entity_ID"
    
    logging.info("Partitioning...")
    plot_gdf["polygon"] = plot_gdf.geometry
    plot_gdf.set_geometry(plot_gdf.geometry.centroid, inplace=True)
    plot_gdf = partition_polygon_gdf(plot_gdf, CONFIG["block_length"])
    plot_gdf.set_geometry(plot_gdf["polygon"], inplace=True, drop=True)
    logging.info(f"Nb. partitions: {len(plot_gdf['partition'].unique())}")
    # save raw plot SR and climate data
    
    megaplot_ar = []
    
    for hab in CONFIG["habitats"]:
        logging.info(f"Generating megaplot dataset for habitat: {hab}")
        # TODO: need to filter out species not in habitat, by creating specific plot ID
        gdf_hab = plot_gdf[plot_gdf["level_1"] == hab]
        habitat_map = eunis.get_habitat_map(hab).where(eunis.raster > -1, np.nan).rio.reproject_match(climate_raster)
        megaplot_data_hab = generate_megaplots(gdf_hab, species_dict, climate_raster, habitat_map)
        megaplot_data_hab["habitat_id"] = hab
        
        assert (megaplot_data_hab.sr > 0).all()

        megaplot_ar.append(megaplot_data_hab)
        
        # Save checkpoint
        checkpoint_path = CONFIG["output_file_path"] / (CONFIG["output_file_name"].stem + f"_checkpoint_{hab}.pkl")
        save_to_pickle(checkpoint_path, megaplot_data=megaplot_data_hab)
        logging.info(f"Checkpoint saved for habitat `{hab}` at {checkpoint_path}")

    # aggregating results and final save
    megaplot_data = pd.concat(megaplot_ar, ignore_index=True)
       
    # exporting megaplot_data to gpkg
    output_path = CONFIG["output_file_path"] / "megaplot_data.gpkg"
    print(f"Exporting {output_path}")
    megaplot_data.to_file(output_path, driver="GPKG")
    
    
    logging.info(f'Full compilation saved at {CONFIG["output_file_path"]}.')

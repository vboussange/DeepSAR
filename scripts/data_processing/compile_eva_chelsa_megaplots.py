"""
Compiles megaplots based on EVA and CHELSA data, for different habitat types.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import math
from tqdm import tqdm
import warnings

from src.generate_sar_data_eva import clip_EVA_SR, generate_random_boxes_from_candidate_pairs, generate_random_boxes, generate_random_squares
from src.data_processing.utils_eva import EVADataset
from src.data_processing.utils_env_pred import CHELSADataset
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
    "block_length": 1e5, # in meters
    "area_range": (1e4, 1e10),  # in m2
    # "side_range": (1e2, 1e5), # in m
    "num_polygon_max": np.inf,
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

def load_and_preprocess_data(check_consistency=False):
    """
    Load and preprocess EVA and landcover data.
    Returns processed EVA and landcover datasets.
    """
    logging.info("Loading EVA data...")
    plot_gdf, species_dict = EVADataset().load()
    if check_consistency:
        print("Checking data consistency...")
        assert all([len(np.unique(species_dict[k])) == r.SR for k, r in plot_gdf.iterrows()])

    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, species_dict, climate_raster


def process_partition(partition, block_plot_gdf, species_dict):
    # print(f"Partition {partition}: Processing EVA data...")
    # boxes_gdf = generate_random_boxes_from_candidate_pairs(block_plot_gdf, min(len(block_plot_gdf), CONFIG["num_polygon_max"]))
    # boxes_gdf = generate_random_boxes(block_plot_gdf, 
    #                                   min(len(block_plot_gdf), CONFIG["num_polygon_max"]), 
    #                                   CONFIG["area_range"],
    #                                   CONFIG["side_range"])
    boxes_gdf = generate_random_squares(block_plot_gdf, 
                                      min(len(block_plot_gdf), CONFIG["num_polygon_max"]), 
                                      CONFIG["area_range"])
    megaplot_data_partition = clip_EVA_SR(block_plot_gdf, species_dict, boxes_gdf, CONFIG["env_vars"])
    # megaplot_data_partition["num_plots"] = megaplot_data_partition['geometry'].apply(lambda geom: len(geom.geoms) if geom.geom_type == 'MultiPoint' else 1)
    megaplot_data_partition["megaplot_area"] = boxes_gdf.area
    megaplot_data_partition["geometry"] = boxes_gdf.geometry
    megaplot_data_partition["partition"] = partition
    megaplot_data_partition = gpd.GeoDataFrame(megaplot_data_partition, crs = plot_gdf.crs, geometry="geometry")
    # print(f"Partition {partition}: Processing climate variables...")
    # megaplot_data_partition = compile_climate_data_megaplot(megaplot_data_partition, climate_raster)
    return megaplot_data_partition

def generate_megaplots(plot_gdf, species_dict):
    """
    Process EVA data and generate synthetic megaplots data based on landcover.
    Returns GeoDataFrame of SAR data.
    """
    total = len(plot_gdf["partition"].unique())
    miniters = max(total // 100, 1)  # Refresh every 1%
    megaplot_data_hab_ar = []
    for partition, block_plot_gdf in tqdm(plot_gdf.groupby("partition"), desc="Processing partitions", total=total, miniters=miniters):
        if len(block_plot_gdf) > 1:
            megaplot_data_hab_ar.append(process_partition(partition, block_plot_gdf, species_dict))
                
    megaplot_data_hab = pd.concat(megaplot_data_hab_ar, ignore_index=True)

    logging.info(f"Nb. megaplots: {len(megaplot_data_hab)} || Nb. plots: {len(plot_gdf)}")

    return megaplot_data_hab[["sr", "area", "megaplot_area", "geometry", "partition"] + CLIMATE_COL_NAMES]

def compile_climate_data_plot(plot_data, climate_raster):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    # climate
    y = plot_data.geometry.y
    x = plot_data.geometry.x
    env_vars = climate_raster.sel(
        x=xr.DataArray(x, dims="z"),
        y=xr.DataArray(y, dims="z"),
        method="nearest",
    )
    env_vars = env_vars.to_numpy().transpose()
    plot_data[CONFIG["env_vars"]] = env_vars
    return plot_data


def format_plot_data(plot_data, species_data):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    for i, row in tqdm(plot_data.iterrows(), desc="Compiling species richness", total=plot_data.shape[0]):
        plot_id = row["plot_id"]
        species = species_data[plot_id]
        sr = len(np.unique(species))
        plot_data.loc[i, "sr"] = sr

    plot_data = plot_data.rename({"SR":"sr", "plot_size": "area", "Level_2":"habitat_id"}, axis=1)
    plot_data = plot_data.set_index("plot_id")
    plot_data.loc[:, [f"std_{var}" for var in CONFIG["env_vars"]]] = 0.
    plot_data["megaplot_area"] = plot_data["area_m2"]
    plot_data["area"] = plot_data["area_m2"]
    plot_data["habitat_id"] = plot_data["level_1"]
    
    plot_data = plot_data[["sr", "area", "megaplot_area", "geometry", "habitat_id", "partition"] + CLIMATE_COL_NAMES]

    return plot_data

if __name__ == "__main__":    
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    CONFIG["output_file_path"]  = CONFIG["output_file_path"] / sha
    CONFIG["output_file_path"].mkdir(parents=True, exist_ok=True)
    CONFIG["output_file_name"] = Path(f"eva_chelsa_augmented_data.pkl")
    
    plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
    # Sample 1000 rows for debugging purposes
    # plot_gdf = plot_gdf.sample(n=1000, random_state=CONFIG["random_state"])
    plot_gdf = compile_climate_data_plot(plot_gdf, climate_raster)
    
    logging.info("Partitioning...")
    plot_gdf = partition_polygon_gdf(plot_gdf, CONFIG["block_length"])
    logging.info(f"Nb. partitions: {len(plot_gdf['partition'].unique())}")
    
    # save raw plot SR and climate data
    logging.info("Compiling plot dataset.")
    plot_data_all = format_plot_data(plot_gdf, species_dict)
    
    megaplot_ar = []
    plot_gdf_by_hab = plot_data_all.groupby("level_1")

    # compiling data for each separate habitat
    for hab in CONFIG["habitats"]:
        logging.info(f"Generating megaplot dataset for habitat: {hab}")
        if hab == "all":
            gdf_hab = plot_data_all
        else:
            gdf_hab = plot_gdf_by_hab.get_group(hab)
        megaplot_data_hab = generate_megaplots(gdf_hab, species_dict, climate_raster)
        megaplot_data_hab["habitat_id"] = hab
        
        assert (megaplot_data_hab.sr > 0).all()

        megaplot_ar.append(megaplot_data_hab)
        
        # Save checkpoint
        checkpoint_path = CONFIG["output_file_path"] / (CONFIG["output_file_name"].stem + f"_checkpoint_{hab}.pkl")
        save_to_pickle(checkpoint_path, megaplot_data=megaplot_data_hab)
        logging.info(f"Checkpoint saved for habitat `{hab}` at {checkpoint_path}")

    # aggregating results and final save
    megaplot_data = pd.concat(megaplot_ar, ignore_index=True)
       
    # export the full compilation to pickle
    output_path = CONFIG["output_file_path"] / CONFIG["output_file_name"]
    print(f"Exporting {output_path}")
    save_to_pickle(output_path, 
                   megaplot_data=megaplot_data, 
                   plot_data_all=plot_data_all,
                   config=CONFIG)
    
    # exporting megaplot_data to gpkg
    output_path = CONFIG["output_file_path"] / "eva_chelsa_megaplot_data.gpkg"
    print(f"Exporting {output_path}")
    megaplot_data.to_file(output_path, driver="GPKG")
    
    # exporting raw plot data tp gpkg
    output_path = CONFIG["output_file_path"] / "eva_chelsa_plot_data.gpkg"
    print(f"Exporting {output_path}")
    plot_data_all.to_file(output_path, driver="GPKG")
    
    logging.info(f'Full compilation saved at {CONFIG["output_file_path"]}.')

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
        f"../../data/processed/EVA_vs_GIFT_compilation/",
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
    "block_length": 1e6, # in meters
    "area_range": (1e4, 1e12),  # in m2
    # "side_range": (1e2, 1e5), # in m
    "num_polygon_max": np.inf,
    "crs": "EPSG:3035",
    # "habitats" : ["all", "T", "Q", "S", "R"],
    "habitats" : ["all"], # TODO: to change for full habitats
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
        logging.info("Checking data consistency...")
        assert all([len(np.unique(species_dict[k])) == r.SR for k, r in plot_gdf.iterrows()])

    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, species_dict, climate_raster


def run_compilation(boxes_gdf, block_plot_gdf, species_dict):
    # print(f"Partition {partition}: Processing EVA data...")
    # boxes_gdf = generate_random_boxes_from_candidate_pairs(block_plot_gdf, min(len(block_plot_gdf), CONFIG["num_polygon_max"]))
    # boxes_gdf = generate_random_boxes(block_plot_gdf, 
    #                                   min(len(block_plot_gdf), CONFIG["num_polygon_max"]), 
    #                                   CONFIG["area_range"],
    #                                   CONFIG["side_range"])

    megaplot_data = clip_EVA_SR(block_plot_gdf, species_dict, boxes_gdf)
    # megaplot_data["num_plots"] = megaplot_data['geometry'].apply(lambda geom: len(geom.geoms) if geom.geom_type == 'MultiPoint' else 1)
    megaplot_data["megaplot_area"] = boxes_gdf.area
    megaplot_data["geometry"] = boxes_gdf.geometry
    megaplot_data = gpd.GeoDataFrame(megaplot_data, crs = plot_gdf.crs, geometry="geometry")
    logging.info(f"Nb. megaplots: {len(megaplot_data_hab)} || Nb. plots: {len(plot_gdf)}")
    return megaplot_data_hab[["sr", "area", "megaplot_area", "geometry"]]

if __name__ == "__main__":    
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    
    path_gift_data = Path(__file__).parent / f"../../data/processed/GIFT_CHELSA_compilation/fb8bc71/megaplot_data.gpkg"
    
    CONFIG["output_file_path"]  = CONFIG["output_file_path"] / sha
    CONFIG["output_file_path"].mkdir(parents=True, exist_ok=True)
    CONFIG["output_file_name"] = Path(f"eva_chelsa_augmented_data.gpkg")
    
    plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
    gift_df = gpd.read_file(path_gift_data)
    
    # selecting all habitat
    gift_df = gift_df[gift_df["habitat_id"] == "all"]

    megaplot_data_all = run_compilation(gift_df, species_dict, climate_raster)
        
    megaplot_ar = []
    plot_gdf_by_hab = plot_data_all.groupby("habitat_id")
    # compiling data for each separate habitat
    for hab in CONFIG["habitats"]:
        logging.info(f"Generating megaplot dataset for habitat: {hab}")
        if hab == "all":
            gdf_hab = plot_data_all
        else:
            gdf_hab = plot_gdf_by_hab.get_group(hab)
        megaplot_data_hab = run_compilation(gdf_hab, species_dict, climate_raster)
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
    logging.info(f"Exporting {output_path}")
    save_to_pickle(output_path, 
                   megaplot_data=megaplot_data, 
                   plot_data_all=plot_data_all,
                   config=CONFIG)
    
    # exporting megaplot_data to gpkg
    output_path = CONFIG["output_file_path"] / "eva_chelsa_megaplot_data.gpkg"
    logging.info(f"Exporting {output_path}")
    megaplot_data.to_file(output_path, driver="GPKG")
    
    # exporting raw plot data tp gpkg
    output_path = CONFIG["output_file_path"] / "eva_chelsa_plot_data.gpkg"
    logging.info(f"Exporting {output_path}")
    plot_data_all.to_file(output_path, driver="GPKG")
    
    logging.info(f'Full compilation saved at {CONFIG["output_file_path"]}.')

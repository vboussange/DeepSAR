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

# working with dictionnary of species
def clip_EVA_SR(plot_gdf, species_data, polygons_gdf, verbose=True):
    data = pd.DataFrame({
        "area": pd.Series(int),
        "sr": pd.Series(int),
        "num_plots": pd.Series(int),
    })
    for i, poly in tqdm(enumerate(polygons_gdf.geometry), desc="Clipping SR", total=len(polygons_gdf), disable=not verbose):
        df_samp = plot_gdf[plot_gdf.within(poly)]
        if len(df_samp) == 0:
            sr = 0
            a = 0
            num_plots = 0
        else:
            species = np.concatenate([species_data[idx] for idx in df_samp.index])
            sr = len(np.unique(species))
            a = np.sum(df_samp['area'])
            # geom = MultiPoint(df_samp.geometry.to_list())
            num_plots = len(df_samp)
        
        data.loc[i, ["area", "sr", "num_plots"]] = [a, sr, num_plots]
    return data

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
    return megaplot_data

if __name__ == "__main__":    
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    
    path_gift_data = Path(__file__).parent / f"../../../../data/processed/GIFT_CHELSA_compilation/fb8bc71/megaplot_data.gpkg"
    
    CONFIG["output_file_path"]  = CONFIG["output_file_path"]
    CONFIG["output_file_path"].mkdir(parents=True, exist_ok=True)

    
    plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
    plot_gdf.set_index("plot_id", inplace=True)
    plot_gdf.rename({"area_m2": "area", "level_1":"habitat_id"}, axis=1, inplace=True)

    gift_df = gpd.read_file(path_gift_data)
    
    # selecting all habitat
    gift_df = gift_df[gift_df["habitat_id"] == "all"]
    gift_df = gift_df[gift_df.is_valid]

    megaplot_data_all = run_compilation(gift_df, plot_gdf, species_dict)
    megaplot_data_all["habitat_id"] = "all"
    
    megaplot_data_all.to_file(CONFIG["output_file_path"] / "EVA_augmented_data.gpkg", driver="GPKG")
    gift_df.to_file(CONFIG["output_file_path"] / "GIFT_data.gpkg", driver="GPKG")

    logging.info(f'Full compilation saved at {CONFIG["output_file_path"]}.')
        

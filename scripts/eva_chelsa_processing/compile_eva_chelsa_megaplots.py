"""
Processing features based on polygons and CHELSA rasters, and EUNIS Esy for
landcover. In plot_megaplot version, we include all single plots to be
predicted, and we generate at least as much megaplots.

The augmented dataset is then processed so that the ratio plot:megaplot is 1:1.
This is to have a more fair evaluation of model, as variation in SR is higher at fine scales.
The corresponding augmented dataset is sometimes called "v2".
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
import math
from tqdm import tqdm

from src.generate_sar_data_eva import clip_EVA_SR_gpu, generate_random_boxes
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

block_relative_extent = 0.2
CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        f"../../data/processed/EVA_CHELSA_raw/",
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
    "block_relative_extent": block_relative_extent,
    "batch_size": 20,
    "area_range": (1e4, 1e11),  # in m2
    "side_range": (1e2, 1e6), # in m
    "num_polygon_max": int(1e6),
    "crs": "EPSG:3035",
    # "habitats" : ["T1"]
    "habitats": ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3"],
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
    plot_gdf, dict_sp = EVADataset().load()
    if check_consistency:
        print("Checking data consistency...")
        assert all([len(np.unique(dict_sp[k])) == r.SR for k, r in plot_gdf.iterrows()])

    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, dict_sp, climate_raster


def generate_megaplots(plot_gdf, dict_sp, climate_raster):
    """
    Process EVA data and generate synthetic megaplots data based on landcover.
    Returns GeoDataFrame of SAR data.
    """
    
    # TOFIX: quick and dirty trick to generate at least `len(plot_gdf)` megaplots with more than one plot
    num_polygons = np.minimum(CONFIG["num_polygon_max"], 2 * len(plot_gdf))

    polygons_gdf = generate_random_boxes(
        plot_gdf,
        num_polygons,
        CONFIG["area_range"],
        CONFIG["side_range"],
    )
    
    megaplot_data_hab_ar = []
    for partition, pol_gdf in polygons_gdf.groupby("partition"):
        logging.info(
            f"Partition {partition}: Processing EVA data..."
        )
        if not pol_gdf.empty:
            eva_data_part = plot_gdf[plot_gdf.partition == partition].copy()
            # if gpu available
            megaplot_data_partition = clip_EVA_SR_gpu(eva_data_part, dict_sp, pol_gdf, CONFIG["batch_size"])
            # if no gpu available
            # megaplot_data = clip_EVA_SR(
            #     eva_data_part, pol_gdf
            # )  # to be modified for gpu
            megaplot_data_partition["num_plots"] = megaplot_data_partition['geometry'].apply(lambda geom: len(geom.geoms) if geom.geom_type == 'MultiPoint' else 1)
            
            # thinning
            megaplot_data_partition = megaplot_data_partition[megaplot_data_partition["num_plots"] > 1]
            megaplot_data_partition["partition"] = partition
            megaplot_data_partition = compile_climate_data_megaplot(megaplot_data_partition, climate_raster)
            megaplot_data_hab_ar.append(megaplot_data_partition)
            
    megaplot_data_hab = gpd.GeoDataFrame(
        pd.concat(megaplot_data_hab_ar, ignore_index=True),
        geometry="geometry",
        crs=CONFIG["crs"],
    )
    assert (megaplot_data_hab["num_plots"] > 1).all()
    logging.info(f"Nb. megaplots: {len(megaplot_data_hab)}, \nNb. plots: {len(plot_gdf)}")

    return megaplot_data_hab[["sr", "area"] + CLIMATE_COL_NAMES]


def compile_climate_data_megaplot(megaplot_data, climate_raster):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    logging.info("Compiling climate...")
    for i, row in tqdm(megaplot_data.iterrows(), total=megaplot_data.shape[0]):
        # climate
        y = [p.y for p in row.geometry.geoms]
        x = [p.x for p in row.geometry.geoms]
        env_vars = climate_raster.sel(
            x=xr.DataArray(x, dims="z"),
            y=xr.DataArray(y, dims="z"),
            method="nearest",
        )
        env_vars = env_vars.to_numpy().reshape((len(CONFIG["env_vars"]), -1, 1))
        _m = np.nanmean(env_vars, axis=(1, 2))
        _std = np.nanstd(env_vars, axis=(1, 2))
        env_pred_stats = np.concatenate([_m, _std])
        megaplot_data.loc[i, CLIMATE_COL_NAMES] = np.stack(env_pred_stats)
    return megaplot_data

def compile_climate_data_plot(plot_data, climate_raster):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    logging.info("Compiling climate for plots...")
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


def format_plot_data(plot_data):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    
    plot_data = plot_data.rename({"SR":"sr", "plot_size": "area", "Level_2":"habitat_id"}, axis=1)
    plot_data.loc[:, "plot_idxs"] = plot_data.index
    plot_data.loc[:, "num_plots"] = 1

    plot_data.loc[:, [f"std_{var}" for var in CONFIG["env_vars"]]] = 0.
    plot_data = plot_data[["sr", "area"] + CLIMATE_COL_NAMES]

    return plot_data

if __name__ == "__main__":
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    CONFIG["ouput_file_name"] = Path(f"EVA_CHELSA_raw_random_state_{CONFIG['random_state']}_{sha}.pkl")
    
    plot_gdf, dict_sp, climate_raster = load_and_preprocess_data()
    plot_gdf = compile_climate_data_plot(plot_gdf, climate_raster)
    
    logging.info("Partitioning...")
    block_length = (plot_gdf.total_bounds[2] - plot_gdf.total_bounds[0]) * CONFIG["block_relative_extent"]
    logging.info(f"Block length = {block_length/1000}km")
    plot_gdf = partition_polygon_gdf(plot_gdf, block_length)
    # Save the indices of plot_gdf as a CSV
    plot_gdf.index.to_series().to_csv(CONFIG["output_file_path"] / "plot_id.csv", index=False)
    # save raw plot SR and climate data
    plot_data_all = format_plot_data(plot_gdf)
    plot_data_all.to_csv(CONFIG["output_file_path"] / "raw_plot_data.csv", index=False)
        
    plot_megaplot_ar = []
    plot_gdf_by_hab = plot_gdf.groupby("Level_2")
    
    # compiling data for each separate habitat
    for hab in CONFIG["habitats"]:
        logging.info(
                f"Generating megaplots based on {hab} plots"
            )
        gdf_hab = plot_gdf_by_hab.get_group(hab)
        megaplot_data_hab = generate_megaplots(gdf_hab, dict_sp, climate_raster)
        megaplot_data_hab["habitat_id"] = hab
        
        assert (megaplot_data_hab.sr > 0).all()

        # logging.info("Checkpointing compiled dataset")
        # save_to_pickle(
        #     CONFIG["output_file_path"]
        #     / (CONFIG["output_file_name"].stem + f"_{hab}.pkl"),
        #     SAR_data=megaplot_data_hab,
        #     config=CONFIG,
        # )
        plot_megaplot_ar.append(megaplot_data_hab)
        
    # compiling data for all habitats
    logging.info(
                f"Generating megaplots based on all plots"
            )
    megaplot_data_hab = generate_megaplots(plot_gdf, dict_sp, climate_raster)
    megaplot_data_hab["habitat_id"] = "all"
    plot_megaplot_ar.append(megaplot_data_hab)

    # appending simple plot data
    plot_megaplot_ar.append(plot_data_all)

    # aggregating results and final save
    SAR_data = gpd.GeoDataFrame(
        pd.concat(plot_megaplot_ar, ignore_index=True),
        geometry="geometry",
        crs=CONFIG["crs"],
    )
    save_to_pickle(CONFIG["output_file_path"] / CONFIG["output_file_name"], SAR_data=SAR_data, config=CONFIG)
    logging.info("Compilation completed successfully")

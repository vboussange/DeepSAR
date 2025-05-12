"""
Compiles megaplots based on EVA and CHELSA data, for different habitat types.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

from src.generate_sar_data_eva import clip_EVA_SR, generate_random_square
from src.data_processing.utils_eva import EVADataset
from src.data_processing.utils_env_pred import CHELSADataset
from src.utils import save_to_pickle

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
    "block_length": 1e6, # in meters
    "area_range_train": (1e4, 1e12),  # in m2
    "area_range_test": (1e4, 1e10),  # in m2
    "raw_plot_test_size": 0.1, # in fraction of total EVA records
    "megaplot_test_size": 0.005, # in fraction of total EVA train records
    # "side_range": (1e2, 1e5), # in m
    "num_polygon_max": np.inf,
    "crs": "EPSG:3035",
    # "habitats" : ["all", "T", "Q", "S", "R"],
    "habitats" : ["all"], # TODO: to change for full habitats
    "random_state": 2,
    "verbose": True,
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


def run_SR_compilation(plot_gdf, species_dict, nb_megaplots, test=False, verbose=CONFIG["verbose"]):
    data = pd.DataFrame({
                        "observed_area": pd.Series(int),
                        "megaplot_area": pd.Series(int),
                        "sr": pd.Series(int),
                        "num_plots": pd.Series(int),
                        "geometry": pd.Series(dtype="object"),
                        })
    miniters = max(nb_megaplots // 100, 1)  # Refresh every 1%
    if test:
        area_range = CONFIG["area_range_test"]
    else:
        area_range = CONFIG["area_range_train"]

    for i in tqdm(range(nb_megaplots),
                desc="Compiling SR", 
                disable=not verbose,
                total=nb_megaplots, 
                miniters=miniters,
                maxinterval=float("inf")):
        box = generate_random_square(plot_gdf, 
                                    area_range)
        plots_within_box = plot_gdf.within(box)
        df_samp = plot_gdf[plots_within_box]
        species = np.concatenate([species_dict[idx] for idx in df_samp.index])
        sr = len(np.unique(species))
        observed_area = np.sum(df_samp['area'])
        megaplot_area = box.area
        # geom = MultiPoint(df_samp.geometry.to_list())
        num_plots = len(df_samp)
        data.loc[i, ["observed_area", "megaplot_area", "sr", "num_plots", "geometry"]] = [observed_area, megaplot_area, sr, num_plots, box]
        if test:
            plot_gdf = plot_gdf[~plots_within_box]
            assert len(plot_gdf) > 0, "Not enough plots left to sample from."
            
    data = gpd.GeoDataFrame(data, crs = plot_gdf.crs, geometry="geometry")
    data["test"] = test
    return data, plot_gdf

def run_climate_compilation(megaplot_data, climate_raster, verbose=CONFIG["verbose"]):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    nb_megaplots = len(megaplot_data)
    miniters = max(nb_megaplots // 100, 1)  # Refresh every 1%
    
    for i, row in tqdm(megaplot_data.iterrows(), 
                       total=nb_megaplots, 
                       desc="Compiling climate", 
                       disable=not verbose,
                       miniters=miniters,
                       maxinterval=float("inf")):
        # climate
        minx, miny, maxx, maxy = row.geometry.bounds
        env_vars = climate_raster.sel(
            x=slice(minx, maxx),
            y=slice(miny, maxy) 
        )
        env_vars = env_vars.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _m = np.nanmean(env_vars, axis=(1, 2))
            _std = np.nanstd(env_vars, axis=(1, 2))
        env_pred_stats = np.concatenate([_m, _std])
        megaplot_data.loc[i, CLIMATE_COL_NAMES] = env_pred_stats
    return megaplot_data

def run_compilation(plot_gdf, species_dict, climate_raster, num_megaplot, test=False):
    megaplot_data, plot_gdf = run_SR_compilation(plot_gdf, species_dict, num_megaplot, test)
    assert (megaplot_data.sr > 0).all()
    megaplot_data = run_climate_compilation(megaplot_data, climate_raster)
    return megaplot_data, plot_gdf

def generate_plot_data(plot_data, species_data, climate_raster):
    """
    Calculate area and convert landcover binary raster to multipoint for each SAR data row.
    Returns processed SAR data.
    """
    logging.info("Calculating climate vars...")
    y = plot_data.geometry.y
    x = plot_data.geometry.x
    env_vars = climate_raster.sel(
        x=xr.DataArray(x, dims="z"),
        y=xr.DataArray(y, dims="z"),
        method="nearest",
    )
    env_vars = env_vars.to_numpy().transpose()
    plot_data[CONFIG["env_vars"]] = env_vars
    miniters = max(plot_data.shape[0] // 100, 1)  # Refresh every 1%
    for i, row in tqdm(plot_data.iterrows(), 
                       desc="Compiling species richness", 
                       total=plot_data.shape[0],
                       disable=not CONFIG["verbose"],
                       miniters=miniters,
                       maxinterval=float("inf")):
        plot_id = row["plot_id"]
        species = species_data[plot_id]
        sr = len(np.unique(species))
        plot_data.loc[i, "sr"] = sr

    plot_data = plot_data.rename({"area_m2": "area", "level_1":"habitat_id"}, axis=1)
    plot_data = plot_data.set_index("plot_id")
    plot_data.loc[:, [f"std_{var}" for var in CONFIG["env_vars"]]] = 0.
    plot_data["megaplot_area"] = plot_data["area"]
    
    plot_data = plot_data[["sr", "area", "megaplot_area", "geometry", "habitat_id"] + CLIMATE_COL_NAMES]

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
    plot_data_all = generate_plot_data(plot_gdf, species_dict, climate_raster)
        
    megaplot_ar = []
    plot_gdf_by_hab = plot_data_all.groupby("habitat_id")
    # compiling data for each separate habitat
    for hab in CONFIG["habitats"]:
        logging.info(f"Generating megaplot dataset for habitat: {hab}")
        if hab == "all":
            gdf_hab = plot_data_all
        else:
            gdf_hab = plot_gdf_by_hab.get_group(hab)
        nb_raw_plots_test = min(int(CONFIG["raw_plot_test_size"] * len(gdf_hab)), CONFIG["num_polygon_max"])
        EVA_raw_idx, EVA_raw_test_idx = train_test_split(gdf_hab.index, test_size=nb_raw_plots_test, random_state=CONFIG["random_state"])
        EVA_raw_test = gdf_hab.loc[EVA_raw_test_idx]
        EVA_raw_test["type"] = "EVA_raw_test"
        EVA_raw_megaplot_train_test = gdf_hab.loc[EVA_raw_idx]
        nb_megaplot_test = min(int(CONFIG["megaplot_test_size"] * len(EVA_raw_megaplot_train_test)), CONFIG["num_polygon_max"])
        logging.info(f"Compiling megaplot test dataset...")
        EVA_megaplot_test, EVA_raw_megaplot_train  = run_compilation(EVA_raw_megaplot_train_test, 
                                                                     species_dict, 
                                                                     climate_raster, 
                                                                     nb_megaplot_test, 
                                                                     test=True)
        EVA_megaplot_test["type"] = "EVA_megaplot_test"
        logging.info(f"Compiling megaplot train dataset...")
        nb_megaplots_train = min(len(EVA_raw_megaplot_train), CONFIG["num_polygon_max"])
        EVA_megaplot_train, _ = run_compilation(EVA_raw_megaplot_train, 
                                                species_dict, 
                                                climate_raster, 
                                                nb_megaplots_train, 
                                                test=False)
        EVA_megaplot_train["type"] = "EVA_megaplot_train"
        
        compiled_data = pd.concat([EVA_raw_test, EVA_megaplot_test, EVA_megaplot_train], ignore_index=True)
        # Save checkpoint
        checkpoint_path = CONFIG["output_file_path"] / (CONFIG["output_file_name"].stem + f"_checkpoint_{hab}.pkl")
        save_to_pickle(checkpoint_path, megaplot_data=compiled_data)
        logging.info(f"Checkpoint saved for habitat `{hab}` at {checkpoint_path}")
        
        megaplot_ar.append(compiled_data)

    # aggregating results and final save
    megaplot_data = pd.concat(megaplot_ar, ignore_index=True)
       
    # export the full compilation to pickle
    output_path = CONFIG["output_file_path"] / CONFIG["output_file_name"]
    logging.info(f"Exporting {output_path}")
    save_to_pickle(output_path, 
                   megaplot_data=megaplot_data, 
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

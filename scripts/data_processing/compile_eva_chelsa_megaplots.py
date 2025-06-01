"""
Compiles megaplots based on EVA and CHELSA data, for habitat "all".
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
    "megaplot_test_size": 0.005, # in fraction of total EVA records
    "megaplot_train_size": 3, # in fraction of total EVA train records
    "num_polygon_max": np.inf,
    "crs": "EPSG:3035",
    "random_state": 2,
    "verbose": True,
}

# Define covariate feature names based on environmental covariates
mean_labels = CONFIG["env_vars"]
std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()

def load_and_preprocess_data():
    """
    Load and preprocess EVA data and environmental covariate raster. Returns
    `plot_gdf` (gdf of plots), `species_dict` (dictionary where each key
    corresponds to plot_gdf.index and value associated species list), and
    `climate_raster`.
    """
    logging.info("Loading EVA data...")
    plot_gdf, species_dict = EVADataset().load()
    plot_gdf = plot_gdf.set_index("plot_id")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    
    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, species_dict, climate_raster


def run_SR_compilation(plot_gdf, 
                       species_dict, 
                       megaplots, # a list of geometries or an int
                       area_range = None, 
                       verbose=CONFIG["verbose"]):
    
    """
    Calculate species richness, area and observed area for each megaplot,
    defined as bags of EVA raw plots contained in `plot_gdf`. Returns a gpd
    dataframe, together with a set of used plots. `megaplots` can be a list of
    geometries corresponding to the megaplots, or an int, in which case random squares are generated.
    """
    data = pd.DataFrame({
        "observed_area": pd.Series(dtype="float"),
        "megaplot_area": pd.Series(dtype="float"),
        "sr": pd.Series(dtype="int"),
        "num_plots": pd.Series(dtype="int"),
        "geometry": pd.Series(dtype="object"),
    })
    if isinstance(megaplots, int):
        nb_megaplots = megaplots
    else:
        nb_megaplots = len(megaplots)
    miniters = max(nb_megaplots // 100, 1)  # Refresh every 1%
    used_plots = set()
    row_idx = 0
    for i in tqdm(range(nb_megaplots),
                desc="Compiling SR", 
                disable=not verbose,
                total=nb_megaplots, 
                miniters=miniters,
                maxinterval=float("inf")):
        
        # generating random square
        if isinstance(megaplots, int):
            box = generate_random_square(plot_gdf, area_range)
        else:
            box = megaplots.iloc[i]
            
        # identifying plots within the square
        plots_within_box = plot_gdf.within(box)
        df_samp = plot_gdf[plots_within_box]
        if len(df_samp) == 0:
            continue
        else:
            x = np.random.uniform(np.log10(1), np.log10(len(df_samp)))
            x = int(10**x)
            df_samp = df_samp.sample(n=x)

        # calculating species richness and feature values
        species = np.concatenate([species_dict[idx] for idx in df_samp.index])
        sr = len(np.unique(species))
        observed_area = np.sum(df_samp['observed_area'])
        megaplot_area = max(box.area, observed_area)
        num_plots = len(df_samp)
        data.loc[row_idx, ["observed_area", "megaplot_area", "sr", "num_plots", "geometry"]] = [observed_area, megaplot_area, sr, num_plots, box]
        used_plots.update(df_samp.index)
        row_idx += 1
            
    data = gpd.GeoDataFrame(data, crs = plot_gdf.crs, geometry="geometry")
    return data, used_plots

def run_climate_compilation(megaplot_data, climate_raster, verbose=CONFIG["verbose"]):
    """
    Based on a gpd dataframe return by `run_SR_compilation`, calculate
    environmental covariates for each megaplot.
    """
    nb_megaplots = len(megaplot_data)
    miniters = max(nb_megaplots // 100, 1)  # Refresh every 1%
    
    for plot_id, row in tqdm(megaplot_data.iterrows(), 
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
        megaplot_data.loc[plot_id, CLIMATE_COL_NAMES] = env_pred_stats
    return megaplot_data

def run_megaplot_compilation(plot_gdf, species_dict, climate_raster, megaplots, area_range):
    """
    Compiles species richness and environmental covariates for each megaplot.
    """
    megaplot_data, used_plots = run_SR_compilation(plot_gdf, species_dict, megaplots, area_range)
    assert (megaplot_data.sr > 0).all()
    megaplot_data = run_climate_compilation(megaplot_data, climate_raster)
    return megaplot_data, used_plots


# TODO: raw plots are not used anymore, we should simplify this
def run_plot_compilation(plot_data, species_dict, climate_raster):
    """
    Compiles species richness and environmental covariates for each raw (EVA) plot.
    """
    miniters = max(plot_data.shape[0] // 100, 1)  # Refresh every 1%
    for plot_id, row in tqdm(plot_data.iterrows(), 
                       desc="Compiling species richness", 
                       total=plot_data.shape[0],
                       disable=not CONFIG["verbose"],
                       miniters=miniters,
                       maxinterval=float("inf")):
        species = species_dict[plot_id]
        sr = len(np.unique(species))
        plot_data.loc[plot_id, "sr"] = sr

    plot_data = plot_data.rename({"area_m2": "observed_area", "level_1":"habitat_id"}, axis=1)
    plot_data.loc[:, [f"std_{var}" for var in CONFIG["env_vars"]]] = 0.
    plot_data["megaplot_area"] = plot_data["observed_area"]
    plot_data["num_plots"] = 1
    
    logging.info("Calculating climate vars...")
    y = plot_data.geometry.y
    x = plot_data.geometry.x
    env_vars = climate_raster.sel(
        x=xr.DataArray(x, dims="z"),
        y=xr.DataArray(y, dims="z"),
        method="nearest",
    )
    env_vars = env_vars.to_numpy().transpose()
    std_env_vars = np.zeros_like(env_vars)
    plot_data[CLIMATE_COL_NAMES] = np.concatenate([env_vars, std_env_vars], axis=1)
    
    plot_data = plot_data[["sr", "observed_area", "megaplot_area", "geometry", "num_plots"] + CLIMATE_COL_NAMES]

    return plot_data

if __name__ == "__main__":
    # Set up the random seed for reproducibility
    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    
    # Set up the output directory
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)
    output_file_path  = CONFIG["output_file_path"] / sha
    output_file_path.mkdir(parents=True, exist_ok=True)
    
    # Loading data
    plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
    # plot_gdf = plot_gdf.sample(n=1000, random_state=CONFIG["random_state"])     # Sample 1000 rows for debugging purposes
    plot_data_all = run_plot_compilation(plot_gdf, species_dict, climate_raster)
    
    # Generating test megaplots
    nb_megaplot_test = min(int(CONFIG["megaplot_test_size"] * len(plot_data_all)), CONFIG["num_polygon_max"])
    logging.info("Compiling megaplot test dataset...")
    EVA_megaplot_test, used_plots = run_megaplot_compilation(plot_data_all, 
                                                             species_dict, 
                                                             climate_raster, 
                                                             nb_megaplot_test,
                                                             CONFIG["area_range_test"])
    
    # Generating training megaplots
    EVA_raw_megaplot_train = plot_data_all[~plot_data_all.index.isin(used_plots)]
    logging.info("Compiling megaplot train dataset...")
    nb_megaplots_train = min(CONFIG["megaplot_train_size"] * len(EVA_raw_megaplot_train), CONFIG["num_polygon_max"])
    EVA_megaplot_train, _ = run_megaplot_compilation(EVA_raw_megaplot_train,
                                                    species_dict,
                                                    climate_raster,
                                                    nb_megaplots_train,
                                                    CONFIG["area_range_train"])
    
    EVA_megaplot_test["test"] = True
    EVA_megaplot_train["test"] = False
    
    # Save checkpoint
    megaplot_data = pd.concat([EVA_megaplot_test, EVA_megaplot_train], ignore_index=True, verify_integrity=True)
    
    # export the full compilation to pickle
    output_path = output_file_path / "eva_chelsa_megaplot_data.pkl"
    logging.info(f"Exporting {output_path}")
    save_to_pickle(output_path, 
                   megaplot_data=megaplot_data, 
                   config=CONFIG)
    
    # exporting megaplot_data to gpkg
    output_path = output_file_path / "eva_chelsa_megaplot_data.gpkg"
    logging.info(f"Exporting {output_path}")
    megaplot_data.to_file(output_path, driver="GPKG")
    
    logging.info(f'Full compilation saved at {output_file_path}.')

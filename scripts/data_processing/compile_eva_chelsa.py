"""
Compiles training samples based on EVA and CHELSA data.
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

from deepsar.generate_sar_data_eva import clip_EVA_SR, generate_random_square
from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.utils_env_pred import CHELSADataset
from deepsar.utils import save_to_pickle

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
    "sp_unit_test_size": 0.005, # in fraction of total EVA records
    "sp_unit_train_size": 3, # in fraction of total EVA train records
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
                       sp_units, # a list of geometries or an int
                       area_range = None, 
                       verbose=CONFIG["verbose"]):
    
    """
    Calculate species richness, area and observed area for each spatial unit,
    defined as bags of EVA raw plots contained in `plot_gdf`. Returns a gpd
    dataframe, together with a set of used plots. `sp_units` can be a list of
    geometries corresponding to the sp_units, or an int, in which case random squares are generated.
    """
    data = pd.DataFrame({
        "observed_area": pd.Series(dtype="float"),
        "sp_unit_area": pd.Series(dtype="float"),
        "sr": pd.Series(dtype="int"),
        "num_plots": pd.Series(dtype="int"),
        "geometry": pd.Series(dtype="object"),
    })
    if isinstance(sp_units, int):
        nb_sp_units = sp_units
    else:
        nb_sp_units = len(sp_units)
    miniters = max(nb_sp_units // 100, 1)  # Refresh every 1%
    used_plots = set()
    row_idx = 0
    for i in tqdm(range(nb_sp_units),
                desc="Compiling SR", 
                disable=not verbose,
                total=nb_sp_units, 
                miniters=miniters,
                maxinterval=float("inf")):
        
        # generating random square
        if isinstance(sp_units, int):
            box = generate_random_square(plot_gdf, area_range)
        else:
            box = sp_units.iloc[i]
            
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
        sp_unit_area = max(box.area, observed_area)
        num_plots = len(df_samp)
        data.loc[row_idx, ["observed_area", "sp_unit_area", "sr", "num_plots", "geometry"]] = [observed_area, sp_unit_area, sr, num_plots, box]
        used_plots.update(df_samp.index)
        row_idx += 1
            
    data = gpd.GeoDataFrame(data, crs = plot_gdf.crs, geometry="geometry")
    return data, used_plots

def run_climate_compilation(sp_unit_data, climate_raster, verbose=CONFIG["verbose"]):
    """
    Based on a gpd dataframe return by `run_SR_compilation`, calculate
    environmental covariates for each sp_unit.
    """
    nb_sp_units = len(sp_unit_data)
    miniters = max(nb_sp_units // 100, 1)  # Refresh every 1%
    
    for plot_id, row in tqdm(sp_unit_data.iterrows(), 
                       total=nb_sp_units, 
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
        sp_unit_data.loc[plot_id, CLIMATE_COL_NAMES] = env_pred_stats
    return sp_unit_data

def run_sp_unit_compilation(plot_gdf, species_dict, climate_raster, sp_units, area_range):
    """
    Compiles species richness and environmental covariates for each spatial unit.
    """
    sp_unit_data, used_plots = run_SR_compilation(plot_gdf, species_dict, sp_units, area_range)
    assert (sp_unit_data.sr > 0).all()
    sp_unit_data = run_climate_compilation(sp_unit_data, climate_raster)
    return sp_unit_data, used_plots


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
    plot_data["sp_unit_area"] = plot_data["observed_area"]
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
    
    plot_data = plot_data[["sr", "observed_area", "sp_unit_area", "geometry", "num_plots"] + CLIMATE_COL_NAMES]

    return plot_data


def export_dataset_statistics(plot_gdf, species_dict, output_file_path):
    """
    Calculate and export dataset statistics to a text file.
    
    Args:
        plot_gdf: GeoDataFrame containing plot data
        species_dict: Dictionary mapping plot IDs to species lists
        output_file_path: Path where statistics file should be saved
    """
    logging.info("Calculating dataset statistics...")
    num_entries = len(plot_gdf)
    all_species = np.concatenate([species_dict[idx] for idx in plot_gdf.index])
    num_distinct_species = len(np.unique(all_species))

    stats_file_path = output_file_path / "dataset_statistics.txt"
    logging.info(f"Exporting dataset statistics to {stats_file_path}")
    with open(stats_file_path, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n")
        f.write(f"Number of entries: {num_entries}\n")
        f.write(f"Number of distinct species: {num_distinct_species}\n")

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

    export_dataset_statistics(plot_gdf, species_dict, output_file_path)
    
    # plot_gdf = plot_gdf.sample(n=1000, random_state=CONFIG["random_state"])     # Sample 1000 rows for debugging purposes
    plot_data_all = run_plot_compilation(plot_gdf, species_dict, climate_raster)
    
    # Generating test sp_units
    nb_sp_unit_test = min(int(CONFIG["sp_unit_test_size"] * len(plot_data_all)), CONFIG["num_polygon_max"])
    logging.info("Compiling sp_unit test dataset...")
    EVA_sp_unit_test, used_plots = run_sp_unit_compilation(plot_data_all, 
                                                             species_dict, 
                                                             climate_raster, 
                                                             nb_sp_unit_test,
                                                             CONFIG["area_range_test"])
    
    # Generating training sp_units
    EVA_raw_sp_unit_train = plot_data_all[~plot_data_all.index.isin(used_plots)]
    logging.info("Compiling sp_unit train dataset...")
    nb_sp_units_train = min(CONFIG["sp_unit_train_size"] * len(EVA_raw_sp_unit_train), CONFIG["num_polygon_max"])
    EVA_sp_unit_train, _ = run_sp_unit_compilation(EVA_raw_sp_unit_train,
                                                    species_dict,
                                                    climate_raster,
                                                    nb_sp_units_train,
                                                    CONFIG["area_range_train"])
    
    EVA_sp_unit_test["test"] = True
    EVA_sp_unit_train["test"] = False
    
    # Save checkpoint
    sp_unit_data = pd.concat([EVA_sp_unit_test, EVA_sp_unit_train], ignore_index=True, verify_integrity=True)
    
    # export the full compilation to pickle
    output_path = output_file_path / "eva_chelsa_compiled_data.pkl"
    logging.info(f"Exporting {output_path}")
    save_to_pickle(output_path, 
                   sp_unit_data=sp_unit_data, 
                   config=CONFIG)
    
    # exporting sp_unit_data to gpkg
    output_path = output_file_path / "eva_chelsa_compiled_data.parquet"
    logging.info(f"Exporting {output_path}")
    sp_unit_data.to_parquet(output_path)
    
    logging.info(f'Full compilation saved at {output_file_path}.')

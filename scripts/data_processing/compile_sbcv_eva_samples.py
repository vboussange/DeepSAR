"""
Compiles training samples based on EVA and environmental feature data using Spatial Block Cross-Validation.

This script generates training data by:
1. Loading EVA species data.
2. Assigning spatial blocks (checkerboard) to each plot.
3. Splitting data into `n_splits` folds based on spatial blocks.
4. For each fold:
    - Generating random spatial units (polygons) for training (using training plots).
    - Generating random spatial units (polygons) for testing (using testing plots).
    - Computing species richness within each polygon.
    - Extracting environmental feature statistics.
    - Saving the datasets.
"""

import geopandas as gpd
from pathlib import Path
import xarray as xr
import logging
import json
import os
import multiprocessing as mp

from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.spatial_folds import assign_checkerboard_folds
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari.data_processing.SR_compilation_ckdtree import run_SR_compilation_ckdtree
from muscari.data_processing.env_feat_compilation import run_environmental_features_compilation_parallel
from muscari.utils import get_git_hash, json_ready

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

_EVA_DF = None
_ENV_DS = None
_LC_DS = None
_OUTPUT_PATH = None

CONFIG = {
    "output_file_path": Path(
        Path(__file__).parent,
        "../../data/processed/training_samples/sbcv",
    ),
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
        "elevation",
        "landcover",
    ],
    "spunit_area_range_test": (2e3**2, 1e6**2),  # in m2
    "spunit_area_range_train": (2e3**2, 1e6**2),  # in m2
    "random_state": 2,
    "verbose": True,
    "num_workers": 100,  # number of parallel workers for env feature compilation
    "n_splits": 5, # number of spatial folds, should be >=3
    "block_size": 1_000,  # Main 1 km spatial-block dataset; edit explicitly for new datasets.
    "ratio_samples_plots": 1.0, # ratio of genrated train/val/test samples to raw plots, should be ~1
}

def write_dataset_metadata(output_path: Path, dataset_id: str, df: gpd.GeoDataFrame) -> None:
    spatial_split_counts = (
        df["spatial_split"].value_counts().sort_index().astype(int).to_dict()
    )
    metadata = {
        "dataset_id": dataset_id,
        "generated_by": str(Path(__file__)),
        "git_hash": get_git_hash(),
        "n_source_plots": int(len(df)),
        "crs": str(df.crs),
        "bounds": [float(x) for x in df.total_bounds],
        "spatial_folds": {
            "method": "checkerboard",
            "n_splits": CONFIG["n_splits"],
            "block_size_m": CONFIG["block_size"],
            "spatial_split_counts": spatial_split_counts,
            "columns": ["grid_x", "grid_y", "spatial_split"],
        },
        "sample_generation": {
            "ratio_samples_plots": CONFIG["ratio_samples_plots"],
            "spunit_area_range_train_m2": CONFIG["spunit_area_range_train"],
            "spunit_area_range_test_m2": CONFIG["spunit_area_range_test"],
            "random_state": CONFIG["random_state"],
            "fold_seed_policy": "random_state + fold_id",
        },
        "environmental_variables": CONFIG["env_vars"],
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(json_ready(metadata), f, indent=2)


def run_sp_unit_compilation(
    df: gpd.GeoDataFrame,
    env_raster: xr.Dataset,
    lc_raster: xr.Dataset,
    n_sp_units: int,
    area_range: tuple,
    random_state: int,
    verbose: bool = CONFIG["verbose"],
    env_var_names: list = CONFIG["env_vars"],
) -> gpd.GeoDataFrame:
    """
    Full pipeline: generate spatial units, compute SR, and extract environmental features.
    """
    # Step 1: Generate spatial units and compute SR
    sp_unit_data = run_SR_compilation_ckdtree(
        df,
        n_sp_units,
        area_range,
        verbose=verbose,
        random_state=random_state,
    )
    
    # Step 2: Validate SR
    assert (sp_unit_data.sr > 0).all(), "Found spatial units with zero species richness"
    
    # Step 3: Extract environmental features
    sp_unit_data = run_environmental_features_compilation_parallel(
        sp_unit_data, env_raster, lc_raster, env_var_names, num_workers=CONFIG["num_workers"], verbose=verbose
    )
    
    return sp_unit_data

def save_compiled_data(
    sp_unit_data: gpd.GeoDataFrame,
    output_path: Path,
    filename: str,
) -> None:
    """
    Save compiled spatial unit data to GeoParquet format.
    """
    file_path = output_path / f"{filename}.parquet"
    logging.info(f"Saving compiled data to {file_path}")
    
    sp_unit_data.to_parquet(
        file_path,
        compression='zstd',
        index=False,
    )
    
    # Also save a lightweight summary
    summary_path = output_path / f"{filename}_summary.json"
    summary = {
        "ratio_samples_plots": CONFIG["ratio_samples_plots"],
        "spatial_block_size_m": CONFIG["block_size"],
        "n_spatial_folds": CONFIG["n_splits"],
        "environmental_variables": CONFIG["env_vars"],
        "n_samples": len(sp_unit_data),
        "columns": list(sp_unit_data.columns),
        "sr_range": [int(sp_unit_data.sr.min()), int(sp_unit_data.sr.max())],
        "spunit_area_range": [float(sp_unit_data.sp_unit_area.min()), float(sp_unit_data.sp_unit_area.max())],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def configure_worker_logging(fold_id: int) -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - Fold {fold_id} - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)


def process_fold(fold_id: int) -> int:
    configure_worker_logging(fold_id)
    logger = logging.getLogger(__name__)
    logger.info(f"Processing Fold {fold_id + 1}/{CONFIG['n_splits']}...")

    df = _EVA_DF
    env_ds = _ENV_DS
    lc_ds = _LC_DS
    output_path = _OUTPUT_PATH
    fold_seed = CONFIG["random_state"] + fold_id

    test_fold_id = fold_id
    val_fold_id = (fold_id + 1) % CONFIG["n_splits"]

    test_df = df[df["spatial_split"] == test_fold_id]
    val_df = df[df["spatial_split"] == val_fold_id]
    train_df = df[(df["spatial_split"] != test_fold_id) & (df["spatial_split"] != val_fold_id)]

    logger.info(
        f"Fold {fold_id}: Train plots: {len(train_df):,}, "
        f"Val plots: {len(val_df):,}, Test plots: {len(test_df):,}"
    )

    logger.info(f"Generating training samples for Fold {fold_id}...")
    train_data = run_sp_unit_compilation(
        train_df,
        env_ds,
        lc_ds,
        n_sp_units=int(CONFIG["ratio_samples_plots"] * len(train_df)),
        area_range=CONFIG["spunit_area_range_train"],
        random_state=fold_seed,
        env_var_names=CONFIG["env_vars"],
    )
    save_compiled_data(train_data, output_path, f"fold_{fold_id}_train")

    logger.info(f"Generating validation samples for Fold {fold_id}...")
    val_data = run_sp_unit_compilation(
        val_df,
        env_ds,
        lc_ds,
        n_sp_units=int(CONFIG["ratio_samples_plots"] * len(val_df)),
        area_range=CONFIG["spunit_area_range_train"],
        random_state=fold_seed,
        env_var_names=CONFIG["env_vars"],
    )
    save_compiled_data(val_data, output_path, f"fold_{fold_id}_val")

    logger.info(f"Generating test samples for Fold {fold_id}...")
    test_data = run_sp_unit_compilation(
        test_df,
        env_ds,
        lc_ds,
        n_sp_units=int(CONFIG["ratio_samples_plots"] * len(test_df)),
        area_range=CONFIG["spunit_area_range_test"],
        random_state=fold_seed,
        env_var_names=CONFIG["env_vars"],
    )
    save_compiled_data(test_data, output_path, f"fold_{fold_id}_test")

    logger.info(f"Completed Fold {fold_id}.")
    return fold_id

if __name__ == "__main__":
    # Set up output directory with git hash
    sha = get_git_hash()
    output_file_path = CONFIG["output_file_path"] / sha
    output_file_path.mkdir(parents=True, exist_ok=True)
    
    # Load EVA data (species matrix format)
    logging.info("Loading EVA data...")
    df = EVADataset.from_source()
    logging.info(f"Loaded {len(df):,} plots, with {df.shape[1] - 2} distinct species.")
    
    # Assign spatial folds
    logging.info("Assigning spatial folds...")
    df = assign_checkerboard_folds(
        df, 
        n_splits=CONFIG["n_splits"], 
        block_size=CONFIG["block_size"]
    )
    write_dataset_metadata(output_file_path, sha, df)
    
    # Load environmental rasters
    logging.info("Loading environmental rasters...")
    env_features = EnvironmentalFeatureDataset()
    chelsa_dem_ds, lc_ds = env_features.load(use_cache=True)
    
    _EVA_DF = df
    _ENV_DS = chelsa_dem_ds
    _LC_DS = lc_ds
    _OUTPUT_PATH = output_file_path

    ctx = mp.get_context("fork")
    n_workers = min(CONFIG["n_splits"], os.cpu_count() or 1)
    logging.info(f"Processing folds in parallel with {n_workers} workers...")
    with ctx.Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(process_fold, range(CONFIG["n_splits"])):
            pass

    with open(output_file_path / "config_used.json", 'w') as f:
        json.dump(json_ready(CONFIG), f, indent=2)
    logging.info("Done!")

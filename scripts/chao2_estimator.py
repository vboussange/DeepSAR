"""
Total species richness of GIFT polygons estimated using Chao2 (incidence-based)
and benchmarked across spatial folds, using the same GIFT/EVA datasets as
`scripts/benchmark.py`.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from sklearn.metrics import (
    d2_absolute_error_score,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from deepsar.data_processing.utils_eva import EVADataset
from deepsar.data_processing.spatial_folds import assign_checkerboard_folds
# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(
    logging.WARNING
)  # see https://stackoverflow.com/questions/65398774/numba-printing-information-regarding-nvidia-driver-to-python-console-when-using

GIFT_SAMPLES_PATH = Path(__file__).parent / "../data/processed/test_samples_GIFT/6dcd90c/compiled_data.parquet"
SBCV_SAMPLES_PATH = Path(__file__).parent / "../data/processed/training_samples/sbcv/fee8771"

CONFIG = {
    "crs": "EPSG:3035",
    "run_folder": Path(Path(__file__).parent, "results", "benchmark"),
    "run_name": f"benchmark_chao2_results_{SBCV_SAMPLES_PATH.name}",
    "seed": 42,  # For reproducibility
    "n_splits": 5,
    "block_size": 20_000,
}

CONFIG["run_folder"].mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data():
    logging.info("Loading EVA plot data and species...")
    eva = EVADataset()
    eva_plots = eva.read_plot_data()
    eva_species = eva.read_species_data()

    eva_plots = eva_plots.to_crs(CONFIG["crs"])
    eva_plots = eva_plots.set_index("record_id")

    logging.info("Building EVA species dict...")
    eva_species_dict = (
        eva_species.groupby("record_id")["anonymised_species_name"].apply(list).to_dict()
    )

    logging.info("Assigning EVA spatial folds...")
    eva_plots = assign_checkerboard_folds(
        eva_plots, n_splits=CONFIG["n_splits"], block_size=CONFIG["block_size"]
    )

    logging.info("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(GIFT_SAMPLES_PATH)
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])

    return eva_plots, eva_species_dict, gift_dataset

def estimate_sr(record_ids, species_dict):
    species_lists = [species_dict[idx] for idx in record_ids if idx in species_dict]
    if not species_lists:
        return np.nan, np.nan, np.nan
    species = np.concatenate(species_lists)
    species_counts = pd.Series(species).value_counts()
    
    # chao2 estimator calculation
    f1 = (species_counts == 1).sum()  # number of singletons
    f2 = (species_counts == 2).sum()  # number of doubletons
    S_obs = len(species_counts)       # observed species richness

    if f2 == 0:
        chao2 = S_obs + (f1 * (f1 - 1)) / 2 / (f2 + 1)  # bias-corrected if no doubletons
        var_chao2 = (f1 * (f1 - 1)) / 2 + ((f1 * (2 * f1 - 1) ** 2) / 4) - f1**4 / (4 * chao2)
    else:
        chao2 = S_obs + (f1 ** 2) / (2 * f2)
        var_chao2 = f2 * ((f1 / f2) ** 4) / 4 + ((f1 ** 3) / (2 * f2 ** 2))    
    
    return S_obs, chao2, var_chao2


eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data()

gift_dataset = gift_dataset[gift_dataset.is_valid]
gift_dataset = gift_dataset[~gift_dataset.is_empty]

fold_paths = sorted(SBCV_SAMPLES_PATH.glob("fold_*_train.parquet"))
n_folds = len(fold_paths) if fold_paths else CONFIG["n_splits"]

results = []

for fold_id in range(n_folds):
    logging.info(f"Processing fold {fold_id}...")
    eva_fold = eva_dataset[eva_dataset["spatial_split"] == fold_id]

    if eva_fold.empty:
        logging.warning(f"No EVA plots found for fold {fold_id}. Skipping.")
        continue

    # Spatial join for plots within GIFT polygons
    joined = gpd.sjoin(
        eva_fold[["geometry", "area_m2"]],
        gift_dataset[["geometry", "sr"]],
        predicate="within",
        how="inner",
    )

    if joined.empty:
        logging.warning(f"No EVA plots intersect GIFT polygons for fold {fold_id}. Skipping.")
        continue

    y_true = []
    y_pred = []

    for gift_idx, group in tqdm(joined.groupby("index_right"), desc=f"Fold {fold_id} polygons"):
        record_ids = group.index
        if len(record_ids) == 0:
            continue

        eva_observed_sr, chao2, var_chao2 = estimate_sr(record_ids, eva_species_dict)
        if np.isnan(chao2):
            continue

        y_true.append(gift_dataset.loc[gift_idx, "sr"])
        y_pred.append(chao2)

    if len(y_true) == 0:
        logging.warning(f"No valid Chao2 predictions for fold {fold_id}. Skipping.")
        continue

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    relative_bias = (y_pred - y_true) / y_true
    median_relative_bias = np.median(relative_bias)
    mean_relative_bias = np.mean(relative_bias)

    metrics = {
        "experiment": "chao2_estimator",
        "fold": fold_id,
        "extrap_r2": r2_score(y_true, y_pred),
        "extrap_d2": d2_absolute_error_score(y_true, y_pred),
        "extrap_rmse": root_mean_squared_error(y_true, y_pred),
        "extrap_mape": mean_absolute_percentage_error(y_true, y_pred),
        "extrap_median_relative_bias": median_relative_bias,
        "extrap_mean_relative_bias": mean_relative_bias,
    }
    results.append(metrics)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(CONFIG["run_folder"] / f"{CONFIG['run_name']}.csv", index=False)

if not metrics_df.empty:
    print("Metrics across folds:")
    print(f"R2: {metrics_df['extrap_r2'].mean():.3f} ± {metrics_df['extrap_r2'].std():.3f}")
    print(f"D2: {metrics_df['extrap_d2'].mean():.3f} ± {metrics_df['extrap_d2'].std():.3f}")
    print(f"RMSE: {metrics_df['extrap_rmse'].mean():.3f} ± {metrics_df['extrap_rmse'].std():.3f}")
    print(f"MAPE: {metrics_df['extrap_mape'].mean():.3f} ± {metrics_df['extrap_mape'].std():.3f}")
    print(f"Median Relative Bias: {metrics_df['extrap_median_relative_bias'].mean():.3f} ± {metrics_df['extrap_median_relative_bias'].std():.3f}")
    print(f"Mean Relative Bias: {metrics_df['extrap_mean_relative_bias'].mean():.3f} ± {metrics_df['extrap_mean_relative_bias'].std():.3f}")
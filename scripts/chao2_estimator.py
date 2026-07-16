"""
Total species richness of GIFT polygons estimated using Chao2 (incidence-based)
and benchmarked across spatial folds, using the same GIFT/EVA datasets as
`scripts/benchmark.py`.
"""
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from muscari.data_processing.utils_eva import EVADataset
from muscari.data_processing.spatial_folds import assign_checkerboard_folds
from muscari.utils import compute_metrics
# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(
    logging.WARNING
)  # see https://stackoverflow.com/questions/65398774/numba-printing-information-regarding-nvidia-driver-to-python-console-when-using

ROOT = Path(__file__).parents[1]
GIFT_DATASET_ID = "418c563"
SBCV_DATASET_ID = "ceacce0"
GIFT_SAMPLES_PATH = (
    ROOT / "data/processed/test_samples_GIFT" / GIFT_DATASET_ID / "compiled_data.parquet"
)
SBCV_SAMPLES_PATH = ROOT / "data/processed/training_samples/sbcv" / SBCV_DATASET_ID
SBCV_CONFIG_PATH = SBCV_SAMPLES_PATH / "config_used.json"

CONFIG = {
    "crs": "EPSG:3035",
    "run_folder": Path(Path(__file__).parent, "results", "benchmark"),
    "run_name": f"benchmark_chao2_results_{SBCV_SAMPLES_PATH.name}",
}


def load_spatial_fold_config() -> tuple[int, int]:
    """Load the exact fold geometry used to compile the SBCV dataset."""
    if not SBCV_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing SBCV generation config: {SBCV_CONFIG_PATH}")
    with open(SBCV_CONFIG_PATH) as handle:
        config = json.load(handle)
    try:
        return int(config["n_splits"]), int(config["block_size"])
    except KeyError as exc:
        raise ValueError(f"Incomplete SBCV generation config: {SBCV_CONFIG_PATH}") from exc


def load_and_preprocess_data(n_splits: int, block_size: int):
    logging.info("Loading EVA plot data and species...")
    eva = EVADataset()
    try:
        eva_plots = eva.read_plot_data()
        eva_species = eva.read_species_data()
    except FileNotFoundError:
        logging.info("Anonymised EVA files not found; falling back to preprocessing files.")
        eva_plots = gpd.read_parquet(eva.data_dir / "preprocessing/plot_data.parquet")
        eva_species = pd.read_parquet(eva.data_dir / "preprocessing/species_data.parquet")
        if "anonymised_species_name" not in eva_species.columns:
            eva_species = eva_species.rename(columns={"species_name": "anonymised_species_name"})

    eva_plots = eva_plots.to_crs(CONFIG["crs"])
    eva_plots = eva_plots.set_index("record_id")

    logging.info("Building EVA species dict...")
    eva_species = eva_species.drop_duplicates(
        subset=["record_id", "anonymised_species_name"]
    )
    eva_species_dict = (
        eva_species.groupby("record_id")["anonymised_species_name"].apply(list).to_dict()
    )

    logging.info("Assigning EVA spatial folds...")
    eva_plots = assign_checkerboard_folds(
        eva_plots,
        n_splits=n_splits,
        block_size=block_size,
    )

    logging.info("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(GIFT_SAMPLES_PATH)
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])

    return eva_plots, eva_species_dict, gift_dataset

def estimate_sr(record_ids, species_dict):
    record_ids = list(dict.fromkeys(record_ids))
    species = [
        species_name
        for idx in record_ids
        if idx in species_dict
        for species_name in set(species_dict[idx])
    ]
    if not species:
        return np.nan
    species_counts = pd.Series(species).value_counts()
    sample_correction = (len(record_ids) - 1) / len(record_ids)
    
    # chao2 estimator calculation
    f1 = (species_counts == 1).sum()  # number of singletons
    f2 = (species_counts == 2).sum()  # number of doubletons
    observed_sr = len(species_counts)

    if f2 == 0:
        unseen = f1 * (f1 - 1) / 2  # Bias-corrected without doubletons.
    else:
        unseen = f1 ** 2 / (2 * f2)
    return observed_sr + sample_correction * unseen


def main() -> None:
    n_splits, block_size = load_spatial_fold_config()
    logging.info(
        "Using %d folds with %d m blocks from %s.",
        n_splits,
        block_size,
        SBCV_CONFIG_PATH,
    )
    eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data(
        n_splits,
        block_size,
    )
    gift_dataset = gift_dataset[gift_dataset.is_valid & ~gift_dataset.is_empty]

    fold_paths = sorted(SBCV_SAMPLES_PATH.glob("fold_*_train.parquet"))
    if len(fold_paths) != n_splits:
        raise ValueError(
            f"Expected {n_splits} SBCV fold files under {SBCV_SAMPLES_PATH}, "
            f"found {len(fold_paths)}."
        )

    results = []
    for fold_id in range(n_splits):
        logging.info("Processing fold %d...", fold_id)
        eva_fold = eva_dataset[eva_dataset["spatial_split"] == fold_id]
        if eva_fold.empty:
            logging.warning("No EVA plots found for fold %d. Skipping.", fold_id)
            continue

        joined = gpd.sjoin(
            eva_fold[["geometry", "area_m2"]],
            gift_dataset[["geometry", "sr"]],
            predicate="within",
            how="inner",
        )
        if joined.empty:
            logging.warning("No EVA plots intersect GIFT polygons for fold %d. Skipping.", fold_id)
            continue

        y_true = []
        y_pred = []
        for gift_idx, group in tqdm(
            joined.groupby("index_right"),
            desc=f"Fold {fold_id} polygons",
        ):
            chao2 = estimate_sr(group.index, eva_species_dict)
            if np.isnan(chao2):
                continue
            y_true.append(gift_dataset.loc[gift_idx, "sr"])
            y_pred.append(chao2)

        if not y_true:
            logging.warning("No valid Chao2 predictions for fold %d. Skipping.", fold_id)
            continue

        y_true_array = np.asarray(y_true)
        extrap_metrics = compute_metrics(y_true_array, np.asarray(y_pred))
        results.append(
            {
                "experiment": "chao2_estimator",
                "fold": fold_id,
                "sbcv_dataset_id": SBCV_DATASET_ID,
                "gift_dataset_id": GIFT_DATASET_ID,
                "spatial_block_size_m": block_size,
                "extrap_sr_mean": float(np.mean(y_true_array)),
                **{f"extrap_{key}": value for key, value in extrap_metrics.items()},
            }
        )

    metrics_df = pd.DataFrame(results)
    CONFIG["run_folder"].mkdir(parents=True, exist_ok=True)
    output_path = CONFIG["run_folder"] / f"{CONFIG['run_name']}.csv"
    metrics_df.to_csv(output_path, index=False)
    if metrics_df.empty:
        raise RuntimeError("No Chao2 metrics were produced.")

    print("Metrics across folds:")
    for metric, label, scale in [
        ("r2", "R2", 1.0),
        ("d2", "D2", 1.0),
        ("nrmse", "NRMSE (%)", 100.0),
        ("mape", "MAPE", 1.0),
        ("median_relative_bias", "Median Relative Bias", 1.0),
        ("mean_relative_bias", "Mean Relative Bias", 1.0),
    ]:
        values = metrics_df[f"extrap_{metric}"] * scale
        print(f"{label}: {values.mean():.3f} ± {values.std():.3f}")


if __name__ == "__main__":
    main()

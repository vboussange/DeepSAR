"""
Total species richness of GIFT polygons estimated using Chao2 estimation (Chao estimator on incidence
data, https://www.uvm.edu/~ngotelli/manuscriptpdfs/Chapter%204.pdf).
"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import logging

from tqdm import tqdm

from deepsar.data_processing.utils_eva import EVADataset

from sklearn.metrics import (d2_absolute_error_score, root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(
    logging.WARNING
)  # see https://stackoverflow.com/questions/65398774/numba-printing-information-regarding-nvidia-driver-to-python-console-when-using

CONFIG = {
    "env_vars": [
        "bio1",
        "pet_penman_mean",
        "sfcWind_mean",
        "bio4",
        "rsds_1981-2010_range_V.2.1",
        "bio12",
        "bio15",
    ],
    "crs": "EPSG:3035",
    "gift_data_dir": Path(__file__).parent / "../data/processed/GIFT_CHELSA_compilation/6c2d61d",
    "run_folder": Path(Path(__file__).parent, 'results', "benchmark"),
    "run_name": "chao2_estimator_benchmark",
    "seed": 42,  # For reproducibility
}

CONFIG["run_folder"].mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data():
    logging.info("Loading EVA data...")
    eva_dataset, eva_species_dict = EVADataset().load()
    eva_dataset = eva_dataset.set_index("plot_id")
    eva_dataset = eva_dataset.to_crs(CONFIG["crs"])
    
    logging.info("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(CONFIG["gift_data_dir"] / "sp_unit_data.parquet")
    gift_dataset = gift_dataset.set_index("entity_ID") 
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    
    return eva_dataset, eva_species_dict, gift_dataset

def estimate_sr(plots, species_dict):
    species = np.concatenate([species_dict[idx] for idx in plots.index])
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
gift_dataset["chao2"] = np.nan
gift_dataset["var_chao2"] = np.nan
gift_dataset["eva_observed_sr"] = np.nan
gift_dataset["eva_observed_area"] = np.nan

for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within_box = eva_dataset.within(geom)
    df_box = eva_dataset[plots_within_box]
    if not df_box.empty:
        eva_observed_sr, chao2, var_chao2 = estimate_sr(df_box, eva_species_dict)
        gift_dataset.at[idx, "chao2"] = chao2
        gift_dataset.at[idx, "var_chao2"] = var_chao2
        gift_dataset.at[idx, "eva_observed_sr"] = eva_observed_sr
        gift_dataset.at[idx, "eva_observed_area"] = df_box["area_m2"].sum()



        print(
            f"Index: {idx} | chao2: {chao2:.2f} | Var(chao2): {var_chao2:.2f} | "
            f"EVA observed SR: {eva_observed_sr} | GIFT observed SR: {row.sr}"
        )
    
y_true = gift_dataset["sr"].values
y_pred = gift_dataset["chao2"].values
y_var_pred = gift_dataset["var_chao2"].values

# Generate multiple replicates based on prediction uncertainty
np.random.seed(CONFIG["seed"])  # For reproducibility
n_replicates = 5
metrics_replicates = []

# Create mask for valid predictions
valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_var_pred))
y_true_valid = y_true[valid_mask]
y_pred_valid = y_pred[valid_mask]
y_std_valid = np.sqrt(y_var_pred[valid_mask])

for i in range(n_replicates):
    # Generate noisy predictions based on uncertainty
    y_pred_replicate = np.random.normal(y_pred_valid, y_std_valid)
    # Ensure predictions are non-negative
    y_pred_replicate = np.maximum(y_pred_replicate, 0)
    
    # Calculate metrics for this replicate
    r2_rep = r2_score(y_true_valid, y_pred_replicate)
    d2_rep = d2_absolute_error_score(y_true_valid, y_pred_replicate)
    rmse_rep = root_mean_squared_error(y_true_valid, y_pred_replicate)
    mape_rep = mean_absolute_percentage_error(y_true_valid, y_pred_replicate)
    gift_relative_bias = (y_true_valid - y_pred_replicate) / y_true_valid
    median_relative_bias = np.median(gift_relative_bias)
    
    metrics_replicates.append({
        'run': i,
        'r2_gift': r2_rep,
        'd2_gift': d2_rep,
        'rmse_gift': rmse_rep,
        'mape_gift': mape_rep,
        'median_relative_bias': median_relative_bias,
    })

# Convert to DataFrame for analysis
metrics_df = pd.DataFrame(metrics_replicates)
metrics_df["model"] = "chao2_estimator"
metrics_df.to_csv(CONFIG["run_folder"] / f'{CONFIG["run_name"]}.csv', index=False)

# Calculate summary statistics
print("Metrics across replicates:")
print(f"R2: {metrics_df['r2_gift'].mean():.3f} ± {metrics_df['r2_gift'].std():.3f}")
print(f"D2: {metrics_df['d2_gift'].mean():.3f} ± {metrics_df['d2_gift'].std():.3f}")
print(f"RMSE: {metrics_df['rmse_gift'].mean():.3f} ± {metrics_df['rmse_gift'].std():.3f}")
print(f"MAPE: {metrics_df['mape_gift'].mean():.3f} ± {metrics_df['mape_gift'].std():.3f}")
print(f"Median Relative Bias: {metrics_df['median_relative_bias'].mean():.3f} ± {metrics_df['median_relative_bias'].std():.3f}")
    
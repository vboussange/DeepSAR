"""
estimating SR with pydistinct
"""
import matplotlib.pyplot as plt
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
from src.data_processing.utils_gift import GIFTDataset
from src.data_processing.utils_env_pred import CHELSADataset
from src.utils import save_to_pickle

import git
import random
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, d2_absolute_error_score, mean_squared_error
from pydistinct.ensemble_estimators import median_estimator

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
}

# Define covariate feature names based on environmental covariates
mean_labels = CONFIG["env_vars"]
std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()

def load_and_preprocess_data():
    logging.info("Loading EVA data...")
    eva_dataset, eva_species_dict = EVADataset().load()
    eva_dataset = eva_dataset.set_index("plot_id")
    eva_dataset = eva_dataset.to_crs(CONFIG["crs"])
    
    logging.info("Loading GIFT data...")
    gift_dataset, gift_species_dict = GIFTDataset().load()
    gift_dataset = gift_dataset.set_index("entity_ID") 
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    
    return eva_dataset, eva_species_dict, gift_dataset, gift_species_dict

def estimate_sr(plots, species_dict):
    species = np.concatenate([species_dict[idx] for idx in plots.index])
    species_counts = pd.Series(species).value_counts()
    S_obs = len(species_counts)       # observed species richness

    return median_estimator(attributes=species_counts.to_dict()), S_obs
    


eva_dataset, eva_species_dict, gift_dataset, gift_species_dict = load_and_preprocess_data()
# eva_dataset = eva_dataset[eva_dataset.level_1 == "S"]
gift_dataset = gift_dataset[gift_dataset.level_1 == "all"]

gift_dataset = gift_dataset[gift_dataset.is_valid]
gift_dataset = gift_dataset[~gift_dataset.is_empty]
gift_dataset["sr_pydistinct"] = 0
gift_dataset["gift_observed_sr"] = 0
gift_dataset["eva_observed_sr"] = 0

for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within_box = eva_dataset.within(geom)
    df_box = eva_dataset[plots_within_box]
    gift_observed_sr = len(np.unique(gift_species_dict[idx]))
    gift_dataset.at[idx, "gift_observed_sr"] = gift_observed_sr
    
    if len(df_box) == 0:
        gift_dataset.at[idx, "sr_pydistinct"] = np.nan
        gift_dataset.at[idx, "eva_observed_sr"] = np.nan
        continue
    sr_pydistinct, eva_observed_sr = estimate_sr(df_box, eva_species_dict)
    gift_dataset.at[idx, "sr_pydistinct"] = sr_pydistinct
    gift_dataset.at[idx, "eva_observed_sr"] = eva_observed_sr

    print(
        f"Index: {idx} | sr_pydistinct: {sr_pydistinct:.2f}"
        f"EVA observed SR: {eva_observed_sr} | GIFT observed SR: {gift_observed_sr}"
    )
    
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(gift_dataset["gift_observed_sr"], gift_dataset["sr_pydistinct"], alpha=0.7)
max_val = np.nanmax([gift_dataset["gift_observed_sr"].max(), gift_dataset["sr_pydistinct"].max()])
axes[0].plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
axes[0].set_xlabel("GIFT observed SR")
axes[0].set_ylabel("pydistinct estimated SR")

# Compute R2, D2, and MSE for Chao1
mask0 = gift_dataset[["gift_observed_sr", "sr_pydistinct"]].dropna()
r2_0 = r2_score(mask0["gift_observed_sr"], mask0["sr_pydistinct"])
d2_0 = d2_absolute_error_score(mask0["gift_observed_sr"], mask0["sr_pydistinct"])
mse_0 = mean_squared_error(mask0["gift_observed_sr"], mask0["sr_pydistinct"])
axes[0].text(
    0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nMSE={mse_0:.2f}",
    transform=axes[0].transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
axes[0].legend()

# Scatter plot: GIFT observed SR vs EVA observed SR
axes[1].scatter(gift_dataset["gift_observed_sr"], gift_dataset["eva_observed_sr"], alpha=0.7)
max_val = np.nanmax([gift_dataset["gift_observed_sr"].max(), gift_dataset["eva_observed_sr"].max()])
axes[1].plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
axes[1].set_xlabel("GIFT observed SR")
axes[1].set_ylabel("EVA observed SR")

# Compute R2, D2, and MSE for EVA observed SR
mask1 = gift_dataset[["gift_observed_sr", "eva_observed_sr"]].dropna()
r2_1 = r2_score(mask1["gift_observed_sr"], mask1["eva_observed_sr"])
d2_1 = d2_absolute_error_score(mask1["gift_observed_sr"], mask1["eva_observed_sr"])
mse_1 = mean_squared_error(mask1["gift_observed_sr"], mask1["eva_observed_sr"])
axes[1].text(
    0.05, 0.95, f"R2={r2_1:.2f}\nD2={d2_1:.2f}\nMSE={mse_1:.2f}",
    transform=axes[1].transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
axes[1].legend()

plt.tight_layout()
plt.show()
   
   
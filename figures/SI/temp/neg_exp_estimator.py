"""
Assesses species richness estimator using rarefaction and negative exponential fit.
Gift dataset is assumed ground truth.
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
from sklearn.metrics import r2_score, d2_absolute_error_score, root_mean_squared_error

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

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

def rarefaction_curve(plots, species_dict):
    n_plots = len(plots)
    if n_plots == 0:
        return np.array([]), np.array([])
    
    # Define systematically spaced sample sizes
    if n_plots <= 20:
        sample_sizes = list(range(1, n_plots + 1))
    else:
        # For larger datasets, create ~20 sample points
        step = max(1, n_plots // 20)
        sample_sizes = list(range(1, n_plots + 1, step))
        if sample_sizes[-1] != n_plots:
            sample_sizes.append(n_plots)
    
    xs = []
    ys = []
    
    # For each sample size, perform multiple random draws
    n_draws = 10
    
    for size in sample_sizes:
        sr_values = []
        for _ in range(n_draws):
            # Sample plots without replacement
            if size == n_plots:
                df_samp = plots
            else:
                df_samp = plots.sample(n=size)
            
            # Get unique species from sampled plots
            species = np.concatenate([species_dict[idx] for idx in df_samp.index])
            sr = len(np.unique(species))
            sr_values.append(sr)
        
        # Calculate mean species richness for this sample size
        xs.append(size)
        ys.append(np.mean(sr_values))
    
    return np.array(xs), np.array(ys)

def neg_exp_func(x, a, b):
    return a*(1. - np.exp(-b * x))

def estimate_sr_negexp(plots, species_dict):
    x, y = rarefaction_curve(plots, species_dict)
    if len(x) < 2:
        return np.nan, np.nan
    # Initial guess: a=max(y),b=0.1
    try:
        popt, pcov = curve_fit(neg_exp_func, x, y, p0=[max(y), 0.1], maxfev=10000)
        a = popt[0]
        b = popt[1]
        return a, b
    except Exception as e:
        logging.warning(f"Curve fit failed: {e}")
        return np.nan, np.nan

eva_dataset, eva_species_dict, gift_dataset, gift_species_dict = load_and_preprocess_data()
eva_dataset = eva_dataset[eva_dataset.level_1 == "R"]
gift_dataset = gift_dataset[gift_dataset.level_1 == "R"]
gift_dataset = gift_dataset[gift_dataset.is_valid]
gift_dataset = gift_dataset[~gift_dataset.is_empty]
gift_dataset["a"] = 0.0
gift_dataset["b"] = 0.0
gift_dataset["gift_observed_sr"] = 0
gift_dataset["eva_observed_sr"] = 0

# test rarefaction and negative exponential fit for a single GIFT plot
test_idx = gift_dataset.index[7]
test_geom = gift_dataset.loc[test_idx].geometry
plots_within = eva_dataset.within(test_geom)
df_test = eva_dataset[plots_within]

if len(df_test) > 0:
    x, y = rarefaction_curve(df_test, eva_species_dict)
    a, b = estimate_sr_negexp(df_test, eva_species_dict)
    plt.figure()
    plt.errorbar(x, y, yerr=None, fmt="o", label="Rarefaction")
    plt.plot(x, neg_exp_func(x, a, b), label=f"NegExp fit (a={a:.1f}, b={b:.2f})")
    plt.xlabel("Number of plots")
    plt.ylabel("Species richness")
    plt.title(f"Rarefaction & NegExp fit for GIFT plot {test_idx}")
    plt.legend()
    plt.show()
else:
    print(f"No EVA plots within GIFT plot {test_idx}")



# estimating SR for each GIFT plot
for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within_box = eva_dataset.within(geom)
    df_box = eva_dataset[plots_within_box]
    gift_observed_sr = len(np.unique(gift_species_dict[idx]))
    gift_dataset.at[idx, "gift_observed_sr"] = gift_observed_sr

    if len(df_box) == 0:
        gift_dataset.at[idx, "a"] = np.nan
        gift_dataset.at[idx, "b"] = np.nan
        gift_dataset.at[idx, "eva_observed_sr"] = np.nan
        continue
    eva_observed_sr = len(np.unique(np.concatenate([eva_species_dict[i] for i in df_box.index])))
    gift_dataset.at[idx, "eva_observed_sr"] = eva_observed_sr

    a, b = estimate_sr_negexp(df_box, eva_species_dict)
    gift_dataset.at[idx, "a"] = a
    gift_dataset.at[idx, "b"] = b

    print(
        f"Index: {idx} | NegExp Asymptote: {a:.2f} | b: {b:.2f} | "
        f"EVA observed SR: {eva_observed_sr} | GIFT observed SR: {gift_observed_sr}"
    )

output_path = Path(__file__).parent / f"{Path(__file__).stem}.parquet"
gift_dataset.to_parquet(output_path)
logging.info(f"Results saved to {output_path}")
"""
Total species richness estimation using rarefaction curve fitted with log linear
model `S = log(sum(plot area)) * b + a`. Gift dataset is assumed ground truth.
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
from sklearn.metrics import r2_score, d2_absolute_error_score

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
    "gift_data_dir": Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered",
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
    gift_dataset = gpd.read_file(CONFIG["gift_data_dir"] / "plot_data.gpkg")
    gift_dataset = gift_dataset.set_index("entity_ID") 
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    
    return eva_dataset, eva_species_dict, gift_dataset

def rarefaction_curve(plots, species_dict):
    n_plots = len(plots)
    if n_plots == 0:
        return np.array([]), np.array([])
    
    # Define log-linearly spaced sample sizes
    if n_plots <= 20:
        sample_sizes = list(range(1, n_plots + 1))
    else:
        # Create ~20 log-linearly spaced sample points
        log_space = np.logspace(np.log10(2), np.log10(n_plots), 20).astype(int)
        # Remove duplicates that can occur due to rounding to integers
        sample_sizes = sorted(list(set(log_space)))
        # Always include the maximum number of plots
        if sample_sizes[-1] != n_plots:
            sample_sizes.append(n_plots)
    
    xs = []
    ys = []
    
    # For each sample size, perform multiple random draws
    n_draws = 10
    
    for size in sample_sizes:
        sr_values = []
        area_values = []
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
            area_values.append(df_samp.area_m2.sum())
        
        # Calculate mean species richness for this sample size
        xs.append(np.mean(area_values))
        ys.append(np.mean(sr_values))
    
    return np.array(xs), np.array(ys)

eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data()

gift_dataset["slope"] = 0.0
gift_dataset["intercept"] = 0.0
gift_dataset["eva_observed_sr"] = 0
gift_dataset["predicted_sr"] = 0.0


for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within = eva_dataset.within(geom)
    df_test = eva_dataset[plots_within]
    gift_polygon_area = row.area
    if len(df_test) > 0:
        x, y = rarefaction_curve(df_test, eva_species_dict)

        logx = np.log(x)

        # Fit linear model between logx and logy
        slope, intercept = np.polyfit(logx, y, 1)
        predicted_logy = slope * logx + intercept

        # Predict species richness at GIFT polygon area
        log_area = np.log(gift_polygon_area)
        predicted_sr = slope * log_area + intercept
        
        # Update values in the gift_dataset directly using loc
        gift_dataset.loc[idx, "slope"] = slope
        gift_dataset.loc[idx, "intercept"] = intercept
        gift_dataset.loc[idx, "predicted_sr"] = predicted_sr
        gift_dataset.loc[idx, "eva_observed_sr"] = len(np.unique(np.concatenate([eva_species_dict[i] for i in df_test.index])))
output_path = Path(__file__).parent / f"{Path(__file__).stem}.parquet"
gift_dataset.to_parquet(output_path)
logging.info(f"Results saved to {output_path}")

# -----------------------------------------------
# Plotting full predictions

fig, ax = plt.subplots()

# Scatter plot: GIFT observed SR vs Chao1
ax.scatter(gift_dataset["gift_observed_sr"], gift_dataset["predicted_sr"], alpha=0.7)
max_val = np.nanmax([gift_dataset["gift_observed_sr"].max(), gift_dataset["predicted_sr"].max()])
ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax.set_xlabel("GIFT observed SR")
ax.set_ylabel("Predicted SR")

# Compute R2, D2, and MSE for Chao1
mask0 = gift_dataset[["gift_observed_sr", "chao1"]].dropna()
r2_0 = r2_score(mask0["gift_observed_sr"], mask0["chao1"])
d2_0 = d2_absolute_error_score(mask0["gift_observed_sr"], mask0["chao1"])
mse_0 = mean_squared_error(mask0["gift_observed_sr"], mask0["chao1"])
ax.text(
    0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nMSE={mse_0:.2f}",
    transform=ax.transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
ax.legend()

# -------------------------------------------
# Plotting predictions for a single GIFT plot
test_idx = gift_dataset[gift_dataset.geo_entity =="Hungary"].index[0]
test_geom = gift_dataset.loc[test_idx].geometry
plots_within = eva_dataset.within(test_geom)
df_test = eva_dataset[plots_within]
gift_observed_sr = gift_dataset.loc[test_idx].sr
gift_polygon_area = gift_dataset.loc[test_idx]["observed_area"]

if len(df_test) > 0:
    x, y = rarefaction_curve(df_test, eva_species_dict)

    logx = np.log(x)

    # Fit linear model between logx and logy
    slope, intercept = np.polyfit(logx, y, 1)
    predicted_logy = slope * logx + intercept

    # Predict species richness at GIFT polygon area
    log_area = np.log(gift_polygon_area)
    predicted_sr = slope * log_area + intercept
    # Plot results with explicit axes definition
    fig, ax = plt.subplots()
    
    # Plot original rarefaction curve
    ax.scatter(x, y, label="EVA rarefaction data")
    
    # Plot the fit line on original scale
    x_fit = np.linspace(min(x), max(x) * 2, 100)
    y_fit = slope * np.log(x_fit) + intercept
    ax.plot(x_fit, y_fit, 'r-', label=f"Power law fit: S = {np.exp(intercept):.2f} * A^{slope:.2f}")

    # Plot the GIFT prediction point
    ax.scatter(gift_polygon_area, predicted_sr, marker='*', s=200, color='green', 
              label=f"Predicted SR: {predicted_sr:.1f}")
    ax.scatter(gift_polygon_area, gift_observed_sr, marker='*', s=200, color='blue',
              label=f"GIFT observed SR: {gift_observed_sr}")

    # Format plot
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Area (m²)')
    ax.set_ylabel('Species Richness')
    ax.set_title('Species-Area Relationship')
    ax.legend()

    # Add text with fit information
    ax.text(0.05, 0.95, f"log(S) = {slope:.4f} * log(A) + {intercept:.4f}\n"
                        f"Predicted SR: {predicted_sr:.1f}\n"
                        f"GIFT observed SR: {gift_observed_sr}",
           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print(f"Linear model: log(S) = {slope:.4f} * log(A) + {intercept:.4f}")
    print(f"Predicted species richness for GIFT plot: {predicted_sr:.1f}")
    print(f"Actual GIFT observed species richness: {gift_observed_sr}")
else:
    print(f"No EVA plots within GIFT plot {test_idx}")
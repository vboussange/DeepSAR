"""
Total species richness estimation using rarefaction curve fitted with `a*(1. -
np.exp(-b * (sum(area plot))))`. Gift dataset is assumed ground truth.
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
from scipy.stats import linregress
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        xs.append(area_values)
        ys.append(sr_values)
    
    return np.array(xs), np.array(ys)

def weibull4(x, b, c, d, e):
    """4-parameter Weibull: f(x) = c + (d - c) * exp(-exp(b * (log(x) - log(e))))"""
    return c + (d - c) * np.exp(-np.exp(b * (np.log(x) - np.log(e))))

def estimate_sr_weibull(x, y):
    if len(x) < 4:  # 4 parameters need at least 4 points
        return np.nan, np.nan, np.nan, np.nan
    # Initial guess: b=1, c=min(y), d=max(y), e=median(x)
    try:
        popt, pcov = curve_fit(
            weibull4, x, y, 
            p0=[1, max(y), min(y), np.median(x)], 
            maxfev=10000
        )
        b, c, d, e = popt
        return b, c, d, e
    except Exception as e:
        logging.warning(f"Weibull curve fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data()


def calculate_rarefaction_data(country, eva_dataset, eva_species_dict, gift_dataset):
    """Calculate rarefaction curve data for a specific country."""
    # Get country data
    country_indices = gift_dataset[gift_dataset.geo_entity == country].index
    if len(country_indices) == 0:
        print(f"No data found for {country}")
        return None
    
    test_idx = country_indices[0]
    test_geom = gift_dataset.loc[test_idx].geometry
    plots_within = eva_dataset.within(test_geom)
    df_test = eva_dataset[plots_within]
    gift_observed_sr = gift_dataset.loc[test_idx].sr
    gift_polygon_area = gift_dataset.loc[test_idx]["megaplot_area"]
    
    if len(df_test) == 0:
        print(f"No EVA plots within GIFT plot for {country}")
        return None
    
    # Generate rarefaction curve
    x, y = rarefaction_curve(df_test, eva_species_dict)
    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    
    # Fit Weibull model
    b, c, d, e = estimate_sr_weibull(np.log(x_mean), y_mean)
    
    # Determine interpolation vs extrapolation ranges
    max_observed_area = max(x.flatten())
    
    # Calculate interpolation range
    x_interp = None
    y_interp = None
    if max_observed_area > min(x.flatten()):
        x_interp = np.logspace(np.log10(min(x.flatten())), np.log10(max_observed_area), 50)
        y_interp = weibull4(np.log(x_interp), b, c, d, e)
    
    # Calculate extrapolation range
    x_extrap = None
    y_extrap = None
    if gift_polygon_area > max_observed_area:
        x_extrap = np.logspace(np.log10(max_observed_area), np.log10(gift_polygon_area), 50)
        y_extrap = weibull4(np.log(x_extrap), b, c, d, e)
    
    # Calculate predicted species richness
    predicted_sr = weibull4(np.log(gift_polygon_area), b, c, d, e)
    
    return {
        'country': country,
        'x': x,
        'y': y,
        'x_interp': x_interp,
        'y_interp': y_interp,
        'x_extrap': x_extrap,
        'y_extrap': y_extrap,
        'gift_polygon_area': gift_polygon_area,
        'gift_observed_sr': gift_observed_sr,
        'predicted_sr': predicted_sr,
        'parameters': (b, c, d, e)
    }

def plot_rarefaction_curves(data_list, colors):
    """Plot rarefaction curves from pre-calculated data."""

# Generate data for both countries
countries = ["Czech Republic", "Sachsen, Germany"]
colors = ["#f72585","#4cc9f0"]

data_list = []
for country in countries:
    data = calculate_rarefaction_data(country, eva_dataset, eva_species_dict, gift_dataset)
    if data:
        data_list.append(data)


# plotting
fig, ax = plt.subplots(figsize=(4, 4))

for data, color in zip(data_list, colors):
    country = data['country']
    
    # Plot training points
    ax.scatter(data['x'], data['y'], alpha=0.3, color=color, s=20,)
    
    # Plot interpolation range (solid line)
    if data['x_interp'] is not None:
        ax.plot(data['x_interp'], data['y_interp'], color=color, linewidth=2, 
                )
    
    # Plot extrapolation range (dashed line)
    if data['x_extrap'] is not None:
        ax.plot(data['x_extrap'], data['y_extrap'], color=color, linewidth=2, 
                linestyle='--')
    
# Format plot
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Species richness', labelpad=10)
ax.set_xlabel('Sampling effort (m²)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Add legend entries for line types
interp_line = mlines.Line2D([], [], color='grey', linewidth=2, label='Interpolation')
extrap_line = mlines.Line2D([], [], color='grey', linewidth=2, linestyle='--', label='Extrapolation')

training_dot = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5, label='Training data')
ax.legend(handles=[training_dot, interp_line, extrap_line], bbox_to_anchor=(0.6, 0.2), loc='upper left')

fig.tight_layout()
fig.savefig(Path(__file__).parent / "rarefaction_curves.svg", bbox_inches='tight')
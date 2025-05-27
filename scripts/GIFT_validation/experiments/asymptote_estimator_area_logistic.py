"""
Total species richness estimation using rarefaction curve fitted with `a*(1. -
np.exp(-b * (sum(area plot))))`. Gift dataset is assumed ground truth.
"""
import matplotlib.pyplot as plt
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
        xs.append(np.mean(area_values))
        ys.append(np.mean(sr_values))
    
    return np.array(xs), np.array(ys)

def logistic_func(x, a, b, c):
    """Logistic function: a/(1 + exp(-b * (x - c)))"""
    return a/(1 + np.exp(-b * (x - c)))

def estimate_sr_logistic(x, y):
    if len(x) < 3:  # Logistic needs at least 3 points for 3 parameters
        return np.nan, np.nan, np.nan
    # Initial guess: a=max(y)*1.2, b=0.1, c=np.median(x)
    try:
        popt, pcov = curve_fit(logistic_func, x, y, p0=[max(y)*1.2, 0.1, np.median(x)], maxfev=10000)
        a = popt[0]  # asymptote
        b = popt[1]  # rate
        c = popt[2]  # midpoint
        return a, b, c
    except Exception as e:
        logging.warning(f"Curve fit failed: {e}")
        return np.nan, np.nan, np.nan

eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data()

country = "Czech Republic"
test_idx = gift_dataset[gift_dataset.geo_entity == country].index[0]
test_geom = gift_dataset.loc[test_idx].geometry
plots_within = eva_dataset.within(test_geom)
df_test = eva_dataset[plots_within]
gift_observed_sr = gift_dataset.loc[test_idx].sr
gift_polygon_area = gift_dataset.loc[test_idx]["megaplot_area"]

if len(df_test) > 0:
    x, y = rarefaction_curve(df_test, eva_species_dict)

    # Fit logistic model
    a, b, c = estimate_sr_logistic(np.log(x), y)

    # Predict species richness at GIFT polygon area
    log_area = np.log(gift_polygon_area)
    predicted_sr = logistic_func(log_area, a, b, c)
    
    # Plot results
    fig, ax = plt.subplots()
    
    # Plot original rarefaction curve
    ax.scatter(x, y, label="EVA rarefaction data")
    
    # Plot the fit line on original scale
    x_fit = np.logspace(np.log10(min(x)), np.log10(max(x) * 2), 100)
    y_fit = logistic_func(np.log(x_fit), a, b, c)
    ax.plot(x_fit, y_fit, 'r-')

    # Plot the GIFT prediction point
    ax.scatter(gift_polygon_area, predicted_sr, marker='*', s=200, color='green', 
                label=f"Predicted SR: {predicted_sr:.1f}")
    ax.scatter(gift_polygon_area, gift_observed_sr, marker='*', s=200, color='blue',
                label=f"GIFT observed SR: {gift_observed_sr}")

    # Format plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Species Richness')
    ax.set_xlabel('Observed area (m²)')
    ax.set_title(f'Rarefaction curve with logistic fit, {country}')
    ax.legend()

    # Add text with fit information
    # ax.text(0.05, 0.8, f"Logistic parameters: a={a:.1f}, b={b:.3f}, c={c:.1f}\n"
    #                     f"Predicted SR: {predicted_sr:.1f}\n"
    #                     f"GIFT observed SR: {gift_observed_sr}",
    #         transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print(f"Predicted species richness for GIFT plot: {predicted_sr:.1f}")
    print(f"Actual GIFT observed species richness: {gift_observed_sr}")
else:
    print(f"No EVA plots within GIFT plot {test_idx}")
    

fit_df = gift_dataset.copy()
fit_df["a"] = 0.0
fit_df["b"] = 0.0
fit_df["c"] = 0.0
fit_df["eva_observed_sr"] = 0
fit_df["eva_observed_area"] = 0
fit_df["predicted_sr"] = 0.0

for idx, row in tqdm(fit_df.iterrows(), total=fit_df.shape[0]):
    geom = fit_df.loc[idx].geometry
    plots_within = eva_dataset.within(geom)
    df_samp = eva_dataset[plots_within]
    gift_observed_sr = fit_df.loc[test_idx].sr
    gift_polygon_area = fit_df.loc[test_idx]["megaplot_area"]
    if len(df_samp) > 0:
        x, y = rarefaction_curve(df_samp, eva_species_dict)

        # Fit logistic model
        a, b, c = estimate_sr_logistic(np.log(x), y)

        # Predict species richness at GIFT polygon area
        log_area = np.log(gift_polygon_area)
        predicted_sr = logistic_func(log_area, a, b, c)
        
        # Update values in the fit_df directly using loc
        fit_df.loc[idx, "predicted_sr"] = predicted_sr
        fit_df.loc[idx, "a"] = a
        fit_df.loc[idx, "b"] = b
        fit_df.loc[idx, "c"] = c
        fit_df.loc[idx, "eva_observed_sr"] = len(np.unique(np.concatenate([eva_species_dict[i] for i in df_test.index])))
        fit_df.loc[idx, "eva_observed_area"] = df_samp.area_m2.sum()
# output_path = Path(__file__).parent / f"{Path(__file__).stem}.parquet"
# fit_df.to_parquet(output_path)
# logging.info(f"Results saved to {output_path}")
fit_df["coverage"] = np.log(fit_df["eva_observed_area"] / fit_df["megaplot_area"])

# -----------------------------------------------
# Plotting full predictions

fig, ax = plt.subplots()
fit_df = fit_df[fit_df["predicted_sr"] > 0]
fit_df = fit_df[fit_df["predicted_sr"] < 1e4]
fit_df["log_predicted_sr"] = np.log(fit_df["predicted_sr"])
fit_df["log_sr"] = np.log(fit_df["sr"])
mask0 = fit_df[["log_predicted_sr", "log_sr"]].dropna()
x = mask0["log_sr"]
y = mask0["log_predicted_sr"]

# Scatter plot: GIFT observed SR vs Chao1
ax.scatter(x, y,c=fit_df["coverage"], 
                         alpha=0.7, cmap='magma_r')
max_val = np.nanmax([x.max(), y.max()])
ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax.set_xlabel("GIFT observed SR")
ax.set_ylabel("Predicted SR")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(x.min(), x.max())

# Compute R2, D2, and MSE for Chao1
r2_0 = r2_score(x, y)
d2_0 = d2_absolute_error_score(x, y)
mse_0 = np.sqrt(mean_squared_error(x, y))
corr_0 = np.corrcoef(x, y)[0, 1]
ax.text(
    0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nRMSE={mse_0:.2f}\nCorr={corr_0:.2f}",
    transform=ax.transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
ax.legend()
ax.set_title("GIFT vs predicted species richness from logistic fit")




# -----------------------------------------------
# Plotting residuals vs coverage
fit_df["log_coverage"] = np.log(fit_df["eva_observed_area"]) / np.log(fit_df["megaplot_area"])

# Calculate residuals (difference between predicted and actual)
fit_df["log_megaplot_area"] = np.log(fit_df["megaplot_area"])
fit_df['residual'] = fit_df['log_predicted_sr'] - fit_df['log_sr']
xaxis = "log_megaplot_area"
fig_res, ax_res = plt.subplots()

scatter = ax_res.scatter(fit_df[xaxis], fit_df['residual'], 
                         c=fit_df['log_sr'], cmap='viridis', alpha=0.7)

# Add a horizontal line at y=0 (perfect prediction)
ax_res.axhline(y=0, color='r', linestyle='--', label='No error')

# Add a trend line to see if there's a pattern
# Filter out NaN values for regression
valid_mask = ~np.isnan(fit_df[xaxis]) & ~np.isnan(fit_df['residual'])
slope, intercept, r_value, p_value, std_err = linregress(
    fit_df.loc[valid_mask, xaxis], 
    fit_df.loc[valid_mask, 'residual']
)
x_vals = np.array([fit_df[xaxis].min(), fit_df[xaxis].max()])
ax_res.plot(x_vals, intercept + slope * x_vals, 'b-', 
            label=f'Trend: y={slope:.3f}x+{intercept:.3f}, r²={r_value**2:.3f}')

# Add labels and title
ax_res.set_xlabel(xaxis)
ax_res.set_ylabel('Residuals (log predicted SR - log observed SR)')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Log observed species richness')

# Add legend
ax_res.legend()

plt.tight_layout()
plt.show()

# Spatial plot of residuals
fig_map, ax_map = plt.subplots(1, 1, figsize=(10, 8))

# Create a copy of the geodataframe with residuals
plot_df = fit_df.copy()

# Remove rows with NaN residuals
plot_df = plot_df[~plot_df['residual'].isna()]

# Create a colormap with a distinct center (white for zero residual)
cmap = mpl.cm.RdBu_r
norm = mpl.colors.TwoSlopeNorm(vmin=plot_df['residual'].min(), vcenter=0, vmax=plot_df['residual'].max())

# Plot polygons with color based on residuals
plot = plot_df.plot(column='residual', ax=ax_map, cmap=cmap, norm=norm, 
                    edgecolor='black', linewidth=0.5)

# Add a smaller colorbar
divider = make_axes_locatable(ax_map)
cax = divider.append_axes("bottom", size="3%", pad=0.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label('Residual (log predicted - log observed)')

# Add title and configure map
ax_map.set_title('Spatial Distribution of Species Richness Prediction Residuals')
ax_map.set_axis_off()

plt.tight_layout()
plt.show()
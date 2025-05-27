"""
Performing GIFT regression with polygon area as predictor.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, d2_absolute_error_score, mean_squared_error

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

CONFIG = {
    "crs": "EPSG:3035",
    "gift_data_dir": Path(__file__).parent / "../../data/processed/GIFT/preprocessing/unfiltered", # TODO: work in progress
}


def load_and_preprocess_data():
    
    logging.info("Loading GIFT data...")
    # gift_dataset, gift_species_dict = GIFTDataset(data_dir=CONFIG["gift_data_dir"]).load()
    gift_dataset = gpd.read_file(CONFIG["gift_data_dir"] / "plot_data.gpkg")
    # gift_dataset = gift_dataset.set_index("entity_ID") 
    # gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    
    return gift_dataset

def fit_linear_model(gift_dataset, predictor):
    """
    Fit a linear regression model to the data.
    """
    
    # Drop rows with NaN values or zeros in predictor
    valid_data = gift_dataset.dropna(subset=[predictor, 'sr'])
    valid_data = valid_data[valid_data[predictor] > 0]

    X = np.log(valid_data[predictor]).values.reshape(-1, 1)
    y = np.log(valid_data['sr'].values)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

def make_regression(gift_dataset, predictor):
    model = fit_linear_model(gift_dataset,  predictor)
    # Filter out zeros and NaN values before prediction
    valid_pred = gift_dataset[gift_dataset[predictor] > 0].copy()
    valid_pred['predicted_sr'] = np.exp(model.predict(np.log(valid_pred[predictor]).values.reshape(-1, 1)))
    gift_dataset = gift_dataset.merge(valid_pred[['predicted_sr']], left_index=True, right_index=True, how='left')
    # Calculate metrics
    mask = gift_dataset[['sr', 'predicted_sr']].dropna()
    r2 = r2_score(mask['sr'], mask['predicted_sr'])
    d2 = d2_absolute_error_score(mask['sr'], mask['predicted_sr'])
    rmse = np.sqrt(mean_squared_error(mask['sr'], mask['predicted_sr']))


    # ---------------------------------------------
    # Plotting the results
    fig, axes = plt.subplots(1,2, figsize=(8, 4))
    ax = axes[0]
    ax.scatter(gift_dataset[predictor], gift_dataset["sr"], alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Area (m²)')
    ax.set_ylabel('Species Richness')

    # Add regression line
    x_range = np.linspace(np.log(1e5), 
                        np.log(1e12), 100)
    y_pred = np.exp(model.predict(x_range.reshape(-1, 1)))
    ax.plot(np.exp(x_range), y_pred, 'r-', 
        label=f'Linear fit (log-log): R² = {r2:.2f}')
    ax.legend()

    # Plot predicted vs observed SR on second axis
    ax = axes[1]
    ax.scatter(gift_dataset['sr'], gift_dataset['predicted_sr'], alpha=0.7)
    max_val = max(gift_dataset['sr'].max(), gift_dataset['predicted_sr'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
    ax.set_xlabel('Observed Species Richness')
    ax.set_ylabel('Predicted Species Richness')
    
    # Calculate correlation coefficient
    mask = gift_dataset[['sr', 'predicted_sr']].dropna()
    corr = np.corrcoef(mask['sr'], mask['predicted_sr'])[0, 1]
    
    ax.text(0.05, 0.95, f'R² = {r2:.2f}\nD² = {d2:.2f}\nRMSE = {rmse:.2f}\nCorr = {corr:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', fc='w', alpha=0.7))
    ax.legend()
    fig.tight_layout()
    return fig, axes

gift_dataset = load_and_preprocess_data()

predictor = "megaplot_area"
fig, axes = make_regression(gift_dataset, predictor)
fig.suptitle("GIFT regression with megaplot area as predictor")
fig.tight_layout()

predictor = "observed_area"
make_regression(gift_dataset, predictor)

# CONCLUSION: pretty poor fit
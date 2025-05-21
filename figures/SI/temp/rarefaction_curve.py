"""
Building simple rarefaction curves

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
from src.data_processing.utils_env_pred import CHELSADataset
from src.utils import save_to_pickle

import git
import random
from scipy.optimize import curve_fit

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

# range to be investigated
# poly_range = (100, 100, 200e3, 200e3) # in meters

def load_and_preprocess_data(check_consistency=False):
    """
    Load and preprocess EVA data and environmental covariate raster. Returns
    `plot_gdf` (gdf of plots), `species_dict` (dictionary where each key
    corresponds to plot_gdf.index and value associated species list), and
    `climate_raster`.
    """
    logging.info("Loading EVA data...")
    plot_gdf, species_dict = EVADataset().load()
    if check_consistency:
        logging.info("Checking data consistency...")
        assert all([len(np.unique(species_dict[k])) == r.SR for k, r in plot_gdf.iterrows()])

    logging.info("Loading climate raster...")
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)

    logging.info(f"Reprojecting to {CONFIG['crs']}...")
    plot_gdf = plot_gdf.to_crs(CONFIG["crs"])
    climate_dataset = climate_dataset.rio.reproject(CONFIG["crs"]).sortby("y")
    climate_raster = climate_dataset.to_array()
    climate_raster = climate_raster.sel(variable=CONFIG["env_vars"])
    
    return plot_gdf, species_dict, climate_raster



plot_gdf, species_dict, climate_raster = load_and_preprocess_data()
plot_gdf = plot_gdf.set_index("plot_id")


# Plotting rarefaction curve
# We select a fixed megaplot and build a rarefaction curve
area_range = (1e6, 2e6)
box = generate_random_square(plot_gdf, area_range)
plots_within_box = plot_gdf.within(box)
df_box = plot_gdf[plots_within_box]

sr_ar = []
observed_area_ar = []
for i in range(1000):
    x = random.randint(1, len(df_box))
    df_samp = df_box.sample(n=x)

    species = np.concatenate([species_dict[idx] for idx in df_samp.index])
    sr = len(np.unique(species))
    observed_area = np.sum(df_samp['area_m2'])
    
    sr_ar.append(sr)
    observed_area_ar.append(observed_area)
    
fig, ax = plt.subplots()
centroid = box.centroid
area = box.area
ax.set_title(
    (   "Rarefaction curve\n"
        r"Centroid location: $(%.2e, %.2e)$" "\n"
        r"$A_\mathrm{megaplot} = %.2e$"  "\n"
        r"$\frac{\log(\Sigma A_\mathrm{obs})}{\log(A_\mathrm{megaplot})} = %.2f$"
    )
    % (
        centroid.x,
        centroid.y,
        area,
        (np.log(df_box.area_m2.sum()) / np.log(area)),
    )
)
ax.scatter(observed_area_ar, sr_ar)
ax.set_xlabel("Observed area (m2)")
ax.set_ylabel("Species richness")
ax.set_yscale("log")
# ax.set_xscale("log")


# plotting SAR curve
area_range = (1e3, 1e6)
sr_ar = []
megaplot_area_ar = []

for i in tqdm(range(1000)):
    box = generate_random_square(plot_gdf, area_range)
    plots_within_box = plot_gdf.within(box)
    df_samp = plot_gdf[plots_within_box]

    species = np.concatenate([species_dict[idx] for idx in df_samp.index])
    sr = len(np.unique(species))
    observed_area = np.sum(df_samp['area_m2'])
    
    sr_ar.append(sr)
    megaplot_area_ar.append(box.area)
        
fig, ax = plt.subplots()
centroid = box.centroid
area = box.area
ax.set_title("SAR curve")
ax.scatter(megaplot_area_ar, sr_ar)
ax.set_xlabel("Megaplot area (m2)")
ax.set_ylabel("Species richness")
# ax.set_yscale("log")
# ax.set_xscale("log")

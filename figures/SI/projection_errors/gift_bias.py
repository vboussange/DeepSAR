import torch
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from src.plotting import boxplot_bypreds
from src.plotting import CMAP_BR
from tqdm import tqdm

import pandas as pd
from matplotlib.patches import Patch
import geopandas as gpd


import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from src.neural_4pweibull import initialize_ensemble_model
from train import Config, Trainer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from src.cld import create_comp_matrix_allpair_t_test, multcomp_letters
from matplotlib.colors import LinearSegmentedColormap

import scipy.stats as stats


# Load model results
path_results = Path("../../../scripts/results/train_seed_1/checkpoint_MSEfit_large_0b85791.pth")
result_modelling = torch.load(path_results, map_location="cpu")
config = result_modelling["config"]

# Extract model components
predictors = result_modelling["predictors"]
feature_scaler = result_modelling["feature_scaler"]
target_scaler = result_modelling["target_scaler"]

# Initialize model
model = initialize_ensemble_model(result_modelling["ensemble_model_state_dict"], predictors, config, "cpu")

eva_dataset = gpd.read_parquet(config.path_eva_data)
eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])


gift_data_dir = Path("../../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")

# Load GIFT dataset
gift_dataset = gpd.read_parquet(gift_data_dir / "megaplot_data.parquet")
gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
gift_dataset["log_observed_area"] = np.log(gift_dataset["megaplot_area"])
gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
gift_dataset = gift_dataset[gift_dataset.geometry.is_valid]  # Filter valid geometries

gift_dataset["eva_observed_area"] = np.nan


for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within_box = eva_dataset.within(geom)
    df_box = eva_dataset[plots_within_box]
    if not df_box.empty:
        gift_dataset.at[idx, "eva_observed_area"] = df_box["observed_area"].sum()

# Make predictions for GIFT
X_gift = gift_dataset[predictors].copy()
X_gift = torch.tensor(feature_scaler.transform(X_gift), dtype=torch.float32)

with torch.no_grad():
    y_pred_gift = model(X_gift).numpy()
    y_pred_gift = target_scaler.inverse_transform(y_pred_gift)

gift_dataset["predicted_sr"] = y_pred_gift.squeeze()
gift_dataset["bias"] = (gift_dataset["sr"] - gift_dataset["predicted_sr"]) / gift_dataset["sr"]
gift_dataset["sampling_effort"] = np.log(gift_dataset["eva_observed_area"] / gift_dataset["megaplot_area"])

# Remove outliers based on the 95th percentile of bias
bias_threshold = gift_dataset["bias"].quantile(0.95)
plot_data = gift_dataset[gift_dataset["bias"] <= bias_threshold]
bias_threshold = plot_data["bias"].quantile(0.05)
plot_data = plot_data[plot_data["bias"] >= bias_threshold]
# plot_data = gift_dataset.copy()

# Define custom colormap
colors = ["#ff9f1c", "#ffbf69", "#ffffff", "#cbf3f0", "#2ec4b6"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 2]})

sns.regplot(data=plot_data, 
            x="sampling_effort", 
            y="bias", 
            ax=ax2, 
            scatter_kws={'alpha': 0.5}, 
            line_kws={'color': 'red'})
ax2.set_xlabel("Relative sampling effort")
ax2.set_ylabel("Relative bias")
ax2.set_position([0.05, 0.3, 0.3, 0.4])  # 

# Filter out rows with NaN or infinite values
valid_data = plot_data[["sampling_effort", "bias"]].replace([np.inf, -np.inf], np.nan).dropna()

# Calculate R²
r2, p_value = stats.pearsonr(valid_data["sampling_effort"], valid_data["bias"])

ax2.text(0.05, 0.25, f"$\\rho = {r2:.2f}$\n$p$-value$ = {p_value:.2g}$", 
         transform=ax2.transAxes, 
         fontsize=12, 
         verticalalignment='top', 
         bbox=dict(boxstyle="round", 
                   facecolor="white", 
                   alpha=0.5, 
                   edgecolor='none'))

# Plot the map with bias
plot_data.plot(column="bias", 
                  cmap=cmap, 
                  legend=True, 
                  ax=ax1, 
                  edgecolor='black',
                  linewidth=0.1,
                  legend_kwds={'label': "Relative bias", 'shrink': 0.5},
                #   vmax=plot_data["bias"].quantile(0.95),
                #   vmin=plot_data["bias"].quantile(0.05)
                )
ax1.set_axis_off()
# ax1.set_aspect('equal', adjustable='datalim')



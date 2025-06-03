"""
Rapid skorch training for hyperparameter choices.
"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import xarray as xr
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings

from src.neural_4pweibull import Neural4PWeibull, MSELogLoss

from sklearn.metrics import r2_score, d2_absolute_error_score
from scipy.stats import linregress
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.callbacks import LRScheduler

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
    "gift_data_dir": Path(__file__).parent / "../../data/processed/GIFT_CHELSA_compilation/6c2d61d/",
    "path_eva_data": Path(__file__).parent / f"../../data/processed/EVA_CHELSA_compilation/0b85791/",
    # "device": "cuda" if torch.cuda.is_available() else "cpu",
    "device": "cuda",  # Use CPU for simplicity in this example
}

def load_and_preprocess_data():
    logging.info("Loading EVA data...")
    eva_dataset = gpd.read_parquet(CONFIG["path_eva_data"] / "eva_chelsa_megaplot_data.parquet")
    eva_dataset = eva_dataset.dropna()
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    
    logging.info("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(CONFIG["gift_data_dir"] / "megaplot_data.parquet")
    gift_dataset = gift_dataset.set_index("entity_ID") 
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["observed_area"])
    gift_dataset = gift_dataset.dropna()
    gift_dataset = gift_dataset[~np.isinf(gift_dataset.select_dtypes(include=[np.number]).values).any(axis=1)]
    return eva_dataset, gift_dataset

def estimate_sr_neural_weibull(x, y):
    p0 = [1e-1, max(y), min(y), np.median(x)]
    # p0 = [3.584, 2397.9, -70.2, 12.1]
    print(f"Initial parameters p0: {p0}")
    model = NeuralNetRegressor(Neural4PWeibull,
                            module__n_features=x.shape[1]-1,
                            module__layer_sizes=[2**6 for _ in range(5)],
                            module__p0=p0,
                            max_epochs=100,
                            lr=1e-3,
                            # optimizer=torch.optim.LBFGS,
                            optimizer=torch.optim.AdamW,
                            # criterion=MSELogLoss(),
                            criterion=torch.nn.MSELoss(),
                            batch_size=2**8,
                            # train_split=0.2,  # Need validation split for early stopping
                            optimizer__weight_decay=1e-3,
                            verbose=1,
                            device=CONFIG["device"],
                            callbacks=[EarlyStopping(patience=5, threshold=1e-4),
                                     LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau, 
                                               patience=3, factor=0.5, verbose=True)],
                        )

    model.fit(x, y)
    return model

eva_dataset, gift_dataset = load_and_preprocess_data()

eva_dataset = eva_dataset[eva_dataset.num_plots > 200].sample(frac=0.6, random_state=42)

# Plot distributions of key variables for both datasets
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Log Megaplot Area Distribution
axes[0].hist(eva_dataset['log_megaplot_area'], bins=50, alpha=0.7, color='blue', edgecolor='black', label='EVA', density=True)
axes[0].hist(gift_dataset['log_megaplot_area'], bins=50, alpha=0.7, color='red', edgecolor='black', label='GIFT', density=True)
axes[0].set_xlabel('Log Megaplot Area')
axes[0].set_ylabel('Density')
axes[0].legend()

# Log Observed Area Distribution
axes[1].hist(eva_dataset['log_observed_area'], bins=50, alpha=0.7, color='blue', edgecolor='black', label='EVA', density=True)
axes[1].hist(gift_dataset['log_observed_area'], bins=50, alpha=0.7, color='red', edgecolor='black', label='GIFT', density=True)
axes[1].set_xlabel('Log Observed Area')
axes[1].set_ylabel('Density')
axes[1].legend()

# Species Richness Distribution
axes[2].hist(eva_dataset['sr'], bins=50, alpha=0.7, color='blue', edgecolor='black', label='EVA', density=True)
axes[2].hist(gift_dataset['sr'], bins=50, alpha=0.7, color='red', edgecolor='black', label='GIFT', density=True)
axes[2].set_xlabel('Species Richness')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.tight_layout()
plt.show()


# Initialize scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

predictors = ["log_observed_area", "log_megaplot_area"] + CONFIG["env_vars"] + ["std_" + var for var in CONFIG["env_vars"]]
X = eva_dataset[predictors].values.astype("float32")
X_scaled = feature_scaler.fit_transform(X).astype("float32")

y = eva_dataset["sr"].values.astype("float32").reshape(-1, 1)
y_scaled = target_scaler.fit_transform(y).astype("float32")

model = estimate_sr_neural_weibull(X_scaled, y_scaled)


# Make predictions on the EVA test set
eva_sr_predicted_scaled = model.predict(X_scaled).reshape(-1, 1)
eva_sr_predicted = target_scaler.inverse_transform(eva_sr_predicted_scaled).squeeze()

# Plot EVA predictions vs true values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: EVA observed SR vs Predicted SR
x_eva = eva_dataset["sr"].values.astype("float32")
y_eva = eva_sr_predicted
ax1.scatter(x_eva, y_eva, alpha=0.7, color='blue')
max_val = np.nanmax([x_eva.max(), y_eva.max()])
ax1.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax1.set_xlabel("EVA observed SR")
ax1.set_ylabel("Predicted SR")
ax1.set_xlim(x_eva.min(), x_eva.max())
ax1.set_ylim(x_eva.min(), x_eva.max())

# Compute metrics for EVA
r2_eva = r2_score(x_eva, y_eva)
d2_eva = d2_absolute_error_score(x_eva, y_eva)
mse_eva = np.sqrt(mean_squared_error(x_eva, y_eva))
corr_eva = np.corrcoef(x_eva, y_eva)[0, 1]
ax1.text(
    0.05, 0.95, f"R2={r2_eva:.2f}\nD2={d2_eva:.2f}\nRMSE={mse_eva:.2f}\nCorr={corr_eva:.2f}",
    transform=ax1.transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
ax1.legend()

# Second subplot: Residuals against area for EVA
residuals_eva = (x_eva - y_eva) / y_eva
areas_eva = eva_dataset["log_megaplot_area"].values.astype("float32")

ax2.scatter(areas_eva, residuals_eva, alpha=0.7, color='blue')
ax2.axhline(y=0, color='r', linestyle='--', label='Zero residual line')
ax2.set_xlabel("Log megaplot area")
ax2.set_ylabel("Relative residuals (Observed - Predicted) / Predicted")
ax2.legend()

plt.tight_layout()
plt.show()


# TESTING ON GIFTS
X_gift = gift_dataset[predictors].values.astype("float32")
X_gift_scaled = feature_scaler.transform(X_gift)

# Make predictions and inverse transform
with torch.no_grad():
    X_gift_tensor = torch.tensor(X_gift_scaled[:,1:]).to(CONFIG["device"])
    gift_sr_predicted_scaled = model.module_.predict_sr(X_gift_tensor).cpu().numpy().reshape(-1, 1)
    gift_sr_predicted = target_scaler.inverse_transform(gift_sr_predicted_scaled).squeeze()

fig, (ax1, ax2) = plt.subplots(1, 2)

# First subplot: GIFT observed SR vs Predicted SR
x = gift_dataset["sr"].values.astype("float32")
y = gift_sr_predicted
ax1.scatter(x, y, alpha=0.7, cmap='magma_r')
max_val = np.nanmax([x.max(), y.max()])
ax1.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax1.set_xlabel("GIFT observed SR")
ax1.set_ylabel("Predicted SR")
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(x.min(), x.max())

# Compute R2, D2, and MSE for Chao1
r2_0 = r2_score(x, y)
d2_0 = d2_absolute_error_score(x, y)
mse_0 = np.sqrt(mean_squared_error(x, y))
corr_0 = np.corrcoef(x, y)[0, 1]
ax1.text(
    0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nRMSE={mse_0:.2f}\nCorr={corr_0:.2f}",
    transform=ax1.transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
)
ax1.legend()

# Second subplot: Residuals against area
residuals = (x - y)/y
areas = gift_dataset["log_megaplot_area"].values.astype("float32")

ax2.scatter(areas, residuals, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', label='Zero residual line')
ax2.set_xlabel("Log megaplot area")
ax2.set_ylabel("Relative residuals (Observed - Predicted)/Predicted")
ax2.set_title("Residuals vs Log Observed Area")
ax2.legend()
ax2.set_yscale('symlog')


plt.tight_layout()
plt.show()
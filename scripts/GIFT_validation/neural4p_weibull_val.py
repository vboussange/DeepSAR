"""
Evaluating Neural4PWeibull trained against the GIFT dataset.
"""
import logging
from pathlib import Path
from src.neural_4pweibull import initialize_ensemble_model
from src.data_processing.utils_eva import EVADataset
import torch
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, d2_absolute_error_score

import sys
sys.path.append(str(Path(__file__).parent / Path("../")))
from train import Config


gift_data_dir = Path(__file__).parent / "../../data/processed/GIFT_CHELSA_compilation/6c2d61d/"
path_results = Path(f"../results/train_seed_1/checkpoint_MSEfit_large_0b85791.pth")    

def load_and_preprocess_data():
    print("Loading model")
    result_modelling = torch.load(path_results, map_location="cpu")

    print("Loading EVA data...")
    eva_dataset, eva_species_dict = EVADataset().load()
    eva_dataset = eva_dataset.set_index("plot_id")
    eva_dataset = eva_dataset.to_crs("EPSG:3035")
    
    print("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(gift_data_dir/ "megaplot_data.parquet")
    
    
    return (eva_dataset, eva_species_dict), gift_dataset, result_modelling


(eva_dataset, eva_species_dict), gift_dataset, result_modelling = load_and_preprocess_data()

config = result_modelling["config"]
predictors = result_modelling["predictors"]
feature_scaler = result_modelling["feature_scaler"]
target_scaler = result_modelling["target_scaler"]
model = initialize_ensemble_model(result_modelling["ensemble_model_state_dict"], predictors, config, "cpu")

gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
gift_dataset["log_observed_area"] = np.log(gift_dataset["megaplot_area"]) # not used
gift_dataset = gift_dataset.dropna()
gift_dataset = gift_dataset.replace([np.inf, -np.inf], np.nan).dropna()

X = gift_dataset[predictors].copy()
X = torch.tensor(feature_scaler.transform(X), dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X).numpy()
    y_pred = target_scaler.inverse_transform(y_pred)
    
gift_dataset["predicted_sr"] = y_pred.squeeze()



fig, ax = plt.subplots()
mask0 = gift_dataset[["sr", "predicted_sr"]].dropna()
x = np.log(mask0["sr"])
y = np.log(mask0["predicted_sr"])

ax.scatter(x, y, alpha=0.7)
max_val = np.nanmax([x.max(), y.max()])
min_val = np.nanmin([x.min(), y.min()])
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
ax.set_xlabel("Log GIFT observed SR")
ax.set_ylabel("Log Predicted SR")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(x.min(), x.max())

# Compute R2, D2, and MSE for log values
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


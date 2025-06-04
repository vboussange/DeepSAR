"""
Total species richness of GIFT polygons estimated using Chao2 estimation (chao2 on incidence
data).
"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import logging

from tqdm import tqdm

from src.data_processing.utils_eva import EVADataset

from sklearn.metrics import (d2_absolute_error_score, root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
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
    "gift_data_dir": Path(__file__).parent / "../data/processed/GIFT_CHELSA_compilation/6c2d61d",
    "run_folder": Path(Path(__file__).parent, 'results', "benchmark"),
    "run_name": "chao2_estimator_benchmark",
    "seed": 42,  # For reproducibility
}

# Define covariate feature names based on environmental covariates
mean_labels = CONFIG["env_vars"]
std_labels = [f"std_{var}" for var in CONFIG["env_vars"]]
CLIMATE_COL_NAMES = np.hstack((mean_labels, std_labels)).tolist()
CONFIG["run_folder"].mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data():
    logging.info("Loading EVA data...")
    eva_dataset, eva_species_dict = EVADataset().load()
    eva_dataset = eva_dataset.set_index("plot_id")
    eva_dataset = eva_dataset.to_crs(CONFIG["crs"])
    
    logging.info("Loading GIFT data...")
    gift_dataset = gpd.read_parquet(CONFIG["gift_data_dir"] / "megaplot_data.parquet")
    gift_dataset = gift_dataset.set_index("entity_ID") 
    gift_dataset = gift_dataset.to_crs(CONFIG["crs"])
    
    return eva_dataset, eva_species_dict, gift_dataset

def estimate_sr(plots, species_dict):
    species = np.concatenate([species_dict[idx] for idx in plots.index])
    species_counts = pd.Series(species).value_counts()
    
    # chao2 estimator calculation
    f1 = (species_counts == 1).sum()  # number of singletons
    f2 = (species_counts == 2).sum()  # number of doubletons
    S_obs = len(species_counts)       # observed species richness

    if f2 == 0:
        chao2 = S_obs + (f1 * (f1 - 1)) / 2 / (f2 + 1)  # bias-corrected if no doubletons
        var_chao2 = (f1 * (f1 - 1)) / 2 + ((f1 * (2 * f1 - 1) ** 2) / 4) - f1**4 / (4 * chao2)
    else:
        chao2 = S_obs + (f1 ** 2) / (2 * f2)
        var_chao2 = f2 * ((f1 / f2) ** 4) / 4 + ((f1 ** 3) / (2 * f2 ** 2))    
    
    return S_obs, chao2, var_chao2


eva_dataset, eva_species_dict, gift_dataset = load_and_preprocess_data()

gift_dataset = gift_dataset[gift_dataset.is_valid]
gift_dataset = gift_dataset[~gift_dataset.is_empty]
gift_dataset["chao2"] = np.nan
gift_dataset["var_chao2"] = np.nan
gift_dataset["eva_observed_sr"] = np.nan
gift_dataset["eva_observed_area"] = np.nan

for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
    geom = row.geometry
    plots_within_box = eva_dataset.within(geom)
    df_box = eva_dataset[plots_within_box]
    if not df_box.empty:
        eva_observed_sr, chao2, var_chao2 = estimate_sr(df_box, eva_species_dict)
        gift_dataset.at[idx, "chao2"] = chao2
        gift_dataset.at[idx, "var_chao2"] = var_chao2
        gift_dataset.at[idx, "eva_observed_sr"] = eva_observed_sr
        gift_dataset.at[idx, "eva_observed_area"] = df_box["area_m2"].sum()



        print(
            f"Index: {idx} | chao2: {chao2:.2f} | Var(chao2): {var_chao2:.2f} | "
            f"EVA observed SR: {eva_observed_sr} | GIFT observed SR: {row.sr}"
        )
    
y_true = gift_dataset["sr"].values
y_pred = gift_dataset["chao2"].values
y_var_pred = gift_dataset["var_chao2"].values

# Generate multiple replicates based on prediction uncertainty
np.random.seed(CONFIG["seed"])  # For reproducibility
n_replicates = 5
metrics_replicates = []

# Create mask for valid predictions
valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_var_pred))
y_true_valid = y_true[valid_mask]
y_pred_valid = y_pred[valid_mask]
y_std_valid = np.sqrt(y_var_pred[valid_mask])

for i in range(n_replicates):
    # Generate noisy predictions based on uncertainty
    y_pred_replicate = np.random.normal(y_pred_valid, y_std_valid)
    # Ensure predictions are non-negative
    y_pred_replicate = np.maximum(y_pred_replicate, 0)
    
    # Calculate metrics for this replicate
    r2_rep = r2_score(y_true_valid, y_pred_replicate)
    d2_rep = d2_absolute_error_score(y_true_valid, y_pred_replicate)
    rmse_rep = root_mean_squared_error(y_true_valid, y_pred_replicate)
    mape_rep = mean_absolute_percentage_error(y_true_valid, y_pred_replicate)
    
    metrics_replicates.append({
        'run': i,
        'r2_gift': r2_rep,
        'd2_gift': d2_rep,
        'rmse_gift': rmse_rep,
        'mape_gift': mape_rep
    })

# Convert to DataFrame for analysis
metrics_df = pd.DataFrame(metrics_replicates)
metrics_df["model"] = "chao2_estimator"
metrics_df.to_csv(CONFIG["run_folder"] / f'{CONFIG["run_name"]}.csv', index=False)

# Calculate summary statistics
print("Metrics across replicates:")
print(f"R2: {metrics_df['r2_gift'].mean():.3f} ± {metrics_df['r2'].std():.3f}")
print(f"D2: {metrics_df['d2_gift'].mean():.3f} ± {metrics_df['d2'].std():.3f}")
print(f"RMSE: {metrics_df['rmse_gift'].mean():.3f} ± {metrics_df['rmse'].std():.3f}")
print(f"MAPE: {metrics_df['mape_gift'].mean():.3f} ± {metrics_df['mape'].std():.3f}")
    
# output_path = Path(__file__).parent / f"{Path(__file__).stem}.parquet"
# gift_dataset.to_parquet(output_path)
# logging.info(f"Results saved to {output_path}")

# gift_dataset["coverage"] = np.log(gift_dataset["eva_observed_area"] / gift_dataset["megaplot_area"])

# fig, ax = plt.subplots()
# gift_dataset = gift_dataset[gift_dataset["chao2"] > 0]
# gift_dataset = gift_dataset[gift_dataset["chao2"] < 1e4]
# gift_dataset["log_chao2"] = np.log(gift_dataset["chao2"])
# gift_dataset["log_sr"] = np.log(gift_dataset["sr"])

# mask0 = gift_dataset[["log_chao2", "log_sr"]].dropna()
# x = mask0["log_sr"]
# y = mask0["log_chao2"]

# # Scatter plot: GIFT observed SR vs Chao1
# scatter = ax.scatter(x, y, c=np.clip(gift_dataset["var_chao2"], None, 1000), 
#                      alpha=0.7, cmap='magma_r')
# cbar = plt.colorbar(scatter, ax=ax, label='var_chao2')
# max_val = np.nanmax([x.max(), y.max()])
# ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
# ax.set_xlabel("GIFT observed SR")
# ax.set_ylabel("Predicted SR")
# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(x.min(), x.max())

# # Compute R2, D2, and MSE for Chao1
# r2_0 = r2_score(x, y)
# d2_0 = d2_absolute_error_score(x, y)
# mse_0 = np.sqrt(mean_squared_error(x, y))
# corr_0 = np.corrcoef(x, y)[0, 1]
# ax.text(
#     0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nRMSE={mse_0:.2f}\nCorr={corr_0:.2f}",
#     transform=ax.transAxes,
#     verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
# )
# ax.legend()
# ax.set_title("GIFT vs predicted species richness from Chao2")

# # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# # # Scatter plot: GIFT observed SR vs chao2
# # scatter = axes[0].scatter(gift_dataset["sr"], gift_dataset["chao2"], c=gift_dataset["coverage"], 
# #                          alpha=0.7, cmap='magma_r')
# # plt.colorbar(scatter, ax=axes[0], label='Log coverage ratio')
# # max_val = np.nanmax([gift_dataset["sr"].max(), gift_dataset["chao2"].max()])
# # axes[0].plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
# # axes[0].set_xlabel("GIFT observed SR")
# # axes[0].set_ylabel("chao2 estimated SR")

# # # Compute R2, D2, MSE and correlation for chao2
# # mask0 = gift_dataset[["sr", "chao2"]].dropna()
# # r2_0 = r2_score(mask0["sr"], mask0["chao2"])
# # d2_0 = d2_absolute_error_score(mask0["sr"], mask0["chao2"])
# # mse_0 = np.sqrt(mean_squared_error(mask0["sr"], mask0["chao2"]))
# # corr_0 = mask0["sr"].corr(mask0["chao2"])
# # axes[0].text(
# #     0.05, 0.95, f"R2={r2_0:.2f}\nD2={d2_0:.2f}\nRMSE={mse_0:.2f}\nCorr={corr_0:.2f}",
# #     transform=axes[0].transAxes,
# #     verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
# # )
# # axes[0].legend()

# # # Scatter plot: GIFT observed SR vs EVA observed SR
# # axes[1].scatter(gift_dataset["sr"], gift_dataset["eva_observed_sr"], alpha=0.7)
# # max_val = np.nanmax([gift_dataset["sr"].max(), gift_dataset["eva_observed_sr"].max()])
# # axes[1].plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
# # axes[1].set_xlabel("GIFT observed SR")
# # axes[1].set_ylabel("EVA observed SR")

# # # Compute R2, D2, MSE and correlation for EVA observed SR
# # mask1 = gift_dataset[["sr", "eva_observed_sr"]].dropna()
# # r2_1 = r2_score(mask1["sr"], mask1["eva_observed_sr"])
# # d2_1 = d2_absolute_error_score(mask1["sr"], mask1["eva_observed_sr"])
# # mse_1 = np.sqrt(mean_squared_error(mask1["sr"], mask1["eva_observed_sr"]))
# # corr_1 = mask1["sr"].corr(mask1["eva_observed_sr"])
# # axes[1].text(
# #     0.05, 0.95, f"R2={r2_1:.2f}\nD2={d2_1:.2f}\nRMSE={mse_1:.2f}\nCorr={corr_1:.2f}",
# #     transform=axes[1].transAxes,
# #     verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
# # )
# # axes[1].legend()
# # fig.suptitle("Chao2 and EVA observed SR predictions")
# # plt.tight_layout()
# # plt.show()
   
   
# # Calculate z-value for chao2 estimator
# gift_dataset['z_value'] = (gift_dataset['chao2'] - gift_dataset['sr']) / np.sqrt(gift_dataset['var_chao2'])
# gift_dataset = gift_dataset[~gift_dataset['z_value'].isna() & ~np.isinf(gift_dataset['z_value'])]
# # Create a new figure for z-values
# fig_z, ax_z = plt.subplots(figsize=(10, 6))

# # Plot histogram of z-values
# ax_z.hist(gift_dataset['z_value'].dropna(), bins=30, alpha=0.7, color='steelblue')
# ax_z.axvline(x=1.96, color='r', linestyle='--', label='z=1.96 (95% CI)')
# ax_z.axvline(x=-1.96, color='r', linestyle='--')
# ax_z.axvline(x=0, color='k', linestyle='-', label='z=0')

# # Add labels and title
# ax_z.set_xlabel('z-value')
# ax_z.set_ylabel('Frequency')
# ax_z.set_title('Distribution of z-values for chao2 Estimator')
# ax_z.legend()

# # Calculate the percentage of values beyond ±1.96
# beyond_ci = (abs(gift_dataset['z_value']) > 1.96).mean() * 100
# ax_z.text(0.05, 0.95, f"{beyond_ci:.1f}% of values outside 95% CI", 
#           transform=ax_z.transAxes, verticalalignment='top',
#           bbox=dict(boxstyle="round", fc="w", alpha=0.7))

# plt.tight_layout()
# plt.show()
    
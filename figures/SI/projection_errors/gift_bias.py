import torch
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
from src.neural_4pweibull import initialize_ensemble_model
from src.plotting import CMAP_GO

import sys
sys.path.append(str(Path(__file__).parent / "../../../scripts/"))
from train import Config, Trainer

def load_data(config_path, gift_data_dir):
    """Load and preprocess evaluation and GIFT datasets."""
    eva_dataset = gpd.read_parquet(config_path)
    eva_dataset["log_megaplot_area"] = np.log(eva_dataset["megaplot_area"])
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])

    gift_dataset = gpd.read_parquet(gift_data_dir / "megaplot_data.parquet")
    gift_dataset["log_megaplot_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["megaplot_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    gift_dataset = gift_dataset[gift_dataset.geometry.is_valid]
    gift_dataset["eva_observed_area"] = np.nan

    return eva_dataset, gift_dataset

def calculate_observed_area(eva_dataset, gift_dataset):
    """Calculate observed area for GIFT dataset."""
    for idx, row in tqdm(gift_dataset.iterrows(), total=gift_dataset.shape[0]):
        geom = row.geometry
        plots_within_box = eva_dataset.within(geom)
        df_box = eva_dataset[plots_within_box]
        if not df_box.empty:
            gift_dataset.at[idx, "eva_observed_area"] = df_box["observed_area"].sum()
    return gift_dataset

def make_predictions(model, feature_scaler, target_scaler, gift_dataset, predictors):
    """Make predictions for GIFT dataset."""
    X_gift = gift_dataset[predictors].copy()
    X_gift = torch.tensor(feature_scaler.transform(X_gift), dtype=torch.float32)
    with torch.no_grad():
        y_pred_gift = model(X_gift).numpy()
        y_pred_gift = target_scaler.inverse_transform(y_pred_gift)
    gift_dataset["predicted_sr"] = y_pred_gift.squeeze()
    gift_dataset["bias"] = (gift_dataset["sr"] - gift_dataset["predicted_sr"]) / gift_dataset["sr"]
    gift_dataset["sampling_effort"] = np.log(gift_dataset["eva_observed_area"] / gift_dataset["megaplot_area"])
    return gift_dataset

    

if __name__ == "__main__":
   
    # Define the path to save the processed dataset
    processed_data_path = Path("processed_gift_dataset.parquet")

    # Check if the processed dataset exists
    if processed_data_path.exists():
        # Load the processed dataset
        gift_dataset = gpd.read_parquet(processed_data_path)
    else:
        path_results = Path("../../../scripts/results/train_seed_1/checkpoint_MSEfit_large_0b85791.pth")
        gift_data_dir = Path("../../../data/processed/GIFT_CHELSA_compilation/6c2d61d/")

        # Load model and data
        result_modelling = torch.load(path_results, map_location="cpu")
        config = result_modelling["config"]
        predictors = result_modelling["predictors"]
        feature_scaler = result_modelling["feature_scaler"]
        target_scaler = result_modelling["target_scaler"]
        model = initialize_ensemble_model(result_modelling["ensemble_model_state_dict"], predictors, config, "cpu")
        eva_dataset, gift_dataset = load_data(config.path_eva_data, gift_data_dir)

        # Process data
        gift_dataset = calculate_observed_area(eva_dataset, gift_dataset)
        gift_dataset.to_parquet(processed_data_path)

    # Filter outliers
    lower_threshold = gift_dataset["bias"].quantile(0.05)
    upper_threshold = gift_dataset["bias"].quantile(0.95)
    plot_data = gift_dataset[(gift_dataset["bias"] >= lower_threshold) & (gift_dataset["bias"] <= upper_threshold)]

    # Generate plots
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 2]})

    # Regression plot
    sns.regplot(data=plot_data, x="sampling_effort", y="bias", ax=ax2, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    ax2.set_xlabel("Relative sampling effort")
    ax2.set_ylabel("Relative bias")
    ax2.set_position([0.05, 0.3, 0.3, 0.4])

    # Calculate and display R²
    valid_data = plot_data[["sampling_effort", "bias"]].replace([np.inf, -np.inf], np.nan).dropna()
    r2, p_value = pearsonr(valid_data["sampling_effort"], valid_data["bias"])
    ax2.text(0.05, 0.25, f"$\\rho = {r2:.2f}$\n$p$-value$ = {p_value:.2g}$", 
             transform=ax2.transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor='none'))

    # Map plot
    plot_data.plot(column="bias", cmap=CMAP_GO, legend=True, ax=ax1, edgecolor='black', linewidth=0.1,
                   legend_kwds={'label': "Relative bias", 'shrink': 0.5})
    ax1.set_axis_off()
    # Annotate panels
    ax2.text(-0.15, 1.05, "a", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")
    ax1.text(-0.1, 1.05, "b", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")

    fig.savefig("figure_gift_bias.pdf", dpi=300, bbox_inches="tight")
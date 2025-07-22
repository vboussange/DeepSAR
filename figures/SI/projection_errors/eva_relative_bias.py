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
from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.plotting import CMAP_GO
from deepsar.ensemble_trainer import EnsembleConfig

from matplotlib.colors import TwoSlopeNorm

def load_data(config_path):
    """Load and preprocess evaluation and GIFT datasets."""
    eva_dataset = gpd.read_parquet(config_path)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["megaplot_area"]) #TODO: change, legacy name
    eva_dataset["log_observed_area"] = np.log(eva_dataset["observed_area"])

    return eva_dataset

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
    gift_dataset["bias"] = (gift_dataset["predicted_sr"] - gift_dataset["sr"]) / gift_dataset["sr"]
    gift_dataset["sampling_effort"] = gift_dataset["eva_observed_area"] / gift_dataset["sp_unit_area"]
    return gift_dataset

    
if __name__ == "__main__":
   
    # Define the path to save the processed dataset
    processed_data_path = Path("processed_gift_dataset.parquet")
    MODEL_NAME = "deep4pweibull_basearch6_0b85791"

    path_results = Path(f"../../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")

    # Load model and data
    checkpoint = torch.load(path_results, map_location="cpu", weights_only=False)
    model = Deep4PWeibull.initialize_ensemble(checkpoint, "cpu")
    eva_dataset = load_data(checkpoint["config"].path_eva_data)
    
    
    y_pred = model.predict_mean_sr(eva_dataset)
    y_true = eva_dataset["sr"].values
    
    
    bias = (y_pred - y_true) / y_true
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eva_dataset["log_sp_unit_area"], bias, alpha=0.2, label="Individual plots")
    ax.set_ylim(-100, 100)
    
    # Compute mean bias in bins of log_sp_unit_area
    bins = np.linspace(eva_dataset["log_sp_unit_area"].min(), eva_dataset["log_sp_unit_area"].max(), 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_bias = [
        np.nanmean(bias[(eva_dataset["log_sp_unit_area"] >= bins[i]) & (eva_dataset["log_sp_unit_area"] < bins[i+1])])
        for i in range(len(bins)-1)
    ]
    ax.plot(bin_centers, mean_bias, color="red", lw=2, label="Mean bias")
    ax.legend()
    # ax.set_yscale('symlog')
    

    # Process data
    gift_dataset = calculate_observed_area(eva_dataset, gift_dataset)
    gift_dataset = make_predictions(model, feature_scaler, target_scaler, gift_dataset, predictors)
    gift_dataset.to_parquet(processed_data_path)

    # Filter outliers
    lower_threshold = gift_dataset["bias"].quantile(0.05)
    upper_threshold = gift_dataset["bias"].quantile(0.95)
    plot_data = gift_dataset[(gift_dataset["bias"] >= lower_threshold) & (gift_dataset["bias"] <= upper_threshold)]
    plot_data["log_sampling_effort"] = np.log(plot_data["sampling_effort"])
    
    # Generate plots
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 2]})

    # Regression plot
    sns.regplot(data=plot_data, x="log_sp_unit_area", y="bias", ax=ax2, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    ax2.set_xlabel("Relative sampling effort")
    ax2.set_ylabel("Relative bias")
    ax2.set_position([0.05, 0.3, 0.3, 0.4])

    # Calculate and display R²
    valid_data = plot_data[["log_sampling_effort", "bias"]].replace([np.inf, -np.inf], np.nan).dropna()
    r2, p_value = pearsonr(valid_data["log_sampling_effort"], valid_data["bias"])
    ax2.text(0.05, 0.25, f"$\\rho = {r2:.2f}$\n$p$-value$ = {p_value:.2g}$", 
             transform=ax2.transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor='none'))

    # Map plot
    # Normalize the colormap so that 0 maps to the center (white)

    vmin = plot_data["bias"].min()
    vmax = plot_data["bias"].max()
    vcenter = 0.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plot_data.plot(
        column="bias",
        cmap=CMAP_GO,
        legend=True,
        ax=ax1,
        edgecolor='black',
        linewidth=0.1,
        norm=norm,
        legend_kwds={'label': "Relative bias", 'shrink': 0.5}
    )
    ax1.set_axis_off()
    # Annotate panels
    ax2.text(-0.15, 1.05, "a", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")
    ax1.text(-0.1, 0.85, "b", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")

    fig.savefig("figure_gift_bias.pdf", dpi=300, bbox_inches="tight")
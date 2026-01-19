"""Plotting species richness against area and coverage for EVA and GIFT datasets."""

from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from deepsar.plotting import CMAP_BR

SBCV_PATH = Path(__file__).parents[3] / "data" / "processed" / "training_samples" / "sbcv" / "606e055"
GIFT_PATH = Path(__file__).parents[3] / "data" / "processed" / "test_samples_GIFT" / "606e055" / "compiled_data.parquet"

FOLD_ID = 0

def load_data():
    root = Path(__file__).parents[3]

    # Load train dataset from SBCV fold
    train_path = SBCV_PATH / f"fold_{FOLD_ID}_train.parquet"
    train_dataset = gpd.read_parquet(train_path)
    train_dataset["log_sp_unit_area"] = np.log(train_dataset["sp_unit_area"])
    train_dataset["log_observed_area"] = np.log(train_dataset["observed_area"])
    train_dataset["coverage"] = train_dataset["log_observed_area"] / train_dataset["log_sp_unit_area"]
    train_dataset = train_dataset.sample(n=5000, random_state=42)  # Sample 1000 points for visualization
    
    # Load GIFT dataset
    gift_dataset = gpd.read_parquet(GIFT_PATH)
    gift_dataset["log_sp_unit_area"] = np.log(gift_dataset["sp_unit_area"])
    gift_dataset["log_observed_area"] = np.log(gift_dataset["observed_area"])
    gift_dataset = gift_dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate log-transformed species richness (log_sr) for both datasets
    train_dataset["log_sr"] = np.log(train_dataset["sr"])
    gift_dataset["log_sr"] = np.log(gift_dataset["sr"])
    
    return train_dataset, gift_dataset


if __name__ == "__main__":
    train_dataset, gift_dataset = load_data()
    colors = ["#f72585","#4cc9f0"]
    
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot train dataset
    scatter = ax.scatter(
        np.exp(train_dataset["log_sp_unit_area"]) / 1e6,
        np.exp(train_dataset["log_sr"]),
        c=train_dataset["coverage"],
        cmap=CMAP_BR,
        alpha=0.6,
        vmax = 1,
        label="Train Dataset",
        s=10
    )
    # Add a colorbar to indicate log_observed_area
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Sampling effort")

    # Plot GIFT dataset
    ax.scatter(
        np.exp(gift_dataset["log_sp_unit_area"]) / 1e6,
        np.exp(gift_dataset["log_sr"]),
        color=colors[0],
        alpha=1,
        label="GIFT Dataset",
        s=20,
        marker="x",  # Use a different marker
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Area (km²)")
    ax.set_ylabel("Species richness")
    ax.legend()
    fig.savefig("figure_datasets.pdf", dpi=300, bbox_inches="tight")
    

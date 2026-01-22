"""Plotting species richness against area and coverage for EVA and GIFT datasets."""

from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd

from deepsar.plotting import CMAP_BR

SBCV_PATH = Path(__file__).parents[3] / "data" / "processed" / "training_samples" / "sbcv" / "d5eb0a5"
GIFT_PATH = Path(__file__).parents[3] / "data" / "processed" / "test_samples_GIFT" / "1cb3898" / "compiled_data.parquet"

FOLD_ID = 0

def load_data():

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
    gift_dataset["coverage"] = 1.0
    
    # Calculate log-transformed species richness (log_sr) for both datasets
    train_dataset["log_sr"] = np.log(train_dataset["sr"])
    gift_dataset["log_sr"] = np.log(gift_dataset["sr"])
    
    return train_dataset, gift_dataset


def export_sampling_effort_stats(train_dataset, gift_dataset, output_path: Path) -> None:
    def summarize(series: np.ndarray) -> dict:
        values = np.asarray(series)
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    stats = {
        "train": summarize(train_dataset["coverage"].dropna().values),
        "gift": summarize(gift_dataset["coverage"].dropna().values),
    }
    pd.DataFrame(stats).T.to_csv(output_path, index_label="dataset")


if __name__ == "__main__":
    train_dataset, gift_dataset = load_data()
    colors = ["#f72585","#4cc9f0"]

    export_sampling_effort_stats(
        train_dataset,
        gift_dataset,
        Path(__file__).with_name("sampling_effort_stats.csv"),
    )
    
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot train dataset
    coverage = train_dataset["coverage"].values
    norm = plt.Normalize(vmin=0, vmax=1)
    scatter = ax.scatter(
        np.exp(train_dataset["log_sp_unit_area"]) / 1e6,
        np.exp(train_dataset["log_sr"]),
        c=coverage,
        cmap=CMAP_BR,
        alpha=0.6,
        norm=norm,
        label="Train Dataset",
        s=10
    )
    # Add a colorbar to indicate log_observed_area
    mappable = plt.cm.ScalarMappable(cmap=CMAP_BR, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
    cbar.set_label("Relative sampling effort")

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
    

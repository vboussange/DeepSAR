import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import geopandas as gpd
from deepsar.deep4pweibull import Deep4PWeibull
from deepsar.plotting import CMAP_GO


def load_data(config_path):
    """Load and preprocess evaluation and GIFT datasets."""
    eva_dataset = gpd.read_parquet(config_path)
    eva_dataset["log_sp_unit_area"] = np.log(eva_dataset["sp_unit_area"]) #TODO: change, legacy name
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
    colors = ["#4cc9f0", "#f72585"]

    # Define the path to save the processed dataset
    processed_data_path = Path("processed_gift_dataset.parquet")
    MODEL_NAME = "deep4pweibull_basearch6_0b85791"

    path_results = Path(f"../../../scripts/results/train/checkpoint_{MODEL_NAME}.pth")

    # Load model and data
    checkpoint = torch.load(path_results, map_location="cpu", weights_only=False)
    model = Deep4PWeibull.initialize_ensemble(checkpoint, "cpu")
    eva_dataset = load_data(checkpoint["config"].path_eva_data)
    eva_test_dataset = eva_dataset[eva_dataset["test"] == True]
    
    y_pred = model.predict_mean_sr(eva_test_dataset)
    y_true = eva_test_dataset["sr"].values

    bias = (y_pred - y_true) / y_true
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Compute mean bias in bins of log_sp_unit_area
    bins = np.linspace(eva_test_dataset["log_sp_unit_area"].min(), eva_test_dataset["log_sp_unit_area"].max(), 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_bias = [
        np.nanmedian(bias[(eva_test_dataset["log_sp_unit_area"] >= bins[i]) & (eva_test_dataset["log_sp_unit_area"] < bins[i+1])])
        for i in range(len(bins)-1)
    ]
    
    ax.scatter(np.exp(eva_test_dataset["log_sp_unit_area"]), 
               bias,
               alpha=0.2, 
               color=colors[0])

    ax.plot(np.exp(bin_centers), mean_bias, color=colors[1], lw=2, label="Median bias")

    ax.legend()
    ax.set_ylim(-2, 2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Area (m²)")
    ax.set_xscale("log")
    ax.set_ylabel("Relative bias")

    fig.savefig(Path(__file__).stem + ".pdf", dpi=300, bbox_inches="tight")

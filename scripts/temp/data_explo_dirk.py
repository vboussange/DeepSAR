"""
Projecting spatially MLP, 
where 
- heterogeneity value if interpolated to have more sensible predictions
- dlogSR/dlogA is calculated by accounting for the change in heterogeneity with the change in area

The script is not fully checked.

Using ensemble methods.
"""
import torch
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.utils import save_to_pickle
from scripts.train import Config
from src.ensemble_model import initialize_ensemble_model
import figures.figure_5.calculate_sar as calculate_sar
from pathlib import Path
import pandas as pd
from src.mlp import scale_feature_tensor, inverse_transform_scale_feature_tensor, get_gradient
from eva_chelsa_processing.preprocess_eva_chelsa_megaplots import load_preprocessed_data
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == "__main__":
    seed = 1
    MODEL = "large"
    HASH = "71f9fc7"


    # predicting for plots in `gift_richness_climate_reduced`
    # csv_path = Path("gift_richness_climate_reduced.csv")
    # data = pd.read_csv(csv_path, sep=";")
    
    checkpoint_path = Path(f"results/train_dSRdA_weight_1e+00_seed_{seed}/checkpoint_{MODEL}_model_full_physics_informed_constraint_{HASH}.pth")    
    results_fit_split_all = torch.load(checkpoint_path, map_location="cpu")    
    results_fit_split = results_fit_split_all["all"]
    model = initialize_ensemble_model(results_fit_split, results_fit_split_all["config"], "cuda")

    predictors = results_fit_split["predictors"]
    feature_scaler = results_fit_split["feature_scaler"]
    target_scaler = results_fit_split["target_scaler"]
    
    # data["log_area"] = np.log(data["area"] * 1e6) # m2 to km2
    # features = torch.tensor(data[predictors].values.astype(np.float32), device=next(model.parameters()).device)
    # X = scale_feature_tensor(features, feature_scaler)
    # y = model(X)
    # log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
    # predicted_sr = np.exp(log_SR.detach().cpu().numpy())
    # data["predicted_sr"] = predicted_sr
    # output_path = Path("predicted_species_richness.csv")
    # data.to_csv(output_path, index=False)
    
    
    # -------------------------------------
    # calculating predictions against original plot data using batches
    data = pd.read_csv(Path(__file__).parent / "../data/processed/EVA_CHELSA_raw/raw_plot_data.csv")
    data["log_area"] = np.log(data["area"]) # m2 to km2

    batch_size = 1024  # Adjust the batch size according to your GPU memory
    predicted_sr = []

    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch_data = data.iloc[start:end]
        features = torch.tensor(batch_data[predictors].values.astype(np.float32), device=next(model.parameters()).device)
        X = scale_feature_tensor(features, feature_scaler)
        y = model(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        predicted_sr_batch = np.exp(log_SR.detach().cpu().numpy()).squeeze()
        predicted_sr.extend(predicted_sr_batch)

    data["predicted_sr"] = predicted_sr
    data.dropna(subset=predictors, inplace=True)
    data.to_csv("results/raw_plot_data_SR_pred.csv", index=False)


    # data = data[data.sr > 10]

    # Plot predicted species richness against ground truth
    fig, ax = plt.subplots()
    ax.scatter(data["sr"], data["predicted_sr"], alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Observed SR")
    ax.set_ylabel("Predicted SR")
    
    r2 = r2_score(data["sr"], data["predicted_sr"])
    correlation = np.corrcoef(data["sr"], data["predicted_sr"])[0, 1]
    ax.text(0.05, 0.90, f'Correlation: {correlation:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.05, 0.95, f'R2: {r2:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    fig.savefig("predicted_vs_observed_SR.png", dpi=300)

    # -------------------------------------
    # plotting extent area vs plot data
    gdf = load_preprocessed_data("all", HASH, seed)
    
    def calculate_extent_area(geometry):
        minx, miny, maxx, maxy = geometry.bounds
        return (maxx - minx) * (maxy - miny)
    
    gdf["extent_area"] = gdf.geometry.apply(calculate_extent_area)
    
    import matplotlib.pyplot as plt


    fig, ax = plt.subplots()
    ax.scatter(gdf["area"], gdf["sr"], label="SR vs Area", alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Plot area (m2)")
    ax.set_ylabel("Species Richness")
    fig.savefig("plot_area.png", dpi=300)
    
    
    fig, ax = plt.subplots()
    ax.scatter(gdf["extent_area"], gdf["sr"], label="SR vs Area", alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Extent area (m2)")
    ax.set_ylabel("Species Richness")
    fig.savefig("extent_area.png", dpi=300)
    
    gdf[predictors + ["sr", "area", "extent_area"]].to_csv("training_data_all.csv", sep=";")
    
    _gdf = gdf[gdf["num_plots"] > 1]
    fig, ax = plt.subplots()
    # Plot extent_area vs area and display the coefficient of proportionality
    # Fit a linear model to obtain the relationship
    X = np.log(_gdf["area"].values.astype(np.float32).reshape(-1, 1))
    y = np.log(_gdf["extent_area"]).values
    model = LinearRegression()
    model.fit(X, y)
    
    ax.scatter(_gdf["area"], _gdf["extent_area"], alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Plot area (m2)")
    ax.set_ylabel("Extent area (m2)")

    slope = model.coef_[0]
    intercept = model.intercept_
    
    print("Slope:", slope)
    print("Intercept:", intercept)
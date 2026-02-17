"""
Projecting spatially the predictions of an ensembled `Deep4PWeibull` model,
and saving SR, std_SR, dSR/dlogA and std_dSR/dlogA to geotiffs.
"""
import torch
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from deepsar.utils import load_ensemble_from_folds
from deepsar.data_processing.utils_features import EnvironmentalFeatureDataset
from deepsar.data_processing.utils_eva import COUNTRY_DATA, COUNTRY_LIST
from deepsar.plotting import CMAP_BR

ROOT = Path(__file__).parents[2]
TRAINING_DATASET_SEED = "a9a058d"
RUN_DIR = ROOT / "scripts" / "results" / "train" / f"{TRAINING_DATASET_SEED}_no_lc_features"

def create_raster(X_map, ypred):
    Xy_map = X_map.copy()
    Xy_map["pred"] = ypred
    rast = Xy_map["pred"].to_xarray().sortby(["y","x"])
    rast = xr.DataArray(rast.values, dims=["y", "x"], coords={
                            "x": rast.x.values,  # X coordinates (easting)
                            "y": rast.y.values,  # Y coordinates (northing)
                        },
                        name="pred")
    rast = rast.rio.write_crs("EPSG:3035")
    return rast

def plot_raster(rast, label, ax, cmap, vmin=None, vmax=None):
        # world.boundary.plot(ax=ax, linewidth=0.1, edgecolor='black')
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, "label":"","pad":0.05, "location":"bottom"} #if display_cbar else {}
        # rolling window for smoothing
        rast.where(rast > 0.).plot(ax=ax,
                                    cmap=cmap, 
                                    cbar_kwargs=cbar_kwargs, 
                                    vmin=vmin, 
                                    vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        
def create_features(model, env_dataset, lc_dataset, res):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
    resolution = abs(env_dataset.rio.resolution()[0])
    ncells = max(1, int(res / resolution))

    # determine which environmental variables are needed
    env_vars = [
        v for v in env_dataset.data_vars
        if (v in model.feature_names) or (f"std_{v}" in model.feature_names)
    ]
    env_subset = env_dataset[env_vars]
    coarse = env_subset.coarsen(x=ncells, y=ncells, boundary="trim")

    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035")
    coarse_std = coarse.std().rio.write_crs("EPSG:3035")

    mean_cols = [c for c in coarse_mean.data_vars if c in model.feature_names]
    std_cols = [c for c in coarse_std.data_vars if f"std_{c}" in model.feature_names]
    mean_ds = coarse_mean[mean_cols]
    std_ds = coarse_std[std_cols].rename({c: f"std_{c}" for c in std_cols})
    X_map = xr.merge([mean_ds, std_ds])

    # landcover fractions
    lc_frac_cols = [c for c in model.feature_names if c.startswith("lc_frac_")]
    if lc_frac_cols:
        lc_da = lc_dataset["landcover"].where(lc_dataset["landcover"] >= 0)
        for col in sorted(lc_frac_cols, key=lambda c: int(c.split("_")[-1])):
            idx = int(col.split("_")[-1])
            frac = (lc_da == idx).coarsen(x=ncells, y=ncells, boundary="trim").mean()
            X_map[col] = frac

    X_map = X_map.assign(log_sp_unit_area=np.log(res**2))
    return X_map[model.feature_names]


def align_features_to_reference(features_ref: xr.Dataset, features_other: xr.Dataset) -> xr.Dataset:
    if features_ref.rio.crs is None:
        features_ref = features_ref.rio.write_crs("EPSG:3035")
    if features_other.rio.crs is None:
        features_other = features_other.rio.write_crs(features_ref.rio.crs)
    return features_other.rio.reproject_match(features_ref).assign_coords(x=features_ref.x, y=features_ref.y)
        
# we use batches, otherwise model and data may not fit in memory
def get_SR_dSR_stats(model, env_dataset, lc_dataset, res0, batch_size=2**15):
    """
    Calculate SR, std_SR and dlogSR_dlogA for the given model and climate
    dataset at a specified resolution. dSR is obtained as a gradient of SR with
    respect to log_sp_unit_area. Does not account for changes in climate
    features with area.
    """
    
    resolution = abs(env_dataset.rio.resolution()[0])
    ncells0 = max(1, int(res0 / resolution))
    ncells1 = 2*ncells0
    res1 = ncells1 * resolution

    print(f"Creating features for res0: {res0}m")
    features0 = create_features(model, env_dataset, lc_dataset, res0)
    print(f"Creating features for res1: {res1}m")
    features1 = create_features(model, env_dataset, lc_dataset, res1)

    # Align features to a common grid for dSR/dlogA computation
    features1 = align_features_to_reference(features0, features1)

    features0_df = features0.to_dataframe()
    features1_df = features1.to_dataframe()

    features0_df = features0_df.dropna(how="all")
    features1_df = features1_df.reindex_like(features0_df)
    total_length = len(features0_df)
    

    percent_step = max(1, total_length // batch_size // 100)
    
    SR01_list = []
    for features in [features0_df, features1_df]:
        SR_list = []
        for i in tqdm(range(0, total_length, batch_size), desc="Calculating SR and stdSR", miniters=percent_step, maxinterval=float("inf")):
            with torch.no_grad():
                current_batch_size = min(batch_size, total_length - i)
                X = features.iloc[i:i+current_batch_size,:]
                if X.empty:
                    continue
                SRs = [m.predict_sr_tot(X) for m in model.models]
                SR_list.append(np.concatenate(SRs, axis=1))
        SR01_list.append(np.concatenate(SR_list, axis=0))


    mean_SR = np.mean(SR01_list[0], axis=1)
    # mean_SR = SR01_list[0][:, 2]
    std_SR = np.std(SR01_list[0], axis=1)
        
    dSR_dlogA = (SR01_list[1] - SR01_list[0]) / (res1 - res0)
    mean_dSR_dlogA = np.nanmean(dSR_dlogA, axis=1)
    std_dSR_dlogA = np.std(dSR_dlogA, axis=1)
    return features0_df, mean_SR, std_SR, mean_dSR_dlogA, std_dSR_dlogA


def load_environmental_features() -> tuple[xr.Dataset, xr.Dataset]:
    env_features = EnvironmentalFeatureDataset()
    env_ds, lc_ds = env_features.load(use_cache=True)
    env_ds = env_ds.rio.write_crs("EPSG:3035")
    lc_ds = lc_ds.rio.write_crs("EPSG:3035")

    countries_gdf = gpd.read_file(COUNTRY_DATA)
    eva_countries_gdf = countries_gdf[countries_gdf["NAME_EN"].isin(COUNTRY_LIST)]
    if eva_countries_gdf.crs != "EPSG:3035":
        eva_countries_gdf = eva_countries_gdf.to_crs("EPSG:3035")

    env_ds = env_ds.rio.clip(eva_countries_gdf.geometry, drop=True)
    lc_ds = lc_ds.rio.clip(eva_countries_gdf.geometry, drop=True)
    return env_ds, lc_ds

if __name__ == "__main__":
    plotting = True

    model_name = RUN_DIR.name

    projection_path = Path(__file__).parents[2] / Path(f"data/processed/projections/{model_name}")
    projection_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ensemble_from_folds(RUN_DIR, device=device)
    env_dataset, lc_dataset = load_environmental_features()

    for res in [5e3, 5e4]:
        print(f"Calculating SR, and stdSR for resolution: {res}m")

        features0, mean_SR, std_SR, mean_dSR_dlogA, std_dSR_dlogA = get_SR_dSR_stats(
            model, env_dataset, lc_dataset, res
        )
        if len(features0) == 0:
            print(f"No features for resolution {res}m; skipping.")
            continue

        raster_configs = [
            ("SR", mean_SR, "SR"),
            ("std_SR", std_SR, "Standard Deviation of SR"),
            ("dSR_dlogA", mean_dSR_dlogA, "dSR/dlogA"),
            ("std_dSR_dlogA", std_dSR_dlogA, "Standard Deviation of dSR/dlogA"),
        ]

        for raster_name, data, plot_title in raster_configs:
            rast = create_raster(features0, data)
            rast.rio.to_raster(projection_path / f"{raster_name}_raster_{model_name}_{res:.0f}m.tif")

            if plotting:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                rast_renamed = rast.rename(plot_title)
                rast_renamed.plot(ax=ax, cmap=CMAP_BR, vmin=rast.quantile(0.01), vmax=rast.quantile(0.99))
                ax.set_title(f"{plot_title} - Res: {res}m")
                fig.savefig(
                    projection_path / f"{raster_name}_raster_{model_name}_{res:.0f}m.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

        print(f"Saved rasters in {projection_path}")
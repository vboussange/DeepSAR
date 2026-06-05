"""
Projecting spatially the predictions of the ensembled `MuScaRi` model, and saving to geotiff files.
The ensemble must already be exported by `muscari.trainer.Trainer`.
"""
import json
import os
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone

from muscari import MuScaRiEnsemble
from muscari.data_processing.utils_features import EnvironmentalFeatureDataset
from muscari.plotting import CMAP_BR
from muscari.utils import get_git_hash
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from data_processing.eva_preprocessing import COUNTRY_DATA, COUNTRY_LIST


DEFAULT_RUN_DIR = Path(__file__).parent / "results" / "train" / "6dcd90c"
RUN_DIR = Path(os.environ.get("MUSCARI_PROJECT_RUN_DIR", DEFAULT_RUN_DIR))
EXPORT_DIR = RUN_DIR / "ensemble_pretrained"
MODEL_NAME = RUN_DIR.name
PROJECTION_PATH = Path(__file__).parents[1] / "data/processed/projections" / MODEL_NAME
RESOLUTIONS_M = [r * 1e3 for r in [2, 2**3, 2**6, 2**7]]
PLOTTING = True
SMOKE_TEST = os.environ.get("MUSCARI_PROJECT_SMOKE", "0") == "1"

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
        # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
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

    # See: https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.reproject_match
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035")
    coarse_std = coarse.std().rio.write_crs("EPSG:3035")
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)

    mean_cols = [c for c in df_mean.columns if c in model.feature_names]
    std_cols = [c for c in df_std.columns if c in model.feature_names]
    X_map = pd.concat([df_mean[mean_cols], df_std[std_cols]], axis=1)

    # landcover fractions
    lc_frac_cols = [c for c in model.feature_names if c.startswith("lc_frac_")]
    if lc_frac_cols:
        lc_da = lc_dataset["landcover"].where(lc_dataset["landcover"] >= 0)
        for col in sorted(lc_frac_cols, key=lambda c: int(c.split("_")[-1])):
            idx = int(col.split("_")[-1])
            frac = (lc_da == idx).coarsen(x=ncells, y=ncells, boundary="trim").mean()
            X_map[col] = frac.to_dataframe(name=col)[col]

    X_map = X_map.assign(log_sp_unit_area=np.log(res**2))
    return X_map[model.feature_names]
        
# we use batches, otherwise model and data may not fit in memory
def batch_predict(model, env_dataset, lc_dataset, res, batch_size=4096):
    """
    Calculate SR, std_SR and dlogSR_dlogA for the given model and environmental
    datasets at a specified resolution. dSR is obtained as a gradient of SR with
    respect to log_sp_unit_area. Does not account for changes in climate
    features with area.
    """
    mean_SR_list = []
    std_SR_list = []
    features = create_features(model, env_dataset, lc_dataset, res)
    total_length = len(features)

    percent_step = max(1, total_length // batch_size // 100)
    
    for i in tqdm(range(0, total_length, batch_size), desc = "Calculating SR and stdSR", miniters=percent_step, maxinterval=float("inf")):
        with torch.no_grad():
            current_batch_size = min(batch_size, total_length - i)
            X = features.iloc[i:i+current_batch_size,:]
            mean_SR_list.append(model.predict_mean_sr_tot(X))
            std_SR_list.append(model.get_std_sr_tot(X))
        
    mean_SR = np.concatenate(mean_SR_list, axis=0)
    std_SR = np.concatenate(std_SR_list, axis=0)
    return features, mean_SR, std_SR
        
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


def load_exported_model(export_dir: Path, device: str) -> MuScaRiEnsemble:
    if not export_dir.exists():
        raise FileNotFoundError(
            f"Missing exported model at {export_dir}. Run the unified Trainer with "
            "export_pretrained=True before projecting."
        )
    model = MuScaRiEnsemble.from_pretrained(export_dir)
    return model.to(device).eval()


def write_projection_metadata(projection_path: Path, device: str):
    metadata = {
        "run_dir": str(RUN_DIR),
        "export_dir": str(EXPORT_DIR),
        "model_name": MODEL_NAME,
        "projection_path": str(projection_path),
        "resolutions_m": RESOLUTIONS_M,
        "device": device,
        "git_hash": get_git_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(projection_path / "projection_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_exported_model(EXPORT_DIR, device)
    PROJECTION_PATH.mkdir(parents=True, exist_ok=True)
    write_projection_metadata(PROJECTION_PATH, device)
    if SMOKE_TEST:
        print(f"Loaded exported model {EXPORT_DIR} on {device}.")
        raise SystemExit(0)

    env_dataset, lc_dataset = load_environmental_features()

    for res in RESOLUTIONS_M:
        print(f"Calculating SR, and stdSR for resolution: {res}m")
        features, SR, std_SR = batch_predict(model, env_dataset, lc_dataset, res)

        SR_rast = create_raster(features, SR)
        SR_rast.rio.to_raster(PROJECTION_PATH / f"SR_raster_{MODEL_NAME}_{res:.0f}m.tif")
        if PLOTTING:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            # colors = ["#dad7cd","#a3b18a","#588157","#3a5a40","#344e41"]
            SR_rast = SR_rast.rename("SR")
            SR_rast.plot(ax=ax, cmap=CMAP_BR, vmin=SR_rast.quantile(0.01), vmax=SR_rast.quantile(0.99))
            ax.set_title(f"Res: {res}m")
            fig.savefig(PROJECTION_PATH / f"SR_raster_{MODEL_NAME}_{res:.0f}m.png", dpi=300, bbox_inches='tight')
        
        # std_SR_rast = create_raster(features, std_SR)
        # std_SR_rast.rio.to_raster(projection_path / f"std_SR_raster_{model_name}_{res:.0f}m.tif")
        
        # print(f"Saved SR, std_SR, dlogSR_dlogA for resolution: {res}m in {projection_path}")
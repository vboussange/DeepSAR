"""
Validating EVA augmentation against GIFT data
"""
import torch
import numpy as np
import xarray as xr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.mlp import MLP, CustomMSELoss
from src.trainer import Trainer
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from pathlib import Path
from src.data_processing.utils_env_pred import CHELSADataset
from src.dataset import AugmentedDataset, create_dataloader
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


@dataclass
class Config:
    device: str
    batch_size: int = 1024
    num_workers: int = 4
    test_size: float = 0.1
    val_size: float = 0.1
    lr: float = 5e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    n_epochs: int = 50
    dSRdA_weight: float = 1e0
    weight_decay: float = 1e-3
    seed: int = 1
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T", "Q", "S", "R"])
    n_ensembles: int = 5  # Number of models in the ensemble
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/fb8bc71/eva_chelsa_augmented_data.pkl"
    path_gift_data: str = Path(__file__).parent / f"../data/processed/GIFT_CHELSA_compilation/fb8bc71/megaplot_data.gpkg"

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
        
        
def create_features(climate_dataset, res):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars

    resolution = abs(climate_dataset.rio.resolution()[0])
    ncells = max(1, int(res / resolution))
    coarse = climate_dataset.coarsen(x=ncells, y=ncells, boundary="trim")
    
    # See: https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.reproject_match
    coarse_mean = coarse.mean().rio.write_crs("EPSG:3035")#.rio.reproject_match(ref)
    coarse_std = coarse.std().rio.write_crs("EPSG:3035")#.rio.reproject_match(ref)
    df_mean = coarse_mean.to_dataframe()
    df_std = coarse_std.to_dataframe()
    df_std = df_std.rename({col: "std_" + col for col in df_std.columns}, axis=1)
    X_map = pd.concat([df_mean, df_std], axis=1)
    return X_map[climate_features]
        
# we use batches, otherwise model and data may not fit in memory
def get_SR(model, raster_features, feature_scaler, target_scaler, batch_size=4096):
    SR_all = []
    total_length = len(raster_features)

    percent_step = max(1, total_length // batch_size // 100)
    
    for i in tqdm(range(0, total_length, batch_size), desc = "Calculating SR and stdSR", miniters=percent_step, maxinterval=float("inf")):
        with torch.no_grad():
            current_batch_size = min(batch_size, total_length - i)
            # features = get_true_sar.interpolate_features(X_map_dict, log_area_tensor, res_climate_pixel, predictors, batch_index=slice(i, i + current_batch_size))
            X = raster_features.iloc[i:i+current_batch_size,:]
            X = feature_scaler.transform(X.values)
            X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
            y = model(X)
            SR = np.exp(target_scaler.inverse_transform(y.cpu().numpy()))

            SR_all.append(SR)
        
    SR_all = np.concatenate(SR_all, axis=0)
    return SR_all
        
        
def load_chelsa_and_reproject():
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in config.climate_variables]]
    climate_dataset = climate_dataset.rio.reproject("EPSG:3035")
    return climate_dataset

if __name__ == "__main__":    
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    config = Config(device=device)
    

    augmented_dataset = AugmentedDataset(path_eva_data = config.path_eva_data,
                                         path_gift_data = config.path_gift_data,
                                         seed = config.seed)
    
    df = augmented_dataset.compile_training_data("all")
    df["coverage"] = df["log_area"] / df["log_megaplot_area"]
    # proportion_area = df[df["type"]=="GIFT"]["area"].mean() / df[df["type"]=="GIFT"]["megaplot_area"].mean() #TODO: this is to be modified
    

    climate_dataset = load_chelsa_and_reproject()
    raster_features = create_features(climate_dataset, 1e4)
    
    
    # plotting values of raster_features vs training features
    # n_features = len(predictors)
    # ncols = 3
    # nrows = int(np.ceil(n_features / ncols))
    # fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    # for idx, feature in enumerate(predictors):
    #     df_reduced = df.sample(n=10000, random_state=config.seed)
    #     raster_features_reduced = raster_features.sample(n=10000, random_state=config.seed)
    #     ax = axes[idx // ncols, idx % ncols]
    #     sns.kdeplot(df_reduced[feature], ax=ax, label="Training", fill=True, color="blue")
    #     sns.kdeplot(raster_features_reduced[feature], ax=ax, label="Raster", fill=True, color="orange")
    #     ax.set_title(feature)
    #     ax.legend()
    #     ax.set_xlabel(feature)
    #     ax.set_ylabel("Density")
    #     # ax.set_yscale("log")

    # # Hide any unused subplots
    # for i in range(n_features, nrows * ncols):
    #     fig.delaxes(axes[i // ncols, i % ncols])

    # plt.tight_layout()
    # plt.show()
    # xgb_params ={
    #     "booster": "gbtree",
    #     "learning_rate": 0.05,
    #     # "max_depth": 4,
    #     # "lambda": 10,
    #     "objective": "reg:squarederror",  # can be reg:squarederror, reg:squaredlogerror
    #     # "min_child_weight": 1.0,
    #     # "device" : "cuda",
    #     "tree_method": "hist",
    # }
    reg = XGBRegressor()
    # reg = LinearRegression()
    # reg = MLPRegressor(
    #     hidden_layer_sizes=(16, 16, 16),
    #     activation="relu",
    #     solver="adam")
    
    # predictors = config.climate_variables + ["std_"+c for c in config.climate_variables] + ["coverage", "log_megaplot_area"]
    # predictors = ["coverage", "log_megaplot_area"]
    
    ## PREDICTING RAW EVA plots
    predictors = config.climate_variables + ["std_"+c for c in config.climate_variables] + ["log_area"]

    df_raw_plots = df[df["type"] == "EVA_raw"]
    df_raw_plots = df_raw_plots[df_raw_plots["habitat_id"] == "S"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(range(len(df_raw_plots)), groups=df_raw_plots["partition"]))
    X_train, y_train = df_raw_plots.iloc[train_idx][predictors].values, df_raw_plots.iloc[train_idx]["log_sr"].values
    X_test, y_test = df_raw_plots.iloc[test_idx][predictors].values, df_raw_plots.iloc[test_idx]["log_sr"].values
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    plt.scatter(y_test, y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.text(0.05, 0.95, f"R2: {r2:.4f}\nCorr: {corr:.4f}", transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.xlabel("True log(SR)")
    plt.ylabel("Predicted log(SR)")
    plt.legend()
    plt.show()
    
    
    
    ## PREDICTING RAW GIFT plots
    predictors = config.climate_variables + ["std_"+c for c in config.climate_variables] + ["log_megaplot_area"]
    # predictors =  ["log_megaplot_area"]

    df_raw_plots = df[df["type"] == "GIFT"]
    train_idx, test_idx = train_test_split(range(len(df_raw_plots)), test_size=0.2, random_state=42)
    X_train, y_train = df_raw_plots.iloc[train_idx][predictors].values, df_raw_plots.iloc[train_idx]["log_sr"].values
    X_test, y_test = df_raw_plots.iloc[test_idx][predictors].values, df_raw_plots.iloc[test_idx]["log_sr"].values
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    mse = np.mean((y_test - y_pred) ** 2)
    plt.scatter(y_test, y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.text(0.05, 0.95, f"R2: {r2:.4f}\nCorr: {corr:.4f}\nMSE: {mse:.4f}", transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.xlabel("True log(SR)")
    plt.ylabel("Predicted log(SR)")
    plt.legend()
    plt.show()

    
    ## PREDICTING SAR
    predictors = config.climate_variables + ["std_" + c for c in config.climate_variables] + ["log_area"]
    # predictors = config.climate_variables + ["log_area"]


    df_megaplots = df[df["type"].isin(["EVA_megaplot", "GIFT"])]
    # #### GROUP SHUFFLE SPLIT
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # train_idx, test_idx = next(gss.split(range(len(df_megaplots)), groups=df_megaplots["partition"]))
    # df_train, df_test = df_megaplots.iloc[train_idx], df_megaplots.iloc[test_idx]
    # X_train, y_train = df_train[predictors].values, df_train["log_sr"].values
    # # y_train, y_test = y[train_idx], y[test_idx]
    # # df_test = df_test[df_test["type"] == "GIFT"]
    # X_test, y_test = df_test[predictors].values, df_test["log_sr"].values
    
    #### TRAINING ON MEGAPLOT ONLY, EVALUATING ON GIFT
    df_train = df_megaplots[df_megaplots["type"]=="EVA_megaplot"]
    df_test = df_megaplots[df_megaplots["type"]=="GIFT"]
    
    X_train = df_train[predictors].values
    y_train = df_train["log_sr"].values

    reg.fit(X_train, y_train)

    X_test = df_test[predictors].values
    y_test = df_test["log_sr"].values

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    mse = np.mean((y_test - y_pred) ** 2)
    plt.scatter(y_test, y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.text(0.05, 0.95, f"R2: {r2:.4f}\nCorr: {corr:.4f}\nMSE: {mse:.4f}", transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.xlabel("True log(SR)")
    plt.ylabel("Predicted log(SR)")
    plt.legend()
    plt.show()
    
    
    SR = np.exp(reg.predict(raster_features.values))

    SR_rast = create_raster(raster_features, SR)
    SR_rast.plot()

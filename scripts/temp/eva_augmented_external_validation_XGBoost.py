"""
Validating EVA augmentation against external GIFT data.
"""
import torch
import numpy as np
import xarray as xr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.mlp import MLP, CustomMSELoss
from src.trainer import Trainer
from dataclasses import dataclass, field
import geopandas as gpd

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
from src.utils import choose_device
import pprint

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


class ExternalValidator():
    def __init__(self, df, include_raw_plots, random_state=42):
        GIFT_valid_df = df[df["type"] == "GIFT"]
        if include_raw_plots:
            EVA_augmented_df = df[df["type"] != "GIFT"]
        else:
            EVA_augmented_df = df[df["type"] == "EVA_megaplot"]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        train_idx, test_idx = next(gss.split(range(len(EVA_augmented_df)), groups=EVA_augmented_df["partition"]))
        
        EVA_test_augmented_df = EVA_augmented_df.iloc[test_idx]
        test_partitions = EVA_test_augmented_df.partition.unique()
        EVA_raw_valid_df = df[(df["type"] == "EVA_raw") & (df["partition"].isin(test_partitions))]
        EVA_megaplot_valid_df = EVA_test_augmented_df[EVA_test_augmented_df["type"] == "EVA_megaplot"]

        self.GIFT_valid_df = GIFT_valid_df
        self.EVA_raw_valid_df = EVA_raw_valid_df
        self.EVA_megaplot_valid_df = EVA_megaplot_valid_df
        self.training_df = EVA_augmented_df.iloc[train_idx]
        
    def validate(self, model, predictors, sample_weight=None):
        linear_model = LinearRegression()
        linear_model.fit(self.training_df[["log_area"]].values, self.training_df["log_sr"].values)
        model.fit(self.training_df[predictors].values, 
                  self.training_df["log_sr"].values,
                  sample_weight=sample_weight)
        
        bench_log = {}
        for df, name in zip([self.GIFT_valid_df, self.EVA_raw_valid_df, self.EVA_megaplot_valid_df], ["GIFT validation", "EVA raw validation", "EVA megaplot validation"]):
            y_val = df["log_sr"].values

            y_pred_linear_model = linear_model.predict(df[["log_area"]].values)
            mse_value_linear_model = np.mean((y_val - y_pred_linear_model) ** 2)
            
            y_pred_model = model.predict(df[predictors].values)
            mse_value_model = np.mean((y_val - y_pred_model) ** 2)
            
            model_score = 1 - mse_value_model/mse_value_linear_model
            model_r2 = r2_score(y_val, y_pred_model)
            bench_log[name] = {"model_r2": model_r2, "model_mse": mse_value_model, "model_score": model_score}
        return bench_log

if __name__ == "__main__":
    @dataclass
    class Config:
        device: str = choose_device()
        climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
        path_eva_data: str = Path(__file__).parent / f"../../data/processed/EVA_CHELSA_compilation/627173c/eva_chelsa_megaplot_data.gpkg"
        path_gift_data: str = Path(__file__).parent / f"../../data/processed/GIFT_CHELSA_compilation/fb8bc71/megaplot_data.gpkg"
        seed: int = 2

    config = Config()
            
    # quick and dirty test
    eva_augmented_data = gpd.read_file(config.path_eva_data)
    
    eva_augmented_data_train = eva_augmented_data[eva_augmented_data["type"] == "EVA_megaplot_train"]
    eva_augmented_data_test = eva_augmented_data[eva_augmented_data["type"] == "EVA_megaplot_test"]
    print(len(eva_augmented_data_train))
    print(len(eva_augmented_data_test))
    fig, ax = plt.subplots()
    ax.scatter(eva_augmented_data_train["observed_area"].astype(np.float32), 
              eva_augmented_data_train["megaplot_area"].astype(np.float32), 
              alpha=0.1, 
              c="tab:blue")
    ax.scatter(eva_augmented_data_test["observed_area"].astype(np.float32), 
              eva_augmented_data_test["megaplot_area"].astype(np.float32), 
            #   alpha=0.1, 
              c="tab:red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # TODO: you may also want to plot the distribution of the ratio of observed area to megaplot area, or scatter megaplot area vs observed area
    

    # ax.hexbin(eva_augmented_data["observed_area"], 
    #           eva_augmented_data["sr"], 
    #         #   alpha=0.1, 
    #           gridsize=30,
    #           xscale="log",
    #           yscale="log",
    #           cmap="magma",)
    
    
    augmented_dataset = AugmentedDataset(path_eva_data = config.path_eva_data,
                                         path_gift_data = config.path_gift_data,
                                         seed = config.seed)
    
    df = augmented_dataset.compile_training_data("all")
    df["coverage"] = df["log_area"] / df["log_megaplot_area"]
    df.dropna(inplace=True)
    
    EVA_megaplot_r2 = []
    EVA_raw_r2 = []
    GIFT_r2 = []
    for i in range(5):
        validator = ExternalValidator(df, include_raw_plots=True, random_state=i)
        pprint.pprint({
            "training_df": len(validator.training_df),
            "GIFT_valid_df": len(validator.GIFT_valid_df),
            "EVA_raw_valid_df": len(validator.EVA_raw_valid_df),
            "EVA_megaplot_valid_df": len(validator.EVA_megaplot_valid_df)
        })
        predictors = config.climate_variables + ["std_"+c for c in config.climate_variables] + ["log_area", "log_megaplot_area"]
        reg = XGBRegressor(max_depth=5,
                            min_child_weight=5,    # Higher values prevent overfitting
                            n_estimators=1000, 
                            learning_rate=0.01, 
                            subsample=0.5, 
                            colsample_bytree=0.5,
                            reg_alpha=0.1,         # L1 regularization
                            reg_lambda=1.0,        # L2 regularization
                            )
        xgb_stats = validator.validate(reg, 
                                       predictors, 
                                       validator.training_df["coverage"].values.clip(0., 1)
                                       )
        EVA_megaplot_r2.append(xgb_stats["EVA megaplot validation"]["model_r2"])
        EVA_raw_r2.append(xgb_stats["EVA raw validation"]["model_r2"])
        GIFT_r2.append(xgb_stats["GIFT validation"]["model_r2"])
        # print("EVA megaplot r2:", EVA_megaplot_r2[-1])
        # print("EVA raw r2:", EVA_raw_r2[-1])
        # print("GIFT r2:", GIFT_r2[-1])
        pprint.pprint(xgb_stats)
        

    print("------------------ SUMMARY STATS------------------")
    print("EVA megaplot r2:", np.mean(EVA_megaplot_r2), np.std(EVA_megaplot_r2))
    print("EVA raw r2:", np.mean(EVA_raw_r2), np.std(EVA_raw_r2))
    print("GIFT r2:", np.mean(GIFT_r2), np.std(GIFT_r2))
    
    
    
    
    
    
    
    # FURTHER TESTS
    # WITH GIFT DATA
    reg = XGBRegressor(max_depth=5, n_estimators=1000, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5)
    predictors = config.climate_variables + ["std_"+c for c in config.climate_variables] + ["log_area", "log_megaplot_area"]
    # SEEMS LIKE IT WORKS BEST WITH VERY SMALL TRAINING DATA SIZE
    training_data = df[df["type"] != "GIFT"].sample(frac=0.8, random_state=15)
    reg.fit(training_data[predictors].values, training_data["log_sr"].values)
    test_data = df[df["type"] == "GIFT"]
    y_pred = reg.predict(test_data[predictors].values)
    print("R2:", r2_score(test_data["log_sr"].values, y_pred))
    
    # WITH EVA DATA
    training_data = df[df["type"] == "EVA_megaplot"].sample(frac=0.01, random_state=1)
    reg.fit(training_data[predictors].values, training_data["log_sr"].values)
    test_data = df[df["type"] == "EVA_raw"]
    y_pred = reg.predict(test_data[predictors].values)
    print("R2:", r2_score(test_data["log_sr"].values, y_pred))

    
    # PROJECTIONS
    climate_dataset = load_chelsa_and_reproject()
    res = 1e3
    proj_features = create_features(climate_dataset, res)
    proj_features["log_area"] = np.log(res**2)
    proj_features["log_megaplot_area"] = np.log(res**2)
    
    SR = np.exp(reg.predict(proj_features[predictors].values))

    SR_rast_1e3 = create_raster(proj_features, SR)
    # SR_rast_1e3.plot()
    
    res = 1e4
    proj_features = create_features(climate_dataset, res)
    proj_features["log_area"] = np.log(res**2)
    proj_features["log_megaplot_area"] = np.log(res**2)
    
    SR = np.exp(reg.predict(proj_features[predictors].values))

    SR_rast_1e4 = create_raster(proj_features, SR)
    SR_rast_1e4 = SR_rast_1e4.rio.reproject_match(SR_rast_1e3)
    (SR_rast_1e4 - SR_rast_1e3).plot()
    
    
    res = 5e4
    proj_features = create_features(climate_dataset, res)
    proj_features["log_area"] = np.log(res**2)
    proj_features["log_megaplot_area"] = np.log(res**2)
    
    SR = np.exp(reg.predict(proj_features[predictors].values))

    SR_rast_1e5 = create_raster(proj_features, SR)
    SR_rast_1e5 = SR_rast_1e5.rio.reproject_match(SR_rast_1e4)
    (SR_rast_1e5 - SR_rast_1e4).plot()
    
    
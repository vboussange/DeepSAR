"""
Projecting spatially MLP and saving to geotiff files.
# TODO: this version is more recent than export_SR_maps.py, but 
# 1. simplifies the code, with no interpolation of features
# 2. loads output from train_single_habitat; should be modified to load from train.py
# 3. We should also export sensitivity maps, which are not exported here.
"""
import torch
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
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


MODEL_ARCHITECTURE = {"small":[16, 16, 16],
                    "large": [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7],
                    "medium":[2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]}
MODEL = "small"
HASH = "fb8bc71"
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
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    habitats: list = field(default_factory=lambda: ["all", "T", "Q", "S", "R"])
    n_ensembles: int = 5  # Number of models in the ensemble
    layer_sizes: list = field(default_factory=lambda: MODEL_ARCHITECTURE[MODEL]) # [16, 16, 16] # [2**11, 2**11, 2**11, 2**11, 2**11, 2**11, 2**9, 2**7] # [2**8, 2**8, 2**8, 2**8, 2**8, 2**8, 2**6, 2**4]
    # test_partitions: list = field(default_factory=lambda: [684, 546, 100, 880, 1256, 296]) # ["T1", "T3", "R1", "R2", "Q5", "Q2", "S2", "S3", "all"]
    path_eva_data: str = Path(__file__).parent / f"../data/processed/EVA_CHELSA_compilation/{HASH}/eva_chelsa_augmented_data.pkl"
    path_gift_data: str = Path(__file__).parent / f"../data/processed/GIFT_CHELSA_compilation/{HASH}/megaplot_data.gpkg"

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
        
        
def create_features(predictor_labels, climate_dataset, res):
    # see: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.coarsen.html
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
    
    X_map = X_map.assign(log_area=np.log(res**2), log_megaplot_area=np.log(res**2)) #TODO: change res
    return X_map[predictor_labels]
        
# we use batches, otherwise model and data may not fit in memory
def get_SR(model, raster_features, feature_scaler, target_scaler, batch_size=4096):
    SR_all = []
    total_length = len(features)

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
        
        
def load_chelsa_and_reproject(predictors):
    climate_dataset = xr.open_dataset(CHELSADataset().cache_path)
    climate_dataset = climate_dataset[[v for v in climate_dataset.data_vars if v in predictors]]
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
    # df = df[df["type"].isin(["EVA_raw", "GIFT"])]
    # proportion_area = df[df["type"]=="GIFT"]["area"].mean() / df[df["type"]=="GIFT"]["megaplot_area"].mean() #TODO: this is to be modified

    climate_vars = config.climate_variables
    std_climate_vars = ["std_" + env for env in climate_vars]
    climate_features = climate_vars + std_climate_vars
    predictors = ["log_area"] + climate_features
    

    climate_dataset = load_chelsa_and_reproject(predictors)
    raster_features = create_features(predictors, climate_dataset, 1e4)
    
    
    # plotting values of raster_features vs training features
    n_features = len(predictors)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, feature in enumerate(predictors):
        df_reduced = df.sample(n=10000, random_state=config.seed)
        raster_features_reduced = raster_features.sample(n=10000, random_state=config.seed)
        ax = axes[idx // ncols, idx % ncols]
        sns.kdeplot(df_reduced[feature], ax=ax, label="Training", fill=True, color="blue")
        sns.kdeplot(raster_features_reduced[feature], ax=ax, label="Raster", fill=True, color="orange")
        ax.set_title(feature)
        ax.legend()
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        # ax.set_yscale("log")

    # Hide any unused subplots
    for i in range(n_features, nrows * ncols):
        fig.delaxes(axes[i // ncols, i % ncols])

    plt.tight_layout()
    plt.show()
    

    train_val_idx, test_idx = train_test_split(df.index,
                                            test_size= config.test_size,
                                            random_state=config.seed)
    gdf_train_val = df.loc[train_val_idx]
    _, feature_scaler, target_scaler = create_dataloader(gdf_train_val, predictors, config.batch_size, config.num_workers)

    train_idx, val_idx = train_test_split(train_val_idx,
                                        test_size=config.val_size,
                                        random_state=config.seed)
    gdf_train, gdf_val, gdf_test = df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

    train_loader, _, _ = create_dataloader(gdf_train, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)
    val_loader, _, _ = create_dataloader(gdf_val, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)
    test_loader, _, _ = create_dataloader(gdf_test, predictors, config.batch_size, config.num_workers, feature_scaler, target_scaler)

    model = MLP(len(predictors), config.layer_sizes)

    trainer = Trainer(config=config, 
                        model=model, 
                        feature_scaler=feature_scaler, 
                        target_scaler=target_scaler, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        test_loader=test_loader, 
                        compute_loss=CustomMSELoss(config.dSRdA_weight),
                        device=config.device,
                    # compute_loss = nn.MSELoss()
                        )
    best_model, _ = trainer.train(n_epochs=config.n_epochs, metrics=["mean_squared_error", "r2_score"])
    
    # TODO: fine tuning with GIFT dataset
    # trainer = Trainer(config=config, 
    #                 model=model, 
    #                 feature_scaler=feature_scaler, 
    #                 target_scaler=target_scaler, 
    #                 train_loader=train_loader, 
    #                 val_loader=val_loader, 
    #                 test_loader=test_loader, 
    #                 compute_loss=CustomMSELoss(config.dSRdA_weight),
    #                 device=config.device,
    #             # compute_loss = nn.MSELoss()
    #                 )
    # best_model, _ = trainer.train(n_epochs=config.n_epochs, metrics=["mean_squared_error", "r2_score"])
    
    SR = get_SR(best_model, raster_features, feature_scaler, target_scaler)

    SR_rast = create_raster(raster_features, SR)
    SR_rast.plot()

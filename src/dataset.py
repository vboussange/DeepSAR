import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from src.plotting import read_result
import geopandas as gpd

class AugmentedDataset:
    def __init__(self, path_eva_data, path_gift_data, seed):

        self.eva_data = read_result(path_eva_data)
        self.gift_data = gpd.read_file(path_gift_data)
        self.seed = seed
    
    def compile_training_data(self, hab):
        eva_data = self.eva_data
        gift_data = self.gift_data
        seed = self.seed
        if hab == "all":
            eva_plot_data = eva_data["plot_data_all"]
        else:
            eva_plot_data = eva_data["plot_data_all"][eva_data["plot_data_all"]["habitat_id"] == hab]
        eva_megaplot_data = eva_data["megaplot_data"][eva_data["megaplot_data"]["habitat_id"] == hab]
        gift_plot_data = gift_data[gift_data["habitat_id"] == hab]
        augmented_data = pd.concat([eva_plot_data, eva_megaplot_data, gift_plot_data], ignore_index=True)
        
        # stack with raw plot data
        augmented_data.loc[:, "log_area"] = np.log(augmented_data["area"].astype(np.float32)) 
        augmented_data.loc[:, "log_megaplot_area"] = np.log(augmented_data["megaplot_area"].astype(np.float32))
        augmented_data.loc[:, "log_sr"] = np.log(augmented_data["sr"].astype(np.float32))
        augmented_data = augmented_data.dropna()
        augmented_data = augmented_data[~augmented_data.isin([np.inf, -np.inf]).any(axis=1)]
        augmented_data = augmented_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        return augmented_data
    
class CustomDataLoader(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    

def scale_features_targets(gdf, predictors, feature_scaler=None, target_scaler=None):
    features = gdf[predictors].values.astype(np.float32)
    target = gdf["log_sr"].values.astype(np.float32)

    if feature_scaler is None:
        feature_scaler, target_scaler = MinMaxScaler(), MinMaxScaler()
        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def create_dataloader(gdf, predictors, batch_size, num_workers, feature_scaler=None, target_scaler=None, shuffle=True):
    X, y, feature_scaler, target_scaler = scale_features_targets(gdf, predictors, feature_scaler, target_scaler)
    dataset = CustomDataLoader(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, feature_scaler, target_scaler

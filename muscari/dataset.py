import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

class CustomDataLoader(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    

def scale_features_targets(gdf, feature_names, feature_scaler=None, target_scaler=None):
    features = gdf[["log_observed_area"] + feature_names].values.astype(np.float32)
    target = gdf["sr"].values.astype(np.float32)

    if feature_scaler is None:
        feature_scaler, target_scaler = MinMaxScaler(), MaxAbsScaler()
        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def create_dataloader(gdf, feature_names, batch_size, num_workers, feature_scaler=None, target_scaler=None, shuffle=True):
    X, y, feature_scaler, target_scaler = scale_features_targets(gdf, feature_names, feature_scaler, target_scaler)
    dataset = CustomDataLoader(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, feature_scaler, target_scaler

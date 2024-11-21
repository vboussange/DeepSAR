import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SpeciesAreaEnvDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.targets[index], self.features[index]
    

def scale_features_targets(gdf, predictors, feature_scaler=None, target_scaler=None):
    features = gdf[predictors].values.astype(np.float32)
    target = gdf["log_sr"].values.astype(np.float32)

    if feature_scaler is None:
        feature_scaler, target_scaler = StandardScaler(), StandardScaler()
        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def create_dataloader(gdf, predictors, batch_size, num_workers, feature_scaler=None, target_scaler=None, shuffle=True):
    X, y, feature_scaler, target_scaler = scale_features_targets(gdf, predictors, feature_scaler, target_scaler)
    dataset = SpeciesAreaEnvDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, feature_scaler, target_scaler

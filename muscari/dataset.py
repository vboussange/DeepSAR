import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from muscari.scaler_serialization import Log1pMaxScaler

class CustomDataLoader(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    

def _make_target_scaler(target_transform):
    if target_transform == "maxabs":
        return MaxAbsScaler()
    if target_transform == "log1p_max":
        return Log1pMaxScaler()
    raise ValueError("target_transform must be 'maxabs' or 'log1p_max'")


def scale_features_targets(
    gdf,
    feature_names,
    feature_scaler=None,
    target_scaler=None,
    target_transform="maxabs",
):
    if (feature_scaler is None) != (target_scaler is None):
        raise ValueError("feature_scaler and target_scaler must be provided together")
    features = gdf[["log_observed_area"] + feature_names].values.astype(np.float32)
    target = gdf["sr"].values.astype(np.float32)

    if feature_scaler is None:
        feature_scaler, target_scaler = MinMaxScaler(), _make_target_scaler(target_transform)
        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def create_dataloader(
    gdf,
    feature_names,
    batch_size,
    num_workers,
    feature_scaler=None,
    target_scaler=None,
    target_transform="maxabs",
    shuffle=True,
    pin_memory=None,
):
    X, y, feature_scaler, target_scaler = scale_features_targets(
        gdf,
        feature_names,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_transform=target_transform,
    )
    dataset = CustomDataLoader(X, y)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, feature_scaler, target_scaler

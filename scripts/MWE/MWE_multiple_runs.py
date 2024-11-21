# This script evaluates the standard deviations of predictions across different runs



import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
from shapely.geometry import Point, MultiPoint
from pathlib import Path
from MLP import CustomMSELoss, MLP, inverse_transform_scale_feature_tensor
    
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
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(features)

    else:
        features = feature_scaler.transform(features)
        
    if target_scaler is None:
        target_scaler = StandardScaler()
        target = target_scaler.fit_transform(target.reshape(-1,1))
    else:
        target = target_scaler.transform(target.reshape(-1,1))
        
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), feature_scaler, target_scaler

def create_dataloaders(gdf_train, gdf_test, predictors, batch_size, num_workers):
    X_train, y_train, feature_scaler, target_scaler = scale_features_targets(gdf_train, predictors)
    X_test, y_test, _, _ = scale_features_targets(gdf_test, predictors, feature_scaler, target_scaler)
    
    train_dataset = SpeciesAreaEnvDataset(X_train, y_train)
    test_dataset = SpeciesAreaEnvDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader, feature_scaler, target_scaler

def train_and_evaluate(model, gdf_full, predictors, criterion, optimizer, scheduler, config, seed):
    best_val_MSE, step = float('inf'), 0
    train_idx, test_idx = train_test_split(gdf_full.index, test_size=config['test_size'], random_state=seed)

    gdf_train, gdf_test = gdf_full.loc[train_idx], gdf_full.loc[test_idx]
    train_loader, val_loader, feature_scaler, target_scaler = create_dataloaders(gdf_train, gdf_test, predictors, config['batch_size'], config['num_workers'])

    for epoch in range(config['n_epochs']):
        running_train_loss = 0
        running_train_MSE = 0
        for log_sr, inputs in train_loader:
            model.train()
            inputs, log_sr = inputs.to(config['device']), log_sr.to(config['device'])
            if isinstance(criterion, torch.nn.MSELoss):
                outputs = model(inputs)
                loss = criterion(outputs, log_sr)
            else:
                inputs.requires_grad_(True)
                outputs = model(inputs)
                loss = criterion(outputs, inputs, log_sr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            step += log_sr.shape[0]
            with torch.no_grad():
                model.eval()
                y_pred = inverse_transform_scale_feature_tensor(outputs, target_scaler)
                y_true = inverse_transform_scale_feature_tensor(log_sr, target_scaler)
                train_MSE = torch.mean((y_pred-y_true)**2) * inputs.size(0)
                running_train_MSE += train_MSE.item()
            
        model.eval()
        running_val_MSE = 0
        with torch.no_grad():
            for log_sr, inputs in val_loader:
                inputs, log_sr = inputs.to(config['device']), log_sr.to(config['device'])
                outputs = model(inputs)
                                
                y_pred = inverse_transform_scale_feature_tensor(outputs, target_scaler)
                y_true = inverse_transform_scale_feature_tensor(log_sr, target_scaler)
                val_MSE = torch.mean((y_pred-y_true)**2) * inputs.size(0)
                running_val_MSE += val_MSE

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        avg_train_MSE = running_train_MSE / len(train_loader.dataset)
        avg_val_MSE = running_val_MSE / len(val_loader.dataset)

        print(f"Epoch {epoch} | Training Loss: {avg_train_loss:.4f} | Training MSE: {avg_train_MSE:.4f} | Validation MSE: {avg_val_MSE:.4f}")
        
        if (avg_val_MSE < best_val_MSE):
            best_model = copy.deepcopy(model).to("cpu")
            best_val_MSE = avg_val_MSE                
        
        scheduler.step(avg_val_MSE)    
    return {'best_model': best_model,
            'best_validation_loss': avg_val_MSE,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            "predictors": predictors,
            "train_idx": train_idx}

    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
        
    config = {
        "device": device,
        "batch_size": 1024,
        "num_workers": 0,
        "test_size": 0.1,
        "lr": 5e-3,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 20,
        "n_epochs": 50,
        "dSRdA_weight":1e0,
        "weight_decay": 1e-3,
        "seed": 2,
        "data_seed": 2,
        "hash": "aab842b",
        "climate_variables": ["bio1",
                                "pet_penman_mean",
                                "sfcWind_mean",
                                "bio4",
                                "rsds_1981-2010_range_V.2.1",
                                "bio12",
                                "bio15",],
    }
    run_name = Path(str(Path(__file__).parent / Path("results") /  Path(__file__).stem) + f"_dSRdA_weight_{config["dSRdA_weight"]:.0e}_seed_{config["seed"]}")

    hab = "all"    
    print(f"Training model with habitat {hab}")
    gdf = pd.read_csv("MWE_all.csv")
    climate_predictors = config["climate_variables"] + ["std_" + env for env in config["climate_variables"]]
    predictors = ["log_area"] + climate_predictors
    
    config['run_name'], config['run_folder'] = "checkpoint", os.path.join('results', run_name)
    os.makedirs(config['run_folder'], exist_ok=True)

    results_all = {}
    seeds = range(4)
    for seed in seeds:
        print(f"Training starting for seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = MLP(len(predictors)).to(config['device'])
        criterion = CustomMSELoss(config["dSRdA_weight"]).to(config['device'])
        # criterion = torch.nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, factor=config['lr_scheduler_factor'], patience=config['lr_scheduler_patience'])

        results= train_and_evaluate(model, gdf, predictors, criterion, optimizer, scheduler, config, seed)
        results_all[seed] = results

    print(f"Saving results in {config['run_folder']}")
    torch.save(results_all, os.path.join(config['run_folder'], f'{config["run_name"]}.pth'))
    
    log_SR_array = []
    for seed in seeds:
        results = results_all[seed]
        feature_scaler = results["feature_scaler"]
        target_scaler = results["target_scaler"]
        model = results["best_model"]
        X, _, _, _, = scale_features_targets(gdf, predictors, feature_scaler=feature_scaler)
        # X = X.to(device)
        y = model(X)
        log_SR = inverse_transform_scale_feature_tensor(y, target_scaler)
        log_SR = log_SR.cpu().detach().numpy()
        log_SR_array.append(log_SR)
        
    res = np.hstack(log_SR_array)
    rel_std = res.std(axis=1) / res.mean(axis=1)

    print(f"Relative std across runs: {rel_std.mean():0.4f}")
# This model evaluates physics loss only at specific data points
# we only feed log_area and not min lat min long
# here we do not split the data based on spatial location.


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
from MLP import MLP, inverse_transform_scale_feature_tensor

def max_lat_lon_difference(multipoint):
    if isinstance(multipoint, Point):
        return 0., 0.
    else:
        x_coords = [point.x for point in multipoint.geoms]
        y_coords = [point.y for point in multipoint.geoms]
        max_lon_diff = max(x_coords) - min(x_coords)
        max_lat_diff = max(y_coords) - min(y_coords)
        return max_lat_diff, max_lon_diff
    
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

def create_dataloaders(gdf_train, gdf_test, predictors, batch_size, num_workers):
    X_train, y_train, feature_scaler, target_scaler = scale_features_targets(gdf_train, predictors)
    X_test, y_test, _, _ = scale_features_targets(gdf_test, predictors, feature_scaler, target_scaler)
    
    train_dataset = SpeciesAreaEnvDataset(X_train, y_train)
    test_dataset = SpeciesAreaEnvDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader, feature_scaler, target_scaler


def compute_feature_range(gdf, predictors, feature_scaler):
    """Compute the scaled feature range for random sampling."""
    features = gdf[predictors].values.astype(np.float32)
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)

    scaled_feature_min = (feature_min - feature_scaler.mean_) / feature_scaler.scale_
    scaled_feature_max = (feature_max - feature_scaler.mean_) / feature_scaler.scale_

    return torch.tensor(scaled_feature_min, dtype=torch.float32), torch.tensor(
        scaled_feature_max, dtype=torch.float32
    )

def train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer, scheduler, config
):
    """Train and evaluate the model."""
    best_val_loss = float("inf")
    device = config["device"]
    dSRdA_weight = config["dSRdA_weight"]
    scaled_feature_min = config["scaled_feature_min"].to(device)
    scaled_feature_max = config["scaled_feature_max"].to(device)

    for epoch in range(config["n_epochs"]):
        model.train()
        total_loss = 0

        for log_sr, inputs in train_loader:
            inputs, log_sr = inputs.to(device), log_sr.to(device)
            inputs.requires_grad_(True)

            # Forward pass
            outputs = model(inputs)
            loss_mse = criterion(outputs, log_sr)

            # Gradient penalty on training data
            grads = torch.autograd.grad(
                outputs.sum(), inputs, create_graph=True
            )[0]
            grads_a = grads[:, 0]  # Assuming 'log_area' is the first predictor
            loss_grad = dSRdA_weight * torch.mean(torch.relu(-grads_a) ** 2)

            # Random sampling across data range
            batch_size = inputs.size(0)
            random_inputs = torch.rand(batch_size, inputs.size(1)).to(device)
            random_inputs = scaled_feature_min + random_inputs * (
                scaled_feature_max - scaled_feature_min
            )
            random_inputs.requires_grad_(True)

            # Gradient penalty on random inputs
            random_outputs = model(random_inputs)
            grads_random = torch.autograd.grad(
                random_outputs.sum(), random_inputs, create_graph=True
            )[0]
            grads_a_random = grads_random[:, 0]
            loss_grad_random = dSRdA_weight * torch.mean(torch.relu(-grads_a_random) ** 2)

            # Total loss
            loss = loss_mse + loss_grad + loss_grad_random

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for log_sr, inputs in val_loader:
                inputs, log_sr = inputs.to(device), log_sr.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, log_sr)
                val_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{config['n_epochs']}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Save the best model
        if config["save"] and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config["run_folder"], f"{config['run_name']}_best.pth"),
            )

    return model

def train_and_evaluate(model, gdf_full, predictors, optimizer, scheduler, config):
    best_val_MSE, step = float('inf'), 0
    train_idx, test_idx = train_test_split(gdf_full.index, test_size=config['test_size'], random_state=config['seed'])

    gdf_train, gdf_test = gdf_full.loc[train_idx], gdf_full.loc[test_idx]
    train_loader, val_loader, feature_scaler, target_scaler = create_dataloaders(gdf_train, gdf_test, predictors, config['batch_size'], config['num_workers'])

    # Compute feature range for random sampling
    scaled_feature_min, scaled_feature_max = compute_feature_range(
        gdf_train, predictors, feature_scaler
    )
    dSRdA_weight = config["dSRdA_weight"]
    scaled_feature_min = scaled_feature_min.to(device)
    scaled_feature_max = scaled_feature_max.to(device)
    
    for epoch in range(config['n_epochs']):
        running_train_loss = 0
        running_train_MSE = 0
        for log_sr, inputs in train_loader:
            model.train()
            inputs, log_sr = inputs.to(config['device']), log_sr.to(config['device'])
            inputs.requires_grad_(True)
            
            # Forward pass
            outputs = model(inputs)
            loss_mse = torch.mean((outputs-log_sr) ** 2)

            # Gradient penalty on training data
            grads = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
            grads_a = grads[:, 0]  # Assuming 'log_area' is the first predictor
            # we add a penalty only for negative gradients
            loss_grad = dSRdA_weight * torch.mean(torch.relu(-grads_a) ** 2)

            # THEOPHILE: when this block is uncommented, something strange happen
            # i.e. although the `loss_grad_random` term is not incorporated to the 
            # final loss, the behavior of the convergence is completely changing
            
            
            # # Random sampling across data range
            # batch_size = inputs.size(0)
            # random_inputs = torch.rand(batch_size, inputs.size(1)).to(device)
            # random_inputs = scaled_feature_min + random_inputs * (
            #     scaled_feature_max - scaled_feature_min
            # )
            # random_inputs.requires_grad_(True)

            # # Gradient penalty on random inputs
            # random_outputs = model(random_inputs)
            # grads_random = torch.autograd.grad(random_outputs.sum(), random_inputs, create_graph=True)[0]
            # grads_a_random = grads_random[:, 0]
            # loss_grad_random = dSRdA_weight * torch.mean(torch.relu(-grads_a_random) ** 2)

            # Total loss
            loss = loss_mse + loss_grad # + loss_grad_random
                
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
    return {'model_state_dict': best_model.state_dict(),
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
        "n_epochs": 2,
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

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    model = MLP(len(predictors)).to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, factor=config['lr_scheduler_factor'], patience=config['lr_scheduler_patience'])

    results= train_and_evaluate(model, gdf, predictors, optimizer, scheduler, config)

    print(f"Saving results in {config['run_folder']}")
    torch.save(results, os.path.join(config['run_folder'], f'{config["run_name"]}.pth'))
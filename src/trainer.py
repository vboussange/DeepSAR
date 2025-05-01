import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass, field
from src.mlp import MLP, CustomMSELoss, inverse_transform_scale_feature_tensor
from src.ensemble_model import EnsembleModel
from src.dataset import create_dataloader
from src.plotting import read_result

from sklearn.metrics import mean_squared_error, r2_score

class Trainer:
    def __init__(self, 
                 config,
                 model, 
                 feature_scaler, 
                 target_scaler,
                 train_loader,
                 val_loader,
                 test_loader,
                 compute_loss):
        
        self.model = model.to(config.device)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.compute_loss = compute_loss.to(config.device)
        self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience,
            )
        self.device = config.device
        
    def get_model_predictions(self, loader):
        # we expect the loader to return transformed data
        self.model.eval()
        preds = []
        with torch.no_grad():
            for log_sr, inputs in loader:
                inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
                outputs = self.model(inputs)
                preds.append(outputs.cpu())
        preds = torch.cat(preds, dim=0)
        return preds

    def get_true_pred_target(self, loader):
        preds = self.get_model_predictions(loader)
        preds = preds.cpu()

        target = [y for y, _ in loader]
        target = torch.cat(target, dim=0)
        target = target.cpu()
        
        sr_pred = self.target_scaler.inverse_transform(preds)
        sr_target = self.target_scaler.inverse_transform(target)
        
        return sr_pred, sr_target
    
    def train_step(self):
        self.model.train()
        running_train_loss = 0.0

        for log_sr, inputs in self.train_loader:
            inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
            self.optimizer.zero_grad()
            if isinstance(self.compute_loss, torch.nn.MSELoss):
                outputs = self.model(inputs)
                batch_loss = self.compute_loss(outputs, log_sr)
            else:
                inputs.requires_grad_(True)
                outputs = self.model(inputs)
                batch_loss = self.compute_loss(self.model, outputs, inputs, log_sr)

            batch_loss.backward()
            self.optimizer.step()
            running_train_loss += float(batch_loss.item()) * inputs.size(0)
        avg_train_loss = running_train_loss / len(self.train_loader.dataset)

        self.model.eval()
        val_loss = 0.0
        for log_sr, inputs in self.val_loader:
            inputs, log_sr = inputs.to(self.device), log_sr.to(self.device)
            if isinstance(self.compute_loss, torch.nn.MSELoss):
                with torch.no_grad():
                    outputs = self.model(inputs)
                    batch_loss = self.compute_loss(outputs, log_sr)
            else:
                inputs.requires_grad_(True)
                outputs = self.model(inputs)
                batch_loss = self.compute_loss(self.model, outputs, inputs, log_sr)
            val_loss +=  batch_loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        
        self.scheduler.step(avg_val_loss)

        return avg_train_loss, avg_val_loss
    
    def train(self, n_epochs, metrics = []):
        best_val_loss = float('inf')
        best_model_log = None
        best_model = None

        for epoch in range(n_epochs):
            train_loss, val_loss = self.train_step()
            print(f"Epoch {epoch + 1}/{n_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}") #todo: change for print
            metric_log = {}
            for loader, name in zip([self.train_loader, self.val_loader, self.test_loader], ["train", "val", "test"]):
                true_pred, true_target = self.get_true_pred_target(loader)
                for m in metrics:
                    metric_value = eval(m)(true_pred, true_target)
                    metric_log[name + "_" + m] = metric_value
                    print(f"Epoch {epoch + 1}/{n_epochs} | {m} {name}: {metric_log[name + '_' + m]:.4f}") #todo: change for print

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model).to('cpu')
                best_model_log = metric_log

        return best_model, best_model_log



if __name__ == "__main__":
    # TODO: it feels like there is a problem : The training loss and validation loss do not return the same number, although they should after the model has converged
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from torch.utils.data import DataLoader, TensorDataset
    from src.mlp import CustomMSELoss
    
    logging.basicConfig(level=logging.INFO)

    X = torch.randn(1000, 3) + 10
    y = X.sum(dim=1, keepdim=True)  # Mock target variable
    
    class MockConfig:
        device: str = "cpu"
        num_workers: int = 0
        test_size: float = 0.1
        val_size: float = 0.1
        lr: float = 1e-2
        lr_scheduler_factor: float = 0.5
        lr_scheduler_patience: int = 20
        n_epochs: int = 100
        weight_decay: float = 1e-3
        seed: int = 1

    
    # Create mock scalers
    feature_scaler = StandardScaler()
    X_scaled = torch.tensor(feature_scaler.fit_transform(X.numpy()), dtype=torch.float32)
    
    target_scaler = StandardScaler()  
    y_scaled = torch.tensor(target_scaler.fit_transform(y.numpy()), dtype=torch.float32)
    
    # Use the scaled data for the model
    X = X_scaled
    y = y_scaled
    
    # Create mock dataloaders
    dataset = TensorDataset(y, X)
    train_loader = DataLoader(dataset, batch_size=256)
    val_loader = DataLoader(dataset, batch_size=256)
    test_loader = DataLoader(dataset, batch_size=256)
    
    # Initialize model and trainer
    model = MLP(3, [16, 16, 16])

    compute_loss = CustomMSELoss(0.5)
    
    trainer = Trainer(
        config=MockConfig(),
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        compute_loss=compute_loss,
    )
    
    # # Override evaluate_SR_metric to avoid eval() call
    # trainer.evaluate_SR_metric = lambda metric, loader: 0.5
    

    # Test train method
    best_model, best_model_log = trainer.train(n_epochs=100, metrics=["mean_squared_error", "r2_score"])

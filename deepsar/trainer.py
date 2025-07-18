import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
from deepsar.dataset import create_dataloader

class Trainer:
    """
    Trainer class for managing the training, validation, and testing of a model.
    """
    
    def __init__(self, 
                 config,
                 model, 
                 feature_scaler, 
                 target_scaler,
                 train_loader,
                 val_loader,
                 compute_loss,
                 test_loader = None,
                 device = "cpu"):
        
        self.model = model.to(device)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.compute_loss = compute_loss.to(device)
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
        self.device = device
        
    def get_model_predictions(self, loader):
        # we expect the loader to return transformed data
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X, y in loader:
                X, y_true = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                preds.append(y_pred.cpu())
                targets.append(y_true.cpu())
                
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        return targets, preds
    
    def train_step(self):
        self.model.train()
        running_train_loss = 0.0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            batch_loss = self.compute_loss(outputs, y)

            batch_loss.backward()
            self.optimizer.step()
            running_train_loss += float(batch_loss.item()) * X.size(0)
        avg_train_loss = running_train_loss / len(self.train_loader.dataset)

        self.model.eval()
        val_loss = 0.0
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            outputs = self.model(X)
            batch_loss = self.compute_loss(outputs, y)
            val_loss +=  batch_loss.item() * X.size(0)
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        
        self.scheduler.step(avg_val_loss)

        return avg_train_loss, avg_val_loss
    
    def train(self, n_epochs, metrics = []):
        best_val_loss = float('inf')
        best_model_log = None
        best_model = None
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            train_loss, val_loss = self.train_step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{n_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            metric_log = {}
            if self.test_loader:
                loader, name = self.test_loader, "test"
                targets, preds = self.get_model_predictions(loader)
                pred_trs = self.target_scaler.inverse_transform(preds)
                target_trs = self.target_scaler.inverse_transform(targets)
                for m in metrics:
                    metric_value = eval(m)(target_trs, pred_trs)
                    metric_log[name + "_" + m] = metric_value
                    print(f"Epoch {epoch + 1}/{n_epochs} | {m} {name}: {metric_log[name + '_' + m]:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model).to('cpu')
                best_model_log = metric_log.copy()  # ensure a copy is stored

        # Append loss vectors to the best_model_log
        if best_model_log is None:
            best_model_log = {}
        best_model_log["train_losses"] = train_losses
        best_model_log["val_losses"] = val_losses

        self.model = best_model
        return best_model, best_model_log

if __name__ == "__main__":
    # testing the Trainer class with a mock dataset and model
    from deepsar.mlp import MLP
    
    logging.basicConfig(level=logging.INFO)


    A1 = np.random.rand(1000) * 1000
    A2 = np.random.rand(1000)
    X = np.random.randn(1000, 3)
    y =  np.exp(0.1 * X[:, 0] + 0.2 * X[:, 1] * X[:, 2]) * (A1 * A2)**(0.8)  # Mock target variable
    
    df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    df["log_sr"] = np.log(y)
    df["log_area"] = np.log(A1)
    df["log_sp_unit_area"] = np.log(A2)
    
    predictors = ["log_area", "log_sp_unit_area" , "feature1", "feature2", "feature3"]
    
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
        batch_size: int = 32    

    config = MockConfig()
    train_val_df, test_df = train_test_split(df, test_size = config.test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size = config.val_size, random_state=42)
    
    train_loader, feature_scaler, target_scaler = create_dataloader(train_df, predictors, config.batch_size, config.num_workers)
    val_loader, _, _ = create_dataloader(val_df, predictors, config.batch_size, config.num_workers, feature_scaler=feature_scaler, target_scaler=target_scaler)
    test_loader, _, _ = create_dataloader(test_df, predictors, config.batch_size, config.num_workers, feature_scaler=feature_scaler, target_scaler=target_scaler)

    # Initialize model and trainer
    model = MLP(5, [16, 16, 16])

    loss = nn.MSELoss()
    
    trainer = Trainer(
        config=config,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        compute_loss=loss,
    )
    # sr_pred, sr_true = trainer.get_true_pred_target(test_loader)
    # # Override evaluate_SR_metric to avoid eval() call
    # trainer.evaluate_SR_metric = lambda metric, loader: 0.5
    

    # Test train method
    best_model, best_model_log = trainer.train(n_epochs=100, metrics=["root_mean_squared_error", "r2_score"])
    for k in best_model_log.keys():
        print(k, best_model_log[k])
        
    assert best_model_log["test_r2_score"] > 0.97
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class DeepSARModel(nn.Module, ABC):
    """
    Abstract class to be inherited by a deep species-area relationship model.
    """
    def __init__(self, feature_names = [], feature_scaler = None, target_scaler = None):
        super().__init__()
        self.feature_names = feature_names
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        
    def predict_sr(self, df: pd.DataFrame):
        """
        Public method to predict species richness given sampling effort informed by column `log_observed_area`.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # `X` includes the sampling effort `log_observed_area`, `x` does not
            X = df[["log_observed_area"] + self.feature_names].values.astype(np.float32)
            if self.feature_scaler is not None:
                X = self.feature_scaler.transform(X)
            X = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
            y_pred = self(X).cpu().numpy()
            if self.target_scaler is not None:
                y_pred = self.target_scaler.inverse_transform(y_pred)
            return y_pred
        
        
    def predict_sr_tot(self, df: pd.DataFrame):
        """
        Public method to predict total species richness. 
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # `X` includes the sampling effort `log_observed_area`, `x` does not
            x = df[self.feature_names].values.astype(np.float32)
            # Add a column of zeros to match the expected input shape of feature scaling
            x = np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1) 
            if self.feature_scaler is not None:
                x = self.feature_scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            # x = x.unsqueeze(0)
            x = x[:, 1:]  # Exclude the first column (log_observed_area)
            y_pred = self._predict_sr_tot(x) # predicting asymptote, no need to feed log_observed_area
            if self.target_scaler is not None:
                y_pred = self.target_scaler.inverse_transform(y_pred.cpu().numpy())
            return y_pred

    @staticmethod
    @abstractmethod
    def initialize(checkpoint, device):
        """
        Abstract static method to initialize a model from a checkpoint.
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_ensemble(checkpoint, device):
        """
        Abstract static method to initialize an ensemble of models from a checkpoint.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Abstract method for forward pass.
        """
        pass

    @abstractmethod
    def _predict_sr_tot(self, x):
        """
        Abstract method to predict total species richness.
        """
        pass
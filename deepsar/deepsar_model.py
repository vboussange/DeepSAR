import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class DeepSARModel(nn.Module):
    """
    Deep species-area relationship model.
    This is a base class providing public methods for predicting species richness.
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
            X = df[["log_observed_area"] + self.feature_names].values
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
            x = df[self.feature_names].values
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

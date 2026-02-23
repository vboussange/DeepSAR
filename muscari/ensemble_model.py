from muscari.muscari_model import MuScaRiModel
import torch
from torch import nn
import numpy as np
import pandas as pd

class MuScaRiEnsembleModel(nn.Module):
    def __init__(self, models, **kwargs):
        super(MuScaRiEnsembleModel, self).__init__()

        assert all(isinstance(model, MuScaRiModel) for model in models), "All models must be instances of MuScaRiModel."
        self.models = nn.ModuleList(models)
        
    @property
    def feature_names(self):
        feature_names = self.models[0].feature_names
        assert all(model.feature_names == feature_names for model in self.models), "All models must have the same feature_names."
        return feature_names
    
    def predict_mean_sr(self, df: pd.DataFrame):
        SRs = [model.predict_sr(df) for model in self.models]
        return np.mean(SRs, axis=0).squeeze()
    
    def get_std_sr(self, df: pd.DataFrame):
        SRs = [model.predict_sr(df) for model in self.models]
        return np.std(SRs, axis=0).squeeze()
    
    def predict_mean_sr_tot(self, df: pd.DataFrame):
        """
        Predict mean species richness using the ensemble model;
        `x` should be a 2D array where each row corresponds to a set of (log_sp_unit_area, environmental features).
        """
        SRs = [model.predict_sr_tot(df) for model in self.models]
        return np.mean(SRs, axis=0).squeeze()
    
    def get_std_sr_tot(self, df: pd.DataFrame):
        """
        Predict standard deviation of species richness using the ensemble model;
        `x` should be a 2D array where each row corresponds to a set of (`log_sp_unit_area`, environmental features).
        """
        SRs = [model.predict_sr_tot(df) for model in self.models]
        return np.std(SRs, axis=0).squeeze()


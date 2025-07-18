from deepsar.deepsar_model import DeepSARModel
import torch
from torch import nn
import numpy as np
import pandas as pd

class DeepSAREnsembleModel(nn.Module):
    def __init__(self, models, **kwargs):
        super(DeepSAREnsembleModel, self).__init__()

        assert all(isinstance(model, DeepSARModel) for model in models), "All models must be instances of DeepSARModel."
        self.models = nn.ModuleList(models)
        
    @property
    def feature_names(self):
        feature_names = self.models[0].feature_names
        assert all(model.feature_names == feature_names for model in self.models), "All models must have the same feature_names."
        return feature_names
    
    def predict_mean_s(self, df: pd.DataFrame):
        SRs = [model.predict_s(df) for model in self.models]
        return np.mean(SRs, axis=0)
    
    def get_std_s(self, df: pd.DataFrame):
        SRs = [model.predict_s(df) for model in self.models]
        return np.std(SRs, axis=0)
    
    def predict_mean_s_tot(self, df: pd.DataFrame):
        """
        Predict mean species richness using the ensemble model;
        `x` should be a 2D array where each row corresponds to a set of (log_sp_unit_area, environmental features).
        """
        SRs = [model.predict_s_tot(df) for model in self.models]
        return np.mean(SRs, axis=0)
    
    def get_std_s_tot(self, df: pd.DataFrame):
        """
        Predict standard deviation of species richness using the ensemble model;
        `x` should be a 2D array where each row corresponds to a set of (`log_sp_unit_area`, environmental features).
        """
        SRs = [model.predict_s_tot(df) for model in self.models]
        return np.std(SRs, axis=0)


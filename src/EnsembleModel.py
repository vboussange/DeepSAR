import torch
from torch import nn
from src.MLP import MLP
    
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
    def std(self, x):
        outputs = [model(x) for model in self.models]
        return torch.std(torch.stack(outputs), dim=0)
    
def initialize_ensemble_model(n_ensembles, *args, **kwargs):
    models = [MLP(*args, **kwargs) for _ in range(n_ensembles)]
    return EnsembleModel(models)

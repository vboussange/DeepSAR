import torch
from torch import nn
from deepsar.mlp import MLP
    
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

def initialize_ensemble_model(result, config, device):        
    predictors = result['predictors']
    models = [MLP(len(predictors), layer_sizes=config.layer_sizes).to(device) for _ in range(config.n_ensembles)]
    model = EnsembleModel(models)
    
    # Load model weights and other components
    model.load_state_dict(result['ensemble_model_state_dict'])
    model.eval()
    return model
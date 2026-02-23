import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
# https://github.com/jlager/BINNs/blob/master/Modules/Utils/Gradient.py


class FFNNBatchNormBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(FFNNBatchNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x
    
class FFNN(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=1):
        super(FFNN, self).__init__()
        layer_sizes = [input_dim] + layer_sizes
        self.fully_connected_layers = nn.ModuleList(
            [FFNNBatchNormBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.last_fully_connected = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        return x

class FFNNExp(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(FFNNExp, self).__init__()
        self.nn = FFNN(input_dim, layer_sizes, 1)

    def forward(self, preds):
        x = self.nn(preds)
        return torch.exp(x)

def load_model_checkpoint(model_state, predictors, layer_sizes):
        """Load the model and scalers from the saved checkpoint."""
        model = FFNNExp(len(predictors), layer_sizes=layer_sizes)
        model.load_state_dict(model_state)
        model.eval()
        return model
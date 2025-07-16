import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
# https://github.com/jlager/BINNs/blob/master/Modules/Utils/Gradient.py


class FullyConnectedBatchNormBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(FullyConnectedBatchNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x
    
    
class SimpleNNBatchNorm(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=1):
        super(SimpleNNBatchNorm, self).__init__()
        layer_sizes = [input_dim] + layer_sizes
        self.fully_connected_layers = nn.ModuleList(
            [FullyConnectedBatchNormBlock(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.last_fully_connected = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x):
        for fully_connected_layer in self.fully_connected_layers:
            x = fully_connected_layer(x)
        x = self.last_fully_connected(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(MLP, self).__init__()
        self.nn = SimpleNNBatchNorm(input_dim, layer_sizes, 1)

    def forward(self, preds):
        x = self.nn(preds)
        return x

def load_model_checkpoint(model_state, predictors, layer_sizes):
        """Load the model and scalers from the saved checkpoint."""
        model = MLP(len(predictors), layer_sizes=layer_sizes)
        model.load_state_dict(model_state)
        model.eval()
        return model
    
def scale_feature_tensor(x, scaler):
    assert len(x.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32, device = x.device).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32, device = x.device).reshape(1, -1)
    features_scaled = (x - mean_tensor) / scale_tensor
    return features_scaled

def inverse_transform_scale_feature_tensor(y, scaler):
    assert len(y.shape)
    mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32, device = y.device).reshape(1, -1)
    scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32, device = y.device).reshape(1, -1)
    invy = y * scale_tensor + mean_tensor
    return invy